//! Long-prefix same-model kernel drift diagnostic; not a lossless pack teacher.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const native = @import("glm5_diagnostic.zig");
const forward = @import("glm5_forward.zig");
const scores = @import("glm5_kld.zig");
const kld = @import("kld.zig");
const batch = @import("glm5_mla_prefill_batch.zig");
const Arr = mlx.mlx_array;
fn number(comptime name: [:0]const u8, fallback: usize) !usize {
    return if (std.c.getenv(name)) |raw| std.fmt.parseInt(usize, std.mem.span(raw), 10) else fallback;
}
fn prefix(net: *const forward.Model, req: *forward.Request, ids: []const u32, chunk: usize) !Arr {
    var logits = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(logits);
    var cursor: usize = 0;
    while (cursor < ids.len) {
        const end = @min(ids.len, cursor + chunk);
        const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &.{ 1, @intCast(end - cursor) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(input);
        const next = try net.forwardLast(req, input, true);
        defer _ = mlx.mlx_array_free(next);
        try mlx.check(mlx.mlx_array_set(&logits, next));
        cursor = end;
    }
    try mlx.check(mlx.mlx_array_eval(logits));
    return logits;
}
fn advance(net: *const forward.Model, req: *forward.Request, id: u32) !Arr {
    const input = mlx.mlx_array_new_data(&id, &.{ 1, 1 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(input);
    return net.forwardLast(req, input, true);
}
fn request(a: std.mem.Allocator, layers: u32) !forward.Request {
    var req = try forward.Request.init(a, layers);
    req.dense_prefill = true;
    req.prefill_async = true;
    req.prefill_sync_layers = 2;
    req.decode_async = true;
    return req;
}
test "GLM long prefix headbatch kernel drift real checkpoint" {
    const raw_model = std.c.getenv("SUSHI_GLM_PREFILL_DRIFT_MODEL") orelse return error.SkipZigTest;
    const raw_ids = std.c.getenv("SUSHI_GLM_PREFILL_DRIFT_IDS") orelse return error.MissingGlmDiagnosticPrompt;
    const raw_out = std.c.getenv("SUSHI_GLM_PREFILL_DRIFT_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const kind_raw = std.c.getenv("SUSHI_GLM_PREFILL_DRIFT_KIND");
    const direct_kind = if (kind_raw) |v| std.mem.eql(u8, std.mem.span(v), "direct") else false;
    const direct_module = @import("glm5_attention_prefill.zig");
    const previous_direct = direct_module.override;
    defer direct_module.override = previous_direct;
    direct_module.override = false;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const count = try number("SUSHI_GLM_PREFILL_DRIFT_PREFIX", 4096);
    const rows = try number("SUSHI_GLM_PREFILL_DRIFT_ROWS", 64);
    const chunk = try number("SUSHI_GLM_PREFILL_DRIFT_CHUNK", 2048);
    if (count < 4096 or count > 16384 or rows < 1 or rows > 512 or chunk != 2048) return error.InvalidGlmDiagnosticBudget;
    const s = mlx.gpuStream();
    var previous_mem: usize = 0;
    var previous_wired: usize = 0;
    var previous_cache: usize = 0;
    var ignored: usize = 0;
    const memory = 110 * 1024 * 1024 * 1024;
    try mlx.check(mlx.mlx_set_memory_limit(&previous_mem, memory));
    defer _ = mlx.mlx_set_memory_limit(&ignored, previous_mem);
    try mlx.check(mlx.mlx_set_wired_limit(&previous_wired, @min(memory, mlx.maxRecommendedWorkingSet())));
    defer _ = mlx.mlx_set_wired_limit(&ignored, previous_wired);
    try mlx.check(mlx.mlx_set_cache_limit(&previous_cache, 2 * 1024 * 1024 * 1024));
    defer _ = mlx.mlx_set_cache_limit(&ignored, previous_cache);
    const path = std.mem.span(raw_model);
    var cfg = try model.parseConfig(io, a, path);
    defer cfg.deinit(a);
    if (!cfg.isGlm5() or cfg.expert_layout != .exl3_k4) return error.UnsupportedGlmDiagnosticLayout;
    try @import("mimo_source.zig").validateExl3Pack(io, a, path, &cfg);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(raw_ids), a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    const list = if (parsed.value == .object) parsed.value.object.get("ids") orelse return error.InvalidGlmPrompt else parsed.value;
    if (list != .array or list.array.items.len == 0) return error.InvalidGlmPrompt;
    const ids = try a.alloc(u32, count);
    defer a.free(ids);
    for (ids, 0..) |*id, i| {
        const value = list.array.items[i % list.array.items.len];
        if (value != .integer or value.integer < 0 or value.integer >= cfg.vocab_size) return error.InvalidGlmPrompt;
        id.* = @intCast(value.integer);
    }
    var weights = try native.loadWeights(io, a, path, s);
    defer weights.deinit();
    var net = try forward.Model.load(a, cfg, &weights, s);
    defer net.deinit();
    var reference = try request(a, cfg.num_hidden_layers);
    defer reference.deinit();
    const teacher = try a.alloc(f32, rows * cfg.vocab_size);
    defer a.free(teacher);
    const tokens = try a.alloc(u32, rows);
    defer a.free(tokens);
    var reference_ns: u64 = 0;
    {
        const mode = batch.bind(false);
        defer mode.restore();
        const timer = @import("io_util.zig").Stopwatch.init(io);
        var logits = try prefix(&net, &reference, ids, chunk);
        defer _ = mlx.mlx_array_free(logits);
        reference_ns = timer.read();
        for (0..rows) |i| {
            try scores.copyLogits(s, logits, teacher[i * cfg.vocab_size ..][0..cfg.vocab_size]);
            tokens[i] = try native.greedy(logits, s);
            if (i + 1 < rows) {
                const next = try advance(&net, &reference, tokens[i]);
                defer _ = mlx.mlx_array_free(next);
                try mlx.check(mlx.mlx_array_set(&logits, next));
            }
        }
    }
    reference.reset();
    var candidate = try request(a, cfg.num_hidden_layers);
    defer candidate.deinit();
    const student = try a.alloc(f32, cfg.vocab_size);
    defer a.free(student);
    var totals: scores.Totals = .{};
    var logit_bit_mismatches: usize = 0;
    const per_row = try a.alloc(f64, rows);
    defer a.free(per_row);
    var candidate_ns: u64 = 0;
    var query_calls: usize = 0;
    var value_calls: usize = 0;
    {
        direct_module.override = direct_kind;
        const mode = batch.bind(!direct_kind);
        defer mode.restore();
        batch.resetDispatchCount();
        const timer = @import("io_util.zig").Stopwatch.init(io);
        var logits = try prefix(&net, &candidate, ids, chunk);
        defer _ = mlx.mlx_array_free(logits);
        candidate_ns = timer.read();
        query_calls = batch.dispatchCount(.query);
        value_calls = batch.dispatchCount(.value);
        if (!direct_kind and (query_calls == 0 or value_calls == 0)) return error.GlmHeadbatchNotEngaged;
        for (0..rows) |i| {
            try scores.copyLogits(s, logits, student);
            for (teacher[i * cfg.vocab_size ..][0..cfg.vocab_size], student) |x, y| {
                logit_bit_mismatches += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
            }
            const row = try kld.scoreRow(teacher[i * cfg.vocab_size ..][0..cfg.vocab_size], student, tokens[i]);
            totals.add(row);
            per_row[i] = row.kld;
            if (i + 1 < rows) {
                const next = try advance(&net, &candidate, tokens[i]);
                defer _ = mlx.mlx_array_free(next);
                try mlx.check(mlx.mlx_array_set(&logits, next));
            }
        }
    }
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const data = try std.json.Stringify.valueAlloc(a, .{ .complete = true, .scope = "same quantized model kernel drift; not a lossless BF16 pack teacher", .candidate = if (direct_kind) "direct_attention" else "headbatch", .model = path, .prefix_tokens = count, .chunk = chunk, .positions = rows, .teacher_ids = tokens, .metrics = totals.metrics(), .logit_bit_mismatches = logit_bit_mismatches, .first_position_kld = per_row[0], .per_position_kld = per_row, .reference_prefill_ns = reference_ns, .candidate_prefill_ns = candidate_ns, .timing = "cold reference then candidate; diagnostic only, not interleaved benchmark", .query_calls = query_calls, .value_calls = value_calls, .peak_bytes = peak, .cache = "BF16 MLA, FP32 KDA" }, .{ .whitespace = .indent_2 });
    defer a.free(data);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(raw_out), .data = data });
}
