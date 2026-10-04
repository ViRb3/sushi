//! Native streamed GLM capture; standard KLD files, independent engine provenance.
const std = @import("std");
const model = @import("model.zig");
const kld = @import("kld.zig");
const mlx = @import("mlx.zig");
const native = @import("glm5_diagnostic.zig");
const forward = @import("glm5_forward.zig");
const streaming = @import("glm5_stream.zig");
const Arr = mlx.mlx_array;
pub fn accepts(cfg: *const model.ModelConfig, opts: kld.Options) !bool {
    if (!cfg.isGlm5() or opts.command != .capture) return false;
    // The FP8 release is a block quantization of the BF16 one, never a teacher.
    if (cfg.expert_layout == .fp8_individual) return error.NativeGlmTeacherRequiresLosslessStreaming;
    // A pack captures its own served path (a student reference, not the teacher) at a BF16 latent.
    if (cfg.expert_layout != .bf16_individual) {
        if (opts.kv_quant_config.isQuant()) return error.GlmStudentReferenceNeedsBf16Latent;
        return false;
    }
    if (cfg.quant_bits != 0 or
        opts.tokens == 0 or opts.ssd_budget_bytes == 0 or opts.expert_cache_bytes != 0 or opts.pick_tolerance != 0 or
        !opts.no_template or opts.enable_mtp or opts.hidden_out.len != 0 or
        opts.kv_quant_config.scheme != .off or opts.wired_margin_bytes != 0)
        return error.NativeGlmTeacherRequiresLosslessStreaming;
    return true;
}
pub fn tryRun(a: std.mem.Allocator, io: std.Io, opts: kld.Options, out: *kld.Out) !bool {
    if (opts.command != .capture) return false;
    var cfg = try model.parseConfig(io, a, opts.model_dir);
    defer cfg.deinit(a);
    if (!try accepts(&cfg, opts)) return false;
    try run(a, io, &cfg, opts, out);
    return true;
}
test "GLM native KLD capture takes the lossless teacher before the generic loader" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    const opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30 };
    if (!(try accepts(&cfg, opts))) return error.MissingNativeGlmCapture;
}

const HeaderAudit = struct { trunk_bytes: u64 = 0, trunk_bf16: usize = 0, trunk_f32: usize = 0, expert_bf16: usize = 0, shard_stat_sha256: [64]u8 = @splat(0) };

fn textKey(key: []const u8, layers: usize) bool {
    if (std.mem.startsWith(u8, key, "model.language_model.layers.")) {
        const tail = key["model.language_model.layers.".len..];
        const end = std.mem.indexOfScalar(u8, tail, '.') orelse return false;
        if ((std.fmt.parseInt(usize, tail[0..end], 10) catch return false) >= layers) return false;
    }
    return std.mem.indexOf(u8, key, ".mtp.") == null and
        (std.mem.startsWith(u8, key, "model.language_model.") or std.mem.startsWith(u8, key, "lm_head."));
}
fn expertKey(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".mlp.experts.") != null or std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null;
}
fn auditTensor(value: std.json.Value, expert: bool, file_bytes: u64, base: u64, audit: *HeaderAudit) !void {
    if (value != .object) return error.InvalidGlmTeacherHeader;
    const dtype = value.object.get("dtype") orelse return error.InvalidGlmTeacherHeader;
    if (dtype != .string) return error.InvalidGlmTeacherHeader;
    const bf16 = std.mem.eql(u8, dtype.string, "BF16");
    const fp32 = std.mem.eql(u8, dtype.string, "F32");
    if ((!bf16 and !fp32) or (expert and !bf16)) return error.NativeGlmTeacherUnsupportedDtype;
    const shape = value.object.get("shape") orelse return error.InvalidGlmTeacherHeader;
    if (shape != .array) return error.InvalidGlmTeacherHeader;
    var elements: u64 = 1;
    for (shape.array.items) |dim| {
        if (dim != .integer or dim.integer <= 0) return error.InvalidGlmTeacherHeader;
        elements = try std.math.mul(u64, elements, @intCast(dim.integer));
    }
    const offsets = value.object.get("data_offsets") orelse return error.InvalidGlmTeacherHeader;
    if (offsets != .array or offsets.array.items.len != 2) return error.InvalidGlmTeacherHeader;
    const start = offsets.array.items[0];
    const end = offsets.array.items[1];
    if (start != .integer or end != .integer or start.integer < 0 or end.integer < start.integer) return error.InvalidGlmTeacherHeader;
    const bytes: u64 = @intCast(end.integer - start.integer);
    if (bytes != try std.math.mul(u64, elements, if (bf16) 2 else 4) or
        try std.math.add(u64, base, @intCast(end.integer)) > file_bytes) return error.InvalidGlmTeacherHeader;
    if (expert) audit.expert_bf16 += 1 else {
        audit.trunk_bytes = try std.math.add(u64, audit.trunk_bytes, bytes);
        if (bf16) audit.trunk_bf16 += 1 else audit.trunk_f32 += 1;
    }
}

/// Read only index-owned text tensor headers before the lazy loader can evaluate.
fn auditHeaders(a: std.mem.Allocator, io: std.Io, directory: []const u8, cfg: *const model.ModelConfig, check_tensors: bool) !HeaderAudit {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    const index = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer index.deinit();
    if (index.value != .object) return error.InvalidGlmWeightIndex;
    const wm = index.value.object.get("weight_map") orelse return error.InvalidGlmWeightIndex;
    if (wm != .object) return error.InvalidGlmWeightIndex;
    var files = std.StringHashMap(void).init(a);
    defer files.deinit();
    var entries = wm.object.iterator();
    while (entries.next()) |entry| {
        if (!textKey(entry.key_ptr.*, cfg.num_hidden_layers)) continue;
        const file = entry.value_ptr.*;
        if (file != .string or file.string.len == 0 or std.mem.indexOfAny(u8, file.string, "/\\") != null or std.mem.eql(u8, file.string, "..")) return error.InvalidGlmShardName;
        try files.put(file.string, {});
    }
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(a);
    var shards = files.keyIterator();
    while (shards.next()) |shard| try names.append(a, shard.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, left: []const u8, right: []const u8) bool {
            return std.mem.lessThan(u8, left, right);
        }
    }.less);
    var audit: HeaderAudit = .{};
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (names.items) |shard| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ directory, shard }, 0);
        defer a.free(path);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.MissingIndexedGlmWeight;
        defer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0 or st.size < 8) return error.InvalidGlmTeacherHeader;
        const mt = st.mtime();
        const stamp = try std.fmt.allocPrint(a, "{s}:{d}:{d}:{d}:{d}\n", .{ shard, st.ino, st.size, mt.sec, mt.nsec });
        defer a.free(stamp);
        hash.update(stamp);
        if (!check_tensors) continue;
        var length: [8]u8 = undefined;
        try @import("expert_io.zig").readExact(fd, &length, 0);
        const n = std.mem.readInt(u64, &length, .little);
        if (n == 0 or n > 128 * 1024 * 1024 or n > @as(u64, @intCast(st.size)) - 8) return error.InvalidGlmTeacherHeader;
        const header = try a.alloc(u8, @intCast(n));
        defer a.free(header);
        try @import("expert_io.zig").readExact(fd, header, 8);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidGlmTeacherHeader;
        entries = wm.object.iterator();
        while (entries.next()) |entry| {
            if (!textKey(entry.key_ptr.*, cfg.num_hidden_layers) or !std.mem.eql(u8, entry.value_ptr.string, shard)) continue;
            const tensor = parsed.value.object.get(entry.key_ptr.*) orelse return error.MissingIndexedGlmWeight;
            try auditTensor(tensor, expertKey(entry.key_ptr.*), @intCast(st.size), n + 8, &audit);
        }
    }
    if (check_tensors and (audit.trunk_bf16 + audit.trunk_f32 == 0 or audit.expert_bf16 == 0)) return error.MissingIndexedGlmWeight;
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    audit.shard_stat_sha256 = std.fmt.bytesToHex(digest, .lower);
    return audit;
}

fn json(a: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8, value: anytype) !void {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{ .whitespace = .indent_2 });
    defer a.free(raw);
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer dir.close(io);
    const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{name});
    defer a.free(tmp);
    try dir.writeFile(io, .{ .sub_path = tmp, .data = raw });
    try dir.rename(tmp, dir, name, io);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

fn normalizeTeacherTf32() !void {
    if (std.c.getenv("MLX_ENABLE_TF32")) |value| {
        if (!std.mem.eql(u8, std.mem.span(value), "0")) return error.NativeGlmTeacherRequiresTf32Off;
    } else if (setenv("MLX_ENABLE_TF32", "0", 0) != 0) return error.NativeGlmTeacherEnvironmentFailed;
}

fn teacherEnvironment() !void {
    try normalizeTeacherTf32();
    if (model.getConfigOverrides() != null) return error.NativeGlmTeacherOverrides;
    @import("glm5_model.zig").reference_numerics = true;
}

/// Widening BF16 to F32 is exact. Any other native dtype is refused, never repaired.
fn exportRow(s: mlx.mlx_stream, logits: Arr, row: []f32) !mlx.mlx_dtype {
    if (mlx.mlx_array_size(logits) != row.len) return error.InvalidGlmDiagnosticLogits;
    const dtype = mlx.mlx_array_dtype(logits);
    if (dtype != .bfloat16 and dtype != .float32) return error.NativeGlmTeacherUnsupportedLogits;
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(wide));
    @memcpy(row, (mlx.mlx_array_data_float32(wide) orelse return error.MlxArrayDataNull)[0..row.len]);
    var nonzero = false;
    for (row) |v| {
        if (!std.math.isFinite(v)) return error.NonfiniteGlmLogits;
        nonzero = nonzero or v != 0;
    }
    if (!nonzero) return error.NativeGlmTeacherZeroNormLogits;
    return dtype;
}
fn greedy(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}
fn activeBound(limit: u64, remaining: u64) !usize {
    var active: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    if (active > limit or remaining > limit - active) return error.GlmResidentBudgetExceeded;
    return active;
}

fn run(a: std.mem.Allocator, io: std.Io, cfg: *const model.ModelConfig, opts: kld.Options, out: *kld.Out) !void {
    try teacherEnvironment();
    const cwd = std.Io.Dir.cwd();
    if (cwd.access(io, opts.out_dir, .{})) |_| return error.NativeGlmTeacherOutputExists else |err| if (err != error.FileNotFound) return err;
    const staging = try std.fmt.allocPrint(a, "{s}.partial", .{opts.out_dir});
    defer a.free(staging);
    try cwd.createDir(io, staging, .default_dir);
    var prompts = try kld.loadPrompts(a, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    if (prompts.items.len == 0) return error.NoPromptsFound;
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, opts.model_dir);
    defer tok.deinit();
    const inputs = try a.alloc([]u32, prompts.items.len);
    defer a.free(inputs);
    var encoded: usize = 0;
    defer for (inputs[0..encoded]) |ids| a.free(ids);
    var max_tokens: usize = 0;
    for (prompts.items, inputs) |p, *ids| {
        ids.* = if (p.ids) |given| try a.dupe(u32, given) else try tok.encode(a, p.text);
        encoded += 1;
        if (ids.len == 0) return error.EmptyPrompt;
        for (ids.*) |id| if (id >= cfg.vocab_size) return error.InvalidGlmPrompt;
        const capacity = try std.math.add(usize, ids.len, opts.tokens);
        if (capacity > cfg.max_position_embeddings or (opts.ctx_size != 0 and capacity > opts.ctx_size)) return error.GlmContextExceeded;
        max_tokens = @max(max_tokens, capacity);
    }
    const chunk = @min(@as(usize, 512), max_tokens);
    const reserve = @max(@as(u64, 8) << 30, try streaming.minimumReserve(cfg, max_tokens, chunk));
    const trunk_limit = try streaming.trunkLimit(cfg, opts.ssd_budget_bytes, reserve);
    const config_sha = try metadataHash(a, io, opts.model_dir, "config.json");
    const index_sha = try metadataHash(a, io, opts.model_dir, "model.safetensors.index.json");
    const tokenizer_sha = try metadataHash(a, io, opts.model_dir, "tokenizer.json");
    const headers = try auditHeaders(a, io, opts.model_dir, cfg, true);
    if (headers.trunk_bytes > trunk_limit) return error.GlmResidentBudgetExceeded;
    try json(a, io, staging, "source-header-audit.json", headers);
    try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "validated", .prompts = prompts.items.len, .tokens_per_prompt = opts.tokens });
    const limit: usize = std.math.cast(usize, opts.ssd_budget_bytes) orelse return error.InvalidGlmStreamBudget;
    const recommended = mlx.maxRecommendedWorkingSet();
    if (recommended == 0 or limit > recommended) return error.InvalidGlmWiredLimit;
    var old_memory: usize = 0;
    var old_cache: usize = 0;
    var old_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_memory, limit));
    defer _ = mlx.mlx_set_memory_limit(&ignored, old_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, 0));
    defer _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    try mlx.check(mlx.mlx_set_wired_limit(&old_wired, limit));
    defer _ = mlx.mlx_set_wired_limit(&ignored, old_wired);
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    const s = mlx.gpuStream();
    const started = std.Io.Timestamp.now(io, .awake);
    try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "loading" });
    var weights = try native.loadWeightsBounded(io, a, opts.model_dir, s, true, trunk_limit);
    defer weights.deinit();
    const payload = native.storedBytes(&weights);
    if (payload != headers.trunk_bytes) return error.NativeGlmTeacherStoredBytesChanged;
    const budget = try streaming.captureBudget(cfg, opts.ssd_budget_bytes, payload, reserve, max_tokens, chunk);
    var engine = try @import("expert_stream.zig").Engine.initWithOptions(a, opts.model_dir, cfg.expertGeometry(), budget.cache, s, .{ .layout = .bf16_individual });
    defer engine.deinit();
    var net = try forward.Model.loadStreamed(a, cfg.*, &weights, s, .{ .engine = &engine, .max_tokens = max_tokens, .max_chunk = chunk });
    defer net.deinit();
    const loaded_active = try activeBound(limit, reserve);
    var request = try forward.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    request.prefill_async = false;
    request.decode_async = false;
    request.prefill_sync_layers = 1;
    var records: std.ArrayList(kld.PromptRecord) = .empty;
    defer {
        for (records.items) |r| r.deinit(a);
        records.deinit(a);
    }
    const row = try a.alloc(f32, cfg.vocab_size);
    defer a.free(row);
    const generated = try a.alloc(u32, opts.tokens);
    defer a.free(generated);
    var logits_dtype: ?mlx.mlx_dtype = null;
    var final_offset: usize = 0;
    for (prompts.items, inputs, 0..) |p, ids, index| {
        request.reset();
        var writer = try kld.PromptWriter.begin(a, io, staging, index, p.id);
        defer writer.deinit();
        var logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits);
        var cursor: usize = 0;
        while (cursor < ids.len) {
            const end = @min(ids.len, cursor + chunk);
            const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &.{ 1, @intCast(end - cursor) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const next = try net.forwardLast(&request, input, true);
            _ = mlx.mlx_array_free(logits);
            logits = next;
            try mlx.check(mlx.mlx_array_eval(logits));
            _ = try activeBound(limit, 0);
            cursor = end;
            try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "prefill", .prompt = p.id, .prefill_tokens = cursor, .completed_rows = 0 });
        }
        for (generated, 0..) |*chosen, step| {
            const dtype = try exportRow(s, logits, row);
            if (logits_dtype) |prior| {
                if (prior != dtype) return error.NativeGlmTeacherLogitsDtypeChanged;
            } else logits_dtype = dtype;
            chosen.* = greedy(row);
            try writer.appendRow(row, chosen.*);
            _ = try activeBound(limit, 0);
            try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "capture", .prompt = p.id, .completed_prompts = index, .completed_rows = step + 1, .total_rows = opts.tokens, .request_offset = request.offset });
            if (step + 1 == generated.len) break;
            const input = mlx.mlx_array_new_data(chosen, &.{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const next = try net.forwardLast(&request, input, true);
            _ = mlx.mlx_array_free(logits);
            logits = next;
        }
        if (writer.rows != opts.tokens or request.offset != ids.len + opts.tokens - 1) return error.NativeGlmTeacherRowCountMismatch;
        final_offset = request.offset;
        const record = try writer.finish(p.text, p.text, ids, generated);
        errdefer record.deinit(a);
        try records.append(a, record);
    }
    const elapsed = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / 1e9;
    try kld.writeBaseline(a, io, staging, .{ .label = opts.label, .model = opts.model_dir, .run = opts.label, .kv_cache_format = "bf16", .inference_profile = "greedy-native-glm-streamed", .prompt_set = opts.prompts, .ssd_budget_gb = opts.ssd_budget_bytes >> 30, .tokens_per_prompt = opts.tokens, .top_k = opts.top_k, .elapsed_secs = elapsed }, records.items);
    var peak: usize = 0;
    var cached: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    try mlx.check(mlx.mlx_get_cache_memory(&cached));
    const active = try activeBound(limit, 0);
    if (!std.mem.eql(u8, &config_sha, &try metadataHash(a, io, opts.model_dir, "config.json")) or
        !std.mem.eql(u8, &index_sha, &try metadataHash(a, io, opts.model_dir, "model.safetensors.index.json")) or
        !std.mem.eql(u8, &tokenizer_sha, &try metadataHash(a, io, opts.model_dir, "tokenizer.json"))) return error.NativeGlmTeacherSourceChanged;
    const final_shards = try auditHeaders(a, io, opts.model_dir, cfg, false);
    if (!std.mem.eql(u8, &headers.shard_stat_sha256, &final_shards.shard_stat_sha256)) return error.NativeGlmTeacherSourceChanged;
    try completeBaseline(a, io, staging, records.items.len, opts.tokens);
    if (cached != 0 or peak > limit) return error.GlmResidentBudgetExceeded;
    try json(a, io, staging, "identity.json", .{ .schema = "sushi-native-glm-capture-v1", .complete = true, .engine = "sushi-native-glm", .model = opts.model_dir, .source_storage = "indexed BF16/F32 trunk; individual BF16 experts, as stored", .config_sha256 = &config_sha, .index_sha256 = &index_sha, .tokenizer_sha256 = &tokenizer_sha, .kda_unary_modes = try @import("glm5_kda_fused.zig").unaryModes(s), .kda_body_dispatches = @import("glm5_kda_fused.zig").dispatchCount(), .kda_post_dispatches = @import("glm5_kda_fused.zig").postDispatchCount(), .kda_prework_dispatches = @import("glm5_kda_prework.zig").dispatchCount(), .trunk_header_audit = headers, .shard_stat_sha256 = &headers.shard_stat_sha256, .logits_dtype = @tagName(logits_dtype.?), .logits_export = "exact full-vocabulary little-endian float32", .vocab_size = cfg.vocab_size, .tokens_per_prompt = opts.tokens, .prompt_count = records.items.len, .prefix_chunk = chunk, .final_request_offset = final_offset, .kv_cache_format = "bf16", .kda_state_format = "float32", .dense_prefill = true, .synchronous_layers = true, .mtp = false, .dflash = false, .tf32 = false, .template = false, .prefix_reuse = false, .stream_budget = budget, .stream_cache_slots = engine.plan.slots_per_layer, .stream_fill_bytes = engine.fill_bytes_total, .loaded_active_bytes = loaded_active, .active_bytes = active, .peak_bytes = peak, .memory_limit_bytes = limit, .wired_limit_bytes = limit, .allocator_cache_bytes = cached, .elapsed_seconds = elapsed });
    try json(a, io, staging, "progress.json", .{ .complete = true, .phase = "complete", .completed_prompts = records.items.len, .completed_rows = records.items.len * opts.tokens });
    try cwd.renamePreserve(staging, cwd, opts.out_dir, io);
    out.print("[kld] native GLM captured {d} prompts x {d} full-vocabulary rows into {s}\n", .{ records.items.len, opts.tokens, opts.out_dir });
}

test "GLM native KLD capture rejects lossy options" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    var opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30 };
    opts.enable_mtp = true;
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.enable_mtp = false;
    opts.expert_cache_bytes = 1;
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.expert_cache_bytes = 0;
    opts.kv_quant_config = @import("transformer.zig").KVQuantConfig.affine(8);
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.command = .compare;
    try std.testing.expect(!try accepts(&cfg, opts));
}

test "GLM native KLD capture CPU header audit refuses nonteacher storage" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "BF16", "F32", "F16", "U16", "F8_E4M3" }) |dtype| {
        const raw = try std.fmt.allocPrint(a, "{{\"dtype\":\"{s}\",\"shape\":[2],\"data_offsets\":[0,{d}]}}", .{ dtype, if (std.mem.eql(u8, dtype, "F32")) @as(u8, 8) else 4 });
        defer a.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer parsed.deinit();
        var audit: HeaderAudit = .{};
        if (std.mem.eql(u8, dtype, "BF16") or std.mem.eql(u8, dtype, "F32")) {
            try auditTensor(parsed.value, false, 64, 8, &audit);
            try std.testing.expect(audit.trunk_bytes == 4 or audit.trunk_bytes == 8);
        } else try std.testing.expectError(error.NativeGlmTeacherUnsupportedDtype, auditTensor(parsed.value, false, 64, 8, &audit));
        if (std.mem.eql(u8, dtype, "BF16")) {
            try auditTensor(parsed.value, true, 64, 8, &audit);
            try std.testing.expectEqual(@as(usize, 1), audit.expert_bf16);
        } else try std.testing.expectError(error.NativeGlmTeacherUnsupportedDtype, auditTensor(parsed.value, true, 64, 8, &audit));
    }
    try std.testing.expect(!textKey("model.language_model.layers.40.mlp.gate.weight", 40));
    try std.testing.expect(!textKey("model.language_model.mtp.weight", 40));
    try std.testing.expect(!textKey("model.visual.weight", 40));
}

test "GLM native KLD capture CPU exact logits export and greedy full vocabulary" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bits = [_]u16{ 0xbf80, 0x3f80, 0x4080, 0x4080, 0x0001 };
    const x = mlx.mlx_array_new_data(&bits, &.{ 1, 1, 5 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var row: [5]f32 = undefined;
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, try exportRow(s, x, &row));
    for (bits, row) |b, f| try std.testing.expectEqual(@as(u32, b) << 16, @as(u32, @bitCast(f)));
    try std.testing.expectEqual(@as(u32, 2), greedy(&row));
    const half = [_]f16{ 1, 2 };
    const y = mlx.mlx_array_new_data(&half, &.{ 1, 1, 2 }, 3, .float16);
    defer _ = mlx.mlx_array_free(y);
    try std.testing.expectError(error.NativeGlmTeacherUnsupportedLogits, exportRow(s, y, row[0..2]));
}

fn metadataHash(a: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8) ![64]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, name, a, .limited(128 * 1024 * 1024));
    defer a.free(bytes);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}
fn completeBaseline(a: std.mem.Allocator, io: std.Io, directory: []const u8, count: usize, tokens: u32) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "baseline.json", a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadBaselineJson;
    const arena = parsed.arena.allocator();
    try parsed.value.object.put(arena, "complete", .{ .bool = true });
    try parsed.value.object.put(arena, "requested_prompt_count", .{ .integer = @intCast(count) });
    try parsed.value.object.put(arena, "completed_prompt_count", .{ .integer = @intCast(count) });
    try parsed.value.object.put(arena, "requested_tokens_per_prompt", .{ .integer = tokens });
    try parsed.value.object.put(arena, "actual_positions", .{ .integer = @intCast(count * tokens) });
    try parsed.value.object.put(arena, "human_truncated", .{ .bool = false });
    try parsed.value.object.put(arena, "quality_scope", .{ .string = "Requested prompt study; full fixed-length capture. The standard release verdict requires sixteen prompts." });
    try json(a, io, directory, "baseline.json", parsed.value);
}

test "GLM native KLD capture CPU rejects zero norm logits" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const data = [_]f32{ 0, 0 };
    const x = mlx.mlx_array_new_data(&data, &.{ 1, 1, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    var row: [2]f32 = undefined;
    try std.testing.expectError(error.NativeGlmTeacherZeroNormLogits, exportRow(s, x, &row));
}

test "GLM teacher capture selects reference numerics with TF32 off" {
    const a = std.testing.allocator;
    const base = @import("glm5_model.zig");
    const saved = if (std.c.getenv("MLX_ENABLE_TF32")) |value| try a.dupeSentinel(u8, std.mem.span(value), 0) else null;
    defer {
        if (saved) |value| {
            _ = setenv("MLX_ENABLE_TF32", value, 1);
            a.free(value);
        } else _ = unsetenv("MLX_ENABLE_TF32");
        base.reference_numerics = false;
    }
    _ = unsetenv("MLX_ENABLE_TF32");
    try std.testing.expect(!base.reference_numerics);
    try teacherEnvironment();
    try std.testing.expect(base.reference_numerics);
    try std.testing.expectEqualStrings("0", std.mem.span(std.c.getenv("MLX_ENABLE_TF32").?));
    const reference_switches = [_]bool{
        @import("glm5_hc_prefill.zig").enabled(),
        @import("glm5_hc_collapse_simd32.zig").enabled(),
        @import("glm5_kda_prefill_cluster.zig").enabled(),
        @import("glm5_attention_nax_packed.zig").enabled(),
        @import("glm5_indexpool_nax.zig").enabled(),
        @import("glm5_attention_decode_batch.zig").enabled(),
    };
    for (reference_switches) |fast| try std.testing.expect(!fast);
    base.reference_numerics = false;
    _ = setenv("MLX_ENABLE_TF32", "1", 1);
    try std.testing.expectError(error.NativeGlmTeacherRequiresTf32Off, teacherEnvironment());
}

test "GLM KLD capture sends a pack to the generic student reference, at a BF16 latent only" {
    var cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .exl3_k4 };
    const kv = @import("kv_quant.zig").KVQuantConfig;
    var opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 256 };
    try std.testing.expect(!(try accepts(&cfg, opts)));
    for ([_]kv{ kv.affine(4), kv.affine(8) }) |quant| {
        opts.kv_quant_config = quant;
        try std.testing.expectError(error.GlmStudentReferenceNeedsBf16Latent, accepts(&cfg, opts));
    }
    opts.command = .compare;
    try std.testing.expect(!(try accepts(&cfg, opts)));
    cfg.expert_layout = .fp8_individual;
    try std.testing.expect(!(try accepts(&cfg, opts)));
    opts.command = .capture;
    opts.kv_quant_config = .dense;
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    cfg.expert_layout = .bf16_individual;
    opts = .{ .command = .capture, .no_template = true, .tokens = 256 };
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.ssd_budget_bytes = 100 << 30;
    try std.testing.expect(try accepts(&cfg, opts));
}

test "GLM native KLD capture refuses a kv8 latent cache" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    const opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30, .kv_quant_config = @import("kv_quant.zig").KVQuantConfig.affine(8) };
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
}
