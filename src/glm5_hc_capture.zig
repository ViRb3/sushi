//! Private, opt-in HC fixture export. Never loaded by normal serving.
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const native = @import("glm5_diagnostic.zig");
const model = @import("model.zig");

fn required(comptime name: [:0]const u8) ![]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return error.MissingGlmHcCaptureSetting);
}

fn save(a: std.mem.Allocator, dir: []const u8, label: []const u8, records: []const forward.HcSnapshot, ids: []const u32, tokens: mlx.mlx_array, cfg: *const model.ModelConfig, provenance: [:0]const u8) !void {
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_map_string_to_string_insert(meta, "provenance", provenance));
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, "tokens", tokens));
    const params = mlx.mlx_array_new_data(&[_]f32{ cfg.rms_norm_eps, cfg.glm_hc_eps, @floatFromInt(cfg.glm_hc_sinkhorn_iters) }, &.{3}, 1, .float32);
    defer _ = mlx.mlx_array_free(params);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, "rms_eps_hc_eps_sinkhorn_iters", params));
    const names = [_][]const u8{ "x", "w", "scale", "base", "mix", "mixed", "post", "comb" };
    for (ids, 0..) |layer, i| for ([_][]const u8{ "attn", "ffn" }, 0..) |kind, j| {
        for (records[2 * i + j].values, names) |value, name| {
            const key = try std.fmt.allocPrintSentinel(a, "layer{d:0>2}.{s}.{s}", .{ layer, kind, name }, 0);
            defer a.free(key);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, key, value));
        }
    };
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}.safetensors", .{ dir, label }, 0);
    defer a.free(path);
    try mlx.check(mlx.mlx_save_safetensors(path, map, meta));
}

fn digest(io: std.Io, a: std.mem.Allocator, dir: []const u8, file: []const u8) ![64]u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, file });
    defer a.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    defer a.free(raw);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

// Coordinator must grant the full-model slot. The output directory must already exist outside
// tracked data. The caller supplies a unique empty private run directory; never reuse filenames.
test "GLM HC private real checkpoint fixtures" {
    const path_raw = std.c.getenv("SUSHI_GLM_HC_CAPTURE_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.span(path_raw);
    const out = try required("SUSHI_GLM_HC_CAPTURE_OUT");
    if (!std.fs.path.isAbsolute(out)) return error.InvalidGlmHcCapturePath;
    const revision = try required("SUSHI_GLM_HC_CAPTURE_REVISION");
    const binary_sha256 = try required("SUSHI_GLM_HC_CAPTURE_BINARY_SHA256");
    const prose_file = try required("SUSHI_GLM_HC_CAPTURE_PROSE");
    const code_file = try required("SUSHI_GLM_HC_CAPTURE_CODE");
    const io = std.testing.io;
    const a = std.testing.allocator;
    var dir = try std.Io.Dir.openDirAbsolute(io, out, .{ .iterate = true });
    defer dir.close(io);
    var iterator = dir.iterate();
    if (try iterator.next(io) != null) return error.GlmHcCaptureDirectoryNotEmpty;
    const s = mlx.gpuStream();
    var previous_memory: usize = 0;
    var previous_cache: usize = 0;
    var previous_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&previous_memory, 110 * 1024 * 1024 * 1024));
    defer _ = mlx.mlx_set_memory_limit(&ignored, previous_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&previous_cache, 2 * 1024 * 1024 * 1024));
    defer _ = mlx.mlx_set_cache_limit(&ignored, previous_cache);
    try mlx.check(mlx.mlx_set_wired_limit(&previous_wired, @min(110 * 1024 * 1024 * 1024, mlx.maxRecommendedWorkingSet())));
    defer _ = mlx.mlx_set_wired_limit(&ignored, previous_wired);
    var cfg = try model.parseConfig(io, a, path);
    defer cfg.deinit(a);
    if (!cfg.isGlm5() or cfg.num_hidden_layers <= 44 or cfg.expert_layout != .exl3_k4) return error.InvalidGlmHcCaptureModel;
    try @import("mimo_source.zig").validateExl3Pack(io, a, path, &cfg);
    var tokenizer = try @import("tokenizer.zig").loadTokenizer(io, a, path);
    defer tokenizer.deinit();
    var weights = try native.loadWeights(io, a, path, s);
    defer weights.deinit();
    var net = try forward.Model.load(a, cfg, &weights, s);
    defer net.deinit();
    const selected = [_]u32{ 0, 3, 23, 44 };
    for ([_][]const u8{ prose_file, code_file }, [_][]const u8{ "prose", "code" }) |prompt_file, category| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, prompt_file, a, .limited(16 * 1024 * 1024));
        defer a.free(text);
        const ids = try tokenizer.encode(a, text);
        defer a.free(ids);
        if (ids.len < 512) return error.GlmHcCapturePromptTooShort;
        var prompt_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &prompt_hash, .{});
        const provenance_json = try std.json.Stringify.valueAlloc(a, .{
            .format_version = 1,
            .model_path = path,
            .revision = revision,
            .binary_sha256_supplied = binary_sha256,
            .config_sha256 = try digest(io, a, path, "config.json"),
            .index_sha256 = try digest(io, a, path, "model.safetensors.index.json"),
            .prompt_sha256 = std.fmt.bytesToHex(prompt_hash, .lower),
            .category = category,
            .prefill_tokens = 512,
            .decode_tokens = 1,
            .layers = selected,
            .reference = "staged FP32 RMS then exact HC projection",
            .timing_valid = false,
        }, .{});
        defer a.free(provenance_json);
        const provenance = try a.dupeSentinel(u8, provenance_json, 0);
        defer a.free(provenance);
        var request = try forward.Request.init(a, cfg.num_hidden_layers);
        defer request.deinit();
        request.dense_prefill = true;
        request.prefill_async = true;
        var records: [8]forward.HcSnapshot = @splat(.{});
        defer for (&records) |*record| record.deinit();
        var capture = forward.HcCapture{ .ids = &selected, .records = &records };
        request.hc_capture = &capture;
        const input = mlx.mlx_array_new_data(ids.ptr, &.{ 1, 512 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(input);
        const logits = try net.forwardLast(&request, input, true);
        defer _ = mlx.mlx_array_free(logits);
        const prefill_label = try std.fmt.allocPrint(a, "{s}-prefill512", .{category});
        defer a.free(prefill_label);
        try save(a, out, prefill_label, &records, &selected, input, &cfg, provenance);
        const next = try native.greedy(logits, s);
        const token = mlx.mlx_array_new_data(&next, &.{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(token);
        const decoded = try net.forwardLast(&request, token, true);
        defer _ = mlx.mlx_array_free(decoded);
        _ = try native.greedy(decoded, s);
        const decode_label = try std.fmt.allocPrint(a, "{s}-decode1", .{category});
        defer a.free(decode_label);
        try save(a, out, decode_label, &records, &selected, token, &cfg, provenance);
    }
}
