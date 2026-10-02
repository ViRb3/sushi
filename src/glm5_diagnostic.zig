//! Opt-in GLM diagnostics; no public model registration or serving dispatch.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const Arr = mlx.mlx_array;

fn keepTextKey(name: []const u8, layers: usize) bool {
    const prefix = "model.language_model.layers.";
    if (std.mem.startsWith(u8, name, prefix)) {
        const tail = name[prefix.len..];
        const end = std.mem.indexOfScalar(u8, tail, '.') orelse return false;
        const layer = std.fmt.parseInt(usize, tail[0..end], 10) catch return false;
        if (layer >= layers) return false;
    }
    if (std.mem.indexOf(u8, name, ".mtp.") != null or std.mem.startsWith(u8, name, "mtp.")) return false;
    return std.mem.startsWith(u8, name, "model.language_model.") or std.mem.startsWith(u8, name, "lm_head.");
}

pub fn storedBytes(weights: *const model.Weights) u64 {
    var total: u64 = 0;
    var it = weights.map.valueIterator();
    while (it.next()) |v| total += mlx.mlx_array_size(v.*) * mlx.mlx_array_itemsize(v.*);
    return total;
}

pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: mlx.mlx_stream) !model.Weights {
    _ = s; // Safetensors Load has a CPU implementation; unified storage is consumed by GPU ops.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const config_raw = try dir.readFileAlloc(io, "config.json", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(config_raw);
    const config = try std.json.parseFromSlice(std.json.Value, allocator, config_raw, .{});
    defer config.deinit();
    if (config.value != .object) return error.InvalidGlmConfig;
    const text_config = config.value.object.get("text_config") orelse config.value;
    if (text_config != .object) return error.InvalidGlmConfig;
    const layer_value = text_config.object.get("num_hidden_layers") orelse return error.InvalidGlmConfig;
    if (layer_value != .integer or layer_value.integer < 1) return error.InvalidGlmConfig;
    const layers: usize = @intCast(layer_value.integer);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidGlmWeightIndex;
    const wm = parsed.value.object.get("weight_map") orelse return error.InvalidGlmWeightIndex;
    if (wm != .object) return error.InvalidGlmWeightIndex;
    var files = std.StringHashMap(void).init(allocator);
    defer files.deinit();
    var owners = wm.object.iterator();
    var expected: usize = 0;
    while (owners.next()) |entry| {
        if (!keepTextKey(entry.key_ptr.*, layers)) continue;
        const value = entry.value_ptr.*;
        if (value != .string or value.string.len == 0 or std.mem.indexOfAny(u8, value.string, "/\\") != null or std.mem.eql(u8, value.string, "..")) return error.InvalidGlmShardName;
        try files.put(value.string, {});
        expected += 1;
    }
    if (expected == 0) return error.MissingIndexedGlmWeight;
    var result = model.Weights.init(allocator);
    errdefer result.deinit();
    var file_it = files.keyIterator();
    while (file_it.next()) |file| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file.* }, 0);
        defer allocator.free(path);
        const present = try std.Io.Dir.openFileAbsolute(io, path, .{});
        present.close(io);
        var arrays = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(arrays);
        var metadata = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(metadata);
        try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, path, cpu));
        const iterator = mlx.mlx_map_string_to_array_iterator_new(arrays);
        defer _ = mlx.mlx_map_string_to_array_iterator_free(iterator);
        while (true) {
            var key: ?[*:0]const u8 = null;
            var value = mlx.mlx_array_new();
            const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iterator);
            if (rc != 0 or key == null) {
                _ = mlx.mlx_array_free(value);
                break;
            }
            const name = std.mem.span(key.?);
            const owner = wm.object.get(name);
            if (!keepTextKey(name, layers) or owner == null or owner.? != .string or !std.mem.eql(u8, owner.?.string, file.*)) {
                _ = mlx.mlx_array_free(value);
                continue;
            }
            errdefer _ = mlx.mlx_array_free(value);
            if (result.get(name) != null) return error.DuplicateGlmWeight;
            const copy = try allocator.dupe(u8, name);
            errdefer allocator.free(copy);
            try result.map.put(copy, value);
        }
    }
    if (result.count() != expected) return error.MissingIndexedGlmWeight;
    // Filter ownership/vision/MTP before materializing any device allocation.
    var values = result.map.valueIterator();
    while (values.next()) |v| try mlx.check(mlx.mlx_array_eval(v.*));
    return result;
}

fn fixture(dir: std.Io.Dir, name: []const u8, header: []const u8, data: []const u8) !void {
    const a = std.testing.allocator;
    const bytes = try a.alloc(u8, 8 + header.len + data.len);
    defer a.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memcpy(bytes[8 + header.len ..], data);
    try dir.writeFile(std.testing.io, .{ .sub_path = name, .data = bytes });
}

fn tmpPath(tmp: anytype) ![]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.c.getcwd(&buf, buf.len) orelse return error.NoCwd;
    return std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}", .{ std.mem.span(@as([*:0]const u8, @ptrCast(cwd))), tmp.sub_path });
}

test "GLM diagnostic loader preserves stored dtypes and strict index ownership" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const header = "{\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"model.language_model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[4,6]},\"model.language_model.layers.0.x.suh\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[6,8]},\"model.language_model.layers.0.x.trellis\":{\"dtype\":\"U16\",\"shape\":[1],\"data_offsets\":[8,10]},\"unindexed\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[10,14]}}";
    try fixture(tmp.dir, "owned.safetensors", header, &.{ 0, 0, 128, 63, 128, 63, 1, 60, 17, 0, 0, 0, 128, 63 });
    try fixture(tmp.dir, "old.safetensors", "{\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 64 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "vision.safetensors", .data = "must not be opened" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"owned.safetensors\",\"model.language_model.norm.weight\":\"owned.safetensors\",\"model.language_model.layers.0.x.suh\":\"owned.safetensors\",\"model.language_model.layers.0.x.trellis\":\"owned.safetensors\",\"model.visual.weight\":\"vision.safetensors\",\"model.language_model.mtp.weight\":\"vision.safetensors\",\"model.language_model.layers.1.mlp.gate.weight\":\"vision.safetensors\"}}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"text_config\":{\"num_hidden_layers\":1}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    var weights = try loadWeights(std.testing.io, a, path, mlx.gpuStream());
    defer weights.deinit();
    try std.testing.expectEqual(@as(u32, 4), weights.count());
    try std.testing.expectEqual(@as(u64, 10), storedBytes(&weights));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(weights.get("lm_head.weight").?));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.language_model.norm.weight").?));
    const half = weights.get("model.language_model.layers.0.x.suh").?;
    try std.testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(half));
    try mlx.check(mlx.mlx_array_eval(half));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, half, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(wide));
    try std.testing.expectEqual(@as(f32, 1.0009765625), mlx.mlx_array_data_float32(wide).?[0]);
    try std.testing.expectEqual(mlx.mlx_dtype.uint16, mlx.mlx_array_dtype(weights.get("model.language_model.layers.0.x.trellis").?));
}

test "GLM diagnostic loader refuses a missing indexed text tensor" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try fixture(tmp.dir, "a.safetensors", "{\"other\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 0 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"a.safetensors\"}}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"text_config\":{\"num_hidden_layers\":1}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    try std.testing.expectError(error.MissingIndexedGlmWeight, loadWeights(std.testing.io, a, path, mlx.gpuStream()));
}

fn envNumber(comptime name: [:0]const u8, default: usize) !usize {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(raw), 10);
}

fn writeJson(io: std.Io, a: std.mem.Allocator, path: []const u8, value: anytype) !void {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{ .whitespace = .indent_2 });
    defer a.free(raw);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = raw });
}

fn readPromptIds(a: std.mem.Allocator, raw: []const u8, count: usize, vocab: u32) ![]u32 {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    const value = if (parsed.value == .object) parsed.value.object.get("ids") orelse return error.InvalidGlmPrompt else parsed.value;
    if (value != .array or value.array.items.len < count or count == 0) return error.InvalidGlmPrompt;
    const ids = try a.alloc(u32, count);
    errdefer a.free(ids);
    for (ids, value.array.items[0..count]) |*id, item| {
        if (item != .integer or item.integer < 0 or item.integer >= vocab) return error.InvalidGlmPrompt;
        id.* = @intCast(item.integer);
    }
    return ids;
}

test "GLM diagnostic prompt keeps exact supplied prefix and rejects invalid IDs" {
    const a = std.testing.allocator;
    const ids = try readPromptIds(a, "{\"ids\":[7,2,9]}", 2, 10);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 7, 2 }, ids);
    try std.testing.expectError(error.InvalidGlmPrompt, readPromptIds(a, "[1]", 2, 10));
    try std.testing.expectError(error.InvalidGlmPrompt, readPromptIds(a, "[-1]", 1, 10));
    try std.testing.expectError(error.InvalidGlmPrompt, readPromptIds(a, "[10]", 1, 10));
}

const ByteArray = struct {
    bytes: []const u8,
    pub fn jsonStringify(self: ByteArray, writer: anytype) !void {
        try writer.beginArray();
        for (self.bytes) |byte| try writer.write(byte);
        try writer.endArray();
    }
};
const DecodedOutput = struct { output_text: ?[]const u8, output_bytes: ByteArray, output_text_utf8_valid: bool };
fn decodedOutput(bytes: []const u8) DecodedOutput {
    const valid = std.unicode.utf8ValidateSlice(bytes);
    return .{ .output_text = if (valid) bytes else null, .output_bytes = .{ .bytes = bytes }, .output_text_utf8_valid = valid };
}

test "GLM diagnostic output JSON text is a string with byte fallback" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "hi", &.{ 0xe4, 0xb8 } }) |bytes| {
        const raw = try std.json.Stringify.valueAlloc(a, decodedOutput(bytes), .{});
        defer a.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer parsed.deinit();
        const text = parsed.value.object.get("output_text").?;
        if (std.unicode.utf8ValidateSlice(bytes)) {
            try std.testing.expect(text == .string);
            try std.testing.expectEqualStrings(bytes, text.string);
        } else try std.testing.expect(text == .null);
        const raw_bytes = parsed.value.object.get("output_bytes").?;
        try std.testing.expect(raw_bytes == .array);
        for (raw_bytes.array.items, bytes) |item, byte| try std.testing.expectEqual(@as(i64, byte), item.integer);
    }
}

pub fn greedy(logits: Arr, s: mlx.mlx_stream) !u32 {
    var finite = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(finite);
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_isfinite(&finite, logits, s));
    try mlx.check(mlx.mlx_all(&all, finite, false, s));
    var arg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(arg);
    try mlx.check(mlx.mlx_argmax_axis(&arg, logits, -1, false, s));
    const evals = mlx.mlx_vector_array_new_data(&.{ all, arg }, 2);
    defer _ = mlx.mlx_vector_array_free(evals);
    try mlx.check(mlx.mlx_eval(evals));
    if (!mlx.mlx_array_data_bool(all).?[0]) return error.NonfiniteGlmLogits;
    if (mlx.mlx_array_size(arg) != 1) return error.InvalidGlmDiagnosticLogits;
    return mlx.mlx_array_data_uint32(arg).?[0];
}

test "GLM diagnostic greedy batches evaluation and rejects nonfinite logits" {
    const s = mlx.gpuStream();
    const x = mlx.mlx_array_new_data(&[_]f32{ -1, 4, 4, 0 }, &.{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    try std.testing.expectEqual(@as(u32, 1), try greedy(x, s));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        const y = mlx.mlx_array_new_data(&[_]f32{ 1, bad }, &.{ 1, 1, 2 }, 3, .float32);
        defer _ = mlx.mlx_array_free(y);
        try std.testing.expectError(error.NonfiniteGlmLogits, greedy(y, s));
    }
}

// Run only after the coordinator grants the full-model GPU slot.
test "GLM native diagnostic real model" {
    const path_raw = std.c.getenv("SUSHI_GLM_DIAGNOSTIC_MODEL") orelse return error.SkipZigTest;
    const out_raw = std.c.getenv("SUSHI_GLM_DIAGNOSTIC_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const io = std.testing.io;
    const a = std.testing.allocator;
    const path = std.mem.span(path_raw);
    const out = std.mem.span(out_raw);
    const progress = try std.fmt.allocPrint(a, "{s}.progress.json", .{out});
    defer a.free(progress);
    errdefer writeJson(io, a, progress, .{ .phase = "failed", .complete = false }) catch {};
    const log = @import("log.zig");
    const routing_histogram = @import("transformer.zig").diagEnvOn("SUSHI_EXL3_UNION_HIST");
    log.setLevel(if (routing_histogram) .info else .err);
    if (routing_histogram) log.enableStderr();
    defer log.setLevel(.info);
    const count = try envNumber("SUSHI_GLM_DIAGNOSTIC_PREFILL", 512);
    const steps = try envNumber("SUSHI_GLM_DIAGNOSTIC_DECODE", 64);
    const chunk = try envNumber("SUSHI_GLM_DIAGNOSTIC_CHUNK", 128);
    const warmup = try envNumber("SUSHI_GLM_DIAGNOSTIC_WARMUP", 0);
    const components = (try envNumber("SUSHI_GLM_DIAGNOSTIC_COMPONENTS", 0)) != 0;
    const profile = (try envNumber("SUSHI_GLM_DIAGNOSTIC_PROFILE", 1)) != 0 or components;
    const prefill_sync_layers = try envNumber("SUSHI_GLM_DIAGNOSTIC_PREFILL_SYNC_LAYERS", 2);
    if (prefill_sync_layers == 0 or prefill_sync_layers > 8) return error.InvalidGlmDiagnosticBudget;
    const warmup_decode = try envNumber("SUSHI_GLM_DIAGNOSTIC_WARMUP_DECODE", 1);
    if (warmup_decode < 1 or warmup_decode > 4096) return error.InvalidGlmDiagnosticBudget;
    if (warmup > 8) return error.InvalidGlmDiagnosticBudget;
    if (count == 0 or count > 65536 or steps < 2 or steps > 4096 or chunk == 0) return error.InvalidGlmDiagnosticBudget;
    const memory_limit = (try envNumber("SUSHI_GLM_DIAGNOSTIC_MEMORY_GIB", 110)) * 1024 * 1024 * 1024;
    const cache_limit = (try envNumber("SUSHI_GLM_DIAGNOSTIC_CACHE_GIB", 2)) * 1024 * 1024 * 1024;
    if (memory_limit == 0) return error.InvalidGlmDiagnosticBudget;
    const s = mlx.gpuStream();
    const recommended = mlx.maxRecommendedWorkingSet();
    const wired_limit = if (std.c.getenv("SUSHI_GLM_DIAGNOSTIC_WIRED_GIB") != null)
        (try envNumber("SUSHI_GLM_DIAGNOSTIC_WIRED_GIB", 0)) * 1024 * 1024 * 1024
    else
        @min(memory_limit, recommended);
    if (recommended == 0 or wired_limit > recommended) return error.InvalidGlmWiredLimit;
    var prior_memory: usize = 0;
    var prior_cache: usize = 0;
    var prior_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&prior_memory, memory_limit));
    defer _ = mlx.mlx_set_memory_limit(&ignored, prior_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&prior_cache, cache_limit));
    defer _ = mlx.mlx_set_cache_limit(&ignored, prior_cache);
    try mlx.check(mlx.mlx_set_wired_limit(&prior_wired, wired_limit));
    defer _ = mlx.mlx_set_wired_limit(&ignored, prior_wired);
    var cfg = try model.parseConfig(io, a, path);
    defer cfg.deinit(a);
    if (!cfg.isGlm5()) return error.InvalidGlmConfig;
    if (cfg.expert_layout != .exl3_k4) return error.UnsupportedGlmDiagnosticLayout;
    try writeJson(io, a, progress, .{ .phase = "preflight", .complete = false });
    try @import("mimo_source.zig").validateExl3Pack(io, a, path, &cfg);
    const tokenizer = @import("tokenizer.zig");
    var tok = try tokenizer.loadTokenizer(io, a, path);
    defer tok.deinit();
    const ids = if (std.c.getenv("SUSHI_GLM_DIAGNOSTIC_TOKENS_FILE")) |file| blk: {
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(file), a, .limited(16 * 1024 * 1024));
        defer a.free(raw);
        break :blk try readPromptIds(a, raw, count, cfg.vocab_size);
    } else if (std.c.getenv("SUSHI_GLM_DIAGNOSTIC_PROMPT_FILE")) |file| blk: {
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(file), a, .limited(16 * 1024 * 1024));
        defer a.free(raw);
        const encoded = try tok.encode(a, raw);
        defer a.free(encoded);
        if (encoded.len < count) return error.InvalidGlmPrompt;
        break :blk try a.dupe(u32, encoded[0..count]);
    } else return error.MissingGlmDiagnosticPrompt;
    defer a.free(ids);
    try writeJson(io, a, progress, .{ .phase = "loading", .complete = false, .memory_limit_bytes = memory_limit, .wired_limit_bytes = wired_limit, .recommended_working_set_bytes = recommended });
    const load_start = std.Io.Timestamp.now(io, .awake);
    var weights = try loadWeights(io, a, path, s);
    defer weights.deinit();
    const payload = storedBytes(&weights);
    try writeJson(io, a, progress, .{ .phase = "modelbind", .complete = false, .stored_tensor_bytes = payload, .tensors = weights.count() });
    const forward = @import("glm5_forward.zig");
    var net = try forward.Model.load(a, cfg, &weights, s);
    defer net.deinit();
    var request = try forward.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.decode_async = (try envNumber("SUSHI_GLM_DIAGNOSTIC_DECODE_ASYNC", 1)) != 0;
    request.dense_prefill = (try envNumber("SUSHI_GLM_DIAGNOSTIC_DENSE_PREFILL", 0)) != 0;
    request.prefill_async = (try envNumber("SUSHI_GLM_DIAGNOSTIC_PREFILL_ASYNC", 0)) != 0;
    request.prefill_sync_layers = @intCast(prefill_sync_layers);
    var loaded_active: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&loaded_active));
    const load_seconds = @as(f64, @floatFromInt(load_start.untilNow(io, .awake).nanoseconds)) / 1e9;
    const warm_start = std.Io.Timestamp.now(io, .awake);
    for (0..warmup) |round| {
        try writeJson(io, a, progress, .{ .phase = "warmup", .complete = false, .round = round });
        var cursor: usize = 0;
        var next: u32 = 0;
        while (cursor < ids.len) {
            const end = @min(ids.len, cursor + chunk);
            const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &[_]c_int{ 1, @intCast(end - cursor) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const result = try net.forwardLast(&request, input, true);
            defer _ = mlx.mlx_array_free(result);
            next = try greedy(result, s);
            cursor = end;
        }
        for (0..warmup_decode) |_| {
            const input = mlx.mlx_array_new_data(&next, &[_]c_int{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const result = try net.forwardLast(&request, input, true);
            defer _ = mlx.mlx_array_free(result);
            next = try greedy(result, s);
        }
        request.reset();
    }
    const warmup_seconds = if (warmup > 0) @as(f64, @floatFromInt(warm_start.untilNow(io, .awake).nanoseconds)) / 1e9 else 0;
    var timed_wired_limit = wired_limit;
    const wired_policy = if (std.c.getenv("SUSHI_WIRED")) |raw| std.mem.span(raw) else "explicit";
    if (!std.mem.eql(u8, wired_policy, "explicit")) {
        if (std.c.getenv("SUSHI_GLM_DIAGNOSTIC_WIRED_GIB") != null) return error.ConflictingGlmWiredSettings;
        timed_wired_limit = mlx.applyWiredPolicy().target orelse return error.InvalidGlmWiredLimit;
    }
    request.profile = profile;
    request.profile_components = components;
    @import("glm5_decode.zig").resetDispatchCount();
    @import("glm5_kda_fused.zig").resetDispatchCount();
    @import("glm5_kda_fused.zig").resetPostDispatchCount();
    @import("glm5_kda_prework.zig").resetDispatchCount();
    @import("glm5_router.zig").resetCallCount();
    @import("glm5_activation.zig").resetCallCount();
    @import("glm5_hc_fused.zig").resetDispatchCount();
    @import("glm5_hc_prefill.zig").resetDispatchCount();
    @import("sushi_exl3").kernels.resetPairedCooperativeCalls();
    @import("sushi_exl3").kernels.resetClampedMiddleDispatchCount();
    try mlx.check(mlx.mlx_reset_peak_memory());
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    var prefill_ns: i128 = 0;
    var offset: usize = 0;
    while (offset < ids.len) {
        const end = @min(ids.len, offset + chunk);
        try writeJson(io, a, progress, .{ .phase = "prefill", .complete = false, .completed_tokens = offset, .total_tokens = count });
        const start = std.Io.Timestamp.now(io, .awake);
        const input = mlx.mlx_array_new_data(ids[offset..end].ptr, &[_]c_int{ 1, @intCast(end - offset) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(input);
        const result = try net.forwardLast(&request, input, true);
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_array_eval(result));
        try mlx.check(mlx.mlx_array_set(&logits, result));
        prefill_ns += start.untilNow(io, .awake).nanoseconds;
        offset = end;
    }
    const prefill_layer_ns = request.layer_ns;
    const prefill_component_ns = request.component_ns;
    const generated = try a.alloc(u32, steps);
    defer a.free(generated);
    generated[0] = try greedy(logits, s);
    var decode_ns: i128 = 0;
    for (1..steps) |i| {
        try writeJson(io, a, progress, .{ .phase = "decode", .complete = false, .generated_tokens = i, .target_tokens = steps });
        const start = std.Io.Timestamp.now(io, .awake);
        const input = mlx.mlx_array_new_data(generated[i - 1 .. i].ptr, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(input);
        const result = try net.forwardLast(&request, input, true);
        defer _ = mlx.mlx_array_free(result);
        generated[i] = try greedy(result, s);
        decode_ns += start.untilNow(io, .awake).nanoseconds;
    }
    var active: usize = 0;
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const text = try tok.decode(a, generated, false);
    defer a.free(text);
    const prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9;
    const decode_seconds = @as(f64, @floatFromInt(decode_ns)) / 1e9;
    var eos_at: ?usize = null;
    for (generated, 0..) |id, i| if (cfg.isEosToken(id)) {
        eos_at = i;
        break;
    };
    var decode_component_ns = request.component_ns;
    for (&decode_component_ns, prefill_component_ns) |*total, prefill| {
        inline for (comptime std.meta.fieldNames(@import("glm5_forward.zig").ComponentTimes)) |name| @field(total, name) -= @field(prefill, name);
    }
    var decode_layer_ns = request.layer_ns;
    for (&decode_layer_ns, prefill_layer_ns) |*total, prefill| total.* -= prefill;
    const decoded = decodedOutput(text);
    var rate_buf: [32]u8 = undefined;
    try writeJson(io, a, out, .{ .complete = true, .model = path, .expert_k = cfg.expert_quant_rate.kText(&rate_buf), .expert_window = cfg.expert_quant_window.bits(), .stored_tensor_bytes = payload, .loaded_active_bytes = loaded_active, .active_bytes = active, .peak_bytes = peak, .memory_limit_bytes = memory_limit, .cache_limit_bytes = cache_limit, .wired_limit_bytes = timed_wired_limit, .wired_policy = wired_policy, .recommended_working_set_bytes = recommended, .load_seconds = load_seconds, .prefill_tokens = count, .prefill_chunk = chunk, .prefill_seconds = prefill_seconds, .prefill_tokens_per_second = @as(f64, @floatFromInt(count)) / prefill_seconds, .generated_tokens = steps, .decode_forward_tokens = steps - 1, .decode_seconds = decode_seconds, .decode_tokens_per_second = @as(f64, @floatFromInt(steps - 1)) / decode_seconds, .first_token_from_prefill = true, .eos_index = eos_at, .continued_after_eos = eos_at != null and eos_at.? + 1 < steps, .input_ids = ids, .output_ids = generated, .output_text = decoded.output_text, .output_bytes = decoded.output_bytes, .output_text_utf8_valid = decoded.output_text_utf8_valid, .warmup_count = warmup, .warmup_decode_forwards_per_round = if (warmup > 0) warmup_decode else 0, .warmup_seconds = warmup_seconds, .prefix_reuse = false, .routing_histogram_enabled = routing_histogram, .profile_enabled = profile, .component_profile_enabled = components, .prefill_component_ns = prefill_component_ns[0..cfg.num_hidden_layers], .decode_component_ns = decode_component_ns[0..cfg.num_hidden_layers], .dense_prefill = request.dense_prefill, .prefill_schedule = if (request.prefill_async and !profile) (if (request.prefill_sync_layers == 2) "async2" else "async-bounded") else "synchronous", .prefill_sync_layers = if (request.prefill_async and !profile) request.prefill_sync_layers else @as(u8, 1), .qkv_dispatches = @import("glm5_decode.zig").dispatchCount(), .kda_post_dispatches = @import("glm5_kda_fused.zig").postDispatchCount(), .kda_prework_dispatches = @import("glm5_kda_prework.zig").dispatchCount(), .kda_body_dispatches = @import("glm5_kda_fused.zig").dispatchCount(), .router_calls = @import("glm5_router.zig").callCount(), .hc_dispatches = @import("glm5_hc_fused.zig").dispatchCount(), .hc_prefill_dispatches = @import("glm5_hc_prefill.zig").dispatchCount(), .activation_calls = @import("glm5_activation.zig").callCount(), .clamped_middle_dispatches = @import("sushi_exl3").kernels.clampedMiddleDispatchCount(), .paired_expert_calls = @import("sushi_exl3").kernels.pairedCooperativeCalls(), .decode_schedule = if (request.decode_async and !profile) "async4" else "synchronous", .prefill_layer_ns = prefill_layer_ns[0..cfg.num_hidden_layers], .decode_layer_ns = decode_layer_ns[0..cfg.num_hidden_layers], .mtp = false, .kv = "BF16 attention cache; FP32 KDA state", .public_serving_enabled = false });
    try writeJson(io, a, progress, .{ .phase = "complete", .complete = true });
}
