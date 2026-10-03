//! Optional proposal-only coarse head: 3-bit shortlist, stored-head row re-score.
const std = @import("std");
const mlx = @import("mlx.zig");
const native = @import("glm5_model.zig");
const mtp = @import("mtp.zig");
const Arr = mlx.mlx_array;
const Ops = native.Ops;

pub const bits: u32 = 3;
pub const group_size: u32 = 64;
pub const shortlist_size: c_int = 32;
var projection_calls: usize = 0;
pub fn projectionCount() usize {
    return projection_calls;
}
pub fn resetStats() void {
    projection_calls = 0;
}

pub fn residentBytes(vocab: c_int, hidden: c_int) u64 {
    return mtp.rerankCoarseBytes(vocab, hidden, bits);
}

/// Build a separate head from stored source rows; the source remains borrowed.
pub fn build(source: native.Linear, s: mlx.mlx_stream) !mtp.QLinear {
    if (source.output < mtp.TOP32_MIN_ROWS or source.input <= 0 or @rem(source.input, 128) != 0 or source.scales.ctx == null)
        return error.UnsupportedGlmMiniHead;
    const source_bits = try native.storedAffineBits(source.w, source.scales, source.biases);
    var coarse = try mtp.requantizeRows(s, source.w, source.scales, source.biases, 128, @intCast(source_bits), "affine", group_size, bits, 16384);
    errdefer coarse.deinit();
    const values = [_]Arr{ coarse.w, coarse.s, coarse.b };
    const ev = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    return coarse;
}

/// Read an anchor plus draft rows; return only drafts with 32 finite logits.
/// The caller retains the selector's top16 and its conditional normalization.
pub fn project(source: native.Linear, coarse: *const mtp.QLinear, hidden: Arr, s: mlx.mlx_stream) !Arr {
    const sh = mlx.getShape(hidden);
    const qw = mlx.getShape(coarse.w);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 2 or sh[1] > 8 or sh[2] != source.input or
        qw.len != 2 or qw[0] != source.output or qw[1] * 32 != source.input * @as(c_int, bits)) return error.UnsupportedGlmMiniHead;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const coarse_logits = try ops.slot();
    try mlx.check(mlx.mlx_quantized_matmul(coarse_logits, try ops.slice(hidden, 1, 1, sh[1]), coarse.w, coarse.s, coarse.b, true, mlx.mlx_optional_int.some(group_size), mlx.mlx_optional_int.some(bits), "affine", s));
    var parts: [15]Arr = undefined;
    const count: usize = @intCast(sh[1] - 1);
    for (0..count) |i| {
        const lo: c_int = @intCast(i);
        const flat = try ops.reshape(try ops.slice(coarse_logits.*, 1, lo, lo + 1), &.{source.output});
        const ids = try ops.own(try mtp.draftTop32(s, flat, source.output));
        const selected = native.Linear{
            .w = try ops.take(source.w, ids, 0),
            .scales = try ops.take(source.scales, ids, 0),
            .biases = try ops.take(source.biases, ids, 0),
            .input = source.input,
            .output = shortlist_size,
        };
        // Keep the original block width so MLX retains its qmv_wide reduction.
        const all_exact = try selected.apply(&ops, hidden);
        const exact = try ops.slice(all_exact, 1, lo + 1, lo + 2);
        const empty = try ops.slot();
        try mlx.check(mlx.mlx_full(empty, &[_]c_int{ 1, 1, source.output }, 3, try ops.scalar(-std.math.inf(f32), .float32), mlx.mlx_array_dtype(exact), s));
        const masked = try ops.slot();
        try mlx.check(mlx.mlx_put_along_axis(masked, empty.*, try ops.reshape(ids, &.{ 1, 1, shortlist_size }), exact, 2, s));
        parts[i] = masked.*;
    }
    const out = try ops.result(try ops.concat(parts[0..count], 1));
    projection_calls += 1;
    return out;
}

test "GLM DFlash mini head storage includes both BF16 affine grids" {
    try std.testing.expectEqual(@as(u64, 277544960), residentBytes(154880, 4096));
}

test "GLM DFlash mini head re-scores original A6 rows and preserves source storage" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const vocab = mtp.TOP32_MIN_ROWS + 17;
    const hidden = 128;
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 0x53132));
    const dense = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(dense, &[_]c_int{ vocab, hidden }, 2, .bfloat16, 0, 0.03125, key.*, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, dense.*, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(6), "affine", .{}, s));
    var stored: [3]Arr = undefined;
    for (&stored, 0..) |*out, i| {
        const slot = try ops.slot();
        try mlx.check(mlx.mlx_vector_array_get(slot, triple, i));
        out.* = slot.*;
    }
    const source = native.Linear{ .w = stored[0], .scales = stored[1], .biases = stored[2], .input = hidden, .output = vocab };
    const source_handles = stored;
    var coarse = try build(source, s);
    defer coarse.deinit();
    try std.testing.expectEqualSlices(c_int, &.{ vocab, hidden * 3 / 32 }, mlx.getShape(coarse.w));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &[_]c_int{ 1, 4, hidden }, 3, .bfloat16, 0, 1, key.*, s));
    const masked = try ops.own(try project(source, &coarse, x.*, s));
    try mlx.check(mlx.mlx_array_eval(masked));
    const actual = mlx.mlx_array_data_bfloat16(masked) orelse return error.MlxArrayDataNull;
    const original = try source.apply(&ops, x.*);
    try mlx.check(mlx.mlx_array_eval(original));
    const expected = mlx.mlx_array_data_bfloat16(original) orelse return error.MlxArrayDataNull;
    for (0..3) |row| {
        var finite: usize = 0;
        for (0..vocab) |id| {
            const value = actual[row * vocab + id];
            if (value != 0xff80) {
                finite += 1;
                try std.testing.expectEqual(expected[(row + 1) * vocab + id], value);
            }
        }
        try std.testing.expectEqual(@as(usize, 32), finite);
    }
    for (source_handles, stored) |before, after| try std.testing.expectEqual(before.ctx, after.ctx);
    try std.testing.expectEqual(@as(c_int, 6), try native.storedAffineBits(source.w, source.scales, source.biases));
}

test "GLM DFlash mini production head component" {
    const path = std.c.getenv("SUSHI_GLM_MINI_HEAD_SHARD") orelse return error.SkipZigTest;
    const output = std.c.getenv("SUSHI_GLM_MINI_HEAD_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, path, cpu));
    var source_arrays: [3]Arr = undefined;
    for ([_][*:0]const u8{ "lm_head.weight", "lm_head.scales", "lm_head.biases" }, 0..) |name, i| {
        const slot = try ops.slot();
        try mlx.check(mlx.mlx_map_string_to_array_get(slot, arrays, name));
        source_arrays[i] = slot.*;
    }
    const source = native.Linear{ .w = source_arrays[0], .scales = source_arrays[1], .biases = source_arrays[2], .input = mlx.getShape(source_arrays[1])[1] * 128, .output = mlx.getShape(source_arrays[0])[0] };
    var coarse = try build(source, s);
    defer coarse.deinit();
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 0x53132));
    const hidden = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(hidden, &[_]c_int{ 1, 8, source.input }, 3, .bfloat16, 0, 1, key.*, s));
    try mlx.check(mlx.mlx_array_eval(hidden.*));
    const masked = try ops.own(try project(source, &coarse, hidden.*, s));
    const original = try source.apply(&ops, hidden.*);
    try mlx.check(mlx.mlx_array_eval(masked));
    try mlx.check(mlx.mlx_array_eval(original));
    const actual = mlx.mlx_array_data_bfloat16(masked) orelse return error.MlxArrayDataNull;
    const expected = mlx.mlx_array_data_bfloat16(original) orelse return error.MlxArrayDataNull;
    const vocab: usize = @intCast(source.output);
    var finite: usize = 0;
    var unequal: usize = 0;
    for (0..7 * vocab) |i| if (actual[i] != 0xff80) {
        finite += 1;
        if (actual[i] != expected[vocab + i]) unequal += 1;
    };
    var retained_top1: usize = 0;
    var retained_top16: usize = 0;
    for (0..7) |row| {
        var best_values: [16]f32 = @splat(-std.math.inf(f32));
        var best_ids: [16]usize = @splat(0);
        for (0..vocab) |id| {
            const value: f32 = @bitCast(@as(u32, expected[(row + 1) * vocab + id]) << 16);
            for (0..16) |rank| if (value > best_values[rank]) {
                var shift: usize = 15;
                while (shift > rank) : (shift -= 1) {
                    best_values[shift] = best_values[shift - 1];
                    best_ids[shift] = best_ids[shift - 1];
                }
                best_values[rank] = value;
                best_ids[rank] = id;
                break;
            };
        }
        if (actual[row * vocab + best_ids[0]] != 0xff80) retained_top1 += 1;
        for (best_ids) |id| if (actual[row * vocab + id] != 0xff80) {
            retained_top16 += 1;
        };
    }
    const samples = 8;
    var full_ns: [samples]u64 = undefined;
    var mini_ns: [samples]u64 = undefined;
    for (0..samples + 2) |iteration| {
        for (0..2) |arm| {
            const use_mini = (iteration + arm) % 2 == 1;
            var scope = Ops{ .s = s };
            defer scope.deinit();
            const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
            const value = if (use_mini) try scope.own(try project(source, &coarse, hidden.*, s)) else try source.apply(&scope, hidden.*);
            try mlx.check(mlx.mlx_array_eval(value));
            const ns = watch.read();
            if (iteration >= 2) {
                if (use_mini) mini_ns[iteration - 2] = ns else full_ns[iteration - 2] = ns;
            }
        }
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .source_bits = try native.storedAffineBits(source.w, source.scales, source.biases), .source_group = 128, .vocab = vocab, .hidden = source.input, .coarse_bits = bits, .coarse_group = group_size, .coarse_bytes = residentBytes(source.output, source.input), .finite = finite, .expected_finite = 7 * 32, .source_logit_bit_mismatches = unequal, .retained_top1 = retained_top1, .expected_top1 = 7, .retained_top16 = retained_top16, .expected_top16 = 112, .input = "fixed-seed random BF16 assistant-shaped hidden; no full model loaded", .timing = "interleaved full8 vs mini7 readout; two warmup pairs; ns", .full_ns = full_ns, .mini_ns = mini_ns }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
    try std.testing.expectEqual(@as(usize, 7 * 32), finite);
    try std.testing.expectEqual(@as(usize, 0), unequal);
}
