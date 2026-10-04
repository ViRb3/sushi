//! A6 prefill expansion plus dense BF16 GEMM; no persistent weight copy.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

pub const expanded_weight_bytes: usize = 64 * 1024 * 1024;
var dispatches: usize = 0;
pub fn dispatchCount() usize {
    return dispatches;
}
pub fn resetDispatchCount() void {
    dispatches = 0;
}
fn geometry(xs: []const c_int, ws: []const c_int, ss: []const c_int, bs: []const c_int, dtypes: [4]mlx.mlx_dtype) bool {
    if (xs.len != 3 or xs[0] != 1 or xs[1] != 2048 or ws.len != 2) return false;
    const k = xs[2];
    const n = ws[0];
    if (!((k == 4096 and n == 8192) or (k == 8192 and n == 4096)) or ws[1] != @divExact(k * 3, 16)) return false;
    const scale_shape = [_]c_int{ n, @divExact(k, 128) };
    return std.mem.eql(c_int, &scale_shape, ss) and std.mem.eql(c_int, &scale_shape, bs) and
        std.mem.eql(mlx.mlx_dtype, &.{ .bfloat16, .uint32, .bfloat16, .bfloat16 }, &dtypes);
}
pub fn tryPrefill(ops: *Ops, x: Arr, w: Arr, scales: Arr, biases: Arr) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_model.zig").naxArms()) return null;
    const arrays = [_]Arr{ x, w, scales, biases };
    for (arrays) |v| if (v.ctx == null) return null;
    var dtypes: [4]mlx.mlx_dtype = undefined;
    for (arrays, 0..) |v, i| dtypes[i] = mlx.mlx_array_dtype(v);
    if (!geometry(mlx.getShape(x), mlx.getShape(w), mlx.getShape(scales), mlx.getShape(biases), dtypes)) return null;
    const weights = try ops.dequant(w, scales, biases);
    const result = try ops.binary(.mm, x, try ops.transpose(weights, &.{ 1, 0 }));
    dispatches += 1;
    return result;
}
fn bill(pending_layers: usize) !usize {
    return std.math.mul(usize, expanded_weight_bytes * 4, pending_layers);
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    return if (chunk == 2048 and @import("glm5_model.zig").naxArms()) bill(pending_layers) else 0;
}

const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };
fn fixture(ops: *Ops, k: c_int, n: c_int, seed: u64) !Input {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ 1, 2048, k }, 3, .bfloat16, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ n, @divExact(k * 3, 16) }, 2, 4, key.*, ops.s));
    const scale = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(scale, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.01, .bfloat16), &.{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    const bias = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(bias, try ops.scalar(-1.2, .bfloat16), try ops.scalar(-0.1, .bfloat16), &.{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    for ([_]Arr{ x.*, w.*, scale.*, bias.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
    return .{ .x = x.*, .w = w.*, .scales = scale.*, .biases = bias.* };
}
fn expectBits(reference: Arr, candidate: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(reference));
    try mlx.check(mlx.mlx_array_eval(candidate));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(reference), mlx.getShape(candidate));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(candidate));
    const count = mlx.mlx_array_size(reference);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(reference).?[0..count], mlx.mlx_array_data_bfloat16(candidate).?[0..count]);
}

test "GLM A6 dense once guard retains decode small tensor and affine8 paths" {
    const dtype = [_]mlx.mlx_dtype{ .bfloat16, .uint32, .bfloat16, .bfloat16 };
    try std.testing.expect(geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(geometry(&.{ 1, 2048, 8192 }, &.{ 4096, 1536 }, &.{ 4096, 64 }, &.{ 4096, 64 }, dtype));
    for ([_]c_int{ 1, 3, 512, 1024, 2047, 2049 }) |t| try std.testing.expect(!geometry(&.{ 1, t, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 2, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 1024 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 128, 768 }, &.{ 128, 32 }, &.{ 128, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 64 }, &.{ 8192, 64 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, .{ .float32, .uint32, .bfloat16, .bfloat16 }));
}

test "GLM A6 dense once transient bill covers pending four projection layers" {
    try std.testing.expectEqual(@as(usize, 268435456), try bill(1));
    try std.testing.expectEqual(@as(usize, 536870912), try bill(2));
    try std.testing.expectEqual(@as(usize, 2147483648), try bill(8));
    try std.testing.expectError(error.Overflow, bill(std.math.maxInt(usize)));
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(512, 2));
}

test "GLM A6 dense once Linear keeps native bits and declines decode rows" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    for ([_]c_int{ 4096, 8192 }, 0..) |k, shape| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const n: c_int = if (k == 4096) 8192 else 4096;
        const input = try fixture(&ops, k, n, @intCast(1031 + shape));
        const linear = @import("glm5_model.zig").Linear{ .w = input.w, .scales = input.scales, .biases = input.biases, .input = k, .output = n };
        resetDispatchCount();
        const actual = try linear.apply(&ops, input.x);
        try expectBits(try ops.qmm(input.x, input.w, input.scales, input.biases, true), actual);
        try std.testing.expectEqual(@as(usize, 1), dispatchCount());
        const decode = try ops.slice(input.x, 1, 0, 3);
        const short_actual = try linear.apply(&ops, decode);
        try expectBits(try ops.qmm(decode, input.w, input.scales, input.biases, true), short_actual);
        try std.testing.expectEqual(@as(usize, 1), dispatchCount());
    }
    resetDispatchCount();
}
