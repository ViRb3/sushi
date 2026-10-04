//! Head-batched affine MLA prefill projections; banks and cache precision are unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const Direction = enum { query, value };
pub const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };

pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (chunk < 128 or chunk > 2048 or !@import("glm5_model.zig").naxArms()) return 0;
    // Input/output permutation copies total64H*(256+512+512+256)*2 bytes/row.
    return std.math.mul(usize, try std.math.mul(usize, chunk, 196608), pending_layers);
}

fn geometry(xs: []const c_int, ws: []const c_int, ss: []const c_int, bs: []const c_int, dtype: mlx.mlx_dtype, dir: Direction) bool {
    const width: c_int = if (dir == .query) 256 else 512;
    if (xs.len != 4 or xs[0] < 128 or xs[0] > 2048 or xs[1] != 64 or xs[2] != 1 or xs[3] != width or dtype != .bfloat16) return false;
    return std.mem.eql(c_int, &.{ 64, 256, 96 }, ws) and std.mem.eql(c_int, &.{ 64, 256, 4 }, ss) and std.mem.eql(c_int, ss, bs);
}
pub fn run(ops: *Ops, input: Input, dir: Direction) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_model.zig").naxArms()) return null;
    for ([_]Arr{ input.x, input.w, input.scales, input.biases }) |v| if (v.ctx == null) return null;
    if (!geometry(mlx.getShape(input.x), mlx.getShape(input.w), mlx.getShape(input.scales), mlx.getShape(input.biases), mlx.mlx_array_dtype(input.x), dir) or
        mlx.mlx_array_dtype(input.w) != .uint32 or mlx.mlx_array_dtype(input.scales) != .bfloat16 or mlx.mlx_array_dtype(input.biases) != .bfloat16) return null;
    const xs = mlx.getShape(input.x);
    const out: c_int = if (dir == .query) 512 else 256;
    const heads = try ops.contiguous(try ops.reshape(try ops.transpose(input.x, &.{ 1, 0, 2, 3 }), &.{ 64, xs[0], xs[3] }));
    const y = try ops.qmm(heads, input.w, input.scales, input.biases, dir == .value);
    return try ops.reshape(try ops.contiguous(try ops.transpose(y, &.{ 1, 0, 2 })), &.{ xs[0], 64, 1, out });
}

test "GLM MLA headbatch prefill geometry guards original banks" {
    try std.testing.expect(geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(geometry(&.{ 128, 64, 1, 512 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .value));
    for ([_]c_int{ 1, 3, 16, 32, 127, 2049 }) |t| try std.testing.expect(!geometry(&.{ t, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(!geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 128 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(!geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .float32, .query));
}

test "GLM MLA headbatch prefill transient bill" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    try std.testing.expectEqual(@as(usize, 402653184), try transientBudget(2048, 1));
    try std.testing.expectEqual(@as(usize, 805306368), try transientBudget(2048, 2));
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(32, 2));
    try std.testing.expectError(error.Overflow, transientBudget(2048, std.math.maxInt(usize)));
}
