//! Self-contained directed check of the optional exact HC primitive hook.
const std = @import("std");
const mlx = @import("mlx.zig");
const hc = @import("glm5_next.zig");
const helper = @import("glm5_hc_collapse_simd32.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
fn exact(ops: *Ops, a: Arr, b: Arr) !void {
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    const ev = mlx.mlx_vector_array_new_data(&.{ x, y }, 2);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(x), mlx.getShape(y));
    try std.testing.expectEqual(mlx.mlx_array_dtype(x), mlx.mlx_array_dtype(y));
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_dtype(x) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(x).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(y).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(x).?[0..n], mlx.mlx_array_data_bfloat16(y).?[0..n]);
}
fn check(ops: *Ops, x: Arr, mix: Arr, scale: Arr, base: Arr, expected_calls: usize) !void {
    const old = blk: {
        const binding = helper.bind(false);
        defer binding.restore();
        helper.resetDispatchCount();
        break :blk try hc.hcCollapse(x, mix, scale, base, 20, 1e-6, ops.s);
    };
    defer old.deinit();
    try std.testing.expectEqual(@as(usize, 0), helper.dispatchCount());
    const binding = helper.bind(true);
    defer binding.restore();
    const new = try hc.hcCollapse(x, mix, scale, base, 20, 1e-6, ops.s);
    defer new.deinit();
    try std.testing.expectEqual(expected_calls, helper.dispatchCount());
    try exact(ops, old.mixed, new.mixed);
    try exact(ops, old.post, new.post);
    try exact(ops, old.comb, new.comb);
}
test "GLM HC SIMD32 scoped policy restores and resets counters" {
    const off = helper.bind(false);
    defer off.restore();
    try std.testing.expect(!helper.enabled());
    const on = helper.bind(true);
    try std.testing.expect(helper.enabled());
    on.restore();
    try std.testing.expect(!helper.enabled());
    helper.resetDispatchCount();
    try std.testing.expectEqual(@as(usize, 0), helper.dispatchCount());
}
test "GLM HC SIMD32 primitive hook exact T3 specials and original fallback" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const x = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 3, 4, 4096 }, 731, ops.s));
    var values: [72]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 8 - 1;
    const mix = try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 24 }, 3, .float32));
    const scale = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.125, 0.25, 0.0625 }, &.{3}, 1, .float32));
    var bases: [24]f32 = undefined;
    for (&bases, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 16 - 0.25;
    const base = try ops.own(mlx.mlx_array_new_data(&bases, &.{24}, 1, .float32));
    try check(&ops, x, mix, scale, base, 1);
    values[0] = -0.0;
    values[1] = 0.0;
    values[8] = std.math.nan(f32);
    values[9] = std.math.inf(f32);
    values[10] = -std.math.inf(f32);
    values[24 + 8] = 1000;
    values[24 + 9] = -1000;
    const special = try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 24 }, 3, .float32));
    try check(&ops, x, special, scale, base, 1);
    try check(&ops, try ops.slice(x, 1, 0, 2), try ops.slice(mix, 1, 0, 2), scale, base, 0);
    try check(&ops, try ops.cast(x, .float32), mix, scale, base, 0);
}
