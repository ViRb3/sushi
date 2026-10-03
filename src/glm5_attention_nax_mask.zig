//! Isolated indexed-latent NAX attention; no expanded KV bank or score tensor.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const Precision = enum { bf16, float32 };
pub const mask_limit: usize = 2 * 1024 * 1024;
pub fn temporaryBytes(rows: usize, history: usize, precision: Precision) !usize {
    if (rows > try maxRows(history)) return error.UnsupportedGlmNaxAttention;
    const plane = try std.math.mul(usize, rows, history + 1);
    const masks = try std.math.add(usize, try std.math.mul(usize, plane, 2), try std.math.mul(usize, rows, 2051 * 9 + 5));
    const bytes: usize = if (precision == .float32) 4 else 2;
    const copies = try std.math.mul(usize, rows, 64 * 512 * bytes * 4);
    const cache_cast = if (precision == .float32) try std.math.mul(usize, history, 512 * 4) else 0;
    return std.math.add(usize, masks, try std.math.add(usize, copies, cache_cast));
}
pub fn maxRows(history: usize) !usize {
    if (history == 0 or history > 65536) return error.UnsupportedGlmNaxAttention;
    return @min(128, mask_limit / (history + 1));
}
fn validGeometry(q: []const c_int, cache: []const c_int, selected: []const c_int, offset: usize, history: usize) bool {
    const rows = maxRows(history) catch return false;
    return q.len == 3 and q[0] > 8 and q[0] <= rows and q[1] == 64 and q[2] == 512 and
        cache.len == 2 and cache[0] >= history and cache[1] == 512 and selected.len == 2 and
        selected[0] == q[0] and selected[1] == 2051 and offset <= history and q[0] <= history - offset;
}
/// Valid selected IDs are unique per query, as guaranteed by IndexPool selection.
fn membership(ops: *Ops, ids: Arr, offset: usize, history: usize) !Arr {
    const rows = mlx.getShape(ids)[0];
    const nonnegative = try ops.slot();
    try mlx.check(mlx.mlx_greater_equal(nonnegative, ids, try ops.scalar(0, .int32), ops.s));
    const bounded = try ops.slot();
    try mlx.check(mlx.mlx_less(bounded, ids, try ops.scalar(@floatFromInt(history), .int32), ops.s));
    const positions = try ops.slot();
    try mlx.check(mlx.mlx_arange(positions, @floatFromInt(offset), @floatFromInt(offset + @as(usize, @intCast(rows))), 1, .int32, ops.s));
    const causal = try ops.slot();
    try mlx.check(mlx.mlx_less_equal(causal, ids, try ops.reshape(positions.*, &.{ rows, 1 }), ops.s));
    const in_range = try ops.slot();
    try mlx.check(mlx.mlx_logical_and(in_range, nonnegative.*, bounded.*, ops.s));
    const valid = try ops.slot();
    try mlx.check(mlx.mlx_logical_and(valid, in_range.*, causal.*, ops.s));
    const safe = try ops.slot();
    // Invalid slots write a separate dummy column, never a valid key's mask bit.
    try mlx.check(mlx.mlx_where(safe, valid.*, ids, try ops.scalar(@floatFromInt(history), .int32), ops.s));
    const zero = try ops.broadcast(try ops.scalar(0, .bool_), &.{ rows, @intCast(history + 1) });
    const mask = try ops.slot();
    try mlx.check(mlx.mlx_put_along_axis(mask, zero, safe.*, valid.*, 1, ops.s));
    return ops.slice(mask.*, 1, 0, @intCast(history));
}
pub fn run(ops: *Ops, q: Arr, cache: Arr, selected: Arr, offset: usize, history: usize, scale: f32, precision: Precision) !?Arr {
    if (!std.math.isFinite(scale) or scale <= 0) return null;
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
    for ([_]Arr{ q, cache, selected }) |x| if (x.ctx == null) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(cache) != .bfloat16 or mlx.mlx_array_dtype(selected) != .int32 or
        !validGeometry(mlx.getShape(q), mlx.getShape(cache), mlx.getShape(selected), offset, history)) return null;
    const dtype: mlx.mlx_dtype = if (precision == .bf16) .bfloat16 else .float32;
    const rows = mlx.getShape(q)[0];
    const queries = try ops.reshape(try ops.transpose(try ops.cast(q, dtype), &.{ 1, 0, 2 }), &.{ 1, 64, rows, 512 });
    const kv = try ops.reshape(try ops.cast(try ops.slice(cache, 0, 0, @intCast(history)), dtype), &.{ 1, 1, @intCast(history), 512 });
    const mask = try membership(ops, selected, offset, history);
    const out = try ops.slot();
    // Masked D512 normally selects an unfused fallback; force its supported NAX path.
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(out, queries, kv, kv, scale, "array", try ops.reshape(mask, &.{ 1, 1, rows, @intCast(history) }), .{ .ctx = null }, true, ops.s));
    const populated = try ops.slot();
    try mlx.check(mlx.mlx_any_axes(populated, mask, &.{1}, 1, true, ops.s));
    const ordered = try ops.reshape(try ops.transpose(out.*, &.{ 0, 2, 1, 3 }), &.{ rows, 64, 512 });
    const safe = try ops.slot();
    try mlx.check(mlx.mlx_where(safe, try ops.reshape(populated.*, &.{ rows, 1, 1 }), ordered, try ops.scalar(0, dtype), ops.s));
    return try ops.cast(safe.*, .bfloat16);
}
test "GLM NAX mask geometry bounds one membership plane" {
    try std.testing.expectEqual(@as(usize, 63), try maxRows(32768));
    try std.testing.expectEqual(@as(usize, 31), try maxRows(65536));
    try std.testing.expect(!validGeometry(&.{ 128, 64, 512 }, &.{ 32768, 512 }, &.{ 128, 2051 }, 16000, 32768));
    try std.testing.expect(!validGeometry(&.{ 8, 64, 512 }, &.{ 4096, 512 }, &.{ 8, 2051 }, 3072, 4096));
    try std.testing.expect(validGeometry(&.{ 63, 64, 512 }, &.{ 32768, 512 }, &.{ 63, 2051 }, 16000, 32768));
}
test "GLM NAX membership invalid slots cannot clear key zero" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const data = [_]i32{ 0, -1, 1, 7, 3, -1, 0, 1 };
    const ids = try ops.own(mlx.mlx_array_new_data(&data, &.{ 2, 4 }, 2, .int32));
    const mask = try ops.contiguous(try membership(&ops, ids, 1, 4));
    try mlx.check(mlx.mlx_array_eval(mask));
    const expected = [_]bool{ true, true, false, false, true, true, false, false };
    try std.testing.expectEqualSlices(bool, &expected, mlx.mlx_array_data_bool(mask).?[0..8]);
}
