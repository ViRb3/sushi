//! Retained BF16 KDA projections using the serial GEMV kernel's batch grid.
const std = @import("std");
const mlx = @import("mlx.zig");
const native = @import("glm5_model.zig");

var enabled_cache: ?bool = null;
var calls: usize = 0;
pub fn enabled() bool {
    if (enabled_cache) |value| return value;
    const raw = std.c.getenv("SUSHI_GLM_DFLASH_DENSE_ROWS");
    const value = if (raw) |text| std.mem.eql(u8, std.mem.span(text), "1") else true;
    enabled_cache = value;
    return value;
}
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}

pub fn project(ops: *native.Ops, linear: native.Linear, x: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(ops.s) or x.ctx == null or linear.w.ctx == null or linear.scales.ctx != null or linear.biases.ctx != null) return null;
    const sh = mlx.getShape(x);
    const ws = mlx.getShape(linear.w);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 2 or sh[1] > 4 or ws.len != 2 or ws[1] != sh[2] or linear.input != sh[2] or linear.output != ws[0]) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(linear.w) != .bfloat16) return null;
    if (!((sh[2] == 4096 and (ws[0] == 64 or ws[0] == 128)) or (sh[2] == 128 and ws[0] == 8192))) return null;
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, linear.w));
    if (!available) return null;
    const strides = mlx.mlx_array_strides(linear.w);
    if (strides[1] != 1 or strides[0] != ws[1]) return null;

    // A column-vector batch keeps N=1 and cannot merge into MLX's wide GEMV.
    // Its ordinary GEMV template and reduction order match each serial row.
    const columns = try ops.reshape(x, &.{ sh[1], sh[2], 1 });
    const y = try ops.binary(.mm, linear.w, columns);
    const output = try ops.reshape(y, &.{ 1, sh[1], linear.output });
    calls += 1;
    return output;
}

test "GLM DFlash dense column batch rejects unsupported inputs" {
    var ops = native.Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const x = try ops.ones(&.{ 1, 3, 4096 }, .bfloat16);
    const linear = native.Linear{ .w = try ops.zeros(&.{ 128, 4096 }, .bfloat16), .input = 4096, .output = 128 };
    try mlx.check(mlx.mlx_array_eval(linear.w));
    try std.testing.expect((try project(&ops, linear, try ops.ones(&.{ 1, 1, 4096 }, .bfloat16))) == null);
    try std.testing.expect((try project(&ops, linear, try ops.ones(&.{ 1, 5, 4096 }, .bfloat16))) == null);
    try std.testing.expect((try project(&ops, linear, try ops.ones(&.{ 2, 3, 4096 }, .bfloat16))) == null);
    try std.testing.expect((try project(&ops, linear, try ops.cast(x, .float32))) == null);
    var changed = linear;
    changed.w = try ops.cast(linear.w, .float32);
    try std.testing.expect((try project(&ops, changed, x)) == null);
    changed = linear;
    changed.scales = try ops.ones(&.{ 128, 32 }, .bfloat16);
    try std.testing.expect((try project(&ops, changed, x)) == null);
    changed = linear;
    changed.w = try ops.transpose(try ops.zeros(&.{ 4096, 128 }, .bfloat16), &.{ 1, 0 });
    try mlx.check(mlx.mlx_array_eval(changed.w));
    try std.testing.expect((try project(&ops, changed, x)) == null);
    changed = .{ .w = try ops.zeros(&.{ 256, 4096 }, .bfloat16), .input = 4096, .output = 256 };
    try std.testing.expect((try project(&ops, changed, x)) == null);
    const cpu_stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_stream);
    var cpu = native.Ops{ .s = cpu_stream };
    defer cpu.deinit();
    try std.testing.expect((try project(&cpu, linear, x)) == null);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "GLM fast opt-out dense rows defaults on and preserves explicit controls" {
    const a = std.testing.allocator;
    const name = "SUSHI_GLM_DFLASH_DENSE_ROWS";
    const previous_cache = enabled_cache;
    defer enabled_cache = previous_cache;
    const previous = if (std.c.getenv(name)) |value| try a.dupeSentinel(u8, std.mem.span(value), 0) else null;
    defer {
        if (previous) |value| {
            _ = setenv(name, value, 1);
            a.free(value);
        } else _ = unsetenv(name);
    }
    try std.testing.expectEqual(@as(c_int, 0), unsetenv(name));
    enabled_cache = null;
    try std.testing.expect(enabled());
    try std.testing.expectEqual(@as(c_int, 0), setenv(name, "0", 1));
    enabled_cache = null;
    try std.testing.expect(!enabled());
    try std.testing.expectEqual(@as(c_int, 0), setenv(name, "1", 1));
    enabled_cache = null;
    try std.testing.expect(enabled());
}
