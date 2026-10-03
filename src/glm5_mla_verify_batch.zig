//! Isolated three-row MLA native-QMM batching; no verifier callsite change.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const Direction = enum { query, value };
pub const Arm = enum { serial_calls, native_broadcast, head_rows };
pub const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };
threadlocal var enabled_override: ?bool = null;
var calls: [2]usize = .{ 0, 0 };
pub const Binding = struct {
    previous: ?bool,
    pub fn restore(self: Binding) void { enabled_override = self.previous; }
};
pub fn bind(on: bool) Binding {
    const b = Binding{ .previous = enabled_override };
    enabled_override = on;
    return b;
}
pub fn enabled() bool {
    return enabled_override orelse @import("transformer.zig").diagEnvOn("SUSHI_GLM_VERIFY_MLA_BATCH");
}
pub fn resetDispatchCount() void { calls = .{ 0, 0 }; }
pub fn dispatchCount(dir: Direction) usize { return calls[@backingInt(dir)]; }
/// Both banks keep exact M1 broadcast geometry. The changed-rounding M3
/// research arm below is not used by this helper.
pub fn run(ops: *Ops, in: Input, dir: Direction) !?Arr {
    if (!enabled() or !mlx.streamIsGpu(ops.s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
    for ([_]Arr{ in.x, in.w, in.scales, in.biases }) |a| if (a.ctx == null) return null;
    const xs = mlx.getShape(in.x);
    if (!std.mem.eql(c_int, &.{ 3, 64, 1, if (dir == .query) @as(c_int, 256) else 512 }, xs) or
        mlx.mlx_array_dtype(in.x) != .bfloat16 or mlx.mlx_array_dtype(in.w) != .uint32 or
        mlx.mlx_array_dtype(in.scales) != .bfloat16 or mlx.mlx_array_dtype(in.biases) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 64, 256, 96 }, mlx.getShape(in.w)) or !std.mem.eql(c_int, &.{ 64, 256, 4 }, mlx.getShape(in.scales)) or
        !std.mem.eql(c_int, mlx.getShape(in.scales), mlx.getShape(in.biases))) return null;
    const out = try project(ops, in, dir, .native_broadcast);
    calls[@backingInt(dir)] += 1;
    return out;
}
pub fn project(ops: *Ops, in: Input, dir: Direction, arm: Arm) !Arr {
    const sh = mlx.getShape(in.x);
    if (sh.len != 4 or sh[0] != 3 or sh[1] != 64 or sh[2] != 1 or sh[3] != (if (dir == .query) @as(c_int, 256) else 512) or
        mlx.mlx_array_dtype(in.x) != .bfloat16 or !std.mem.eql(c_int, &.{ 64, 256, 96 }, mlx.getShape(in.w)) or
        !std.mem.eql(c_int, &.{ 64, 256, 4 }, mlx.getShape(in.scales)) or !std.mem.eql(c_int, mlx.getShape(in.scales), mlx.getShape(in.biases))) return error.UnsupportedGlmVerifyBatch;
    if (arm == .native_broadcast) return ops.qmm(in.x, in.w, in.scales, in.biases, dir == .value);
    if (arm == .serial_calls) {
        var out: [3]Arr = undefined;
        for (0..3) |row| {
            const one = try ops.slice(in.x, 0, @intCast(row), @intCast(row + 1));
            out[row] = try ops.qmm(one, in.w, in.scales, in.biases, dir == .value);
        }
        return ops.concat(&out, 0);
    }
    const heads = try ops.contiguous(try ops.reshape(try ops.transpose(in.x, &.{ 1, 0, 2, 3 }), &.{ 64, 3, sh[3] }));
    const out = try ops.qmm(heads, in.w, in.scales, in.biases, dir == .value);
    return ops.reshape(try ops.contiguous(try ops.transpose(out, &.{ 1, 0, 2 })), &.{ 3, 64, 1, if (dir == .query) @as(c_int, 512) else 256 });
}
fn fixture(ops: *Ops, dir: Direction) !Input {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, if (dir == .query) 1903 else 1913));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ 3, 64, 1, if (dir == .query) @as(c_int, 256) else 512 }, 4, .bfloat16, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ 64, 256, 96 }, 3, 4, key.*, ops.s));
    const sc = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(sc, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.01, .bfloat16), &.{ 64, 256, 4 }, 3, .bfloat16, key.*, ops.s));
    const bi = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(bi, try ops.scalar(-0.3, .bfloat16), try ops.scalar(0.1, .bfloat16), &.{ 64, 256, 4 }, 3, .bfloat16, key.*, ops.s));
    const values = mlx.mlx_vector_array_new_data(&.{ x.*, w.*, sc.*, bi.* }, 4);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    return .{ .x = x.*, .w = w.*, .scales = sc.*, .biases = bi.* };
}
const Drift = struct { values: usize = 0, bit_mismatches: usize = 0, max_abs: f64 = 0, relative_l2: f64 = 0 };
fn compare(a: Arr, b: Arr) !Drift {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const av = mlx.mlx_array_data_bfloat16(a).?;
    const bv = mlx.mlx_array_data_bfloat16(b).?;
    var d = Drift{ .values = mlx.mlx_array_size(a) };
    var error_sum: f64 = 0;
    var norm: f64 = 0;
    for (0..d.values) |i| {
        const x: f32 = @bitCast(@as(u32, av[i]) << 16);
        const y: f32 = @bitCast(@as(u32, bv[i]) << 16);
        try std.testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
        d.bit_mismatches += @intFromBool(av[i] != bv[i]);
        const delta: f64 = @as(f64, x) - @as(f64, y);
        error_sum += delta * delta;
        norm += @as(f64, x) * @as(f64, x);
        d.max_abs = @max(d.max_abs, @abs(delta));
    }
    d.relative_l2 = @sqrt(error_sum / norm);
    return d;
}
fn timed(in: Input, dir: Direction, arm: Arm) !u64 {
    const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
    var ops = Ops{ .s = mlx.gpuStream() };
    errdefer ops.deinit();
    const out = try project(&ops, in, dir, arm);
    try mlx.check(mlx.mlx_array_eval(out));
    ops.deinit();
    return watch.read();
}
test "GLM MLA VERIFY3 native QMM batching reference and timing" {
    const path = std.c.getenv("SUSHI_GLM_MLA_VERIFY_BATCH_OUT") orelse return error.SkipZigTest;
    var drifts: [2][2]Drift = undefined;
    var times: [2][3][11]u64 = undefined;
    for ([_]Direction{ .query, .value }, 0..) |dir, projection| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const in = try fixture(&ops, dir);
        const reference = try project(&ops, in, dir, .serial_calls);
        const broadcast = try project(&ops, in, dir, .native_broadcast);
        const heads = try project(&ops, in, dir, .head_rows);
        drifts[projection][0] = try compare(reference, broadcast);
        drifts[projection][1] = try compare(reference, heads);
        try std.testing.expectEqual(@as(usize, 0), drifts[projection][0].bit_mismatches);
        for (0..3) |_| for ([_]Arm{ .serial_calls, .native_broadcast, .head_rows }) |arm| {
            _ = try timed(in, dir, arm);
        };
        for (0..11) |round| for (0..3) |position| {
            const arm = if (round % 2 == 0) position else 2 - position;
            times[projection][arm][round] = try timed(in, dir, @enumFromInt(arm));
        };
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .rows = 3, .heads = 64, .banks = "original A6g128 U32/BF16 scale/bias", .arms = .{ "three serial nativecalls", "one native M1 broadcast exactgeometry", "headM3 qvm/query andqmv_wide/value" }, .drift = drifts, .nanoseconds = times, .rounds = 11, .warmups = 3, .timing = "fresh graph; slice/concat or headtranspose/copies/native QMM/eval/free included", .runtime_hook = false }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = json });
}
