//! Isolated MLA head-batched affine prefill; original banks/cache precision stay unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const Direction = enum { query, value };
pub const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };

threadlocal var enabled_override: ?bool = null;
var calls: [2]usize = .{ 0, 0 };
pub const Binding = struct {
    previous: ?bool,
    pub fn restore(self: Binding) void {
        enabled_override = self.previous;
    }
};
pub fn bind(on: bool) Binding {
    const result = Binding{ .previous = enabled_override };
    enabled_override = on;
    return result;
}
pub fn enabled() bool {
    return enabled_override orelse @import("transformer.zig").diagEnvOn("SUSHI_GLM_MLA_PREFILL_BATCH");
}
pub fn resetDispatchCount() void {
    calls = .{ 0, 0 };
}
pub fn dispatchCount(dir: Direction) usize {
    return calls[@backingInt(dir)];
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!enabled() or chunk < 128 or chunk > 2048) return 0;
    // Input/output permutation copies total64H*(256+512+512+256)*2 bytes/row.
    return std.math.mul(usize, try std.math.mul(usize, chunk, 196608), pending_layers);
}

fn geometry(xs: []const c_int, ws: []const c_int, ss: []const c_int, bs: []const c_int, dtype: mlx.mlx_dtype, dir: Direction) bool {
    const width: c_int = if (dir == .query) 256 else 512;
    if (xs.len != 4 or xs[0] < 128 or xs[0] > 2048 or xs[1] != 64 or xs[2] != 1 or xs[3] != width or dtype != .bfloat16) return false;
    return std.mem.eql(c_int, &.{ 64, 256, 96 }, ws) and std.mem.eql(c_int, &.{ 64, 256, 4 }, ss) and std.mem.eql(c_int, ss, bs);
}
fn project(ops: *Ops, input: Input, dir: Direction, batched: bool) !Arr {
    if (!batched) return ops.qmm(input.x, input.w, input.scales, input.biases, dir == .value);
    const xs = mlx.getShape(input.x);
    const out: c_int = if (dir == .query) 512 else 256;
    const heads = try ops.contiguous(try ops.reshape(try ops.transpose(input.x, &.{ 1, 0, 2, 3 }), &.{ 64, xs[0], xs[3] }));
    const y = try ops.qmm(heads, input.w, input.scales, input.biases, dir == .value);
    return ops.reshape(try ops.contiguous(try ops.transpose(y, &.{ 1, 0, 2 })), &.{ xs[0], 64, 1, out });
}
pub fn run(ops: *Ops, input: Input, dir: Direction) !?Arr {
    if (!enabled() or !mlx.streamIsGpu(ops.s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
    for ([_]Arr{ input.x, input.w, input.scales, input.biases }) |v| if (v.ctx == null) return null;
    if (!geometry(mlx.getShape(input.x), mlx.getShape(input.w), mlx.getShape(input.scales), mlx.getShape(input.biases), mlx.mlx_array_dtype(input.x), dir) or
        mlx.mlx_array_dtype(input.w) != .uint32 or mlx.mlx_array_dtype(input.scales) != .bfloat16 or mlx.mlx_array_dtype(input.biases) != .bfloat16) return null;
    const output = try project(ops, input, dir, true);
    calls[@backingInt(dir)] += 1;
    return output;
}
fn fixture(ops: *Ops, tokens: c_int, dtype: mlx.mlx_dtype, dir: Direction, seed: u64) !Input {
    const width: c_int = if (dir == .query) 256 else 512;
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ tokens, 64, 1, width }, 4, dtype, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ 64, 256, 96 }, 3, 4, key.*, ops.s));
    const sc = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(sc, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.01, .bfloat16), &.{ 64, 256, 4 }, 3, .bfloat16, key.*, ops.s));
    const bi = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(bi, try ops.scalar(-0.3, .bfloat16), try ops.scalar(0.1, .bfloat16), &.{ 64, 256, 4 }, 3, .bfloat16, key.*, ops.s));
    for ([_]Arr{ x.*, w.*, sc.*, bi.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
    return .{ .x = x.*, .w = w.*, .scales = sc.*, .biases = bi.* };
}
fn bf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}
fn roundBf(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    return bf16(@truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16));
}
fn ordered(b: u16) u32 {
    return if (b & 0x8000 != 0) 0x8000 - @as(u32, b & 0x7fff) else 0x8000 + @as(u32, b);
}
const Drift = struct { values: usize, raw_mismatches: usize, max_bf16_ulp: u32, ulp_bins: [5]usize, max_abs: f64, rms_difference: f64, relative_l2: f64 };
fn value(a: Arr, i: usize) f32 {
    return if (mlx.mlx_array_dtype(a) == .bfloat16) bf16(mlx.mlx_array_data_bfloat16(a).?[i]) else mlx.mlx_array_data_float32(a).?[i];
}
fn drift(a: Arr, b: Arr) !Drift {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    const n = mlx.mlx_array_size(a);
    var d = Drift{ .values = n, .raw_mismatches = 0, .max_bf16_ulp = 0, .ulp_bins = @splat(0), .max_abs = 0, .rms_difference = 0, .relative_l2 = 0 };
    var square: f64 = 0;
    var norm: f64 = 0;
    for (0..n) |i| {
        const x: f64 = value(a, i);
        const y: f64 = value(b, i);
        try std.testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
        const delta = x - y;
        square += delta * delta;
        norm += x * x;
        d.max_abs = @max(d.max_abs, @abs(delta));
        if (mlx.mlx_array_dtype(a) == .bfloat16) {
            const av = mlx.mlx_array_data_bfloat16(a).?[i];
            const bv = mlx.mlx_array_data_bfloat16(b).?[i];
            d.raw_mismatches += @intFromBool(av != bv);
            const ao = ordered(av);
            const bo = ordered(bv);
            const ulp = if (ao > bo) ao - bo else bo - ao;
            d.max_bf16_ulp = @max(d.max_bf16_ulp, ulp);
            d.ulp_bins[if (ulp == 0) 0 else if (ulp == 1) 1 else if (ulp == 2) 2 else if (ulp <= 8) 3 else 4] += 1;
        } else d.raw_mismatches += @intFromBool(@as(u32, @bitCast(value(a, i))) != @as(u32, @bitCast(value(b, i))));
    }
    d.rms_difference = @sqrt(square / @as(f64, @floatFromInt(n)));
    d.relative_l2 = @sqrt(square / norm);
    return d;
}
const Reference = struct { samples: usize, native_vs_fp32_weight_rms: f64, batch_vs_fp32_weight_rms: f64, native_vs_bf16_weight_rms: f64, batch_vs_bf16_weight_rms: f64, weight_boundary_rms: f64 };
fn reference(input: Input, dir: Direction, a: Arr, b: Arr) !Reference {
    const xs = mlx.getShape(input.x);
    const tokens: usize = @intCast(xs[0]);
    const k: usize = @intCast(xs[3]);
    const n: usize = if (dir == .query) 512 else 256;
    const codes: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint32(input.w).?);
    const scales = mlx.mlx_array_data_bfloat16(input.scales).?;
    const biases = mlx.mlx_array_data_bfloat16(input.biases).?;
    var e: [5]f64 = @splat(0);
    for (0..8) |ti| for (0..8) |hi| for (0..4) |ni| {
        const row = ti * (tokens - 1) / 7;
        const head = hi * 63 / 7;
        const col = ni * (n - 1) / 3;
        var exact: f64 = 0;
        var rounded: f64 = 0;
        for (0..k) |j| {
            const wr = if (dir == .query) j else col;
            const wc = if (dir == .query) col else j;
            const offset = (head * 256 + wr) * 384 + (wc / 4) * 3;
            const pack = @as(u32, codes[offset]) | (@as(u32, codes[offset + 1]) << 8) | (@as(u32, codes[offset + 2]) << 16);
            const code = (pack >> @as(u5, @intCast(wc % 4 * 6))) & 63;
            const group = (head * 256 + wr) * 4 + wc / 128;
            const w = @mulAdd(f32, @floatFromInt(code), bf16(scales[group]), bf16(biases[group]));
            const xv: f64 = value(input.x, (row * 64 + head) * k + j);
            exact += xv * @as(f64, w);
            rounded += xv * @as(f64, roundBf(w));
        }
        const av: f64 = value(a, (row * 64 + head) * n + col);
        const bv: f64 = value(b, (row * 64 + head) * n + col);
        const delta = [_]f64{ av - exact, bv - exact, av - rounded, bv - rounded, exact - rounded };
        for (delta, 0..) |d, i| e[i] += d * d;
    };
    return .{ .samples = 256, .native_vs_fp32_weight_rms = @sqrt(e[0] / 256), .batch_vs_fp32_weight_rms = @sqrt(e[1] / 256), .native_vs_bf16_weight_rms = @sqrt(e[2] / 256), .batch_vs_bf16_weight_rms = @sqrt(e[3] / 256), .weight_boundary_rms = @sqrt(e[4] / 256) };
}
fn write(path: [*:0]const u8, v: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(std.testing.allocator, v, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(text);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = text });
}

test "GLM MLA headbatch prefill geometry guards original banks" {
    try std.testing.expect(geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(geometry(&.{ 128, 64, 1, 512 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .value));
    for ([_]c_int{ 1, 3, 16, 32, 127, 2049 }) |t| try std.testing.expect(!geometry(&.{ t, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(!geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 128 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(!geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .float32, .query));
}
test "GLM MLA headbatch prefill BF16 ULP and F32 reference qualification" {
    const path = std.c.getenv("SUSHI_GLM_MLA_BATCH_PARITY_OUT") orelse return error.SkipZigTest;
    var differences: [2][2][3]Drift = undefined;
    var refs: [2][2][3]Reference = undefined;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }, 0..) |dtype, di| for ([_]Direction{ .query, .value }, 0..) |dir, projection| for ([_]c_int{ 128, 512, 2048 }, 0..) |t, shape| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const input = try fixture(&ops, t, dtype, dir, @intCast(1103 + di * 10 + projection));
        const a = try project(&ops, input, dir, false);
        const b = try project(&ops, input, dir, true);
        differences[di][projection][shape] = try drift(a, b);
        refs[di][projection][shape] = try reference(input, dir, a, b);
    };
    try write(path, .{ .drift = differences, .reference = refs, .dtypes = .{ "BF16", "F32" }, .directions = .{ "query256to512nontransposed", "value512to256transposed" }, .rows = .{ 128, 512, 2048 }, .ulp_bins = "0,1,2,3to8,over8", .weight_storage = "originalU32A6/group128 BF16scale/bias", .cache_change = false, .tolerance_relaxed = false, .full_model = false });
}
fn timed(input: Input, dir: Direction, batched: bool, repetitions: usize) !u64 {
    const timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    for (0..repetitions) |_| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const y = try project(&ops, input, dir, batched);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    return timer.read() / repetitions;
}
test "GLM MLA headbatch prefill inclusive paired timing" {
    const path = std.c.getenv("SUSHI_GLM_MLA_BATCH_BENCH_OUT") orelse return error.SkipZigTest;
    var samples: [2][3][2][11]u64 = undefined;
    var differences: [2][3]Drift = undefined;
    for ([_]Direction{ .query, .value }, 0..) |dir, projection| for ([_]c_int{ 128, 512, 2048 }, 0..) |t, shape| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const input = try fixture(&ops, t, .bfloat16, dir, @intCast(1123 + projection));
        {
            var check = Ops{ .s = ops.s };
            defer check.deinit();
            differences[projection][shape] = try drift(try project(&check, input, dir, false), try project(&check, input, dir, true));
        }
        for (0..6) |_| for (0..2) |arm| {
            _ = try timed(input, dir, arm == 1, 1);
        };
        for (0..11) |round| for (0..2) |position| {
            const arm = if (round % 2 == 0) position else 1 - position;
            samples[projection][shape][arm][round] = try timed(input, dir, arm == 1, 2);
        };
    };
    try write(path, .{ .nanoseconds = samples, .drift = differences, .rows = .{ 128, 512, 2048 }, .arms = .{ "nativeM1qvmqmv", "headbatchednativeQMM" }, .warmups = 6, .rounds = 11, .repetitions = 2, .timing = "apply, all input/output contiguouscopies, nativeQMM, evaluation, frees", .expanded_weights = false, .full_model = false });
}

test "GLM MLA headbatch prefill scoped binding and transient bill" {
    const off = bind(false);
    defer off.restore();
    try std.testing.expect(!enabled());
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(2048, 2));
    {
        const on = bind(true);
        defer on.restore();
        try std.testing.expect(enabled());
        try std.testing.expectEqual(@as(usize, 402653184), try transientBudget(2048, 1));
        try std.testing.expectEqual(@as(usize, 805306368), try transientBudget(2048, 2));
        try std.testing.expectEqual(@as(usize, 0), try transientBudget(32, 2));
        try std.testing.expectError(error.Overflow, transientBudget(2048, std.math.maxInt(usize)));
    }
    try std.testing.expect(!enabled());
}
