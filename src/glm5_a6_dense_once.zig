//! Opt-in A6 prefill expansion plus dense BF16 GEMM; no persistent weight copy.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };

pub const expanded_weight_bytes: usize = 64 * 1024 * 1024;
var dispatches: usize = 0;
pub fn dispatchCount() usize {
    return dispatches;
}
pub fn resetDispatchCount() void {
    dispatches = 0;
}
pub fn enabled() bool {
    return @import("transformer.zig").diagEnvOn("SUSHI_GLM_A6_DENSE_PREFILL");
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
    if (!enabled() or !mlx.streamIsGpu(ops.s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
    const arrays = [_]Arr{ x, w, scales, biases };
    for (arrays) |v| if (v.ctx == null) return null;
    var dtypes: [4]mlx.mlx_dtype = undefined;
    for (arrays, 0..) |v, i| dtypes[i] = mlx.mlx_array_dtype(v);
    if (!geometry(mlx.getShape(x), mlx.getShape(w), mlx.getShape(scales), mlx.getShape(biases), dtypes)) return null;
    const result = try apply(ops, .{ .x = x, .w = w, .scales = scales, .biases = biases }, true);
    dispatches += 1;
    return result;
}
fn bill(pending_layers: usize) !usize {
    return std.math.mul(usize, expanded_weight_bytes * 4, pending_layers);
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    return if (enabled() and chunk == 2048) bill(pending_layers) else 0;
}
// The four KDA banks remain live until the pending layer's graph is reclaimed.
pub fn budgetFits(active: usize, transient: usize, limit: usize) bool {
    return transient <= limit and active <= limit - transient;
}

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
fn apply(ops: *Ops, input: Input, dense: bool) !Arr {
    if (!dense) return ops.qmm(input.x, input.w, input.scales, input.biases, true);
    const weights = try ops.dequant(input.w, input.scales, input.biases);
    return ops.binary(.mm, input.x, try ops.transpose(weights, &.{ 1, 0 }));
}
fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn roundBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
const WeightStats = struct { values: usize, mismatches: usize, first_expected: u16, first_actual: u16 };

// Native NAX converts BF16 scale/bias to FP32 before this same code*s+b store.
// These finite fixture scales times a six-bit integer have an exact FP32 product.
fn weightStats(input: Input, decoded: Arr) !WeightStats {
    try mlx.check(mlx.mlx_array_eval(decoded));
    const sh = mlx.getShape(decoded);
    try std.testing.expectEqual(@as(usize, 2), sh.len);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(decoded));
    const k: usize = @intCast(sh[1]);
    const n: usize = @intCast(sh[0]);
    const codes: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint32(input.w).?);
    const scales = mlx.mlx_array_data_bfloat16(input.scales).?;
    const biases = mlx.mlx_array_data_bfloat16(input.biases).?;
    const actual = mlx.mlx_array_data_bfloat16(decoded).?;
    var stats = WeightStats{ .values = k * n, .mismatches = 0, .first_expected = 0, .first_actual = 0 };
    for (0..n) |row| for (0..k) |col| {
        const offset = row * (k * 3 / 4) + (col / 4) * 3;
        const pack = @as(u32, codes[offset]) | (@as(u32, codes[offset + 1]) << 8) | (@as(u32, codes[offset + 2]) << 16);
        const code = (pack >> @as(u5, @intCast((col % 4) * 6))) & 63;
        const group = row * (k / 128) + col / 128;
        const expected = roundBf16(@mulAdd(f32, @floatFromInt(code), bf16(scales[group]), bf16(biases[group])));
        const got = actual[row * k + col];
        if (expected != got) {
            if (stats.mismatches == 0) {
                stats.first_expected = expected;
                stats.first_actual = got;
            }
            stats.mismatches += 1;
        }
    };
    return stats;
}
const OutputStats = struct { values: usize, mismatches: usize, max_abs: f64, rms_difference: f64, rms_reference: f64, relative_l2: f64 };
fn outputStats(reference: Arr, candidate: Arr) !OutputStats {
    try mlx.check(mlx.mlx_array_eval(reference));
    try mlx.check(mlx.mlx_array_eval(candidate));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(reference), mlx.getShape(candidate));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(candidate));
    const count = mlx.mlx_array_size(reference);
    const a = mlx.mlx_array_data_bfloat16(reference).?;
    const b = mlx.mlx_array_data_bfloat16(candidate).?;
    var mismatches: usize = 0;
    var difference: f64 = 0;
    var norm: f64 = 0;
    var max: f64 = 0;
    for (0..count) |i| {
        const x: f64 = bf16(a[i]);
        const y: f64 = bf16(b[i]);
        try std.testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
        mismatches += @intFromBool(a[i] != b[i]);
        const delta = x - y;
        difference += delta * delta;
        norm += x * x;
        max = @max(max, @abs(delta));
    }
    const size: f64 = @floatFromInt(count);
    return .{ .values = count, .mismatches = mismatches, .max_abs = max, .rms_difference = @sqrt(difference / size), .rms_reference = @sqrt(norm / size), .relative_l2 = @sqrt(difference / norm) };
}
const ReferenceStats = struct { samples: usize, affine_rms_error: f64, dense_rms_error: f64, affine_max_error: f64, dense_max_error: f64 };
fn sourceReference(input: Input, decoded: Arr, affine: Arr, dense: Arr) !ReferenceStats {
    const k: usize = @intCast(mlx.getShape(decoded)[1]);
    const n: usize = @intCast(mlx.getShape(decoded)[0]);
    const x = mlx.mlx_array_data_bfloat16(input.x).?;
    const w = mlx.mlx_array_data_bfloat16(decoded).?;
    const a = mlx.mlx_array_data_bfloat16(affine).?;
    const b = mlx.mlx_array_data_bfloat16(dense).?;
    var err_a: f64 = 0;
    var err_b: f64 = 0;
    var max_a: f64 = 0;
    var max_b: f64 = 0;
    for (0..8) |ti| for (0..32) |ni| {
        const row = ti * 2047 / 7;
        const col = ni * (n - 1) / 31;
        var expected: f64 = 0;
        for (0..k) |j| expected += @as(f64, bf16(x[row * k + j])) * @as(f64, bf16(w[col * k + j]));
        const da = @as(f64, bf16(a[row * n + col])) - expected;
        const db = @as(f64, bf16(b[row * n + col])) - expected;
        err_a += da * da;
        err_b += db * db;
        max_a = @max(max_a, @abs(da));
        max_b = @max(max_b, @abs(db));
    };
    return .{ .samples = 256, .affine_rms_error = @sqrt(err_a / 256), .dense_rms_error = @sqrt(err_b / 256), .affine_max_error = max_a, .dense_max_error = max_b };
}
fn write(path: [*:0]const u8, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(std.testing.allocator, value, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(text);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = text });
}

test "GLM A6 dense once weight rounding and output qualification" {
    const path = std.c.getenv("SUSHI_GLM_A6_DENSE_PARITY_OUT") orelse return error.SkipZigTest;
    var weights: [2]WeightStats = undefined;
    var outputs: [2]OutputStats = undefined;
    var reference: [2]ReferenceStats = undefined;
    for ([_]c_int{ 4096, 8192 }, 0..) |k, shape| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const input = try fixture(&ops, k, if (k == 4096) 8192 else 4096, @intCast(1009 + shape));
        const decoded = try ops.dequant(input.w, input.scales, input.biases);
        weights[shape] = try weightStats(input, decoded);
        if (weights[shape].mismatches != 0) {
            try write(path, .{ .stage = "weight reconstruction mismatch", .shape = shape, .weights = weights[shape] });
            return error.NativeNaxWeightMismatch;
        }
        const affine = try apply(&ops, input, false);
        const dense = try apply(&ops, input, true);
        outputs[shape] = try outputStats(affine, dense);
        reference[shape] = try sourceReference(input, decoded, affine, dense);
    }
    try write(path, .{ .weights = weights, .outputs = outputs, .sampled_fp64_reference = reference, .m = 2048, .input_widths = .{ 4096, 8192 }, .full_model = false });
}
fn timed(input: Input, dense: bool, repetitions: usize) !u64 {
    const timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    for (0..repetitions) |_| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const output = try apply(&ops, input, dense);
        try mlx.check(mlx.mlx_array_eval(output));
    }
    return timer.read() / repetitions;
}
test "GLM A6 dense once inclusive isolated timing" {
    const path = std.c.getenv("SUSHI_GLM_A6_DENSE_BENCH_OUT") orelse return error.SkipZigTest;
    var samples: [2][2][11]u64 = undefined;
    var output_stats: [2]OutputStats = undefined;
    for ([_]c_int{ 4096, 8192 }, 0..) |k, shape| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const input = try fixture(&ops, k, if (k == 4096) 8192 else 4096, @intCast(1019 + shape));
        {
            var check_ops = Ops{ .s = mlx.gpuStream() };
            defer check_ops.deinit();
            const check = try weightStats(input, try check_ops.dequant(input.w, input.scales, input.biases));
            try std.testing.expectEqual(@as(usize, 0), check.mismatches);
            output_stats[shape] = try outputStats(try apply(&check_ops, input, false), try apply(&check_ops, input, true));
        }
        // Each timed graph reconstructs its own temporary W; this check's W is released first.
        for (0..12) |_| for (0..2) |arm| {
            _ = try timed(input, arm == 1, 1);
        };
        for (0..11) |round| for (0..2) |position| {
            const arm = if (round % 2 == 0) position else 1 - position;
            samples[shape][arm][round] = try timed(input, arm == 1, 4);
        };
    }
    try write(path, .{
        .samples_ns = samples,
        .output_stats = output_stats,
        .m = 2048,
        .input_widths = .{ 4096, 8192 },
        .arms = .{ "affine A6 NAX", "runtime BF16 dequant + dense mm" },
        .warmups = 12,
        .rounds = 11,
        .repetitions = 4,
        .temporary_weight_bytes = 67108864,
        .includes_dequant_allocation_graph_eval_free = true,
        .persistent_expanded_weights = false,
        .full_model = false,
    });
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
    try std.testing.expect(budgetFits(1000, 512, 1512));
    try std.testing.expect(!budgetFits(1000, 512, 1511));
    try std.testing.expect(!budgetFits(0, 512, 511));
}

test "GLM A6 dense once opt-in Linear keeps native bits and declines decode rows" {
    if (!enabled() or !@import("glm5_kda_fused.zig").hardwareSupported()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    for ([_]c_int{ 4096, 8192 }, 0..) |k, shape| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const n: c_int = if (k == 4096) 8192 else 4096;
        const input = try fixture(&ops, k, n, @intCast(1031 + shape));
        const linear = @import("glm5_model.zig").Linear{ .w = input.w, .scales = input.scales, .biases = input.biases, .input = k, .output = n };
        resetDispatchCount();
        const actual = try linear.apply(&ops, input.x);
        const expected = try ops.qmm(input.x, input.w, input.scales, input.biases, true);
        try std.testing.expectEqual(@as(usize, 0), (try outputStats(expected, actual)).mismatches);
        try std.testing.expectEqual(@as(usize, 1), dispatchCount());
        const decode = try ops.slice(input.x, 1, 0, 3);
        const short_actual = try linear.apply(&ops, decode);
        const short_expected = try ops.qmm(decode, input.w, input.scales, input.biases, true);
        try std.testing.expectEqual(@as(usize, 0), (try outputStats(short_expected, short_actual)).mismatches);
        try std.testing.expectEqual(@as(usize, 1), dispatchCount());
    }
    resetDispatchCount();
}
