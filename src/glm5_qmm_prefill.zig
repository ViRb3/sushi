//! Unintegrated affine8/group128 NAX scheduling experiment. No dequantization work is removed.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
const HEADER = @embedFile("kernels/glm5_qmm_prefill_header.metal");
const SOURCE =
    \\uint3 tile = threadgroup_position_in_grid;
    \\tile.y = tile.y * uint(GROUP_M) + tile.x % uint(GROUP_M);
    \\tile.x /= uint(GROUP_M);
    \\threadgroup bfloat16_t Ws[64 * 72];
    \\qmm_t_nax_tgp_impl<bfloat16_t,128,8,true,64,64,64,2,2>(
    \\  w, scales, biases, x, y, Ws, K, N, M, tile,
    \\  thread_index_in_threadgroup, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };

fn ready(a: Arr) !bool {
    if (a.ctx == null) return false;
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, a));
    if (!available) return false;
    const sh = mlx.getShape(a);
    const strides = mlx.mlx_array_strides(a);
    var expected: usize = 1;
    var i = sh.len;
    while (i > 0) {
        i -= 1;
        if (sh[i] < 1 or strides[i] != expected) return false;
        expected = try std.math.mul(usize, expected, @intCast(sh[i]));
    }
    return true;
}

pub fn apply(s: mlx.mlx_stream, input: Input, group_m: u32) !?Arr {
    if (!mlx.streamIsGpu(s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
    if (group_m != 1 and group_m != 2 and group_m != 4 and group_m != 8) return error.InvalidQmmSchedule;
    for ([_]Arr{ input.x, input.w, input.scales, input.biases }) |a| if (!try ready(a)) return null;
    const xs = mlx.getShape(input.x);
    const ws = mlx.getShape(input.w);
    if (xs.len != 3 or xs[0] != 1 or xs[1] != 512 or ws.len != 2) return null;
    const k = xs[2];
    const n = ws[0];
    if (!((k == 4096 and n == 8192) or (k == 8192 and n == 4096)) or ws[1] != @divExact(k, 4)) return null;
    const ss = [_]c_int{ n, @divExact(k, 128) };
    if (!std.mem.eql(c_int, &ss, mlx.getShape(input.scales)) or !std.mem.eql(c_int, &ss, mlx.getShape(input.biases)) or
        mlx.mlx_array_dtype(input.x) != .bfloat16 or mlx.mlx_array_dtype(input.w) != .uint32 or
        mlx.mlx_array_dtype(input.scales) != .bfloat16 or mlx.mlx_array_dtype(input.biases) != .bfloat16) return null;
    if (kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "w", "scales", "biases", "x", "K", "N", "M" }, 7);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"y"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const value = mlx.mlx_fast_metal_kernel_new("sushi_glm_qmm_prefill_schedule", ins, outs, SOURCE, HEADER, false, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        kernel = value;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 512, n }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GROUP_M", @intCast(group_m)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(n, 64) * @as(c_int, @intCast(group_m)) * 32, @intCast(16 / group_m), 2));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 2));
    const ka = mlx.mlx_array_new_int(k);
    defer _ = mlx.mlx_array_free(ka);
    const na = mlx.mlx_array_new_int(n);
    defer _ = mlx.mlx_array_free(na);
    const ma = mlx.mlx_array_new_int(512);
    defer _ = mlx.mlx_array_free(ma);
    const inputs = mlx.mlx_vector_array_new_data(&.{ input.w, input.scales, input.biases, input.x, ka, na, ma }, 7);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.?, inputs, cfg, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    return y;
}

fn fixture(ops: *Ops, k: c_int, n: c_int, seed: u64) !Input {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ 1, 512, k }, 3, .bfloat16, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ n, @divExact(k, 4) }, 2, 4, key.*, ops.s));
    const scales = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(scales, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.01, .bfloat16), &.{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    const biases = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(biases, try ops.scalar(-1.2, .bfloat16), try ops.scalar(-0.1, .bfloat16), &.{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    for ([_]Arr{ x.*, w.*, scales.*, biases.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
    return .{ .x = x.*, .w = w.*, .scales = scales.*, .biases = biases.* };
}

fn native(s: mlx.mlx_stream, in: Input) !Arr {
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_quantized_matmul(&result, in.x, in.w, in.scales, in.biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", s));
    return result;
}

fn exact(a: Arr, b: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(a));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(b));
    const n = mlx.mlx_array_size(a);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
}

test "GLM QMM prefill G1 matches native at QKV and output geometry" {
    const s = mlx.gpuStream();
    for ([_]c_int{ 4096, 8192 }) |k| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const input = try fixture(&ops, k, if (k == 4096) 8192 else 4096, 19);
        const reference = try native(s, input);
        defer _ = mlx.mlx_array_free(reference);
        const got = (try apply(s, input, 1)) orelse return error.TestExpectedQmm;
        defer _ = mlx.mlx_array_free(got);
        try exact(reference, got);
    }
}

test "GLM QMM prefill swizzles preserve every native BF16 output bit" {
    const s = mlx.gpuStream();
    for ([_]c_int{ 4096, 8192 }) |k| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const input = try fixture(&ops, k, if (k == 4096) 8192 else 4096, 211);
        const reference = try native(s, input);
        defer _ = mlx.mlx_array_free(reference);
        for ([_]u32{ 2, 4, 8 }) |group| {
            const got = (try apply(s, input, group)) orelse return error.TestExpectedQmm;
            defer _ = mlx.mlx_array_free(got);
            try exact(reference, got);
        }
    }
}

test "GLM QMM prefill rejects unsupported layouts without materialization" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const input = try fixture(&ops, 4096, 8192, 23);
    try std.testing.expectError(error.InvalidQmmSchedule, apply(s, input, 3));
    var changed = input;
    changed.w = try ops.transpose(input.w, &.{ 1, 0 });
    try mlx.check(mlx.mlx_array_eval(changed.w));
    try std.testing.expect((try apply(s, changed, 1)) == null);
    changed = input;
    changed.x = try ops.slice(input.x, 1, 0, 128);
    try mlx.check(mlx.mlx_array_eval(changed.x));
    try std.testing.expect((try apply(s, changed, 1)) == null);
    changed = input;
    changed.scales = try ops.cast(input.scales, .float32);
    try std.testing.expect((try apply(s, changed, 1)) == null);
    try mlx.check(mlx.mlx_array_eval(changed.scales));
    try std.testing.expect((try apply(s, changed, 1)) == null);
}

fn arm(s: mlx.mlx_stream, input: Input, group: u32) !Arr {
    return if (group == 0) try native(s, input) else (try apply(s, input, group)) orelse error.TestExpectedQmm;
}

fn timed(s: mlx.mlx_stream, banks: []const Input, group: u32, start: usize, repetitions: usize) !u64 {
    const timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    for (0..repetitions) |i| {
        const y = try arm(s, banks[(start + i) % banks.len], group);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    return timer.read() / repetitions;
}

test "GLM QMM prefill isolated scheduling timing" {
    const path = std.c.getenv("SUSHI_GLM_QMM_PREFILL_BENCH_OUT") orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    const groups = [_]u32{ 0, 1, 2, 4, 8 };
    const widths = [_]c_int{ 4096, 8192 };
    var samples: [2][2][11][groups.len]u64 = undefined;
    for (widths, 0..) |k, geometry| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var banks: [4]Input = undefined;
        for (&banks, 0..) |*bank, bi| bank.* = try fixture(&ops, k, if (k == 4096) 8192 else 4096, @intCast(531 + bi));
        for (banks) |bank| {
            const expected = try native(s, bank);
            defer _ = mlx.mlx_array_free(expected);
            for (groups[1..]) |group| {
                const y = try arm(s, bank, group);
                defer _ = mlx.mlx_array_free(y);
                try exact(expected, y);
            }
        }
        for ([_]usize{ 1, 4 }, 0..) |bank_count, rotation| {
            const input = banks[0..bank_count];
            for (groups) |group| _ = try timed(s, input, group, 0, 12);
            for (0..11) |round| for (0..groups.len) |step| {
                const ai = if (round % 2 == 0) step else groups.len - 1 - step;
                samples[geometry][rotation][round][ai] = try timed(s, input, groups[ai], round, 8);
            };
        }
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .complete = true,
        .exact_against_native = true,
        .timing = "host apply+eval+free; alternating arm order; evaluated inputs; no model",
        .shapes = .{ .{ .m = 512, .n = 8192, .k = 4096 }, .{ .m = 512, .n = 4096, .k = 8192 } },
        .groups = groups,
        .group_zero = "native MLX",
        .bank_counts = .{ 1, 4 },
        .warmup = 12,
        .evaluations_per_sample = 8,
        .nanoseconds = samples,
    }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = json });
}
