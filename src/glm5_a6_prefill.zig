//! Isolated affine6/group128 loader experiment; the native NAX tile/body stays unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
const HEADER = @embedFile("kernels/glm5_qmm_prefill_header.metal");
const SOURCE =
    \\uint3 tile = threadgroup_position_in_grid;
    \\
    \\threadgroup bfloat16_t Ws[64 * 72];
    \\qmm_t_nax_tgp_impl<bfloat16_t,128,6,true,64,64,64,2,2>(
    \\  w, scales, biases, x, y, Ws, K, N, M, tile,
    \\  thread_index_in_threadgroup, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
;
var kernels: [3]?mlx.mlx_fast_metal_kernel = .{ null, null, null };

const WORDS =
    \\
    \\template <typename T>
    \\inline void glm_a6_words32(const device uint8_t* src, T scale, T bias, threadgroup T* dst) {
    \\  const float s=float(scale), b=float(bias);
    \\  const device uint* words=reinterpret_cast<const device uint*>(src);
    \\  const uint w0=words[0];
    \\  const uint w1=words[1];
    \\  const uint w2=words[2];
    \\  const uint w3=words[3];
    \\  const uint w4=words[4];
    \\  const uint w5=words[5];
    \\  dst[0]=static_cast<T>(((w0 >> 0u) & 63u)*s+b);
    \\  dst[1]=static_cast<T>(((w0 >> 6u) & 63u)*s+b);
    \\  dst[2]=static_cast<T>(((w0 >> 12u) & 63u)*s+b);
    \\  dst[3]=static_cast<T>(((w0 >> 18u) & 63u)*s+b);
    \\  dst[4]=static_cast<T>(((w0 >> 24u) & 63u)*s+b);
    \\  dst[5]=static_cast<T>((((w0 >> 30u) | (w1 << 2u)) & 63u)*s+b);
    \\  dst[6]=static_cast<T>(((w1 >> 4u) & 63u)*s+b);
    \\  dst[7]=static_cast<T>(((w1 >> 10u) & 63u)*s+b);
    \\  dst[8]=static_cast<T>(((w1 >> 16u) & 63u)*s+b);
    \\  dst[9]=static_cast<T>(((w1 >> 22u) & 63u)*s+b);
    \\  dst[10]=static_cast<T>((((w1 >> 28u) | (w2 << 4u)) & 63u)*s+b);
    \\  dst[11]=static_cast<T>(((w2 >> 2u) & 63u)*s+b);
    \\  dst[12]=static_cast<T>(((w2 >> 8u) & 63u)*s+b);
    \\  dst[13]=static_cast<T>(((w2 >> 14u) & 63u)*s+b);
    \\  dst[14]=static_cast<T>(((w2 >> 20u) & 63u)*s+b);
    \\  dst[15]=static_cast<T>(((w2 >> 26u) & 63u)*s+b);
    \\  dst[16]=static_cast<T>(((w3 >> 0u) & 63u)*s+b);
    \\  dst[17]=static_cast<T>(((w3 >> 6u) & 63u)*s+b);
    \\  dst[18]=static_cast<T>(((w3 >> 12u) & 63u)*s+b);
    \\  dst[19]=static_cast<T>(((w3 >> 18u) & 63u)*s+b);
    \\  dst[20]=static_cast<T>(((w3 >> 24u) & 63u)*s+b);
    \\  dst[21]=static_cast<T>((((w3 >> 30u) | (w4 << 2u)) & 63u)*s+b);
    \\  dst[22]=static_cast<T>(((w4 >> 4u) & 63u)*s+b);
    \\  dst[23]=static_cast<T>(((w4 >> 10u) & 63u)*s+b);
    \\  dst[24]=static_cast<T>(((w4 >> 16u) & 63u)*s+b);
    \\  dst[25]=static_cast<T>(((w4 >> 22u) & 63u)*s+b);
    \\  dst[26]=static_cast<T>((((w4 >> 28u) | (w5 << 4u)) & 63u)*s+b);
    \\  dst[27]=static_cast<T>(((w5 >> 2u) & 63u)*s+b);
    \\  dst[28]=static_cast<T>(((w5 >> 8u) & 63u)*s+b);
    \\  dst[29]=static_cast<T>(((w5 >> 14u) & 63u)*s+b);
    \\  dst[30]=static_cast<T>(((w5 >> 20u) & 63u)*s+b);
    \\  dst[31]=static_cast<T>(((w5 >> 26u) & 63u)*s+b);
    \\}
;

const LOAD =
    \\    for (int i = 0; i < n_reads; i++) {
    \\      dequantize<T, pack_factor, bits>(
    \\          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    \\    }
;
const UNROLLED =
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < n_reads; i++) {
    \\      dequantize<T, pack_factor, bits>(
    \\          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    \\    }
;
const WORD_LOAD =
    \\    static_assert(bits == 6 && n_reads == 8 && pack_factor == 4,
    \\                  "word loader requires native 32-coefficient A6 span");
    \\    glm_a6_words32(src, scale, bias, dst);
;
fn header(allocator: std.mem.Allocator, arm: u32) ![:0]u8 {
    if (arm == 0) return allocator.dupeSentinel(u8, HEADER, 0);
    if (std.mem.count(u8, HEADER, LOAD) != 1) return error.A6VendorSourceMismatch;
    const body = try std.mem.replaceOwned(u8, allocator, HEADER, LOAD, if (arm == 1) UNROLLED else WORD_LOAD);
    defer allocator.free(body);
    if (arm == 1) return allocator.dupeSentinel(u8, body, 0);
    const with_helper = try std.mem.replaceOwned(u8, allocator, body, "template <\n    typename T,\n    short BROWS,", WORDS ++ "\ntemplate <\n    typename T,\n    short BROWS,");
    defer allocator.free(with_helper);
    return allocator.dupeSentinel(u8, with_helper, 0);
}
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

pub fn apply(s: mlx.mlx_stream, input: Input, arm: u32) !?Arr {
    if (!mlx.streamIsGpu(s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
    if (arm > 2) return error.InvalidA6UnpackArm;
    for ([_]Arr{ input.x, input.w, input.scales, input.biases }) |a| if (!try ready(a)) return null;
    const xs = mlx.getShape(input.x);
    const ws = mlx.getShape(input.w);
    if (xs.len != 3 or xs[0] != 1 or (xs[1] != 512 and xs[1] != 2048) or ws.len != 2) return null;
    const k = xs[2];
    const n = ws[0];
    const m = xs[1];
    if (!((k == 4096 and n == 8192) or (k == 8192 and n == 4096)) or ws[1] != @divExact(k * 3, 16)) return null;
    const ss = [_]c_int{ n, @divExact(k, 128) };
    if (!std.mem.eql(c_int, &ss, mlx.getShape(input.scales)) or !std.mem.eql(c_int, &ss, mlx.getShape(input.biases)) or
        mlx.mlx_array_dtype(input.x) != .bfloat16 or mlx.mlx_array_dtype(input.w) != .uint32 or
        mlx.mlx_array_dtype(input.scales) != .bfloat16 or mlx.mlx_array_dtype(input.biases) != .bfloat16) return null;
    if (kernels[arm] == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "w", "scales", "biases", "x", "K", "N", "M" }, 7);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"y"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const h = try header(std.heap.page_allocator, arm);
        defer std.heap.page_allocator.free(h);
        const names = [_][*:0]const u8{ "sushi_glm_a6_native", "sushi_glm_a6_unrolled", "sushi_glm_a6_words" };
        const value = mlx.mlx_fast_metal_kernel_new(names[arm], ins, outs, SOURCE, h.ptr, false, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        kernels[arm] = value;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, m, n }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(n, 64) * 32, @divExact(m, 64) * 2, 2));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 2));
    const ka = mlx.mlx_array_new_int(k);
    defer _ = mlx.mlx_array_free(ka);
    const na = mlx.mlx_array_new_int(n);
    defer _ = mlx.mlx_array_free(na);
    const ma = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(ma);
    const inputs = mlx.mlx_vector_array_new_data(&.{ input.w, input.scales, input.biases, input.x, ka, na, ma }, 7);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernels[arm].?, inputs, cfg, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    return y;
}

fn fixture(ops: *Ops, m: c_int, k: c_int, n: c_int, seed: u64) !Input {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ 1, m, k }, 3, .bfloat16, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ n, @divExact(k * 3, 16) }, 2, 4, key.*, ops.s));
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
    try mlx.check(mlx.mlx_quantized_matmul(&result, in.x, in.w, in.scales, in.biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(6), "affine", s));
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

test "GLM A6 unpack preserves native output at actual QKV and output widths" {
    const s = mlx.gpuStream();
    for ([_]c_int{ 512, 2048 }) |m| for ([_]c_int{ 4096, 8192 }) |k| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const input = try fixture(&ops, m, k, if (k == 4096) 8192 else 4096, 751);
        const expected = try native(s, input);
        defer _ = mlx.mlx_array_free(expected);
        for ([_]u32{ 0, 1, 2 }) |arm| {
            const got = (try apply(s, input, arm)) orelse return error.TestExpectedA6;
            defer _ = mlx.mlx_array_free(got);
            try exact(expected, got);
        }
    };
}

test "GLM A6 unpack declines unsupported shapes and layouts" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const input = try fixture(&ops, 512, 4096, 8192, 757);
    try std.testing.expectError(error.InvalidA6UnpackArm, apply(s, input, 3));
    var changed = input;
    changed.x = try ops.slice(input.x, 1, 0, 128);
    try mlx.check(mlx.mlx_array_eval(changed.x));
    try std.testing.expect((try apply(s, changed, 2)) == null);
    changed = input;
    changed.scales = try ops.cast(input.scales, .float32);
    try mlx.check(mlx.mlx_array_eval(changed.scales));
    try std.testing.expect((try apply(s, changed, 2)) == null);
    changed = input;
    changed.w = try ops.transpose(input.w, &.{ 1, 0 });
    try mlx.check(mlx.mlx_array_eval(changed.w));
    try std.testing.expect((try apply(s, changed, 2)) == null);
}

fn timed(s: mlx.mlx_stream, banks: []const Input, arm: u32, start: usize, repetitions: usize) !u64 {
    const timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    for (0..repetitions) |i| {
        const input = banks[(start + i) % banks.len];
        const y = if (arm == 3) try native(s, input) else (try apply(s, input, arm)) orelse return error.TestExpectedA6;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    return timer.read() / repetitions;
}

test "GLM A6 unpack isolated timing" {
    const path = std.c.getenv("SUSHI_GLM_A6_UNPACK_BENCH_OUT") orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    var samples: [2][2][2][11][4]u64 = undefined;
    for ([_]c_int{ 512, 2048 }, 0..) |m, mi| for ([_]c_int{ 4096, 8192 }, 0..) |k, ki| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var banks: [4]Input = undefined;
        for (&banks, 0..) |*bank, bi| bank.* = try fixture(&ops, m, k, if (k == 4096) 8192 else 4096, @intCast(761 + bi));
        for (banks) |input| {
            const expected = try native(s, input);
            defer _ = mlx.mlx_array_free(expected);
            for ([_]u32{ 0, 1, 2 }) |arm| {
                const got = (try apply(s, input, arm)) orelse return error.TestExpectedA6;
                defer _ = mlx.mlx_array_free(got);
                try exact(expected, got);
            }
        }
        for ([_]usize{ 1, 4 }, 0..) |count, rotation| {
            for (0..4) |arm| _ = try timed(s, banks[0..count], @intCast(arm), 0, 12);
            for (0..11) |round| for (0..4) |position| {
                const arm = if (round % 2 == 0) position else 3 - position;
                samples[mi][ki][rotation][round][arm] = try timed(s, banks[0..count], @intCast(arm), round, 8);
            };
        }
    };
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .nanoseconds = samples,
        .rows = .{ 512, 2048 },
        .input_widths = .{ 4096, 8192 },
        .arms = .{ "same-body bytes", "same-body unrolled bytes", "same-body words", "native MLX" },
        .bank_counts = .{ 1, 4 },
        .warmups = 12,
        .rounds = 11,
        .repetitions = 8,
        .method = "single-process alternating forward/reverse; inputs materialized; fresh apply/eval/free",
        .exact_against_native = true,
        .full_model = false,
    }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = json });
}
