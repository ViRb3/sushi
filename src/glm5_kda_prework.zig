//! Parallel GLM KDA prework; recurrence and output normalization remain separate.
const std = @import("std");
const mlx = @import("mlx.zig");
const unary = @import("glm5_kda_fused.zig");
const Arr = mlx.mlx_array;
pub const Inputs = struct {
    qkv: Arr,
    a: Arr,
    beta: Arr,
    conv_weight: Arr,
    exp_a: Arr,
    dt_bias: Arr,
    conv_state: ?Arr = null,
    heads: c_int,
    lower: f32 = -5,
};
pub const Result = struct {
    q: Arr,
    k: Arr,
    v: Arr,
    decay: Arr,
    beta: Arr,
    conv: Arr,
    pub fn arrays(self: Result) [6]Arr {
        return .{ self.q, self.k, self.v, self.decay, self.beta, self.conv };
    }
    pub fn deinit(self: Result) void {
        for (self.arrays()) |a| {
            _ = mlx.mlx_array_free(a);
        }
    }
};
var count: usize = 0;
var force_reference_for_tests = false;
pub fn forceReferenceForTest(on: bool) void {
    if (@import("builtin").is_test) force_reference_for_tests = on;
}

pub fn dispatchCount() usize {
    return count;
}
pub fn resetDispatchCount() void {
    count = 0;
}
// Arithmetic follows the qualified one-token body and oMLX's prework mapping;
// retained BF16 convolution/SiLU stores and FP32 normalization order are explicit.
const HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\using namespace metal;
    \\template<typename T,bool P> inline T sigmoid(T x) {
    \\ auto y=1/(1+(P?metal::precise::exp(metal::abs(x)):metal::exp(metal::abs(x))));
    \\ return x<0?y:1-y;
    \\}
;
const SOURCE: [:0]const u8 =
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    \\const uint head=threadgroup_position_in_grid.x,row=threadgroup_position_in_grid.y;
    \\const uint width=uint(H)*128u,channels=3u*width;
    \\const uint rows=uint(length);
    \\threadgroup T qs[128],ks[128];
    \\const uint base=(row*uint(H)+head)*128u;
    \\for(uint part=0;part<3u;++part) {
    \\ const uint channel=part*width+head*128u+tid;
    \\ float acc=0.0f;
    \\ for(uint tap=0;tap<4u;++tap) {
    \\   const uint pos=row+tap;
    \\   T x=pos<3u?(HAS_CONV?previous[pos*channels+channel]:T(0)):qkv[(pos-3u)*channels+channel];
    \\   acc+=float(x)*float(conv_w[channel*4u+tap]);
    \\ }
    \\ const T co=T(acc),sig=sigmoid<T,SIG_B>(co);
    \\ const T act=co*sig;
    \\ if(part==0u) qs[tid]=act; else if(part==1u) ks[tid]=act; else v_out[base+tid]=act;
    \\ if(row==0u) {
    \\   for(uint j=0;j<3u;++j) {
    \\     const uint pos=rows+j;
    \\     const ushort raw=pos<3u?(HAS_CONV?((const device ushort*)previous)[pos*channels+channel]:ushort(0)):((const device ushort*)qkv)[(pos-3u)*channels+channel];
    \\     // Single-token views preserve subnormal bits without a float conversion.
    \\     if(rows==1u) ((device ushort*)conv_out)[j*channels+channel]=raw;
    \\     else conv_out[j*channels+channel]=T(float(as_type<T>(raw))+constants[3]);
    \\   }
    \\ }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(sg<2u) {
    \\ threadgroup T* values=sg==0u?qs:ks;
    \\ float x[4],total=0.0f;
    \\ for(uint i=0;i<4u;++i) {x[i]=float(values[lane*4u+i]);float sq=x[i]*x[i];total=sq+total;}
    \\ total=simd_sum(total);
    \\ const float arg=total+constants[1];
    \\ const float inv=RSQ_F?metal::precise::rsqrt(arg):metal::rsqrt(arg);
    \\ for(uint i=0;i<4u;++i) {
    \\   const float normalized=x[i]*inv;
    \\   if(sg==0u) q_out[base+lane*4u+i]=T(normalized*constants[0]);
    \\   else k_out[base+lane*4u+i]=T(normalized);
    \\ }
    \\}
    \\const float shift=float(a[base+tid])+dt_bias[head*128u+tid];
    \\const float scaled=exp_a[head]*shift;
    \\const float forget=sigmoid<float,SIG_F>(scaled);
    \\const float gate=constants[2]*forget;
    \\decay_out[base+tid]=EXP_F?metal::precise::exp(gate):metal::exp(gate);
    \\if(tid==0u) beta_out[row*uint(H)+head]=sigmoid<T,SIG_B>(beta[row*uint(H)+head]);
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;
fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(&.{ "qkv", "a", "beta", "conv_w", "exp_a", "dt_bias", "previous", "length", "constants" }, 9);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&.{ "q_out", "k_out", "v_out", "decay_out", "beta_out", "conv_out" }, 6);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_kda_prework", iv, ov, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}
fn geometryFits(rows: c_int, heads: c_int) bool {
    if (rows < 1 or heads < 1) return false;
    const channels = std.math.mul(u64, @intCast(heads), 384) catch return false;
    const input_count = std.math.mul(u64, @intCast(rows), channels) catch return false;
    const weights = std.math.mul(u64, channels, 4) catch return false;
    const tail = std.math.mul(u64, channels, 3) catch return false;
    const grid = std.math.mul(u64, @intCast(heads), 128) catch return false;
    // Shader offsets are uint32; a span of 2^32 elements has last index UINT_MAX.
    return channels <= std.math.maxInt(c_int) and input_count <= std.math.maxInt(c_int) and
        weights - 1 <= std.math.maxInt(u32) and tail - 1 <= std.math.maxInt(u32) and grid <= std.math.maxInt(c_int);
}

pub fn apply(s: mlx.mlx_stream, in: Inputs) !?Result {
    if (@import("builtin").is_test and force_reference_for_tests) return null;
    if (!mlx.streamIsGpu(s) or in.heads < 1 or in.heads > @divTrunc(std.math.maxInt(c_int), 384) or in.lower != -5) return null;
    for ([_]Arr{ in.qkv, in.a, in.beta, in.conv_weight, in.exp_a, in.dt_bias }) |a| if (a.ctx == null) return null;
    const shape = mlx.getShape(in.qkv);
    const width = in.heads * 128;
    if (shape.len != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] != 3 * width or
        @as(i64, shape[1]) * shape[2] > std.math.maxInt(c_int) or mlx.mlx_array_dtype(in.qkv) != .bfloat16) return null;
    const rows = shape[1];
    if (!geometryFits(rows, in.heads)) return null;
    for ([_]Arr{ in.a, in.beta, in.conv_weight }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
    if (!std.mem.eql(c_int, &.{ 1, rows, width }, mlx.getShape(in.a)) or !std.mem.eql(c_int, &.{ 1, rows, in.heads }, mlx.getShape(in.beta)) or
        !std.mem.eql(c_int, &.{ 3 * width, 4, 1 }, mlx.getShape(in.conv_weight)) or
        !std.mem.eql(c_int, &.{in.heads}, mlx.getShape(in.exp_a)) or mlx.mlx_array_dtype(in.exp_a) != .float32 or
        !std.mem.eql(c_int, &.{width}, mlx.getShape(in.dt_bias)) or mlx.mlx_array_dtype(in.dt_bias) != .float32) return null;
    if (in.conv_state) |a| if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16 or !std.mem.eql(c_int, &.{ 1, 3, 3 * width }, mlx.getShape(a))) return null;
    const mode = (try unary.unaryModes(s)) orelse return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for ([_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .bfloat16, .float32 }) |dtype| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, in.heads, 128 }, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, in.heads }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, 3 * width }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * in.heads, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", in.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HAS_CONV", @intFromBool(in.conv_state != null)));
    inline for (.{ "SIG_B", "SIG_F", "EXP_F", "RSQ_F" }, .{ mode.sig_b, mode.sig_f, mode.exp_f, mode.rsq_f }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, @intFromBool(value)));
    const length = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(length);
    const constants = mlx.mlx_array_new_data(&[_]f32{ 1 / @sqrt(@as(f32, 128)), 1e-6, in.lower, 0 }, &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(constants);
    const iv = mlx.mlx_vector_array_new_data(&.{ in.qkv, in.a, in.beta, in.conv_weight, in.exp_a, in.dt_bias, in.conv_state orelse in.qkv, length, constants }, 9);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getKernel(), iv, cfg, s));
    var arrays: [6]Arr = undefined;
    var made: usize = 0;
    errdefer for (arrays[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&arrays, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    }
    count += 1;
    return .{ .q = arrays[0], .k = arrays[1], .v = arrays[2], .decay = arrays[3], .beta = arrays[4], .conv = arrays[5] };
}

fn reference(ops: *@import("glm5_model.zig").Ops, in: Inputs) ![6]Arr {
    const rows = mlx.getShape(in.qkv)[1];
    const width = in.heads * 128;
    const previous = in.conv_state orelse try ops.zeros(&.{ 1, 3, 3 * width }, .bfloat16);
    const joined = try ops.concat(&.{ previous, in.qkv }, 1);
    const conv = try ops.silu(try ops.conv(joined, in.conv_weight, 3 * width));
    const shape = [_]c_int{ 1, rows, in.heads, 128 };
    const q = try ops.cast(try ops.reshape(try ops.slice(conv, 2, 0, width), &shape), .float32);
    const k = try ops.cast(try ops.reshape(try ops.slice(conv, 2, width, 2 * width), &shape), .float32);
    const v = try ops.reshape(try ops.slice(conv, 2, 2 * width, 3 * width), &shape);
    const eps = try ops.scalar(1e-6, .float32);
    const qnorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, q, q), -1, false, true), eps));
    const knorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, k, k), -1, false, true), eps));
    const qr = try ops.cast(try ops.binary(.mul, try ops.binary(.mul, q, qnorm), try ops.scalar(1 / @sqrt(@as(f32, 128)), .float32)), .bfloat16);
    const kr = try ops.cast(try ops.binary(.mul, k, knorm), .bfloat16);
    const a = try ops.reshape(try ops.cast(in.a, .float32), &shape);
    const shift = try ops.reshape(in.dt_bias, &.{ 1, 1, in.heads, 128 });
    const magnitude = try ops.reshape(in.exp_a, &.{ 1, 1, in.heads, 1 });
    const forget = try ops.unary(.sigmoid, try ops.binary(.mul, magnitude, try ops.binary(.add, a, shift)));
    const decay = try ops.unary(.exp, try ops.binary(.mul, forget, try ops.scalar(in.lower, .float32)));
    const beta = try ops.unary(.sigmoid, in.beta);
    const tail = try ops.slice(joined, 1, rows, rows + 3);
    const compact = if (rows > 1) try ops.own(try @import("transformer.zig").materializedOwnedCopy(ops.s, tail)) else try ops.contiguous(tail);
    return .{ qr, kr, v, decay, beta, compact };
}
fn randomArray(ops: *@import("glm5_model.zig").Ops, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64, scale: f32) !Arr {
    var size: usize = 1;
    for (shape) |d| size *= @intCast(d);
    var random = std.Random.DefaultPrng.init(seed);
    const rng = random.random();
    const allocator = std.testing.allocator;
    if (dtype == .bfloat16) {
        const data = try allocator.alloc(u16, size);
        defer allocator.free(data);
        for (data, 0..) |*v, i| {
            const f = (rng.float(f32) - 0.5) * scale;
            v.* = if (i % 257 == 0) 1 else if (i % 17 == 0) 0x8000 else @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        return ops.own(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dtype));
    }
    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);
    for (data) |*v| v.* = (rng.float(f32) - 0.5) * scale;
    return ops.own(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dtype));
}
fn expectBits(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ca = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ca);
    var cb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cb);
    try mlx.check(mlx.mlx_contiguous(&ca, a, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_contiguous(&cb, b, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(ca));
    try mlx.check(mlx.mlx_array_eval(cb));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(ca).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(cb).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(ca).?[0..n], mlx.mlx_array_data_bfloat16(cb).?[0..n]);
}

test "GLM KDA prework exact raw outputs and compact tails across token boundaries" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!unary.hardwareSupported()) return error.SkipZigTest;
    for ([_]c_int{ 1, 3, 64 }) |heads| {
        for ([_]c_int{ 1, 2, 3, 4, 17, 128, 512 }) |rows| {
            for ([_]bool{ false, true }) |hot| {
                var ops = Ops{ .s = s };
                defer ops.deinit();
                const width = heads * 128;
                const exp_a = try ops.unary(.exp, try randomArray(&ops, &.{heads}, .float32, 55, 1));
                const in = Inputs{
                    .qkv = try randomArray(&ops, &.{ 1, rows, 3 * width }, .bfloat16, 3, 3),
                    .a = try randomArray(&ops, &.{ 1, rows, width }, .bfloat16, 15, 12),
                    .beta = try randomArray(&ops, &.{ 1, rows, heads }, .bfloat16, 33, 8),
                    .conv_weight = try randomArray(&ops, &.{ 3 * width, 4, 1 }, .bfloat16, 45, 0.5),
                    .exp_a = exp_a,
                    .dt_bias = try randomArray(&ops, &.{width}, .float32, 67, 2),
                    .conv_state = if (hot) try randomArray(&ops, &.{ 1, 3, 3 * width }, .bfloat16, 89, 2) else null,
                    .heads = heads,
                };
                const want = try reference(&ops, in);
                const got = (try apply(s, in)) orelse return error.TestExpectedPrework;
                defer got.deinit();
                for (want, got.arrays(), 0..) |expected, actual, field| expectBits(expected, actual) catch |err| {
                    std.debug.print("prework mismatch H={d} rows={d} hot={} field={d}\n", .{ heads, rows, hot, field });
                    return err;
                };
            }
        }
    }
}

test "GLM KDA prework refuses unsupported inputs before dispatch" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const count_before = dispatchCount();
    const valid = Inputs{
        .qkv = try ops.zeros(&.{ 1, 2, 384 }, .bfloat16),
        .a = try ops.zeros(&.{ 1, 2, 128 }, .bfloat16),
        .beta = try ops.zeros(&.{ 1, 2, 1 }, .bfloat16),
        .conv_weight = try ops.zeros(&.{ 384, 4, 1 }, .bfloat16),
        .exp_a = try ops.ones(&.{1}, .float32),
        .dt_bias = try ops.zeros(&.{128}, .float32),
        .heads = 1,
    };
    var bad = valid;
    bad.lower = -6;
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.heads = 0;
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.qkv = .{ .ctx = null };
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.a = try ops.cast(valid.a, .float32);
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.conv_state = try ops.zeros(&.{ 1, 2, 384 }, .bfloat16);
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.exp_a = try ops.cast(valid.exp_a, .bfloat16);
    try std.testing.expect((try apply(ops.s, bad)) == null);
    try std.testing.expectEqual(count_before, dispatchCount());
}

fn timedPrework(in: Inputs, fused: bool) !u64 {
    var ops = @import("glm5_model.zig").Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const actual = if (fused) (try apply(ops.s, in)) orelse return error.TestExpectedPrework else null;
    defer if (actual) |r| r.deinit();
    const arrays = if (actual) |r| r.arrays() else try reference(&ops, in);
    const outputs = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_eval(outputs));
    return timer.read();
}
test "GLM KDA prework production microbenchmark" {
    const path = std.c.getenv("SUSHI_GLM_PREWORK_BENCH_OUT") orelse return error.SkipZigTest;
    var ops = @import("glm5_model.zig").Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const heads = 64;
    const rows = 512;
    const width = heads * 128;
    const in = Inputs{
        .qkv = try randomArray(&ops, &.{ 1, rows, 3 * width }, .bfloat16, 3, 3),
        .a = try randomArray(&ops, &.{ 1, rows, width }, .bfloat16, 15, 12),
        .beta = try randomArray(&ops, &.{ 1, rows, heads }, .bfloat16, 33, 8),
        .conv_weight = try randomArray(&ops, &.{ 3 * width, 4, 1 }, .bfloat16, 45, 0.5),
        .exp_a = try ops.unary(.exp, try randomArray(&ops, &.{heads}, .float32, 55, 1)),
        .dt_bias = try randomArray(&ops, &.{width}, .float32, 67, 2),
        .conv_state = try randomArray(&ops, &.{ 1, 3, 3 * width }, .bfloat16, 89, 2),
        .heads = heads,
    };
    for (0..3) |_| {
        _ = try timedPrework(in, false);
        _ = try timedPrework(in, true);
    }
    var reference_ns: [24]u64 = undefined;
    var fused_ns: [24]u64 = undefined;
    for (0..24) |i| {
        if (i % 2 == 0) {
            reference_ns[i] = try timedPrework(in, false);
            fused_ns[i] = try timedPrework(in, true);
        } else {
            fused_ns[i] = try timedPrework(in, true);
            reference_ns[i] = try timedPrework(in, false);
        }
    }
    const text = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .reference_ns = reference_ns,
        .fused_ns = fused_ns,
        .warm_pairs = 3,
        .timed_pairs = 24,
        .rows = rows,
        .heads = heads,
        .head_dim = 128,
        .all_six_outputs_evaluated = true,
        .method = "single-process alternating AB/BA; graph build plus evaluation; hot nonzero conv tail",
        .full_model = false,
    }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(text);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = text });
}

test "GLM KDA prework geometry guards every uint32 shader offset" {
    try std.testing.expect(geometryFits(512, 64));
    const largest_heads: c_int = @intCast((@as(u64, std.math.maxInt(u32)) + 1) / (384 * 4));
    try std.testing.expect(geometryFits(1, largest_heads));
    try std.testing.expect(geometryFits(2, largest_heads));
    try std.testing.expect(!geometryFits(1, largest_heads + 1));
    try std.testing.expect(!geometryFits(1, 3_000_000));
    const longest: c_int = @divTrunc(std.math.maxInt(c_int), 64 * 384);
    try std.testing.expect(geometryFits(longest, 64));
    try std.testing.expect(!geometryFits(longest + 1, 64));
    try std.testing.expect(!geometryFits(0, 64));
    try std.testing.expect(!geometryFits(1, 0));
    try std.testing.expect(!geometryFits(std.math.maxInt(c_int), std.math.maxInt(c_int)));
}
