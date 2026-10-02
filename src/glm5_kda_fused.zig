//! GLM one-token KDA body; BF16 activations and FP32 vector decay/state.
//! Adapted from oMLX 6745c39c (Apache-2.0); sigmoid expressions derive from MLX (MIT).
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\using namespace metal;
    \\template<typename T, bool precise> inline T glm_sigmoid(T x) {
    \\  auto y = 1 / (1 + (precise ? metal::precise::exp(metal::abs(x)) : metal::exp(metal::abs(x))));
    \\  return x < 0 ? y : 1 - y;
    \\}
    \\template<bool precise> inline float glm_exp(float x) { return precise ? metal::precise::exp(x) : metal::exp(x); }
    \\template<bool precise> inline float glm_rsqrt(float x) { return precise ? metal::precise::rsqrt(x) : metal::rsqrt(x); }
;
const SOURCE: [:0]const u8 =
    \\
    \\  constexpr int CK = 4;
    \\  constexpr int NROW = CK - 1 + TOK;
    \\  constexpr int NP = 3 * QKV;
    \\  const uint tid = thread_position_in_threadgroup.x;
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const uint sg = simdgroup_index_in_threadgroup;
    \\  const int h = int(threadgroup_position_in_grid.x);
    \\  const float q_scale = consts[0];
    \\  const float l2_eps = consts[1];
    \\  const float norm_eps = consts[2];
    \\  const float lower = consts[3];
    \\  const float inv_n = consts[4];
    \\
    \\  threadgroup T qs[TOK][DK];
    \\  threadgroup T ks[TOK][DK];
    \\  threadgroup T vs[TOK][DK];
    \\  threadgroup T as_[TOK][DK];
    \\  threadgroup T gates[TOK][DK];
    \\  threadgroup T ys[TOK][DK];
    \\  threadgroup float gs[TOK][DK];
    \\  threadgroup T betas[TOK];
    \\
    \\  if (tid < uint(3 * DK)) {
    \\    const int part = int(tid) / DK;
    \\    const int i = int(tid) % DK;
    \\    const int gc = part * QKV + h * DK + i;
    \\    T win[NROW];
    \\    for (int r = 0; r < CK - 1; r++) {
    \\      win[r] = conv_state[r * NP + gc];
    \\    }
    \\    for (int t = 0; t < TOK; t++) {
    \\      win[CK - 1 + t] = proj[t * PROJ_W + gc];
    \\    }
    \\    const device T* w = conv_w + gc * CK;
    \\    for (int t = 0; t < TOK; t++) {
    \\      float acc = 0.0;
    \\      for (int j = 0; j < CK; ++j) {
    \\        acc += static_cast<float>(win[t + j]) * w[j];
    \\      }
    \\      T co = static_cast<T>(acc);
    \\      T sgm = glm_sigmoid<T, SIG_B>(co);
    \\      T sv = co * sgm;
    \\      if (part == 0) {
    \\        qs[t][i] = sv;
    \\      } else if (part == 1) {
    \\        ks[t][i] = sv;
    \\      } else {
    \\        vs[t][i] = sv;
    \\      }
    \\    }
    \\    for (int r = 0; r < CK - 1; r++) {
    \\      conv_state_out[r * NP + gc] = win[TOK + r];
    \\    }
    \\  }
    \\
    \\  for (int e = int(tid); e < TOK * DK; e += 1024) {
    \\    const int t = e / DK;
    \\    const int i = e % DK;
    \\    as_[t][i] = a_pre[t * QKV + h * DK + i];
    \\    gates[t][i] = gate_pre[t * QKV + h * DK + i];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  if (sg < uint(2 * TOK)) {
    \\    const int t = int(sg) / 2;
    \\    const bool is_q = (sg % 2) == 0;
    \\    threadgroup T* row = is_q ? qs[t] : ks[t];
    \\    float x[4];
    \\    float tot = 0.0f;
    \\    for (int e = 0; e < 4; e++) {
    \\      x[e] = static_cast<float>(row[4 * lane + e]);
    \\      float sq = x[e] * x[e];
    \\      tot = sq + tot;
    \\    }
    \\    tot = simd_sum(tot);
    \\    float u = tot + l2_eps;
    \\    float r = glm_rsqrt<RSQ_F>(u);
    \\    for (int e = 0; e < 4; e++) {
    \\      float xn = x[e] * r;
    \\      if (is_q) {
    \\        float xs = xn * q_scale;
    \\        row[4 * lane + e] = static_cast<T>(xs);
    \\      } else {
    \\        row[4 * lane + e] = static_cast<T>(xn);
    \\      }
    \\    }
    \\  }
    \\
    \\  if (tid < uint(TOK * DK)) {
    \\    const int t = int(tid) / DK;
    \\    const int i = int(tid) % DK;
    \\    float ea = exp_a[h];
    \\    float af = static_cast<float>(as_[t][i]);
    \\    float s1 = af + dt_bias[h * DK + i];
    \\    float s2 = ea * s1;
    \\    float s3 = glm_sigmoid<float, SIG_F>(s2);
    \\    float s4 = lower * s3;
    \\    gs[t][i] = glm_exp<EXP_F>(s4);
    \\  }
    \\  if (tid < uint(TOK)) {
    \\    betas[tid] = glm_sigmoid<T, SIG_B>(proj[tid * PROJ_W + OFF_B + h]);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  for (int j = 0; j < DK / 32; j++) {
    \\    const int dv_idx = int(sg) + 32 * j;
    \\    constexpr int n_per_t = DK / 32;
    \\    const int dk_idx = int(lane);
    \\    float state[n_per_t];
    \\    for (int i = 0; i < n_per_t; ++i) {
    \\      auto s_idx = n_per_t * dk_idx + i;
    \\      state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
    \\    }
    \\    for (int t = 0; t < TOK; ++t) {
    \\      float kv_mem = 0.0f;
    \\      for (int i = 0; i < n_per_t; ++i) {
    \\        auto s_idx = n_per_t * dk_idx + i;
    \\        state[i] = state[i] * gs[t][s_idx];
    \\        kv_mem += state[i] * ks[t][s_idx];
    \\      }
    \\      kv_mem = simd_sum(kv_mem);
    \\
    \\      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];
    \\
    \\      float out = 0.0f;
    \\      for (int i = 0; i < n_per_t; ++i) {
    \\        auto s_idx = n_per_t * dk_idx + i;
    \\        state[i] = state[i] + ks[t][s_idx] * delta;
    \\        out += state[i] * qs[t][s_idx];
    \\      }
    \\      out = simd_sum(out);
    \\      if (thread_index_in_simdgroup == 0) {
    \\        ys[t][dv_idx] = static_cast<T>(out);
    \\      }
    \\    }
    \\    for (int i = 0; i < n_per_t; ++i) {
    \\      auto s_idx = n_per_t * dk_idx + i;
    \\      state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  if (sg < uint(TOK)) {
    \\    const int t = int(sg);
    \\    float x[4];
    \\    float tot = 0.0f;
    \\    for (int e = 0; e < 4; e++) {
    \\      x[e] = static_cast<float>(ys[t][4 * lane + e]);
    \\      float sq = x[e] * x[e];
    \\      tot = sq + tot;
    \\    }
    \\    tot = simd_sum(tot);
    \\    float var = tot * inv_n;
    \\    float u = var + norm_eps;
    \\    float r = glm_rsqrt<RSQ_F>(u);
    \\    for (int e = 0; e < 4; e++) {
    \\      const int c = 4 * lane + e;
    \\      float xn = x[e] * r;
    \\      float wf = static_cast<float>(norm_w[c]);
    \\      float wx = wf * xn;
    \\      float gf = static_cast<float>(gates[t][c]);
    \\      float gsg = glm_sigmoid<float, SIG_F>(gf);
    \\      float o = wx * gsg;
    \\      y[t * QKV + h * DK + c] = static_cast<T>(o);
    \\    }
    \\  }
;

const Modes = struct { sig_b: bool, sig_f: bool, exp_f: bool, rsq_f: bool };
var modes_checked = false;
var modes_cached: ?Modes = null;
var kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var calls: usize = 0;
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}
fn applyKernel(k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) !mlx.mlx_vector_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    return ov;
}

const PROBE: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(SIZE))return;
    \\if(OP==0) { fast[i]=glm_sigmoid<T,false>(x[i]); precise[i]=glm_sigmoid<T,true>(x[i]); }
    \\else if(OP==1) { fast[i]=T(glm_exp<false>(float(x[i]))); precise[i]=T(glm_exp<true>(float(x[i]))); }
    \\else { fast[i]=T(glm_rsqrt<false>(float(x[i]))); precise[i]=T(glm_rsqrt<true>(float(x[i]))); }
;
fn sameBytes(a: Arr, b: Arr) bool {
    const n = mlx.mlx_array_size(a);
    return if (mlx.mlx_array_dtype(a) == .bfloat16)
        std.mem.eql(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n])
    else
        std.mem.eql(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n]));
}
fn probe(s: mlx.mlx_stream, dtype: mlx.mlx_dtype, operation: c_int) !?bool {
    const n = 65536;
    const a = std.heap.page_allocator;
    var input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(input);
    if (dtype == .bfloat16) {
        const values = try a.alloc(u16, n);
        defer a.free(values);
        for (values, 0..) |*v, i| {
            const bits: u16 = @intCast(i);
            v.* = if (bits & 0x7f80 == 0x7f80) 0 else bits;
        }
        const raw = mlx.mlx_array_new_data(values.ptr, &[_]c_int{n}, 1, dtype);
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_array_set(&input, raw));
    } else {
        const values = try a.alloc(f32, n);
        defer a.free(values);
        for (values, 0..) |*v, i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, n - 1);
            v.* = switch (operation) {
                0 => 48 * u - 24,
                1 => 5 * u - 5,
                else => @exp2(-20 + 40 * u),
            };
        }
        const raw = mlx.mlx_array_new_data(values.ptr, &[_]c_int{n}, 1, dtype);
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_array_set(&input, raw));
    }
    const k = try makeKernel("sushi_glm_kda_math_probe", &.{"x"}, &.{ "fast", "precise" }, PROBE);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{n}, 1, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, n, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SIZE", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "OP", operation));
    const outputs = try applyKernel(k, &.{input}, cfg, s);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var candidates = [_]Arr{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (candidates) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (&candidates, 0..) |*v, i| try mlx.check(mlx.mlx_vector_array_get(v, outputs, i));
    var expected = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expected);
    try mlx.check(switch (operation) {
        0 => mlx.mlx_sigmoid(&expected, input, s),
        1 => mlx.mlx_exp(&expected, input, s),
        else => mlx.mlx_rsqrt(&expected, input, s),
    });
    const evals = mlx.mlx_vector_array_new_data(&[_]Arr{ expected, candidates[0], candidates[1] }, 3);
    defer _ = mlx.mlx_vector_array_free(evals);
    try mlx.check(mlx.mlx_eval(evals));
    if (sameBytes(expected, candidates[1])) return true;
    if (sameBytes(expected, candidates[0])) return false;
    return null;
}
fn modes(s: mlx.mlx_stream) !?Modes {
    if (modes_checked) return modes_cached;
    const result: ?Modes = blk: {
        break :blk .{ .sig_b = (try probe(s, .bfloat16, 0)) orelse break :blk null, .sig_f = (try probe(s, .float32, 0)) orelse break :blk null, .exp_f = (try probe(s, .float32, 1)) orelse break :blk null, .rsq_f = (try probe(s, .float32, 2)) orelse break :blk null };
    };
    modes_cached = result;
    modes_checked = true;
    return result;
}

pub const Inputs = struct {
    qkv: Arr,
    beta: Arr,
    a: Arr,
    gate: Arr,
    conv_weight: Arr,
    exp_a: Arr,
    dt_bias: Arr,
    norm: Arr,
    conv_state: Arr,
    state: Arr,
    heads: c_int,
    norm_eps: f32,
    lower: f32,
};
pub const Result = struct {
    y: Arr,
    conv: Arr,
    state: Arr,
    pub fn deinit(self: Result) void {
        _ = mlx.mlx_array_free(self.y);
        _ = mlx.mlx_array_free(self.conv);
        _ = mlx.mlx_array_free(self.state);
    }
};
var hardware_cached: ?bool = null;
pub fn hardwareSupported() bool {
    if (hardware_cached == null) hardware_cached = std.mem.startsWith(u8, @import("transformer.zig").naxStatus(), "on");
    return hardware_cached.?;
}
const Config = struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool };
const ConfigEntry = struct { heads: c_int, value: mlx.mlx_fast_metal_kernel_config };
var configs: [8]?ConfigEntry = @splat(null);
fn configuration(heads: c_int, mode: Modes) !Config {
    for (configs) |entry| if (entry) |e| {
        if (e.heads == heads) return .{ .value = e.value, .cached = true };
    };
    const d = heads * 128;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 1, d }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, d * 3 }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, heads, 128, 128 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * heads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    inline for (.{ "TOK", "DK", "QKV", "PROJ_W", "OFF_B" }, .{ @as(c_int, 1), @as(c_int, 128), d, d * 3 + heads, d * 3 }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
    inline for (.{ "SIG_B", "SIG_F", "EXP_F", "RSQ_F" }, .{ mode.sig_b, mode.sig_f, mode.exp_f, mode.rsq_f }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, @intFromBool(value)));
    for (&configs) |*entry| if (entry.* == null) {
        entry.* = .{ .heads = heads, .value = cfg };
        return .{ .value = cfg, .cached = true };
    };
    return .{ .value = cfg, .cached = false };
}

pub fn step(s: mlx.mlx_stream, in: Inputs) !?Result {
    if (!mlx.streamIsGpu(s) or in.heads < 1 or in.heads > 1024 or !std.math.isFinite(in.norm_eps) or in.norm_eps <= 0 or !std.math.isFinite(in.lower) or in.lower != -5) return null;
    for ([_]Arr{ in.qkv, in.beta, in.a, in.gate, in.conv_weight, in.exp_a, in.dt_bias, in.norm, in.conv_state, in.state }) |value| if (value.ctx == null) return null;
    const d = in.heads * 128;
    if (!std.mem.eql(c_int, &.{ 1, 1, d * 3 }, mlx.getShape(in.qkv)) or mlx.mlx_array_dtype(in.qkv) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, 1, in.heads }, mlx.getShape(in.beta)) or mlx.mlx_array_dtype(in.beta) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, 1, d }, mlx.getShape(in.a)) or mlx.mlx_array_dtype(in.a) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, 1, d }, mlx.getShape(in.gate)) or mlx.mlx_array_dtype(in.gate) != .bfloat16) return null;
    if (!std.mem.eql(c_int, &.{ d * 3, 4, 1 }, mlx.getShape(in.conv_weight)) or mlx.mlx_array_dtype(in.conv_weight) != .bfloat16 or
        mlx.mlx_array_size(in.exp_a) != @as(usize, @intCast(in.heads)) or mlx.mlx_array_dtype(in.exp_a) != .float32 or
        !std.mem.eql(c_int, &.{d}, mlx.getShape(in.dt_bias)) or mlx.mlx_array_dtype(in.dt_bias) != .float32 or
        !std.mem.eql(c_int, &.{128}, mlx.getShape(in.norm)) or mlx.mlx_array_dtype(in.norm) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, 3, d * 3 }, mlx.getShape(in.conv_state)) or mlx.mlx_array_dtype(in.conv_state) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, in.heads, 128, 128 }, mlx.getShape(in.state)) or mlx.mlx_array_dtype(in.state) != .float32) return null;
    if (!hardwareSupported()) return null;
    const mode = (try modes(s)) orelse return null;
    var projection = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(projection);
    const parts = mlx.mlx_vector_array_new_data(&[_]Arr{ in.qkv, in.beta }, 2);
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_concatenate_axis(&projection, parts, -1, s));
    const constants = [_]f32{ 1 / @sqrt(@as(f32, 128)), 1e-6, in.norm_eps, in.lower, 1.0 / 128.0 };
    const consts = mlx.mlx_array_new_data(&constants, &[_]c_int{5}, 1, .float32);
    defer _ = mlx.mlx_array_free(consts);
    if (kernel_cached == null) kernel_cached = try makeKernel("sushi_glm_kda_decode_body", &.{ "proj", "conv_w", "exp_a", "dt_bias", "norm_w", "consts", "conv_state", "state_in", "a_pre", "gate_pre" }, &.{ "y", "conv_state_out", "state_out" }, SOURCE);
    const selected = try configuration(in.heads, mode);
    const cfg = selected.value;
    defer if (!selected.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    const outputs = try applyKernel(kernel_cached.?, &.{ projection, in.conv_weight, in.exp_a, in.dt_bias, in.norm, consts, in.conv_state, in.state, in.a, in.gate }, cfg, s);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = Result{ .y = mlx.mlx_array_new(), .conv = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.y, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.conv, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&result.state, outputs, 2));
    calls += 1;
    return result;
}
