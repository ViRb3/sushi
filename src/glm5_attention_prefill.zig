//! Opt-in single-split prefill finalization; the online attention order is unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;

pub var override: ?bool = null;
pub fn enabled() bool {
    if (override) |value| return value;
    const value = std.c.getenv("SUSHI_GLM_PREFILL_DIRECT") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}
pub fn rowBytes(heads: usize, width: usize, bytes: usize) !usize {
    if (heads == 0 or width == 0 or (bytes != 2 and bytes != 4)) return error.InvalidGlmAttentionShape;
    return std.math.mul(usize, try std.math.mul(usize, heads, width), bytes);
}

pub const common: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane=thread_position_in_threadgroup.x;
    \\const uint head=threadgroup_position_in_grid.y;
    \\const uint row=threadgroup_position_in_grid.z/uint(SPLITS);
    \\const uint part=threadgroup_position_in_grid.z%uint(SPLITS);
    \\constexpr uint ITEMS=(uint(D)+31u)/32u;
    \\float query[ITEMS],acc[ITEMS];
    \\for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u; query[j]=d<uint(D)?float(q[(row*uint(H)+head)*uint(D)+d]):0.0f;acc[j]=0.0f;}
    \\const uint pos=uint(offset)+row;
    \\const uint count=SELECTED?2051u:pos+1u;
    \\const uint chunk=(count+uint(SPLITS)-1u)/uint(SPLITS);
    \\const uint begin=part*chunk,end=min(count,begin+chunk);
    \\float maximum=-INFINITY,denom=0.0f;
    \\for(uint k=begin;k<end;++k) {
    \\  const int token=SELECTED?selected[row*2051u+k]:int(k);
    \\  if(token<0 || uint(token)>pos || uint(token)>=uint(length)) continue;
    \\  float values[ITEMS]; float dot=0.0f;
    \\  for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u;values[j]=d<uint(D)?float(cache[uint(token)*uint(D)+d]):0.0f;dot+=query[j]*values[j];}
    \\  dot=simd_sum(dot)*float(scale);
    \\  float next=max(maximum,dot),old=precise::exp(maximum-next),p=precise::exp(dot-next);
    \\  denom=denom*old+p;
    \\  for(uint j=0;j<ITEMS;++j) acc[j]=acc[j]*old+p*values[j];
    \\  maximum=next;
    \\}
    \\const uint base=((row*uint(H)+head)*uint(SPLITS)+part);
;
pub const partial_tail: [:0]const u8 =
    \\for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u;if(d<uint(D)) partial[base*uint(D)+d]=acc[j];}
    \\if(lane==0) {stats[base*2u]=maximum;stats[base*2u+1u]=denom;}
;
const direct_tail: [:0]const u8 =
    \\const float global_max=max(-INFINITY,maximum);
    \\for(uint j=0;j<ITEMS;++j) {
    \\  uint d=lane+j*32u;
    \\  if(d<uint(D)) {
    \\    float sum=0.0f,normalizer=0.0f;
    \\    if(denom>0.0f) {const float w=precise::exp(maximum-global_max);sum+=w*acc[j];normalizer+=w*denom;}
    \\    out[base*uint(D)+d]=OutT(normalizer>0.0f?sum/normalizer:0.0f);
    \\  }
    \\}
;
const old_merge: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*uint(H)*uint(D)) return;
    \\const uint rowhead=i/uint(D),d=i%uint(D);
    \\float maximum=-INFINITY;
    \\for(uint p=0;p<uint(SPLITS);++p) maximum=max(maximum,stats[(rowhead*uint(SPLITS)+p)*2u]);
    \\float sum=0.0f,denom=0.0f;
    \\for(uint p=0;p<uint(SPLITS);++p) {
    \\ const uint base=rowhead*uint(SPLITS)+p;
    \\ const float n=stats[base*2u+1u];
    \\ if(n>0.0f) {const float w=precise::exp(stats[base*2u]-maximum);sum+=w*partial[base*uint(D)+d];denom+=w*n;}
    \\}
    \\out[i]=OutT(denom>0.0f?sum/denom:0.0f);
;

var cached: ?mlx.mlx_fast_metal_kernel = null;
fn makeKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const result = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (result.ctx == null) return error.MetalKernelCompileFailed;
    return result;
}
fn run(kernel: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) !mlx.mlx_vector_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var result = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(result);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&result, kernel, iv, cfg, stream));
    return result;
}
fn output(values: mlx.mlx_vector_array, index: usize) !Arr {
    var value = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(value);
    try mlx.check(mlx.mlx_vector_array_get(&value, values, index));
    return value;
}
fn argument(cfg: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
}

/// Caller validates shapes and owns query/selection construction and scratch admission.
pub fn attend(q: Arr, cache: Arr, indices: Arr, offset: Arr, length: Arr, scale: Arr, selected: bool, stream: mlx.mlx_stream) !Arr {
    if (!mlx.streamIsGpu(stream)) return error.GlmAttentionGpuRequired;
    const sh = mlx.getShape(q);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, 3, mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, sh[1], sh[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try argument(cfg, "H", sh[1]);
    try argument(cfg, "D", sh[2]);
    try argument(cfg, "SPLITS", 1);
    try argument(cfg, "SELECTED", @intFromBool(selected));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", mlx.mlx_array_dtype(q)));
    if (cached == null) cached = try makeKernel("sushi_glm_prefill_single", &.{ "q", "cache", "selected", "offset", "length", "scale" }, &.{"out"}, common ++ direct_tail);
    const values = try run(cached.?, &.{ q, cache, indices, offset, length, scale }, cfg, stream);
    defer _ = mlx.mlx_vector_array_free(values);
    return output(values, 0);
}

fn reference(q: Arr, cache: Arr, indices: Arr, offset: Arr, length: Arr, scale: Arr, selected: bool, stream: mlx.mlx_stream) !Arr {
    const sh = mlx.getShape(q);
    const pk = try makeKernel("glm_prefill_old_partial", &.{ "q", "cache", "selected", "offset", "length", "scale" }, &.{ "partial", "stats" }, common ++ partial_tail);
    defer _ = mlx.mlx_fast_metal_kernel_free(pk);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ sh[0], sh[1], 1, sh[2] }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ sh[0], sh[1], 1, 2 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, sh[1], sh[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try argument(cfg, "H", sh[1]);
    try argument(cfg, "D", sh[2]);
    try argument(cfg, "SPLITS", 1);
    try argument(cfg, "SELECTED", @intFromBool(selected));
    const old = try run(pk, &.{ q, cache, indices, offset, length, scale }, cfg, stream);
    defer _ = mlx.mlx_vector_array_free(old);
    const partial = try output(old, 0);
    defer _ = mlx.mlx_array_free(partial);
    const stats = try output(old, 1);
    defer _ = mlx.mlx_array_free(stats);
    const mk = try makeKernel("glm_prefill_old_merge", &.{ "partial", "stats" }, &.{"out"}, old_merge);
    defer _ = mlx.mlx_fast_metal_kernel_free(mk);
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, sh.ptr, 3, mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, sh[0] * sh[1] * sh[2], 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, 256, 1, 1));
    try argument(mc, "ROWS", sh[0]);
    try argument(mc, "H", sh[1]);
    try argument(mc, "D", sh[2]);
    try argument(mc, "SPLITS", 1);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OutT", mlx.mlx_array_dtype(q)));
    const ov = try run(mk, &.{ partial, stats }, mc, stream);
    defer _ = mlx.mlx_vector_array_free(ov);
    return output(ov, 0);
}
fn exact(a: Arr, b: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .bfloat16) {
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
    } else {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n]));
    }
}
fn cast(a: Arr, dtype: mlx.mlx_dtype, stream: mlx.mlx_stream) !Arr {
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_astype(&result, a, dtype, stream));
    return result;
}

test "GLM direct prefill output has bounded row storage and is opt-in" {
    try std.testing.expectEqual(@as(usize, 128), (8 * 1024 * 1024) / try rowBytes(64, 512, 2));
    try std.testing.expectEqual(@as(usize, 64), (8 * 1024 * 1024) / try rowBytes(64, 512, 4));
    try std.testing.expectError(error.InvalidGlmAttentionShape, rowBytes(0, 512, 2));
    const before = override;
    defer override = before;
    override = false;
    try std.testing.expect(!enabled());
    override = true;
    try std.testing.expect(enabled());
}

test "GLM direct prefill is raw-bit exact against old merge for BF16 FP32 masked zero NaN" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const stream = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| for ([_]bool{ false, true }) |selected| for (0..4) |kind| {
        const rows: usize = if (kind == 0) 128 else 9;
        const h: usize = if (kind == 0) 64 else 2;
        const d: usize = if (kind == 0) 512 else 4;
        const length: usize = if (kind == 0) 2057 else 17;
        const off = length - rows;
        const alloc = std.testing.allocator;
        const qv = try alloc.alloc(f32, rows * h * d);
        defer alloc.free(qv);
        const cv = try alloc.alloc(f32, length * d);
        defer alloc.free(cv);
        const ids = try alloc.alloc(i32, rows * 2051);
        defer alloc.free(ids);
        for (qv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) / 16 - 0.5;
        for (cv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 31)) / 16 - 0.75;
        if (kind == 2) {
            @memset(qv, -0.0);
            @memset(cv, -0.0);
        }
        if (kind == 3) {
            qv[0] = @bitCast(@as(u32, 0x7fc12345));
            cv[0] = @bitCast(@as(u32, 0x7f800000));
        }
        @memset(ids, -1);
        for (0..rows) |r| {
            if (kind == 1) continue;
            // Retain unsorted order; future and invalid slots must be ignored.
            ids[r * 2051] = @intCast(off + r);
            ids[r * 2051 + 1] = 0;
            ids[r * 2051 + 2] = @intCast(length + 1);
            ids[r * 2051 + 3] = @intCast(@min(length - 1, off + r + 1));
            ids[r * 2051 + 4] = 2;
        }
        const raw_q = mlx.mlx_array_new_data(qv.ptr, &.{ @intCast(rows), @intCast(h), @intCast(d) }, 3, .float32);
        defer _ = mlx.mlx_array_free(raw_q);
        const raw_c = mlx.mlx_array_new_data(cv.ptr, &.{ @intCast(length), @intCast(d) }, 2, .float32);
        defer _ = mlx.mlx_array_free(raw_c);
        const q = try cast(raw_q, dtype, stream);
        defer _ = mlx.mlx_array_free(q);
        const cache = try cast(raw_c, dtype, stream);
        defer _ = mlx.mlx_array_free(cache);
        const indices = mlx.mlx_array_new_data(ids.ptr, &.{ @intCast(rows), 2051 }, 2, .int32);
        defer _ = mlx.mlx_array_free(indices);
        const offset = mlx.mlx_array_new_int(@intCast(off));
        defer _ = mlx.mlx_array_free(offset);
        const len = mlx.mlx_array_new_int(@intCast(length));
        defer _ = mlx.mlx_array_free(len);
        const scale = mlx.mlx_array_new_float(1.0 / 16.0);
        defer _ = mlx.mlx_array_free(scale);
        const got = try attend(q, cache, indices, offset, len, scale, selected, stream);
        defer _ = mlx.mlx_array_free(got);
        const want = try reference(q, cache, indices, offset, len, scale, selected, stream);
        defer _ = mlx.mlx_array_free(want);
        try exact(got, want);
    };
}

test "GLM direct prefill chunking preserves causal pool tails across the sparse boundary" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const attention = @import("glm5_attention.zig");
    const stream = mlx.gpuStream();
    const alloc = std.testing.allocator;
    const previous = override;
    defer override = previous;
    const length = 2176;
    const rows = 129;
    const heads = 64;
    const dim = 512;
    const iw = 4;
    const ih = 2;
    const lv = try alloc.alloc(f32, length * dim);
    defer alloc.free(lv);
    const kv = try alloc.alloc(f32, length * iw);
    defer alloc.free(kv);
    const gv = try alloc.alloc(f32, length * iw);
    defer alloc.free(gv);
    const qv = try alloc.alloc(f32, rows * heads * dim);
    defer alloc.free(qv);
    const iqv = try alloc.alloc(f32, rows * ih * iw);
    defer alloc.free(iqv);
    for (lv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 37)) / 32 - 0.5;
    for (kv, gv, 0..) |*k, *g, i| {
        k.* = @as(f32, @floatFromInt(i % 29)) / 32 - 0.25;
        g.* = @as(f32, @floatFromInt(i % 17)) / 16 - 0.5;
    }
    for (qv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 23)) / 32 - 0.25;
    for (iqv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) / 16 - 0.25;
    const latent_f = mlx.mlx_array_new_data(lv.ptr, &.{ length, dim }, 2, .float32);
    defer _ = mlx.mlx_array_free(latent_f);
    const keys_f = mlx.mlx_array_new_data(kv.ptr, &.{ length, iw }, 2, .float32);
    defer _ = mlx.mlx_array_free(keys_f);
    const gates_f = mlx.mlx_array_new_data(gv.ptr, &.{ length, iw }, 2, .float32);
    defer _ = mlx.mlx_array_free(gates_f);
    const q_f = mlx.mlx_array_new_data(qv.ptr, &.{ rows, heads, dim }, 3, .float32);
    defer _ = mlx.mlx_array_free(q_f);
    const iq_f = mlx.mlx_array_new_data(iqv.ptr, &.{ rows, ih, iw }, 3, .float32);
    defer _ = mlx.mlx_array_free(iq_f);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| {
        const latent = try cast(latent_f, dtype, stream);
        defer _ = mlx.mlx_array_free(latent);
        const keys = try cast(keys_f, dtype, stream);
        defer _ = mlx.mlx_array_free(keys);
        const gates = try cast(gates_f, dtype, stream);
        defer _ = mlx.mlx_array_free(gates);
        const q = try cast(q_f, dtype, stream);
        defer _ = mlx.mlx_array_free(q);
        const iq = try cast(iq_f, dtype, stream);
        defer _ = mlx.mlx_array_free(iq);
        var ape = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ape);
        try mlx.check(mlx.mlx_zeros(&ape, &.{ 4, iw }, 2, dtype, stream));
        var weights = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weights);
        try mlx.check(mlx.mlx_ones(&weights, &.{ rows, ih }, 2, dtype, stream));
        var state = attention.State.init();
        defer state.deinit();
        _ = try state.append(latent, keys, gates, ape, stream);
        override = false;
        const old = try attention.attend(&state, q, iq, weights, 2047, 1.0 / 16.0, stream);
        defer _ = mlx.mlx_array_free(old);
        try mlx.check(mlx.mlx_array_eval(old));
        override = true;
        const fused = try attention.attend(&state, q, iq, weights, 2047, 1.0 / 16.0, stream);
        defer _ = mlx.mlx_array_free(fused);
        try exact(old, fused);
        try std.testing.expectEqual(@as(usize, length), state.processed);
    }
}
