//! Fixed exact T3 mHC coefficients: one uniform SIMD32 subgroup per row.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const Result = @import("glm5_next.zig").HcResult;
const Ops = @import("glm5_model.zig").Ops;
var calls: usize = 0;
pub fn enabled() bool {
    return !@import("glm5_model.zig").reference_numerics;
}
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}
pub fn geometry(x: []const c_int, mixes: []const c_int, scale: []const c_int, base: []const c_int, iters: c_int, epsilon: f32) bool {
    return std.mem.eql(c_int, x, &.{ 1, 3, 4, 4096 }) and std.mem.eql(c_int, mixes, &.{ 1, 3, 24 }) and
        std.mem.eql(c_int, scale, &.{3}) and std.mem.eql(c_int, base, &.{24}) and iters == 20 and std.math.isFinite(epsilon) and epsilon > 0;
}
const SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint row=threadgroup_position_in_grid.x;
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    \\threadgroup float pre[4];
    \\threadgroup float matrix[16];
    \\const float epsilon=float(eps);
    \\if(sg==0u) {
    \\  if(lane<4u) {
    \\    float z=mixes[row*24u+lane]*scale[0]+base[lane];
    \\    pre[lane]=1.0f/(1.0f+precise::exp(-z))+epsilon;
    \\    z=mixes[row*24u+4u+lane]*scale[1]+base[4u+lane];
    \\    post[row*4u+lane]=2.0f/(1.0f+precise::exp(-z));
    \\  }
    \\  float value=0.0f;
    \\  if(lane<16u) value=mixes[row*24u+8u+lane]*scale[2]+base[8u+lane];
    \\  const uint j=(lane&15u)/4u, k=lane&3u;
    \\  float maximum=-INFINITY;
    \\  for(uint c=0u;c<4u;++c) maximum=max(maximum,simd_shuffle(value,j*4u+c));
    \\  value=precise::exp(value-maximum);
    \\  float sum=0.0f;
    \\  for(uint c=0u;c<4u;++c) sum+=simd_shuffle(value,j*4u+c);
    \\  value=value/sum+epsilon;
    \\  for(uint iter=0u;iter<20u;++iter) {
    \\    if(iter!=0u) {
    \\      sum=0.0f;
    \\      for(uint c=0u;c<4u;++c) sum+=simd_shuffle(value,j*4u+c);
    \\      value/=sum+epsilon;
    \\    }
    \\    sum=0.0f;
    \\    for(uint r=0u;r<4u;++r) sum+=simd_shuffle(value,r*4u+k);
    \\    value/=sum+epsilon;
    \\  }
    \\  if(lane<16u) {matrix[lane]=value;comb[row*16u+lane]=value;}
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for(uint d=tid;d<4096u;d+=256u) {
    \\  float value=0.0f;
    \\  for(uint j=0u;j<4u;++j) value+=pre[j]*float(x[(row*4u+j)*4096u+d]);
    \\  mixed[row*4096u+d]=OutT(value);
    \\}
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;
pub fn collapse(x: Arr, mixes: Arr, scale: Arr, base: Arr, iters: c_int, epsilon: f32, s: mlx.mlx_stream) !?Result {
    if (!mlx.streamIsGpu(s)) return null;
    for ([_]Arr{ x, mixes, scale, base }) |a| if (a.ctx == null) return null;
    if (!geometry(mlx.getShape(x), mlx.getShape(mixes), mlx.getShape(scale), mlx.getShape(base), iters, epsilon) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    for ([_]Arr{ mixes, scale, base }) |a| if (mlx.mlx_array_dtype(a) != .float32) return null;
    if (kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "x", "mixes", "scale", "base", "eps" }, 5);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "mixed", "post", "comb" }, 3);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_hc_collapse_simd32", ins, outs, SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, 4096 }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, 4 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, 4, 4 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 3 * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", .bfloat16));
    const eps = mlx.mlx_array_new_float(epsilon);
    defer _ = mlx.mlx_array_free(eps);
    const inputs = mlx.mlx_vector_array_new_data(&.{ x, mixes, scale, base, eps }, 5);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.?, inputs, cfg, s));
    if (mlx.mlx_vector_array_size(outputs) != 3) return error.MetalKernelBadOutputCount;
    var result = Result{ .mixed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.mixed, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.post, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&result.comb, outputs, 2));
    calls += 1;
    return result;
}

fn exact(ops: *Ops, a: Arr, b: Arr) !void {
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    const ev = mlx.mlx_vector_array_new_data(&.{ x, y }, 2);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(x), mlx.getShape(y));
    try std.testing.expectEqual(mlx.mlx_array_dtype(x), mlx.mlx_array_dtype(y));
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_dtype(x) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(x).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(y).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(x).?[0..n], mlx.mlx_array_data_bfloat16(y).?[0..n]);
}
fn check(ops: *Ops, x: Arr, mix: Arr, scale: Arr, base: Arr, expected_calls: usize) !void {
    const hc = @import("glm5_next.zig");
    const model = @import("glm5_model.zig");
    resetDispatchCount();
    const reference = blk: {
        model.reference_numerics = true;
        defer model.reference_numerics = false;
        break :blk try hc.hcCollapse(x, mix, scale, base, 20, 1e-6, ops.s);
    };
    defer reference.deinit();
    try std.testing.expectEqual(@as(usize, 0), dispatchCount());
    const new = try hc.hcCollapse(x, mix, scale, base, 20, 1e-6, ops.s);
    defer new.deinit();
    try std.testing.expectEqual(expected_calls, dispatchCount());
    try exact(ops, reference.mixed, new.mixed);
    try exact(ops, reference.post, new.post);
    try exact(ops, reference.comb, new.comb);
}
test "GLM HC SIMD32 primitive hook exact T3 specials and original fallback" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const x = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 3, 4, 4096 }, 731, ops.s));
    var values: [72]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 8 - 1;
    const mix = try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 24 }, 3, .float32));
    const scale = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.125, 0.25, 0.0625 }, &.{3}, 1, .float32));
    var bases: [24]f32 = undefined;
    for (&bases, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 16 - 0.25;
    const base = try ops.own(mlx.mlx_array_new_data(&bases, &.{24}, 1, .float32));
    try check(&ops, x, mix, scale, base, 1);
    values[0] = -0.0;
    values[1] = 0.0;
    values[8] = std.math.nan(f32);
    values[9] = std.math.inf(f32);
    values[10] = -std.math.inf(f32);
    values[24 + 8] = 1000;
    values[24 + 9] = -1000;
    const special = try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 24 }, 3, .float32));
    try check(&ops, x, special, scale, base, 1);
    try check(&ops, try ops.slice(x, 1, 0, 2), try ops.slice(mix, 1, 0, 2), scale, base, 0);
    try check(&ops, try ops.cast(x, .float32), mix, scale, base, 0);
}
