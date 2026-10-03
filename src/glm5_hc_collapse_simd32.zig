//! Fixed exact T3 mHC coefficients: one uniform SIMD32 subgroup per row.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const Result = @import("glm5_next.zig").HcResult;
threadlocal var enabled_override: ?bool = null;
var calls: usize = 0;
pub const Binding = struct {
    previous: ?bool,
    pub fn restore(self: Binding) void {
        enabled_override = self.previous;
    }
};
pub fn bind(on: bool) Binding {
    const old = Binding{ .previous = enabled_override };
    enabled_override = on;
    return old;
}
pub fn enabled() bool {
    return enabled_override orelse @import("transformer.zig").diagEnvOn("SUSHI_GLM_HC_COLLAPSE_SIMD32");
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
