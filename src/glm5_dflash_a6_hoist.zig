//! Exact A6 masked-coefficient hoist; original row arithmetic retained.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const Ops = @import("glm5_model.zig").Ops;
const Linear = @import("glm5_model.zig").Linear;
// The per-row dot/update order is inherited from glm5_decode's MLX-derived qmv_fast.
// Each admitted width occupies one complete row tile, so its offset is constant.
const SOURCE =
    \\const int lane = int(thread_index_in_simdgroup);
    \\const int output0 = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup) * 2;
    \\const int token0 = 0;
    \\const device uint8_t* codes = reinterpret_cast<const device uint8_t*>(w);
    \\float result[R][2];
    \\for (int m = 0; m < R; ++m) for (int r = 0; r < 2; ++r) result[m][r] = 0.0f;
    \\for (int k = 0; k < K; k += 256) {
    \\  float local[R][8];
    \\  float sum[R];
    \\  for (int m = 0; m < R; ++m) {
    \\    sum[m] = 0.0f;
    \\    if (BITS == 6) {
    \\      for (int i = 0; i < 8; i += 4) {
    \\        const int at = (token0+m)*K+k+lane*8+i;
    \\        if (token0+m < M) sum[m] += x[at]+x[at+1]+x[at+2]+x[at+3];
    \\        local[m][i] = token0+m < M ? float(x[at]) : 0.0f;
    \\        local[m][i+1] = token0+m < M ? float(x[at+1])/64.0f : 0.0f;
    \\        local[m][i+2] = token0+m < M ? float(x[at+2])/16.0f : 0.0f;
    \\        local[m][i+3] = token0+m < M ? float(x[at+3])/4.0f : 0.0f;
    \\      }
    \\    } else {
    \\    for (int i = 0; i < 8; ++i) {
    \\      const float value = token0 + m < M ? float(x[(token0 + m) * K + k + lane * 8 + i]) : 0.0f;
    \\      sum[m] += value;
    \\      local[m][i] = value;
    \\    }
    \\    }
    \\  }
    \\  for (int r = 0; r < 2; ++r) {
    \\    const int output = output0 + r;
    \\    const int group = output * (K / 128) + k / 128 + lane / 16;
    \\    const float scale = float(scales[group]);
    \\    const float bias = float(biases[group]);
    \\    float quant[8];
    \\    if (BITS == 8) for (int i = 0; i < 8; ++i) quant[i] = float(codes[size_t(output) * K + k + lane * 8 + i]);
    \\    float coeff[12];
    \\    if (BITS == 6) {
    \\      const device uint8_t* packed = codes + size_t(output)*(K*3/4) + (k+lane*8)*3/4;
    \\      for (int pack = 0; pack < 2; ++pack) {
    \\        const device uint8_t* q = packed + pack*3;
    \\        const int c = pack*6;
    \\        coeff[c] = float(q[0]&0x3f);
    \\        coeff[c+1] = float(q[0]&0xc0);
    \\        coeff[c+2] = float(q[1]&0x0f);
    \\        coeff[c+3] = float(q[1]&0xf0);
    \\        coeff[c+4] = float(q[2]&0x03);
    \\        coeff[c+5] = float(q[2]&0xfc);
    \\      }
    \\    }
    \\    for (int m = 0; m < R; ++m) {
    \\      float accum = 0.0f;
    \\      if (BITS == 6) {
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\          const int i = pack*4, c = pack*6;
    \\          accum += coeff[c]*local[m][i];
    \\          accum += coeff[c+1]*local[m][i+1];
    \\          accum += coeff[c+2]*(local[m][i+1]*256.0f);
    \\          accum += coeff[c+3]*local[m][i+2];
    \\          accum += coeff[c+4]*(local[m][i+2]*256.0f);
    \\          accum += coeff[c+5]*local[m][i+3];
    \\        }
    \\      } else for (int i = 0; i < 8; ++i) accum += local[m][i] * quant[i];
    \\      result[m][r] += scale * accum + sum[m] * bias;
    \\    }
    \\  }
    \\}
    \\for (int m = 0; m < R; ++m) for (int r = 0; r < 2; ++r) {
    \\  const float value = simd_sum(result[m][r]);
    \\  if (lane == 0 && token0 + m < M) y[(token0 + m) * N + output0 + r] = bfloat(value);
    \\}
;
// The MLA value bank adds a head axis; its per-row arithmetic is identical.
fn replace(comptime source: []const u8, comptime old: []const u8, comptime value: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const at = std.mem.indexOf(u8, source, old) orelse return source ++ "";
    return source[0..at] ++ value ++ replace(source[at + old.len ..], old, value);
}
const HEAD_SOURCE: [:0]const u8 = blk: {
    var body = replace(SOURCE, "int(simdgroup_index_in_threadgroup) * 2", "int(simdgroup_index_in_threadgroup) * 4");
    body = replace(body, "result[R][2]", "result[R][4]");
    body = replace(body, "r < 2", "r < 4");
    body = replace(body, "const int token0 = 0;", "const int token0 = 0;\nconst int head = int(threadgroup_position_in_grid.z);");
    body = replace(body, "reinterpret_cast<const device uint8_t*>(w);", "reinterpret_cast<const device uint8_t*>(w) + size_t(head)*N*K*3/4;");
    body = replace(body, "(token0+m)*K", "((token0+m)*64+head)*K");
    body = replace(body, "(token0 + m) * K", "((token0 + m)*64+head) * K");
    body = replace(body, "const int group = output *", "const int group = head*N*(K/128) + output *");
    body = replace(body, "(token0 + m) * N", "((token0 + m)*64+head) * N");
    break :blk body;
};
var dispatch_count: usize = 0;
pub fn dispatchCount() usize {
    return dispatch_count;
}
pub fn resetDispatchCount() void {
    dispatch_count = 0;
}
var kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);
const Key = struct { m: c_int, n: c_int, k: c_int, bits: c_int, headed: bool };
const Cached = struct { key: Key, config: mlx.mlx_fast_metal_kernel_config };
var configs: [32]?Cached = @splat(null);

fn rowMajorReady(a: Arr) !bool {
    if (a.ctx == null) return false;
    const shape = mlx.getShape(a);
    if (shape.len != 2 and shape.len != 3) return false;
    const strides = mlx.mlx_array_strides(a);
    const last = shape.len - 1;
    if (strides[last] != 1 or strides[last - 1] != shape[last]) return false;
    if (shape.len == 3 and strides[0] != @as(i64, shape[1]) * shape[2]) return false;
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, a));
    return available;
}

pub fn project(stream: mlx.mlx_stream, x: Arr, linear: Linear) !?Arr {
    if (!mlx.streamIsGpu(stream) or x.ctx == null or linear.w.ctx == null or linear.scales.ctx == null or linear.biases.ctx == null) return null;
    const sh = mlx.getShape(x);
    const ws = mlx.getShape(linear.w);
    const headed = sh.len == 4;
    if (headed) {
        if (!std.mem.eql(c_int, sh, &.{ 3, 64, 1, 512 }) or !std.mem.eql(c_int, ws, &.{ 64, 256, 96 })) return null;
    } else if (sh.len != 3 or sh[0] != 1 or (sh[1] != 3 and sh[1] != 4) or sh[2] < 256 or @mod(sh[2], 256) != 0 or ws.len != 2 or ws[0] < 8 or @mod(ws[0], 8) != 0) return null;
    const k = sh[sh.len - 1];
    const n = ws[ws.len - 2];
    const bits: c_int = if (@as(i64, ws[ws.len - 1]) * 16 == @as(i64, k) * 3) 6 else return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(linear.w) != .uint32 or !(try rowMajorReady(linear.w))) return null;
    const grid_shape = [_]c_int{ 64, n, @divExact(k, 128) };
    for ([_]Arr{ linear.scales, linear.biases }) |grid| {
        if (mlx.mlx_array_dtype(grid) != .bfloat16 or !std.mem.eql(c_int, if (headed) &grid_shape else grid_shape[1..], mlx.getShape(grid)) or !(try rowMajorReady(grid))) return null;
    }
    const kernel = &kernels[@intFromBool(headed)];
    if (kernel.* == null) {
        const ins = [_][*:0]const u8{ "x", "w", "scales", "biases" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        kernel.* = mlx.mlx_fast_metal_kernel_new(if (headed) "sushi_glm_value_three_rows" else "sushi_glm_dflash_affine6_hoisted", iv, ov, if (headed) HEAD_SOURCE else SOURCE, "", true, false);
        if (kernel.*.?.ctx == null) {
            kernel.* = null;
            return error.MetalKernelCompileFailed;
        }
    }
    const key = Key{ .m = if (headed) sh[0] else sh[1], .n = n, .k = k, .bits = bits, .headed = headed };
    var cached: ?mlx.mlx_fast_metal_kernel_config = null;
    for (configs) |item| if (item) |entry| if (std.meta.eql(key, entry.key)) {
        cached = entry.config;
        break;
    };
    const config = cached orelse mlx.mlx_fast_metal_kernel_config_new();
    var retained = cached != null;
    defer if (!retained) {
        _ = mlx.mlx_fast_metal_kernel_config_free(config);
    };
    if (cached == null) {
        const tile: c_int = if (key.m < 4) key.m else 4;
        const output_shape: []const c_int = if (headed) &.{ key.m, 64, 1, key.n } else &.{ 1, key.m, key.n };
        const threads: c_int = if (headed) 64 else 128;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, output_shape.ptr, output_shape.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divExact(key.n, 8) * threads, @divTrunc(key.m + tile - 1, tile), if (headed) 64 else 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "M", key.m));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "N", key.n));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", key.k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "R", tile));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", bits));
        for (&configs) |*item| if (item.* == null) {
            item.* = .{ .key = key, .config = config };
            retained = true;
            break;
        };
    }
    const arrays = [_]Arr{ x, linear.w, linear.scales, linear.biases };
    const iv = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.*.?, iv, config, stream));
    var output = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_vector_array_get(&output, ov, 0));
    dispatch_count += 1;
    return output;
}
