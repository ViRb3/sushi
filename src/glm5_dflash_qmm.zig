//! Affine8/group128 tree projections with each serial qmv row's reduction order.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const Ops = @import("glm5_model.zig").Ops;
const Linear = @import("glm5_model.zig").Linear;

// The per-row dot/update order is inherited from glm5_decode's MLX-derived qmv_fast.
const SOURCE =
    \\const int lane = int(thread_index_in_simdgroup);
    \\const int output0 = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup) * 4;
    \\const int token0 = int(threadgroup_position_in_grid.y) * R;
    \\const device uint8_t* codes = reinterpret_cast<const device uint8_t*>(w);
    \\float result[R][4];
    \\for (int m = 0; m < R; ++m) for (int r = 0; r < 4; ++r) result[m][r] = 0.0f;
    \\for (int k = 0; k < K; k += 256) {
    \\  float local[R][8];
    \\  float sum[R];
    \\  for (int m = 0; m < R; ++m) {
    \\    sum[m] = 0.0f;
    \\    for (int i = 0; i < 8; ++i) {
    \\      const float value = token0 + m < M ? float(x[(token0 + m) * K + k + lane * 8 + i]) : 0.0f;
    \\      sum[m] += value;
    \\      local[m][i] = value;
    \\    }
    \\  }
    \\  for (int r = 0; r < 4; ++r) {
    \\    const int output = output0 + r;
    \\    const int group = output * (K / 128) + k / 128 + lane / 16;
    \\    const float scale = float(scales[group]);
    \\    const float bias = float(biases[group]);
    \\    float quant[8];
    \\    for (int i = 0; i < 8; ++i) quant[i] = float(codes[size_t(output) * K + k + lane * 8 + i]);
    \\    for (int m = 0; m < R; ++m) {
    \\      float accum = 0.0f;
    \\      for (int i = 0; i < 8; ++i) accum += local[m][i] * quant[i];
    \\      result[m][r] += scale * accum + sum[m] * bias;
    \\    }
    \\  }
    \\}
    \\for (int m = 0; m < R; ++m) for (int r = 0; r < 4; ++r) {
    \\  const float value = simd_sum(result[m][r]);
    \\  if (lane == 0 && token0 + m < M) y[(token0 + m) * N + output0 + r] = bfloat(value);
    \\}
;
var dispatch_count: usize = 0;
pub fn dispatchCount() usize {
    return dispatch_count;
}
pub fn resetDispatchCount() void {
    dispatch_count = 0;
}
var kernel: ?mlx.mlx_fast_metal_kernel = null;
const Key = struct { m: c_int, n: c_int, k: c_int };
const Cached = struct { key: Key, config: mlx.mlx_fast_metal_kernel_config };
var configs: [32]?Cached = @splat(null);

fn rowMajorReady(a: Arr) !bool {
    if (a.ctx == null) return false;
    const shape = mlx.getShape(a);
    if (shape.len != 2) return false;
    const strides = mlx.mlx_array_strides(a);
    if (strides[1] != 1 or strides[0] != shape[1]) return false;
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, a));
    return available;
}

pub fn project(stream: mlx.mlx_stream, x: Arr, linear: Linear) !?Arr {
    if (!mlx.streamIsGpu(stream) or x.ctx == null or linear.w.ctx == null or linear.scales.ctx == null or linear.biases.ctx == null) return null;
    const sh = mlx.getShape(x);
    const ws = mlx.getShape(linear.w);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 1 or sh[1] > 16 or sh[2] < 256 or @mod(sh[2], 256) != 0 or ws.len != 2 or ws[0] < 8 or @mod(ws[0], 8) != 0 or ws[1] != @divExact(sh[2], 4)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(linear.w) != .uint32 or !(try rowMajorReady(linear.w))) return null;
    for ([_]Arr{ linear.scales, linear.biases }) |grid| {
        if (mlx.mlx_array_dtype(grid) != .bfloat16 or !std.mem.eql(c_int, &.{ ws[0], @divExact(sh[2], 128) }, mlx.getShape(grid)) or !(try rowMajorReady(grid))) return null;
    }
    if (kernel == null) {
        const ins = [_][*:0]const u8{ "x", "w", "scales", "biases" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        kernel = mlx.mlx_fast_metal_kernel_new("sushi_glm_dflash_affine_rows", iv, ov, SOURCE, "", true, false);
        if (kernel.?.ctx == null) {
            kernel = null;
            return error.MetalKernelCompileFailed;
        }
    }
    const key = Key{ .m = sh[1], .n = ws[0], .k = sh[2] };
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
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, key.m, key.n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divExact(key.n, 8) * 64, @divTrunc(key.m + tile - 1, tile), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "M", key.m));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "N", key.n));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", key.k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "R", tile));
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
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, config, stream));
    var output = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_vector_array_get(&output, ov, 0));
    dispatch_count += 1;
    return output;
}

test "GLM DFlash affine row tiles preserve serial qmv bits" {
    const s = mlx.gpuStream();
    const Shape = struct { n: c_int, k: c_int };
    for ([_]Shape{ .{ .n = 32, .k = 256 }, .{ .n = 1536, .k = 4096 }, .{ .n = 8192, .k = 4096 }, .{ .n = 4096, .k = 8192 }, .{ .n = 154880, .k = 4096 } }) |shape| {
        if (shape.n == 154880 and std.c.getenv("SUSHI_GLM_DFLASH_HEAD_FIXTURE") == null) continue;
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(shape.n + shape.k)));
        const codes = try ops.slot();
        try mlx.check(mlx.mlx_random_bits(codes, &[_]c_int{ shape.n, @divExact(shape.k, 4) }, 2, 4, key.*, s));
        const scales = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(scales, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.02, .bfloat16), &[_]c_int{ shape.n, @divExact(shape.k, 128) }, 2, .bfloat16, key.*, s));
        const biases = try ops.binary(.mul, scales.*, try ops.scalar(-127.5, .bfloat16));
        const linear = Linear{ .w = codes.*, .scales = scales.*, .biases = biases, .input = shape.k, .output = shape.n };
        const x = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(x, &[_]c_int{ 1, 16, shape.k }, 3, .bfloat16, 0, 1, key.*, s));
        for ([_]Arr{ linear.w, linear.scales, linear.biases, x.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
        for ([_]c_int{ 1, 2, 3, 4, 5, 8, 16 }) |rows| {
            var scope = Ops{ .s = s };
            defer scope.deinit();
            const input = try scope.slice(x.*, 1, 0, rows);
            const actual = try scope.own((try project(s, input, linear)) orelse return error.TestExpectedGlmTreeQmm);
            try mlx.check(mlx.mlx_array_eval(actual));
            if (rows > 1) {
                const before = dispatchCount();
                const integrated = try @import("glm5_dflash_kda.zig").linearRows(&scope, linear, input, .affine_rows);
                try mlx.check(mlx.mlx_array_eval(integrated));
                try std.testing.expectEqual(before + 1, dispatchCount());
                const total: usize = @intCast(rows * shape.n);
                try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(actual).?[0..total], mlx.mlx_array_data_bfloat16(integrated).?[0..total]);
            }
            for (0..@intCast(rows)) |row| {
                var one = Ops{ .s = s };
                defer one.deinit();
                const expected = try linear.apply(&one, try one.slice(input, 1, @intCast(row), @intCast(row + 1)));
                const got = try one.slice(actual, 1, @intCast(row), @intCast(row + 1));
                try mlx.check(mlx.mlx_array_eval(expected));
                try mlx.check(mlx.mlx_array_eval(got));
                const count: usize = @intCast(shape.n);
                try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..count], mlx.mlx_array_data_bfloat16(got).?[0..count]);
            }
        }
    }
}

test "GLM DFlash affine row tiles decline incompatible grids and layouts" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.ones(&.{ 1, 2, 256 }, .bfloat16);
    const linear = Linear{ .w = try ops.zeros(&.{ 8, 64 }, .uint32), .scales = try ops.ones(&.{ 8, 2 }, .bfloat16), .biases = try ops.zeros(&.{ 8, 2 }, .bfloat16), .input = 256, .output = 8 };
    for ([_]Arr{ linear.w, linear.scales, linear.biases }) |a| try mlx.check(mlx.mlx_array_eval(a));
    const before = dispatchCount();
    var absent = linear;
    absent.scales = .{ .ctx = null };
    try std.testing.expect((try project(s, x, absent)) == null);
    var wrong_group = linear;
    wrong_group.scales = try ops.ones(&.{ 8, 4 }, .bfloat16);
    try std.testing.expect((try project(s, x, wrong_group)) == null);
    var wrong_dtype = linear;
    wrong_dtype.biases = try ops.zeros(&.{ 8, 2 }, .float32);
    try std.testing.expect((try project(s, x, wrong_dtype)) == null);
    var dense = linear;
    dense.w = try ops.zeros(&.{ 8, 256 }, .bfloat16);
    try std.testing.expect((try project(s, x, dense)) == null);
    var strided = linear;
    strided.w = try ops.transpose(try ops.zeros(&.{ 64, 8 }, .uint32), &.{ 1, 0 });
    try mlx.check(mlx.mlx_array_eval(strided.w));
    try std.testing.expect((try project(s, x, strided)) == null);
    try std.testing.expect((try project(s, try ops.ones(&.{ 1, 17, 256 }, .bfloat16), linear)) == null);
    try std.testing.expect((try project(s, try ops.cast(x, .float32), linear)) == null);
    try std.testing.expectEqual(before, dispatchCount());
}
