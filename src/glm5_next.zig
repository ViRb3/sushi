//! GLM-5.3-Flash forward primitives. The architecture is not yet served.
const std = @import("std");
const mlx = @import("mlx.zig");

/// Q/K are already L2-normalized (Q also scaled by Dk^-0.5).
/// Decay is exp(log-gate), not the forget projection or log-gate itself.
pub const KdaInputs = struct {
    q: mlx.mlx_array, // [B,T,H,Dk]
    k: mlx.mlx_array,
    v: mlx.mlx_array, // [B,T,H,Dv]
    decay: mlx.mlx_array, // f32 [B,T,H,Dk]
    beta: mlx.mlx_array, // [B,T,H]
    state: mlx.mlx_array, // f32 [B,H,Dv,Dk]
};

pub const KdaResult = struct {
    y: mlx.mlx_array,
    state: mlx.mlx_array,

    pub fn deinit(self: KdaResult) void {
        _ = mlx.mlx_array_free(self.y);
        _ = mlx.mlx_array_free(self.state);
    }
};

pub fn kda(in: KdaInputs, s: mlx.mlx_stream) !KdaResult {
    if (!mlx.streamIsGpu(s)) return error.KdaGpuRequired;
    const q = mlx.getShape(in.q);
    const v = mlx.getShape(in.v);
    if (q.len != 4 or v.len != 4 or !std.mem.eql(c_int, q, mlx.getShape(in.k)) or
        !std.mem.eql(c_int, q, mlx.getShape(in.decay))) return error.InvalidKdaShape;
    if (q[0] <= 0 or q[1] <= 0 or q[2] <= 0 or q[3] <= 0 or @mod(q[3], 32) != 0 or
        v[3] <= 0 or @mod(v[3], 4) != 0 or !std.mem.eql(c_int, q[0..3], v[0..3]) or
        !std.mem.eql(c_int, q[0..3], mlx.getShape(in.beta))) return error.InvalidKdaShape;
    const state_shape = [_]c_int{ q[0], q[2], v[3], q[3] };
    if (!std.mem.eql(c_int, &state_shape, mlx.getShape(in.state))) return error.InvalidKdaShape;
    const dtype = mlx.mlx_array_dtype(in.q);
    if ((dtype != .bfloat16 and dtype != .float32) or mlx.mlx_array_dtype(in.k) != dtype or
        mlx.mlx_array_dtype(in.v) != dtype or mlx.mlx_array_dtype(in.state) != .float32 or
        mlx.mlx_array_dtype(in.decay) != .float32 or
        (mlx.mlx_array_dtype(in.beta) != .float32 and mlx.mlx_array_dtype(in.beta) != .bfloat16)) return error.InvalidKdaDtype;
    const heads = try std.math.mul(c_int, q[0], q[2]);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, v.ptr, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_shape, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, v[3], heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", q[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", v[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", q[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", q[2]));
    const length = mlx.mlx_array_new_int(q[1]);
    defer _ = mlx.mlx_array_free(length);
    const arrays = [_]mlx.mlx_array{ in.q, in.k, in.v, in.decay, in.beta, in.state, length };
    const inputs = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    const kernel = try @import("transformer.zig").getGdnKernel(true);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, s));
    var result = KdaResult{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.y, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.state, outputs, 1));
    return result;
}

test "GLM KDA per-channel decay preserves FP32 state and agrees with a scalar reference" {
    const t = std.testing;
    const B = 2;
    const T = 3;
    const H = 2;
    const DK = 128;
    const DV = 4;
    var q: [B * T * H * DK]f32 = undefined;
    var k: @TypeOf(q) = undefined;
    var decay: @TypeOf(q) = undefined;
    var v: [B * T * H * DV]f32 = undefined;
    var beta: [B * T * H]f32 = undefined;
    var state: [B * H * DV * DK]f32 = undefined;
    for (&q, &k, &decay, 0..) |*a, *b, *g, i| {
        a.* = (@as(f32, @floatFromInt(i % 17)) - 8) / 128;
        b.* = (@as(f32, @floatFromInt(i % 13)) - 6) / 32;
        g.* = 0.2 + @as(f32, @floatFromInt(i % 19)) / 32;
    }
    for (&v, 0..) |*x, i| x.* = (@as(f32, @floatFromInt(i % 11)) - 5) / 8;
    for (&beta, 0..) |*x, i| x.* = 0.3 + @as(f32, @floatFromInt(i % 7)) / 16;
    for (&state, 0..) |*x, i| x.* = 0.00123 * @as(f32, @floatFromInt(i % 23));
    var expected_state: [state.len]f64 = undefined;
    for (state, &expected_state) |x, *out| out.* = x;
    var expected_y: [v.len]f64 = undefined;
    for (0..B) |b| for (0..T) |pos| for (0..H) |h| {
        const row = (b * T + pos) * H + h;
        for (0..DV) |dv| {
            const base = ((b * H + h) * DV + dv) * DK;
            var memory: f64 = 0;
            for (0..DK) |dk| {
                expected_state[base + dk] *= decay[row * DK + dk];
                memory += expected_state[base + dk] * k[row * DK + dk];
            }
            const delta = (@as(f64, v[row * DV + dv]) - memory) * beta[row];
            var out: f64 = 0;
            for (0..DK) |dk| {
                expected_state[base + dk] += k[row * DK + dk] * delta;
                out += expected_state[base + dk] * q[row * DK + dk];
            }
            expected_y[row * DV + dv] = out;
        }
    };
    const shape = [_]c_int{ B, T, H, DK };
    const arrs = [_]mlx.mlx_array{
        mlx.mlx_array_new_data(&q, &shape, 4, .float32),
        mlx.mlx_array_new_data(&k, &shape, 4, .float32),
        mlx.mlx_array_new_data(&v, &[_]c_int{ B, T, H, DV }, 4, .float32),
        mlx.mlx_array_new_data(&decay, &shape, 4, .float32),
        mlx.mlx_array_new_data(&beta, &[_]c_int{ B, T, H }, 3, .float32),
        mlx.mlx_array_new_data(&state, &[_]c_int{ B, H, DV, DK }, 4, .float32),
    };
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const s = mlx.gpuStream();
    const inputs = KdaInputs{ .q = arrs[0], .k = arrs[1], .v = arrs[2], .decay = arrs[3], .beta = arrs[4], .state = arrs[5] };
    const result = try kda(inputs, s);
    defer result.deinit();
    try mlx.check(mlx.mlx_array_eval(result.y));
    try mlx.check(mlx.mlx_array_eval(result.state));
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(result.state));
    for (expected_y, mlx.mlx_array_data_float32(result.y).?[0..v.len]) |expected, actual| try t.expectApproxEqAbs(expected, actual, 1e-6);
    for (expected_state, mlx.mlx_array_data_float32(result.state).?[0..state.len]) |expected, actual| try t.expectApproxEqAbs(expected, actual, 1e-6);
}

fn expectBitsEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    try mlx.check(mlx.mlx_array_eval(ac));
    try mlx.check(mlx.mlx_array_eval(bc));
    try std.testing.expectEqual(mlx.mlx_array_dtype(ac), mlx.mlx_array_dtype(bc));
    try std.testing.expectEqual(mlx.mlx_array_size(ac), mlx.mlx_array_size(bc));
    if (mlx.mlx_array_dtype(ac) == .float32) {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(ac).?[0..mlx.mlx_array_size(ac)]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(bc).?[0..mlx.mlx_array_size(bc)]));
    } else {
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(ac).?[0..mlx.mlx_array_size(ac)], mlx.mlx_array_data_bfloat16(bc).?[0..mlx.mlx_array_size(bc)]);
    }
}

fn timeSlice(a: mlx.mlx_array, pos: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(a);
    var starts: [4]c_int = @splat(0);
    var stops: [4]c_int = @splat(1);
    const steps: [4]c_int = @splat(1);
    @memcpy(stops[0..shape.len], shape);
    starts[1] = pos;
    stops[1] = pos + 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &starts, shape.len, &stops, shape.len, &steps, shape.len, s));
    return out;
}

test "GLM KDA chunked and serial execution preserve identical outputs and FP32 states" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const shape = [_]c_int{ 2, 3, 2, 128 };
    var values: [2 * 3 * 2 * 128]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 31)) - 15) / 64;
    const source = mlx.mlx_array_new_data(&values, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(source);
    var decay_values: [values.len]f32 = undefined;
    for (&decay_values, 0..) |*v, i| v.* = 0.1 + @as(f32, @floatFromInt(i % 29)) / 32;
    const decay = mlx.mlx_array_new_data(&decay_values, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(decay);
    const beta_values: [2 * 3 * 2]f32 = @splat(0.75);
    const beta = mlx.mlx_array_new_data(&beta_values, &[_]c_int{ 2, 3, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(beta);
    var state_values: [2 * 2 * 128 * 128]f32 = undefined;
    for (&state_values, 0..) |*v, i| v.* = 0.000123 * @as(f32, @floatFromInt(i % 53));
    const initial = mlx.mlx_array_new_data(&state_values, &[_]c_int{ 2, 2, 128, 128 }, 4, .float32);
    defer _ = mlx.mlx_array_free(initial);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| {
        var input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input);
        try mlx.check(mlx.mlx_astype(&input, source, dtype, s));
        const args = KdaInputs{ .q = input, .k = input, .v = input, .decay = decay, .beta = beta, .state = initial };
        const full = try kda(args, s);
        defer full.deinit();
        var serial = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(serial);
        try mlx.check(mlx.mlx_array_set(&serial, initial));
        for (0..3) |pos| {
            const qi = try timeSlice(input, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(qi);
            const gi = try timeSlice(decay, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(gi);
            const bi = try timeSlice(beta, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(bi);
            const step = try kda(.{ .q = qi, .k = qi, .v = qi, .decay = gi, .beta = bi, .state = serial }, s);
            defer step.deinit();
            const expected = try timeSlice(full.y, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(expected);
            try expectBitsEqual(expected, step.y, s);
            try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(step.state));
            try mlx.check(mlx.mlx_array_set(&serial, step.state));
        }
        try expectBitsEqual(full.state, serial, s);
        var rounded_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rounded_state);
        try mlx.check(mlx.mlx_astype(&rounded_state, initial, .bfloat16, s));
        var invalid = args;
        invalid.state = rounded_state;
        try t.expectError(error.InvalidKdaDtype, kda(invalid, s));
        invalid = args;
        invalid.decay = beta;
        try t.expectError(error.InvalidKdaShape, kda(invalid, s));
    }
}
