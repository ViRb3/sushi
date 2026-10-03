//! Exact short-row HC collapse and following BF16 RMS normalization.
const std = @import("std");
const mlx = @import("mlx.zig");
const primitive = @import("glm5_next.zig");
const Arr = mlx.mlx_array;
pub const Output = struct {
    mixed: Arr,
    post: Arr,
    comb: Arr,
    normalized: Arr,
    pub fn deinit(self: Output) void {
        for ([_]Arr{ self.mixed, self.post, self.comb, self.normalized }) |v| {
            _ = mlx.mlx_array_free(v);
        }
    }
};
// Collapse retains glm5_next arithmetic; the 4096-wide RMS maps MLX
// rms_single_row (MIT), including its two separate BF16 rounding boundaries.
const SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint row = threadgroup_position_in_grid.x;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\threadgroup float pre[4];
    \\threadgroup float matrix[16];
    \\const float epsilon = float(eps);
    \\if (tid == 0) {
    \\  for (uint j = 0; j < 4; ++j) {
    \\    float z = mixes[row * 24 + j] * scale[0] + base[j];
    \\    pre[j] = 1.0f / (1.0f + precise::exp(-z)) + epsilon;
    \\    z = mixes[row * 24 + 4 + j] * scale[1] + base[4 + j];
    \\    post[row * 4 + j] = 2.0f / (1.0f + precise::exp(-z));
    \\  }
    \\  for (uint j = 0; j < 4; ++j) {
    \\    float maximum = -INFINITY;
    \\    for (uint k = 0; k < 4; ++k) {
    \\      uint i = j * 4 + k;
    \\      matrix[i] = mixes[row * 24 + 8 + i] * scale[2] + base[8 + i];
    \\      maximum = max(maximum, matrix[i]);
    \\    }
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < 4; ++k) { matrix[j * 4 + k] = precise::exp(matrix[j * 4 + k] - maximum); sum += matrix[j * 4 + k]; }
    \\    for (uint k = 0; k < 4; ++k) matrix[j * 4 + k] = matrix[j * 4 + k] / sum + epsilon;
    \\  }
    \\  for (uint iter = 0; iter < uint(ITERS); ++iter) {
    \\    if (iter != 0) {
    \\      for (uint j = 0; j < 4; ++j) {
    \\        float sum = 0.0f;
    \\        for (uint k = 0; k < 4; ++k) sum += matrix[j * 4 + k];
    \\        for (uint k = 0; k < 4; ++k) matrix[j * 4 + k] /= sum + epsilon;
    \\      }
    \\    }
    \\    for (uint k = 0; k < 4; ++k) {
    \\      float sum = 0.0f;
    \\      for (uint j = 0; j < 4; ++j) sum += matrix[j * 4 + k];
    \\      for (uint j = 0; j < 4; ++j) matrix[j * 4 + k] /= sum + epsilon;
    \\    }
    \\  }
    \\  for (uint i = 0; i < 16; ++i) comb[row * 16 + i] = matrix[i];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float thread_x[4];
    \\for (uint i=0; i<4; ++i) {
    \\  const uint d=tid*4+i;
    \\  float value=0.0f;
    \\  for (uint j=0; j<4; ++j) value+=pre[j]*float(x[(row*4+j)*4096u+d]);
    \\  const bfloat rounded=bfloat(value);
    \\  mixed[row*4096u+d]=rounded;
    \\  thread_x[i]=float(rounded);
    \\}
    \\{
    \\#pragma clang fp contract(fast)
    \\const uint lane=thread_index_in_simdgroup;
    \\const uint sg=simdgroup_index_in_threadgroup;
    \\threadgroup float sums[32];
    \\threadgroup float inverse[1];
    \\float acc=0.0f;
    \\for (uint i=0; i<4; ++i) acc+=thread_x[i]*thread_x[i];
    \\acc=simd_sum(acc);
    \\if(lane==0) sums[sg]=acc;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(sg==0) {acc=simd_sum(sums[lane]); if(lane==0) inverse[0]=metal::precise::rsqrt(acc/4096.0f+float(norm_eps));}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for(uint i=0; i<4; ++i) {
    \\  const uint d=tid*4+i;
    \\  normalized[row*4096u+d]=weight[d]*bfloat(thread_x[i]*inverse[0]);
    \\}
    \\}
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var dispatch_count: usize = 0;
pub fn dispatchCount() usize {
    return dispatch_count;
}
pub fn resetDispatchCount() void {
    dispatch_count = 0;
}
const Key = struct { rows: c_int, iters: c_int };
const Entry = struct { key: Key, config: mlx.mlx_fast_metal_kernel_config };
var configs: [8]?Entry = @splat(null);
pub fn apply(s: mlx.mlx_stream, x: Arr, mixes: Arr, scale: Arr, base: Arr, weight: Arr, iters: c_int, hc_epsilon: f32, norm_epsilon: f32) !?Output {
    if (!mlx.streamIsGpu(s) or !std.math.isFinite(hc_epsilon) or hc_epsilon <= 0 or !std.math.isFinite(norm_epsilon) or norm_epsilon <= 0 or iters < 1 or iters > 100) return null;
    for ([_]Arr{ x, mixes, scale, base, weight }) |v| if (v.ctx == null) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 4 or sh[0] != 1 or sh[1] < 1 or sh[1] > 4 or sh[2] != 4 or sh[3] != 4096 or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(weight) != .bfloat16 or !std.mem.eql(c_int, &.{4096}, mlx.getShape(weight))) return null;
    if (mlx.mlx_array_dtype(mixes) != .float32 or mlx.mlx_array_size(mixes) != @as(usize, @intCast(sh[1])) * 24 or mlx.mlx_array_dtype(scale) != .float32 or mlx.mlx_array_size(scale) != 3 or mlx.mlx_array_dtype(base) != .float32 or mlx.mlx_array_size(base) != 24) return null;
    if (kernel == null) {
        const names = [_][*:0]const u8{ "x", "mixes", "scale", "base", "weight", "eps", "norm_eps" };
        const outs = [_][*:0]const u8{ "mixed", "post", "comb", "normalized" };
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        kernel = mlx.mlx_fast_metal_kernel_new("sushi_glm_hc_collapse_norm", iv, ov, SOURCE, "", true, false);
        if (kernel.?.ctx == null) {
            kernel = null;
            return error.MetalKernelCompileFailed;
        }
    }
    const key = Key{ .rows = sh[1], .iters = iters };
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
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &.{ 1, key.rows, 4096 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &.{ 1, key.rows, 4 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &.{ 1, key.rows, 4, 4 }, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &.{ 1, key.rows, 4096 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, key.rows * 1024, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 1024, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "ITERS", iters));
        for (&configs) |*item| if (item.* == null) {
            item.* = .{ .key = key, .config = config };
            retained = true;
            break;
        };
    }
    const eps = mlx.mlx_array_new_float(hc_epsilon);
    defer _ = mlx.mlx_array_free(eps);
    const norm_eps = mlx.mlx_array_new_float(norm_epsilon);
    defer _ = mlx.mlx_array_free(norm_eps);
    const iv = mlx.mlx_vector_array_new_data(&.{ x, mixes, scale, base, weight, eps, norm_eps }, 7);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, config, s));
    var out = Output{ .mixed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .normalized = mlx.mlx_array_new() };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.mixed, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
    try mlx.check(mlx.mlx_vector_array_get(&out.normalized, ov, 3));
    dispatch_count += 1;
    return out;
}
fn exact(a: Arr, b: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    const count = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .float32) try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..count]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..count])) else try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..count], mlx.mlx_array_data_bfloat16(b).?[0..count]);
}
const Fixture = struct {
    x: Arr,
    mixes: Arr,
    scale: Arr,
    base: Arr,
    weight: Arr,
    fn init(rows: c_int, magnitude: f32) !Fixture {
        const a = std.testing.allocator;
        var random = std.Random.DefaultPrng.init(765743);
        const rnd = random.random();
        const x = try a.alloc(u16, @intCast(rows * 4 * 4096));
        defer a.free(x);
        const mixes = try a.alloc(f32, @intCast(rows * 24));
        defer a.free(mixes);
        var weight: [4096]u16 = undefined;
        var base: [24]f32 = undefined;
        for (x) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) - 0.5) * magnitude)) >> 16);
        for (mixes) |*v| v.* = (rnd.float(f32) - 0.5) * 4;
        for (&base) |*v| v.* = (rnd.float(f32) - 0.5) * 2;
        for (&weight) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) + 0.5)) >> 16);
        const scales = [_]f32{ 0.37, 1.2, -0.9 };
        return .{ .x = mlx.mlx_array_new_data(x.ptr, &.{ 1, rows, 4, 4096 }, 4, .bfloat16), .mixes = mlx.mlx_array_new_data(mixes.ptr, &.{ 1, rows, 24 }, 3, .float32), .scale = mlx.mlx_array_new_data(&scales, &.{3}, 1, .float32), .base = mlx.mlx_array_new_data(&base, &.{24}, 1, .float32), .weight = mlx.mlx_array_new_data(&weight, &.{4096}, 1, .bfloat16) };
    }
    fn deinit(self: Fixture) void {
        for ([_]Arr{ self.x, self.mixes, self.scale, self.base, self.weight }) |v| {
            _ = mlx.mlx_array_free(v);
        }
    }
};
test "GLM HC collapse and norm preserve every output bit" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for ([_]c_int{ 1, 3, 4 }) |rows| for ([_]f32{ 0, 0.0001, 1, 128 }) |magnitude| for ([_]f32{ 1e-6, 1e-5, 0.1 }) |epsilon| {
        const f = try Fixture.init(rows, magnitude);
        defer f.deinit();
        const expected = try primitive.hcCollapse(f.x, f.mixes, f.scale, f.base, 20, 1e-6, s);
        defer expected.deinit();
        var normalized = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normalized);
        try mlx.check(mlx.mlx_fast_rms_norm(&normalized, expected.mixed, f.weight, epsilon, s));
        const got = (try apply(s, f.x, f.mixes, f.scale, f.base, f.weight, 20, 1e-6, epsilon)) orelse return error.TestExpectedHcNorm;
        defer got.deinit();
        try exact(expected.mixed, got.mixed);
        try exact(expected.post, got.post);
        try exact(expected.comb, got.comb);
        try exact(normalized, got.normalized);
    };
}

test "GLM HC collapse norm declines unsupported inputs" {
    const s = mlx.gpuStream();
    const f = try Fixture.init(3, 1);
    defer f.deinit();
    const before = dispatchCount();
    const absent = Arr{ .ctx = null };
    try std.testing.expect((try apply(s, f.x, f.mixes, f.scale, f.base, absent, 20, 1e-6, 1e-5)) == null);
    try std.testing.expect((try apply(s, f.x, f.mixes, f.scale, f.base, f.weight, 0, 1e-6, 1e-5)) == null);
    try std.testing.expect((try apply(s, f.x, f.mixes, f.scale, f.base, f.weight, 20, 0, 1e-5)) == null);
    try std.testing.expect((try apply(s, f.x, f.mixes, f.scale, f.base, f.weight, 20, 1e-6, std.math.nan(f32))) == null);
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, f.weight, .float32, s));
    try std.testing.expect((try apply(s, f.x, f.mixes, f.scale, f.base, wide, 20, 1e-6, 1e-5)) == null);
    var xwide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xwide);
    try mlx.check(mlx.mlx_astype(&xwide, f.x, .float32, s));
    try std.testing.expect((try apply(s, xwide, f.mixes, f.scale, f.base, f.weight, 20, 1e-6, 1e-5)) == null);
    try std.testing.expectEqual(before, dispatchCount());
}

fn chain(f: Fixture, fused: bool, s: mlx.mlx_stream) !Output {
    if (fused) return (try apply(s, f.x, f.mixes, f.scale, f.base, f.weight, 20, 1e-6, 1e-5)) orelse return error.TestExpectedHcNorm;
    const collapse = try primitive.hcCollapse(f.x, f.mixes, f.scale, f.base, 20, 1e-6, s);
    errdefer collapse.deinit();
    var normalized = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_fast_rms_norm(&normalized, collapse.mixed, f.weight, 1e-5, s));
    return .{ .mixed = collapse.mixed, .post = collapse.post, .comb = collapse.comb, .normalized = normalized };
}
fn timed(fixtures: []const Fixture, fused: bool, s: mlx.mlx_stream) !u64 {
    var timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    {
        var results: [16]Output = undefined;
        var made: usize = 0;
        defer for (results[0..made]) |v| v.deinit();
        const iv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(iv);
        for (fixtures, 0..) |f, i| {
            results[i] = try chain(f, fused, s);
            made += 1;
            for ([_]Arr{ results[i].mixed, results[i].post, results[i].comb, results[i].normalized }) |v| try mlx.check(mlx.mlx_vector_array_append_value(iv, v));
        }
        try mlx.check(mlx.mlx_eval(iv));
    }
    return timer.read();
}
test "GLM HC collapse norm queued component timing" {
    const path = std.c.getenv("SUSHI_GLM_HC_NORM_BENCH_OUT") orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    var samples: [2][2][2][31]u64 = undefined;
    for ([_]c_int{ 3, 4 }, 0..) |rows, geometry| {
        var fixtures: [16]Fixture = undefined;
        var made: usize = 0;
        defer for (fixtures[0..made]) |f| f.deinit();
        for (&fixtures) |*f| {
            f.* = try Fixture.init(rows, 1);
            made += 1;
        }
        const iv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(iv);
        for (fixtures) |f| for ([_]Arr{ f.x, f.mixes, f.scale, f.base, f.weight }) |v| try mlx.check(mlx.mlx_vector_array_append_value(iv, v));
        try mlx.check(mlx.mlx_eval(iv));
        for ([_]usize{ 1, 16 }, 0..) |batch, batch_index| {
            for (0..8) |_| for ([_]bool{ false, true }) |fused| {
                _ = try timed(fixtures[0..batch], fused, s);
            };
            for (0..31) |round| for (0..2) |position| {
                const arm = if (round % 2 == 0) position else 1 - position;
                samples[geometry][batch_index][arm][round] = try timed(fixtures[0..batch], arm == 1, s);
            };
        }
    }
    const raw = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .samples_ns = samples, .rows = [_]c_int{ 3, 4 }, .queued_calls = [_]usize{ 1, 16 }, .warmups_per_arm = 8, .pairs = 31, .method = "single-process alternating forward/reverse; build/eval/free; materialized distinct buffers; all four outputs evaluated" }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(raw);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = raw });
}
