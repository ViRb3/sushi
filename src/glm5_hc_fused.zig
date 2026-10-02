//! Experimental one-token GLM HC normalization/mix; integration requires parity qualification.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
var calls: usize = 0;
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}

// RMS reduction mapping follows MLX rms_looped (MIT) and oMLX hc_mix1 (Apache-2.0).
// The projection keeps Sushi's four-SIMD-group order, not oMLX's eight-group GEMV.
const SOURCE: [:0]const u8 =
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    \\const uint output=threadgroup_position_in_grid.x;
    \\threadgroup float sums[32];
    \\threadgroup float inverse[1];
    \\for(uint virtual_group=0;virtual_group<8;++virtual_group) {
    \\  const uint logical=virtual_group*128u+tid;
    \\  float acc=0.0f;
    \\  for(uint r=0;r<16384u;r+=4096u) {
    \\    for(uint i=0;i<4u;++i) {float xi=float(x[r+logical*4u+i]);acc+=xi*xi;}
    \\  }
    \\  acc=simd_sum(acc);
    \\  if(lane==0) sums[virtual_group*4u+sg]=acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(sg==0) {float acc=simd_sum(sums[lane]);if(lane==0) inverse[0]=metal::precise::rsqrt(acc/16384.0f+float(eps));}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float value=0.0f;
    \\for(uint d=tid;d<16384u;d+=128u) {
    \\  volatile float normalized=float(x[d])*inverse[0];
    \\  value+=normalized*float(w[output*16384u+d]);
    \\}
    \\value=simd_sum(value);
    \\if(lane==0) sums[sg]=value;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(tid==0) out[output]=(sums[0]+sums[1])+(sums[2]+sums[3]);
;
const REFERENCE: [:0]const u8 =
    \\const uint output=threadgroup_position_in_grid.x;
    \\const uint lane=thread_position_in_threadgroup.x;
    \\threadgroup float partial[4];
    \\float value=0.0f;
    \\for(uint d=lane;d<16384u;d+=128u) value+=x[d]*float(w[output*16384u+d]);
    \\value=simd_sum(value);
    \\if(thread_index_in_simdgroup==0) partial[simdgroup_index_in_threadgroup]=value;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(lane==0) out[output]=(partial[0]+partial[1])+(partial[2]+partial[3]);
;
var fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reference_kernel: ?mlx.mlx_fast_metal_kernel = null;
var config: ?mlx.mlx_fast_metal_kernel_config = null;
fn launch(s: mlx.mlx_stream, x: Arr, w: Arr, epsilon: f32, fused: bool) !Arr {
    const slot = if (fused) &fused_kernel else &reference_kernel;
    if (slot.* == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "x", "w", "eps" }, 3);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new(if (fused) "sushi_glm_hc_norm_mix" else "sushi_glm_hc_mix_reference", ins, outs, if (fused) SOURCE else REFERENCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = k;
    }
    if (config == null) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, 1, 24 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 24 * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        config = c;
    }
    const eps = mlx.mlx_array_new_float(epsilon);
    defer _ = mlx.mlx_array_free(eps);
    const inputs = mlx.mlx_vector_array_new_data(&.{ x, w, eps }, 3);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, slot.*.?, inputs, config.?, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    return out;
}
pub fn mix(s: mlx.mlx_stream, x: Arr, w: Arr, epsilon: f32) !?Arr {
    if (!mlx.streamIsGpu(s) or x.ctx == null or w.ctx == null or !std.math.isFinite(epsilon) or epsilon <= 0) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or (mlx.mlx_array_dtype(w) != .bfloat16 and mlx.mlx_array_dtype(w) != .float32) or
        !std.mem.eql(c_int, &.{ 1, 1, 4, 4096 }, mlx.getShape(x)) or !std.mem.eql(c_int, &.{ 24, 16384 }, mlx.getShape(w))) return null;
    const out = try launch(s, x, w, epsilon, true);
    calls += 1;
    return out;
}
fn reference(s: mlx.mlx_stream, x: Arr, w: Arr, epsilon: f32) !Arr {
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_astype(&wide, x, .float32, s));
    try mlx.check(mlx.mlx_reshape(&flat, wide, &.{ 1, 1, 16384 }, 3, s));
    try mlx.check(mlx.mlx_fast_rms_norm(&norm, flat, .{ .ctx = null }, epsilon, s));
    return launch(s, norm, w, epsilon, false);
}
fn expectBits(a: Arr, b: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    const n = mlx.mlx_array_size(a);
    try std.testing.expectEqual(n, mlx.mlx_array_size(b));
    if (mlx.mlx_array_dtype(a) == .float32) try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n])) else try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
}

test "GLM HC fused normalization mix matches exact staged geometry" {
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var random = std.Random.DefaultPrng.init(919243);
    const rnd = random.random();
    const wb = try a.alloc(u16, 24 * 16384);
    defer a.free(wb);
    for (wb) |*v| {
        const f = (rnd.float(f32) - 0.5) / 32;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    const w = mlx.mlx_array_new_data(wb.ptr, &[_]c_int{ 24, 16384 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    var xbits: [16384]u16 = undefined;
    for ([_]f32{ 0, 0.0001, 1, 128 }) |magnitude| {
        for (&xbits) |*v| {
            const f = (rnd.float(f32) - 0.5) * magnitude;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const x = mlx.mlx_array_new_data(&xbits, &[_]c_int{ 1, 1, 4, 4096 }, 4, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        for ([_]f32{ 1e-6, 1e-5, 0.1 }) |eps| {
            const old = try reference(s, x, w, eps);
            defer _ = mlx.mlx_array_free(old);
            const got = (try mix(s, x, w, eps)) orelse return error.TestExpectedFusedHc;
            defer _ = mlx.mlx_array_free(got);
            try expectBits(old, got);
        }
    }
}

const Fixture = struct {
    x: Arr,
    w: Arr,
    scale: Arr,
    base: Arr,
    fn deinit(self: Fixture) void {
        for ([_]Arr{ self.x, self.w, self.scale, self.base }) |v| {
            _ = mlx.mlx_array_free(v);
        }
    }
    fn init() !Fixture {
        const a = std.testing.allocator;
        const wb = try a.alloc(u16, 24 * 16384);
        defer a.free(wb);
        var random = std.Random.DefaultPrng.init(315487);
        const rnd = random.random();
        for (wb) |*v| {
            const f = (rnd.float(f32) - 0.5) / 32;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        var xb: [16384]u16 = undefined;
        for (&xb) |*v| {
            const f = (rnd.float(f32) - 0.5) * 3;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        var bases: [24]f32 = undefined;
        for (&bases, 0..) |*v, i| v.* = ([_]f32{ 0.1, -0.2, 0.3, -0.4 })[i % 4];
        return .{
            .x = mlx.mlx_array_new_data(&xb, &[_]c_int{ 1, 1, 4, 4096 }, 4, .bfloat16),
            .w = mlx.mlx_array_new_data(wb.ptr, &[_]c_int{ 24, 16384 }, 2, .bfloat16),
            .scale = mlx.mlx_array_new_data(&[_]f32{ 0.125, -0.25, 0.0625 }, &[_]c_int{3}, 1, .float32),
            .base = mlx.mlx_array_new_data(&bases, &[_]c_int{24}, 1, .float32),
        };
    }
};
fn collapse(ops: *@import("glm5_model.zig").Ops, f: Fixture, fused: bool) !@import("glm5_next.zig").HcResult {
    if (fused) {
        const mixes = try ops.own((try mix(ops.s, f.x, f.w, 1e-5)) orelse return error.TestExpectedFusedHc);
        return @import("glm5_next.zig").hcCollapse(f.x, mixes, f.scale, f.base, 20, 1e-6, ops.s);
    }
    const hc = @import("glm5_model.zig").Hc{ .w = f.w, .scale = f.scale, .base = f.base };
    const cfg = @import("model.zig").ModelConfig{ .rms_norm_eps = 1e-5, .glm_hc_sinkhorn_iters = 20, .glm_hc_eps = 1e-6 };
    if (@hasDecl(@import("glm5_model.zig").Hc, "collapseReference")) return hc.collapseReference(ops, f.x, &cfg);
    return hc.collapse(ops, f.x, &cfg);
}
test "GLM HC fused complete collapse preserves mixed post and comb bits" {
    const f = try Fixture.init();
    defer f.deinit();
    var ops = @import("glm5_model.zig").Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const before = dispatchCount();
    const old = try collapse(&ops, f, false);
    defer old.deinit();
    try std.testing.expectEqual(before, dispatchCount());
    const got = try collapse(&ops, f, true);
    defer got.deinit();
    try expectBits(old.mixed, got.mixed);
    try expectBits(old.post, got.post);
    try expectBits(old.comb, got.comb);
    const wf = try ops.cast(f.w, .float32);
    const mix_old = try ops.own(try reference(ops.s, f.x, wf, 1e-5));
    const mix_new = try ops.own((try mix(ops.s, f.x, wf, 1e-5)).?);
    try expectBits(mix_old, mix_new);
    try std.testing.expect((try mix(ops.s, f.x, f.w, -1)) == null);
    try std.testing.expect((try mix(ops.s, try ops.reshape(f.x, &.{ 1, 4, 4096 }), f.w, 1e-5)) == null);
    try std.testing.expect((try mix(ops.s, try ops.cast(f.x, .float32), f.w, 1e-5)) == null);
}

fn timedCollapse(f: Fixture, fused: bool) !u64 {
    var ops = @import("glm5_model.zig").Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const out = try collapse(&ops, f, fused);
    defer out.deinit();
    const vec = mlx.mlx_vector_array_new_data(&.{ out.mixed, out.post, out.comb }, 3);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    return timer.read();
}
fn timedBatch(fixtures: []const Fixture, fused: bool) !u64 {
    const Ops = @import("glm5_model.zig").Ops;
    var scopes: [16]Ops = undefined;
    var scope_count: usize = 0;
    defer for (scopes[0..scope_count]) |*ops| ops.deinit();
    var results: [16]@import("glm5_next.zig").HcResult = undefined;
    var made: usize = 0;
    defer for (results[0..made]) |out| out.deinit();
    const outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    var timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    for (fixtures, 0..) |f, i| {
        scopes[i] = .{ .s = mlx.gpuStream() };
        scope_count += 1;
        results[i] = try collapse(&scopes[i], f, fused);
        made += 1;
        for ([_]Arr{ results[i].mixed, results[i].post, results[i].comb }) |out|
            try mlx.check(mlx.mlx_vector_array_append_value(outputs, out));
    }
    try mlx.check(mlx.mlx_eval(outputs));
    return timer.read();
}

test "GLM HC fused warmed microbenchmark" {
    const path = std.c.getenv("SUSHI_GLM_HC_BENCH_OUT") orelse return error.SkipZigTest;
    const f = try Fixture.init();
    defer f.deinit();
    for (0..16) |_| {
        _ = try timedCollapse(f, false);
        _ = try timedCollapse(f, true);
    }
    const count = 200;
    var reference_ns: [count]u64 = undefined;
    var fused_ns: [count]u64 = undefined;
    for (0..count) |i| {
        if (i % 2 == 0) {
            reference_ns[i] = try timedCollapse(f, false);
            fused_ns[i] = try timedCollapse(f, true);
        } else {
            fused_ns[i] = try timedCollapse(f, true);
            reference_ns[i] = try timedCollapse(f, false);
        }
    }
    var fixtures: [16]Fixture = undefined;
    var made: usize = 0;
    defer for (fixtures[0..made]) |value| value.deinit();
    for (&fixtures) |*value| {
        value.* = try Fixture.init();
        made += 1;
    }
    for (0..8) |_| {
        _ = try timedBatch(&fixtures, false);
        _ = try timedBatch(&fixtures, true);
    }
    var queued_reference_ns: [64]u64 = undefined;
    var queued_fused_ns: [64]u64 = undefined;
    for (0..64) |i| {
        if (i % 2 == 0) {
            queued_reference_ns[i] = try timedBatch(&fixtures, false);
            queued_fused_ns[i] = try timedBatch(&fixtures, true);
        } else {
            queued_fused_ns[i] = try timedBatch(&fixtures, true);
            queued_reference_ns[i] = try timedBatch(&fixtures, false);
        }
    }
    const raw = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .queued_reference_ns = queued_reference_ns,
        .queued_fused_ns = queued_fused_ns,
        .queued_batch_size = 16,
        .queued_warmup_pairs = 8,
        .queued_distinct_buffers = true,
        .reference_ns = reference_ns,
        .fused_ns = fused_ns,
        .warmup_pairs = 16,
        .timed_pairs = count,
        .geometry = "BF16[1,1,4,4096],BF16[24,16384]",
        .component = "full HC collapse: normalization, mix, Sinkhorn and stream collapse",
        .method = "alternating AB/BA, build plus synchronous evaluation, one process",
        .qos = "foreground taskpolicy -a",
        .full_model = false,
    }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(raw);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = raw });
}
