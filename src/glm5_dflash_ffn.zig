//! Exact decode-width routed experts plus reusable affine shared/dense projections.
const std = @import("std");
const mlx = @import("mlx.zig");
const api = @import("sushi_exl3");
const base = @import("glm5_model.zig");
const forward = @import("glm5_forward.zig");
const rows = @import("glm5_dflash_kda.zig");
const Arr = mlx.mlx_array;
const Ops = base.Ops;

var routed_batches: usize = 0;
pub fn batchCount() usize {
    return routed_batches;
}
pub fn resetBatchCount() void {
    routed_batches = 0;
}

fn dense(linear: base.DenseMlp, ops: *Ops, x: Arr, limit: f32) !Arr {
    const gate = try rows.linearRows(ops, linear.gate, x, .affine_rows);
    const up = try rows.linearRows(ops, linear.up, x, .affine_rows);
    const activated = if (try @import("glm5_activation.zig").apply(ops.s, gate, up, limit)) |value| try ops.own(value) else blk: {
        const hi = try ops.scalar(limit, mlx.mlx_array_dtype(gate));
        const lo = try ops.scalar(-limit, mlx.mlx_array_dtype(up));
        const cg = try ops.binary(.min, gate, hi);
        const cu = try ops.binary(.max, try ops.binary(.min, up, hi), lo);
        break :blk try ops.binary(.mul, try ops.silu(cg), cu);
    };
    return rows.linearRows(ops, linear.down, activated, .affine_rows);
}

pub fn apply(target: *const forward.Model, index: usize, ops: *Ops, x: Arr) !Arr {
    if (index >= target.layers.len) return error.InvalidGlmLayer;
    const shape = mlx.getShape(x);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > 16 or shape[2] != target.cfg.hidden_size) return error.InvalidGlmDraftShape;
    if (shape[1] == 1) {
        switch (target.layers[index].ffn) {
            .moe => @import("glm5_dflash_profile.zig").skipSingleRoute(),
            .dense => {},
        }
        return target.feedForwardLayer(index, ops, x);
    }
    return switch (target.layers[index].ffn) {
        .dense => |layer| dense(layer, ops, x, target.cfg.glm_swiglu_limit),
        .moe => |layer| blk: {
            var timer = @import("glm5_dflash_profile.zig").Timer.start(@intCast(shape[1]));
            const routing: forward.Routed = if (try @import("glm5_router.zig").routeBatch(ops.s, x, layer.weight, layer.correction, @intCast(target.cfg.num_experts_per_tok), target.cfg.router_scaling_factor, target.cfg.moe_route_norm)) |candidate| blk_route: {
                const ids = ops.own(candidate.indices) catch |err| {
                    _ = mlx.mlx_array_free(candidate.scores);
                    return err;
                };
                break :blk_route .{ .indices = ids, .scores = try ops.own(candidate.scores) };
            } else blk_route: {
                var ids: [16]Arr = undefined;
                var scores: [16]Arr = undefined;
                for (0..@intCast(shape[1])) |row| {
                    var one = Ops{ .s = ops.s };
                    defer one.deinit();
                    const route = try target.routeLayer(index, &one, try one.slice(x, 1, @intCast(row), @intCast(row + 1)));
                    ids[row] = try ops.own(try one.result(route.indices));
                    scores[row] = try ops.own(try one.result(route.scores));
                }
                const count: usize = @intCast(shape[1]);
                break :blk_route .{ .indices = try ops.concat(ids[0..count], 1), .scores = try ops.concat(scores[0..count], 1) };
            };
            try @import("glm5_dflash_profile.zig").captureRoutes(ops.s, index, @intCast(shape[1]), target.cfg.num_experts_per_tok, @intCast(mlx.getShape(layer.weight)[0]), routing.indices);
            try timer.finish("ffn_router", &.{ routing.indices, routing.scores });
            const routed = try ops.own(try api.moeClamped(ops.s, x, layer.bank, routing.indices, routing.scores, .{ .codebook = target.cfg.expert_quant_codebook, .window = target.cfg.expert_quant_window }, @intFromFloat(target.cfg.glm_swiglu_limit)));
            try timer.finish("ffn_routed", &.{routed});
            routed_batches += 1;
            if (layer.shared) |shared| {
                const shared_output = try dense(shared, ops, x, target.cfg.glm_swiglu_limit);
                try timer.finish("ffn_shared", &.{shared_output});
                const combined = try ops.binary(.add, routed, shared_output);
                try timer.finish("ffn_combine", &.{combined});
                break :blk combined;
            }
            break :blk routed;
        },
    };
}

fn equal(a: Arr, b: Arr, stream: mlx.mlx_stream) !void {
    _ = stream;
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    const count = mlx.mlx_array_size(a);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..count], mlx.mlx_array_data_bfloat16(b).?[0..count]);
}

fn bank(ops: *Ops, hidden: c_int, intermediate: c_int, rate: c_int) !api.Bank {
    var projections: [3]api.Proj = undefined;
    for (&projections, 0..) |*projection, i| {
        const input = if (i == 2) intermediate else hidden;
        const output = if (i == 2) hidden else intermediate;
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(123 + i)));
        const codes = try ops.slot();
        try mlx.check(mlx.mlx_random_bits(codes, &[_]c_int{ 9, @divExact(input, 16), @divExact(output, 16), rate }, 4, 2, key.*, ops.s));
        const su = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(su, try ops.scalar(-0.2, .float16), try ops.scalar(0.2, .float16), &[_]c_int{ 9, input }, 2, .float16, key.*, ops.s));
        const sv = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(sv, try ops.scalar(-0.05, .float16), try ops.scalar(0.05, .float16), &[_]c_int{ 9, output }, 2, .float16, key.*, ops.s));
        projection.* = .{ .trellis = codes.*, .suh = su.*, .svh = sv.* };
    }
    return .{ .gate = projections[0], .up = projections[1], .down = projections[2] };
}

test "GLM DFlash clamped experts keep serial row bits at all supported rates" {
    const s = mlx.gpuStream();
    for (0..18) |case| {
        const production = case == 17;
        const hidden: c_int = if (production) 4096 else 128;
        const intermediate: c_int = if (production) 2048 else 128;
        const rate: c_int = if (production) 36 else @intCast(32 + case * 2);
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const weights = try bank(&ops, hidden, intermediate, rate);
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 117));
        const input = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(input, &[_]c_int{ 1, 16, hidden }, 3, .bfloat16, 0, 2, key.*, s));
        var id_data: [16 * 8]u32 = undefined;
        var score_data: [16 * 8]f32 = undefined;
        for (&id_data, &score_data, 0..) |*id, *score, i| {
            id.* = @intCast((i / 8 + i % 8) % 9);
            score.* = @as(f32, @floatFromInt(i % 8 + 1)) / 36 * 2.5;
        }
        const ids = try ops.own(mlx.mlx_array_new_data(&id_data, &[_]c_int{ 1, 16, 8 }, 3, .uint32));
        const scores = try ops.own(mlx.mlx_array_new_data(&score_data, &[_]c_int{ 1, 16, 8 }, 3, .float32));
        for ([_]c_int{ 2, 4, 8, 16 }) |count| {
            var scope = Ops{ .s = s };
            defer scope.deinit();
            const x = try scope.slice(input.*, 1, 0, count);
            const ii = try scope.slice(ids, 1, 0, count);
            const ss = try scope.slice(scores, 1, 0, count);
            const got = try scope.own(try api.moeClamped(s, x, weights, ii, ss, .{ .codebook = .mcg, .window = .w12 }, 10));
            for (0..@intCast(count)) |row| {
                var one = Ops{ .s = s };
                defer one.deinit();
                const from: c_int = @intCast(row);
                const want = try one.own(try api.moeClamped(s, try one.slice(x, 1, from, from + 1), weights, try one.slice(ii, 1, from, from + 1), try one.slice(ss, 1, from, from + 1), .{ .codebook = .mcg, .window = .w12 }, 10));
                try equal(want, try one.slice(got, 1, from, from + 1), s);
            }
        }
    }
}

test "GLM DFlash FFN batch matches each target FFN row" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iterator = weights.map.iterator();
    var seed: usize = 133;
    while (iterator.next()) |entry| {
        if (std.mem.indexOf(u8, entry.key_ptr.*, ".mlp.") == null) continue;
        const value = entry.value_ptr;
        const shape = mlx.getShape(value.*);
        const dtype = mlx.mlx_array_dtype(value.*);
        if (shape.len < 2 or dtype != .bfloat16) continue;
        const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(shape, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 4, 128 }, 731, s));
    for ([_]usize{ 0, 3 }) |layer| {
        const got = try apply(&target, layer, &ops, x);
        for (0..4) |row| {
            var one = Ops{ .s = s };
            defer one.deinit();
            const from: c_int = @intCast(row);
            const expected = try target.feedForwardLayer(layer, &one, try one.slice(x, 1, from, from + 1));
            try equal(expected, try one.slice(got, 1, from, from + 1), s);
        }
    }
}

test "GLM DFlash FFN integrates production-width batched routing" {
    const allocator = std.testing.allocator;
    const stream = mlx.gpuStream();
    var fixtures = Ops{ .s = stream };
    defer fixtures.deinit();
    var weights = @import("model.zig").Weights.init(allocator);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var target = try forward.Model.load(allocator, cfg, &weights, stream);
    defer target.deinit();
    target.cfg.hidden_size = 4096;
    target.cfg.num_experts_per_tok = 8;
    const w = try fixtures.zeros(&.{ 288, 4096 }, .bfloat16);
    var correction: [288]f32 = @splat(-10);
    for (correction[0..8]) |*value| value.* = 1;
    const bias = try fixtures.own(mlx.mlx_array_new_data(&correction, &.{288}, 1, .float32));
    target.layers[3].ffn.moe.weight = w;
    target.layers[3].ffn.moe.correction = bias;
    target.layers[3].ffn.moe.bank = try bank(&fixtures, 4096, 128, 36);
    target.layers[3].ffn.moe.shared = null;
    try mlx.check(mlx.mlx_array_eval(w));
    const x = try fixtures.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 4, 4096 }, 731, stream));
    var ops = Ops{ .s = stream };
    defer ops.deinit();
    @import("glm5_router.zig").resetBatchCallCount();
    const actual = try apply(&target, 3, &ops, x);
    try std.testing.expectEqual(@as(usize, 1), @import("glm5_router.zig").batchCallCount());
    for (0..4) |i| {
        var one = Ops{ .s = stream };
        defer one.deinit();
        const row: c_int = @intCast(i);
        const expected = try target.feedForwardLayer(3, &one, try one.slice(x, 1, row, row + 1));
        try equal(expected, try one.slice(actual, 1, row, row + 1), stream);
    }
}
