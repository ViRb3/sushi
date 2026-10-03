//! Production retained weights, synthetic BF16 input; no runtime integration.
const std = @import("std");
const mlx = @import("mlx.zig");
const native = @import("glm5_model.zig");
const cluster = @import("glm5_kda_prefill_cluster.zig");
const Ops = native.Ops;
const Arr = mlx.mlx_array;

fn eval(values: []const Arr) !void {
    const list = mlx.mlx_vector_array_new_data(values.ptr, values.len);
    defer _ = mlx.mlx_vector_array_free(list);
    try mlx.check(mlx.mlx_eval(list));
}
fn products(ops: *Ops, bank: [5]native.Linear, cluster_weight: Arr, x: Arr, candidate: bool, downstream: bool) ![3]Arr {
    var y: [3]Arr = undefined;
    if (candidate) {
        y = try cluster.project(ops, cluster_weight, x);
    } else {
        for (0..3) |i| y[i] = try bank[i].apply(ops, x);
    }
    if (downstream) {
        y[0] = try bank[3].apply(ops, y[0]);
        y[1] = try bank[4].apply(ops, y[1]);
    }
    return y;
}
fn timed(bank: [5]native.Linear, cluster_weight: Arr, x: Arr, candidate: bool, downstream: bool) !u64 {
    const clock = @import("io_util.zig").Stopwatch.init(std.testing.io);
    {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const outputs = try products(&ops, bank, cluster_weight, x, candidate, downstream);
        try eval(&outputs);
    } // Fresh graphs, output compaction, evaluation and teardown are included.
    return clock.read();
}
const Drift = struct { count: usize, mismatches: usize, relative_l2: f64, max_abs: f64 };
fn compare(a: Arr, b: Arr) !Drift {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    const n = mlx.mlx_array_size(a);
    const av = mlx.mlx_array_data_bfloat16(a).?[0..n];
    const bv = mlx.mlx_array_data_bfloat16(b).?[0..n];
    var squared: f64 = 0;
    var error_squared: f64 = 0;
    var max_abs: f64 = 0;
    var unequal: usize = 0;
    for (av, bv) |ab, bb| {
        const left: f64 = @as(f32, @bitCast(@as(u32, ab) << 16));
        const right: f64 = @as(f32, @bitCast(@as(u32, bb) << 16));
        try std.testing.expect(std.math.isFinite(left) and std.math.isFinite(right));
        squared += left * left;
        error_squared += (left - right) * (left - right);
        max_abs = @max(max_abs, @abs(left - right));
        unequal += @intFromBool(ab != bb);
    }
    return .{ .count = n, .mismatches = unequal, .relative_l2 = @sqrt(error_squared / @max(squared, 1e-30)), .max_abs = max_abs };
}

test "GLM prefill cluster production qualification" {
    const path = std.c.getenv("SUSHI_GLM_CLUSTER_SHARD") orelse return error.SkipZigTest;
    const output = std.c.getenv("SUSHI_GLM_CLUSTER_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const s = mlx.gpuStream();
    var setup = Ops{ .s = s };
    defer setup.deinit();
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, path, cpu));
    var bank: [5]native.Linear = undefined;
    for ([_][*:0]const u8{ "model.language_model.layers.0.self_attn.f_a_proj.weight", "model.language_model.layers.0.self_attn.g_a_proj.weight", "model.language_model.layers.0.self_attn.b_proj.weight", "model.language_model.layers.0.self_attn.f_b_proj.weight", "model.language_model.layers.0.self_attn.g_b_proj.weight" }, 0..) |name, i| {
        const slot = try setup.slot();
        try mlx.check(mlx.mlx_map_string_to_array_get(slot, arrays, name));
        const shape = mlx.getShape(slot.*);
        try std.testing.expectEqual(@as(usize, 2), shape.len);
        try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(slot.*));
        bank[i] = .{ .w = slot.*, .input = shape[1], .output = shape[0] };
        try mlx.check(mlx.mlx_array_eval(slot.*));
    }
    const pack_clock = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const cluster_weight = try cluster.prepare(&setup, bank[0..3].*);
    try mlx.check(mlx.mlx_array_eval(cluster_weight));
    const pack_ns = pack_clock.read();
    const key = try setup.slot();
    try mlx.check(mlx.mlx_random_key(key, 0x3202048));
    const x = try setup.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ 1, 2048, 4096 }, 3, .bfloat16, 0, 1, key.*, s));
    try mlx.check(mlx.mlx_array_eval(x.*));
    // Exercise the exact first-stage caller helper and owned preparation, without
    // another complete recurrence or timing sweep. Other KDA fields are unused.
    if (std.c.getenv("SUSHI_GLM_CLUSTER_CALLER_ONLY") != null) {
        var layer: native.KdaLayer = undefined;
        layer.fa = bank[0];
        layer.ga = bank[1];
        layer.beta = bank[2];
        layer.prepared_conv = .{ .ctx = null };
        layer.prepared_decay = .{ .ctx = null };
        layer.prepared_cluster = .{ .ctx = null };
        defer layer.deinit();
        const off = cluster.bind(false);
        try layer.preparePrefillCluster(s);
        try std.testing.expect(layer.prepared_cluster.ctx == null);
        try std.testing.expectEqual(@as(usize, 0), try cluster.transientBudget(2048, 2));
        off.restore();
        const on = cluster.bind(true);
        defer on.restore();
        try layer.preparePrefillCluster(s);
        const owned = layer.prepared_cluster.ctx;
        try std.testing.expect(owned != null);
        try std.testing.expectEqual(cluster.weight_bytes, mlx.mlx_array_size(layer.prepared_cluster) * mlx.mlx_array_itemsize(layer.prepared_cluster));
        var net: @import("glm5_forward.zig").Model = undefined;
        var layers: [1]std.meta.Child(@TypeOf(net.layers)) = undefined;
        layers[0].attn = .{ .kda = layer };
        net.layers = &layers;
        try std.testing.expectEqual(cluster.weight_bytes, net.kdaPrefillClusterBytes());
        try layer.preparePrefillCluster(s);
        try std.testing.expectEqual(owned, layer.prepared_cluster.ctx);
        try std.testing.expectEqual(2 * cluster.transient_bytes, try cluster.transientBudget(2048, 2));
        try std.testing.expectEqual(@as(usize, 0), try cluster.transientBudget(2047, 2));
        var scope = Ops{ .s = s };
        defer scope.deinit();
        cluster.resetDispatchCount();
        try std.testing.expect((try layer.prefillCluster(&scope, try scope.slice(x.*, 1, 0, 3))) == null);
        try std.testing.expectEqual(@as(usize, 0), cluster.dispatchCount());
        const actual = (try layer.prefillCluster(&scope, x.*)) orelse return error.TestExpectedClusterCaller;
        const expected = try products(&scope, bank, cluster_weight, x.*, false, false);
        try eval(&actual);
        try eval(&expected);
        for (expected, actual) |a, b| try std.testing.expectEqual(@as(usize, 0), (try compare(a, b)).mismatches);
        try std.testing.expectEqual(@as(usize, 1), cluster.dispatchCount());
        const disable = cluster.bind(false);
        try std.testing.expect((try layer.prefillCluster(&scope, x.*)) == null);
        disable.restore();
        layer.deinit();
        try std.testing.expect(layer.prepared_cluster.ctx == null);
        layers[0].attn = .{ .kda = layer };
        try std.testing.expectEqual(@as(usize, 0), net.kdaPrefillClusterBytes());
        const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .caller = "KdaLayer.preparePrefillCluster/prefillCluster/deinit", .raw_bf16_equal = true, .prepared_weight_bytes = cluster.weight_bytes, .two_pending_transient_bytes = 2 * cluster.transient_bytes, .unsupported_t3_fallback = true, .disabled_fallback = true, .ownership_idempotent = true, .timing_repeated = false, .full_model = false }, .{ .whitespace = .indent_2 });
        defer std.testing.allocator.free(json);
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
        return;
    }
    var drifts: [2][3]Drift = undefined;
    for ([_]bool{ false, true }, 0..) |downstream, mode| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const original = try products(&ops, bank, cluster_weight, x.*, false, downstream);
        const candidate = try products(&ops, bank, cluster_weight, x.*, true, downstream);
        try eval(&original);
        try eval(&candidate);
        for (original, candidate, 0..) |a, b, i| {
            drifts[mode][i] = try compare(a, b);
            try std.testing.expect(drifts[mode][i].relative_l2 <= 0.003);
        }
    }
    // All three boundaries are identifiable in production outputs; shape and
    // element comparisons include every row, including the beta ragged tile.
    var samples: [2][21][2]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |downstream, mode| {
        for (0..5) |_| {
            for ([_]bool{ false, true }) |arm| _ = try timed(bank, cluster_weight, x.*, arm, downstream);
        }
        for (0..21) |round| for (0..2) |position| {
            const arm = if (round % 2 == 0) position else 1 - position;
            samples[mode][round][arm] = try timed(bank, cluster_weight, x.*, arm == 1, downstream);
        };
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .nanoseconds = samples, .modes = .{ "three first-stage projections", "five projection chain" }, .arms = .{ "three original BF16 calls", "prepared BF16 cluster320 plus compact slices" }, .drifts = drifts, .prepared_weight_bytes = cluster.weight_bytes, .one_time_pack_ns = pack_ns, .input_rows = 2048, .warmup_pairs = 5, .sample_pairs = 21, .method = "layer0 original retained BF16 weights; materialized normal BF16 x; FP32 native NAX accumulation, BF16 outputs; fresh graphs/compaction/eval/free included; prepared weight concatenation excluded and separately measured; AB/BA pairs", .runtime_integration = false, .full_model = false }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
