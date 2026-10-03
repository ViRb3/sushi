//! One-leaf FP32 recurrence retention qualification; no model load.
const std = @import("std");
const mlx = @import("mlx.zig");
const kda = @import("glm5_dflash_kda.zig");
const primitive = @import("glm5_next.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
test "GLM KDA leaf picks first ancestry terminal" {
    try std.testing.expectEqual(@as(u32, 2), kda.cachedLeafRow(&.{ -1, 0, 1 }));
    try std.testing.expectEqual(@as(u32, 1), kda.cachedLeafRow(&.{ -1, 0, 0 }));
}

test "GLM KDA leaf state and replay fallback preserve raw FP32 bits" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const input = try fixture(&ops, 2, 32, 16);
    for ([_][3]i32{ .{ -1, 0, 1 }, .{ -1, 0, 0 } }) |parents| {
        const retained = try kda.recurrentLeaf(input, &parents, s);
        defer retained.deinit();
        try exact(try ops.own(try kda.recurrent(input, &parents, s)), retained.y);
        var tape = try makeTape(&ops, input, &parents, retained);
        defer tape.deinit();
        kda.resetLeafStats();
        const leaf_path: []const u32 = if (parents[2] == 1) &.{ 0, 1, 2 } else &.{ 0, 1 };
        const leaf = try tape.replay(leaf_path, s);
        defer _ = mlx.mlx_array_free(leaf.conv_state);
        defer _ = mlx.mlx_array_free(leaf.ssm_state);
        try exact(leaf.ssm_state, try sequential(&ops, input, leaf_path));
        try std.testing.expectEqual(mlx.mlx_array_data_float32(retained.state), mlx.mlx_array_data_float32(leaf.ssm_state));
        const miss = try tape.replay(&.{0}, s);
        defer _ = mlx.mlx_array_free(miss.conv_state);
        defer _ = mlx.mlx_array_free(miss.ssm_state);
        try exact(miss.ssm_state, try sequential(&ops, input, &.{0}));
        try std.testing.expectEqual(@as(usize, 1), kda.leafHits());
        try std.testing.expectEqual(@as(usize, 1), kda.leafMisses());
    }
}

fn normal(ops: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64, deviation: f32) !Arr {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(out, shape.ptr, shape.len, dtype, 0, deviation, key.*, ops.s));
    return out.*;
}
fn fixture(ops: *Ops, h: c_int, dk: c_int, dv: c_int) !primitive.KdaInputs {
    return .{ .q = try normal(ops, &.{ 1, 3, h, dk }, .bfloat16, 31, 0.02), .k = try normal(ops, &.{ 1, 3, h, dk }, .bfloat16, 32, 0.02), .v = try normal(ops, &.{ 1, 3, h, dv }, .bfloat16, 33, 0.1), .decay = try ops.binary(.mul, try ops.ones(&.{ 1, 3, h, dk }, .float32), try ops.scalar(0.98, .float32)), .beta = try ops.binary(.mul, try ops.ones(&.{ 1, 3, h }, .bfloat16), try ops.scalar(0.5, .bfloat16)), .state = try normal(ops, &.{ 1, h, dv, dk }, .float32, 34, 0.05) };
}
fn exact(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .float32) try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n])) else try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
}
fn sequential(ops: *Ops, input: primitive.KdaInputs, path: []const u32) !Arr {
    const ids = try ops.own(mlx.mlx_array_new_data(path.ptr, &.{@intCast(path.len)}, 1, .uint32));
    const result = try primitive.kda(.{ .q = try ops.take(input.q, ids, 1), .k = try ops.take(input.k, ids, 1), .v = try ops.take(input.v, ids, 1), .decay = try ops.take(input.decay, ids, 1), .beta = try ops.take(input.beta, ids, 1), .state = input.state }, ops.s);
    defer result.deinit();
    return ops.own(try ops.result(result.state));
}
fn makeTape(ops: *Ops, input: primitive.KdaInputs, parents: []const i32, retained: ?kda.LeafResult) !kda.Tape {
    var tape = kda.Tape{ .inputs = .{ .q = .{ .ctx = null }, .k = .{ .ctx = null }, .v = .{ .ctx = null }, .decay = .{ .ctx = null }, .beta = .{ .ctx = null }, .state = .{ .ctx = null } }, .conv_input = .{ .ctx = null }, .count = parents.len };
    errdefer tape.deinit();
    inline for (.{ "q", "k", "v", "decay", "beta", "state" }) |name| @field(tape.inputs, name) = try ops.result(@field(input, name));
    const shape = mlx.getShape(input.q);
    tape.conv_input = try ops.result(try ops.zeros(&.{ 1, 6, shape[2] * shape[3] * 3 }, .bfloat16));
    @memcpy(tape.parents[0..parents.len], parents);
    if (retained) |value| {
        tape.retained_state = try ops.result(value.state);
        tape.retained_row = value.row;
    }
    return tape;
}
fn timed(input: primitive.KdaInputs, retain: bool, path: []const u32, s: mlx.mlx_stream) !u64 {
    const timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    var ops = Ops{ .s = s };
    errdefer ops.deinit();
    const parents = [_]i32{ -1, 0, 1 };
    const saved: ?kda.LeafResult = if (retain) try kda.recurrentLeaf(input, &parents, s) else null;
    defer if (saved) |value| value.deinit();
    const y = if (saved) |value| try ops.own(try ops.result(value.y)) else try ops.own(try kda.recurrent(input, &parents, s));
    var tape = try makeTape(&ops, input, &parents, saved);
    defer tape.deinit();
    const result = try tape.replay(path, s);
    defer _ = mlx.mlx_array_free(result.conv_state);
    defer _ = mlx.mlx_array_free(result.ssm_state);
    const ev = mlx.mlx_vector_array_new_data(&.{ y, result.conv_state, result.ssm_state }, 3);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    ops.deinit();
    return timer.read();
}

test "GLM KDA leaf production hit miss traffic and timing" {
    const out = std.c.getenv("SUSHI_GLM_KDA_LEAF_OUT") orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const input = try fixture(&ops, 64, 128, 128);
    const values = [_]Arr{ input.q, input.k, input.v, input.decay, input.beta, input.state };
    const ev = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    const saved = try kda.recurrentLeaf(input, &.{ -1, 0, 1 }, s);
    defer saved.deinit();
    try exact(saved.y, try ops.own(try kda.recurrent(input, &.{ -1, 0, 1 }, s)));
    var tape = try makeTape(&ops, input, &.{ -1, 0, 1 }, saved);
    defer tape.deinit();
    const committed = try tape.replay(&.{ 0, 1, 2 }, s);
    defer _ = mlx.mlx_array_free(committed.conv_state);
    defer _ = mlx.mlx_array_free(committed.ssm_state);
    try exact(committed.ssm_state, try sequential(&ops, input, &.{ 0, 1, 2 }));
    const samples = 8;
    var ns: [2][2][samples]u64 = undefined;
    for ([_][]const u32{ &.{ 0, 1, 2 }, &.{0} }, 0..) |path, endpoint| {
        for (0..2) |_| {
            _ = try timed(input, false, path, s);
            _ = try timed(input, true, path, s);
        }
        for (0..samples) |sample| for (0..2) |slot| {
            const retain = (sample + slot) % 2 == 1;
            ns[endpoint][@intFromBool(retain)][sample] = try timed(input, retain, path, s);
        };
    }
    const data = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .heads = 64, .keys = 128, .values = 128, .nodes = 3, .cached_row = 2, .cache_dtype = "FP32", .extra_state_bytes_per_layer = 4194304, .extra_state_bytes_34layers = 142606336, .bit_parity = true, .samples_ns = ns, .axes = "[hit endpoint2/miss endpoint0][baseline/retained][sample]", .timing = "tree verification + commit state and conv tail + evaluate; fresh scopes; two warmups; eight alternating pairs", .default = "off", .model_load = false }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(data);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(out), .data = data });
}
