//! Isolated B3 native attention versus current scalar split-eight cadence.
const std = @import("std");
const mlx = @import("mlx.zig");
const attention = @import("glm5_attention.zig");
const batch_attention = @import("glm5_attention_decode_batch.zig");
const overlay = @import("glm5_attention_overlay.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
const samples = 11;

const Fixture = struct {
    ops: Ops,
    q: Arr,
    iq: Arr,
    weights: Arr,
    prefix: Arr,
    tape: Arr,
    pooled: Arr,
    prefix_rows: usize = 16381,
    fn init(path: [*:0]const u8) !Fixture {
        var ops = Ops{ .s = mlx.gpuStream() };
        errdefer ops.deinit();
        var arrays = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(arrays);
        var metadata = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(metadata);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, path, cpu));
        var values: [5]Arr = undefined;
        for ([_][*:0]const u8{ "q", "index_q", "weights", "latent", "pooled" }, &values) |name, *value| {
            const slot = try ops.slot();
            try mlx.check(mlx.mlx_map_string_to_array_get(slot, arrays, name));
            value.* = slot.*;
        }
        try std.testing.expectEqualSlices(c_int, &.{ 2048, 64, 512 }, mlx.getShape(values[0]));
        try std.testing.expectEqualSlices(c_int, &.{ 16384, 512 }, mlx.getShape(values[3]));
        const q = try ops.slice(values[0], 0, 2045, 2048);
        const iq = try ops.slice(values[1], 0, 2045, 2048);
        const weights = try ops.slice(values[2], 0, 2045, 2048);
        const tape = try ops.slice(values[3], 0, 16381, 16384);
        const ev = mlx.mlx_vector_array_new_data(&.{ q, iq, weights, values[3], values[4], tape }, 6);
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_eval(ev));
        return .{ .ops = ops, .q = q, .iq = iq, .weights = weights, .prefix = values[3], .tape = tape, .pooled = values[4] };
    }
    fn deinit(self: *Fixture) void {
        self.ops.deinit();
    }
    fn branchViews(self: *const Fixture, fork: bool) [3]batch_attention.Branch {
        const base = self.prefix_rows;
        return .{
            .{ .offset = base, .length = base + 1, .path = .{ 0, 0, 0 } },
            .{ .offset = base + 1, .length = base + 2, .path = .{ 0, 1, 0 } },
            if (fork) .{ .offset = base + 1, .length = base + 2, .path = .{ 0, 2, 0 } } else .{ .offset = base + 2, .length = base + 3, .path = .{ 0, 1, 2 } },
        };
    }
    fn selectBranches(self: *const Fixture, ops: *Ops, branches: []const batch_attention.Branch) ![3]Arr {
        var ids: [3]Arr = undefined;
        for (branches, &ids, 0..) |branch, *out, row| {
            const state = attention.State{ .latent = self.prefix, .pooled = self.pooled, .processed = branch.length };
            out.* = try ops.own(try attention.probeSelect(&state, try ops.slice(self.iq, 0, @intCast(row), @intCast(row + 1)), try ops.slice(self.weights, 0, @intCast(row), @intCast(row + 1)), branch.offset, ops.s));
        }
        return ids;
    }
    fn outputs(self: *const Fixture, ops: *Ops, fork: bool, arm: usize) !mlx.mlx_vector_array {
        const branches = self.branchViews(fork);
        const selected = try self.selectBranches(ops, &branches);
        const result = mlx.mlx_vector_array_new();
        errdefer _ = mlx.mlx_vector_array_free(result);
        if (arm == 2) {
            const out = (try batch_attention.run(ops, self.q, self.prefix, self.prefix_rows, self.tape, &branches, try ops.concat(&selected, 0), 1.0 / 16.0)) orelse return error.ExpectedDecodeBatch;
            try mlx.check(mlx.mlx_vector_array_append_value(result, out));
        } else for (branches, selected, 0..) |branch, ids, row| {
            const q = try ops.slice(self.q, 0, @intCast(row), @intCast(row + 1));
            const out = if (arm == 1) (try batch_attention.run(ops, q, self.prefix, self.prefix_rows, self.tape, &.{branch}, ids, 1.0 / 16.0)) orelse return error.ExpectedDecodeSingle else blk: {
                const depth = branch.length - self.prefix_rows;
                const path = try ops.own(mlx.mlx_array_new_data(&branch.path, &.{@as(c_int, @intCast(depth))}, 1, .uint32));
                const tail = try ops.take(self.tape, path, 0);
                const state = attention.State{ .latent = self.prefix, .pooled = self.pooled, .processed = branch.length };
                break :blk try ops.own(try attention.probeSelectedOverlay(&state, q, ids, branch.offset, 1.0 / 16.0, .{ .prefix = self.prefix, .prefix_rows = self.prefix_rows, .tail = tail }, ops.s));
            };
            try mlx.check(mlx.mlx_vector_array_append_value(result, out));
        }
        return result;
    }
};

const Proof = struct {
    native_b1_b3_bit_values: usize = 0,
    finite_failures: usize = 0,
    scalar_bit_mismatches: usize = 0,
    scalar_relative_l2: f64 = 0,
    scalar_max_abs: f64 = 0,
    peak_candidate_delta_bytes: usize = 0,
};
fn outputAt(ops: *Ops, values: mlx.mlx_vector_array, index: usize) !Arr {
    const out = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(out, values, index));
    return out.*;
}
fn proof(fx: *const Fixture) !Proof {
    var report = Proof{};
    var norm: f64 = 0;
    var squared: f64 = 0;
    for ([_]bool{ false, true }) |fork| {
        var ops = Ops{ .s = fx.ops.s };
        defer ops.deinit();
        const scalar = try fx.outputs(&ops, fork, 0);
        defer _ = mlx.mlx_vector_array_free(scalar);
        try mlx.check(mlx.mlx_eval(scalar));
        const singles = try fx.outputs(&ops, fork, 1);
        defer _ = mlx.mlx_vector_array_free(singles);
        try mlx.check(mlx.mlx_eval(singles));
        var active: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&active));
        try mlx.check(mlx.mlx_reset_peak_memory());
        const batched = try fx.outputs(&ops, fork, 2);
        defer _ = mlx.mlx_vector_array_free(batched);
        try mlx.check(mlx.mlx_eval(batched));
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        report.peak_candidate_delta_bytes = @max(report.peak_candidate_delta_bytes, peak -| active);
        try std.testing.expect(peak -| active <= batch_attention.scratch_limit);
        const b = mlx.mlx_array_data_bfloat16(try outputAt(&ops, batched, 0)).?;
        for (0..3) |row| {
            const a = mlx.mlx_array_data_bfloat16(try outputAt(&ops, singles, row)).?;
            const old = mlx.mlx_array_data_bfloat16(try outputAt(&ops, scalar, row)).?;
            const count = 64 * 512;
            try std.testing.expectEqualSlices(u16, a[0..count], b[row * count ..][0..count]);
            report.native_b1_b3_bit_values += count;
            for (0..count) |i| {
                const av: f64 = @as(f32, @bitCast(@as(u32, old[i]) << 16));
                const bv: f64 = @as(f32, @bitCast(@as(u32, b[row * count + i]) << 16));
                report.finite_failures += @intFromBool(!std.math.isFinite(av) or !std.math.isFinite(bv));
                report.scalar_bit_mismatches += @intFromBool(old[i] != b[row * count + i]);
                norm += av * av;
                squared += (av - bv) * (av - bv);
                report.scalar_max_abs = @max(report.scalar_max_abs, @abs(av - bv));
            }
        }
    }
    report.scalar_relative_l2 = @sqrt(squared / norm);
    try std.testing.expectEqual(@as(usize, 0), report.finite_failures);
    return report;
}
fn timed(fx: *const Fixture, batch: bool) !u64 {
    const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
    var ops = Ops{ .s = fx.ops.s };
    errdefer ops.deinit();
    const outputs = try fx.outputs(&ops, true, if (batch) 2 else 0);
    errdefer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_eval(outputs));
    _ = mlx.mlx_vector_array_free(outputs);
    ops.deinit();
    return watch.read();
}

fn directedGuards(s: mlx.mlx_stream) !void {
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var cache: [4 * 512]u16 = undefined;
    for (0..4) |row| for (0..512) |d| {
        cache[row * 512 + d] = if (row == 0) 0x3f00 else if (row == 1) 0x7fc0 else 0x7f80;
    };
    var suffix: [3 * 512]u16 = undefined;
    for (0..3) |row| @memset(suffix[row * 512 ..][0..512], ([_]u16{ 0x3e80, 0xbf00, 0x3f80 })[row]);
    const prefix = try ops.own(mlx.mlx_array_new_data(&cache, &.{ 4, 512 }, 2, .bfloat16));
    const tape = try ops.own(mlx.mlx_array_new_data(&suffix, &.{ 3, 512 }, 2, .bfloat16));
    const q = try ops.zeros(&.{ 3, 64, 512 }, .bfloat16);
    const branches = [_]batch_attention.Branch{
        .{ .offset = 1, .length = 2, .path = .{ 0, 0, 0 } },
        .{ .offset = 2, .length = 3, .path = .{ 0, 1, 0 } },
        .{ .offset = 2, .length = 3, .path = .{ 0, 2, 0 } },
    };
    var indices: [3 * 2051]i32 = @splat(-1);
    indices[0] = std.math.minInt(i32);
    indices[1] = std.math.maxInt(i32);
    indices[2] = 3; // A physically present row is future for this branch.
    indices[2051 + 10] = 0; // Key zero must remain valid.
    indices[2 * 2051] = 3; // Fork row2 has position2, not position3.
    indices[3 * 2051 - 1] = 2; // Last slot maps to tape2, not its sibling tape1.
    const selected = try ops.own(mlx.mlx_array_new_data(&indices, &.{ 3, 2051 }, 2, .int32));
    const out = (try batch_attention.run(&ops, q, prefix, 1, tape, &branches, selected, 1.0 / 16.0)) orelse return error.ExpectedGuardBatch;
    const all = mlx.mlx_vector_array_new_data(&.{out}, 1);
    defer _ = mlx.mlx_vector_array_free(all);
    var controls: [6]Arr = undefined;
    for (branches, 0..) |branch, row| {
        const qc = try ops.slice(q, 0, @intCast(row), @intCast(row + 1));
        const ids = try ops.slice(selected, 0, @intCast(row), @intCast(row + 1));
        controls[row] = (try batch_attention.run(&ops, qc, prefix, 1, tape, &.{branch}, ids, 1.0 / 16.0)) orelse return error.ExpectedGuardSingle;
        try mlx.check(mlx.mlx_vector_array_append_value(all, controls[row]));
        const depth = branch.length - 1;
        const path = try ops.own(mlx.mlx_array_new_data(&branch.path, &.{@as(c_int, @intCast(depth))}, 1, .uint32));
        const state = attention.State{ .latent = prefix, .processed = branch.length };
        controls[row + 3] = try ops.own(try attention.probeSelectedOverlay(&state, qc, ids, branch.offset, 1.0 / 16.0, .{ .prefix = prefix, .prefix_rows = 1, .tail = try ops.take(tape, path, 0) }, s));
        try mlx.check(mlx.mlx_vector_array_append_value(all, controls[row + 3]));
    }
    try mlx.check(mlx.mlx_eval(all));
    const bits = mlx.mlx_array_data_bfloat16(out).?;
    for (0..3) |row| {
        const expected = ([_]u16{ 0, 0x3f00, 0x3f80 })[row];
        const chunk = bits[row * 64 * 512 ..][0 .. 64 * 512];
        for (chunk) |bit| try std.testing.expectEqual(expected, bit);
        try std.testing.expectEqualSlices(u16, chunk, mlx.mlx_array_data_bfloat16(controls[row]).?[0 .. 64 * 512]);
        try std.testing.expectEqualSlices(u16, chunk, mlx.mlx_array_data_bfloat16(controls[row + 3]).?[0 .. 64 * 512]);
    }
}

test "GLM decode batch true B3 component proof and inclusive timing" {
    const fixture = std.c.getenv("SUSHI_GLM_DECODE_BATCH_FIXTURE") orelse return error.SkipZigTest;
    const output = std.c.getenv("SUSHI_GLM_DECODE_BATCH_OUT") orelse return error.MissingDecodeBatchOutput;
    mlx.installErrorHandler();
    var fx = try Fixture.init(fixture);
    defer fx.deinit();
    try directedGuards(fx.ops.s);
    const exact = try proof(&fx);
    var ns: [2][samples]u64 = undefined;
    for (0..3) |_| for ([_]bool{ false, true }) |batch| {
        _ = try timed(&fx, batch);
    };
    for (0..samples) |round| for (0..2) |position| {
        const arm = if (round % 2 == 0) position else 1 - position;
        ns[arm][round] = try timed(&fx, arm == 1);
    };
    const data = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .proof = exact,
        .fixture = std.mem.span(fixture),
        .source_prefix_rows = fx.prefix_rows,
        .construction = "captured Q/index Q/weights last3; actual latent last3 tape; chain/fork paths constructed, not speculative captures",
        .arms = .{ "three current scalar split8 overlays", "true native B3 overlay gather + rank4 SDPA" },
        .native_equivalence = "B3 vs three same-helper nativeB1 calls, one endpoint vector settle",
        .timing = "fresh original per-node scalar selectors, ordered IDs, ancestry, gather/scalarpartial+merge/nativeSDPA, ONE endpoint vector eval and frees",
        .selection_in_timing = true,
        .warmup_pairs = 3,
        .pairs = samples,
        .nanoseconds = ns,
        .precision = "BF16 Q/KV/cache/output; native FP32 accumulators; no precision restoration",
        .peak_baseline = "resident fixture plus retained scalar and native B1 outputs/graphs; candidate output included, not net versus scalar",
        .pending_layer_bound_bytes = batch_attention.scratch_limit,
        .runtime_hook = false,
    }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(data);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = data });
}

test "GLM decode batch coupled serial overlay and B3 share target math" {
    const fixture = std.c.getenv("SUSHI_GLM_DECODE_BATCH_FIXTURE") orelse return error.SkipZigTest;
    var fx = try Fixture.init(fixture);
    defer fx.deinit();
    const mode = batch_attention.bind(true);
    defer mode.restore();
    batch_attention.resetCalls();
    for ([_]bool{ false, true }) |fork| {
        var ops = Ops{ .s = fx.ops.s };
        defer ops.deinit();
        const branches = fx.branchViews(fork);
        const selected = try fx.selectBranches(&ops, &branches);
        const batched = (try batch_attention.run(&ops, fx.q, fx.prefix, fx.prefix_rows, fx.tape, &branches, try ops.concat(&selected, 0), 1.0 / 16.0)) orelse return error.ExpectedCoupledBatch;
        const outputs = mlx.mlx_vector_array_new_data(&.{batched}, 1);
        defer _ = mlx.mlx_vector_array_free(outputs);
        var single: [3]Arr = undefined;
        var ordinary: [3]Arr = undefined;
        for (branches, 0..) |branch, row| {
            const qc = try ops.slice(fx.q, 0, @intCast(row), @intCast(row + 1));
            const iq = try ops.slice(fx.iq, 0, @intCast(row), @intCast(row + 1));
            const weights = try ops.slice(fx.weights, 0, @intCast(row), @intCast(row + 1));
            const depth = branch.length - fx.prefix_rows;
            const path = try ops.own(mlx.mlx_array_new_data(&branch.path, &.{@as(c_int, @intCast(depth))}, 1, .uint32));
            const state = attention.State{ .latent = fx.prefix, .pooled = fx.pooled, .processed = branch.length };
            single[row] = try ops.own(try attention.attendOverlay(&state, qc, iq, weights, branch.offset, 1.0 / 16.0, .{ .prefix = fx.prefix, .prefix_rows = fx.prefix_rows, .tail = try ops.take(fx.tape, path, 0) }, ops.s));
            try mlx.check(mlx.mlx_vector_array_append_value(outputs, single[row]));
            if (!fork) {
                ordinary[row] = try ops.own(try attention.attend(&state, qc, iq, weights, branch.offset, 1.0 / 16.0, ops.s));
                try mlx.check(mlx.mlx_vector_array_append_value(outputs, ordinary[row]));
            }
        }
        try mlx.check(mlx.mlx_eval(outputs));
        const bits = mlx.mlx_array_data_bfloat16(batched).?;
        for (0..3) |row| {
            const chunk = bits[row * 64 * 512 ..][0 .. 64 * 512];
            try std.testing.expectEqualSlices(u16, chunk, mlx.mlx_array_data_bfloat16(single[row]).?[0 .. 64 * 512]);
            if (!fork) try std.testing.expectEqualSlices(u16, chunk, mlx.mlx_array_data_bfloat16(ordinary[row]).?[0 .. 64 * 512]);
        }
    }
    try std.testing.expectEqual(@as(usize, 9), batch_attention.b1Calls());
    try std.testing.expectEqual(@as(usize, 2), batch_attention.b3Calls());
}
