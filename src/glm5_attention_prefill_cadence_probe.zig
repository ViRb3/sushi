//! Inclusive packed attention cadence qualification on one captured MLA block.
const std = @import("std");
const mlx = @import("mlx.zig");
const attention = @import("glm5_attention.zig");
const packed_attention = @import("glm5_attention_nax_packed.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
const samples = 11;

const Fixture = struct {
    ops: Ops,
    state: attention.State,
    q: Arr,
    iq: Arr,
    weights: Arr,
    offset: usize,
    scale: f32,
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
        var values: [8]Arr = undefined;
        for ([_][*:0]const u8{ "q", "index_q", "weights", "latent", "pooled", "offset", "processed", "scale" }, &values) |name, *value| {
            const slot = try ops.slot();
            try mlx.check(mlx.mlx_map_string_to_array_get(slot, arrays, name));
            value.* = slot.*;
        }
        const ev = mlx.mlx_vector_array_new_data(&values, values.len);
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_eval(ev));
        const history: usize = mlx.mlx_array_data_uint32(values[6]).?[0];
        try std.testing.expect(history == 8192 or history == 16384);
        try std.testing.expectEqualSlices(c_int, &.{ 2048, 64, 512 }, mlx.getShape(values[0]));
        return .{ .ops = ops, .state = .{ .latent = values[3], .pooled = values[4], .processed = history }, .q = values[0], .iq = values[1], .weights = values[2], .offset = mlx.mlx_array_data_uint32(values[5]).?[0], .scale = mlx.mlx_array_data_float32(values[7]).?[0] };
    }
    fn deinit(self: *Fixture) void {
        self.ops.deinit();
    }
    fn run(self: *const Fixture, pair: bool, rows: usize) !Arr {
        const binding = attention.bindPackedCadence(pair);
        defer binding.restore();
        var ops = Ops{ .s = self.ops.s };
        defer ops.deinit();
        const q = try ops.slice(self.q, 0, 0, @intCast(rows));
        const iq = try ops.slice(self.iq, 0, 0, @intCast(rows));
        const weights = try ops.slice(self.weights, 0, 0, @intCast(rows));
        return attention.attend(&self.state, q, iq, weights, self.offset, self.scale, self.ops.s);
    }
};

fn timed(fx: *const Fixture, pair: bool) !u64 {
    const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const value = try fx.run(pair, 2048);
    errdefer _ = mlx.mlx_array_free(value);
    try mlx.check(mlx.mlx_array_eval(value));
    _ = mlx.mlx_array_free(value);
    return watch.read();
}

test "GLM prefill cadence scoped control restores serial fallback" {
    const off = attention.bindPackedCadence(false);
    defer off.restore();
    try std.testing.expect(!attention.packedCadenceEnabled());
    {
        const on = attention.bindPackedCadence(true);
        defer on.restore();
        try std.testing.expect(attention.packedCadenceEnabled());
    }
    try std.testing.expect(!attention.packedCadenceEnabled());
}

test "GLM prefill cadence extra budget covers only a second packed tile" {
    const legacy16 = @import("glm5_attention_nax_packed.zig").bind32(false);
    defer legacy16.restore();
    const pack = packed_attention.bind(true);
    defer pack.restore();
    const off = attention.bindPackedCadence(false);
    defer off.restore();
    try std.testing.expectEqual(@as(usize, 0), try attention.packedCadenceTransientBudget(2048, 2));
    const on = attention.bindPackedCadence(true);
    defer on.restore();
    try std.testing.expectEqual(@as(usize, 0), try attention.packedCadenceTransientBudget(16, 2));
    try std.testing.expectEqual(packed_attention.scratch_limit, try attention.packedCadenceTransientBudget(33, 1));
    try std.testing.expectEqual(packed_attention.scratch_limit * 2, try attention.packedCadenceTransientBudget(2048, 2));
    try std.testing.expectError(error.Overflow, attention.packedCadenceTransientBudget(2048, std.math.maxInt(usize)));
    const unpacked = packed_attention.bind(false);
    defer unpacked.restore();
    try std.testing.expectEqual(@as(usize, 0), try attention.packedCadenceTransientBudget(2048, 2));
}

test "GLM prefill cadence real T2048 inclusive bits memory and paired timing" {
    const path = std.c.getenv("SUSHI_GLM_PREFILL_CADENCE_FIXTURE") orelse return error.SkipZigTest;
    const output = std.c.getenv("SUSHI_GLM_PREFILL_CADENCE_OUT") orelse return error.MissingCadenceOutput;
    const binding = packed_attention.bind(true);
    defer binding.restore();
    mlx.installErrorHandler();
    var fx = try Fixture.init(path);
    defer fx.deinit();
    attention.resetPackedCadenceCalls();
    var exact_values: usize = 0;
    var peak_delta: usize = 0;
    // A final one-row tile exercises an unpaired graph and ragged cleanup.
    for ([_]usize{ 33, 2048 }) |rows| {
        const a = try fx.run(false, rows);
        defer _ = mlx.mlx_array_free(a);
        try mlx.check(mlx.mlx_array_eval(a));
        var active: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&active));
        try mlx.check(mlx.mlx_reset_peak_memory());
        const b = try fx.run(true, rows);
        defer _ = mlx.mlx_array_free(b);
        try mlx.check(mlx.mlx_array_eval(b));
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        peak_delta = @max(peak_delta, peak -| active);
        const count = rows * 64 * 512;
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..count], mlx.mlx_array_data_bfloat16(b).?[0..count]);
        exact_values += count;
        // Two packed banks plus retained small outputs and final concatenation.
        try std.testing.expect(peak -| active <= 2 * packed_attention.scratch_limit + 2 * rows * 64 * 512 * 2 + 16 * 1024 * 1024);
    }
    var ns: [2][samples]u64 = undefined;
    for (0..3) |_| for ([_]bool{ false, true }) |pair| {
        _ = try timed(&fx, pair);
    };
    for (0..samples) |round| for (0..2) |position| {
        const arm = if (round % 2 == 0) position else 1 - position;
        ns[arm][round] = try timed(&fx, arm == 1);
    };
    try std.testing.expectEqual(@as(usize, 16), attention.packedCadenceCalls());
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .history = fx.state.processed, .offset = fx.offset, .query_rows = 2048, .fixture = std.mem.span(path), .exact_bf16_values = exact_values, .peak_candidate_extra_bytes = peak_delta, .peak_baseline = "resident fixture plus held serial reference output; includes candidate output, not net serial-vs-candidate overhead", .packed_cadence_calls = attention.packedCadenceCalls(), .warmups = 3, .pairs = samples, .arms = .{ "serial16", "async2x16" }, .nanoseconds = ns, .timing = "fresh host graphs, selection, gather, native unchanged packed SDPA, settle, output concat, endpoint eval and free", .precision = "BF16 inputs/cache/output; original native FP32 accumulators" }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
