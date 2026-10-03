//! Fixed native B32 packed attention; exact original T16 selectors and two graphs.
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
        try std.testing.expect(history == 16384);
        try std.testing.expectEqualSlices(c_int, &.{ 2048, 64, 512 }, mlx.getShape(values[0]));
        return .{ .ops = ops, .state = .{ .latent = values[3], .pooled = values[4], .processed = history }, .q = values[0], .iq = values[1], .weights = values[2], .offset = mlx.mlx_array_data_uint32(values[5]).?[0], .scale = mlx.mlx_array_data_float32(values[7]).?[0] };
    }
    fn deinit(self: *Fixture) void {
        self.ops.deinit();
    }
    fn run(self: *const Fixture, on: bool, rows: usize) !Arr {
        const wide = packed_attention.bind32(on);
        defer wide.restore();
        const cadence = attention.bindPackedCadence(true);
        defer cadence.restore();
        var ops = Ops{ .s = self.ops.s };
        defer ops.deinit();
        return attention.attend(&self.state, try ops.slice(self.q, 0, 0, @intCast(rows)), try ops.slice(self.iq, 0, 0, @intCast(rows)), try ops.slice(self.weights, 0, 0, @intCast(rows)), self.offset, self.scale, self.ops.s);
    }
};

fn exact(a: Arr, b: Arr, dtype: mlx.mlx_dtype) !void {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const ca = try ops.contiguous(a);
    const cb = try ops.contiguous(b);
    const values = mlx.mlx_vector_array_new_data(&.{ ca, cb }, 2);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(ca), mlx.getShape(cb));
    const count = mlx.mlx_array_size(ca);
    var differs: usize = 0;
    if (dtype == .bfloat16) {
        for (mlx.mlx_array_data_bfloat16(ca).?[0..count], mlx.mlx_array_data_bfloat16(cb).?[0..count]) |av, bv| { if (av != bv) differs += 1; }
    } else {
        for (mlx.mlx_array_data_int32(ca).?[0..count], mlx.mlx_array_data_int32(cb).?[0..count]) |av, bv| { if (av != bv) differs += 1; }
    }
    if (differs != 0) std.debug.print("B32 parity mismatch: values={d},different={d}\n", .{ count, differs });
    try std.testing.expectEqual(@as(usize, 0), differs);
}
fn control32(ops: *Ops, q: Arr, cache: Arr, ids: Arr, offset: usize, history: usize) !Arr {
    const binding = packed_attention.bind32(false);
    defer binding.restore();
    var out: [2]Arr = undefined;
    for (&out, 0..) |*value, half| {
        const start: c_int = @intCast(half * 16);
        value.* = (try packed_attention.run(ops, try ops.slice(q, 0, start, start + 16), cache, try ops.slice(ids, 0, start, start + 16), offset + half * 16, history, 1.0 / 16.0)) orelse return error.ExpectedCurrentPacked;
    }
    return ops.concat(&out, 0);
}
test "GLM packed32 native bits nonfinite empty causal and head guards" {
    const path = std.c.getenv("SUSHI_GLM_PACKED32_RAW_OUT") orelse return error.SkipZigTest;
    @import("log.zig").enableStderr();
    mlx.installErrorHandler();
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const alloc = std.testing.allocator;
    const query = try alloc.alloc(u16, 32 * 64 * 512);
    defer alloc.free(query);
    @memset(query, 0);
    for (0..32) |row| for (0..64) |head| { query[(row * 64 + head) * 512] = if (head % 2 == 0) 0x4180 else 0xc180; };
    const q = try ops.own(mlx.mlx_array_new_data(query.ptr, &.{ 32, 64, 512 }, 3, .bfloat16));
    var bank: [36 * 512]u16 = @splat(0);
    for (0..512) |d| {
        bank[d] = if (d % 2 == 0) 0x7fc0 else 0x7f80; // valid key0 is nonfinite
        bank[35 * 512 + d] = bank[d]; // future for all32 actual queries
    }
    bank[512] = 0x3f80;
    bank[1024] = 0xbf80;
    const cache = try ops.own(mlx.mlx_array_new_data(&bank, &.{ 36, 512 }, 2, .bfloat16));
    const selected = try alloc.alloc(i32, 32 * 2051);
    defer alloc.free(selected);
    @memset(selected, -1);
    selected[0] = std.math.minInt(i32);
    selected[1] = std.math.maxInt(i32);
    selected[2] = 35; // row0 empty, invalid rows must not read nonfinite key0
    for (1..31) |row| {
        selected[row * 2051] = 1;
        selected[row * 2051 + 2050] = 2;
        selected[row * 2051 + 1] = 35;
    }
    selected[31 * 2051] = 0; // genuinely valid selected nonfinite must propagate
    const ids = try ops.own(mlx.mlx_array_new_data(selected.ptr, &.{ 32, 2051 }, 2, .int32));
    const control = try control32(&ops, q, cache, ids, 3, 36);
    const wide = packed_attention.bind32(true);
    defer wide.restore();
    const actual = (try packed_attention.run(&ops, q, cache, ids, 3, 36, 1.0 / 16.0)) orelse return error.MissingPacked32;
    try exact(control, actual, .bfloat16);
    const bits = mlx.mlx_array_data_bfloat16(actual).?;
    for (0..64 * 512) |i| try std.testing.expectEqual(@as(u16, 0), bits[i]);
    for (1..31) |row| for (0..64) |head| {
        const start = (row * 64 + head) * 512;
        const value: f32 = @bitCast(@as(u32, bits[start]) << 16);
        if (head % 2 == 0) try std.testing.expect(value > 0.7 and value < 0.8) else try std.testing.expect(value < -0.7 and value > -0.8);
        for (1..512) |d| try std.testing.expectEqual(@as(u16, 0), bits[start + d]);
    };
    for (31 * 64 * 512..32 * 64 * 512) |i| {
        const value: f32 = @bitCast(@as(u32, bits[i]) << 16);
        try std.testing.expect(!std.math.isFinite(value));
    }
    for ([_]c_int{ 17, 24, 31 }) |rows| try std.testing.expect((try packed_attention.run(&ops, try ops.slice(q, 0, 0, rows), cache, try ops.slice(ids, 0, 0, rows), 3, 36, 1.0 / 16.0)) == null);
    const json = try std.json.Stringify.valueAlloc(alloc, .{ .bf16_values_exact = 32 * 64 * 512, .head_row_key0_lastslot_order = true, .invalid_future_nonfinite_zero_before_read = true, .valid_selected_nonfinite_propagates = true, .empty_positive_zero = true, .batch17to31_declined = true, .precision = "unchanged native D512 Q64/head1/K2051 arithmetic;batch32 only" }, .{ .whitespace = .indent_2 });
    defer alloc.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = json });
}

fn selectedProof(fx: *const Fixture) !usize {
    var checked: usize = 0;
    for (0..64) |block| {
        var ops = Ops{ .s = fx.ops.s };
        defer ops.deinit();
        const start: c_int = @intCast(block * 32);
        var expected: [2]Arr = undefined;
        for (&expected, 0..) |*out, half| {
            const first = start + @as(c_int, @intCast(half * 16));
            out.* = try ops.own(try attention.probeSelect(&fx.state, try ops.slice(fx.iq, 0, first, first + 16), try ops.slice(fx.weights, 0, first, first + 16), fx.offset + @as(usize, @intCast(first)), fx.ops.s));
        }
        const current_ids = try ops.concat(&expected, 0);
        const candidate_ids = try ops.own(try attention.probePackedSelect(&fx.state, try ops.slice(fx.iq, 0, start, start + 32), try ops.slice(fx.weights, 0, start, start + 32), fx.offset + @as(usize, @intCast(start)), fx.ops.s));
        try exact(current_ids, candidate_ids, .int32);
        checked += 32 * 2051;
    }
    return checked;
}
fn timed(fx: *const Fixture, on: bool) !u64 {
    const clock = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const out = try fx.run(on, 2048);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    _ = mlx.mlx_array_free(out);
    return clock.read();
}
test "GLM packed32 current T16 selector whole attention ragged bits and pairs" {
    const path = std.c.getenv("SUSHI_GLM_PACKED32_FIXTURE") orelse return error.SkipZigTest;
    const output = std.c.getenv("SUSHI_GLM_PACKED32_OUT") orelse return error.MissingPacked32Output;
    @import("log.zig").enableStderr();
    mlx.installErrorHandler();
    const pack = packed_attention.bind(true);
    defer pack.restore();
    var fx = try Fixture.init(path);
    defer fx.deinit();
    const ids = try selectedProof(&fx);
    var checked: usize = 0;
    var peak_delta: usize = 0;
    for ([_]usize{ 33, 49, 2048 }) |rows| {
        const control = try fx.run(false, rows);
        defer _ = mlx.mlx_array_free(control);
        try mlx.check(mlx.mlx_array_eval(control));
        var active: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&active));
        try mlx.check(mlx.mlx_reset_peak_memory());
        packed_attention.resetDispatchCount();
        const candidate = try fx.run(true, rows);
        defer _ = mlx.mlx_array_free(candidate);
        try exact(control, candidate, .bfloat16);
        try std.testing.expectEqual(@as(usize, if (rows == 2048) 64 else 1), packed_attention.wideDispatchCount());
        try std.testing.expectEqual(@as(usize, if (rows == 33) 2 else if (rows == 49) 3 else 64), packed_attention.dispatchCount());
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        peak_delta = @max(peak_delta, peak -| active);
        try std.testing.expect(peak -| active <= 2 * packed_attention.wide_scratch_limit + 2 * rows * 64 * 512 * 2 + 16 * 1024 * 1024);
        checked += rows * 64 * 512;
    }
    const before_index = @import("glm5_indexpool_nax.zig").dispatchCount();
    var ns: [2][samples]u64 = undefined;
    var batches: [2][samples]usize = undefined;
    var wide_batches: [2][samples]usize = undefined;
    for (0..3) |_| for ([_]bool{ false, true }) |on| { _ = try timed(&fx, on); };
    for (0..samples) |round| for (0..2) |position| {
        const arm = if (round % 2 == 0) position else 1 - position;
        packed_attention.resetDispatchCount();
        ns[arm][round] = try timed(&fx, arm == 1);
        batches[arm][round] = packed_attention.dispatchCount();
        wide_batches[arm][round] = packed_attention.wideDispatchCount();
        try std.testing.expectEqual(@as(usize, if (arm == 1) 64 else 128), batches[arm][round]);
        try std.testing.expectEqual(@as(usize, if (arm == 1) 64 else 0), wide_batches[arm][round]);
    };
    try std.testing.expectEqual(@as(usize, 28 * 128), @import("glm5_indexpool_nax.zig").dispatchCount() - before_index);
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .history = fx.state.processed, .query_rows = 2048, .offset = fx.offset, .ordered_selected_ids_exact = ids, .bf16_values_exact = checked, .ragged = .{ 33, 49 }, .native_contract = "Q64/head1/D512/K2051,unchanged native body and precision;B16 versus B32 only", .selector_contract = "128 original T16 calls per full arm;no T32 scorer,no pool overlap", .warmups = 3, .pairs = samples, .arms = .{ "current B16 two pending", "B32 two pending" }, .packed_batches = batches, .b32_batches = wide_batches, .nax_selector_calls = @import("glm5_indexpool_nax.zig").dispatchCount() - before_index, .peak_candidate_extra_bytes = peak_delta, .peak_scope = "fixture plus held current output;candidate output and owner graphs included", .graph_allowance_bytes = packed_attention.wide_scratch_limit, .nanoseconds = ns, .timing = "both original selectors/ordered ID concat,gather,unchanged native SDPA/zeroing,two graph settlement,result concat,endpoint evaluation and everyfree" }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
