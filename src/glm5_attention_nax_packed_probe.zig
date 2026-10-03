//! Isolated sparse head-packed BF16 D512 attention component probe.
const std = @import("std");
const mlx = @import("mlx.zig");
const nax = @import("glm5_attention_nax_mask.zig");
const headpack = @import("glm5_attention_nax_packed.zig");
const old = @import("glm5_attention_prefill.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
var rows: usize = 31;
const heads: usize = 64;
const width: usize = 512;
const selected_width: usize = 2051;

test "GLM packed NAX probe selection is unique unsorted causal with masked rows" {
    const a = std.testing.allocator;
    for ([_]usize{ 4096, 16384, 32768 }) |history| {
        const ids = try makeSelection(a, history);
        defer a.free(ids);
        const seen = try a.alloc(bool, history);
        defer a.free(seen);
        const offset = history - rows;
        for (0..rows) |row| {
            @memset(seen, false);
            var live: usize = 0;
            for (ids[row * selected_width ..][0..selected_width]) |id| {
                if (id < 0 or id >= history or id > offset + row) continue;
                try std.testing.expect(!seen[@intCast(id)]);
                seen[@intCast(id)] = true;
                live += 1;
            }
            if (row == 0) try std.testing.expectEqual(@as(usize, 0), live);
            if (row == 1) {
                try std.testing.expectEqual(@as(usize, 1), live);
                try std.testing.expect(seen[0]);
            }
            if (row > 2) try std.testing.expectEqual(@as(usize, 2048) + (offset + row + 1) % 4, live);
        }
    }
}

test "GLM packed NAX probe BF16 D512 full component qualification" {
    const path = std.c.getenv("SUSHI_GLM_PACKED_NAX_PROBE_OUT") orelse return error.SkipZigTest;
    try qualify(std.mem.span(path));
}

fn makeSelection(a: std.mem.Allocator, history: usize) ![]i32 {
    const ids = try a.alloc(i32, rows * selected_width);
    errdefer a.free(ids);
    @memset(ids, -1);
    const pools = try a.alloc(i32, history / 4);
    defer a.free(pools);
    var prng = std.Random.DefaultPrng.init(@intCast(history + 53));
    const offset = history - rows;
    ids[selected_width] = 0;
    for (2..rows) |row| {
        const pos = offset + row;
        const complete = (pos + 1) / 4;
        for (pools[0..complete], 0..) |*pool, i| pool.* = @intCast(i);
        prng.random().shuffle(i32, pools[0..complete]);
        for (0..512) |slot| for (0..4) |j| {
            ids[row * selected_width + slot * 4 + j] = pools[slot] * 4 + @as(i32, @intCast(j));
        };
        const rem = (pos + 1) % 4;
        for (0..rem) |j| ids[row * selected_width + 2048 + j] = @intCast(pos + 1 - rem + j);
    }
    ids[2 * selected_width] = -1;
    ids[2 * selected_width + 4] = @intCast(history);
    ids[2 * selected_width + 8] = @intCast(offset + 3);
    ids[2 * selected_width + 12] = @intCast(history + 17);
    return ids;
}

const Fixture = struct {
    ops: Ops,
    q: Arr,
    cache: Arr,
    selected: Arr,
    history: usize,
    fn init(history: usize, s: mlx.mlx_stream) !Fixture {
        var ops = Ops{ .s = s };
        errdefer ops.deinit();
        const key_q = try ops.slot();
        const key_k = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key_q, @intCast(history + 431)));
        try mlx.check(mlx.mlx_random_key(key_k, @intCast(history + 977)));
        const q = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(q, &.{ @as(c_int, @intCast(rows)), heads, width }, 3, .bfloat16, 0, 1, key_q.*, s));
        const cache = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(cache, &.{ @intCast(history), width }, 2, .bfloat16, 0, 0.5, key_k.*, s));
        const ids = try makeSelection(std.testing.allocator, history);
        defer std.testing.allocator.free(ids);
        const selected = try ops.own(mlx.mlx_array_new_data(ids.ptr, &.{ @as(c_int, @intCast(rows)), selected_width }, 2, .int32));
        const values = [_]Arr{ q.*, cache.*, selected };
        const ev = mlx.mlx_vector_array_new_data(&values, values.len);
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_eval(ev));
        return .{ .ops = ops, .q = q.*, .cache = cache.*, .selected = selected, .history = history };
    }
    fn deinit(self: *Fixture) void {
        self.ops.deinit();
    }
    fn original(self: *const Fixture, ops: *Ops) !Arr {
        const offset: u32 = @intCast(self.history - rows);
        const length: u32 = @intCast(self.history);
        const oa = try ops.own(mlx.mlx_array_new_data(&offset, &.{}, 0, .uint32));
        const la = try ops.own(mlx.mlx_array_new_data(&length, &.{}, 0, .uint32));
        return ops.own(try old.attend(self.q, self.cache, self.selected, oa, la, try ops.scalar(1.0 / 16.0, .float32), true, ops.s));
    }
    fn fullHistory(self: *const Fixture, ops: *Ops) !Arr {
        const value = (try nax.run(ops, self.q, self.cache, self.selected, self.history - rows, self.history, 1.0 / 16.0, .bf16)) orelse return error.ExpectedForcedD512Nax;
        try std.testing.expectEqualSlices(c_int, &.{ @as(c_int, @intCast(rows)), heads, width }, mlx.getShape(value));
        try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(value));
        return value;
    }

    fn candidate(self: *const Fixture, ops: *Ops) !Arr {
        const parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        var start: usize = 0;
        while (start < rows) {
            const end = @min(rows, start + headpack.max_rows);
            var scope = Ops{ .s = ops.s };
            defer scope.deinit();
            const q = try scope.slice(self.q, 0, @intCast(start), @intCast(end));
            const selected = try scope.slice(self.selected, 0, @intCast(start), @intCast(end));
            const value = (try headpack.run(&scope, q, self.cache, selected, self.history - rows + start, self.history, 1.0 / 16.0)) orelse return error.ExpectedPackedNax;
            // Settle each bounded bank before releasing the scope. Retain only
            // the small output plane, then concatenate after all chunks.
            try mlx.check(mlx.mlx_array_eval(value));
            try mlx.check(mlx.mlx_vector_array_append_value(parts, value));
            start = end;
        }
        const result = try ops.slot();
        try mlx.check(mlx.mlx_concatenate_axis(result, parts, 0, ops.s));
        return result.*;
    }
};

const samples = 6;
const Comparison = struct {
    history: usize,
    query_rows: usize,
    q_heads: usize = heads,
    latent_width: usize = width,
    selected_width: usize = selected_width,
    compared: usize = 0,
    finite_failures: usize = 0,
    bf16_bit_mismatches: usize = 0,
    all_invalid_nonzero: usize = 0,
    key_zero_mismatches: usize = 0,
    normal_relative_l2: f64 = 0,
    normal_max_abs: f64 = 0,
    normal_cosine: f64 = 0,
    normal_rms_ratio: f64 = 0,
    full_history_relative_l2: f64 = 0,
    packed_vs_full_history_relative_l2: f64 = 0,
    planned_temporary_bytes: usize,
    active_before_accuracy: usize = 0,
    peak_accuracy_bytes: usize = 0,
    original_ns: [samples]u64 = @splat(0),
    nax_ns: [samples]u64 = @splat(0),
    full_history_ns: [samples]u64 = @splat(0),
};
fn bf16(value: u16) f32 {
    return @bitCast(@as(u32, value) << 16);
}
fn accuracy(fx: *const Fixture) !Comparison {
    var scope = Ops{ .s = fx.ops.s };
    defer scope.deinit();
    var report = Comparison{ .history = fx.history, .query_rows = rows, .planned_temporary_bytes = try headpack.temporaryBytes(@min(rows, headpack.max_rows)) + rows * heads * width * 2 * 4 };
    try mlx.check(mlx.mlx_get_active_memory(&report.active_before_accuracy));
    try mlx.check(mlx.mlx_reset_peak_memory());
    const original = try fx.original(&scope);
    const full_history = try fx.fullHistory(&scope);
    const candidate = try fx.candidate(&scope);
    try mlx.check(mlx.mlx_array_eval(original));
    try mlx.check(mlx.mlx_array_eval(candidate));
    try mlx.check(mlx.mlx_array_eval(full_history));
    try mlx.check(mlx.mlx_get_peak_memory(&report.peak_accuracy_bytes));
    const a = mlx.mlx_array_data_bfloat16(original) orelse return error.MlxArrayDataNull;
    const b = mlx.mlx_array_data_bfloat16(candidate) orelse return error.MlxArrayDataNull;
    const c = mlx.mlx_array_data_bfloat16(full_history) orelse return error.MlxArrayDataNull;
    const cache = mlx.mlx_array_data_bfloat16(fx.cache) orelse return error.MlxArrayDataNull;
    var ref_sum: f64 = 0;
    var out_sum: f64 = 0;
    var error_sum: f64 = 0;
    var cross: f64 = 0;
    var full_error: f64 = 0;
    var packed_full_error: f64 = 0;
    report.compared = rows * heads * width;
    for (0..report.compared) |i| {
        const av: f64 = bf16(a[i]);
        const bv: f64 = bf16(b[i]);
        const cv: f64 = bf16(c[i]);
        if (!std.math.isFinite(av) or !std.math.isFinite(bv) or !std.math.isFinite(cv)) report.finite_failures += 1;
        if (a[i] != b[i]) report.bf16_bit_mismatches += 1;
        if (i < heads * width and (a[i] != 0 or b[i] != 0 or c[i] != 0)) report.all_invalid_nonzero += 1;
        if (i >= heads * width and i < 2 * heads * width and (a[i] != cache[i % width] or b[i] != cache[i % width] or c[i] != cache[i % width])) report.key_zero_mismatches += 1;
        if (i < 2 * heads * width) continue;
        const diff = av - bv;
        ref_sum += av * av;
        out_sum += bv * bv;
        cross += av * bv;
        error_sum += diff * diff;
        full_error += (av - cv) * (av - cv);
        packed_full_error += (bv - cv) * (bv - cv);
        report.normal_max_abs = @max(report.normal_max_abs, @abs(diff));
    }
    report.normal_relative_l2 = @sqrt(error_sum / ref_sum);
    report.full_history_relative_l2 = @sqrt(full_error / ref_sum);
    report.packed_vs_full_history_relative_l2 = @sqrt(packed_full_error / ref_sum);
    report.normal_cosine = cross / @sqrt(ref_sum * out_sum);
    report.normal_rms_ratio = @sqrt(out_sum / ref_sum);
    return report;
}
fn timed(fx: *const Fixture, arm: usize) !u64 {
    const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
    var scope = Ops{ .s = fx.ops.s };
    errdefer scope.deinit();
    const value = switch (arm) { 0 => try fx.original(&scope), 1 => try fx.fullHistory(&scope), else => try fx.candidate(&scope) };
    try mlx.check(mlx.mlx_array_eval(value));
    scope.deinit();
    return watch.read();
}
fn write(path: []const u8, reports: []const Comparison, complete: bool) !void {
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .complete = complete,
        .dtype = "BF16 Q/K/V/cache/output; native float score/output accumulators",
        .input = "independent fixed-seed BF16 normal Q(sd1) and cache(sd0.5); unsorted unique 512 pools plus causal tail; all-invalid row0; sole key0 row1; mixed invalid/future row2",
        .reference = "glm5_attention_prefill.attend exact online scalar indexed traversal, direct finalization",
        .candidate = "glm5_attention_nax_headpack.run, real rows=fake batch; 64 real heads=fakeQ64; one masked gather sharedK/V; BF16 native fullNAX FP32acc; force_fused true",
        .normal_accuracy_scope = "rows2..T-1; excludes all-invalid and sole-key rows",
        .timing = "fresh bounded Ops per packedchunk; gather/mask/reshape/SDPA/eval/free and concat included; same fixtures; two warmups, six ABC/CBA rounds; arms scalar/fullhistoryNAX/headpackedNAX",
        .runtime_integration = false,
        .comparisons = reports,
    }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = json });
}
fn qualify(path: []const u8) !void {
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    var reports: [6]Comparison = undefined;
    var count: usize = 0;
    for ([_]usize{ 16, 31 }) |query_rows| {
      rows = query_rows;
      for ([_]usize{ 4096, 16384, 32768 }) |history| {
        var fixture = try Fixture.init(history, s);
        defer fixture.deinit();
        reports[count] = try accuracy(&fixture);
        count += 1;
        try write(path, reports[0..count], false);
        const report = &reports[count - 1];
        try std.testing.expectEqual(@as(usize, 0), report.finite_failures);
        try std.testing.expect(report.peak_accuracy_bytes -| report.active_before_accuracy <= report.planned_temporary_bytes + 16 * 1024 * 1024);
        try std.testing.expectEqual(@as(usize, 0), report.all_invalid_nonzero);
        try std.testing.expectEqual(@as(usize, 0), report.key_zero_mismatches);
        try std.testing.expect(report.normal_relative_l2 < 0.01);
        try std.testing.expect(report.normal_cosine > 0.9999);
        try std.testing.expect(report.normal_rms_ratio > 0.99 and report.normal_rms_ratio < 1.01);
        for (0..2) |_| {
            for (0..3) |arm| { _ = try timed(&fixture, arm); }
        }
        for (0..samples) |sample| for (0..3) |slot| {
            const arm = if (sample % 2 == 0) slot else 2 - slot;
            const ns = try timed(&fixture, arm);
            switch (arm) { 0 => report.original_ns[sample] = ns, 1 => report.full_history_ns[sample] = ns, else => report.nax_ns[sample] = ns, }
        };
        try write(path, reports[0..count], false);
    }
    }
    try write(path, reports[0..count], true);
}
