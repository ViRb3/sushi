//! Isolated stock BF16 row-batching parity experiment; no runtime integration.
const std = @import("std");
const mlx = @import("mlx.zig");
const native = @import("glm5_model.zig");
const rows = @import("glm5_dflash_kda.zig");
const Ops = native.Ops;

const Case = struct { name: []const u8, n: c_int, k: c_int };
const cases = [_]Case{
    .{ .name = "FA", .n = 128, .k = 4096 },
    .{ .name = "GA", .n = 128, .k = 4096 },
    .{ .name = "beta", .n = 64, .k = 4096 },
    .{ .name = "FB", .n = 8192, .k = 128 },
    .{ .name = "GB", .n = 8192, .k = 128 },
};
const Comparison = struct {
    name: []const u8,
    arm: []const u8,
    n: c_int,
    k: c_int,
    m: c_int,
    seed: u64,
    compared: usize,
    mismatches: usize,
    first: ?usize,
    serial_bits: ?u16,
    batched_bits: ?u16,
};

test "GLM DFlash dense stock batch raw BF16 parity probe" {
    const out = std.c.getenv("SUSHI_GLM_DENSE_BATCH_OUT") orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    var reports: [cases.len * 3 * 3 * 2]Comparison = undefined;
    var count: usize = 0;
    var total_mismatches: usize = 0;
    for (cases, 0..) |case, ci| for (0..3) |trial| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const seed: u64 = @intCast(6191 + ci * 19 + trial);
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, seed));
        const w = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(w, &[_]c_int{ case.n, case.k }, 2, .bfloat16, 0, 0.03125, key.*, s));
        const x = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(x, &[_]c_int{ 1, 4, case.k }, 3, .bfloat16, 0, 1, key.*, s));
        for ([_]mlx.mlx_array{ w.*, x.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
        const linear = native.Linear{ .w = w.*, .input = case.k, .output = case.n };
        for ([_]c_int{ 2, 3, 4 }) |m| {
            var scope = Ops{ .s = s };
            defer scope.deinit();
            const input = try scope.slice(x.*, 1, 0, m);
            const serial = try rows.linearRows(&scope, linear, input, .serial_rows);
            try mlx.check(mlx.mlx_array_eval(serial));
            const size: usize = @intCast(m * case.n);
            const a = mlx.mlx_array_data_bfloat16(serial).?[0..size];
            for ([_]bool{ false, true }) |column_batch| {
                const batched = if (column_batch) try columnRows(&scope, linear, input) else try linear.apply(&scope, input);
                try mlx.check(mlx.mlx_array_eval(batched));
                const b = mlx.mlx_array_data_bfloat16(batched).?[0..size];
                var report = Comparison{ .name = case.name, .arm = if (column_batch) "column_batch_gemv" else "stock_multirow", .n = case.n, .k = case.k, .m = m, .seed = seed, .compared = size, .mismatches = 0, .first = null, .serial_bits = null, .batched_bits = null };
                for (a, b, 0..) |av, bv, i| if (av != bv) {
                    report.mismatches += 1;
                    if (report.first == null) {
                        report.first = i;
                        report.serial_bits = av;
                        report.batched_bits = bv;
                    }
                };
                if (column_batch) total_mismatches += report.mismatches;
                reports[count] = report;
                count += 1;
            }
        }
    };
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .method = "materialized BF16 normal weights/inputs; actual Linear.apply vs exact current linearRows serial fallback; compare every uint16 output", .total_mismatches = total_mismatches, .comparisons = reports[0..count], .runtime_integration = false }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(out), .data = json });
    try std.testing.expectEqual(@as(usize, 0), total_mismatches);
}

fn columnRows(ops: *Ops, linear: native.Linear, x: mlx.mlx_array) !mlx.mlx_array {
    return (try @import("glm5_dflash_dense_rows.zig").project(ops, linear, x)) orelse error.TestExpectedDenseColumnRows;
}

fn chain(ops: *Ops, bank: [5]native.Linear, x: mlx.mlx_array, column_batch: bool) ![3]mlx.mlx_array {
    var y: [5]mlx.mlx_array = undefined;
    for (bank, 0..) |linear, i| {
        const input = if (i == 3) y[0] else if (i == 4) y[1] else x;
        y[i] = if (column_batch) try columnRows(ops, linear, input) else try rows.linearRows(ops, linear, input, .serial_rows);
    }
    return .{ y[2], y[3], y[4] };
}

fn eval(values: []const mlx.mlx_array) !void {
    const v = mlx.mlx_vector_array_new_data(values.ptr, values.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
}

fn timed(s: mlx.mlx_stream, bank: [5]native.Linear, x: mlx.mlx_array, column_batch: bool) !u64 {
    const timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    for (0..4) |_| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const y = try chain(&ops, bank, x, column_batch);
        try eval(&y);
    }
    return timer.read() / 4;
}

test "GLM DFlash dense column batch isolated timing" {
    const out = std.c.getenv("SUSHI_GLM_DENSE_BATCH_BENCH_OUT") orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    var samples: [3][31][2]u64 = undefined;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var bank: [5]native.Linear = undefined;
    for (cases, 0..) |case, i| {
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(6300 + i)));
        const w = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(w, &[_]c_int{ case.n, case.k }, 2, .bfloat16, 0, 0.03125, key.*, s));
        try mlx.check(mlx.mlx_array_eval(w.*));
        bank[i] = .{ .w = w.*, .input = case.k, .output = case.n };
    }
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 6333));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &[_]c_int{ 1, 4, 4096 }, 3, .bfloat16, 0, 1, key.*, s));
    try mlx.check(mlx.mlx_array_eval(x.*));
    for ([_]c_int{ 2, 3, 4 }, 0..) |m, mi| {
        const input = try ops.slice(x.*, 1, 0, m);
        var scope = Ops{ .s = s };
        defer scope.deinit();
        const expected = try chain(&scope, bank, input, false);
        const actual = try chain(&scope, bank, input, true);
        try eval(&expected);
        try eval(&actual);
        for (expected, actual) |a, b| {
            const size = mlx.mlx_array_size(a);
            try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..size], mlx.mlx_array_data_bfloat16(b).?[0..size]);
        }
        for (0..10) |_| {
            for ([_]bool{ false, true }) |arm| _ = try timed(s, bank, input, arm);
        }
        for (0..31) |round| for (0..2) |position| {
            const arm = if (round % 2 == 0) position else 1 - position;
            samples[mi][round][arm] = try timed(s, bank, input, arm == 1);
        };
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .nanoseconds = samples, .rows = .{ 2, 3, 4 }, .arms = .{ "exact serial Linear.apply rows", "column batch ordinary GEMV" }, .warmup_pairs = 10, .sample_pairs = 31, .repetitions_per_sample = 4, .method = "fresh five-projection chain; materialized synthetic weights/inputs; one eval vector per chain; alternating AB/BA in one process", .full_model = false, .chain_parity = true }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(out), .data = json });
}
