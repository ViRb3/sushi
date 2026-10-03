//! Inclusive two-arm grid scheduling probe on actual T2048 routes and original L20 banks.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const api = @import("root.zig");
const grid = @import("glm_prefill_grid.zig");
const Arr = mlx.mlx_array;
const a = std.testing.allocator;
const Fixture = struct {
    values: [12]Arr = undefined,
    count: usize = 0,
    x: Arr = .{ .ctx = null },
    ids: Arr = .{ .ctx = null },
    scores: Arr = .{ .ctx = null },
    bank: api.Bank = undefined,
    fn deinit(self: *Fixture) void {
        for (self.values[0..self.count]) |v| {
            _ = mlx.mlx_array_free(v);
        }
    }
    fn own(self: *Fixture, arrays: mlx.mlx_map_string_to_array, name: [*:0]const u8) !Arr {
        var v = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_map_string_to_array_get(&v, arrays, name));
        if (v.ctx == null) return error.MissingGlmGridTensor;
        self.values[self.count] = v;
        self.count += 1;
        return v;
    }
    fn init(path: [*:0]const u8, model_path: []const u8) !Fixture {
        var f: Fixture = .{};
        errdefer f.deinit();
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var arrays = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(arrays);
        var metadata = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(metadata);
        try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, path, cpu));
        f.x = try f.own(arrays, "x");
        f.ids = try f.own(arrays, "indices");
        f.scores = try f.own(arrays, "scores");
        inline for (.{ "gate", "up", "down" }) |name| {
            const joined = try std.fs.path.join(a, &.{ model_path, "model-exl3-L20-" ++ name ++ ".safetensors" });
            defer a.free(joined);
            const zpath = try a.dupeSentinel(u8, joined, 0);
            defer a.free(zpath);
            var bank_arrays = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(bank_arrays);
            try mlx.check(mlx.mlx_load_safetensors(&bank_arrays, &metadata, zpath, cpu));
            var p: api.Proj = undefined;
            inline for (.{ "trellis", "suh", "svh" }) |kind| @field(p, kind) = try f.own(bank_arrays, "model.language_model.layers.20.mlp.switch_mlp." ++ name ++ "_proj." ++ kind);
            @field(f.bank, name) = p;
        }
        const ev = mlx.mlx_vector_array_new_data(&f.values, f.count);
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_eval(ev));
        return f;
    }
    fn run(self: Fixture, transposed: bool) !Arr {
        if (transposed) {
            const binding = grid.bind(true);
            defer binding.restore();
            return (try grid.tryMoe(mlx.gpuStream(), self.x, self.bank, self.ids, self.scores, .{ .codebook = .mcg, .window = .w12 }, 10)) orelse error.ExpectedGlmGridCandidate;
        }
        return api.moeClamped(mlx.gpuStream(), self.x, self.bank, self.ids, self.scores, .{ .codebook = .mcg, .window = .w12 }, 10);
    }
};
fn timed(f: Fixture, transposed: bool) !u64 {
    const watch = @import("mlx_host").io_util.Stopwatch.init(std.testing.io);
    const y = try f.run(transposed);
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    _ = mlx.mlx_array_free(y);
    return watch.read();
}
test "GLM prefill grid actual T2048 L20 inclusive parity and timing" {
    const path = std.c.getenv("SUSHI_GLM_PREFILL_GRID_FIXTURE") orelse return error.SkipZigTest;
    const model_path = std.c.getenv("SUSHI_GLM_PREFILL_GRID_MODEL") orelse return error.MissingGlmGridModel;
    const output = std.c.getenv("SUSHI_GLM_PREFILL_GRID_OUT") orelse return error.MissingGlmGridOutput;
    if (std.c.getenv("SUSHI_EXL3_GEMM_WIN")) |value| if (!std.mem.eql(u8, std.mem.span(value), "32")) return error.ExpectedNativeWin32;
    if (std.c.getenv("SUSHI_EXL3_WIN_ALIGN")) |value| if (value[0] == '0') return error.ExpectedAlignedNativeWindows;
    var f = try Fixture.init(path, std.mem.span(model_path));
    defer f.deinit();
    grid.resetDispatchCount();
    const old = try f.run(false);
    defer _ = mlx.mlx_array_free(old);
    const new = try f.run(true);
    defer _ = mlx.mlx_array_free(new);
    try mlx.check(mlx.mlx_array_eval(old));
    try mlx.check(mlx.mlx_array_eval(new));
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(old).?[0 .. 2048 * 4096], mlx.mlx_array_data_bfloat16(new).?[0 .. 2048 * 4096]);
    try std.testing.expectEqual(@as(usize, 1), grid.dispatchCount());
    var ns: [2][11]u64 = undefined;
    for (0..3) |_| for ([_]bool{ false, true }) |arm| {
        _ = try timed(f, arm);
    };
    for (0..11) |round| for (0..2) |position| {
        const arm = if (round % 2 == 0) position else 1 - position;
        ns[arm][round] = try timed(f, arm == 1);
    };
    try std.testing.expectEqual(@as(usize, 15), grid.dispatchCount());
    const json = try std.json.Stringify.valueAlloc(a, .{ .layer = 20, .tokens = 2048, .assignments = 16384, .experts = 288, .window_capacity = 800, .bank_layout = "original full 288 expert banks, no compaction", .exact_bf16_values = 2048 * 4096, .arms = .{ "native grid output-stripe then window", "transposed grid window then output-stripe" }, .warmup_pairs = 3, .pairs = 11, .nanoseconds = ns, .timing = "whole routed chain: sort, metadata/inverse, prepare, three unchanged NAX GEMMs, middle, finish, allocation, endpoint eval/free; actual BF16 input/routes/scores; resident original banks", .runtime_hook = false }, .{ .whitespace = .indent_2 });
    defer a.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
test {
    _ = grid;
}
