//! Source-only IndexPool score/selection component probe; no model hook.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const score = @import("glm5_indexpool_nax.zig");
const Arr = mlx.mlx_array;
const rows = 16;
const samples = 6;
const Fixture = struct {
    ops: Ops,
    q: Arr,
    keys: Arr,
    weights: Arr,
    pools: usize,
    offset: usize,
    fn init(history: usize, s: mlx.mlx_stream) !Fixture {
        var ops = Ops{ .s = s };
        errdefer ops.deinit();
        const key = try ops.slot();
        const key_k = try ops.slot();
        const key_w = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(history + 703)));
        try mlx.check(mlx.mlx_random_key(key_k, @intCast(history + 971)));
        try mlx.check(mlx.mlx_random_key(key_w, @intCast(history + 1103)));
        const q = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(q, &.{ rows, 32, 128 }, 3, .bfloat16, 0, 1, key.*, s));
        const keys = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(keys, &.{ @as(c_int, @intCast(history / 4)), 128 }, 2, .bfloat16, 0, 0.5, key_k.*, s));
        const weights = try ops.slot();
        // Positive and negative BF16 weights exercise the real score contract.
        try mlx.check(mlx.mlx_random_normal(weights, &.{ rows, 32 }, 2, .bfloat16, 0, 0.125, key_w.*, s));
        const values = mlx.mlx_vector_array_new_data(&.{ q.*, keys.*, weights.* }, 3);
        defer _ = mlx.mlx_vector_array_free(values);
        try mlx.check(mlx.mlx_eval(values));
        return .{ .ops = ops, .q = q.*, .keys = keys.*, .weights = weights.*, .pools = history / 4, .offset = history - rows };
    }
    fn deinit(self: *Fixture) void { self.ops.deinit(); }
    fn scores(self: *const Fixture, ops: *Ops, nax: bool) !Arr {
        return if (nax) ops.own(try score.scores(self.q, self.keys, self.weights, self.offset, self.pools, ops.s)) else score.reference(ops, self.q, self.keys, self.weights, self.offset, self.pools);
    }
};
const EXPAND: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=16u*2051u) return;
    \\const uint row=i/2051u,col=i%2051u,pos=uint(offset)+row;
    \\int token=-1;
    \\if(col<2048u) {const int p=int(selected[row*512u+col/4u]);if(p>=0 && (uint(p)+1u)*4u<=pos+1u) token=p*4+int(col%4u);}
    \\else {const uint rem=(pos+1u)%4u,j=col-2048u;if(j<rem) token=int(pos+1u-rem+j);}
    \\out[i]=token;
;
var expand_kernel: ?mlx.mlx_fast_metal_kernel = null;
fn expand(ops: *Ops, selected: Arr, offset: usize) !Arr {
    if (expand_kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "selected", "offset" }, 2);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_index_probe_expand", ins, outs, EXPAND, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        expand_kernel = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, 2051 }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 2051, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const off: u32 = @intCast(offset);
    const oa = try ops.own(mlx.mlx_array_new_data(&off, &.{}, 0, .uint32));
    const iv = mlx.mlx_vector_array_new_data(&.{ selected, oa }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, expand_kernel.?, iv, cfg, ops.s));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(out, ov, 0));
    return out.*;
}
const Report = struct {
    history: usize,
    rows: usize = rows,
    score_values: usize = 0,
    score_bit_mismatches: usize = 0,
    max_abs: f64 = 0,
    relative_l2: f64 = 0,
    future_score_mismatches: usize = 0,
    bf16_boundary_failures: usize = 0,
    pool_overlap: [rows]usize = @splat(0),
    reference_cutoff_ties: [rows]usize = @splat(0),
    candidate_cutoff_ties: [rows]usize = @splat(0),
    scalar_score_ns: [samples]u64 = @splat(0),
    nax_score_ns: [samples]u64 = @splat(0),
    scalar_select_ns: [samples]u64 = @splat(0),
    nax_select_ns: [samples]u64 = @splat(0),
};
fn audit(f: *const Fixture) !Report {
    var ops = Ops{ .s = f.ops.s };
    defer ops.deinit();
    const a = try f.scores(&ops, false);
    const b = try f.scores(&ops, true);
    const ta = try ops.contiguous(try score.topPools(&ops, a));
    const tb = try ops.contiguous(try score.topPools(&ops, b));
    const ea = try expand(&ops, ta, f.offset);
    const eb = try expand(&ops, tb, f.offset);
    const values = mlx.mlx_vector_array_new_data(&.{ a, b, ta, tb, ea, eb }, 6);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    const av = mlx.mlx_array_data_float32(a).?;
    const bv = mlx.mlx_array_data_float32(b).?;
    const ai = mlx.mlx_array_data_uint32(ta).?;
    const bi = mlx.mlx_array_data_uint32(tb).?;
    var report = Report{ .history = f.pools * 4, .score_values = rows * f.pools };
    const seen = try std.testing.allocator.alloc(u8, f.pools);
    defer std.testing.allocator.free(seen);
    var square: f64 = 0;
    var norm: f64 = 0;
    for (0..rows) |row| {
        @memset(seen, 0);
        const complete = (f.offset + row + 1) / 4;
        var cutoff_a: f32 = std.math.inf(f32);
        var cutoff_b: f32 = std.math.inf(f32);
        for (0..512) |slot| {
            const pa = ai[row * 512 + slot];
            const pb = bi[row * 512 + slot];
            try std.testing.expect(pa < complete and pb < complete);
            try std.testing.expect((seen[pa] & 1) == 0 and (seen[pb] & 2) == 0);
            seen[pa] |= 1;
            seen[pb] |= 2;
            cutoff_a = @min(cutoff_a, av[row * f.pools + pa]);
            cutoff_b = @min(cutoff_b, bv[row * f.pools + pb]);
        }
        for (seen) |flags| report.pool_overlap[row] += @intFromBool(flags == 3);
        for (0..f.pools) |pool| {
            const index = row * f.pools + pool;
            const x = av[index];
            const y = bv[index];
            if (pool >= complete) {
                report.future_score_mismatches += @intFromBool(x != -std.math.inf(f32) or y != x);
                continue;
            }
            try std.testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
            report.reference_cutoff_ties[row] += @intFromBool(x == cutoff_a);
            report.candidate_cutoff_ties[row] += @intFromBool(y == cutoff_b);
            const xb: u32 = @bitCast(x);
            const yb: u32 = @bitCast(y);
            report.score_bit_mismatches += @intFromBool(xb != yb);
            report.bf16_boundary_failures += @intFromBool((xb & 0xffff) != 0 or (yb & 0xffff) != 0);
            const delta: f64 = @as(f64, x) - @as(f64, y);
            square += delta * delta;
            norm += @as(f64, x) * @as(f64, x);
            report.max_abs = @max(report.max_abs, @abs(delta));
        }
        // Expansion must remain unique, causal, and have the exact partial tail.
        const token_seen = try std.testing.allocator.alloc(bool, f.pools * 4);
        defer std.testing.allocator.free(token_seen);
        for ([_]Arr{ ea, eb }) |tokens| {
            @memset(token_seen, false);
            const ids = mlx.mlx_array_data_int32(tokens).?;
            var live: usize = 0;
            for (ids[row * 2051 ..][0..2051]) |id| {
                if (id < 0) continue;
                try std.testing.expect(id <= f.offset + row);
                try std.testing.expect(!token_seen[@intCast(id)]);
                token_seen[@intCast(id)] = true;
                live += 1;
            }
            try std.testing.expectEqual(@as(usize, 2048) + (f.offset + row + 1) % 4, live);
        }
    }
    report.relative_l2 = @sqrt(square / norm);
    try std.testing.expectEqual(@as(usize, 0), report.future_score_mismatches);
    try std.testing.expectEqual(@as(usize, 0), report.bf16_boundary_failures);
    return report;
}
fn timed(f: *const Fixture, nax: bool, selection: bool) !u64 {
    const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
    var ops = Ops{ .s = f.ops.s };
    errdefer ops.deinit();
    var out = try f.scores(&ops, nax);
    if (selection) out = try expand(&ops, try score.topPools(&ops, out), f.offset);
    try mlx.check(mlx.mlx_array_eval(out));
    ops.deinit();
    return watch.read();
}
test "GLM IndexPool NAX score and full selector component probe" {
    const path = std.c.getenv("SUSHI_GLM_INDEX_NAX_PROBE_OUT") orelse return error.SkipZigTest;
    var reports: [3]Report = undefined;
    for ([_]usize{ 4096, 16384, 32768 }, 0..) |history, i| {
        var fixture = try Fixture.init(history, mlx.gpuStream());
        defer fixture.deinit();
        reports[i] = try audit(&fixture);
        for (0..2) |_| for ([_]bool{ false, true }) |selection| for ([_]bool{ false, true }) |nax| {
            _ = try timed(&fixture, nax, selection);
        };
        for (0..samples) |round| for (0..4) |position| {
            const arm = if (round % 2 == 0) position else 3 - position;
            const ns = try timed(&fixture, arm % 2 == 1, arm >= 2);
            switch (arm) {
                0 => reports[i].scalar_score_ns[round] = ns,
                1 => reports[i].nax_score_ns[round] = ns,
                2 => reports[i].scalar_select_ns[round] = ns,
                3 => reports[i].nax_select_ns[round] = ns,
                else => unreachable,
            }
        };
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .reports = reports, .dtype = "BF16 inputs/dot/product/finalscore boundaries; FP32 dotacc/headsum", .selection = "existing negative/argpartition512 and expand logic", .timing = "fresh scopes; tiled dots/epilogue/eval/free included; fullselector adds negative/partition/slice/expand", .runtime_hook = false }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(path), .data = json });
}
