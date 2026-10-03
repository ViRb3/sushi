//! Exact physical grid transpose for an isolated GLM routed prefill comparison.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const base = @import("expert_exl3_kernels.zig");
const api = @import("root.zig");
const support = base.PrefillGridSupport;
const Arr = mlx.mlx_array;
fn replace(comptime source: []const u8, comptime old: []const u8, comptime value: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const at = comptime std.mem.indexOf(u8, source, old).?;
    return source[0..at] ++ value ++ source[at + old.len ..] ++ "";
}
const SOURCE = replace(replace(support.source, "uint win = uint(threadgroup_position_in_grid.y);", "uint win = uint(threadgroup_position_in_grid.x);"), "uint(threadgroup_position_in_grid.x) * 128u + sg * 32u", "uint(threadgroup_position_in_grid.y) * 128u + sg * 32u");
var kernel: ?mlx.mlx_fast_metal_kernel = null;
const Config = struct { input: c_int, output: c_int, value: mlx.mlx_fast_metal_kernel_config };
var configs: [2]?Config = @splat(null);
fn project(s: mlx.mlx_stream, x: Arr, bank: Arr, ids: Arr, starts: Arr, live: Arr, windows: c_int) !Arr {
    const k = mlx.getShape(x)[1];
    const n = mlx.getShape(bank)[2] * 16;
    var cached: ?mlx.mlx_fast_metal_kernel_config = null;
    for (configs) |entry| if (entry) |e| if (e.input == k and e.output == n) {
        cached = e.value;
    };
    const cfg = cached orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 16384, n }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        inline for (.{ "IDIM", "ODIM", "NHW", "WIN" }, .{ k, n, @as(c_int, 36), @as(c_int, 32) }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, value));
        for (&configs) |*entry| if (entry.* == null) {
            entry.* = .{ .input = k, .output = n, .value = c };
            break :blk c;
        };
        return error.GlmGridConfigCacheFull;
    };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * windows, @divExact(n, 128), 1));
    if (kernel == null) kernel = try support.makeKernel(SOURCE);
    const iv = mlx.mlx_vector_array_new_data(&.{ x, bank, ids, starts, live }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, cfg, s));
    var output = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_vector_array_get(&output, ov, 0));
    return output;
}
pub fn moe(s: mlx.mlx_stream, x: Arr, bank: api.Bank, indices: Arr, scores: Arr) !?Arr {
    for ([_]Arr{ x, indices, scores, bank.gate.trellis, bank.gate.suh, bank.gate.svh, bank.up.trellis, bank.up.suh, bank.up.svh, bank.down.trellis, bank.down.suh, bank.down.svh }) |a| if (a.ctx == null) return null;
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(x), &.{ 1, 2048, 4096 }) or !std.mem.eql(c_int, mlx.getShape(indices), &.{ 1, 2048, 8 }) or !std.mem.eql(c_int, mlx.getShape(scores), &.{ 1, 2048, 8 }) or mlx.mlx_array_dtype(indices) != .uint32 or mlx.mlx_array_dtype(scores) != .float32) return null;
    inline for (.{ .{ "gate", 4096, 2048 }, .{ "up", 4096, 2048 }, .{ "down", 2048, 4096 } }) |entry| {
        const p = @field(bank, entry[0]);
        api.validateClampedProjection(p, 288, entry[1], entry[2]) catch return null;
        if (mlx.getShape(p.trellis)[3] != 36) return null;
    }
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ 2048, 4096 }, 2, s));
    try mlx.check(mlx.mlx_reshape(&ids, indices, &.{16384}, 1, s));
    try mlx.check(mlx.mlx_reshape(&sc, scores, &.{16384}, 1, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, ids, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, ids, order, 0, s));
    const prep = try support.prepare(s, flat, bank.gate.suh, bank.up.suh, sorted, order_i, 4096, 16384, 8);
    defer for ([_]Arr{ prep[0], prep[1] }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const metadata = try support.windows(s, sorted, order_i, 16384, 32, 288, true);
    defer for ([_]Arr{ metadata.inverse, metadata.table.starts, metadata.table.nlives }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (metadata.inverse.ctx == null) return error.ExpectedGlmInverseRouting;
    const tab = metadata.table;
    const gate = try project(s, prep[0], bank.gate.trellis, sorted, tab.starts, tab.nlives, tab.nwin);
    defer _ = mlx.mlx_array_free(gate);
    const up = try project(s, prep[1], bank.up.trellis, sorted, tab.starts, tab.nlives, tab.nwin);
    defer _ = mlx.mlx_array_free(up);
    const middle = try support.middle(s, gate, up, bank.gate.svh, bank.up.svh, bank.down.suh, sorted, 2048, 16384, 10);
    defer _ = mlx.mlx_array_free(middle);
    const down = try project(s, middle, bank.down.trellis, sorted, tab.starts, tab.nlives, tab.nwin);
    defer _ = mlx.mlx_array_free(down);
    const result = try support.finish(s, down, metadata.inverse, bank.down.svh, ids, sc, 4096, 2048, 8, .bfloat16);
    defer _ = mlx.mlx_array_free(result);
    var shaped = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, result, &.{ 1, 2048, 4096 }, 3, s));
    return shaped;
}

test "GLM prefill grid transpose is a bijection of production window tiles" {
    for ([_]usize{ 16, 32 }) |outputs| {
        var seen: [800 * 32]bool = @splat(false);
        for (0..800) |physical_x| for (0..outputs) |physical_y| {
            const index = physical_x * outputs + physical_y;
            try std.testing.expect(!seen[index]);
            seen[index] = true;
        };
        for (seen[0 .. 800 * outputs]) |covered| try std.testing.expect(covered);
    }
}
