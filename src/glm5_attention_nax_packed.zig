//! Sparse head-packed native NAX SDPA with bounded query batches.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const scratch_limit: usize = 64 * 1024 * 1024;
pub const max_rows: usize = 16;
pub const wide_rows: usize = 32;
pub const wide_scratch_limit: usize = 128 * 1024 * 1024;
threadlocal var batch32_override: ?bool = null;
pub const Batch32Binding = struct {
    previous: ?bool,
    pub fn restore(self: Batch32Binding) void {
        batch32_override = self.previous;
    }
};
pub fn bind32(on: bool) Batch32Binding {
    const old = Batch32Binding{ .previous = batch32_override };
    batch32_override = on;
    return old;
}
pub fn batch32Enabled() bool {
    return batch32_override orelse @import("transformer.zig").diagEnvOn("SUSHI_GLM_PREFILL_PACKED32");
}
pub fn batchRows() usize {
    return if (batch32Enabled()) wide_rows else max_rows;
}
pub fn scratchLimit() usize {
    return if (batch32Enabled()) wide_scratch_limit else scratch_limit;
}
var wide_calls: usize = 0;
pub fn wideDispatchCount() usize {
    return wide_calls;
}
threadlocal var enabled_override: ?bool = null;
var calls: usize = 0;
pub const Binding = struct {
    previous: ?bool,
    pub fn restore(self: Binding) void {
        enabled_override = self.previous;
    }
};
pub fn bind(on: bool) Binding {
    const result = Binding{ .previous = enabled_override };
    enabled_override = on;
    return result;
}
pub fn enabled() bool {
    if (enabled_override) |on| return on;
    return @import("transformer.zig").diagEnvOn("SUSHI_GLM_ATTENTION_PACKED");
}
pub fn resetDispatchCount() void {
    calls = 0;
    wide_calls = 0;
}
pub fn dispatchCount() usize {
    return calls;
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!enabled() or chunk <= 8) return 0;
    return std.math.mul(usize, scratchLimit(), pending_layers);
}
pub fn temporaryBytes(rows: usize) !usize {
    if (rows == 0 or (rows > max_rows and rows != wide_rows)) return error.UnsupportedGlmPackedAttention;
    // One contiguous gathered BF16 KV bank; native K/V share it without copies.
    // Query,
    // result, result-zeroing and native contiguity copies; expanded bool mask,
    // predicate/index bookkeeping. No [Q,H,K] floating score tensor.
    return std.math.mul(usize, rows, 2051 * 512 * 2 + 64 * 512 * 2 * 4 + 64 * 2051 + 2051 * 9 + 16);
}
const gather_source: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*2051u*128u) return;
    \\const uint slot=i/128u,d=(i%128u)*4u,row=slot/2051u;
    \\const int id=selected[slot];
    \\const bool valid=id>=0 && uint(id)<uint(length) && uint(id)<=uint(offset)+row;
    \\if(d==0u) mask[slot]=valid;
    \\if(!valid) {for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=OutT(0);return;}
    \\for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=cache[size_t(uint(id))*512u+d+j];
;
var gather_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gather_configs: [wide_rows]?mlx.mlx_fast_metal_kernel_config = @splat(null);
fn gather(ops: *Ops, cache: Arr, ids: Arr, offset: usize, history: usize) !struct { kv: Arr, mask: Arr } {
    const rows = mlx.getShape(ids)[0];
    const index: usize = @intCast(rows - 1);
    const cfg = gather_configs[index] orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, 2051, 512 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, 2051 }, 2, .bool_));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, rows * 2051 * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", .bfloat16));
        gather_configs[index] = c;
        break :blk c;
    };
    if (gather_kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "cache", "selected", "offset", "length" }, 4);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "kv", "mask" }, 2);
        defer _ = mlx.mlx_vector_string_free(outs);
        // Strides are explicitly admitted below; disable the wrapper's flag-
        // based copies so false row-contiguous flags cannot copy full history.
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_sparse_head_gather", ins, outs, gather_source, "", false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        gather_kernel = k;
    }
    const off: u32 = @intCast(offset);
    const len: u32 = @intCast(history);
    const oa = try ops.own(mlx.mlx_array_new_data(&off, &.{}, 0, .uint32));
    const la = try ops.own(mlx.mlx_array_new_data(&len, &.{}, 0, .uint32));
    const iv = mlx.mlx_vector_array_new_data(&.{ cache, ids, oa, la }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, gather_kernel.?, iv, cfg, ops.s));
    if (mlx.mlx_vector_array_size(outputs) != 2) return error.MetalKernelBadOutputCount;
    const kv = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(kv, outputs, 0));
    const mask = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(mask, outputs, 1));
    return .{ .kv = kv.*, .mask = mask.* };
}
fn geometry(q: []const c_int, cache: []const c_int, ids: []const c_int, offset: usize, history: usize) bool {
    return q.len == 3 and q[0] > 0 and (q[0] <= max_rows or (q[0] == wide_rows and batch32Enabled())) and q[1] == 64 and q[2] == 512 and
        cache.len == 2 and cache[0] >= history and cache[1] == 512 and ids.len == 2 and ids[0] == q[0] and ids[1] == 2051 and
        history > 0 and history <= 1048576 and offset <= history and q[0] <= history - offset;
}
/// Valid IDs must be unique per real query, as guaranteed by IndexPool selection.
pub fn run(ops: *Ops, q: Arr, cache: Arr, selected: Arr, offset: usize, history: usize, scale: f32) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_kda_fused.zig").hardwareSupported() or !std.math.isFinite(scale) or scale <= 0) return null;
    for ([_]Arr{ q, cache, selected }) |a| if (a.ctx == null) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(cache) != .bfloat16 or mlx.mlx_array_dtype(selected) != .int32 or
        !geometry(mlx.getShape(q), mlx.getShape(cache), mlx.getShape(selected), offset, history)) return null;
    // Refuse a cache that would require the generic gather wrapper to copy
    // full history; actual State latent buffers have these contiguous strides.
    const cs = mlx.mlx_array_strides(cache);
    if (cs[0] != 512 or cs[1] != 1) return null;
    const rows = mlx.getShape(q)[0];
    if (try temporaryBytes(@intCast(rows)) > scratchLimit()) return error.GlmPackedAttentionScratchBudget;
    const is = mlx.mlx_array_strides(selected);
    const ids = if (is[0] == 2051 and is[1] == 1) selected else try ops.contiguous(selected);
    const bank = try gather(ops, cache, ids, offset, history);
    const queries = try ops.reshape(q, &.{ rows, 1, 64, 512 });
    const kv = try ops.reshape(bank.kv, &.{ rows, 1, 2051, 512 });
    const mask = try ops.reshape(bank.mask, &.{ rows, 1, 1, 2051 });
    const out = try ops.slot();
    // Fake Q positions are original heads; array mode has no position bias.
    // Q64/GQA1 selects native full D512 NAX, and force_fused forbids fallback.
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(out, queries, kv, kv, scale, "array", mask, .{ .ctx = null }, true, ops.s));
    const populated = try ops.slot();
    try mlx.check(mlx.mlx_any_axes(populated, bank.mask, &.{1}, 1, true, ops.s));
    const safe = try ops.slot();
    try mlx.check(mlx.mlx_where(safe, try ops.reshape(populated.*, &.{ rows, 1, 1, 1 }), out.*, try ops.scalar(0, .bfloat16), ops.s));
    const result = try ops.reshape(safe.*, &.{ rows, 64, 512 });
    calls += 1;
    if (rows == wide_rows) wide_calls += 1;
    return result;
}
test "GLM packed NAX geometry and fixed scratch bound" {
    try std.testing.expect(geometry(&.{ 16, 64, 512 }, &.{ 32768, 512 }, &.{ 16, 2051 }, 32752, 32768));
    try std.testing.expect(!geometry(&.{ 17, 64, 512 }, &.{ 32768, 512 }, &.{ 17, 2051 }, 32751, 32768));
    try std.testing.expect(!geometry(&.{ 16, 64, 512 }, &.{ 32768, 512 }, &.{ 16, 2051 }, 32753, 32768));
    try std.testing.expect(try temporaryBytes(16) <= scratch_limit);
}
test "GLM packed NAX scoped controls bill every pending layer" {
    const off = bind(false);
    defer off.restore();
    try std.testing.expect(!enabled());
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(2048, 2));
    {
        const on = bind(true);
        defer on.restore();
        try std.testing.expect(enabled());
        try std.testing.expectEqual(scratch_limit, try transientBudget(16, 1));
        try std.testing.expectEqual(scratch_limit * 2, try transientBudget(2048, 2));
        try std.testing.expectEqual(@as(usize, 0), try transientBudget(8, 2));
        try std.testing.expectError(error.Overflow, transientBudget(2048, std.math.maxInt(usize)));
    }
    try std.testing.expect(!enabled());
}
test "GLM packed NAX invalid gather cannot poison output with masked nonfinite key zero" {
    if (std.c.getenv("SUSHI_GLM_PACKED_NAX_PROBE_OUT") == null) return error.SkipZigTest;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var data: [4 * 512]u16 = @splat(0);
    for (0..512) |i| {
        data[i] = if (i % 2 == 0) 0x7fc0 else 0x7f80;
        data[512 + i] = 0x3e80; // exact BF16 0.25
    }
    const cache = try ops.own(mlx.mlx_array_new_data(&data, &.{ 4, 512 }, 2, .bfloat16));
    var indices: [2 * 2051]i32 = @splat(-1);
    indices[0] = std.math.minInt(i32);
    indices[1] = std.math.maxInt(i32);
    indices[2] = 3; // future at real query position1
    indices[2051 + 100] = 1;
    const selected = try ops.own(mlx.mlx_array_new_data(&indices, &.{ 2, 2051 }, 2, .int32));
    const q = try ops.zeros(&.{ 2, 64, 512 }, .bfloat16);
    const out = (try run(&ops, q, cache, selected, 1, 4, 1.0 / 16.0)) orelse return error.ExpectedPackedNax;
    try mlx.check(mlx.mlx_array_eval(out));
    const bits = mlx.mlx_array_data_bfloat16(out).?;
    for (0..64 * 512) |i| {
        try std.testing.expectEqual(@as(u16, 0), bits[i]);
        try std.testing.expectEqual(@as(u16, 0x3e80), bits[64 * 512 + i]);
    }
}
test "GLM packed NAX preserves head order real-row selection and final ragged slot" {
    if (std.c.getenv("SUSHI_GLM_PACKED_NAX_PROBE_OUT") == null) return error.SkipZigTest;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var data: [4 * 512]u16 = @splat(0);
    data[0] = 0x3f80; // BF16 +1
    data[512] = 0xbf80; // BF16 -1
    const cache = try ops.own(mlx.mlx_array_new_data(&data, &.{ 4, 512 }, 2, .bfloat16));
    var query: [2 * 64 * 512]u16 = @splat(0);
    for (0..2) |row| for (0..64) |head| {
        query[(row * 64 + head) * 512] = if (head % 2 == 0) 0x4180 else 0xc180; // +/-16
    };
    const q = try ops.own(mlx.mlx_array_new_data(&query, &.{ 2, 64, 512 }, 3, .bfloat16));
    var indices: [2 * 2051]i32 = @splat(-1);
    indices[0] = 0;
    indices[2050] = 1; // last ragged K position must participate
    indices[2051 + 10] = 0; // a different real query has a different key set
    const selected = try ops.own(mlx.mlx_array_new_data(&indices, &.{ 2, 2051 }, 2, .int32));
    const out = (try run(&ops, q, cache, selected, 1, 4, 1.0 / 16.0)) orelse return error.ExpectedPackedNax;
    try mlx.check(mlx.mlx_array_eval(out));
    const bits = mlx.mlx_array_data_bfloat16(out).?;
    for (0..64) |head| {
        const value: f32 = @bitCast(@as(u32, bits[head * 512]) << 16);
        if (head % 2 == 0) try std.testing.expect(value > 0.7 and value < 0.8) else try std.testing.expect(value < -0.7 and value > -0.8);
        try std.testing.expectEqual(@as(u16, 0x3f80), bits[(64 + head) * 512]);
        for (1..512) |d| {
            try std.testing.expectEqual(@as(u16, 0), bits[head * 512 + d]);
            try std.testing.expectEqual(@as(u16, 0), bits[(64 + head) * 512 + d]);
        }
    }
}

test "GLM packed32 fixed policy geometry and conservative graph bill" {
    const off = bind32(false);
    defer off.restore();
    try std.testing.expectEqual(@as(usize, 16), batchRows());
    try std.testing.expectEqual(scratch_limit, scratchLimit());
    try std.testing.expect(!geometry(&.{ 32, 64, 512 }, &.{ 16384, 512 }, &.{ 32, 2051 }, 16352, 16384));
    const on = bind32(true);
    try std.testing.expect(batch32Enabled());
    try std.testing.expectEqual(@as(usize, 32), batchRows());
    try std.testing.expectEqual(wide_scratch_limit, scratchLimit());
    try std.testing.expect(geometry(&.{ 32, 64, 512 }, &.{ 16384, 512 }, &.{ 32, 2051 }, 16352, 16384));
    for ([_]c_int{ 17, 24, 31 }) |rows| try std.testing.expect(!geometry(&.{ rows, 64, 512 }, &.{ 16384, 512 }, &.{ rows, 2051 }, 16352, 16384));
    try std.testing.expect(!geometry(&.{ 32, 64, 512 }, &.{ 16384, 512 }, &.{ 32, 2051 }, 16353, 16384));
    try std.testing.expect(try temporaryBytes(32) <= wide_scratch_limit);
    for ([_]usize{ 17, 24, 31 }) |rows| try std.testing.expectError(error.UnsupportedGlmPackedAttention, temporaryBytes(rows));
    on.restore();
    try std.testing.expect(!batch32Enabled());
}
