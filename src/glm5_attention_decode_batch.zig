//! Opt-in mode-matched B1/B3 native decode attention.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const scratch_limit: usize = 8 * 1024 * 1024;
threadlocal var mode_override: ?bool = null;
var b1_calls: usize = 0;
var b3_calls: usize = 0;
pub const Binding = struct {
    previous: ?bool,
    pub fn restore(self: Binding) void {
        mode_override = self.previous;
    }
};
pub fn bind(on: bool) Binding {
    const result = Binding{ .previous = mode_override };
    mode_override = on;
    return result;
}
pub fn enabled() bool {
    return mode_override orelse @import("transformer.zig").diagEnvOn("SUSHI_GLM_DECODE_BATCH");
}
pub fn b1Calls() usize {
    return b1_calls;
}
pub fn b3Calls() usize {
    return b3_calls;
}
pub fn resetCalls() void {
    b1_calls = 0;
    b3_calls = 0;
}
pub fn admit(cfg: *const @import("model.zig").ModelConfig, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    if (!enabled()) return;
    if (cfg.num_attention_heads != 64 or cfg.mla_kv_lora_rank != 512 or cfg.mla_qk_nope_head_dim != 256 or
        cfg.indexer_n_heads != 32 or cfg.indexer_head_dim != 128 or dtype != .bfloat16 or
        !mlx.streamIsGpu(s) or !@import("glm5_kda_fused.zig").hardwareSupported()) return error.GlmDecodeNativeUnsupported;
}
pub const Branch = struct {
    offset: usize,
    length: usize,
    path: [3]u32,
};
pub fn temporaryBytes(batch: usize) !usize {
    if (batch != 1 and batch != 3) return error.UnsupportedGlmDecodeBatch;
    return @import("glm5_attention_nax_packed.zig").temporaryBytes(batch);
}
pub fn transientBudget(pending_layers: usize) !usize {
    if (!enabled()) return 0;
    return std.math.mul(usize, scratch_limit, pending_layers);
}
fn geometry(q: []const c_int, prefix: []const c_int, prefix_rows: usize, tape: []const c_int, ids: []const c_int, branches: []const Branch) bool {
    if (q.len != 3 or (q[0] != 1 and q[0] != 3) or q[1] != 64 or q[2] != 512 or branches.len != q[0] or
        prefix.len != 2 or prefix[0] < prefix_rows or prefix[1] != 512 or prefix_rows > 1048576 or
        tape.len != 2 or tape[0] < 1 or tape[0] > 3 or tape[1] != 512 or
        ids.len != 2 or ids[0] != q[0] or ids[1] != 2051) return false;
    for (branches) |branch| {
        if (branch.length <= prefix_rows or branch.length - prefix_rows > tape[0] or branch.offset != branch.length - 1) return false;
        const depth = branch.length - prefix_rows;
        for (branch.path[0..depth], 0..) |node, i| {
            if (node >= tape[0]) return false;
            for (branch.path[0..i]) |prior| if (node == prior) return false;
        }
    }
    return true;
}
const gather_source: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(batch_count)*2051u*128u) return;
    \\const uint slot=i/128u,d=(i%128u)*4u,row=slot/2051u;
    \\const int id=selected[slot];
    \\const bool valid=id>=0 && uint(id)<lengths[row] && uint(id)<=offsets[row];
    \\if(d==0u) mask[slot]=valid;
    \\if(!valid) {for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=OutT(0);return;}
    \\const device OutT* values=uint(id)<uint(prefix_length)?prefix+size_t(uint(id))*512u:
    \\ tape+size_t(paths[row*3u+uint(id)-uint(prefix_length)])*512u;
    \\for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=values[d+j];
;
var gather_kernel: ?mlx.mlx_fast_metal_kernel = null;
var configs: [2]?mlx.mlx_fast_metal_kernel_config = @splat(null);
pub fn run(ops: *Ops, q: Arr, prefix: Arr, prefix_rows: usize, tape: Arr, branches: []const Branch, selected: Arr, scale: f32) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_kda_fused.zig").hardwareSupported() or !std.math.isFinite(scale) or scale <= 0) return null;
    for ([_]Arr{ q, prefix, tape, selected }) |a| if (a.ctx == null) return null;
    for ([_]Arr{ q, prefix, tape }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(selected) != .int32 or !geometry(mlx.getShape(q), mlx.getShape(prefix), prefix_rows, mlx.getShape(tape), mlx.getShape(selected), branches)) return null;
    for ([_]Arr{ prefix, tape }) |a| {
        const strides = mlx.mlx_array_strides(a);
        if (strides[0] != 512 or strides[1] != 1) return null;
    }
    const qs = mlx.mlx_array_strides(q);
    if (qs[1] != 512 or qs[2] != 1) return null;
    const batch: c_int = @intCast(branches.len);
    if (try temporaryBytes(branches.len) > scratch_limit) return error.GlmDecodeBatchScratchBudget;
    var offsets: [3]u32 = undefined;
    var lengths: [3]u32 = undefined;
    var paths: [9]u32 = undefined;
    for (branches, 0..) |branch, row| {
        offsets[row] = @intCast(branch.offset);
        lengths[row] = @intCast(branch.length);
        @memcpy(paths[row * 3 ..][0..3], &branch.path);
    }
    const oa = try ops.own(mlx.mlx_array_new_data(&offsets, &.{batch}, 1, .uint32));
    const la = try ops.own(mlx.mlx_array_new_data(&lengths, &.{batch}, 1, .uint32));
    const pa = try ops.own(mlx.mlx_array_new_data(&paths, &.{ batch, 3 }, 2, .uint32));
    const base: u32 = @intCast(prefix_rows);
    const ba = try ops.own(mlx.mlx_array_new_data(&base, &.{}, 0, .uint32));
    const batches: u32 = @intCast(branches.len);
    const batches_array = try ops.own(mlx.mlx_array_new_data(&batches, &.{}, 0, .uint32));
    const ids = try ops.contiguous(selected);
    const cfg_index: usize = if (batch == 1) 0 else 1;
    const cfg = configs[cfg_index] orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ batch, 2051, 512 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ batch, 2051 }, 2, .bool_));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, batch * 2051 * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", .bfloat16));
        configs[cfg_index] = c;
        break :blk c;
    };
    if (gather_kernel == null) {
        const inputs = mlx.mlx_vector_string_new_data(&.{ "prefix", "tape", "selected", "offsets", "lengths", "paths", "prefix_length", "batch_count" }, 8);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&.{ "kv", "mask" }, 2);
        defer _ = mlx.mlx_vector_string_free(outputs);
        // Admitted strides avoid wrapper copies of the immutable full prefix.
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_decode_batch_gather", inputs, outputs, gather_source, "", false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        gather_kernel = k;
    }
    const inputs = mlx.mlx_vector_array_new_data(&.{ prefix, tape, ids, oa, la, pa, ba, batches_array }, 8);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, gather_kernel.?, inputs, cfg, ops.s));
    const bank = try ops.slot();
    const mask = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(bank, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(mask, outputs, 1));
    const queries = try ops.reshape(try ops.contiguous(q), &.{ batch, 1, 64, 512 });
    const kv = try ops.reshape(bank.*, &.{ batch, 1, 2051, 512 });
    const masked = try ops.reshape(mask.*, &.{ batch, 1, 1, 2051 });
    const out = try ops.slot();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(out, queries, kv, kv, scale, "array", masked, .{ .ctx = null }, true, ops.s));
    const populated = try ops.slot();
    try mlx.check(mlx.mlx_any_axes(populated, mask.*, &.{1}, 1, true, ops.s));
    const safe = try ops.slot();
    try mlx.check(mlx.mlx_where(safe, try ops.reshape(populated.*, &.{ batch, 1, 1, 1 }), out.*, try ops.scalar(0, .bfloat16), ops.s));
    const result = try ops.reshape(safe.*, &.{ batch, 64, 512 });
    if (batch == 1) b1_calls += 1 else b3_calls += 1;
    return result;
}
test "GLM decode batch geometry ancestry and conservative scratch" {
    const branches = [_]Branch{
        .{ .offset = 16381, .length = 16382, .path = .{ 0, 0, 0 } },
        .{ .offset = 16382, .length = 16383, .path = .{ 0, 1, 0 } },
        .{ .offset = 16382, .length = 16383, .path = .{ 0, 2, 0 } },
    };
    try std.testing.expect(geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &branches));
    try std.testing.expect(!geometry(&.{ 0, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 0, 2051 }, &.{}));
    var bad = branches;
    bad[2].offset += 1;
    try std.testing.expect(!geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &bad));
    bad = branches;
    bad[2].path = .{ 0, 0, 0 };
    try std.testing.expect(!geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &bad));
    try std.testing.expect(try temporaryBytes(3) <= scratch_limit);
    const off = bind(false);
    defer off.restore();
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(4));
    {
        const on = bind(true);
        defer on.restore();
        try std.testing.expect(enabled());
        try std.testing.expectEqual(scratch_limit * 4, try transientBudget(4));
        try std.testing.expectError(error.Overflow, transientBudget(std.math.maxInt(usize)));
    }
    try std.testing.expect(!enabled());
}
