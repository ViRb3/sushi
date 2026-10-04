//! Mode-matched B1/B3 native decode attention: FP32 GEMMs over one gathered bank on every GPU.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const latent_store = @import("glm5_latent.zig");
const packed_attention = @import("glm5_attention_nax_packed.zig");
const Latent = latent_store.Latent;
const Arr = mlx.mlx_array;
var b1_calls: usize = 0;
var b3_calls: usize = 0;
pub fn enabled() bool {
    return !@import("glm5_model.zig").reference_numerics;
}
/// Per pending layer: the FP32 composite's three-row planes.
pub fn scratchLimit() usize {
    return 32 * 1024 * 1024;
}
pub fn supportedConfig(cfg: *const @import("model.zig").ModelConfig, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) bool {
    return cfg.num_attention_heads == 64 and cfg.mla_kv_lora_rank == 512 and cfg.mla_qk_nope_head_dim == 256 and
        cfg.indexer_n_heads == 32 and cfg.indexer_head_dim == 128 and dtype == .bfloat16 and mlx.streamIsGpu(s);
}
pub fn supportedQuery(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) bool {
    return shape.len == 3 and shape[0] > 0 and shape[0] <= 8 and shape[1] == 64 and shape[2] == 512 and
        dtype == .bfloat16 and mlx.streamIsGpu(s);
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
/// Ancestry rows a branch may read from the verification tape (the gather's `paths` stride).
pub const max_tail = 4;
pub const Branch = struct {
    offset: usize,
    length: usize,
    path: [max_tail]u32,
};
pub fn temporaryBytes(batch: usize) !usize {
    if (batch != 1 and batch != 3) return error.UnsupportedGlmDecodeBatch;
    return packed_attention.compositeBytes(batch);
}
pub fn transientBudget(pending_layers: usize) !usize {
    if (!enabled()) return 0;
    return std.math.mul(usize, scratchLimit(), pending_layers);
}
fn geometry(q: []const c_int, prefix: []const c_int, prefix_rows: usize, tape: []const c_int, ids: []const c_int, branches: []const Branch) bool {
    if (q.len != 3 or (q[0] != 1 and q[0] != 3) or q[1] != 64 or q[2] != 512 or branches.len != q[0] or
        prefix.len != 2 or prefix[0] < prefix_rows or prefix[1] != 512 or prefix_rows > 1048576 or
        tape.len != 2 or tape[0] < 1 or tape[0] > max_tail or tape[1] != 512 or
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
    \\if(uint(id)<uint(prefix_length)) {for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=SUSHI_LATENT(prefix,uint(id),d+j,512u);return;}
    \\const device OutT* values=tape+size_t(paths[row*4u+uint(id)-uint(prefix_length)])*512u;
    \\for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=values[d+j];
;
var gather_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var configs: [2]?mlx.mlx_fast_metal_kernel_config = @splat(null);
pub fn run(ops: *Ops, q: Arr, prefix: Latent, prefix_rows: usize, tape: Arr, branches: []const Branch, selected: Arr, scale: f32) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !std.math.isFinite(scale) or scale <= 0) return null;
    for ([_]Arr{ q, prefix.data, tape, selected }) |a| if (a.ctx == null) return null;
    if (!prefix.rowMajor() or prefix.dtype() != .bfloat16) return null;
    for ([_]Arr{ q, tape }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(selected) != .int32 or !geometry(mlx.getShape(q), &.{ prefix.rows(), prefix.width() }, prefix_rows, mlx.getShape(tape), mlx.getShape(selected), branches)) return null;
    const ts = mlx.mlx_array_strides(tape);
    if (ts[0] != 512 or ts[1] != 1) return null;
    const qs = mlx.mlx_array_strides(q);
    if (qs[1] != 512 or qs[2] != 1) return null;
    const batch: c_int = @intCast(branches.len);
    if (try temporaryBytes(branches.len) > scratchLimit()) return error.GlmDecodeBatchScratchBudget;
    var offsets: [3]u32 = undefined;
    var lengths: [3]u32 = undefined;
    var paths: [3 * max_tail]u32 = undefined;
    for (branches, 0..) |branch, row| {
        offsets[row] = @intCast(branch.offset);
        lengths[row] = @intCast(branch.length);
        @memcpy(paths[row * max_tail ..][0..max_tail], &branch.path);
    }
    const oa = try ops.own(mlx.mlx_array_new_data(&offsets, &.{batch}, 1, .uint32));
    const la = try ops.own(mlx.mlx_array_new_data(&lengths, &.{batch}, 1, .uint32));
    const pa = try ops.own(mlx.mlx_array_new_data(&paths, &.{ batch, max_tail }, 2, .uint32));
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
    const quantized = prefix.quantized();
    const kernel = &gather_kernels[@intFromBool(quantized)];
    if (kernel.* == null) {
        const names: []const [*:0]const u8 = if (quantized)
            &.{ "prefix", "prefix_scales", "prefix_biases", "tape", "selected", "offsets", "lengths", "paths", "prefix_length", "batch_count" }
        else
            &.{ "prefix", "tape", "selected", "offsets", "lengths", "paths", "prefix_length", "batch_count" };
        const inputs = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&.{ "kv", "mask" }, 2);
        defer _ = mlx.mlx_vector_string_free(outputs);
        // Admitted strides avoid wrapper copies of the immutable full prefix.
        const k = mlx.mlx_fast_metal_kernel_new(if (quantized) "sushi_glm_decode_batch_gather8" else "sushi_glm_decode_batch_gather", inputs, outputs, gather_source, latent_store.header(quantized), false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel.* = k;
    }
    const inputs = if (quantized)
        mlx.mlx_vector_array_new_data(&.{ prefix.data, prefix.scales, prefix.biases, tape, ids, oa, la, pa, ba, batches_array }, 10)
    else
        mlx.mlx_vector_array_new_data(&.{ prefix.data, tape, ids, oa, la, pa, ba, batches_array }, 8);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.*.?, inputs, cfg, ops.s));
    const bank = try ops.slot();
    const mask = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(bank, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(mask, outputs, 1));
    // A NAX GPU runs FP32 GEMMs as TF32 by default; the block-masked GEMM keeps them FP32 there.
    const how: packed_attention.Gemm = if (@import("glm5_model.zig").naxArms()) .block_masked else .matmul;
    if (b1_calls + b3_calls == 0) @import("log.zig").info("[glm-attn] FP32 composite B1/B3 decode attention engaged ({s})\n", .{@tagName(how)});
    // Row by row: MLX picks GEMM tiles and split-K by batch size, and B3 must equal three B1.
    var rows: [3]Arr = undefined;
    for (rows[0..branches.len], 0..) |*row, r| {
        const at: c_int = @intCast(r);
        row.* = try packed_attention.compositeWith(ops, try ops.slice(q, 0, at, at + 1), try ops.slice(bank.*, 0, at, at + 1), try ops.slice(mask.*, 0, at, at + 1), scale, how);
    }
    const out = if (branches.len == 1) rows[0] else try ops.concat(rows[0..branches.len], 0);
    const populated = try ops.slot();
    try mlx.check(mlx.mlx_any_axes(populated, mask.*, &.{1}, 1, true, ops.s));
    const safe = try ops.slot();
    try mlx.check(mlx.mlx_where(safe, try ops.reshape(populated.*, &.{ batch, 1, 1, 1 }), out, try ops.scalar(0, .bfloat16), ops.s));
    const result = try ops.reshape(safe.*, &.{ batch, 64, 512 });
    if (batch == 1) b1_calls += 1 else b3_calls += 1;
    return result;
}
test "GLM decode batch geometry ancestry and conservative scratch" {
    const branches = [_]Branch{
        .{ .offset = 16381, .length = 16382, .path = .{ 0, 0, 0, 0 } },
        .{ .offset = 16382, .length = 16383, .path = .{ 0, 1, 0, 0 } },
        .{ .offset = 16382, .length = 16383, .path = .{ 0, 2, 0, 0 } },
    };
    try std.testing.expect(geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &branches));
    try std.testing.expect(!geometry(&.{ 0, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 0, 2051 }, &.{}));
    var bad = branches;
    bad[2].offset += 1;
    try std.testing.expect(!geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &bad));
    bad = branches;
    bad[2].path = .{ 0, 0, 0, 0 };
    try std.testing.expect(!geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &bad));
    try std.testing.expect(try temporaryBytes(3) <= scratchLimit());
    try std.testing.expectEqual(scratchLimit() * 4, try transientBudget(4));
    try std.testing.expectError(error.Overflow, transientBudget(std.math.maxInt(usize)));
    const model = @import("glm5_model.zig");
    model.reference_numerics = true;
    defer model.reference_numerics = false;
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(4));
}

test "GLM kv8 B1 and B3 decode gathers match BF16 over the round-tripped prefix" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const prefix_rows = 3000;
    var q = try latent_store.quantize(try ops.own(try latent_store.randomRows(prefix_rows, 512, 61, s)), s);
    defer q.deinit();
    const kv8 = Latent{ .data = q.q, .scales = q.scales, .biases = q.biases };
    const bf16 = Latent{ .data = try ops.own(try kv8.dense(0, prefix_rows, s)) };
    const tape = try ops.own(try latent_store.readable(try ops.own(try latent_store.randomRows(3, 512, 62, s)), latent_store.kv8_bits, s));
    const branches = [_]Branch{
        .{ .offset = prefix_rows, .length = prefix_rows + 1, .path = .{ 0, 0, 0, 0 } },
        .{ .offset = prefix_rows + 1, .length = prefix_rows + 2, .path = .{ 0, 1, 0, 0 } },
        .{ .offset = prefix_rows + 1, .length = prefix_rows + 2, .path = .{ 0, 2, 0, 0 } },
    };
    var ids: [3 * 2051]i32 = undefined;
    for (branches, 0..) |branch, row| for (0..2051) |k| {
        ids[row * 2051 + k] = @intCast(branch.length - 2051 + k);
    };
    const selected = try ops.own(mlx.mlx_array_new_data(&ids, &.{ 3, 2051 }, 2, .int32));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 63));
    const query = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(query, &.{ 3, 64, 512 }, 3, .bfloat16, 0, 0.05, key, s));
    resetCalls();
    const got = (try run(&ops, query.*, kv8, prefix_rows, tape, &branches, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    const want = (try run(&ops, query.*, bf16, prefix_rows, tape, &branches, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    try @import("glm5_attention.zig").expectSameBits(want, got);
    const one = try ops.slice(query.*, 0, 0, 1);
    const one_ids = try ops.slice(selected, 0, 0, 1);
    const got1 = (try run(&ops, one, kv8, prefix_rows, tape, branches[0..1], one_ids, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    const want1 = (try run(&ops, one, bf16, prefix_rows, tape, branches[0..1], one_ids, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    try @import("glm5_attention.zig").expectSameBits(want1, got1);
    try std.testing.expectEqual(@as(usize, 2), b3Calls());
    try std.testing.expectEqual(@as(usize, 2), b1Calls());
}

test "GLM B1 reads a four-row ancestry tail exactly as those rows committed to the prefix" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const prefix_rows = 3000;
    const rows = try ops.own(try latent_store.randomRows(prefix_rows + 4, 512, 71, s));
    const committed = Latent{ .data = rows };
    const prefix = Latent{ .data = try ops.slice(rows, 0, 0, prefix_rows) };
    const tail4 = try ops.contiguous(try ops.slice(rows, 0, prefix_rows, prefix_rows + 4));
    const tail1 = try ops.contiguous(try ops.slice(rows, 0, prefix_rows + 3, prefix_rows + 4));
    const length = prefix_rows + 4;
    var ids: [2051]i32 = undefined;
    for (&ids, 0..) |*id, k| id.* = @intCast(length - 2051 + k);
    const selected = try ops.own(mlx.mlx_array_new_data(&ids, &.{ 1, 2051 }, 2, .int32));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 72));
    const query = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(query, &.{ 1, 64, 512 }, 3, .bfloat16, 0, 0.05, key, s));
    const deep = Branch{ .offset = length - 1, .length = length, .path = .{ 0, 1, 2, 3 } };
    const flat = Branch{ .offset = length - 1, .length = length, .path = .{ 0, 0, 0, 0 } };
    const got = (try run(&ops, query.*, prefix, prefix_rows, tail4, &.{deep}, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    const want = (try run(&ops, query.*, committed, prefix_rows + 3, tail1, &.{flat}, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    try @import("glm5_attention.zig").expectSameBits(want, got);
}
