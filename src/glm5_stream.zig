//! Native GLM routed experts over Sushi's shared expert cache: BF16 source experts or EXL3 pack banks.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const stream = @import("expert_stream.zig");
const exl3 = @import("sushi_exl3");
const fp8_block = @import("fp8_block.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

pub const Budget = struct { total: u64, trunk: u64, reserve: u64, cache: u64, fixed: u64 };
fn bytesPerExpert(g: stream.Geometry) !u64 {
    return stream.expertBytes(try std.math.mul(u32, 2, g.intermediate), g.hidden, g.intermediate);
}

/// Retained lazy trunk metadata must fit this bound before tensor evaluation.
pub fn trunkLimit(cfg: *const model.ModelConfig, total: u64, reserve: u64) !u64 {
    const g = cfg.expertGeometry();
    const bytes = try bytesPerExpert(g);
    const base = try plan(total, 0, reserve, g, bytes);
    const minimum_cache = try std.math.mul(u64, g.layers - g.first_moe_layer, bytes);
    return total - base.fixed - minimum_cache;
}

pub fn plan(total: u64, trunk: u64, reserve: u64, geometry: stream.Geometry, expert_bytes: u64) !Budget {
    if (geometry.layers <= geometry.first_moe_layer or geometry.experts == 0 or expert_bytes == 0 or reserve == 0) return error.InvalidGlmStreamBudget;
    const layers: u64 = geometry.layers - geometry.first_moe_layer;
    const workspace = std.math.mul(u64, geometry.experts, expert_bytes) catch return error.InvalidGlmStreamBudget;
    // Covers page rounding, imported-array metadata and prepared trunk copies.
    const overhead = 64 * 1024 * 1024;
    var fixed = std.math.add(u64, trunk, reserve) catch return error.InvalidGlmStreamBudget;
    for ([_]u64{ workspace, stream.BOUNCE_BYTES, overhead }) |n| fixed = std.math.add(u64, fixed, n) catch return error.InvalidGlmStreamBudget;
    if (fixed >= total) return error.SsdBudgetBelowResident;
    const per_slot = std.math.mul(u64, layers, expert_bytes) catch return error.InvalidGlmStreamBudget;
    const slots = @min((total - fixed) / per_slot, geometry.experts);
    if (slots == 0) return error.SsdBudgetBelowResident;
    return .{ .total = total, .trunk = trunk, .reserve = reserve, .cache = slots * per_slot, .fixed = fixed };
}

test "GLM stream CPU budget refuses underfunding and includes every resident slab" {
    const t = std.testing;
    const g = stream.Geometry{ .layers = 4, .first_moe_layer = 3, .experts = 4, .hidden = 32, .intermediate = 16 };
    const b = try plan(1024 * 1024 * 1024, 100, 200, g, 3072);
    try t.expectEqual(@as(u64, 4 * 3072), b.cache);
    try t.expect(b.fixed >= 100 + 200 + stream.BOUNCE_BYTES + 4 * 3072);
    try t.expect(b.fixed + b.cache <= b.total);
    try t.expectError(error.SsdBudgetBelowResident, plan(100, 100, 200, g, 3072));
    try t.expectError(error.InvalidGlmStreamBudget, plan(std.math.maxInt(u64), std.math.maxInt(u64), 1, g, 3072));
}

test "GLM stream CPU trunk bound admits one slot before trunk materialization" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .num_hidden_layers = 4, .first_k_dense_replace = 3, .num_experts = 4, .hidden_size = 32, .moe_intermediate_size = 16 };
    const total = 1024 * 1024 * 1024;
    const reserve = 128 * 1024 * 1024;
    const max_trunk = try trunkLimit(&cfg, total, reserve);
    const g = stream.Geometry{ .layers = 4, .first_moe_layer = 3, .experts = 4, .hidden = 32, .intermediate = 16 };
    const b = try plan(total, max_trunk, reserve, g, 3072);
    try std.testing.expectEqual(@as(u64, 3072), b.cache);
    try std.testing.expectEqual(total, b.fixed + b.cache);
    try std.testing.expectError(error.SsdBudgetBelowResident, plan(total, max_trunk + 1, reserve, g, 3072));
}

/// Conservative one-request live tensor bound. Streamed forwards synchronize each
/// layer; speculative tapes and concurrent requests are outside this admission.
pub fn minimumReserve(cfg: *const model.ModelConfig, tokens: usize, chunk: usize) !u64 {
    if (tokens == 0 or tokens > cfg.max_position_embeddings or chunk == 0 or chunk > 512 or chunk > tokens or cfg.full_attention_interval == 0 or cfg.num_hidden_layers == 0) return error.InvalidGlmStreamBudget;
    const n: u128 = tokens;
    const t: u128 = chunk;
    const h: u128 = cfg.hidden_size;
    const heads: u128 = cfg.num_attention_heads;
    const attention_layers: u128 = cfg.num_hidden_layers / cfg.full_attention_interval;
    const linear_layers: u128 = cfg.num_hidden_layers - attention_layers;
    const d: u128 = cfg.linear_key_head_dim;
    const width: u128 = @as(u128, cfg.linear_num_value_heads) * d;
    const recurrent = linear_layers * (width * d * 4 + width * 3 * 3 * 2);
    // Capacity growth and update may retain both old and new arrays.
    const caches = 4 * n * attention_layers * (@as(u128, cfg.mla_kv_lora_rank) * 2 + @as(u128, cfg.indexer_head_dim) * 8);
    const activations = t * (h * 128 + width * 96 + @as(u128, cfg.vocab_size) * 8 + @as(u128, cfg.num_experts_per_tok) * (@as(u128, cfg.moe_intermediate_size) * 16 + h * 8));
    const attention_scratch = heads * t * n * 8 + t * heads * (@as(u128, cfg.mla_qk_nope_head_dim) + cfg.mla_v_head_dim) * 8;
    const prepared = attention_layers * heads * (@as(u128, cfg.mla_qk_nope_head_dim) + cfg.mla_v_head_dim) * @as(u128, cfg.mla_kv_lora_rank) * 2;
    return std.math.cast(u64, prepared + recurrent * 2 + caches + activations + attention_scratch + 64 * 1024 * 1024) orelse error.InvalidGlmStreamBudget;
}

/// A lossless teacher capture's BF16 expert budget: one request of `max_tokens`, chunks of at most 512.
pub fn captureBudget(cfg: *const model.ModelConfig, total: u64, trunk: u64, reserve: u64, max_tokens: usize, max_chunk: usize) !Budget {
    if (!cfg.isGlm5() or max_tokens == 0 or max_tokens > cfg.max_position_embeddings or max_chunk == 0 or max_chunk > max_tokens or max_chunk > 512) return error.InvalidGlmStreamBudget;
    if (reserve < try minimumReserve(cfg, max_tokens, max_chunk)) return error.GlmStreamReserveTooSmall;
    const g = cfg.expertGeometry();
    return plan(total, trunk, reserve, g, try bytesPerExpert(g));
}

/// The caller owns the engine. One inference thread may use a stream at a time;
/// requests own only their caches.
pub const Stream = struct {
    engine: *stream.Engine,
    max_tokens: usize,
    max_chunk: usize,
    request_owner: ?*const anyopaque = null,

    /// Serving admits each request itself; the stream bounds only the model's own context.
    pub fn serving(engine: *stream.Engine, cfg: *const model.ModelConfig) Stream {
        return .{ .engine = engine, .max_tokens = cfg.max_position_embeddings, .max_chunk = cfg.max_position_embeddings };
    }
    pub fn claim(self: *Stream, owner: *const anyopaque) !void {
        if (self.request_owner) |current| if (current != owner) return error.GlmStreamRequestBusy;
        self.request_owner = owner;
    }
    pub fn release(self: *Stream, owner: *const anyopaque) void {
        if (self.request_owner == owner) self.request_owner = null;
    }
    pub fn admit(self: *const Stream, offset: usize, rows: usize) !void {
        if (rows == 0 or rows > self.max_chunk or offset > self.max_tokens or rows > self.max_tokens - offset) return error.GlmStreamRequestBudgetExceeded;
    }
    /// Router ids reach the slabs as slot ids; scores, clamps and the reduction stay the resident path's.
    pub fn apply(self: *Stream, layer: u16, ops: *Ops, x: Arr, ids: Arr, scores: Arr, cfg: *const model.ModelConfig) !Arr {
        if (!mlx.streamIsGpu(ops.s)) return error.GlmStreamRequiresGpu;
        const shape = mlx.getShape(x);
        const ish = mlx.getShape(ids);
        if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > self.max_chunk or shape[2] != self.engine.geometry.hidden or mlx.mlx_array_dtype(x) != .bfloat16 or
            (mlx.mlx_array_dtype(ids) != .uint32 and mlx.mlx_array_dtype(ids) != .int32) or ish.len != 3 or ish[0] != 1 or ish[1] != shape[1] or ish[2] < 1 or ish[2] > self.engine.geometry.experts or !std.mem.eql(c_int, ish, mlx.getShape(scores)) or mlx.mlx_array_dtype(scores) != .float32 or cfg.glm_swiglu_limit != 10) return error.InvalidGlmStreamInput;
        var scope = Ops{ .s = ops.s };
        defer scope.deinit();
        const raw_ids = try scope.contiguous(try scope.cast(ids, .uint32));
        try mlx.check(mlx.mlx_array_eval(raw_ids));
        const count = mlx.mlx_array_size(raw_ids);
        const data = mlx.mlx_array_data_uint32(raw_ids) orelse return error.InvalidGlmStreamInput;
        const host = try self.engine.allocator.alloc(u16, count);
        defer self.engine.allocator.free(host);
        for (host, 0..) |*v, i| {
            if (data[i] >= self.engine.geometry.experts) return error.ExpertOutOfRange;
            v.* = @intCast(data[i]);
        }
        var prepared = try self.engine.prepareHost(layer, host);
        defer prepared.deinit();
        errdefer _ = mlx.mlx_synchronize(ops.s);
        const local = try self.engine.allocator.alloc(u32, count);
        defer self.engine.allocator.free(local);
        for (local, prepared.remapped) |*v, remap| v.* = remap;
        const remapped = try scope.own(mlx.mlx_array_new_data(local.ptr, ish.ptr, @intCast(ish.len), .uint32));
        const out = switch (self.engine.store.layout()) {
            .bf16_individual => try bf16Routed(&scope, x, prepared.gate, prepared.up, prepared.down, remapped, scores, cfg.glm_swiglu_limit),
            .exl3_k4 => try @import("glm5_forward.zig").routedExl3(&scope, x, exl3Bank(&prepared.quant_raw), remapped, scores, cfg),
            .fp8_individual => try fp8Routed(&scope, self.engine.allocator, x, &prepared.quant_raw, local, ish, scores, cfg.glm_swiglu_limit),
            else => return error.GlmStreamLayoutUnsupported,
        };
        // Complete every slab reader before another layer can refill the union.
        try mlx.check(mlx.mlx_array_eval(out));
        return ops.own(try scope.result(out));
    }
};

fn exl3Bank(raw: *const [stream.quant.component_count]Arr) exl3.Bank {
    const C = stream.quant.Component;
    const proj = struct {
        fn of(r: *const [stream.quant.component_count]Arr, w: C, s: C, b: C) exl3.Proj {
            return .{ .trellis = r[@backingInt(w)], .suh = r[@backingInt(s)], .svh = r[@backingInt(b)] };
        }
    }.of;
    return .{ .gate = proj(raw, .gate_w, .gate_s, .gate_b), .up = proj(raw, .up_w, .up_s, .up_b), .down = proj(raw, .down_w, .down_s, .down_b) };
}

/// FP8 experts by the bf16-dequant route: the routed slots' `[out, in]` e4m3 weights become
/// bf16(code * block scale) (`fp8_block.dequantize`), then the BF16 routed composite. `slots`
/// index `bank`'s leading axis; only the distinct routed slots are dequantized.
pub fn fp8Routed(ops: *Ops, a: std.mem.Allocator, x: Arr, bank: *const [stream.quant.component_count]Arr, slots: []const u32, ids_shape: []const c_int, scores: Arr, limit: f32) !Arr {
    const C = stream.quant.Component;
    const sorted = try a.dupe(u32, slots);
    defer a.free(sorted);
    std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
    var distinct: usize = 0;
    for (sorted) |slot| if (distinct == 0 or sorted[distinct - 1] != slot) {
        sorted[distinct] = slot;
        distinct += 1;
    };
    const used = sorted[0..distinct];
    const position = try a.alloc(u32, used[used.len - 1] + 1);
    defer a.free(position);
    for (used, 0..) |slot, i| position[slot] = @intCast(i);
    const dense = try a.alloc(u32, slots.len);
    defer a.free(dense);
    for (dense, slots) |*d, slot| d.* = position[slot];
    const n: c_int = @intCast(used.len);
    // A union slab holds exactly its routed experts in slots 0..n-1: no gather copy.
    const contiguous = used[used.len - 1] == used.len - 1;
    const pick = try ops.own(mlx.mlx_array_new_data(used.ptr, &.{n}, 1, .uint32));
    var views: [3]Arr = undefined;
    for ([_][2]C{ .{ .gate_w, .gate_s }, .{ .up_w, .up_s }, .{ .down_w, .down_s } }, &views) |pair, *view| {
        var parts: [2]Arr = undefined;
        for (pair, &parts) |c, *part| {
            const all = bank[@backingInt(c)];
            part.* = if (contiguous) try ops.slice(all, 0, 0, n) else try ops.take(all, pick, 0);
        }
        const ws = mlx.getShape(parts[0]);
        const ss = mlx.getShape(parts[1]);
        const rows = n * ws[1];
        var out: [1]Arr = .{.{}};
        try fp8_block.dequantize(ops.s, try ops.reshape(parts[0], &.{ rows, ws[2] }), try ops.reshape(parts[1], &.{ n * ss[1], ss[2] }), fp8_block.RowSplit.dense(@intCast(rows)), &out);
        view.* = try ops.transpose(try ops.reshape(try ops.own(out[0]), &.{ n, ws[1], ws[2] }), &.{ 0, 2, 1 });
    }
    const ids = try ops.own(mlx.mlx_array_new_data(dense.ptr, ids_shape.ptr, @intCast(ids_shape.len), .uint32));
    return bf16Routed(ops, x, views[0], views[1], views[2], ids, scores, limit);
}

/// Banks are [experts,input,output] views. Scores remain FP32 until the final cast.
pub fn bf16Routed(ops: *Ops, x: Arr, gate: Arr, up: Arr, down: Arr, ids: Arr, scores: Arr, limit: f32) !Arr {
    if (!mlx.streamIsGpu(ops.s)) return error.GlmStreamRequiresGpu;
    const xs = mlx.getShape(x);
    const ish = mlx.getShape(ids);
    if (xs.len != 3 or ish.len != 3) return error.InvalidGlmStreamInput;
    const expanded = try ops.reshape(x, &.{ xs[0], xs[1], 1, 1, xs[2] });
    const g = try gather(ops, expanded, gate, ids);
    const u = try gather(ops, expanded, up, ids);
    const hi = try ops.scalar(limit, .bfloat16);
    const lo = try ops.scalar(-limit, .bfloat16);
    const activation = try ops.binary(.mul, try ops.silu(try ops.binary(.min, g, hi)), try ops.binary(.max, try ops.binary(.min, u, hi), lo));
    const y = try gather(ops, activation, down, ids);
    const y4 = try ops.reshape(y, &.{ xs[0], xs[1], ish[2], xs[2] });
    const weights = try ops.reshape(scores, &.{ xs[0], xs[1], ish[2], 1 });
    return ops.cast(try ops.reduce(try ops.binary(.mul, y4, weights), -2, false, false), .bfloat16);
}
fn gather(ops: *Ops, x: Arr, w: Arr, ids: Arr) !Arr {
    const out = try ops.slot();
    try mlx.check(mlx.mlx_gather_mm(out, x, w, .{ .ctx = null }, ids, false, ops.s));
    return out.*;
}

test "GLM stream GPU BF16 routed output matches resident through eviction union and retained output" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const fixture = @import("glm_stream_fixture.zig");
    try fixture.write(t.allocator, tmp.dir, .none);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path);
    const cpu = mlx.gpuStream();
    const geometry = stream.Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
    var engine = try stream.Engine.initWithOptions(t.allocator, path[0..n], geometry, 2 * 3072, cpu, .{ .layout = .bf16_individual, .bounce_size = 4096, .io_workers = 1 });
    defer engine.deinit();
    var store = Stream{ .engine = &engine, .max_tokens = 16, .max_chunk = 3 };
    const cfg = model.ModelConfig{ .glm_swiglu_limit = 10 };
    var banks: [3]Arr = undefined;
    for (&banks, 0..) |*bank, pi| {
        var raw: [4 * 512]u16 = undefined;
        for (0..4) |e| @memset(raw[e * 512 ..][0..512], fixture.value(e, pi));
        const dims = if (pi == 2) [_]c_int{ 4, 32, 16 } else [_]c_int{ 4, 16, 32 };
        const source = mlx.mlx_array_new_data(&raw, &dims, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(source);
        bank.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(bank, source, &.{ 0, 2, 1 }, 3, cpu));
    }
    defer for (banks) |b| {
        _ = mlx.mlx_array_free(b);
    };
    const routes = [_][6]u32{ .{ 0, 1, 0, 1, 0, 1 }, .{ 2, 3, 2, 3, 2, 3 }, .{ 3, 2, 1, 0, 3, 0 }, .{ 0, 1, 0, 1, 0, 1 } };
    var old: Arr = .{ .ctx = null };
    defer if (old.ctx != null) {
        _ = mlx.mlx_array_free(old);
    };
    var first: [96]u16 = undefined;
    for (routes, 0..) |route_ids, iteration| {
        var ops = Ops{ .s = cpu };
        defer ops.deinit();
        var values: [96]f32 = undefined;
        for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 5)) / 4;
        const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 32 }, 3, .float32)), .bfloat16);
        const ids = try ops.own(mlx.mlx_array_new_data(&route_ids, &.{ 1, 3, 2 }, 3, .uint32));
        const scores = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75, 0.4, 0.6, 0.7, 0.3 }, &.{ 1, 3, 2 }, 3, .float32));
        const actual = try store.apply(3, &ops, x, ids, scores, &cfg);
        const expected = try bf16Routed(&ops, x, banks[0], banks[1], banks[2], ids, scores, 10);
        try mlx.check(mlx.mlx_array_eval(expected));
        const bits = mlx.mlx_array_data_bfloat16(actual).?;
        try t.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..96], bits[0..96]);
        try t.expect(bits[0] != 0);
        if (iteration == 0) {
            old = try ops.result(actual);
            @memcpy(&first, bits[0..96]);
        } else try t.expectEqualSlices(u16, &first, mlx.mlx_array_data_bfloat16(old).?[0..96]);
        try t.expectError(error.ExpertLayerAbsent, store.apply(0, &ops, x, ids, scores, &cfg));
        const bad = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 4, 0, 0, 0, 0, 0 }, &.{ 1, 3, 2 }, 3, .uint32));
        const filled = store.engine.fill_bytes_total;
        try t.expectError(error.ExpertOutOfRange, store.apply(3, &ops, x, bad, scores, &cfg));
        try t.expectEqual(filled, store.engine.fill_bytes_total);
    }
    try t.expect(store.engine.fill_bytes_total > 4 * 3072);
    try t.expectError(error.GlmStreamRequestBudgetExceeded, store.admit(15, 2));
    try t.expectError(error.GlmStreamRequestBudgetExceeded, store.admit(0, 4));
    try store.admit(13, 3);
}

/// The fixture's FP8 bank resident: `[experts, out, in]` codes and `[experts, out/128, in/128]` scales.
fn fp8FixtureBank() [stream.quant.component_count]Arr {
    const fixture = @import("glm_stream_fixture.zig");
    const C = stream.quant.Component;
    var bank: [stream.quant.component_count]Arr = @splat(.{ .ctx = null });
    var codes: [4 * 128 * 128]u8 = undefined;
    var scales: [4]f32 = undefined;
    for ([_][2]C{ .{ .gate_w, .gate_s }, .{ .up_w, .up_s }, .{ .down_w, .down_s } }, 0..) |pair, pi| {
        for (0..4) |e| {
            for (0..128 * 128) |i| codes[e * 128 * 128 + i] = fixture.fp8Code(e, pi, i);
            scales[e] = fixture.fp8Scale(e, pi, 0);
        }
        bank[@backingInt(pair[0])] = mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 128, 128 }, 3, .uint8);
        bank[@backingInt(pair[1])] = mlx.mlx_array_new_data(&scales, &[_]c_int{ 4, 1, 1 }, 3, .float32);
    }
    return bank;
}

test "GLM stream GPU FP8 routed output matches the resident FP8 bank through eviction and the union" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try @import("glm_stream_fixture.zig").writeStorage(t.allocator, tmp.dir, .none, 128, 128, .fp8);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path);
    const geometry = stream.Geometry{ .layers = 4, .experts = 4, .hidden = 128, .intermediate = 128, .first_moe_layer = 3 };
    const per_expert = try stream.expertBytesFor(t.allocator, path[0..n], geometry, .fp8_individual);
    var engine = try stream.Engine.initWithOptions(t.allocator, path[0..n], geometry, 2 * per_expert, s, .{ .layout = .fp8_individual, .bounce_size = 1 << 20, .io_workers = 1 });
    defer engine.deinit();
    var store = Stream{ .engine = &engine, .max_tokens = 16, .max_chunk = 3 };
    const cfg = model.ModelConfig{ .glm_swiglu_limit = 10 };
    var bank = fp8FixtureBank();
    defer for (bank) |b| if (b.ctx != null) {
        _ = mlx.mlx_array_free(b);
    };
    var union_seen = false;
    for ([_][6]u32{ .{ 0, 1, 2, 3, 2, 0 }, .{ 2, 3, 2, 3, 2, 3 }, .{ 3, 2, 1, 0, 3, 0 }, .{ 0, 1, 0, 1, 0, 1 }, .{ 1, 3, 1, 3, 3, 1 } }) |route_ids| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var values: [3 * 128]f32 = undefined;
        for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 8;
        const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 128 }, 3, .float32)), .bfloat16);
        const ids = try ops.own(mlx.mlx_array_new_data(&route_ids, &.{ 1, 3, 2 }, 3, .uint32));
        const scores = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75, 0.4, 0.6, 0.7, 0.3 }, &.{ 1, 3, 2 }, 3, .float32));
        const actual = try store.apply(3, &ops, x, ids, scores, &cfg);
        union_seen = union_seen or engine.last.union_members > 0 or engine.layers[3].stats.union_members > 0;
        const expected = try fp8Routed(&ops, t.allocator, x, &bank, &route_ids, &.{ 1, 3, 2 }, scores, 10);
        try mlx.check(mlx.mlx_array_eval(expected));
        const want = mlx.mlx_array_data_bfloat16(expected).?[0..3 * 128];
        try t.expectEqualSlices(u16, want, mlx.mlx_array_data_bfloat16(actual).?[0..3 * 128]);
        var nonzero = false;
        for (want) |bits| nonzero = nonzero or bits & 0x7fff != 0;
        try t.expect(nonzero);
    }
    try t.expect(union_seen);
    try t.expect(engine.fill_experts_total > 4);
}

test "GLM FP8 expert dequant is bf16 of the e4m3 code times its block scale, against an f32 oracle" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0xf8f8);
    const rnd = prng.random();
    var codes: [2 * 256 * 256]u8 = undefined;
    for (&codes) |*c| {
        c.* = rnd.int(u8);
        if (c.* & 0x7f == 0x7f) c.* ^= 1;
    }
    var scales: [2 * 2 * 2]f32 = undefined;
    for (&scales) |*v| v.* = (0.5 + rnd.float(f32)) * 4.0e-5;
    var bank: [stream.quant.component_count]Arr = @splat(.{ .ctx = null });
    defer for (bank) |b| if (b.ctx != null) {
        _ = mlx.mlx_array_free(b);
    };
    const C = stream.quant.Component;
    for ([_]C{ .gate_w, .up_w, .down_w }) |c| bank[@backingInt(c)] = mlx.mlx_array_new_data(&codes, &[_]c_int{ 2, 256, 256 }, 3, .uint8);
    for ([_]C{ .gate_s, .up_s, .down_s }) |c| bank[@backingInt(c)] = mlx.mlx_array_new_data(&scales, &[_]c_int{ 2, 2, 2 }, 3, .float32);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var out: [1]Arr = .{.{}};
    try fp8_block.dequantize(s, try ops.reshape(bank[@backingInt(C.gate_w)], &.{ 512, 256 }), try ops.reshape(bank[@backingInt(C.gate_s)], &.{ 4, 2 }), fp8_block.RowSplit.dense(512), &out);
    defer _ = mlx.mlx_array_free(out[0]);
    try mlx.check(mlx.mlx_array_eval(out[0]));
    const got = mlx.mlx_array_data_bfloat16(out[0]).?;
    for (0..2) |e| for (0..256) |r| for (0..256) |col| {
        const code = codes[(e * 256 + r) * 256 + col];
        const sign: f32 = if (code & 0x80 != 0) -1 else 1;
        const exp: i32 = @intCast((code >> 3) & 0xf);
        const man: f32 = @floatFromInt(code & 7);
        const magnitude: f32 = if (exp == 0) man / 8 * std.math.pow(f32, 2, -6) else (1 + man / 8) * std.math.pow(f32, 2, @floatFromInt(exp - 7));
        const scale = scales[(e * 2 + r / 128) * 2 + col / 128];
        const bits: u32 = @bitCast(sign * magnitude * scale);
        const want: u16 = @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        try t.expectEqual(want, got[(e * 256 + r) * 256 + col]);
    };
}

test "GLM stream CPU only one request holds the admitted cache reserve" {
    var store = Stream{ .engine = undefined, .max_tokens = 16, .max_chunk = 4 };
    var a: u8 = 0;
    var b: u8 = 0;
    try store.claim(&a);
    try store.claim(&a);
    try std.testing.expectError(error.GlmStreamRequestBusy, store.claim(&b));
    store.release(&b);
    try std.testing.expectError(error.GlmStreamRequestBusy, store.claim(&b));
    store.release(&a);
    try store.claim(&b);
    store.release(&b);
    try std.testing.expect(store.request_owner == null);
}

test "GLM stream CPU refuses unsupported BF16 gather before reading arrays" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var store = Stream{ .engine = undefined, .max_tokens = 16, .max_chunk = 4 };
    var ops = Ops{ .s = cpu };
    defer ops.deinit();
    const nil = Arr{ .ctx = null };
    try std.testing.expectError(error.GlmStreamRequiresGpu, store.apply(0, &ops, nil, nil, nil, &.{ .glm_swiglu_limit = 10 }));
    try std.testing.expectError(error.GlmStreamRequiresGpu, bf16Routed(&ops, nil, nil, nil, nil, nil, nil, 10));
    try std.testing.expectEqual(@as(usize, 0), ops.count);
}
