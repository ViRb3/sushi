//! Native GLM routed BF16 experts over Sushi's fixed cache and I/O slabs.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const stream = @import("expert_stream.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

pub const Budget = struct { total: u64, trunk: u64, reserve: u64, cache: u64, fixed: u64 };
fn geometryFor(cfg: *const model.ModelConfig) !stream.Geometry {
    return .{ .layers = std.math.cast(u16, cfg.num_hidden_layers) orelse return error.InvalidGlmStreamBudget, .experts = std.math.cast(u16, cfg.num_experts) orelse return error.InvalidGlmStreamBudget, .hidden = cfg.hidden_size, .intermediate = cfg.moe_intermediate_size, .first_moe_layer = std.math.cast(u16, cfg.first_k_dense_replace) orelse return error.InvalidGlmStreamBudget };
}
fn bytesPerExpert(g: stream.Geometry) !u64 {
    return stream.expertBytes(try std.math.mul(u32, 2, g.intermediate), g.hidden, g.intermediate);
}

/// Retained lazy trunk metadata must fit this bound before tensor evaluation.
pub fn trunkLimit(cfg: *const model.ModelConfig, total: u64, reserve: u64) !u64 {
    const g = try geometryFor(cfg);
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

/// The caller owns the trunk and this engine; destroy Model before either owner.
/// One inference thread may use a stream at a time. Requests own only their caches.
pub const Bf16 = struct {
    engine: stream.Engine,
    budget: Budget,
    max_tokens: usize,
    max_chunk: usize,
    request_owner: ?*const anyopaque = null,

    pub fn init(a: std.mem.Allocator, directory: []const u8, cfg: *const model.ModelConfig, total: u64, trunk: u64, reserve: u64, max_tokens: usize, max_chunk: usize, s: mlx.mlx_stream) !Bf16 {
        if (!mlx.streamIsGpu(s)) return error.GlmStreamRequiresGpu;
        if (!cfg.isGlm5() or max_tokens == 0 or max_tokens > cfg.max_position_embeddings or max_chunk == 0 or max_chunk > max_tokens or max_chunk > 512) return error.InvalidGlmStreamBudget;
        if (reserve < try minimumReserve(cfg, max_tokens, max_chunk)) return error.GlmStreamReserveTooSmall;
        const g = try geometryFor(cfg);
        const expert_bytes = try bytesPerExpert(g);
        const budget = try plan(total, trunk, reserve, g, expert_bytes);
        return .{ .engine = try stream.Engine.initWithOptions(a, directory, g, budget.cache, s, .{ .layout = .bf16_individual }), .budget = budget, .max_tokens = max_tokens, .max_chunk = max_chunk };
    }
    pub fn deinit(self: *Bf16) void {
        self.engine.deinit();
    }
    pub fn claim(self: *Bf16, owner: *const anyopaque) !void {
        if (self.request_owner) |current| if (current != owner) return error.GlmStreamRequestBusy;
        self.request_owner = owner;
    }
    pub fn release(self: *Bf16, owner: *const anyopaque) void {
        if (self.request_owner == owner) self.request_owner = null;
    }
    pub fn admit(self: *const Bf16, offset: usize, rows: usize) !void {
        if (rows == 0 or rows > self.max_chunk or offset > self.max_tokens or rows > self.max_tokens - offset) return error.GlmStreamRequestBudgetExceeded;
    }
    pub fn apply(self: *Bf16, layer: u16, ops: *Ops, x: Arr, ids: Arr, scores: Arr, limit: f32) !Arr {
        if (!mlx.streamIsGpu(ops.s)) return error.GlmStreamRequiresGpu;
        const shape = mlx.getShape(x);
        const ish = mlx.getShape(ids);
        if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > self.max_chunk or shape[2] != self.engine.geometry.hidden or mlx.mlx_array_dtype(x) != .bfloat16 or
            (mlx.mlx_array_dtype(ids) != .uint32 and mlx.mlx_array_dtype(ids) != .int32) or ish.len != 3 or ish[0] != 1 or ish[1] != shape[1] or ish[2] < 1 or ish[2] > self.engine.geometry.experts or !std.mem.eql(c_int, ish, mlx.getShape(scores)) or mlx.mlx_array_dtype(scores) != .float32 or limit != 10) return error.InvalidGlmStreamInput;
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
        const out = try bf16Routed(&scope, x, prepared.gate, prepared.up, prepared.down, remapped, scores, limit);
        // Complete every slab reader before another layer can refill the union.
        try mlx.check(mlx.mlx_array_eval(out));
        return ops.own(try scope.result(out));
    }
};

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
    var store = Bf16{ .engine = try stream.Engine.initWithOptions(t.allocator, path[0..n], geometry, 2 * 3072, cpu, .{ .layout = .bf16_individual, .bounce_size = 4096, .io_workers = 1 }), .budget = undefined, .max_tokens = 16, .max_chunk = 3 };
    defer store.deinit();
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
        const actual = try store.apply(3, &ops, x, ids, scores, 10);
        const expected = try bf16Routed(&ops, x, banks[0], banks[1], banks[2], ids, scores, 10);
        try mlx.check(mlx.mlx_array_eval(expected));
        const bits = mlx.mlx_array_data_bfloat16(actual).?;
        try t.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..96], bits[0..96]);
        try t.expect(bits[0] != 0);
        if (iteration == 0) {
            old = try ops.result(actual);
            @memcpy(&first, bits[0..96]);
        } else try t.expectEqualSlices(u16, &first, mlx.mlx_array_data_bfloat16(old).?[0..96]);
        try t.expectError(error.ExpertLayerAbsent, store.apply(0, &ops, x, ids, scores, 10));
        const bad = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 4, 0, 0, 0, 0, 0 }, &.{ 1, 3, 2 }, 3, .uint32));
        const filled = store.engine.fill_bytes_total;
        try t.expectError(error.ExpertOutOfRange, store.apply(3, &ops, x, bad, scores, 10));
        try t.expectEqual(filled, store.engine.fill_bytes_total);
    }
    try t.expect(store.engine.fill_bytes_total > 4 * 3072);
    try t.expectError(error.GlmStreamRequestBudgetExceeded, store.admit(15, 2));
    try t.expectError(error.GlmStreamRequestBudgetExceeded, store.admit(0, 4));
    try store.admit(13, 3);
}

test "GLM stream CPU only one request holds the admitted cache reserve" {
    var store = Bf16{ .engine = undefined, .budget = undefined, .max_tokens = 16, .max_chunk = 4 };
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
    var store = Bf16{ .engine = undefined, .budget = undefined, .max_tokens = 16, .max_chunk = 4 };
    var ops = Ops{ .s = cpu };
    defer ops.deinit();
    const nil = Arr{ .ctx = null };
    try std.testing.expectError(error.GlmStreamRequiresGpu, store.apply(0, &ops, nil, nil, nil, 10));
    try std.testing.expectError(error.GlmStreamRequiresGpu, bf16Routed(&ops, nil, nil, nil, nil, nil, nil, 10));
    try std.testing.expectEqual(@as(usize, 0), ops.count);
}
