//! Diagnostic GLM model composition, kept separate from served architecture dispatch.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const base = @import("glm5_model.zig");
const primitive = @import("glm5_next.zig");
const exl3 = @import("sushi_exl3");
const Arr = mlx.mlx_array;
const Ops = base.Ops;
const Linear = base.Linear;

pub const Routed = struct { indices: Arr, scores: Arr };

fn route(ops: *Ops, x: Arr, weight: Arr, correction: Arr, top: c_int, scale: f32, normalize: bool) !Routed {
    if (try @import("glm5_router.zig").route(ops.s, x, weight, correction, top, scale, normalize)) |fused| {
        const ids = ops.own(fused.indices) catch |e| {
            _ = mlx.mlx_array_free(fused.scores);
            return e;
        };
        return .{ .indices = ids, .scores = try ops.own(fused.scores) };
    }
    return routeReference(ops, x, weight, correction, top, scale, normalize);
}

fn routeReference(ops: *Ops, x: Arr, weight: Arr, correction: Arr, top: c_int, scale: f32, normalize: bool) !Routed {
    const logits = try ops.binary(.mm, try ops.cast(x, .float32), try ops.transpose(weight, &.{ 1, 0 }));
    const scores = try ops.unary(.sigmoid, logits);
    const selection = try ops.binary(.add, scores, correction);
    const count = mlx.getShape(scores)[2];
    if (top <= 0 or top > count) return error.InvalidGlmRouter;
    const order = try ops.slot();
    try mlx.check(mlx.mlx_argpartition_axis(order, try ops.unary(.negative, selection), top - 1, -1, ops.s));
    const ids = try ops.slice(order.*, 2, 0, top);
    const picked = try ops.slot();
    try mlx.check(mlx.mlx_take_along_axis(picked, scores, ids, -1, ops.s));
    const probs = if (normalize) try ops.binary(.div, picked.*, try ops.reduce(picked.*, -1, false, true)) else picked.*;
    return .{ .indices = try ops.cast(ids, .uint32), .scores = try ops.binary(.mul, probs, try ops.scalar(scale, .float32)) };
}

test "GLM router correction changes selection but not normalized weights" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.ones(&.{ 1, 1, 2 }, .float32);
    const w = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0, 0, 1, 0, -1, 0 }, &[_]c_int{ 3, 2 }, 2, .float32));
    const bias = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0, 0, 2 }, &[_]c_int{3}, 1, .float32));
    const r = try route(&ops, x, w, bias, 2, 2.5, true);
    try mlx.check(mlx.mlx_array_eval(r.indices));
    try mlx.check(mlx.mlx_array_eval(r.scores));
    const ids = mlx.mlx_array_data_uint32(r.indices).?;
    const scores = mlx.mlx_array_data_float32(r.scores).?;
    var sum: f32 = 0;
    for (0..2) |i| {
        try std.testing.expect(ids[i] == 1 or ids[i] == 2);
        const expected: f32 = if (ids[i] == 1) 2.5 / (1 + @exp(@as(f32, -1))) else 2.5 / (1 + @exp(@as(f32, 1)));
        try std.testing.expectApproxEqAbs(expected, scores[i], 1e-5);
        sum += scores[i];
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), sum, 1e-5);
}

fn tensor(weights: *const model.Weights, prefix: []const u8, suffix: []const u8) !Arr {
    var buf: [256]u8 = undefined;
    return weights.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix })) orelse error.MissingGlmWeight;
}

fn linear(weights: *const model.Weights, prefix: []const u8, suffix: []const u8, input: u32) !Linear {
    var buf: [256]u8 = undefined;
    return Linear.load(weights, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix }), input);
}

pub const Mla = struct {
    qa: Linear,
    qb: Linear,
    kva: Linear,
    out: Linear,
    qa_norm: Arr,
    kv_norm: Arr,
    iq: Linear,
    ik: Linear,
    iw: Linear,
    ik_norm: Arr,
    ik_bias: Arr,
    compress: Arr,
    ape: Arr,
    wk: Arr,
    wv: Arr,
    sk: Arr,
    sv: Arr,
    bk: Arr,
    bv: Arr,
    quantized: bool,
    prepared: Ops,

    pub fn load(weights: *const model.Weights, prefix: []const u8, cfg: *const model.ModelConfig, s: mlx.mlx_stream) !Mla {
        var prep = Ops{ .s = s };
        errdefer prep.deinit();
        const kvb = try linear(weights, prefix, "kv_b_proj", cfg.mla_kv_lora_rank);
        const h: c_int = @intCast(cfg.num_attention_heads);
        const kd: c_int = @intCast(cfg.mla_qk_nope_head_dim);
        const vd: c_int = @intCast(cfg.mla_v_head_dim);
        const latent: c_int = @intCast(cfg.mla_kv_lora_rank);
        if (kvb.output != h * (kd + vd)) return error.InvalidGlmMlaWeight;
        const quant = kvb.scales.ctx != null;
        const w = try prep.reshape(kvb.w, &.{ h, kd + vd, if (quant) @divExact(latent, 4) else latent });
        const wk = try prep.contiguous(try prep.slice(w, 1, 0, kd));
        const wv = try prep.contiguous(try prep.slice(w, 1, kd, kd + vd));
        var sk: Arr = .{ .ctx = null };
        var sv: Arr = .{ .ctx = null };
        var bk: Arr = .{ .ctx = null };
        var bv: Arr = .{ .ctx = null };
        if (quant) {
            const scales = try prep.reshape(kvb.scales, &.{ h, kd + vd, @divExact(latent, 128) });
            const biases = try prep.reshape(kvb.biases, &.{ h, kd + vd, @divExact(latent, 128) });
            sk = try prep.contiguous(try prep.slice(scales, 1, 0, kd));
            sv = try prep.contiguous(try prep.slice(scales, 1, kd, kd + vd));
            bk = try prep.contiguous(try prep.slice(biases, 1, 0, kd));
            bv = try prep.contiguous(try prep.slice(biases, 1, kd, kd + vd));
        }
        const result = Mla{
            .qa = try linear(weights, prefix, "q_a_proj", cfg.hidden_size),
            .qb = try linear(weights, prefix, "q_b_proj", cfg.mla_q_lora_rank),
            .kva = try linear(weights, prefix, "kv_a_proj_with_mqa", cfg.hidden_size),
            .out = try linear(weights, prefix, "o_proj", cfg.num_attention_heads * cfg.mla_v_head_dim),
            .qa_norm = try tensor(weights, prefix, "q_a_layernorm.weight"),
            .kv_norm = try tensor(weights, prefix, "kv_a_layernorm.weight"),
            .iq = try linear(weights, prefix, "indexer.wq_b", cfg.mla_q_lora_rank),
            .ik = try linear(weights, prefix, "indexer.wk", cfg.hidden_size),
            .iw = try linear(weights, prefix, "indexer.weights_proj", cfg.hidden_size),
            .ik_norm = try tensor(weights, prefix, "indexer.k_norm.weight"),
            .ik_bias = try tensor(weights, prefix, "indexer.k_norm.bias"),
            .compress = try tensor(weights, prefix, "indexer.index_kpool_compress_gate"),
            .ape = try tensor(weights, prefix, "indexer.index_kpool_compress_ape"),
            .wk = wk,
            .wv = wv,
            .sk = sk,
            .sv = sv,
            .bk = bk,
            .bv = bv,
            .quantized = quant,
            .prepared = prep,
        };
        const expected_outputs = [_]u32{ cfg.mla_q_lora_rank, cfg.num_attention_heads * cfg.mla_qk_nope_head_dim, cfg.mla_kv_lora_rank, cfg.hidden_size, cfg.indexer_n_heads * cfg.indexer_head_dim, cfg.indexer_head_dim, cfg.indexer_n_heads };
        for ([_]Linear{ result.qa, result.qb, result.kva, result.out, result.iq, result.ik, result.iw }, expected_outputs) |proj, expected| {
            if (proj.output != @as(c_int, @intCast(expected))) return error.InvalidGlmMlaWeight;
        }
        for ([_]Arr{ result.qa_norm, result.kv_norm, result.ik_norm, result.ik_bias }, [_]u32{ cfg.mla_q_lora_rank, cfg.mla_kv_lora_rank, cfg.indexer_head_dim, cfg.indexer_head_dim }) |value, width| {
            const dtype = mlx.mlx_array_dtype(value);
            if (!std.mem.eql(c_int, &.{@intCast(width)}, mlx.getShape(value)) or (dtype != .bfloat16 and dtype != .float32)) return error.InvalidGlmMlaWeight;
        }
        if (!std.mem.eql(c_int, &.{ @intCast(cfg.indexer_head_dim), @intCast(cfg.hidden_size) }, mlx.getShape(result.compress)) or
            !std.mem.eql(c_int, &.{ 4, @intCast(cfg.indexer_head_dim) }, mlx.getShape(result.ape))) return error.InvalidGlmMlaWeight;
        for ([_]Arr{ result.compress, result.ape }) |value| {
            const dtype = mlx.mlx_array_dtype(value);
            if (dtype != .bfloat16 and dtype != .float32) return error.InvalidGlmMlaWeight;
        }
        const evals = mlx.mlx_vector_array_new_data(prep.values[0..prep.count].ptr, prep.count);
        defer _ = mlx.mlx_vector_array_free(evals);
        try mlx.check(mlx.mlx_eval(evals));
        return result;
    }

    pub fn deinit(self: *Mla) void {
        self.prepared.deinit();
    }

    pub fn densePrefillEligible(enabled: bool, rows: c_int, offset: usize, cfg: *const model.ModelConfig, dtype: mlx.mlx_dtype) bool {
        return enabled and rows > 8 and offset <= 2051 and @as(usize, @intCast(rows)) <= 2051 - offset and
            cfg.mla_qk_nope_head_dim == 256 and cfg.mla_v_head_dim == 256 and dtype == .bfloat16;
    }

    fn densePrefill(self: *const Mla, ops: *Ops, q: Arr, cfg: *const model.ModelConfig, state: anytype) !Arr {
        const rows = mlx.getShape(q)[0];
        const latent: c_int = @intCast(cfg.mla_kv_lora_rank);
        const valid_cache = try ops.slice(state.latent, 0, 0, @intCast(state.processed));
        const cached = try ops.reshape(valid_cache, &.{ 1, 1, @intCast(state.processed), latent });
        const keys = if (self.quantized) try ops.qmm(cached, self.wk, self.sk, self.bk, true) else try ops.binary(.mm, cached, try ops.transpose(self.wk, &.{ 0, 2, 1 }));
        const values = if (self.quantized) try ops.qmm(cached, self.wv, self.sv, self.bv, true) else try ops.binary(.mm, cached, try ops.transpose(self.wv, &.{ 0, 2, 1 }));
        const query = try ops.transpose(q, &.{ 2, 1, 0, 3 });
        const attended = try ops.slot();
        // Causal SDPA aligns the query's final row with the final cached key,
        // so the same mask is valid for both cold and cached prefill chunks.
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(attended, query, try ops.cast(keys, .bfloat16), try ops.cast(values, .bfloat16), 1 / @sqrt(@as(f32, @floatFromInt(cfg.mla_qk_nope_head_dim))), "causal", .{ .ctx = null }, .{ .ctx = null }, true, ops.s));
        const output = try ops.reshape(try ops.transpose(attended.*, &.{ 0, 2, 1, 3 }), &.{ 1, rows, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
        return self.out.apply(ops, output);
    }

    pub fn apply(self: *const Mla, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, state: anytype) !Arr {
        return self.applyMode(ops, x, cfg, state, false);
    }

    pub fn applyMode(self: *const Mla, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, state: anytype, dense_prefill: bool) !Arr {
        const sh = mlx.getShape(x);
        if (sh[0] != 1) return error.GlmBatchUnsupported;
        const t = sh[1];
        const h: c_int = @intCast(cfg.num_attention_heads);
        const kd: c_int = @intCast(cfg.mla_qk_nope_head_dim);
        const latent: c_int = @intCast(cfg.mla_kv_lora_rank);
        const qr = try ops.rms(try self.qa.apply(ops, x), self.qa_norm, cfg.rms_norm_eps);
        const q = try ops.reshape(try self.qb.apply(ops, qr), &.{ t, h, 1, kd });
        const dense = densePrefillEligible(dense_prefill, t, state.processed, cfg, mlx.mlx_array_dtype(q));
        const absorbed: Arr = if (dense) .{ .ctx = null } else if (self.quantized) try ops.qmm(q, self.wk, self.sk, self.bk, false) else try ops.binary(.mm, q, self.wk);
        const kv = try ops.rms(try self.kva.apply(ops, x), self.kv_norm, cfg.rms_norm_eps);
        const iq = try ops.reshape(try self.iq.apply(ops, qr), &.{ t, @intCast(cfg.indexer_n_heads), @intCast(cfg.indexer_head_dim) });
        const ik = try ops.layerNorm(try self.ik.apply(ops, x), self.ik_norm, self.ik_bias, 1e-6);
        const iw = try ops.cast(try ops.binary(.mul, try self.iw.apply(ops, x), try ops.scalar(1 / @sqrt(@as(f32, @floatFromInt(cfg.indexer_n_heads * cfg.indexer_head_dim))), .float32)), mlx.mlx_array_dtype(iq));
        const gates = try ops.binary(.mm, x, try ops.transpose(self.compress, &.{ 1, 0 }));
        const offset = try state.append(try ops.reshape(kv, &.{ t, latent }), try ops.reshape(ik, &.{ t, @intCast(cfg.indexer_head_dim) }), try ops.reshape(gates, &.{ t, @intCast(cfg.indexer_head_dim) }), self.ape, ops.s);
        if (dense) return self.densePrefill(ops, q, cfg, state);
        const y = try ops.own(try @import("glm5_attention.zig").attend(state, try ops.reshape(absorbed, &.{ t, h, latent }), iq, try ops.reshape(iw, &.{ t, @intCast(cfg.indexer_n_heads) }), offset, 1 / @sqrt(@as(f32, @floatFromInt(kd))), ops.s));
        const y4 = try ops.reshape(y, &.{ t, h, 1, latent });
        const values = if (self.quantized) try ops.qmm(y4, self.wv, self.sv, self.bv, true) else try ops.binary(.mm, y4, try ops.transpose(self.wv, &.{ 0, 2, 1 }));
        return self.out.apply(ops, try ops.reshape(values, &.{ 1, t, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) }));
    }
};

const Moe = struct {
    weight: Arr,
    correction: Arr,
    bank: exl3.Bank,
    shared: ?base.DenseMlp,
    fn load(weights: *const model.Weights, prefix: []const u8, cfg: *const model.ModelConfig) !Moe {
        var projs: [3]exl3.Proj = undefined;
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, i| {
            var buf: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "{s}.switch_mlp.{s}", .{ prefix, p });
            projs[i] = .{ .trellis = try tensor(weights, name, "trellis"), .suh = try tensor(weights, name, "suh"), .svh = try tensor(weights, name, "svh") };
        }
        var buf: [256]u8 = undefined;
        return .{ .weight = try tensor(weights, prefix, "gate.weight"), .correction = try tensor(weights, prefix, "gate.e_score_correction_bias"), .bank = .{ .gate = projs[0], .up = projs[1], .down = projs[2] }, .shared = if (cfg.shared_expert_intermediate_size > 0) try base.DenseMlp.load(weights, try std.fmt.bufPrint(&buf, "{s}.shared_experts", .{prefix}), cfg.hidden_size, cfg.shared_expert_intermediate_size) else null };
    }
    fn apply(self: Moe, ops: *Ops, x: Arr, cfg: *const model.ModelConfig) !Arr {
        return self.applyTracked(ops, x, cfg, null);
    }
    fn applyTracked(self: Moe, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, component: ?*ComponentTimer) !Arr {
        const routing = try route(ops, x, self.weight, self.correction, @intCast(cfg.num_experts_per_tok), cfg.router_scaling_factor, cfg.moe_route_norm);
        try ComponentTimer.mark(component, "router", &.{ routing.indices, routing.scores }, null);
        const routed = try ops.own(try exl3.moeClamped(ops.s, x, self.bank, routing.indices, routing.scores, .{ .codebook = cfg.expert_quant_codebook, .window = cfg.expert_quant_window }, @intFromFloat(cfg.glm_swiglu_limit)));
        try ComponentTimer.mark(component, "routed", &.{routed}, null);
        if (self.shared) |shared| {
            const y = try shared.apply(ops, x, cfg.glm_swiglu_limit);
            try ComponentTimer.mark(component, "shared", &.{y}, null);
            const combined = try ops.binary(.add, routed, y);
            try ComponentTimer.mark(component, "combine_ffn", &.{combined}, null);
            return combined;
        }
        return routed;
    }
};

const Attention = union(enum) { kda: base.KdaLayer, mla: Mla };
const Ffn = union(enum) { dense: base.DenseMlp, moe: Moe };
const Layer = struct {
    attn: Attention,
    ffn: Ffn,
    hc_attn: base.Hc,
    hc_ffn: base.Hc,
    norm_attn: Arr,
    norm_ffn: Arr,
    fn deinit(self: *Layer) void {
        switch (self.attn) {
            .mla => |*a| a.deinit(),
            .kda => |*a| {
                if (@hasDecl(base.KdaLayer, "deinit")) a.deinit();
            },
        }
    }
};
const LayerState = struct {
    recurrent: @import("transformer.zig").SSMCacheEntry,
    attention: @import("glm5_attention.zig").State,
    fn init() LayerState {
        return .{ .recurrent = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false }, .attention = .init() };
    }
    fn deinit(self: *LayerState) void {
        _ = mlx.mlx_array_free(self.recurrent.conv_state);
        _ = mlx.mlx_array_free(self.recurrent.ssm_state);
        self.attention.deinit();
    }
};

pub const ComponentTimes = struct {
    hc_attn: u64 = 0,
    attn_norm: u64 = 0,
    attention: u64 = 0,
    expand_attn: u64 = 0,
    hc_ffn: u64 = 0,
    ffn_norm: u64 = 0,
    router: u64 = 0,
    routed: u64 = 0,
    shared: u64 = 0,
    combine_ffn: u64 = 0,
    dense_ffn: u64 = 0,
    expand_ffn: u64 = 0,
};

const ComponentTimer = struct {
    totals: *ComponentTimes,
    clock: @import("io_util.zig").Stopwatch,

    fn mark(self: ?*ComponentTimer, comptime field: []const u8, outputs: []const Arr, state: ?*LayerState) !void {
        const timer = self orelse return;
        const values = mlx.mlx_vector_array_new_data(outputs.ptr, outputs.len);
        defer _ = mlx.mlx_vector_array_free(values);
        if (state) |st| try appendLayerState(values, st);
        try mlx.check(mlx.mlx_eval(values));
        @field(timer.totals, field) += timer.clock.read();
        timer.clock.reset();
    }
};

pub const Capture = struct { ids: []const u32, out: []Arr };

/// Caller owns snapshots. Each selected layer has attention then FFN records.
/// Tensor order is stable for private fixture consumers; stored weights retain their dtype.
pub const HcSnapshot = struct {
    pub const names = [_][:0]const u8{ "input", "fn", "scale", "base", "mixes", "mixed", "post", "comb" };
    values: [8]Arr = @splat(.{ .ctx = null }),
    pub fn deinit(self: *HcSnapshot) void {
        for (self.values) |v| if (v.ctx != null) {
            _ = mlx.mlx_array_free(v);
        };
        self.* = .{};
    }
};
pub const HcCapture = struct {
    ids: []const u32,
    records: []HcSnapshot,
    fn retain(self: *HcCapture, index: usize, ffn: bool, ops: *Ops, hc: base.Hc, x: Arr, cfg: *const model.ModelConfig) !void {
        for (self.ids, 0..) |id, i| if (id == index) {
            const mixes = try hc.mixReference(ops, x, cfg);
            const result = try primitive.hcCollapse(x, mixes, hc.scale, hc.base, @intCast(cfg.glm_hc_sinkhorn_iters), cfg.glm_hc_eps, ops.s);
            defer result.deinit();
            var record = HcSnapshot{};
            errdefer record.deinit();
            for (&record.values, [_]Arr{ x, hc.w, hc.scale, hc.base, mixes, result.mixed, result.post, result.comb }) |*out, value| out.* = try ops.result(value);
            const at = i * 2 + @intFromBool(ffn);
            self.records[at].deinit();
            self.records[at] = record;
        };
    }
};

pub const Request = struct {
    allocator: std.mem.Allocator,
    layers: []LayerState,
    offset: usize = 0,
    failed: bool = false,
    profile: bool = false,
    profile_components: bool = false,
    component_ns: [128]ComponentTimes = @splat(.{}),
    decode_async: bool = true,
    dense_prefill: bool = false,
    prefill_async: bool = false,
    capture: ?*Capture = null,
    hc_capture: ?*HcCapture = null,
    layer_ns: [128]u64 = @splat(0),
    pub fn init(allocator: std.mem.Allocator, count: usize) !Request {
        const layers = try allocator.alloc(LayerState, count);
        for (layers) |*layer| layer.* = .init();
        return .{ .allocator = allocator, .layers = layers };
    }
    pub fn deinit(self: *Request) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
    }
    pub fn reset(self: *Request) void {
        for (self.layers) |*layer| {
            layer.deinit();
            layer.* = .init();
        }
        self.offset = 0;
        self.failed = false;
        self.layer_ns = @splat(0);
        self.component_ns = @splat(.{});
    }
};

var schedule_test_syncs: usize = 0;
var schedule_test_asyncs: usize = 0;

fn appendLayerState(evals: mlx.mlx_vector_array, state: *const LayerState) !void {
    if (state.recurrent.initialized) {
        try mlx.check(mlx.mlx_vector_array_append_value(evals, state.recurrent.conv_state));
        try mlx.check(mlx.mlx_vector_array_append_value(evals, state.recurrent.ssm_state));
    }
    for (state.attention.arrays()) |cache| if (cache.ctx != null) {
        try mlx.check(mlx.mlx_vector_array_append_value(evals, cache));
    };
}

fn appendCaptures(evals: mlx.mlx_vector_array, capture: ?*Capture, hc_capture: ?*HcCapture, first: usize, end: usize) !void {
    if (capture) |c| for (c.ids, 0..) |id, i| {
        if (id >= first and id < end) try mlx.check(mlx.mlx_vector_array_append_value(evals, c.out[i]));
    };
    if (hc_capture) |c| for (c.ids, 0..) |id, i| {
        if (id >= first and id < end) for (c.records[i * 2 ..][0..2]) |record| {
            for (record.values) |v| if (v.ctx != null) {
                try mlx.check(mlx.mlx_vector_array_append_value(evals, v));
            };
        };
    };
}

pub const Model = struct {
    allocator: std.mem.Allocator,
    cfg: model.ModelConfig,
    layers: []Layer,
    embedding: Linear,
    head: Linear,
    norm: Arr,
    s: mlx.mlx_stream,

    pub fn load(allocator: std.mem.Allocator, cfg: model.ModelConfig, weights: *const model.Weights, s: mlx.mlx_stream) !Model {
        if (!cfg.isGlm5()) return error.InvalidGlmConfig;
        const layers = try allocator.alloc(Layer, cfg.num_hidden_layers);
        var loaded: usize = 0;
        errdefer {
            for (layers[0..loaded]) |*layer| layer.deinit();
            allocator.free(layers);
        }
        const emb = try linear(weights, cfg.weight_prefix, "embed_tokens", cfg.hidden_size);
        const head = try Linear.load(weights, "lm_head", cfg.hidden_size);
        if (emb.output != cfg.vocab_size or head.output != cfg.vocab_size) return error.InvalidGlmVocabulary;
        for (layers, 0..) |*layer, i| {
            var prefix_buf: [256]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buf, "{s}.layers.{d}", .{ cfg.weight_prefix, i });
            const hc_attn = try base.Hc.load(weights, prefix, "hc_attn", cfg.hidden_size);
            const hc_ffn = try base.Hc.load(weights, prefix, "hc_ffn", cfg.hidden_size);
            const norm_attn = try tensor(weights, prefix, "input_layernorm.weight");
            const norm_ffn = try tensor(weights, prefix, "post_attention_layernorm.weight");
            var buf: [256]u8 = undefined;
            const ffn: Ffn = if (i < cfg.first_k_dense_replace)
                .{ .dense = try base.DenseMlp.load(weights, try std.fmt.bufPrint(&buf, "{s}.mlp", .{prefix}), cfg.hidden_size, cfg.intermediate_size) }
            else
                .{ .moe = try Moe.load(weights, try std.fmt.bufPrint(&buf, "{s}.mlp", .{prefix}), &cfg) };
            const name = try std.fmt.bufPrint(&buf, "{s}.self_attn", .{prefix});
            const attn: Attention = if ((i + 1) % cfg.full_attention_interval == 0) .{ .mla = try Mla.load(weights, name, &cfg, s) } else blk: {
                var kda = try base.KdaLayer.load(weights, name, &cfg);
                if (@hasDecl(base.KdaLayer, "prepare")) try kda.prepare(s);
                break :blk .{ .kda = kda };
            };
            layer.* = .{ .attn = attn, .ffn = ffn, .hc_attn = hc_attn, .hc_ffn = hc_ffn, .norm_attn = norm_attn, .norm_ffn = norm_ffn };
            loaded += 1;
        }
        return .{ .allocator = allocator, .cfg = cfg, .layers = layers, .embedding = emb, .head = head, .norm = try tensor(weights, cfg.weight_prefix, "norm.weight"), .s = s };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
    }

    pub fn routeLayer(self: *const Model, index: usize, ops: *Ops, x: Arr) !Routed {
        if (index >= self.layers.len) return error.InvalidGlmLayer;
        return switch (self.layers[index].ffn) {
            .moe => |moe| route(ops, x, moe.weight, moe.correction, @intCast(self.cfg.num_experts_per_tok), self.cfg.router_scaling_factor, self.cfg.moe_route_norm),
            .dense => error.GlmLayerNotRouted,
        };
    }

    pub fn feedForwardLayer(self: *const Model, index: usize, ops: *Ops, x: Arr) !Arr {
        if (index >= self.layers.len) return error.InvalidGlmLayer;
        return switch (self.layers[index].ffn) {
            .dense => |dense| dense.apply(ops, x, self.cfg.glm_swiglu_limit),
            .moe => |moe| moe.apply(ops, x, &self.cfg),
        };
    }

    pub fn rawEmbedding(self: *const Model, ids: Arr) !Arr {
        var ops = Ops{ .s = self.s };
        defer ops.deinit();
        const e = self.embedding;
        const code = try ops.take(e.w, ids, 0);
        const hidden = if (e.scales.ctx != null)
            try ops.dequant(code, try ops.take(e.scales, ids, 0), try ops.take(e.biases, ids, 0))
        else
            code;
        return ops.result(hidden);
    }
    pub fn projectHead(self: *const Model, hidden: Arr) !Arr {
        const sh = mlx.getShape(hidden);
        if (sh.len < 2 or sh[sh.len - 1] != self.cfg.hidden_size) return error.InvalidGlmInput;
        var ops = Ops{ .s = self.s };
        defer ops.deinit();
        return ops.result(try self.head.apply(&ops, hidden));
    }

    pub fn forward(self: *const Model, request: *Request, ids: Arr) !Arr {
        return self.forwardLast(request, ids, false);
    }

    pub fn forwardLast(self: *const Model, request: *Request, ids: Arr, last_only: bool) !Arr {
        const ish = mlx.getShape(ids);
        if (ish.len != 2 or ish[0] != 1 or ish[1] < 1 or request.layers.len != self.layers.len) return error.InvalidGlmInput;
        if (request.capture) |capture| {
            if (capture.ids.len != capture.out.len) return error.InvalidGlmCapture;
            for (capture.ids, 0..) |id, i| {
                if (id >= self.layers.len or (i > 0 and id <= capture.ids[i - 1]) or capture.out[i].ctx == null) return error.InvalidGlmCapture;
            }
        }
        if (request.hc_capture) |capture| {
            if (capture.ids.len == 0 or capture.ids.len > self.layers.len or capture.records.len != capture.ids.len * 2) return error.InvalidGlmHcCapture;
            for (capture.ids, 0..) |id, i| if (id >= self.layers.len or (i > 0 and id <= capture.ids[i - 1])) return error.InvalidGlmHcCapture;
        }
        if (request.failed) return error.GlmRequestNeedsReset;
        if (request.offset + @as(usize, @intCast(ish[1])) > self.cfg.max_position_embeddings) return error.GlmContextExceeded;
        errdefer request.failed = true;
        const staged_decode = ish[1] == 1 and request.decode_async and !request.profile and !request.profile_components;
        const staged_prefill = ish[1] > 1 and request.prefill_async and !request.profile and !request.profile_components;
        var h: Arr = undefined;
        {
            var ops = Ops{ .s = self.s };
            defer ops.deinit();
            const e = self.embedding;
            const code = try ops.take(e.w, ids, 0);
            const hidden = if (e.scales.ctx != null) try ops.dequant(code, try ops.take(e.scales, ids, 0), try ops.take(e.biases, ids, 0)) else code;
            h = try ops.result(try ops.contiguous(try ops.broadcast(try ops.reshape(hidden, &.{ 1, ish[1], 1, @intCast(self.cfg.hidden_size) }), &.{ 1, ish[1], 4, @intCast(self.cfg.hidden_size) })));
        }
        defer _ = mlx.mlx_array_free(h);
        for (self.layers, request.layers, 0..) |*layer, *state, layer_index| {
            const timer = if (request.profile) @import("io_util.zig").Stopwatch.init(std.Io.Threaded.global_single_threaded.io()) else null;
            var ops = Ops{ .s = self.s };
            defer ops.deinit();
            var component_clock: ComponentTimer = undefined;
            const component: ?*ComponentTimer = if (request.profile_components) blk: {
                try mlx.check(mlx.mlx_array_eval(h));
                component_clock = .{ .totals = &request.component_ns[layer_index], .clock = .init(std.Io.Threaded.global_single_threaded.io()) };
                break :blk &component_clock;
            } else null;
            if (request.hc_capture) |capture| try capture.retain(layer_index, false, &ops, layer.hc_attn, h, &self.cfg);
            const pre = try layer.hc_attn.collapse(&ops, h, &self.cfg);
            defer pre.deinit();
            try ComponentTimer.mark(component, "hc_attn", &.{ pre.mixed, pre.post, pre.comb }, null);
            const x = try ops.rms(pre.mixed, layer.norm_attn, self.cfg.rms_norm_eps);
            try ComponentTimer.mark(component, "attn_norm", &.{x}, null);
            const a = switch (layer.attn) {
                .kda => |kda| try kda.apply(&ops, x, &self.cfg, &state.recurrent),
                .mla => |*mla| try mla.applyMode(&ops, x, &self.cfg, &state.attention, request.dense_prefill),
            };
            try ComponentTimer.mark(component, "attention", &.{a}, state);
            const joined = try ops.own(try primitive.hcExpand(h, a, pre.post, pre.comb, self.s));
            try ComponentTimer.mark(component, "expand_attn", &.{joined}, null);
            if (request.hc_capture) |capture| try capture.retain(layer_index, true, &ops, layer.hc_ffn, joined, &self.cfg);
            const ff = try layer.hc_ffn.collapse(&ops, joined, &self.cfg);
            defer ff.deinit();
            try ComponentTimer.mark(component, "hc_ffn", &.{ ff.mixed, ff.post, ff.comb }, null);
            const fx = try ops.rms(ff.mixed, layer.norm_ffn, self.cfg.rms_norm_eps);
            try ComponentTimer.mark(component, "ffn_norm", &.{fx}, null);
            const y = switch (layer.ffn) {
                .dense => |dense| blk: {
                    const y = try dense.apply(&ops, fx, self.cfg.glm_swiglu_limit);
                    try ComponentTimer.mark(component, "dense_ffn", &.{y}, null);
                    break :blk y;
                },
                .moe => |moe| try moe.applyTracked(&ops, fx, &self.cfg, component),
            };
            const next = try ops.own(try primitive.hcExpand(joined, y, ff.post, ff.comb, self.s));
            try ComponentTimer.mark(component, "expand_ffn", &.{next}, null);
            if (request.capture) |capture| {
                for (capture.ids, 0..) |id, i| {
                    if (id == layer_index) try mlx.check(mlx.mlx_array_set(&capture.out[i], try ops.reduce(next, 2, true, false)));
                }
            }
            if (staged_prefill) {
                const evals = mlx.mlx_vector_array_new_value(next);
                defer _ = mlx.mlx_vector_array_free(evals);
                const first = layer_index - layer_index % 2;
                try appendCaptures(evals, request.capture, request.hc_capture, first, layer_index + 1);
                for (request.layers[first .. layer_index + 1]) |*pending| try appendLayerState(evals, pending);
                if (layer_index % 2 == 0) {
                    try mlx.check(mlx.mlx_async_eval(evals));
                    if (@import("builtin").is_test) schedule_test_asyncs += 1;
                } else {
                    try mlx.check(mlx.mlx_eval(evals));
                    if (@import("builtin").is_test) schedule_test_syncs += 1;
                }
            } else if (!staged_decode or (layer_index + 1) % 4 == 0) {
                const evals = mlx.mlx_vector_array_new_value(next);
                defer _ = mlx.mlx_vector_array_free(evals);
                if (staged_decode) {
                    try appendCaptures(evals, request.capture, request.hc_capture, layer_index - 3, layer_index + 1);
                    for (request.layers[layer_index - 3 .. layer_index + 1]) |*pending| try appendLayerState(evals, pending);
                    try mlx.check(mlx.mlx_async_eval(evals));
                    if (@import("builtin").is_test) schedule_test_asyncs += 1;
                } else {
                    try appendCaptures(evals, request.capture, request.hc_capture, layer_index, layer_index + 1);
                    try appendLayerState(evals, state);
                    try mlx.check(mlx.mlx_eval(evals));
                    if (@import("builtin").is_test) schedule_test_syncs += 1;
                }
            }
            try mlx.check(mlx.mlx_array_set(&h, next));
            if (timer) |clock| request.layer_ns[layer_index] += clock.read();
        }
        var ops = Ops{ .s = self.s };
        defer ops.deinit();
        const chosen = if (last_only) try ops.slice(h, 1, ish[1] - 1, ish[1]) else h;
        const normalized = try ops.rms(try ops.reduce(chosen, 2, true, false), self.norm, self.cfg.rms_norm_eps);
        const logits = try self.head.apply(&ops, normalized);
        const result = try ops.result(logits);
        errdefer _ = mlx.mlx_array_free(result);
        if (staged_decode or staged_prefill or request.capture != null or request.hc_capture != null) {
            // Cache side outputs must settle even when they are not ancestors of logits.
            const evals = mlx.mlx_vector_array_new_value(result);
            defer _ = mlx.mlx_vector_array_free(evals);
            for (request.layers) |*state| try appendLayerState(evals, state);
            try appendCaptures(evals, request.capture, request.hc_capture, 0, self.layers.len);
            try mlx.check(mlx.mlx_eval(evals));
            if (@import("builtin").is_test) schedule_test_syncs += 1;
        }
        request.offset += @intCast(ish[1]);
        return result;
    }
};

fn fixtureTensor(weights: *model.Weights, prefix: []const u8, suffix: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype, ones: bool) !void {
    const key = try std.fmt.allocPrint(std.testing.allocator, "{s}.{s}", .{ prefix, suffix });
    errdefer std.testing.allocator.free(key);
    var arr = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(arr);
    if (ones) try mlx.check(mlx.mlx_ones(&arr, shape.ptr, shape.len, dtype, mlx.gpuStream())) else try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, shape.len, dtype, mlx.gpuStream()));
    try weights.map.put(key, arr);
}

fn fixtureMlp(weights: *model.Weights, prefix: []const u8) !void {
    for ([_][]const u8{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" }) |leaf|
        try fixtureTensor(weights, prefix, leaf, &.{ 128, 128 }, .bfloat16, false);
}

pub fn completeFixture(weights: *model.Weights) !model.ModelConfig {
    const a = std.testing.allocator;
    const cfg = model.ModelConfig{
        .model_type = "glm5_next",
        .weight_prefix = "model.language_model",
        .hidden_size = 128,
        .vocab_size = 4,
        .num_hidden_layers = 4,
        .first_k_dense_replace = 3,
        .intermediate_size = 128,
        .moe_intermediate_size = 128,
        .full_attention_interval = 4,
        .num_experts = 2,
        .num_experts_per_tok = 1,
        .shared_expert_intermediate_size = 128,
        .linear_num_value_heads = 1,
        .linear_key_head_dim = 128,
        .linear_conv_kernel_dim = 4,
        .kda_gate_lower_bound = -5,
        .hc_count = 4,
        .glm_hc_sinkhorn_iters = 20,
        .glm_hc_eps = 1e-6,
        .glm_swiglu_limit = 10,
        .mla_q_lora_rank = 128,
        .mla_kv_lora_rank = 128,
        .mla_qk_nope_head_dim = 128,
        .mla_v_head_dim = 128,
        .num_attention_heads = 1,
        .indexer_n_heads = 1,
        .indexer_head_dim = 128,
        .indexer_compress_ratio = 4,
        .indexer_budget = 2048,
        .max_position_embeddings = 16,
        .rms_norm_eps = 1e-5,
        .expert_quant_codebook = .mcg,
        .expert_quant_window = .w12,
    };
    const codes: [4 * 32]u32 = @splat(0x01010101);
    for ([_][]const u8{ "model.language_model.embed_tokens", "lm_head" }) |name| {
        const key = try std.fmt.allocPrint(a, "{s}.weight", .{name});
        const code = mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 32 }, 2, .uint32);
        try weights.map.put(key, code);
        try fixtureTensor(weights, name, "scales", &.{ 4, 1 }, .bfloat16, true);
        try fixtureTensor(weights, name, "biases", &.{ 4, 1 }, .bfloat16, false);
    }
    try fixtureTensor(weights, cfg.weight_prefix, "norm.weight", &.{128}, .bfloat16, true);

    for (0..4) |i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}", .{i});
        for ([_][]const u8{ "hc_attn_fn", "hc_ffn_fn" }) |k| try fixtureTensor(weights, p, k, &.{ 24, 512 }, .float32, false);
        for ([_][]const u8{ "hc_attn_scale", "hc_ffn_scale" }) |k| try fixtureTensor(weights, p, k, &.{3}, .float32, false);
        for ([_][]const u8{ "hc_attn_base", "hc_ffn_base" }) |k| try fixtureTensor(weights, p, k, &.{24}, .float32, false);
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |k| try fixtureTensor(weights, p, k, &.{128}, .bfloat16, true);
        var sb: [160]u8 = undefined;
        const ap = try std.fmt.bufPrint(&sb, "{s}.self_attn", .{p});
        if (i < 3) {
            for ([_][]const u8{ "q_proj.weight", "k_proj.weight", "v_proj.weight", "f_a_proj.weight", "f_b_proj.weight", "g_a_proj.weight", "g_b_proj.weight", "o_proj.weight" }) |k| try fixtureTensor(weights, ap, k, &.{ 128, 128 }, .bfloat16, false);
            try fixtureTensor(weights, ap, "b_proj.weight", &.{ 1, 128 }, .bfloat16, false);
            for ([_][]const u8{ "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight" }) |k| try fixtureTensor(weights, ap, k, &.{ 128, 1, 4 }, .bfloat16, false);
            try fixtureTensor(weights, ap, "A_log", &.{1}, .float32, false);
            try fixtureTensor(weights, ap, "dt_bias", &.{128}, .float32, false);
            try fixtureTensor(weights, ap, "o_norm.weight", &.{128}, .bfloat16, true);
        } else {
            for ([_][]const u8{ "q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight", "o_proj.weight", "indexer.wq_b.weight", "indexer.wk.weight" }) |k| try fixtureTensor(weights, ap, k, &.{ 128, 128 }, .bfloat16, false);
            try fixtureTensor(weights, ap, "kv_b_proj.weight", &.{ 256, 128 }, .bfloat16, false);
            for ([_][]const u8{ "q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight" }) |k| try fixtureTensor(weights, ap, k, &.{128}, .bfloat16, true);
            try fixtureTensor(weights, ap, "indexer.k_norm.bias", &.{128}, .bfloat16, false);
            try fixtureTensor(weights, ap, "indexer.weights_proj.weight", &.{ 1, 128 }, .bfloat16, false);
            try fixtureTensor(weights, ap, "indexer.index_kpool_compress_gate", &.{ 128, 128 }, .bfloat16, false);
            try fixtureTensor(weights, ap, "indexer.index_kpool_compress_ape", &.{ 4, 128 }, .bfloat16, false);
        }
        const mp = try std.fmt.bufPrint(&sb, "{s}.mlp", .{p});
        if (i < 3) try fixtureMlp(weights, mp) else {
            try fixtureTensor(weights, mp, "gate.weight", &.{ 2, 128 }, .float32, false);
            try fixtureTensor(weights, mp, "gate.e_score_correction_bias", &.{2}, .float32, false);
            var pb: [220]u8 = undefined;
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
                const name = try std.fmt.bufPrint(&pb, "{s}.switch_mlp.{s}", .{ mp, proj });
                try fixtureTensor(weights, name, "trellis", &.{ 2, 8, 8, 36 }, .uint16, false);
                try fixtureTensor(weights, name, "suh", &.{ 2, 128 }, .float16, false);
                try fixtureTensor(weights, name, "svh", &.{ 2, 128 }, .float16, false);
            }
            try fixtureMlp(weights, try std.fmt.bufPrint(&pb, "{s}.shared_experts", .{mp}));
        }
    }
    return cfg;
}

test "GLM complete diagnostic forward advances and resets request state" {
    const a = std.testing.allocator;
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const cfg = try completeFixture(&weights);
    var mdl = try Model.load(a, cfg, &weights, mlx.gpuStream());
    defer mdl.deinit();
    var req = try Request.init(a, 4);
    defer req.deinit();
    const ids = mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &[_]c_int{ 1, 3 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const out = try mdl.forward(&req, ids);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    try std.testing.expectEqualSlices(c_int, &.{ 1, 3, 4 }, mlx.getShape(out));
    try std.testing.expectEqual(@as(usize, 3), req.offset);
    try std.testing.expectEqual(@as(usize, 3), req.layers[3].attention.processed);
    req.reset();
    try std.testing.expectEqual(@as(usize, 0), req.offset);
    const again = try mdl.forwardLast(&req, ids, true);
    defer _ = mlx.mlx_array_free(again);
    try mlx.check(mlx.mlx_array_eval(again));
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const fp = try ops.cast(again, .float32);
    try mlx.check(mlx.mlx_array_eval(fp));
    for (mlx.mlx_array_data_float32(fp).?[0..4]) |v| try std.testing.expectApproxEqAbs(@as(f32, 128), v, 0.01);
}

fn mlaOrientationFixture(weights: *model.Weights) !model.ModelConfig {
    const cfg = model.ModelConfig{ .hidden_size = 128, .mla_q_lora_rank = 128, .mla_kv_lora_rank = 128, .mla_qk_nope_head_dim = 128, .mla_v_head_dim = 128, .num_attention_heads = 2, .indexer_n_heads = 2, .indexer_head_dim = 128 };
    for ([_][]const u8{ "q_a_proj.weight", "kv_a_proj_with_mqa.weight", "indexer.wk.weight" }) |name| try fixtureTensor(weights, "mla", name, &.{ 128, 128 }, .bfloat16, false);
    for ([_][]const u8{ "q_b_proj.weight", "indexer.wq_b.weight" }) |name| try fixtureTensor(weights, "mla", name, &.{ 256, 128 }, .bfloat16, false);
    try fixtureTensor(weights, "mla", "o_proj.weight", &.{ 128, 256 }, .bfloat16, false);
    try fixtureTensor(weights, "mla", "indexer.weights_proj.weight", &.{ 2, 128 }, .bfloat16, false);
    for ([_][]const u8{ "q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight" }) |name| try fixtureTensor(weights, "mla", name, &.{128}, .bfloat16, true);
    try fixtureTensor(weights, "mla", "indexer.k_norm.bias", &.{128}, .bfloat16, false);
    try fixtureTensor(weights, "mla", "indexer.index_kpool_compress_gate", &.{ 128, 128 }, .bfloat16, false);
    try fixtureTensor(weights, "mla", "indexer.index_kpool_compress_ape", &.{ 4, 128 }, .bfloat16, false);
    var codes: [512 * 32]u32 = undefined;
    var scales: [512]u16 = undefined;
    var biases: [512]u16 = undefined;
    for (&codes, 0..) |*v, i| {
        const b: @TypeOf(v.*) = @intCast(1 + (i % 7));
        v.* = b | (b + 1) << 8 | (b + 2) << 16 | (b + 3) << 24;
    }
    for (&scales, &biases, 0..) |*sc, *bias, i| {
        const f: f32 = @as(f32, @floatFromInt(1 + i % 3)) / 128;
        sc.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        const b: f32 = -@as(f32, @floatFromInt(i % 3)) / 64;
        bias.* = @truncate(@as(u32, @bitCast(b)) >> 16);
    }
    const a = std.testing.allocator;
    try weights.map.put(try a.dupe(u8, "mla.kv_b_proj.weight"), mlx.mlx_array_new_data(&codes, &[_]c_int{ 512, 32 }, 2, .uint32));
    try weights.map.put(try a.dupe(u8, "mla.kv_b_proj.scales"), mlx.mlx_array_new_data(&scales, &[_]c_int{ 512, 1 }, 2, .bfloat16));
    try weights.map.put(try a.dupe(u8, "mla.kv_b_proj.biases"), mlx.mlx_array_new_data(&biases, &[_]c_int{ 512, 1 }, 2, .bfloat16));
    return cfg;
}

test "GLM MLA loader refuses invalid projection and norm geometry" {
    var weights = model.Weights.init(std.testing.allocator);
    defer weights.deinit();
    const cfg = try mlaOrientationFixture(&weights);
    const Case = struct { name: []const u8, shape: []const c_int };
    for ([_]Case{ .{ .name = "mla.q_a_layernorm.weight", .shape = &.{1} }, .{ .name = "mla.q_b_proj.weight", .shape = &.{ 128, 128 } } }) |case| {
        const slot = weights.map.getPtr(case.name).?;
        const original = slot.*;
        slot.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&slot.*, case.shape.ptr, case.shape.len, .bfloat16, mlx.gpuStream()));
        defer {
            _ = mlx.mlx_array_free(slot.*);
            slot.* = original;
        }
        if (Mla.load(&weights, "mla", &cfg, mlx.gpuStream())) |loaded| {
            var bad = loaded;
            bad.deinit();
            return error.TestExpectedError;
        } else |err| try std.testing.expectEqual(error.InvalidGlmMlaWeight, err);
    }
}

test "GLM MLA stored affine kv rows preserve both projection orientations" {
    const s = mlx.gpuStream();
    var weights = model.Weights.init(std.testing.allocator);
    defer weights.deinit();
    const cfg = try mlaOrientationFixture(&weights);
    var layer = try Mla.load(&weights, "mla", &cfg, s);
    defer layer.deinit();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const kvb = try linear(&weights, "mla", "kv_b_proj", 128);
    const dense = try ops.reshape(try ops.dequant(kvb.w, kvb.scales, kvb.biases), &.{ 2, 256, 128 });
    const dk = try ops.slice(dense, 1, 0, 128);
    const dv = try ops.slice(dense, 1, 128, 256);
    var input: [2 * 2 * 128]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 9)) - 4) / 32;
    const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&input, &[_]c_int{ 2, 2, 1, 128 }, 4, .float32)), .bfloat16);
    const key_quant = try ops.cast(try ops.qmm(x, layer.wk, layer.sk, layer.bk, false), .float32);
    const key_dense = try ops.cast(try ops.binary(.mm, x, dk), .float32);
    const value_quant = try ops.cast(try ops.qmm(x, layer.wv, layer.sv, layer.bv, true), .float32);
    const value_dense = try ops.cast(try ops.binary(.mm, x, try ops.transpose(dv, &.{ 0, 2, 1 })), .float32);
    for ([_]Arr{ key_quant, key_dense, value_quant, value_dense }) |v| try mlx.check(mlx.mlx_array_eval(v));
    try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(key_dense).?[0..512], mlx.mlx_array_data_float32(key_quant).?[0..512]);
    try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(value_dense).?[0..512], mlx.mlx_array_data_float32(value_quant).?[0..512]);
    try std.testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(layer.wk));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(layer.sk));
}

fn nonzeroDecodeFixture(weights: *model.Weights) !model.ModelConfig {
    var cfg = try completeFixture(weights);
    cfg.max_position_embeddings = 512;
    var iter = weights.map.iterator();
    var seed: usize = 0;
    while (iter.next()) |entry| {
        const name = entry.key_ptr.*;
        const old = entry.value_ptr.*;
        const dtype = mlx.mlx_array_dtype(old);
        if (dtype != .bfloat16 and dtype != .float32 and dtype != .float16) continue;
        if (std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases") or std.mem.indexOf(u8, name, "norm.weight") != null) continue;
        const count = mlx.mlx_array_size(old);
        const host = try std.testing.allocator.alloc(f32, count);
        defer std.testing.allocator.free(host);
        for (host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i + seed) % 11)) - 5) / 256;
        if (std.mem.endsWith(u8, name, ".suh") or std.mem.endsWith(u8, name, ".svh")) @memset(host, 0.125);
        if (std.mem.endsWith(u8, name, "_scale")) @memset(host, 0.25);
        const shape = mlx.getShape(old);
        const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(f);
        var value = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(value);
        try mlx.check(mlx.mlx_astype(&value, f, dtype, mlx.gpuStream()));
        _ = mlx.mlx_array_free(old);
        entry.value_ptr.* = value;
        seed += 1;
    }
    return cfg;
}

fn expectArrayBits(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const av = try ops.cast(a, .float32);
    const bv = try ops.cast(b, .float32);
    try mlx.check(mlx.mlx_array_eval(av));
    try mlx.check(mlx.mlx_array_eval(bv));
    const count = mlx.mlx_array_size(av);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(av).?[0..count]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(bv).?[0..count]));
}

fn expectRequestBits(a: *Request, b: *Request) !void {
    try std.testing.expectEqual(a.offset, b.offset);
    for (a.layers, b.layers) |*left, *right| {
        try std.testing.expectEqual(left.recurrent.initialized, right.recurrent.initialized);
        if (left.recurrent.initialized) {
            try expectArrayBits(left.recurrent.conv_state, right.recurrent.conv_state);
            try expectArrayBits(left.recurrent.ssm_state, right.recurrent.ssm_state);
        }
        try std.testing.expectEqual(left.attention.processed, right.attention.processed);
        for (left.attention.arrays(), right.attention.arrays()) |x, y| {
            try std.testing.expectEqual(x.ctx == null, y.ctx == null);
            if (x.ctx != null) try expectArrayBits(x, y);
        }
    }
}

test "GLM async decode preserves nonzero logits and every cache state" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const cfg = try nonzeroDecodeFixture(&weights);
    var net = try Model.load(a, cfg, &weights, s);
    defer net.deinit();
    var sync = try Request.init(a, 4);
    defer sync.deinit();
    sync.decode_async = false;
    var staged = try Request.init(a, 4);
    defer staged.deinit();
    staged.decode_async = true;
    for ([_]usize{ 3, 1, 1, 1, 1, 1, 1, 1 }) |width| {
        const tokens = [_]u32{ 1, 2, 3 };
        const ids = mlx.mlx_array_new_data(&tokens, &[_]c_int{ 1, @intCast(width) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const x = try net.forward(&sync, ids);
        defer _ = mlx.mlx_array_free(x);
        schedule_test_syncs = 0;
        schedule_test_asyncs = 0;
        const y = try net.forward(&staged, ids);
        defer _ = mlx.mlx_array_free(y);
        try std.testing.expectEqual(@as(usize, if (width == 1) 1 else 4), schedule_test_syncs);
        try std.testing.expectEqual(@as(usize, if (width == 1) 1 else 0), schedule_test_asyncs);
        try expectArrayBits(x, y);
        try expectRequestBits(&sync, &staged);
    }
    const state = sync.layers[0].recurrent.ssm_state;
    try mlx.check(mlx.mlx_array_eval(state));
    var nonzero: usize = 0;
    for (mlx.mlx_array_data_float32(state).?[0..mlx.mlx_array_size(state)]) |v| {
        if (v != 0) nonzero += 1;
    }
    try std.testing.expect(nonzero > 0);
    sync.reset();
    staged.reset();
    try expectRequestBits(&sync, &staged);
}

test "GLM async decode settles graphs and profiling retains layer boundaries" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const cfg = try nonzeroDecodeFixture(&weights);
    var net = try Model.load(a, cfg, &weights, s);
    defer net.deinit();
    var request = try Request.init(a, 4);
    defer request.deinit();
    const ids = mlx.mlx_array_new_data(&[_]u32{1}, &[_]c_int{ 1, 1 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    var baseline: usize = 0;
    for (0..80) |i| {
        const result = try net.forwardLast(&request, ids, true);
        _ = mlx.mlx_array_free(result);
        if (i == 15) try mlx.check(mlx.mlx_get_active_memory(&baseline));
    }
    var active: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    // Both snapshots end on a complete pool with the same reserved cache capacity.
    try std.testing.expect(active <= baseline + 64 * 1024);
    try std.testing.expectEqual(@as(usize, 80), request.offset);
    request.profile = true;
    schedule_test_syncs = 0;
    schedule_test_asyncs = 0;
    const result = try net.forwardLast(&request, ids, true);
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_array_eval(result));
    try std.testing.expectEqual(@as(usize, 4), schedule_test_syncs);
    try std.testing.expectEqual(@as(usize, 0), schedule_test_asyncs);
    for (request.layer_ns[0..4]) |ns| try std.testing.expect(ns > 0);
}

test "GLM two-layer prefill schedule preserves every nonzero cache bit" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const cfg = try nonzeroDecodeFixture(&weights);
    var net = try Model.load(a, cfg, &weights, s);
    defer net.deinit();
    var baseline = try Request.init(a, 4);
    defer baseline.deinit();
    var pipelined = try Request.init(a, 4);
    defer pipelined.deinit();
    pipelined.prefill_async = true;
    for ([_]usize{ 17, 33, 2 }) |width| {
        var tokens: [33]u32 = undefined;
        for (tokens[0..width], 0..) |*v, i| v.* = @intCast(i % 4);
        const ids = mlx.mlx_array_new_data(&tokens, &[_]c_int{ 1, @intCast(width) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const expected = try net.forwardLast(&baseline, ids, true);
        defer _ = mlx.mlx_array_free(expected);
        schedule_test_asyncs = 0;
        schedule_test_syncs = 0;
        const got = try net.forwardLast(&pipelined, ids, true);
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(usize, 2), schedule_test_asyncs);
        try std.testing.expectEqual(@as(usize, 3), schedule_test_syncs);
        try expectArrayBits(expected, got);
        try expectRequestBits(&baseline, &pipelined);
    }
    pipelined.profile = true;
    const ids = mlx.mlx_array_new_data(&[_]u32{ 1, 2 }, &[_]c_int{ 1, 2 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    schedule_test_asyncs = 0;
    schedule_test_syncs = 0;
    const y = try net.forwardLast(&pipelined, ids, true);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    try std.testing.expectEqual(@as(usize, 0), schedule_test_asyncs);
    try std.testing.expectEqual(@as(usize, 4), schedule_test_syncs);
}

test "GLM post-layer capture and shared draft projections retain their contracts" {
    const a = std.testing.allocator;
    const stream = mlx.gpuStream();
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const cfg = try completeFixture(&weights);
    var net = try Model.load(a, cfg, &weights, stream);
    defer net.deinit();
    var request = try Request.init(a, 4);
    defer request.deinit();
    var outputs = [_]Arr{ mlx.mlx_array_new_float(0), mlx.mlx_array_new_float(0) };
    defer for (outputs) |v| {
        _ = mlx.mlx_array_free(v);
    };
    var capture = Capture{ .ids = &.{ 0, 3 }, .out = &outputs };
    request.capture = &capture;
    const ids = mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &[_]c_int{ 1, 3 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try net.forwardLast(&request, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    var ops = Ops{ .s = stream };
    defer ops.deinit();
    for (outputs) |out| {
        try std.testing.expectEqualSlices(c_int, &.{ 1, 3, 128 }, mlx.getShape(out));
        const f = try ops.cast(out, .float32);
        try mlx.check(mlx.mlx_array_eval(f));
        for (mlx.mlx_array_data_float32(f).?[0..384]) |v| try std.testing.expectEqual(@as(f32, 1), v);
    }
    const embedding = try ops.own(try net.rawEmbedding(ids));
    try expectArrayBits(embedding, outputs[0]);
    const twice = try ops.binary(.mul, embedding, try ops.scalar(2, .bfloat16));
    const projected = try ops.own(try net.projectHead(twice));
    const f = try ops.cast(projected, .float32);
    try mlx.check(mlx.mlx_array_eval(f));
    for (mlx.mlx_array_data_float32(f).?[0..12]) |v| try std.testing.expectEqual(@as(f32, 256), v);
    request.reset();
    capture.ids = &.{ 3, 0 };
    try std.testing.expectError(error.InvalidGlmCapture, net.forwardLast(&request, ids, true));
    try std.testing.expectEqual(@as(usize, 0), request.offset);
}

test "GLM fused router matches FP32 scores and selected order at production width" {
    const a = std.testing.allocator;
    const stream = mlx.gpuStream();
    if (!@import("glm5_kda_fused.zig").hardwareSupported()) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(58289);
    const rnd = random.random();
    const matrix = try a.alloc(f32, 288 * 4096);
    defer a.free(matrix);
    var vector: [4096]f32 = undefined;
    var bias: [288]f32 = undefined;
    for (matrix) |*v| v.* = (rnd.float(f32) - 0.5) / 16;
    for (&vector) |*v| v.* = (rnd.float(f32) - 0.5) * 4;
    for (&bias) |*v| v.* = (rnd.float(f32) - 0.5) / 4;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |weight_dtype| for ([_]bool{ true, false }) |norm| for ([_]bool{ false, true }) |ties| {
        if (ties) {
            @memset(matrix, 0);
            @memset(&bias, 0);
        } else {
            for (matrix) |*v| v.* = (rnd.float(f32) - 0.5) / 16;
            for (&bias) |*v| v.* = (rnd.float(f32) - 0.5) / 4;
        }
        var ops = Ops{ .s = stream };
        defer ops.deinit();
        const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&vector, &[_]c_int{ 1, 1, 4096 }, 3, .float32)), dtype);
        const w = try ops.cast(try ops.own(mlx.mlx_array_new_data(matrix.ptr, &[_]c_int{ 288, 4096 }, 2, .float32)), weight_dtype);
        try mlx.check(mlx.mlx_array_eval(w));
        const correction = try ops.own(mlx.mlx_array_new_data(&bias, &[_]c_int{288}, 1, .float32));
        const expected = try routeReference(&ops, x, w, correction, 8, 2.5, norm);
        const got = (try @import("glm5_router.zig").route(stream, x, w, correction, 8, 2.5, norm)) orelse return error.TestExpectedFusedRouter;
        _ = try ops.own(got.indices);
        _ = try ops.own(got.scores);
        try expectArrayBits(got.indices, expected.indices);
        try expectArrayBits(got.scores, expected.scores);
    };
}

test "GLM component profiling preserves logits and cache state and resets counters" {
    const allocator = std.testing.allocator;
    const stream = mlx.gpuStream();
    var weights = model.Weights.init(allocator);
    defer weights.deinit();
    const cfg = try nonzeroDecodeFixture(&weights);
    var net = try Model.load(allocator, cfg, &weights, stream);
    defer net.deinit();
    var reference = try Request.init(allocator, 4);
    defer reference.deinit();
    var measured = try Request.init(allocator, 4);
    defer measured.deinit();
    measured.profile_components = true;
    for ([_]usize{ 3, 1, 1 }) |width| {
        const tokens = [_]u32{ 1, 2, 3 };
        const ids = mlx.mlx_array_new_data(&tokens, &.{ 1, @intCast(width) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const x = try net.forward(&reference, ids);
        defer _ = mlx.mlx_array_free(x);
        schedule_test_asyncs = 0;
        const y = try net.forward(&measured, ids);
        defer _ = mlx.mlx_array_free(y);
        try std.testing.expectEqual(@as(usize, 0), schedule_test_asyncs);
        try expectArrayBits(x, y);
        try expectRequestBits(&reference, &measured);
    }
    for (measured.component_ns[0..4]) |value| {
        try std.testing.expect(value.hc_attn > 0);
        try std.testing.expect(value.attention > 0);
        try std.testing.expect(value.hc_ffn > 0);
        try std.testing.expect(value.expand_ffn > 0);
    }
    try std.testing.expect(measured.component_ns[0].dense_ffn > 0);
    try std.testing.expect(measured.component_ns[3].routed > 0);
    measured.reset();
    try std.testing.expectEqual(ComponentTimes{}, measured.component_ns[0]);
    try std.testing.expectEqual(ComponentTimes{}, measured.component_ns[3]);
}

test "GLM HC capture retains reference fixtures without changing logits or state" {
    const a = std.testing.allocator;
    var weights = model.Weights.init(a);
    defer weights.deinit();
    var cfg = try nonzeroDecodeFixture(&weights);
    cfg.max_position_embeddings = 64;
    var net = try Model.load(a, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    var plain = try Request.init(a, 4);
    defer plain.deinit();
    var captured = try Request.init(a, 4);
    defer captured.deinit();
    var records: [4]HcSnapshot = @splat(.{});
    defer for (&records) |*r| r.deinit();
    var capture = HcCapture{ .ids = &.{ 0, 3 }, .records = &records };
    captured.hc_capture = &capture;
    for ([_]c_int{ 17, 1 }) |rows| {
        var ops = Ops{ .s = net.s };
        defer ops.deinit();
        const ids = try ops.zeros(&.{ 1, rows }, .uint32);
        const expected = try ops.own(try net.forwardLast(&plain, ids, true));
        const got = try ops.own(try net.forwardLast(&captured, ids, true));
        try expectArrayBits(expected, got);
        try expectRequestBits(&plain, &captured);
        for (&records) |*r| {
            try std.testing.expectEqualSlices(c_int, &.{ 1, rows, 4, 128 }, mlx.getShape(r.values[0]));
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(r.values[4]));
            for (r.values) |v| try std.testing.expect(v.ctx != null);
        }
    }
    const offset = captured.offset;
    var ops = Ops{ .s = net.s };
    defer ops.deinit();
    const ids = try ops.zeros(&.{ 1, 1 }, .uint32);
    for ([_][]const u32{ &.{ 3, 0 }, &.{ 0, 0 }, &.{ 0, 4 }, &.{0}, &.{} }) |bad| {
        capture.ids = bad;
        try std.testing.expectError(error.InvalidGlmHcCapture, net.forwardLast(&captured, ids, true));
        try std.testing.expectEqual(offset, captured.offset);
        try std.testing.expect(!captured.failed);
        try expectRequestBits(&plain, &captured);
    }
}

test "GLM HC prefill policy preserves nonzero small model logits and every cache bit" {
    const hc_prefill = @import("glm5_hc_prefill.zig");
    defer hc_prefill.testSetEnabled(null);
    var weights = model.Weights.init(std.testing.allocator);
    defer weights.deinit();
    const cfg = try nonzeroDecodeFixture(&weights);
    var net = try Model.load(std.testing.allocator, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    var reference = try Request.init(std.testing.allocator, 4);
    defer reference.deinit();
    var candidate = try Request.init(std.testing.allocator, 4);
    defer candidate.deinit();
    const before = hc_prefill.dispatchCount();
    for ([_]c_int{ 128, 17, 1 }) |rows| {
        var ops = Ops{ .s = net.s };
        defer ops.deinit();
        var tokens: [128]u32 = undefined;
        for (tokens[0..@intCast(rows)], 0..) |*v, i| v.* = @intCast(i % 4);
        const ids = try ops.own(mlx.mlx_array_new_data(&tokens, &.{ 1, rows }, 2, .uint32));
        hc_prefill.testSetEnabled(false);
        const expected = try ops.own(try net.forwardLast(&reference, ids, true));
        hc_prefill.testSetEnabled(true);
        const actual = try ops.own(try net.forwardLast(&candidate, ids, true));
        try expectArrayBits(expected, actual);
        try expectRequestBits(&reference, &candidate);
    }
    // Hidden128 intentionally falls back. Production-width engagement is covered by HC fixtures.
    try std.testing.expectEqual(before, hc_prefill.dispatchCount());
}

test "GLM DFlash asynchronous schedules preserve nonzero tapes captures and committed state" {
    const verifier = @import("glm5_dflash_model.zig");
    var weights = model.Weights.init(std.testing.allocator);
    defer weights.deinit();
    const cfg = try nonzeroDecodeFixture(&weights);
    var net = try Model.load(std.testing.allocator, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    var request = try Request.init(std.testing.allocator, 4);
    defer request.deinit();
    var ops = Ops{ .s = net.s };
    defer ops.deinit();
    const prefix = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &.{ 1, 3 }, 2, .uint32));
    _ = try ops.own(try net.forwardLast(&request, prefix, true));
    const tokens = [_]u32{ 1, 0, 2, 0, 3 };
    const parents = [_]i32{ -1, 0, 0, 1, 2 };
    const taps = [_]u32{ 0, 3 };
    const baseline_binding = try verifier.bindSchedule(0);
    defer baseline_binding.restore();
    var reference = try verifier.verify(&net, &request, &tokens, &parents, &taps, .affine_rows_ffn);
    defer reference.deinit();
    for ([_]usize{ 2, 4 }) |cadence| {
        const binding = try verifier.bindSchedule(cadence);
        defer binding.restore();
        verifier.resetStats();
        var candidate = try verifier.verify(&net, &request, &tokens, &parents, &taps, .affine_rows_ffn);
        defer candidate.deinit();
        try std.testing.expectEqual(@as(usize, 4) / cadence, verifier.asyncDispatchCount());
        try std.testing.expectEqual(@as(usize, 1), verifier.syncDispatchCount());
        try std.testing.expectEqualSlices(u32, reference.targets[0..reference.count], candidate.targets[0..candidate.count]);
        for (reference.captures.hook.out, candidate.captures.hook.out) |x, y| try expectArrayBits(x, y);
        for (reference.layers, candidate.layers) |x, y| switch (x.?) {
            .kda => |left| {
                const right = y.?.kda;
                for ([_]Arr{ left.inputs.q, left.inputs.k, left.inputs.v, left.inputs.decay, left.inputs.beta, left.inputs.state, left.conv_input }, [_]Arr{ right.inputs.q, right.inputs.k, right.inputs.v, right.inputs.decay, right.inputs.beta, right.inputs.state, right.conv_input }) |a, b| try expectArrayBits(a, b);
            },
            .mla => |left| {
                const right = y.?.mla;
                for ([_]Arr{ left.latent, left.keys, left.gates }, [_]Arr{ right.latent, right.keys, right.gates }) |a, b| try expectArrayBits(a, b);
            },
        };
        for ([_]usize{ 1, 3, 5 }) |budget| {
            var expected = try reference.prepareCommit(&request, budget, &.{}, net.s);
            defer expected.deinit();
            var actual = try candidate.prepareCommit(&request, budget, &.{}, net.s);
            defer actual.deinit();
            for (&expected.states, &actual.states) |*a, *b| {
                try std.testing.expectEqual(a.* != null, b.* != null);
                if (a.*) |*left| try expectRequestBits(left, &b.*.?);
            }
        }
        var profile: @import("glm5_dflash_profile.zig").Profile = .{};
        const profile_binding = @import("glm5_dflash_profile.zig").bind(&profile);
        defer profile_binding.restore();
        verifier.resetStats();
        var profiled = try verifier.verify(&net, &request, &tokens, &parents, &taps, .affine_rows_ffn);
        defer profiled.deinit();
        try std.testing.expectEqual(@as(usize, 0), verifier.asyncDispatchCount());
        try std.testing.expectEqual(@as(usize, 5), verifier.syncDispatchCount());
        try std.testing.expectEqualSlices(u32, reference.targets[0..reference.count], profiled.targets[0..profiled.count]);
    }
    try std.testing.expectError(error.InvalidGlmVerifySchedule, verifier.bindSchedule(3));
}
