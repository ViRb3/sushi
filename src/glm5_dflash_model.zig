//! Layerwise GLM tree verification with bounded branch-state lifetime.
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const base = @import("glm5_model.zig");
const primitive = @import("glm5_next.zig");
const attention = @import("glm5_attention.zig");
const kda = @import("glm5_dflash_kda.zig");
const tree = @import("glm5_dflash_tree.zig");
const adapter = @import("glm5_dflash.zig");
const Arr = mlx.mlx_array;
const Ops = base.Ops;

fn ancestry(parents: []const i32, row: usize, out: *[16]u32) []const u32 {
    var count: usize = 0;
    var at: i32 = @intCast(row);
    while (at >= 0) : (at = parents[@intCast(at)]) {
        out[count] = @intCast(at);
        count += 1;
    }
    std.mem.reverse(u32, out[0..count]);
    return out[0..count];
}

pub const MlaTape = struct {
    latent: Arr,
    keys: Arr,
    gates: Arr,
    ape: Arr,
    fn deinit(self: *MlaTape) void {
        for ([_]Arr{ self.latent, self.keys, self.gates, self.ape }) |a| _ = mlx.mlx_array_free(a);
    }
    fn append(self: *const MlaTape, state: *attention.State, path: []const u32, s: mlx.mlx_stream) !void {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const indices = try ops.own(mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .uint32));
        _ = try state.append(try ops.take(self.latent, indices, 0), try ops.take(self.keys, indices, 0), try ops.take(self.gates, indices, 0), self.ape, s);
    }
};

fn forkAttention(source: *const attention.State) !attention.State {
    var state = attention.State{ .processed = source.processed };
    errdefer state.deinit();
    inline for (.{ "latent", "pooled", "tail_keys", "tail_gates" }) |name| {
        const a = @field(source.*, name);
        if (a.ctx != null) {
            @field(state, name) = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&@field(state, name), a));
        }
    }
    return state;
}

fn mlaTree(layer: *const forward.Mla, ops: *Ops, x: Arr, cfg: *const @import("model.zig").ModelConfig, state: *const attention.State, parents: []const i32, mode: kda.ProjectionMode) !struct { output: Arr, tape: MlaTape } {
    const t: c_int = @intCast(parents.len);
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kd: c_int = @intCast(cfg.mla_qk_nope_head_dim);
    const width: c_int = @intCast(cfg.mla_kv_lora_rank);
    const ih: c_int = @intCast(cfg.indexer_n_heads);
    const iw: c_int = @intCast(cfg.indexer_head_dim);
    const qr = try ops.rms(try kda.linearRows(ops, layer.qa, x, mode), layer.qa_norm, cfg.rms_norm_eps);
    const q = try ops.reshape(try kda.linearRows(ops, layer.qb, qr, mode), &.{ t, heads, 1, kd });
    // Head-batched projections keep each node's serial [1,H,1,D] geometry.
    var absorbed: [16]Arr = undefined;
    for (0..parents.len) |row| {
        const one = try ops.slice(q, 0, @intCast(row), @intCast(row + 1));
        absorbed[row] = if (layer.quantized) try ops.qmm(one, layer.wk, layer.sk, layer.bk, false) else try ops.binary(.mm, one, layer.wk);
    }
    const qa = try ops.reshape(try ops.concat(absorbed[0..parents.len], 0), &.{ t, heads, width });
    const latent = try ops.reshape(try ops.rms(try kda.linearRows(ops, layer.kva, x, mode), layer.kv_norm, cfg.rms_norm_eps), &.{ t, width });
    const index_q = try ops.reshape(try kda.linearRows(ops, layer.iq, qr, mode), &.{ t, ih, iw });
    const keys = try ops.reshape(try ops.layerNorm(try kda.linearRows(ops, layer.ik, x, mode), layer.ik_norm, layer.ik_bias, 1e-6), &.{ t, iw });
    const index_weights = try ops.reshape(try ops.cast(try ops.binary(.mul, try kda.linearRows(ops, layer.iw, x, mode), try ops.scalar(1 / @sqrt(@as(f32, @floatFromInt(ih * iw))), .float32)), mlx.mlx_array_dtype(index_q)), &.{ t, ih });
    const compress = base.Linear{ .w = layer.compress, .scales = .{ .ctx = null }, .biases = .{ .ctx = null }, .input = @intCast(cfg.hidden_size), .output = @intCast(cfg.indexer_head_dim) };
    const gates = try ops.reshape(try kda.linearRows(ops, compress, x, mode), &.{ t, iw });
    var tape = MlaTape{ .latent = .{ .ctx = null }, .keys = .{ .ctx = null }, .gates = .{ .ctx = null }, .ape = .{ .ctx = null } };
    errdefer tape.deinit();
    tape.latent = try ops.result(latent);
    tape.keys = try ops.result(keys);
    tape.gates = try ops.result(gates);
    tape.ape = try ops.result(layer.ape);
    var result_rows: [16]Arr = undefined;
    for (0..parents.len) |row| {
        var branch = try forkAttention(state);
        defer branch.deinit();
        var path: [16]u32 = undefined;
        const kept = ancestry(parents, row, &path);
        try tape.append(&branch, kept, ops.s);
        const from: c_int = @intCast(row);
        const y = try ops.own(try attention.attend(&branch, try ops.slice(qa, 0, from, from + 1), try ops.slice(index_q, 0, from, from + 1), try ops.slice(index_weights, 0, from, from + 1), state.processed + kept.len - 1, 1 / @sqrt(@as(f32, @floatFromInt(kd))), ops.s));
        // Bound copies of the committed latent prefix to one branch at a time.
        try mlx.check(mlx.mlx_array_eval(y));
        const y4 = try ops.reshape(y, &.{ 1, heads, 1, width });
        const values = if (layer.quantized) try ops.qmm(y4, layer.wv, layer.sv, layer.bv, true) else try ops.binary(.mm, y4, try ops.transpose(layer.wv, &.{ 0, 2, 1 }));
        result_rows[row] = try ops.reshape(values, &.{ 1, 1, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
    }
    const output = try kda.linearRows(ops, layer.out, try ops.concat(result_rows[0..parents.len], 1), mode);
    return .{ .output = output, .tape = tape };
}

const LayerTape = union(enum) {
    kda: kda.Tape,
    mla: MlaTape,
    fn deinit(self: *LayerTape) void {
        switch (self.*) {
            .kda => |*t| t.deinit(),
            .mla => |*t| t.deinit(),
        }
    }
};

pub const Verified = struct {
    allocator: std.mem.Allocator,
    layers: []?LayerTape,
    captures: adapter.Captures,
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    targets: [16]u32 = undefined,
    count: usize,
    offset: usize,
    pub fn deinit(self: *Verified) void {
        for (self.layers) |*maybe| if (maybe.*) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.captures.deinit();
    }
    /// Replays only the accepted KDA prework and rebuilds only its IndexPool path.
    pub fn prepareCommit(self: *const Verified, source: *const forward.Request, budget: usize, eos: []const u32, s: mlx.mlx_stream) !adapter.Verification {
        if (source.offset != self.offset or source.layers.len != self.layers.len) return error.InvalidGlmDraftOffset;
        const accepted = try tree.accept(self.tokens[0..self.count], self.parents[0..self.count], self.targets[0..self.count], budget, eos);
        const path = accepted.rows[0..accepted.count];
        const last = path[path.len - 1];
        var result = adapter.Verification{ .count = self.count, .offset = self.offset, .tokens = self.tokens, .parents = self.parents, .targets = self.targets };
        errdefer result.deinit();
        result.states[last] = try adapter.cloneRequest(source);
        const next = &result.states[last].?;
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const arrays = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(arrays);
        for (self.layers, next.layers) |maybe, *state| switch (maybe orelse return error.IncompleteGlmDraftVerify) {
            .kda => |tape| {
                const updated = try tape.replay(path, s);
                _ = mlx.mlx_array_free(state.recurrent.conv_state);
                _ = mlx.mlx_array_free(state.recurrent.ssm_state);
                state.recurrent = updated;
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, updated.conv_state));
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, updated.ssm_state));
            },
            .mla => |tape| {
                try tape.append(&state.attention, path, s);
                for (state.attention.arrays()) |a| if (a.ctx != null) {
                    try mlx.check(mlx.mlx_vector_array_append_value(arrays, a));
                };
            },
        };
        for (path) |row| {
            result.captures[row] = try adapter.Captures.init(self.allocator, self.captures.hook.ids);
            for (self.captures.hook.out, result.captures[row].?.hook.out) |value, *out| {
                try mlx.check(mlx.mlx_array_set(out, try ops.slice(value, 1, @intCast(row), @intCast(row + 1))));
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, out.*));
            }
        }
        try mlx.check(mlx.mlx_eval(arrays));
        next.offset += path.len;
        return result;
    }
};

pub fn verify(target: *const forward.Model, request: *const forward.Request, tokens: []const u32, parents: []const i32, taps: []const u32, mode: kda.ProjectionMode) !Verified {
    try tree.validate(tokens, parents);
    if (request.failed) return error.GlmRequestNeedsReset;
    if (request.capture != null or request.layers.len != target.layers.len) return error.InvalidGlmDraftOffset;
    for (tokens) |token| if (token >= target.cfg.vocab_size) return error.InvalidGlmDraftToken;
    for (0..parents.len) |row| {
        var path: [16]u32 = undefined;
        const length = ancestry(parents, row, &path).len;
        if (request.offset >= target.cfg.max_position_embeddings or length > target.cfg.max_position_embeddings - request.offset) return error.GlmContextExceeded;
    }
    for (taps, 0..) |id, i| if (id >= target.layers.len or (i > 0 and id <= taps[i - 1])) return error.InvalidGlmCapture;
    const allocator = request.allocator;
    const layers = try allocator.alloc(?LayerTape, target.layers.len);
    @memset(layers, null);
    const captures = adapter.Captures.init(allocator, taps) catch |err| {
        allocator.free(layers);
        return err;
    };
    var result = Verified{ .allocator = allocator, .layers = layers, .captures = captures, .count = tokens.len, .offset = request.offset };
    errdefer result.deinit();
    @memcpy(result.tokens[0..tokens.len], tokens);
    @memcpy(result.parents[0..tokens.len], parents);
    const rows: c_int = @intCast(tokens.len);
    var h: Arr = undefined;
    {
        var ops = Ops{ .s = target.s };
        defer ops.deinit();
        const ids = try ops.own(mlx.mlx_array_new_data(tokens.ptr, &[_]c_int{ 1, rows }, 2, .uint32));
        const embedding = try ops.own(try target.rawEmbedding(ids));
        h = try ops.result(try ops.contiguous(try ops.broadcast(try ops.reshape(embedding, &.{ 1, rows, 1, @intCast(target.cfg.hidden_size) }), &.{ 1, rows, 4, @intCast(target.cfg.hidden_size) })));
    }
    defer _ = mlx.mlx_array_free(h);
    for (target.layers, request.layers, 0..) |*layer, *state, index| {
        var ops = Ops{ .s = target.s };
        defer ops.deinit();
        const pre = try layer.hc_attn.collapse(&ops, h, &target.cfg);
        defer pre.deinit();
        const x = try ops.rms(pre.mixed, layer.norm_attn, target.cfg.rms_norm_eps);
        const attended = switch (layer.attn) {
            .kda => |weights| blk: {
                const computed = try kda.applyLayer(weights, &ops, x, &target.cfg, &state.recurrent, parents, mode);
                result.layers[index] = .{ .kda = computed.tape };
                break :blk computed.output;
            },
            .mla => |*weights| blk: {
                const computed = try mlaTree(weights, &ops, x, &target.cfg, &state.attention, parents, mode);
                result.layers[index] = .{ .mla = computed.tape };
                break :blk computed.output;
            },
        };
        const joined = try ops.own(try primitive.hcExpand(h, attended, pre.post, pre.comb, target.s));
        const ff = try layer.hc_ffn.collapse(&ops, joined, &target.cfg);
        defer ff.deinit();
        const fx = try ops.rms(ff.mixed, layer.norm_ffn, target.cfg.rms_norm_eps);
        const ffout = if (mode == .batched) try target.feedForwardLayer(index, &ops, fx) else blk: {
            var outputs: [16]Arr = undefined;
            var made: usize = 0;
            defer for (outputs[0..made]) |value| {
                _ = mlx.mlx_array_free(value);
            };
            for (0..tokens.len) |row| {
                var one = Ops{ .s = target.s };
                defer one.deinit();
                outputs[row] = try one.result(try target.feedForwardLayer(index, &one, try one.slice(fx, 1, @intCast(row), @intCast(row + 1))));
                made += 1;
            }
            break :blk try ops.concat(outputs[0..made], 1);
        };
        const next = try ops.own(try primitive.hcExpand(joined, ffout, ff.post, ff.comb, target.s));
        const evals = mlx.mlx_vector_array_new_value(next);
        defer _ = mlx.mlx_vector_array_free(evals);
        switch (result.layers[index].?) {
            .kda => |tape| for ([_]Arr{ tape.inputs.q, tape.inputs.k, tape.inputs.v, tape.inputs.decay, tape.inputs.beta, tape.inputs.state, tape.conv_input }) |a| {
                try mlx.check(mlx.mlx_vector_array_append_value(evals, a));
            },
            .mla => |tape| for ([_]Arr{ tape.latent, tape.keys, tape.gates }) |a| {
                try mlx.check(mlx.mlx_vector_array_append_value(evals, a));
            },
        }
        for (taps, 0..) |id, tap| if (id == index) {
            try mlx.check(mlx.mlx_array_set(&result.captures.hook.out[tap], try ops.reduce(next, 2, true, false)));
            try mlx.check(mlx.mlx_vector_array_append_value(evals, result.captures.hook.out[tap]));
        };
        try mlx.check(mlx.mlx_eval(evals));
        try mlx.check(mlx.mlx_array_set(&h, next));
    }
    var ops = Ops{ .s = target.s };
    defer ops.deinit();
    const normalized = try ops.rms(try ops.reduce(h, 2, true, false), target.norm, target.cfg.rms_norm_eps);
    const logits = try kda.linearRows(&ops, target.head, normalized, mode);
    const decisions = try ops.slot();
    try mlx.check(mlx.mlx_argmax_axis(decisions, logits, -1, false, target.s));
    const u = try ops.cast(decisions.*, .uint32);
    try mlx.check(mlx.mlx_array_eval(u));
    @memcpy(result.targets[0..tokens.len], (mlx.mlx_array_data_uint32(u) orelse return error.MlxArrayDataNull)[0..tokens.len]);
    return result;
}
