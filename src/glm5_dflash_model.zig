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
// Layers per asynchronous evaluation group; zero settles every layer synchronously.
threadlocal var async_layers: usize = 0;
var async_dispatches: usize = 0;
var sync_dispatches: usize = 0;
pub const ScheduleBinding = struct {
    previous: usize,
    pub fn restore(self: ScheduleBinding) void {
        async_layers = self.previous;
    }
};
pub fn bindSchedule(layers: usize) !ScheduleBinding {
    if (layers != 0 and layers != 2 and layers != 4) return error.InvalidGlmVerifySchedule;
    const old = ScheduleBinding{ .previous = async_layers };
    async_layers = layers;
    return old;
}
pub fn asyncDispatchCount() usize {
    return async_dispatches;
}
pub fn syncDispatchCount() usize {
    return sync_dispatches;
}
fn appendTape(evals: mlx.mlx_vector_array, tape: LayerTape) !void {
    switch (tape) {
        .kda => |t| for ([_]Arr{ t.inputs.q, t.inputs.k, t.inputs.v, t.inputs.decay, t.inputs.beta, t.inputs.state, t.conv_input }) |a| {
            try mlx.check(mlx.mlx_vector_array_append_value(evals, a));
        },
        .mla => |t| for ([_]Arr{ t.latent, t.keys, t.gates }) |a| {
            try mlx.check(mlx.mlx_vector_array_append_value(evals, a));
        },
    }
}
var mla_branch_flushes: usize = 0;
var mla_scratch_bound: usize = 0;
pub fn branchFlushCount() usize {
    return mla_branch_flushes;
}
pub fn scratchBoundBytes() usize {
    return mla_scratch_bound;
}
pub fn resetStats() void {
    async_dispatches = 0;
    sync_dispatches = 0;
    mla_branch_flushes = 0;
    mla_scratch_bound = 0;
}

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
    fn appendIndex(self: *const MlaTape, state: *attention.State, path: []const u32, s: mlx.mlx_stream) !Arr {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const indices = try ops.own(mlx.mlx_array_new_data(path.ptr, &.{@intCast(path.len)}, 1, .uint32));
        const tail = try ops.take(self.latent, indices, 0);
        _ = try state.appendIndexOnly(tail, try ops.take(self.keys, indices, 0), try ops.take(self.gates, indices, 0), self.ape, s);
        return ops.result(tail);
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
    const native = @import("glm5_attention_decode_batch.zig");
    const native_mode = native.enabled() and native.supportedConfig(cfg, mlx.mlx_array_dtype(x), ops.s);
    if (native_mode and parents.len > 3) return error.GlmDecodeNativeTreeUnsupported;
    const t: c_int = @intCast(parents.len);
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kd: c_int = @intCast(cfg.mla_qk_nope_head_dim);
    const width: c_int = @intCast(cfg.mla_kv_lora_rank);
    const ih: c_int = @intCast(cfg.indexer_n_heads);
    const iw: c_int = @intCast(cfg.indexer_head_dim);
    const latent_capacity: usize = if (state.latent.ctx != null) @intCast(mlx.getShape(state.latent)[0]) else 0;
    const pool_capacity: usize = if (state.pooled.ctx != null) @intCast(mlx.getShape(state.pooled)[0]) else 0;
    var scratch = try @import("glm5_dflash_memory.zig").plan(state.processed, latent_capacity, pool_capacity, @intCast(width), @intCast(iw), @intCast(heads), parents.len, mlx.mlx_array_itemsize(x));
    if (native_mode) {
        const limit = @import("glm5_dflash_memory.zig").limit_bytes;
        if (scratch.common_bytes >= limit - native.scratch_limit) return error.GlmTreeScratchLimit;
        scratch.branches = @min(parents.len, (limit - scratch.common_bytes - native.scratch_limit) / scratch.per_branch_bytes);
        if (scratch.branches == 0) return error.GlmTreeScratchLimit;
        scratch.live_bytes = scratch.common_bytes + scratch.branches * scratch.per_branch_bytes + native.scratch_limit;
    }
    mla_scratch_bound = @max(mla_scratch_bound, scratch.live_bytes);
    const qr = try ops.rms(try kda.linearRows(ops, layer.qa, x, mode), layer.qa_norm, cfg.rms_norm_eps);
    const q = try ops.reshape(try kda.linearRows(ops, layer.qb, qr, mode), &.{ t, heads, 1, kd });
    const verify_batch = @import("glm5_mla_verify_batch.zig");
    const broadcast_q = if (layer.quantized) try verify_batch.run(ops, .{ .x = q, .w = layer.wk, .scales = layer.sk, .biases = layer.bk }, .query) else null;
    const qa = try ops.reshape(if (broadcast_q) |out| out else blk: {
        var absorbed: [16]Arr = undefined;
        for (0..parents.len) |row| {
            const one = try ops.slice(q, 0, @intCast(row), @intCast(row + 1));
            absorbed[row] = if (layer.quantized) try ops.qmm(one, layer.wk, layer.sk, layer.bk, false) else try ops.binary(.mm, one, layer.wk);
        }
        break :blk try ops.concat(absorbed[0..parents.len], 0);
    }, &.{ t, heads, width });
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
    // Dense prefixes do not consume index_q/index_weights. Keep them lazy here;
    // sparse branches naturally bill their necessary indexer work in mla_branches.
    var result_rows: [16]Arr = undefined;
    var attention_rows: [16]Arr = undefined;
    var pending: [16]Arr = undefined;
    var pending_count: usize = 0;
    const batched_native = native_mode and parents.len == 3 and scratch.branches == 3;
    var native_ids: [3]Arr = undefined;
    var native_branches: [3]native.Branch = undefined;
    for (0..parents.len) |row| {
        var branch = try forkAttention(state);
        defer branch.deinit();
        var path: [16]u32 = undefined;
        const kept = ancestry(parents, row, &path);
        const overlay = parents.len <= 3;
        const tail = if (overlay) try ops.own(try tape.appendIndex(&branch, kept, ops.s)) else blk: {
            try tape.append(&branch, kept, ops.s);
            break :blk Arr{ .ctx = null };
        };
        const from: c_int = @intCast(row);
        const query = try ops.slice(qa, 0, from, from + 1);
        const index_query = try ops.slice(index_q, 0, from, from + 1);
        const weights = try ops.slice(index_weights, 0, from, from + 1);
        const offset = state.processed + kept.len - 1;
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(kd)));
        if (batched_native) {
            native_ids[row] = try ops.own(try attention.decodeSelected(&branch, index_query, weights, offset, ops.s));
            native_branches[row] = .{ .offset = offset, .length = branch.processed, .path = .{ 0, 0, 0 } };
            @memcpy(native_branches[row].path[0..kept.len], kept);
            continue;
        }
        const y = try ops.own(if (overlay)
            try attention.attendOverlay(&branch, query, index_query, weights, offset, scale, .{ .prefix = state.latent, .prefix_rows = state.processed, .tail = tail }, ops.s)
        else
            try attention.attend(&branch, query, index_query, weights, offset, scale, ops.s));
        pending[pending_count] = y;
        pending_count += 1;
        // The final group settles at the enclosing layer boundary.
        if (pending_count == scratch.branches and row + 1 < parents.len) {
            const group = mlx.mlx_vector_array_new_data(&pending, pending_count);
            defer _ = mlx.mlx_vector_array_free(group);
            try mlx.check(mlx.mlx_eval(group));
            mla_branch_flushes += 1;
            pending_count = 0;
        }
        const y4 = try ops.reshape(y, &.{ 1, heads, 1, width });
        attention_rows[row] = y4;
        if (broadcast_q == null) {
            const values = if (layer.quantized) try ops.qmm(y4, layer.wv, layer.sv, layer.bv, true) else try ops.binary(.mm, y4, try ops.transpose(layer.wv, &.{ 0, 2, 1 }));
            result_rows[row] = try ops.reshape(values, &.{ 1, 1, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
        }
    }
    if (batched_native) {
        const prefix = if (state.processed == 0) latent else state.latent;
        const selected = try ops.concat(&native_ids, 0);
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(kd)));
        const batched = try native.run(ops, qa, prefix, state.processed, latent, &native_branches, selected, scale);
        for (0..parents.len) |row| {
            const y = if (batched) |all| try ops.slice(all, 0, @intCast(row), @intCast(row + 1)) else (try native.run(ops, try ops.slice(qa, 0, @intCast(row), @intCast(row + 1)), prefix, state.processed, latent, native_branches[row .. row + 1], native_ids[row], scale)) orelse return error.GlmDecodeNativeUnsupported;
            const y4 = try ops.reshape(y, &.{ 1, heads, 1, width });
            attention_rows[row] = y4;
            if (broadcast_q == null) {
                const values = if (layer.quantized) try ops.qmm(y4, layer.wv, layer.sv, layer.bv, true) else try ops.binary(.mm, y4, try ops.transpose(layer.wv, &.{ 0, 2, 1 }));
                result_rows[row] = try ops.reshape(values, &.{ 1, 1, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
            }
        }
    }
    if (broadcast_q != null) {
        const combined = try ops.concat(attention_rows[0..parents.len], 0);
        if (try verify_batch.run(ops, .{ .x = combined, .w = layer.wv, .scales = layer.sv, .biases = layer.bv }, .value)) |values| {
            for (0..parents.len) |row| result_rows[row] = try ops.reshape(try ops.slice(values, 0, @intCast(row), @intCast(row + 1)), &.{ 1, 1, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
        } else for (0..parents.len) |row| {
            const values = try ops.qmm(attention_rows[row], layer.wv, layer.sv, layer.bv, true);
            result_rows[row] = try ops.reshape(values, &.{ 1, 1, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
        }
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
    logits: Arr = .{ .ctx = null },
    pub fn deinit(self: *Verified) void {
        if (self.logits.ctx != null) _ = mlx.mlx_array_free(self.logits);
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
    if (target.expert_stream != null) return error.GlmStreamingSpecUnsupported;
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
    const cadence = async_layers;
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
        const ffout = if (mode == .affine_rows_ffn) try @import("glm5_dflash_ffn.zig").apply(target, index, &ops, fx) else blk: {
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
        for (taps, 0..) |id, tap| if (id == index) {
            try mlx.check(mlx.mlx_array_set(&result.captures.hook.out[tap], try ops.reduce(next, 2, true, false)));
        };
        if (cadence == 0 or (index + 1) % cadence == 0) {
            const evals = mlx.mlx_vector_array_new_value(next);
            defer _ = mlx.mlx_vector_array_free(evals);
            const first = if (cadence == 0) index else index + 1 - cadence;
            for (result.layers[first .. index + 1]) |tape| try appendTape(evals, tape.?);
            for (taps, 0..) |id, tap| if (id >= first and id <= index) {
                try mlx.check(mlx.mlx_vector_array_append_value(evals, result.captures.hook.out[tap]));
            };
            if (cadence == 0) {
                try mlx.check(mlx.mlx_eval(evals));
                sync_dispatches += 1;
            } else {
                try mlx.check(mlx.mlx_async_eval(evals));
                async_dispatches += 1;
            }
        }
        try mlx.check(mlx.mlx_array_set(&h, next));
    }
    var ops = Ops{ .s = target.s };
    defer ops.deinit();
    const normalized = try ops.rms(try ops.reduce(h, 2, true, false), target.norm, target.cfg.rms_norm_eps);
    const logits = try target.samplingLogits(&ops, try kda.linearRows(&ops, target.head, normalized, mode));
    const decisions = try ops.slot();
    try mlx.check(mlx.mlx_argmax_axis(decisions, logits, -1, false, target.s));
    const u = try ops.cast(decisions.*, .uint32);
    // Head dependencies do not necessarily consume every replay/capture array.
    // Settle them explicitly before returning ownership to commit/replay.
    const final = mlx.mlx_vector_array_new_value(u);
    defer _ = mlx.mlx_vector_array_free(final);
    if (cadence != 0) {
        for (result.layers) |tape| try appendTape(final, tape.?);
        for (result.captures.hook.out) |value| try mlx.check(mlx.mlx_vector_array_append_value(final, value));
    }
    try mlx.check(mlx.mlx_eval(final));
    sync_dispatches += 1;
    @memcpy(result.targets[0..tokens.len], (mlx.mlx_array_data_uint32(u) orelse return error.MlxArrayDataNull)[0..tokens.len]);
    result.logits = try ops.result(logits);
    return result;
}

test {
    _ = @import("glm5_dflash_memory.zig");
}

fn expectArrayBits(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    if (a.ctx == null or b.ctx == null) return std.testing.expect(a.ctx == null and b.ctx == null);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    try mlx.check(mlx.mlx_array_eval(x));
    try mlx.check(mlx.mlx_array_eval(y));
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_dtype(x) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(x).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(y).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(x).?[0..n], mlx.mlx_array_data_bfloat16(y).?[0..n]);
}

test "GLM latent overlay three-node verifier commits independent serial ancestry" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iterator = weights.map.iterator();
    var seed: usize = 23;
    while (iterator.next()) |entry| {
        const value = entry.value_ptr;
        const sh = mlx.getShape(value.*);
        if (sh.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(sh, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    var request = try forward.Request.init(a, target.layers.len);
    defer request.deinit();
    const ids = mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &.{ 1, 3 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try target.forwardLast(&request, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    var tokens = [_]u32{ 1, 0, 0 };
    const taps = [_]u32{ 0, 3 };
    for ([_][3]i32{ .{ -1, 0, 1 }, .{ -1, 0, 0 } }) |parents| {
        tokens[2] = if (parents[2] == 0) 2 else 0;
        var oracle = try adapter.verifyTreeOracle(&target, &request, &tokens, &parents, &taps);
        defer oracle.deinit();
        var computed = try verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
        defer computed.deinit();
        try std.testing.expectEqualSlices(u32, oracle.targets[0..oracle.count], computed.targets[0..computed.count]);
        var captures = Ops{ .s = s };
        defer captures.deinit();
        for (0..3) |row| for (computed.captures.hook.out, oracle.captures[row].?.hook.out) |left, right| {
            try expectArrayBits(try captures.slice(left, 1, @intCast(row), @intCast(row + 1)), right, s);
        };
        for ([_]usize{ 1, 2, 3 }) |budget| {
            var committed = try computed.prepareCommit(&request, budget, &.{}, s);
            defer committed.deinit();
            const accepted = try tree.accept(&tokens, &parents, oracle.targets[0..oracle.count], budget, &.{});
            const last = accepted.rows[accepted.count - 1];
            const left = committed.states[last].?;
            const right = oracle.states[last].?;
            try std.testing.expectEqual(right.offset, left.offset);
            for (left.layers, right.layers) |x, y| {
                for (x.attention.arrays(), y.attention.arrays()) |u, v| try expectArrayBits(u, v, s);
                try expectArrayBits(x.recurrent.conv_state, y.recurrent.conv_state, s);
                try expectArrayBits(x.recurrent.ssm_state, y.recurrent.ssm_state, s);
            }
        }
    }
}
