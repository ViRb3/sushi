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
const profiling = @import("glm5_dflash_profile.zig");
const Ops = base.Ops;
var mla_branch_flushes: usize = 0;
var mla_scratch_bound: usize = 0;
pub fn branchFlushCount() usize {
    return mla_branch_flushes;
}
pub fn scratchBoundBytes() usize {
    return mla_scratch_bound;
}
pub fn resetStats() void {
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
    const latent_capacity: usize = if (state.latent.ctx != null) @intCast(mlx.getShape(state.latent)[0]) else 0;
    const pool_capacity: usize = if (state.pooled.ctx != null) @intCast(mlx.getShape(state.pooled)[0]) else 0;
    const scratch = try @import("glm5_dflash_memory.zig").plan(state.processed, latent_capacity, pool_capacity, @intCast(width), @intCast(iw), @intCast(heads), parents.len, mlx.mlx_array_itemsize(x));
    mla_scratch_bound = @max(mla_scratch_bound, scratch.live_bytes);
    var profile = profiling.Timer.start(parents.len);
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
    // Dense prefixes do not consume index_q/index_weights. Keep them lazy here;
    // sparse branches naturally bill their necessary indexer work in mla_branches.
    try profile.finish("mla_common", &.{ qa, latent, keys, gates });
    var result_rows: [16]Arr = undefined;
    var pending: [16]Arr = undefined;
    var pending_count: usize = 0;
    for (0..parents.len) |row| {
        var branch = try forkAttention(state);
        defer branch.deinit();
        var path: [16]u32 = undefined;
        const kept = ancestry(parents, row, &path);
        try tape.append(&branch, kept, ops.s);
        const from: c_int = @intCast(row);
        const y = try ops.own(try attention.attend(&branch, try ops.slice(qa, 0, from, from + 1), try ops.slice(index_q, 0, from, from + 1), try ops.slice(index_weights, 0, from, from + 1), state.processed + kept.len - 1, 1 / @sqrt(@as(f32, @floatFromInt(kd))), ops.s));
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
        const values = if (layer.quantized) try ops.qmm(y4, layer.wv, layer.sv, layer.bv, true) else try ops.binary(.mm, y4, try ops.transpose(layer.wv, &.{ 0, 2, 1 }));
        result_rows[row] = try ops.reshape(values, &.{ 1, 1, @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim) });
    }
    try profile.finish("mla_branches", result_rows[0..parents.len]);
    const output = try kda.linearRows(ops, layer.out, try ops.concat(result_rows[0..parents.len], 1), mode);
    try profile.finish("mla_out", &.{output});
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
    var embedding_profile = profiling.Timer.start(tokens.len);
    var h: Arr = undefined;
    {
        var ops = Ops{ .s = target.s };
        defer ops.deinit();
        const ids = try ops.own(mlx.mlx_array_new_data(tokens.ptr, &[_]c_int{ 1, rows }, 2, .uint32));
        const embedding = try ops.own(try target.rawEmbedding(ids));
        h = try ops.result(try ops.contiguous(try ops.broadcast(try ops.reshape(embedding, &.{ 1, rows, 1, @intCast(target.cfg.hidden_size) }), &.{ 1, rows, 4, @intCast(target.cfg.hidden_size) })));
    }
    defer _ = mlx.mlx_array_free(h);
    try embedding_profile.finish("embedding", &.{h});
    for (target.layers, request.layers, 0..) |*layer, *state, index| {
        const scope = try profiling.enterLayer(index);
        defer scope.restore();
        var profile = profiling.Timer.start(tokens.len);
        var ops = Ops{ .s = target.s };
        defer ops.deinit();
        const pre = try layer.hc_attn.collapse(&ops, h, &target.cfg);
        defer pre.deinit();
        try profile.finish("hc_attn", &.{ pre.mixed, pre.post, pre.comb });
        const x = try ops.rms(pre.mixed, layer.norm_attn, target.cfg.rms_norm_eps);
        try profile.finish("attn_norm", &.{x});
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
        profile = profiling.Timer.start(tokens.len);
        const joined = try ops.own(try primitive.hcExpand(h, attended, pre.post, pre.comb, target.s));
        try profile.finish("expand_attn", &.{joined});
        const ff = try layer.hc_ffn.collapse(&ops, joined, &target.cfg);
        defer ff.deinit();
        try profile.finish("hc_ffn", &.{ ff.mixed, ff.post, ff.comb });
        const fx = try ops.rms(ff.mixed, layer.norm_ffn, target.cfg.rms_norm_eps);
        try profile.finish("ffn_norm", &.{fx});
        const ffout = if (mode == .affine_rows_ffn) try @import("glm5_dflash_ffn.zig").apply(target, index, &ops, fx) else if (mode == .batched) try target.feedForwardLayer(index, &ops, fx) else blk: {
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
        try profile.finish("ffn_overall", &.{ffout});
        const next = try ops.own(try primitive.hcExpand(joined, ffout, ff.post, ff.comb, target.s));
        try profile.finish("expand_ffn", &.{next});
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
        profile.record("layer_settle");
        try mlx.check(mlx.mlx_array_set(&h, next));
    }
    var ops = Ops{ .s = target.s };
    defer ops.deinit();
    var head_profile = profiling.Timer.start(tokens.len);
    const normalized = try ops.rms(try ops.reduce(h, 2, true, false), target.norm, target.cfg.rms_norm_eps);
    const logits = try kda.linearRows(&ops, target.head, normalized, mode);
    const decisions = try ops.slot();
    try mlx.check(mlx.mlx_argmax_axis(decisions, logits, -1, false, target.s));
    const u = try ops.cast(decisions.*, .uint32);
    try mlx.check(mlx.mlx_array_eval(u));
    head_profile.record("head");
    @memcpy(result.targets[0..tokens.len], (mlx.mlx_array_data_uint32(u) orelse return error.MlxArrayDataNull)[0..tokens.len]);
    return result;
}

test {
    _ = @import("glm5_dflash_memory.zig");
}

fn profileArrayEqual(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
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

test "GLM DFlash component profiling preserves nonzero captures and complete committed state" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const profile = @import("glm5_dflash_profile.zig");
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iterator = weights.map.iterator();
    var seed: usize = 13;
    while (iterator.next()) |entry| {
        const value = entry.value_ptr;
        const sh = mlx.getShape(value.*);
        if (sh.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const next = try @import("dflash.zig").TinyFix.bf16ArrShaped(sh, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = next;
        seed += 1;
    }
    var codes: [4 * 32]u32 = undefined;
    for (&codes, 0..) |*v, i| v.* = 0x01030205 + @as(u32, @intCast(i % 4)) * 0x01010101;
    const embedding = weights.map.getPtr("model.language_model.embed_tokens.weight").?;
    _ = mlx.mlx_array_free(embedding.*);
    embedding.* = mlx.mlx_array_new_data(&codes, &.{ 4, 32 }, 2, .uint32);
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    var request = try forward.Request.init(a, target.layers.len);
    defer request.deinit();
    const ids = mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &.{ 1, 3 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try target.forwardLast(&request, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    const tokens = [_]u32{ 1, 0, 2, 0, 3 };
    const parents = [_]i32{ -1, 0, 0, 1, 2 };
    const taps = [_]u32{ 0, 3 };
    const disabled = profile.bind(null);
    defer disabled.restore();
    var ordinary = try verify(&target, &request, &tokens, &parents, &taps, .affine_rows_ffn);
    defer ordinary.deinit();
    var collected: profile.Profile = .{};
    var measured = blk: {
        const binding = profile.bind(&collected);
        defer binding.restore();
        break :blk try verify(&target, &request, &tokens, &parents, &taps, .affine_rows_ffn);
    };
    defer measured.deinit();
    var disabled_again = try verify(&target, &request, &tokens, &parents, &taps, .affine_rows_ffn);
    defer disabled_again.deinit();
    try std.testing.expectEqualSlices(u32, ordinary.targets[0..ordinary.count], disabled_again.targets[0..disabled_again.count]);
    for (ordinary.captures.hook.out, disabled_again.captures.hook.out) |left_capture, right_capture| try profileArrayEqual(left_capture, right_capture, s);
    try std.testing.expectEqualSlices(u32, ordinary.targets[0..ordinary.count], measured.targets[0..measured.count]);
    for (ordinary.captures.hook.out, measured.captures.hook.out) |left, right| try profileArrayEqual(left, right, s);
    const capture = measured.captures.hook.out[0];
    var nonzero = false;
    const values = try @import("dflash.zig").TinyFix.readF32(capture, a, s);
    defer a.free(values);
    for (values) |v| if (v != 0 and std.math.isFinite(v)) {
        nonzero = true;
        break;
    };
    try std.testing.expect(nonzero);
    var committed = try ordinary.prepareCommit(&request, 3, &.{}, s);
    defer committed.deinit();
    var profiled_commit = try measured.prepareCommit(&request, 3, &.{}, s);
    defer profiled_commit.deinit();
    const accepted = try tree.accept(&tokens, &parents, ordinary.targets[0..ordinary.count], 3, &.{});
    const last = accepted.rows[accepted.count - 1];
    const left = committed.states[last].?;
    const right = profiled_commit.states[last].?;
    try std.testing.expectEqual(left.offset, right.offset);
    for (left.layers, right.layers) |x, y| {
        try std.testing.expectEqual(x.recurrent.initialized, y.recurrent.initialized);
        try profileArrayEqual(x.recurrent.conv_state, y.recurrent.conv_state, s);
        try profileArrayEqual(x.recurrent.ssm_state, y.recurrent.ssm_state, s);
        try std.testing.expectEqual(x.attention.processed, y.attention.processed);
        for (x.attention.arrays(), y.attention.arrays()) |xx, yy| try profileArrayEqual(xx, yy, s);
    }
    try std.testing.expectEqual(@as(u64, 1), collected.global.embedding.calls);
    try std.testing.expectEqual(@as(u64, 1), collected.global.head.calls);
    var kda_calls: u64 = 0;
    var mla_calls: u64 = 0;
    for (collected.layers[0..target.layers.len]) |layer| {
        try std.testing.expectEqual(@as(u64, 1), layer.hc_attn.calls);
        try std.testing.expectEqual(@as(u64, 1), layer.ffn_overall.calls);
        try std.testing.expect(layer.ffn_overall.ns > 0);
        inline for (.{ "kda_qkv", "kda_lowrank_beta", "kda_recurrence", "kda_gate_post", "kda_out" }) |field|
            try std.testing.expectEqual(layer.kda_prework.calls, @field(layer, field).calls);
        try std.testing.expectEqual(layer.mla_common.calls, layer.mla_branches.calls);
        try std.testing.expectEqual(layer.mla_common.calls, layer.mla_out.calls);
        kda_calls += layer.kda_prework.calls;
        mla_calls += layer.mla_common.calls;
    }
    try std.testing.expect(kda_calls > 0 and mla_calls > 0);
}
