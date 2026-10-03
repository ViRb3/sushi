//! GLM DFlash2 target adapter and exact branch verification oracle.
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const draft = @import("dflash.zig");
const tree = @import("glm5_dflash_tree.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

test "GLM DFlash request fork keeps rejected pool and recurrence state isolated" {
    var source = try forward.Request.init(std.testing.allocator, 1);
    defer source.deinit();
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const latent = try ops.ones(&.{ 3, 4 }, .float32);
    const keys = try ops.ones(&.{ 3, 2 }, .float32);
    const gates = try ops.zeros(&.{ 3, 2 }, .float32);
    const ape = try ops.zeros(&.{ 4, 2 }, .float32);
    _ = try source.layers[0].attention.append(latent, keys, gates, ape, ops.s);
    try source.layers[0].attention.evaluate();
    source.offset = 3;
    try mlx.check(mlx.mlx_array_set(&source.layers[0].recurrent.ssm_state, try ops.ones(&.{ 1, 1, 2, 2 }, .float32)));
    try mlx.check(mlx.mlx_array_set(&source.layers[0].recurrent.conv_state, try ops.ones(&.{ 1, 3, 6 }, .float32)));
    source.layers[0].recurrent.initialized = true;
    source.dense_prefill = true;
    source.prefill_async = true;
    source.prefill_sync_layers = 4;
    var branch = try cloneRequest(&source);
    defer branch.deinit();
    try std.testing.expectEqual(@as(u8, 4), branch.prefill_sync_layers);
    try std.testing.expect(branch.dense_prefill and branch.prefill_async);
    _ = try branch.layers[0].attention.append(try ops.ones(&.{ 1, 4 }, .float32), try ops.zeros(&.{ 1, 2 }, .float32), try ops.zeros(&.{ 1, 2 }, .float32), ape, ops.s);
    try branch.layers[0].attention.evaluate();
    try mlx.check(mlx.mlx_array_set(&branch.layers[0].recurrent.ssm_state, try ops.zeros(&.{ 1, 1, 2, 2 }, .float32)));
    try std.testing.expectEqual(@as(usize, 3), source.layers[0].attention.processed);
    try std.testing.expectEqual(@as(usize, 4), branch.layers[0].attention.processed);
    try std.testing.expect(source.layers[0].attention.pooled.ctx == null);
    try std.testing.expectEqual(@as(c_int, 3), mlx.getShape(source.layers[0].attention.tail_keys)[0]);
    try mlx.check(mlx.mlx_array_eval(source.layers[0].recurrent.ssm_state));
    try std.testing.expectEqual(@as(f32, 1), mlx.mlx_array_data_float32(source.layers[0].recurrent.ssm_state).?[0]);
    source.failed = true;
    try std.testing.expectError(error.GlmRequestNeedsReset, cloneRequest(&source));
}

/// All arrays are immutable values; retaining handles keeps each parent's history alive.
pub fn cloneRequest(source: *const forward.Request) !forward.Request {
    if (source.failed) return error.GlmRequestNeedsReset;
    var copy = try forward.Request.init(source.allocator, source.layers.len);
    errdefer copy.deinit();
    copy.offset = source.offset;
    copy.profile = source.profile;
    copy.decode_async = source.decode_async;
    copy.dense_prefill = source.dense_prefill;
    copy.prefill_async = source.prefill_async;
    copy.prefill_sync_layers = source.prefill_sync_layers;
    for (copy.layers, source.layers) |*dst, src| {
        if (src.recurrent.initialized) {
            try mlx.check(mlx.mlx_array_set(&dst.recurrent.conv_state, src.recurrent.conv_state));
            try mlx.check(mlx.mlx_array_set(&dst.recurrent.ssm_state, src.recurrent.ssm_state));
            dst.recurrent.initialized = true;
        }
        dst.attention.processed = src.attention.processed;
        inline for (.{ "latent", "pooled", "tail_keys", "tail_gates" }) |field| {
            const value = @field(src.attention, field);
            if (value.ctx != null) {
                @field(dst.attention, field) = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&@field(dst.attention, field), value));
            }
        }
    }
    return copy;
}

pub const Captures = struct {
    allocator: std.mem.Allocator,
    hook: forward.Capture,
    pub fn init(allocator: std.mem.Allocator, ids: []const u32) !Captures {
        const out = try allocator.alloc(Arr, ids.len);
        for (out) |*a| a.* = mlx.mlx_array_new_float(0);
        return .{ .allocator = allocator, .hook = .{ .ids = ids, .out = out } };
    }
    pub fn deinit(self: *Captures) void {
        for (self.hook.out) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(self.hook.out);
    }
};

fn argmax(logits: Arr, s: mlx.mlx_stream) !u32 {
    const sh = mlx.getShape(logits);
    if (sh.len != 3 or sh[0] != 1 or sh[1] != 1 or sh[2] < 1) return error.InvalidGlmDraftLogits;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const idx = try ops.slot();
    try mlx.check(mlx.mlx_argmax_axis(idx, logits, -1, false, s));
    const u = try ops.cast(idx.*, .uint32);
    try mlx.check(mlx.mlx_array_eval(u));
    return (mlx.mlx_array_data_uint32(u) orelse return error.MlxArrayDataNull)[0];
}

/// Correctness oracle: one serial forward per node with an isolated ancestor state.
/// This deliberately does not claim batched tree verification or a decode speedup.
pub const Verification = struct {
    states: [16]?forward.Request = @splat(null),
    captures: [16]?Captures = @splat(null),
    targets: [16]u32 = undefined,
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    count: usize,
    offset: usize,
    pub fn deinit(self: *Verification) void {
        for (&self.states) |*item| if (item.*) |*state| state.deinit();
        for (&self.captures) |*item| if (item.*) |*capture| capture.deinit();
    }
};

pub fn verifyTreeOracle(target: *const forward.Model, request: *const forward.Request, tokens: []const u32, parents: []const i32, taps: []const u32) !Verification {
    try tree.validate(tokens, parents);
    for (tokens) |token| if (token >= target.cfg.vocab_size) return error.InvalidGlmDraftToken;
    if (request.capture != null) return error.GlmCaptureAlreadyActive;
    var result = Verification{ .count = tokens.len, .offset = request.offset };
    errdefer result.deinit();
    @memcpy(result.tokens[0..tokens.len], tokens);
    @memcpy(result.parents[0..parents.len], parents);
    for (tokens, parents, 0..) |token, parent, row| {
        const input = if (parent < 0) request else &result.states[@intCast(parent)].?;
        result.states[row] = try cloneRequest(input);
        result.captures[row] = try Captures.init(request.allocator, taps);
        const state = &result.states[row].?;
        state.capture = &result.captures[row].?.hook;
        defer state.capture = null;
        const ids = mlx.mlx_array_new_data(&token, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const logits = try target.forwardLast(state, ids, true);
        defer _ = mlx.mlx_array_free(logits);
        result.targets[row] = try argmax(logits, target.s);
        state.capture = null;
    }
    return result;
}

fn cloneContext(assistant: *const draft.DflashModel, source: *const draft.DflashCtx) !draft.DflashCtx {
    var snapshot = try source.cache.snapshot();
    defer snapshot.deinit();
    var copy = try draft.DflashCtx.init(assistant.allocator, assistant, source.base_pos);
    errdefer copy.deinit();
    try copy.cache.restore(&snapshot);
    return copy;
}

var commit_window_calls: usize = 0;
pub fn commitWindowCalls() usize {
    return commit_window_calls;
}

fn cropCommitContext(assistant: *const draft.DflashModel, next: *draft.DflashCtx) !bool {
    const cfg = &assistant.config;
    if (assistant.layers.len != 5 or cfg.block_size != 8 or cfg.sliding_window != 2048 or next.cache.config.scheme != .off or next.cache.step < 2048) return false;
    const diag = @import("transformer.zig");
    if (!diag.diagEnvOn("SUSHI_GLM_DFLASH_BLOCK_TAIL") or !diag.diagEnvOn("SUSHI_GLM_DFLASH_COMMIT_WINDOW")) return false;
    for (assistant.layers, next.cache.entries) |layer, entry| {
        if (layer.layer_type != .sliding_attention or !entry.initialized or entry.ringed or entry.base != 0 or entry.offset != next.cache.step or mlx.mlx_array_dtype(entry.keys) != .bfloat16 or mlx.mlx_array_dtype(entry.values) != .bfloat16) return false;
    }
    const keep = cfg.sliding_window - 1;
    const drop = next.cache.step - keep;
    var ops = Ops{ .s = assistant.s };
    defer ops.deinit();
    for (next.cache.entries) |*entry| {
        const k = try ops.slice(entry.keys, 2, @intCast(drop), @intCast(next.cache.step));
        const v = try ops.slice(entry.values, 2, @intCast(drop), @intCast(next.cache.step));
        try mlx.check(mlx.mlx_array_set(&entry.keys, k));
        try mlx.check(mlx.mlx_array_set(&entry.values, v));
        entry.offset = keep;
    }
    next.cache.step = keep;
    next.base_pos += drop;
    return true;
}

fn evaluateContext(context: *draft.DflashCtx) !void {
    const arrays = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(arrays);
    context.appendEvalArrays(arrays);
    try mlx.check(mlx.mlx_eval(arrays));
}

/// Both target and assistant publish only after the new assistant state has evaluated.
pub fn commitVerified(assistant: *const draft.DflashModel, context: *draft.DflashCtx, request: *forward.Request, verified: *Verification, budget: usize, eos: []const u32) !tree.Accepted {
    if (request.failed or request.offset != verified.offset or context.absLen() != verified.offset) return error.InvalidGlmDraftOffset;
    const accepted = try tree.accept(verified.tokens[0..verified.count], verified.parents[0..verified.count], verified.targets[0..verified.count], budget, eos);
    const last = accepted.rows[accepted.count - 1];
    if (verified.states[last] == null) return error.GlmDraftAlreadyCommitted;
    for (accepted.rows[0..accepted.count]) |row| {
        const capture = verified.captures[row] orelse return error.InvalidGlmDraftCaptures;
        if (!std.mem.eql(u32, capture.hook.ids, assistant.config.target_layer_ids)) return error.InvalidGlmDraftCaptures;
    }
    if (verified.states[last].?.offset != verified.offset + accepted.count) return error.InvalidGlmDraftOffset;
    var next_context = try cloneContext(assistant, context);
    errdefer next_context.deinit();
    const windowed = try cropCommitContext(assistant, &next_context);
    var ops = Ops{ .s = assistant.s };
    defer ops.deinit();
    var projected: [16]Arr = undefined;
    if (assistant.config.target_layer_ids.len > projected.len) return error.InvalidGlmDraftCaptures;
    for (0..assistant.config.target_layer_ids.len) |tap| {
        var rows: [16]Arr = undefined;
        for (accepted.rows[0..accepted.count], 0..) |row, i| rows[i] = verified.captures[row].?.hook.out[tap];
        projected[tap] = try ops.concat(rows[0..accepted.count], 1);
    }
    try draft.appendContext(assistant, &next_context, projected[0..assistant.config.target_layer_ids.len], verified.offset);
    try evaluateContext(&next_context);
    request.deinit();
    request.* = verified.states[last].?;
    verified.states[last] = null;
    context.deinit();
    context.* = next_context;
    if (windowed) commit_window_calls += 1;
    return accepted;
}

pub fn validatePair(assistant: *const draft.DflashModel, target: *const forward.Model) !void {
    if (target.expert_stream != null) return error.GlmStreamingSpecUnsupported;
    const cfg = &assistant.config;
    if (!target.cfg.isGlm5() or !cfg.isDflash2() or cfg.hidden_size != target.cfg.hidden_size or cfg.mask_token_id >= target.cfg.vocab_size or
        cfg.anchor_row_drafts or assistant.markov != null or assistant.selector == null or cfg.target_layer_ids.len == 0 or cfg.target_layer_ids.len > 16 or
        cfg.block_size < 2 or cfg.block_size > 16 or cfg.conv_kernel_size != 2) return error.GlmDraftTargetMismatch;
    try draft.validateTargetLayers(cfg.target_layer_ids, target.cfg.num_hidden_layers);
    const selector = &assistant.selector.?;
    const pred = mlx.getShape(selector.pred_codebook);
    if (pred.len != 2 or pred[0] != target.cfg.vocab_size or pred[1] != cfg.selector_rank or !std.mem.eql(c_int, pred, mlx.getShape(selector.succ_codebook))) return error.GlmDraftTargetMismatch;
}

pub const Proposal = struct {
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    count: usize,
};

pub fn proposeTree(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, pending: u32, max_nodes: usize) !Proposal {
    return proposeTreeWithChildren(assistant, context, target, pending, max_nodes, 4);
}

var readout_horizon_calls: usize = 0;
pub fn readoutHorizonCalls() usize {
    return readout_horizon_calls;
}
fn readoutHorizonEnabled(max_nodes: usize, block_size: usize, mini: bool) bool {
    if (max_nodes != 2 or block_size != 8 or mini) return false;
    return @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_READOUT_HORIZON");
}

pub fn proposeTreeWithChildren(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, pending: u32, max_nodes: usize, children: usize) !Proposal {
    try validatePair(assistant, target);
    if (max_nodes == 0 or max_nodes > 15 or children == 0 or children > 16 or pending >= target.cfg.vocab_size) return error.InvalidGlmDraftTree;
    // forwardBlock uses spare cache rows; fork so an evaluation failure cannot alter committed context.
    var work = try cloneContext(assistant, context);
    defer work.deinit();
    var ops = Ops{ .s = assistant.s };
    defer ops.deinit();
    var noise: [16]u32 = @splat(assistant.config.mask_token_id);
    noise[0] = pending;
    const ids = try ops.own(mlx.mlx_array_new_data(&noise, &[_]c_int{ 1, @intCast(assistant.config.block_size) }, 2, .uint32));
    const embeds = try ops.own(try target.rawEmbedding(ids));
    const hidden = try ops.own(try draft.forwardBlock(assistant, &work, embeds, context.absLen()));
    const bounded = readoutHorizonEnabled(max_nodes, assistant.config.block_size, assistant.draft_head != null);
    const readout_input = if (bounded) try ops.slice(hidden, 1, 1, 3) else hidden;
    const projected = if (assistant.draft_head) |*coarse|
        try ops.own(try @import("glm5_dflash_mini.zig").project(target.head, coarse, hidden, assistant.s))
    else
        try ops.own(try target.projectHead(readout_input));
    const transformed = try ops.own(try draft.applyLogitTransforms(projected, assistant.config.output_multiplier, assistant.config.logit_softcap, assistant.s));
    const logits = if (bounded or assistant.draft_head != null) transformed else try ops.slice(transformed, 1, 1, @intCast(assistant.config.block_size));
    const selector_input = if (bounded) try ops.slice(hidden, 1, 0, 3) else hidden;
    var lattice = try tree.lattice(assistant.allocator, &assistant.selector.?, assistant.config.selector_top_k, selector_input, logits, pending, assistant.s);
    if (bounded) readout_horizon_calls += 1;
    defer lattice.deinit(assistant.allocator);
    var branches = try tree.bestFirstTree(assistant.allocator, &lattice, .{ .max_nodes = max_nodes, .children = children });
    defer branches.deinit(assistant.allocator);
    var result = Proposal{ .count = branches.tokens.len + 1 };
    result.tokens[0] = pending;
    result.parents[0] = -1;
    for (branches.tokens, branches.parents, 1..) |token, parent, row| {
        result.tokens[row] = token;
        result.parents[row] = parent + 1;
    }
    return result;
}

fn sameArray(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    try std.testing.expectEqual(a.ctx == null, b.ctx == null);
    if (a.ctx == null) return;
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    if (mlx.mlx_array_size(a) == 0) return;
    const av = try draft.TinyFix.readF32(a, std.testing.allocator, s);
    defer std.testing.allocator.free(av);
    const bv = try draft.TinyFix.readF32(b, std.testing.allocator, s);
    defer std.testing.allocator.free(bv);
    try std.testing.expectEqualSlices(f32, av, bv);
}

fn sameRequest(a: *const forward.Request, b: *const forward.Request, s: mlx.mlx_stream) !void {
    try std.testing.expectEqual(a.offset, b.offset);
    for (a.layers, b.layers) |left, right| {
        try std.testing.expectEqual(left.recurrent.initialized, right.recurrent.initialized);
        if (left.recurrent.initialized) {
            try sameArray(left.recurrent.conv_state, right.recurrent.conv_state, s);
            try sameArray(left.recurrent.ssm_state, right.recurrent.ssm_state, s);
        }
        try std.testing.expectEqual(left.attention.processed, right.attention.processed);
        for (left.attention.arrays(), right.attention.arrays()) |x, y| try sameArray(x, y, s);
    }
}

fn fixtureLinear(ops: *Ops, input: c_int, output: c_int) !draft.DflashLinear {
    return .{ .w = try ops.binary(.mul, try ops.ones(&.{ input, output }, .bfloat16), try ops.scalar(0.002, .bfloat16)), .scales = try ops.own(mlx.mlx_array_new()), .biases = try ops.own(mlx.mlx_array_new()) };
}

test "GLM DFlash actual branch oracle and commit match independent serial states" {
    const allocator = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(allocator);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iter = weights.map.iterator();
    var seed: usize = 5;
    while (iter.next()) |entry| {
        const value = entry.value_ptr;
        const shape = mlx.getShape(value.*);
        if (shape.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try draft.TinyFix.bf16ArrShaped(shape, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var embedding_codes: [4 * 32]u32 = undefined;
    for (&embedding_codes, 0..) |*value, i| {
        const row = i / 32;
        const col = i % 32;
        value.* = @as(u32, @intCast(1 + (row + col) % 7)) |
            (@as(u32, @intCast(1 + (3 * row + col) % 5)) << 8) |
            (@as(u32, @intCast(1 + (row + 2 * col) % 11)) << 16) |
            (@as(u32, @intCast(1 + (2 * row + col) % 13)) << 24);
    }
    const embedding = weights.map.getPtr("model.language_model.embed_tokens.weight").?;
    _ = mlx.mlx_array_free(embedding.*);
    embedding.* = mlx.mlx_array_new_data(&embedding_codes, &[_]c_int{ 4, 32 }, 2, .uint32);
    var target = try forward.Model.load(allocator, cfg, &weights, s);
    defer target.deinit();
    var request = try forward.Request.init(allocator, 4);
    defer request.deinit();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var taps = [_]u32{ 0, 3 };
    var captures = try Captures.init(allocator, &taps);
    defer captures.deinit();
    request.capture = &captures.hook;
    const prefix_ids = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &[_]c_int{ 1, 3 }, 2, .uint32));
    _ = try ops.own(try target.forward(&request, prefix_ids));
    request.capture = null;
    var original = try cloneRequest(&request);
    defer original.deinit();
    const norm = try ops.ones(&.{128}, .bfloat16);
    const lin = try fixtureLinear(&ops, 128, 128);
    const conv = draft.DynConv{ .base_kernel = try ops.ones(&.{ 2, 2, 128 }, .bfloat16), .kernel_projection = try fixtureLinear(&ops, 128, 32) };
    var layers = [_]draft.DflashLayer{.{ .layer_type = .sliding_attention, .input_norm = norm, .post_attn_norm = norm, .q = lin, .q_norm = norm, .k = lin, .k_norm = norm, .v = lin, .o = lin, .gate = lin, .up = lin, .down = lin, .attention_conv = conv, .mlp_conv = conv }};
    var types = [_]draft.LayerType{.sliding_attention};
    var assistant = draft.DflashModel{
        .allocator = allocator,
        .s = s,
        .config = .{ .hidden_size = 128, .num_hidden_layers = 1, .num_attention_heads = 1, .num_key_value_heads = 1, .head_dim = 128, .intermediate_size = 128, .rms_norm_eps = 1e-5, .rope_theta = 10000, .sliding_window = 8, .layer_types = &types, .block_size = 4, .mask_token_id = 3, .target_layer_ids = &taps, .selector_rank = 2, .selector_top_k = 2, .conv_kernel_size = 2, .conv_group_size = 16 },
        .fc = try fixtureLinear(&ops, 256, 128),
        .enc_norm = norm,
        .final_norm = norm,
        .layers = &layers,
        .selector = .{ .pred_codebook = try ops.ones(&.{ 4, 2 }, .bfloat16), .succ_codebook = try ops.ones(&.{ 4, 2 }, .bfloat16), .hidden_projection = try fixtureLinear(&ops, 128, 2) },
    };
    var context = try draft.DflashCtx.init(allocator, &assistant, 0);
    defer context.deinit();
    try draft.appendContext(&assistant, &context, captures.hook.out, 0);
    try evaluateContext(&context);
    const proposal = try proposeTree(&assistant, &context, &target, 1, 4);
    try std.testing.expectEqual(@as(usize, 5), proposal.count);
    const explicit_default = try proposeTreeWithChildren(&assistant, &context, &target, 1, 4, 4);
    try std.testing.expectEqualSlices(u32, proposal.tokens[0..proposal.count], explicit_default.tokens[0..explicit_default.count]);
    try std.testing.expectEqualSlices(i32, proposal.parents[0..proposal.count], explicit_default.parents[0..explicit_default.count]);
    const chain = try proposeTreeWithChildren(&assistant, &context, &target, 1, 3, 1);
    try std.testing.expectEqual(@as(usize, 4), chain.count);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2 }, chain.parents[0..chain.count]);
    try std.testing.expectError(error.InvalidGlmDraftTree, proposeTreeWithChildren(&assistant, &context, &target, 1, 3, 0));
    try std.testing.expectError(error.InvalidGlmDraftTree, proposeTreeWithChildren(&assistant, &context, &target, 1, 3, 17));
    try std.testing.expectEqual(@as(usize, 3), context.absLen());
    // The fixture head ties at token zero; its ancestry is root 1 -> token 0 -> token 0.
    const tokens = [_]u32{ 1, 0, 2, 0, 3 };
    const parents = [_]i32{ -1, 0, 0, 1, 2 };
    var verified = try verifyTreeOracle(&target, &request, &tokens, &parents, &taps);
    defer verified.deinit();
    try sameRequest(&request, &original, s);
    const left = try draft.TinyFix.readF32(verified.states[1].?.layers[0].recurrent.ssm_state, allocator, s);
    defer allocator.free(left);
    const right = try draft.TinyFix.readF32(verified.states[2].?.layers[0].recurrent.ssm_state, allocator, s);
    defer allocator.free(right);
    try std.testing.expect(!std.mem.eql(f32, left, right));
    for (0..tokens.len) |row| {
        var serial = try cloneRequest(&original);
        defer serial.deinit();
        var path: [16]usize = undefined;
        var count: usize = 0;
        var at: i32 = @intCast(row);
        while (at >= 0) : (at = parents[@intCast(at)]) {
            path[count] = @intCast(at);
            count += 1;
        }
        var cap = try Captures.init(allocator, &taps);
        defer cap.deinit();
        serial.capture = &cap.hook;
        while (count > 0) {
            count -= 1;
            const id = tokens[path[count]];
            const input = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const logits = try target.forwardLast(&serial, input, true);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
        serial.capture = null;
        try sameRequest(&serial, &verified.states[row].?, s);
        for (cap.hook.out, verified.captures[row].?.hook.out) |x, y| try sameArray(x, y, s);
    }
    @import("glm5_dflash_model.zig").resetStats();
    var layerwise = try @import("glm5_dflash_model.zig").verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
    defer layerwise.deinit();
    try std.testing.expectEqual(@as(usize, 0), @import("glm5_dflash_model.zig").branchFlushCount());
    try std.testing.expect(@import("glm5_dflash_model.zig").scratchBoundBytes() > 0);
    try std.testing.expectEqualSlices(u32, verified.targets[0..verified.count], layerwise.targets[0..layerwise.count]);
    const qmm_before = @import("glm5_dflash_qmm.zig").dispatchCount();
    const ffn_before = @import("glm5_dflash_ffn.zig").batchCount();
    var affine = try @import("glm5_dflash_model.zig").verify(&target, &request, &tokens, &parents, &taps, .affine_rows_ffn);
    defer affine.deinit();
    try std.testing.expectEqual(ffn_before + 1, @import("glm5_dflash_ffn.zig").batchCount());
    try std.testing.expectEqual(qmm_before, @import("glm5_dflash_qmm.zig").dispatchCount());
    try std.testing.expectEqualSlices(u32, layerwise.targets[0..layerwise.count], affine.targets[0..affine.count]);
    var affine_committed = try affine.prepareCommit(&request, 3, &.{}, s);
    defer affine_committed.deinit();
    try sameRequest(&affine_committed.states[3].?, &verified.states[3].?, s);
    var replayed = try layerwise.prepareCommit(&request, 3, &.{}, s);
    defer replayed.deinit();
    try sameRequest(&replayed.states[3].?, &verified.states[3].?, s);
    for ([_]usize{ 0, 1, 3 }) |row| {
        for (replayed.captures[row].?.hook.out, verified.captures[row].?.hook.out) |xcap, ycap| try sameArray(xcap, ycap, s);
    }
    const accepted = try commitVerified(&assistant, &context, &request, &verified, 3, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3 }, accepted.rows[0..accepted.count]);
    try std.testing.expectEqual(@as(usize, 6), request.offset);
    try std.testing.expectEqual(request.offset, context.absLen());
    try std.testing.expectError(error.InvalidGlmDraftOffset, commitVerified(&assistant, &context, &request, &verified, 3, &.{}));
    request.reset();
    context.deinit();
    context = try draft.DflashCtx.init(allocator, &assistant, 0);
    const pending = try prefill(&assistant, &context, &target, &request, prefix_ids);
    const round = try roundTreeOracle(std.Io.Threaded.global_single_threaded.io(), &assistant, &context, &target, &request, pending, 4, 2, &.{});
    try std.testing.expect(round.count >= 1 and round.count <= 2);
    try std.testing.expectEqual(@as(usize, 5), round.verified_rows);
    try std.testing.expectEqual(@as(usize, 3) + round.count, request.offset);
    try std.testing.expectEqual(request.offset, context.absLen());
    const stopped = try roundTreeLayerwise(std.Io.Threaded.global_single_threaded.io(), &assistant, &context, &target, &request, 0, 4, 3, &.{0});
    try std.testing.expect(stopped.stopped and stopped.pending == null);
    try std.testing.expectEqual(@as(usize, 1), stopped.verified_rows);
    target.cfg.max_position_embeddings = 4096;
    for ([_]usize{ 1, 2, 4, 255, 2051 }) |prefix_length| {
        request.reset();
        const prompt = try allocator.alloc(u32, prefix_length);
        defer allocator.free(prompt);
        for (prompt, 0..) |*token, i| token.* = @intCast(i % 4);
        const inputs = mlx.mlx_array_new_data(prompt.ptr, &[_]c_int{ 1, @intCast(prefix_length) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(inputs);
        const logits = try target.forwardLast(&request, inputs, true);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        var reference = try verifyTreeOracle(&target, &request, &tokens, &parents, &taps);
        defer reference.deinit();
        var batch = try @import("glm5_dflash_model.zig").verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
        defer batch.deinit();
        try std.testing.expectEqualSlices(u32, reference.targets[0..reference.count], batch.targets[0..batch.count]);
        for ([_]usize{ 1, 2, 3 }) |budget| {
            var committed = try batch.prepareCommit(&request, budget, &.{}, s);
            defer committed.deinit();
            const keep = try tree.accept(&tokens, &parents, reference.targets[0..reference.count], budget, &.{});
            const last = keep.rows[keep.count - 1];
            try sameRequest(&committed.states[last].?, &reference.states[last].?, s);
            for (keep.rows[0..keep.count]) |row| for (committed.captures[row].?.hook.out, reference.captures[row].?.hook.out) |xcap, ycap| try sameArray(xcap, ycap, s);
        }
    }
}

/// Stored assistant matrices are used directly: no load-time requantization.
pub fn loadAssistantStored(io: std.Io, allocator: std.mem.Allocator, directory: []const u8, target: *const forward.Model) !draft.DflashModel {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "config.json", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.GlmDraftTargetMismatch;
    const vocab = parsed.value.object.get("vocab_size") orelse return error.GlmDraftTargetMismatch;
    const count = parsed.value.object.get("num_target_layers") orelse return error.GlmDraftTargetMismatch;
    if (vocab != .integer or vocab.integer != target.cfg.vocab_size or count != .integer or count.integer != target.cfg.num_hidden_layers) return error.GlmDraftTargetMismatch;
    var assistant = try draft.loadDflashQuant(io, allocator, target.s, directory, 0);
    errdefer assistant.deinit();
    try validatePair(&assistant, target);
    _ = try assistantStorage(&assistant);
    if (@import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_MINI_HEAD")) {
        if (assistant.config.block_size > 8 or assistant.config.selector_top_k > 32 or assistant.config.logit_softcap > 0 or
            !std.math.isFinite(assistant.config.output_multiplier) or assistant.config.output_multiplier <= 0) return error.UnsupportedGlmMiniHead;
        const mini = @import("glm5_dflash_mini.zig");
        assistant.draft_head = try mini.build(target.head, target.s);
        assistant.draft_head_bits = mini.bits;
        assistant.draft_head_group = mini.group_size;
        @import("log.zig").info("[glm-dflash] mini draft readout: {d}-bit/gs{d} -> top-32 -> stored-head re-score; selector top-{d}; {d} resident bytes\n", .{ mini.bits, mini.group_size, assistant.config.selector_top_k, mini.residentBytes(target.head.output, target.head.input) });
    }
    return assistant;
}

/// Retained for callers that explicitly require the original BF16 assistant.
pub fn loadAssistantBf16(io: std.Io, allocator: std.mem.Allocator, directory: []const u8, target: *const forward.Model) !draft.DflashModel {
    var assistant = try loadAssistantStored(io, allocator, directory, target);
    errdefer assistant.deinit();
    if ((try assistantStorage(&assistant)).affine_linears != 0) return error.GlmDraftRequiresBf16;
    return assistant;
}

pub const AssistantStorage = struct {
    bits: u32,
    group_size: u32,
    dense_linears: usize = 0,
    affine_linears: usize = 0,
    pub fn label(self: AssistantStorage) []const u8 {
        return if (self.affine_linears == 0) "BF16" else if (self.bits == 4 and self.group_size == 64) "A4g64" else if (self.bits == 8) "A8g128" else "A6g128";
    }
    fn add(self: *AssistantStorage, linear: *const draft.DflashLinear) !void {
        if (linear.w.ctx == null) return error.UnsupportedGlmDraftStorage;
        const ws = mlx.getShape(linear.w);
        if (ws.len != 2 or ws[0] <= 0 or ws[1] <= 0) return error.UnsupportedGlmDraftStorage;
        if (linear.bits == 0) {
            if (mlx.mlx_array_dtype(linear.w) != .bfloat16 or linear.scales.ctx != null or linear.biases.ctx != null) return error.UnsupportedGlmDraftStorage;
            self.dense_linears += 1;
            return;
        }
        if (!((linear.bits == 4 and linear.group_size == 64) or ((linear.bits == 6 or linear.bits == 8) and linear.group_size == 128)) or linear.bits != self.bits or linear.group_size != self.group_size or
            mlx.mlx_array_dtype(linear.w) != .uint32 or linear.scales.ctx == null or linear.biases.ctx == null or
            mlx.mlx_array_dtype(linear.scales) != .bfloat16 or mlx.mlx_array_dtype(linear.biases) != .bfloat16) return error.UnsupportedGlmDraftStorage;
        const ss = mlx.getShape(linear.scales);
        if (ss.len != 2 or ss[0] != ws[0] or ss[1] <= 0 or !std.mem.eql(c_int, ss, mlx.getShape(linear.biases)) or
            @as(u64, @intCast(ws[1])) * 32 != @as(u64, @intCast(ss[1])) * linear.group_size * linear.bits) return error.UnsupportedGlmDraftStorage;
        self.affine_linears += 1;
    }
};

/// Validate every contracted matrix, including dynamic convolutions and selector.
/// Dense small matrices may be retained alongside one uniform stored affine rate.
pub fn assistantStorage(assistant: *const draft.DflashModel) !AssistantStorage {
    var result = AssistantStorage{ .bits = assistant.fc.bits, .group_size = assistant.fc.group_size };
    try result.add(&assistant.fc);
    for (assistant.layers) |*layer| {
        for ([_]*const draft.DflashLinear{ &layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down }) |linear| try result.add(linear);
        if (layer.attention_conv) |*conv| try result.add(&conv.kernel_projection);
        if (layer.mlp_conv) |*conv| try result.add(&conv.kernel_projection);
    }
    if (assistant.selector) |*selector| try result.add(&selector.hidden_projection);
    return result;
}

/// Context capture is evaluated per caller-selected prefill chunk, without retaining full hidden history.
pub fn prefill(assistant: *const draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, ids: Arr) !u32 {
    try validatePair(assistant, target);
    if (request.offset != context.absLen() or request.capture != null) return error.InvalidGlmDraftOffset;
    var next = try cloneRequest(request);
    errdefer next.deinit();
    var next_context = try cloneContext(assistant, context);
    errdefer next_context.deinit();
    var captures = try Captures.init(request.allocator, assistant.config.target_layer_ids);
    defer captures.deinit();
    next.capture = &captures.hook;
    const logits = try target.forwardLast(&next, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    next.capture = null;
    const pending = try argmax(logits, target.s);
    try draft.appendContext(assistant, &next_context, captures.hook.out, request.offset);
    try evaluateContext(&next_context);
    request.deinit();
    request.* = next;
    context.deinit();
    context.* = next_context;
    return pending;
}

pub const RoundResult = struct {
    tokens: [16]u32 = undefined,
    count: usize,
    pending: ?u32,
    stopped: bool,
    verified_rows: usize,
    accepted_drafts: usize,
    draft_ns: u64,
    verify_ns: u64,
    commit_ns: u64,
    replay_ns: u64 = 0,
    verifier: []const u8 = "serial_branch_oracle",
};

/// Explicit diagnostic entry point; all candidate nodes use the unchanged serial arithmetic.
pub fn roundTreeOracle(io: std.Io, assistant: *draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32) !RoundResult {
    if (budget == 0) return error.InvalidGlmDraftBudget;
    if (request.offset != context.absLen()) return error.InvalidGlmDraftOffset;
    const clock = @import("io_util.zig").Stopwatch;
    var timer = clock.init(io);
    const proposal = if (budget == 1 or std.mem.indexOfScalar(u32, eos, pending) != null) blk: {
        var one = Proposal{ .count = 1 };
        one.tokens[0] = pending;
        one.parents[0] = -1;
        break :blk one;
    } else try proposeTree(assistant, context, target, pending, max_nodes);
    const draft_ns = timer.read();
    timer.reset();
    var verified = try verifyTreeOracle(target, request, proposal.tokens[0..proposal.count], proposal.parents[0..proposal.count], assistant.config.target_layer_ids);
    defer verified.deinit();
    const verify_ns = timer.read();
    timer.reset();
    const kept = try commitVerified(assistant, context, request, &verified, budget, eos);
    var result = RoundResult{ .count = kept.count, .pending = kept.pending, .stopped = kept.stopped, .verified_rows = proposal.count, .accepted_drafts = kept.count - 1, .draft_ns = draft_ns, .verify_ns = verify_ns, .commit_ns = timer.read() };
    for (kept.rows[0..kept.count], 0..) |row, i| result.tokens[i] = proposal.tokens[row];
    return result;
}

test {
    _ = @import("glm5_dflash_kda.zig");
}

test {
    _ = @import("glm5_dflash_model.zig");
}

pub fn roundTreeLayerwise(io: std.Io, assistant: *draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32) !RoundResult {
    return roundTreeLayerwiseMode(io, assistant, context, target, request, pending, max_nodes, budget, eos, .serial_rows);
}

pub fn roundTreeLayerwiseMode(io: std.Io, assistant: *draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32, mode: @import("glm5_dflash_kda.zig").ProjectionMode) !RoundResult {
    return roundTreeLayerwiseConfigured(io, assistant, context, target, request, pending, max_nodes, budget, eos, mode, 4);
}

pub fn roundTreeLayerwiseConfigured(io: std.Io, assistant: *draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32, mode: @import("glm5_dflash_kda.zig").ProjectionMode, children: usize) !RoundResult {
    if (children == 0 or children > 16) return error.InvalidGlmDraftTree;
    if (mode == .batched) return error.GlmBatchedVerifyUnqualified;
    try validatePair(assistant, target);
    if (budget == 0) return error.InvalidGlmDraftBudget;
    if (request.offset != context.absLen()) return error.InvalidGlmDraftOffset;
    var timer = @import("io_util.zig").Stopwatch.init(io);
    const proposal = if (budget == 1 or std.mem.indexOfScalar(u32, eos, pending) != null) blk: {
        var one = Proposal{ .count = 1 };
        one.tokens[0] = pending;
        one.parents[0] = -1;
        break :blk one;
    } else try proposeTreeWithChildren(assistant, context, target, pending, max_nodes, children);
    const draft_ns = timer.read();
    timer.reset();
    var layerwise = try @import("glm5_dflash_model.zig").verify(target, request, proposal.tokens[0..proposal.count], proposal.parents[0..proposal.count], assistant.config.target_layer_ids, mode);
    defer layerwise.deinit();
    const verify_ns = timer.read();
    timer.reset();
    var verified = try layerwise.prepareCommit(request, budget, eos, target.s);
    defer verified.deinit();
    const replay_ns = timer.read();
    timer.reset();
    const kept = try commitVerified(assistant, context, request, &verified, budget, eos);
    var result = RoundResult{ .count = kept.count, .pending = kept.pending, .stopped = kept.stopped, .verified_rows = proposal.count, .accepted_drafts = kept.count - 1, .draft_ns = draft_ns, .verify_ns = verify_ns, .replay_ns = replay_ns, .commit_ns = timer.read(), .verifier = if (mode == .affine_rows_ffn) "layerwise_tree_affine_ffn_tiles" else if (mode == .affine_rows) "layerwise_tree_affine_row_tiles" else "layerwise_tree_serial_projections" };
    for (kept.rows[0..kept.count], 0..) |row, i| result.tokens[i] = proposal.tokens[row];
    return result;
}

test {
    _ = @import("glm5_dflash_diagnostic.zig");
}

test {
    _ = @import("glm5_dflash_qmm.zig");
}

test {
    _ = @import("glm5_dflash_ffn.zig");
}

test "GLM draft stored BF16 A4 A6 and A8 validate packed geometry without changing arrays" {
    const nil = mlx.mlx_array{ .ctx = null };
    const codes: [32]u32 = @splat(0);
    const scales: [2]u16 = .{ 0x3f80, 0 };
    for ([_][2]u32{ .{ 4, 64 }, .{ 6, 128 }, .{ 8, 128 } }) |spec| {
        const bits = spec[0];
        const group = spec[1];
        const cols: c_int = @intCast(group * bits / 32);
        const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ 1, cols }, 2, .uint32);
        defer _ = mlx.mlx_array_free(w);
        const scale = mlx.mlx_array_new_data(&scales, &[_]c_int{ 1, 1 }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(scale);
        var linear = draft.DflashLinear{ .w = w, .scales = scale, .biases = scale, .bits = bits, .group_size = group };
        var storage = AssistantStorage{ .bits = bits, .group_size = group };
        try storage.add(&linear);
        try std.testing.expectEqual(@as(usize, 1), storage.affine_linears);
        try std.testing.expectEqual(w.ctx, linear.w.ctx);
        try std.testing.expectEqual(scale.ctx, linear.scales.ctx);
        try std.testing.expectEqual(scale.ctx, linear.biases.ctx);
        try std.testing.expectEqualStrings(if (bits == 4) "A4g64" else if (bits == 6) "A6g128" else "A8g128", storage.label());
        linear.group_size = if (group == 64) 128 else 64;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        linear.group_size = group;
        linear.biases = nil;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        const bad = mlx.mlx_array_new_data(&scales, &[_]c_int{ 1, 2 }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(bad);
        linear.biases = bad;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        linear.scales = bad;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        linear.scales = scale;
        linear.biases = scale;
        linear.bits = if (bits == 6) 8 else 6;
        linear.group_size = 128;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
    }
    const w = mlx.mlx_array_new_data(&scales, &[_]c_int{ 1, 2 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    const dense = draft.DflashLinear{ .w = w, .scales = nil, .biases = nil };
    var storage = AssistantStorage{ .bits = 0, .group_size = 0 };
    try storage.add(&dense);
    try std.testing.expectEqual(@as(usize, 1), storage.dense_linears);
    try std.testing.expectEqualStrings("BF16", storage.label());
}

test "GLM N2 best-first selection is independent of unused future lattice depths" {
    const k = 4;
    var cands: [7 * k]i32 = undefined;
    var unary: [7 * k]f32 = undefined;
    var anchor: [k]f32 = undefined;
    var edges: [6 * k * k]f32 = undefined;
    for (&cands, &unary, 0..) |*id, *v, i| {
        id.* = @intCast(100 + i);
        v.* = @as(f32, @floatFromInt((i * 7) % 13)) / 8 - 0.5;
    }
    for (&anchor, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 4;
    for (&edges, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 3) % 17)) / 8 - 0.75;
    const full = tree.Lattice{ .m = 7, .k = k, .cands = &cands, .unary = &unary, .e0 = &anchor, .e = &edges };
    const short = tree.Lattice{ .m = 2, .k = k, .cands = cands[0 .. 2 * k], .unary = unary[0 .. 2 * k], .e0 = &anchor, .e = edges[0 .. k * k] };
    for ([_]usize{ 1, 2, 4 }) |children| {
        var original = try tree.bestFirstTree(std.testing.allocator, &full, .{ .max_nodes = 2, .children = children });
        defer original.deinit(std.testing.allocator);
        var bounded = try tree.bestFirstTree(std.testing.allocator, &short, .{ .max_nodes = 2, .children = children });
        defer bounded.deinit(std.testing.allocator);
        try std.testing.expectEqualSlices(u32, original.tokens, bounded.tokens);
        try std.testing.expectEqualSlices(i32, original.parents, bounded.parents);
        try std.testing.expectEqualSlices(u32, original.depth, bounded.depth);
    }
}

test "GLM N2 horizon production head and selector component" {
    const head_path = std.c.getenv("SUSHI_GLM_HORIZON_HEAD_SHARD") orelse return error.SkipZigTest;
    const selector_path = std.c.getenv("SUSHI_GLM_HORIZON_SELECTOR_SHARD") orelse return error.MissingGlmDiagnosticInput;
    const output = std.c.getenv("SUSHI_GLM_HORIZON_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, head_path, cpu));
    var head_arrays: [3]Arr = undefined;
    for ([_][*:0]const u8{ "lm_head.weight", "lm_head.scales", "lm_head.biases" }, 0..) |name, i| {
        const slot = try ops.slot();
        try mlx.check(mlx.mlx_map_string_to_array_get(slot, arrays, name));
        head_arrays[i] = slot.*;
    }
    const native = @import("glm5_model.zig");
    const head = native.Linear{ .w = head_arrays[0], .scales = head_arrays[1], .biases = head_arrays[2], .input = mlx.getShape(head_arrays[1])[1] * 128, .output = mlx.getShape(head_arrays[0])[0] };
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, selector_path, cpu));
    var selector_arrays: [3]Arr = undefined;
    for ([_][*:0]const u8{ "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook", "candidate_selector.hidden_projection.weight" }, 0..) |name, i| {
        const slot = try ops.slot();
        try mlx.check(mlx.mlx_map_string_to_array_get(slot, arrays, name));
        selector_arrays[i] = slot.*;
    }
    const selector = draft.Selector{ .pred_codebook = selector_arrays[0], .succ_codebook = selector_arrays[1], .hidden_projection = .{ .w = try ops.transpose(selector_arrays[2], &.{ 1, 0 }), .scales = .{ .ctx = null }, .biases = .{ .ctx = null } } };
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 0x53132));
    const hidden = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(hidden, &[_]c_int{ 1, 8, head.input }, 3, .bfloat16, 0, 1, key.*, s));
    try mlx.check(mlx.mlx_array_eval(hidden.*));
    const original = try head.apply(&ops, hidden.*);
    const bounded = try head.apply(&ops, try ops.slice(hidden.*, 1, 1, 3));
    try mlx.check(mlx.mlx_array_eval(original));
    try mlx.check(mlx.mlx_array_eval(bounded));
    const vocab: usize = @intCast(head.output);
    const expected = mlx.mlx_array_data_bfloat16(original) orelse return error.MlxArrayDataNull;
    const actual = mlx.mlx_array_data_bfloat16(bounded) orelse return error.MlxArrayDataNull;
    var logit_mismatches: usize = 0;
    for (0..2 * vocab) |i| if (expected[vocab + i] != actual[i]) {
        logit_mismatches += 1;
    };
    // Use the shipped selector on the same immutable eight-row fixture.
    var full = try tree.lattice(std.testing.allocator, &selector, 16, hidden.*, try ops.slice(original, 1, 1, 8), 1, s);
    defer full.deinit(std.testing.allocator);
    var short = try tree.lattice(std.testing.allocator, &selector, 16, try ops.slice(hidden.*, 1, 0, 3), bounded, 1, s);
    defer short.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(i32, full.cands[0..32], short.cands);
    try std.testing.expectEqualSlices(f32, full.unary[0..32], short.unary);
    try std.testing.expectEqualSlices(f32, full.e0, short.e0);
    try std.testing.expectEqualSlices(f32, full.e[0..256], short.e);
    for ([_]usize{ 1, 2, 4 }) |children| {
        var before = try tree.bestFirstTree(std.testing.allocator, &full, .{ .max_nodes = 2, .children = children });
        defer before.deinit(std.testing.allocator);
        var after = try tree.bestFirstTree(std.testing.allocator, &short, .{ .max_nodes = 2, .children = children });
        defer after.deinit(std.testing.allocator);
        try std.testing.expectEqualSlices(u32, before.tokens, after.tokens);
        try std.testing.expectEqualSlices(i32, before.parents, after.parents);
        try std.testing.expectEqualSlices(u32, before.depth, after.depth);
    }
    // Isolated readout+lattice phase only; the proposal hot path has no clocks.
    var full_ns: [4]u64 = undefined;
    var horizon_ns: [4]u64 = undefined;
    for (0..6) |iteration| {
        for (0..2) |arm| {
            const use_horizon = (iteration + arm) % 2 == 1;
            var scope = Ops{ .s = s };
            defer scope.deinit();
            const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
            const input = if (use_horizon) try scope.slice(hidden.*, 1, 1, 3) else hidden.*;
            const projected = try head.apply(&scope, input);
            const logits = if (use_horizon) projected else try scope.slice(projected, 1, 1, 8);
            const selector_input = if (use_horizon) try scope.slice(hidden.*, 1, 0, 3) else hidden.*;
            var lat = try tree.lattice(std.testing.allocator, &selector, 16, selector_input, logits, 1, s);
            defer lat.deinit(std.testing.allocator);
            var selected = try tree.bestFirstTree(std.testing.allocator, &lat, .{ .max_nodes = 2, .children = 4 });
            defer selected.deinit(std.testing.allocator);
            const ns = watch.read();
            if (iteration >= 2) {
                if (use_horizon) horizon_ns[iteration - 2] = ns else full_ns[iteration - 2] = ns;
            }
        }
    }
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .head_bits = try native.storedAffineBits(head.w, head.scales, head.biases), .vocab = vocab, .hidden = head.input, .selector_rank = mlx.getShape(selector.pred_codebook)[1], .input = "fixed-seed random BF16 hidden, seed 0x53132; real target A6 head and real assistant BF16 selector; no full model loaded", .first_two_logit_bit_mismatches = logit_mismatches, .candidate_rows_equal = 2, .unary_rows_equal = 2, .anchor_and_first_edge_equal = true, .n2_tree_children_equal = .{ 1, 2, 4 }, .timing = "readout+lattice+N2 tree only; four interleaved pairs after two warmup pairs; ns", .full_ns = full_ns, .horizon_ns = horizon_ns }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
    try std.testing.expectEqual(@as(usize, 0), logit_mismatches);
}

test "GLM assistant commit window actual A6 preserves KV and proposals" {
    const dir = std.c.getenv("SUSHI_GLM_COMMIT_WINDOW_ASSISTANT") orelse return error.SkipZigTest;
    const head_path = std.c.getenv("SUSHI_GLM_COMMIT_WINDOW_HEAD_SHARD") orelse return error.MissingGlmDiagnosticInput;
    const output = std.c.getenv("SUSHI_GLM_COMMIT_WINDOW_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const allocator = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    var assistant = try draft.loadDflashQuant(io, allocator, s, std.mem.span(dir), 0);
    defer assistant.deinit();
    var source = try draft.DflashCtx.init(allocator, &assistant, 37);
    defer source.deinit();
    source.cache.reserve_tokens = 33024;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 0x53132));
    for (0..5) |li| {
        const k = try ops.slot();
        const v = try ops.slot();
        for ([_]*Arr{ k, v }) |a| try mlx.check(mlx.mlx_random_normal(a, &[_]c_int{ 1, 8, 32768, 128 }, 4, .bfloat16, 0, 1, key.*, s));
        _ = try source.cache.update(@intCast(li), k.*, v.*, s, 0);
    }
    try evaluateContext(&source);
    const CacheHash = struct {
        fn run(ctx: *const draft.DflashCtx) ![32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            for (ctx.cache.entries) |entry| for ([_]Arr{ entry.keys, entry.values }) |a| {
                const data = mlx.mlx_array_data_bfloat16(a) orelse return error.MlxArrayDataNull;
                hash.update(std.mem.sliceAsBytes(data[0..mlx.mlx_array_size(a)]));
            };
            return hash.finalResult();
        }
    };
    const source_hash = try CacheHash.run(&source);
    var captures: [5]Arr = undefined;
    for (&captures) |*capture| {
        const a = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(a, &[_]c_int{ 1, 3, 4096 }, 3, .bfloat16, 0, 1, key.*, s));
        capture.* = a.*;
        try mlx.check(mlx.mlx_array_eval(a.*));
    }
    var full = try cloneContext(&assistant, &source);
    defer full.deinit();
    try draft.appendContext(&assistant, &full, &captures, source.absLen());
    try evaluateContext(&full);
    var windowed = try cloneContext(&assistant, &source);
    defer windowed.deinit();
    try std.testing.expect(try cropCommitContext(&assistant, &windowed));
    try std.testing.expectEqual(source.absLen(), windowed.absLen());
    try std.testing.expectEqual(@as(usize, 37 + 32768 - 2047), windowed.base_pos);
    try draft.appendContext(&assistant, &windowed, &captures, source.absLen());
    try evaluateContext(&windowed);
    try std.testing.expectEqual(full.absLen(), windowed.absLen());
    try std.testing.expectEqual(@as(usize, 32768 + 3), full.cache.step);
    try std.testing.expectEqual(@as(usize, 2050), windowed.cache.step);
    var compared: usize = 0;
    for (full.cache.entries, windowed.cache.entries) |old, new| for ([_][2]Arr{ .{ old.keys, new.keys }, .{ old.values, new.values } }) |pair| {
        const expected = try ops.contiguous(try ops.slice(pair[0], 2, 32768 - 2047, 32768 + 3));
        const actual = try ops.contiguous(try ops.slice(pair[1], 2, 0, 2050));
        try mlx.check(mlx.mlx_array_eval(expected));
        try mlx.check(mlx.mlx_array_eval(actual));
        const n = mlx.mlx_array_size(expected);
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..n], mlx.mlx_array_data_bfloat16(actual).?[0..n]);
        compared += n;
    };
    const noise = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(noise, &[_]c_int{ 1, 8, 4096 }, 3, .bfloat16, 0, 1, key.*, s));
    const hidden_full = try ops.own(try draft.forwardBlock(&assistant, &full, noise.*, full.absLen()));
    const hidden_window = try ops.own(try draft.forwardBlock(&assistant, &windowed, noise.*, windowed.absLen()));
    try mlx.check(mlx.mlx_array_eval(hidden_full));
    try mlx.check(mlx.mlx_array_eval(hidden_window));
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(hidden_full).?[0 .. 8 * 4096], mlx.mlx_array_data_bfloat16(hidden_window).?[0 .. 8 * 4096]);
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, head_path, cpu));
    var head_arrays: [3]Arr = undefined;
    for ([_][*:0]const u8{ "lm_head.weight", "lm_head.scales", "lm_head.biases" }, 0..) |name, i| {
        const a = try ops.slot();
        try mlx.check(mlx.mlx_map_string_to_array_get(a, arrays, name));
        head_arrays[i] = a.*;
    }
    const head = @import("glm5_model.zig").Linear{ .w = head_arrays[0], .scales = head_arrays[1], .biases = head_arrays[2], .input = 4096, .output = mlx.getShape(head_arrays[0])[0] };
    var logits: [2]Arr = undefined;
    var lattices: [2]tree.Lattice = undefined;
    for ([_]Arr{ hidden_full, hidden_window }, 0..) |hidden, arm| {
        logits[arm] = try ops.own(try draft.applyLogitTransforms(try head.apply(&ops, try ops.slice(hidden, 1, 1, 3)), assistant.config.output_multiplier, assistant.config.logit_softcap, s));
        lattices[arm] = try tree.lattice(allocator, &assistant.selector.?, 16, try ops.slice(hidden, 1, 0, 3), logits[arm], 1, s);
    }
    defer for (&lattices) |*lat| lat.deinit(allocator);
    const logit_count = mlx.mlx_array_size(logits[0]);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(logits[0]).?[0..logit_count], mlx.mlx_array_data_bfloat16(logits[1]).?[0..logit_count]);
    try std.testing.expectEqualSlices(i32, lattices[0].cands, lattices[1].cands);
    try std.testing.expectEqualSlices(f32, lattices[0].unary, lattices[1].unary);
    try std.testing.expectEqualSlices(f32, lattices[0].e0, lattices[1].e0);
    try std.testing.expectEqualSlices(f32, lattices[0].e, lattices[1].e);
    for ([_]usize{ 1, 2, 4 }) |children| {
        var before = try tree.bestFirstTree(allocator, &lattices[0], .{ .max_nodes = 2, .children = children });
        defer before.deinit(allocator);
        var after = try tree.bestFirstTree(allocator, &lattices[1], .{ .max_nodes = 2, .children = children });
        defer after.deinit(allocator);
        try std.testing.expectEqualSlices(u32, before.tokens, after.tokens);
        try std.testing.expectEqualSlices(i32, before.parents, after.parents);
        try std.testing.expectEqualSlices(u32, before.depth, after.depth);
    }
    var original_ns: [4]u64 = undefined;
    var window_ns: [4]u64 = undefined;
    for (0..6) |iteration| for (0..2) |arm| {
        const crop = (iteration + arm) % 2 == 1;
        const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
        var work = try cloneContext(&assistant, &source);
        defer work.deinit();
        if (crop) try std.testing.expect(try cropCommitContext(&assistant, &work));
        try draft.appendContext(&assistant, &work, &captures, source.absLen());
        try evaluateContext(&work);
        const ns = watch.read();
        if (iteration >= 2) {
            if (crop) window_ns[iteration - 2] = ns else original_ns[iteration - 2] = ns;
        }
    };
    try std.testing.expectEqualSlices(u8, &source_hash, &(try CacheHash.run(&source)));
    try std.testing.expectEqual(@as(usize, 32768), source.cache.step);
    try std.testing.expectEqual(@as(usize, 37), source.base_pos);
    const json = try std.json.Stringify.valueAlloc(allocator, .{ .input = "actual A6 assistant/head, fixedseed BF16 KV32768 +three accepted capture rows +full8 block, base37", .source_hash_unchanged = true, .retained_and_appended_kv_bits_equal = compared, .absolute_length = windowed.absLen(), .window_base = windowed.base_pos, .window_rows = windowed.cache.step, .replacement_capacity = mlx.getShape(windowed.cache.entries[0].keys)[2], .full8_hidden_bits_equal = 8 * 4096, .first_two_readout_bits_equal = logit_count, .lattice_scores_equal = true, .n2_trees_equal = true, .timing = "clone+optionalcrop+acceptedfeatureprojection+KVappend+eval; four alternating pairs after two warmups; ns", .full_ns = original_ns, .window_ns = window_ns }, .{ .whitespace = .indent_2 });
    defer allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
