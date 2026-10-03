//! Explicit full-checkpoint DFlash2 correctness/timing diagnostic; never a serving gate.
const std = @import("std");
const mlx = @import("mlx.zig");
const adapter = @import("glm5_dflash.zig");
const native = @import("glm5_diagnostic.zig");
const forward = @import("glm5_forward.zig");
const Arr = mlx.mlx_array;

fn writeJson(io: std.Io, a: std.mem.Allocator, path: []const u8, value: anytype) !void {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{ .whitespace = .indent_2 });
    defer a.free(raw);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = raw });
}
fn number(comptime name: [:0]const u8, fallback: usize) !usize {
    return if (std.c.getenv(name)) |p| std.fmt.parseInt(usize, std.mem.span(p), 10) else fallback;
}
fn greedy(logits: Arr, stream: mlx.mlx_stream) !u32 {
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_argmax_axis(&out, logits, -1, false, stream));
    try mlx.check(mlx.mlx_array_eval(out));
    if (mlx.mlx_array_size(out) != 1) return error.InvalidGlmDraftLogits;
    return (mlx.mlx_array_data_uint32(out) orelse return error.MlxArrayDataNull)[0];
}
fn arrayMatches(a: Arr, b: Arr, stream: mlx.mlx_stream) !bool {
    if (a.ctx == null or b.ctx == null) return a.ctx == null and b.ctx == null;
    if (mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b) or !std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return false;
    var equal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(equal);
    try mlx.check(mlx.mlx_array_equal(&equal, a, b, false, stream));
    try mlx.check(mlx.mlx_array_eval(equal));
    return (mlx.mlx_array_data_bool(equal) orelse return error.MlxArrayDataNull)[0];
}
fn stateMatches(a: *const forward.Request, b: *const forward.Request, stream: mlx.mlx_stream) !bool {
    if (a.offset != b.offset or a.layers.len != b.layers.len) return false;
    for (a.layers, b.layers) |x, y| {
        if (x.recurrent.initialized != y.recurrent.initialized or x.attention.processed != y.attention.processed) return false;
        if (x.recurrent.initialized) {
            if (!try arrayMatches(x.recurrent.ssm_state, y.recurrent.ssm_state, stream) or !try arrayMatches(x.recurrent.conv_state, y.recurrent.conv_state, stream)) return false;
        }
        for (x.attention.arrays(), y.attention.arrays()) |left, right| if (!try arrayMatches(left, right, stream)) return false;
    }
    return true;
}
const Round = struct { emitted: usize, accepted_drafts: usize, verified_rows: usize, draft_ns: u64, verify_ns: u64, replay_ns: u64, commit_ns: u64 };

test "GLM DFlash real checkpoint diagnostic" {
    const model_env = std.c.getenv("SUSHI_GLM_DFLASH_MODEL") orelse return error.SkipZigTest;
    const assistant_env = std.c.getenv("SUSHI_GLM_DFLASH_ASSISTANT") orelse return error.MissingGlmDraftAssistant;
    const tokens_env = std.c.getenv("SUSHI_GLM_DFLASH_TOKENS_FILE") orelse return error.MissingGlmDiagnosticPrompt;
    const output_env = std.c.getenv("SUSHI_GLM_DFLASH_OUT") orelse return error.MissingGlmDiagnosticOutput;
    const model_path = std.mem.span(model_env);
    const assistant_path = std.mem.span(assistant_env);
    const output_path = std.mem.span(output_env);
    const a = std.testing.allocator;
    const io = std.testing.io;
    const stream = mlx.gpuStream();
    const progress = try std.fmt.allocPrint(a, "{s}.progress.json", .{output_path});
    defer a.free(progress);
    errdefer writeJson(io, a, progress, .{ .complete = false, .phase = "failed" }) catch {};
    const count = try number("SUSHI_GLM_DFLASH_PREFILL", 32);
    const steps = try number("SUSHI_GLM_DFLASH_DECODE", 8);
    const nodes = try number("SUSHI_GLM_DFLASH_NODES", 3);
    const children = try number("SUSHI_GLM_DFLASH_CHILDREN", 4);
    if (children == 0 or children > 16) return error.InvalidGlmDraftTree;
    const async_layers = try number("SUSHI_GLM_DFLASH_ASYNC_LAYERS", 0);
    const schedule_binding = try @import("glm5_dflash_model.zig").bindSchedule(async_layers);
    defer schedule_binding.restore();
    const affine_rows = @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_AFFINE_ROWS");
    const batch_ffn = @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_BATCH_FFN");
    const profile_enabled = @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_PROFILE");
    const capture_routes = @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_CAPTURE_ROUTES");
    if (capture_routes and !batch_ffn) return error.GlmRouteCaptureRequiresBatchedFfn;
    const route_capacity = if (capture_routes) try number("SUSHI_GLM_DFLASH_ROUTE_CAPACITY", 4096) else 0;
    var route_capture: ?@import("glm5_dflash_profile.zig").RouteCapture = if (capture_routes) try @import("glm5_dflash_profile.zig").RouteCapture.init(a, route_capacity) else null;
    defer if (route_capture) |*capture| capture.deinit();
    const synchronization_perturbed = profile_enabled or capture_routes;
    if (batch_ffn and !affine_rows) return error.GlmFfnRequiresAffineRows;
    const verify_mode: @import("glm5_dflash_kda.zig").ProjectionMode = if (batch_ffn) .affine_rows_ffn else if (affine_rows) .affine_rows else .serial_rows;
    const warmup = try number("SUSHI_GLM_DFLASH_WARMUP", 0);
    if (warmup > 4) return error.InvalidGlmDiagnosticBudget;
    const chunk = try number("SUSHI_GLM_DFLASH_CHUNK", 128);
    const dense_prefill = @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_DENSE_PREFILL");
    const prefill_async = @import("transformer.zig").diagEnvOn("SUSHI_GLM_DFLASH_PREFILL_ASYNC");
    const prefill_sync_layers = try number("SUSHI_GLM_DFLASH_PREFILL_SYNC_LAYERS", 2);
    if (prefill_sync_layers == 0 or prefill_sync_layers > 8) return error.InvalidGlmPrefillSchedule;
    if (count == 0 or count > 65536 or steps == 0 or steps > 4096 or nodes == 0 or nodes > 15 or chunk == 0) return error.InvalidGlmDiagnosticBudget;
    const memory = (try number("SUSHI_GLM_DFLASH_MEMORY_GIB", 110)) * 1024 * 1024 * 1024;
    const wired = @min(memory, mlx.maxRecommendedWorkingSet());
    var old_memory: usize = 0;
    var old_cache: usize = 0;
    var old_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_memory, memory));
    defer _ = mlx.mlx_set_memory_limit(&ignored, old_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, 2 * 1024 * 1024 * 1024));
    defer _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    try mlx.check(mlx.mlx_set_wired_limit(&old_wired, wired));
    defer _ = mlx.mlx_set_wired_limit(&ignored, old_wired);
    var cfg = try @import("model.zig").parseConfig(io, a, model_path);
    defer cfg.deinit(a);
    try @import("glm5_attention_decode_batch.zig").admit(&cfg, .bfloat16, stream);
    if (!cfg.isGlm5() or cfg.expert_layout != .exl3_k4) return error.UnsupportedGlmDiagnosticLayout;
    try @import("mimo_source.zig").validateExl3Pack(io, a, model_path, &cfg);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(tokens_env), a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    const ids_value = if (parsed.value == .object) parsed.value.object.get("ids") orelse return error.InvalidGlmPrompt else parsed.value;
    if (ids_value != .array or ids_value.array.items.len < count) return error.InvalidGlmPrompt;
    const ids = try a.alloc(u32, count);
    defer a.free(ids);
    for (ids, ids_value.array.items[0..count]) |*id, value| {
        if (value != .integer or value.integer < 0 or value.integer >= cfg.vocab_size) return error.InvalidGlmPrompt;
        id.* = @intCast(value.integer);
    }
    try writeJson(io, a, progress, .{ .complete = false, .phase = "loading_target" });
    var weights = try native.loadWeights(io, a, model_path, stream);
    defer weights.deinit();
    var target = try forward.Model.load(a, cfg, &weights, stream);
    defer target.deinit();
    try writeJson(io, a, progress, .{ .complete = false, .phase = "loading_assistant" });
    var assistant = try adapter.loadAssistantStored(io, a, assistant_path, &target);
    defer assistant.deinit();
    const assistant_storage = try adapter.assistantStorage(&assistant);
    var context = try @import("dflash.zig").DflashCtx.init(a, &assistant, 0);
    defer context.deinit();
    var request = try forward.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = dense_prefill;
    request.prefill_async = prefill_async;
    request.prefill_sync_layers = @intCast(prefill_sync_layers);
    const warm_start = @import("io_util.zig").Stopwatch.init(io);
    for (0..warmup) |iteration| {
        try writeJson(io, a, progress, .{ .complete = false, .phase = "warmup", .iteration = iteration });
        var at: usize = 0;
        var next: u32 = 0;
        while (at < count) {
            const end = @min(count, at + chunk);
            const input = mlx.mlx_array_new_data(ids[at..end].ptr, &[_]c_int{ 1, @intCast(end - at) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            next = try adapter.prefill(&assistant, &context, &target, &request, input);
            at = end;
        }
        {
            var serial_warm = try adapter.cloneRequest(&request);
            defer serial_warm.deinit();
            var token = next;
            for (0..2) |_| {
                const input = mlx.mlx_array_new_data(&token, &[_]c_int{ 1, 1 }, 2, .uint32);
                defer _ = mlx.mlx_array_free(input);
                const logits = try target.forwardLast(&serial_warm, input, true);
                defer _ = mlx.mlx_array_free(logits);
                token = try greedy(logits, stream);
            }
        }
        const warmed = try adapter.roundTreeLayerwiseConfigured(io, &assistant, &context, &target, &request, next, nodes, 32, cfg.eosTokenSlice(), verify_mode, children);
        if (warmed.pending) |last| _ = try adapter.roundTreeLayerwiseConfigured(io, &assistant, &context, &target, &request, last, nodes, 1, cfg.eosTokenSlice(), verify_mode, children);
        const empty = try @import("dflash.zig").DflashCtx.init(a, &assistant, 0);
        context.deinit();
        context = empty;
        request.reset();
    }
    const warmup_ns = if (warmup > 0) warm_start.read() else 0;
    var timer = @import("io_util.zig").Stopwatch.init(io);
    var pending: u32 = 0;
    var cursor: usize = 0;
    var prefill_ns: u64 = 0;
    while (cursor < count) {
        const end = @min(count, cursor + chunk);
        try writeJson(io, a, progress, .{ .complete = false, .phase = "prefill", .processed = cursor });
        timer.reset();
        const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &[_]c_int{ 1, @intCast(end - cursor) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(input);
        pending = try adapter.prefill(&assistant, &context, &target, &request, input);
        prefill_ns += timer.read();
        cursor = end;
    }
    var serial = try adapter.cloneRequest(&request);
    defer serial.deinit();
    var serial_pending = pending;
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(a);
    var rounds: std.ArrayList(Round) = .empty;
    defer rounds.deinit(a);
    try mlx.check(mlx.mlx_reset_peak_memory());
    @import("glm5_dflash_qmm.zig").resetDispatchCount();
    @import("glm5_dflash_dense_rows.zig").resetDispatchCount();
    @import("glm5_dflash_mini.zig").resetStats();
    @import("glm5_dflash_ffn.zig").resetBatchCount();
    @import("glm5_dflash_ffn.zig").resetGroup2BatchCount();
    @import("glm5_router.zig").resetBatchCallCount();
    @import("glm5_kda_prework.zig").resetTreeDispatchCount();
    @import("glm5_kda_fused.zig").resetPostDispatchCount();
    @import("glm5_dflash_model.zig").resetStats();
    @import("sushi_exl3").kernels.resetLanePairCalls();
    @import("sushi_exl3").kernels.resetLanePairChainCalls();
    @import("sushi_exl3").kernels.resetDownLaneCalls();
    var component_profile: @import("glm5_dflash_profile.zig").Profile = .{};
    const profile_binding = @import("glm5_dflash_profile.zig").bind(if (profile_enabled) &component_profile else null);
    defer profile_binding.restore();
    const route_binding = @import("glm5_dflash_profile.zig").bindRoutes(if (route_capture) |*capture| capture else null);
    defer route_binding.restore();
    const block_tail_before = @import("dflash.zig").blockTailCalls();
    const commit_window_before = adapter.commitWindowCalls();
    @import("glm5_dflash_a6_hoist.zig").resetDispatchCount();
    @import("glm5_hc_collapse_simd32.zig").resetDispatchCount();
    var decode_ns: u64 = 0;
    while (generated.items.len < steps) {
        try writeJson(io, a, progress, .{ .complete = false, .phase = "tree_decode", .generated = generated.items.len, .rounds = rounds.items.len });
        try @import("glm5_dflash_profile.zig").beginRouteRound(rounds.items.len);
        timer.reset();
        const round = try adapter.roundTreeLayerwiseConfigured(io, &assistant, &context, &target, &request, pending, nodes, steps - generated.items.len, cfg.eosTokenSlice(), verify_mode, children);
        decode_ns += timer.read();
        try generated.appendSlice(a, round.tokens[0..round.count]);
        try rounds.append(a, .{ .emitted = round.count, .accepted_drafts = round.accepted_drafts, .verified_rows = round.verified_rows, .draft_ns = round.draft_ns, .verify_ns = round.verify_ns, .replay_ns = round.replay_ns, .commit_ns = round.commit_ns });
        if (round.stopped) break;
        pending = round.pending orelse return error.MissingGlmPendingToken;
    }
    const lane_pair_calls = @import("sushi_exl3").kernels.lanePairCalls();
    const lane_pair_chain_calls = @import("sushi_exl3").kernels.lanePairChainCalls();
    const down_lane_calls = @import("sushi_exl3").kernels.downLaneCalls();
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    try writeJson(io, a, progress, .{ .complete = false, .phase = "serial_parity" });
    const reference = try a.alloc(u32, generated.items.len);
    defer a.free(reference);
    var serial_decode_ns: u64 = 0;
    for (reference) |*id| {
        timer.reset();
        id.* = serial_pending;
        {
            const input = mlx.mlx_array_new_data(&serial_pending, &[_]c_int{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const logits = try target.forwardLast(&serial, input, true);
            defer _ = mlx.mlx_array_free(logits);
            serial_pending = try greedy(logits, stream);
        }
        serial_decode_ns += timer.read();
    }
    const ids_match = std.mem.eql(u32, reference, generated.items);
    const state_match = try stateMatches(&request, &serial, stream);
    const affine_dispatches = @import("glm5_dflash_qmm.zig").dispatchCount();
    const ffn_batches = @import("glm5_dflash_ffn.zig").batchCount();
    const engaged = (!affine_rows or affine_dispatches > 0) and (!batch_ffn or ffn_batches > 0);
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, model_path);
    defer tok.deinit();
    const text = try tok.decode(a, generated.items, false);
    defer a.free(text);
    try writeJson(io, a, output_path, .{ .complete = ids_match and state_match and engaged, .hc_collapse_simd32 = @import("glm5_hc_collapse_simd32.zig").enabled(), .hc_collapse_simd32_calls = @import("glm5_hc_collapse_simd32.zig").dispatchCount(), .model = model_path, .assistant = assistant_path, .assistant_precision = assistant_storage.label(), .assistant_storage = assistant_storage, .kv = "BF16 compressed MLA cache; FP32 KDA state", .lane_pair_calls = lane_pair_calls, .lane_pair_chain_calls = lane_pair_chain_calls, .down_lane_calls = down_lane_calls, .lane_pair_counter_scope = "speculative rounds only; excludes warmup and serial reference", .verifier = if (batch_ffn) "layerwise_tree_affine_ffn_tiles" else if (affine_rows) "layerwise_tree_affine_row_tiles" else "layerwise_tree_serial_projections", .affine_row_tiles = affine_rows, .affine_row_dispatches = affine_dispatches, .dense_row_dispatches = @import("glm5_dflash_dense_rows.zig").dispatchCount(), .mini_head_bits = assistant.draft_head_bits, .mini_head_group = assistant.draft_head_group, .mini_head_bytes = if (assistant.draft_head != null) @import("glm5_dflash_mini.zig").residentBytes(target.head.output, target.head.input) else @as(u64, 0), .mini_head_projection_calls = @import("glm5_dflash_mini.zig").projectionCount(), .ffn_batches = ffn_batches, .group2_batches = @import("glm5_dflash_ffn.zig").group2BatchCount(), .block_tail_calls = @import("dflash.zig").blockTailCalls() - block_tail_before, .commit_window_calls = adapter.commitWindowCalls() - commit_window_before, .affine6_hoist_calls = @import("glm5_dflash_a6_hoist.zig").dispatchCount(), .kda_prefill_cluster_calls = @import("glm5_kda_prefill_cluster.zig").dispatchCount(), .kda_prefill_cluster_bytes = target.kdaPrefillClusterBytes(), .packed_cadence_calls = @import("glm5_attention.zig").packedCadenceCalls(), .prefill_grid_calls = @import("sushi_exl3").glm_prefill_grid.dispatchCount(), .decode_native_b1_calls = @import("glm5_attention_decode_batch.zig").b1Calls(), .decode_native_b3_calls = @import("glm5_attention_decode_batch.zig").b3Calls(), .router_batch_calls = @import("glm5_router.zig").batchCallCount(), .tree_kda_prework_dispatches = @import("glm5_kda_prework.zig").treeDispatchCount(), .tree_kda_post_dispatches = @import("glm5_kda_fused.zig").postDispatchCount(), .batch_ffn = batch_ffn, .mla_branch_flushes = @import("glm5_dflash_model.zig").branchFlushCount(), .mla_scratch_bound_bytes = @import("glm5_dflash_model.zig").scratchBoundBytes(), .mla_scratch_cap_bytes = @import("glm5_dflash_memory.zig").limit_bytes, .sampling = "greedy", .warmup = warmup, .warmup_ns = warmup_ns, .timing_provisional = true, .synchronization_perturbed = synchronization_perturbed, .throughput_comparable = !synchronization_perturbed, .route_capture = if (route_capture) |*capture| .{ .scope = "eligible multirow routed FFNs only; token-major top-k order", .capacity = capture.records.len, .record_count = capture.count, .skipped_single_calls = capture.skipped_single_calls, .complete = true, .records = capture.records[0..capture.count] } else null, .component_profile = if (profile_enabled) .{ .units = "nanoseconds", .method = "host construction plus forced evaluation; synchronization-perturbed", .ffn_overall_includes_children = true, .sparse_indexer_billed_to = "mla_branches", .layers = component_profile.layers[0..target.layers.len], .global = component_profile.global } else null, .prefill_tokens = count, .requested_output_tokens = steps, .generated_tokens = generated.items.len, .prefill_tokens_per_second = @as(f64, @floatFromInt(count)) * 1e9 / @as(f64, @floatFromInt(prefill_ns)), .input_ids = ids, .output_ids = generated.items, .serial_output_ids = reference, .output_text = if (std.unicode.utf8ValidateSlice(text)) text else null, .token_parity = ids_match, .state_parity = state_match, .prefill_ns = prefill_ns, .decode_ns = decode_ns, .rate_denominator = "committed_input_tokens_including_final_emitted_token", .serial_reference_ns = serial_decode_ns, .serial_reference_tokens_per_second = @as(f64, @floatFromInt(reference.len)) * 1e9 / @as(f64, @floatFromInt(serial_decode_ns)), .speedup_vs_matched_serial = if (!synchronization_perturbed and ids_match and state_match) @as(f64, @floatFromInt(serial_decode_ns)) / @as(f64, @floatFromInt(decode_ns)) else @as(?f64, null), .serial_reference_prefill = "same captured prefix and target state", .serial_reference_decode_capture = false, .serial_reference_async4 = serial.decode_async, .dense_prefill = request.dense_prefill, .prefill_async = request.prefill_async, .prefill_sync_layers = request.prefill_sync_layers, .prefill_chunk = chunk, .warmup_serial_steps = 2 * warmup, .decode_tokens_per_second = @as(f64, @floatFromInt(generated.items.len)) * 1e9 / @as(f64, @floatFromInt(decode_ns)), .peak_bytes = peak, .peak_scope = "decode_after_prefill_reset", .memory_limit_bytes = memory, .wired_limit_bytes = wired, .nodes = nodes, .children = children, .verify_async_layers = if (profile_enabled) 0 else async_layers, .verify_async_dispatches = @import("glm5_dflash_model.zig").asyncDispatchCount(), .verify_sync_dispatches = @import("glm5_dflash_model.zig").syncDispatchCount(), .rounds = rounds.items, .public_serving_enabled = false });
    if (!ids_match or !state_match) return error.GlmDflashSerialParityFailed;
    if (!engaged) return error.GlmAffineRowsNotEngaged;
    try writeJson(io, a, progress, .{ .complete = true, .phase = "complete" });
}
