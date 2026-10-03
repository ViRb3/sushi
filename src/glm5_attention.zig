//! Request-local IndexPool and NoPE latent attention. All caches are lossless.
//! GLM production policy is BF16 compressed MLA cache; do not inherit generic KV8
//! defaults when integrating this state into serving. KDA separately keeps FP32 state.
const std = @import("std");
const mlx = @import("mlx.zig");
const prefill_direct = @import("glm5_attention_prefill.zig");
const packed_nax = @import("glm5_attention_nax_packed.zig");
const latent_overlay = @import("glm5_attention_overlay.zig");
const Arr = mlx.mlx_array;
const Ops = @import("glm5_model.zig").Ops;
const nil = Arr{ .ctx = null };
const pool_size = 4;
const pool_budget = 512;
const selected_width = pool_size * pool_budget + pool_size - 1;
pub const score_scratch_bytes: usize = 2 * 1024 * 1024;
pub const attention_scratch_bytes: usize = 8 * 1024 * 1024;
threadlocal var packed_cadence: ?bool = null;
var cadence_calls: usize = 0;
pub const CadenceBinding = struct {
    previous: ?bool,
    pub fn restore(self: CadenceBinding) void {
        packed_cadence = self.previous;
    }
};
pub fn bindPackedCadence(on: bool) CadenceBinding {
    const binding = CadenceBinding{ .previous = packed_cadence };
    packed_cadence = on;
    return binding;
}
pub fn packedCadenceEnabled() bool {
    return packed_cadence orelse @import("transformer.zig").diagEnvOn("SUSHI_GLM_PREFILL_CADENCE");
}
pub fn packedCadenceCalls() usize {
    return cadence_calls;
}
pub fn resetPackedCadenceCalls() void {
    cadence_calls = 0;
}
pub fn packedCadenceTransientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!packed_nax.enabled() or !packedCadenceEnabled() or chunk <= packed_nax.max_rows) return 0;
    return std.math.mul(usize, packed_nax.scratch_limit, pending_layers);
}

threadlocal var captured_cadence: bool = false;
fn captureCadence(state: *const State, q: Arr, iq: Arr, weights: Arr, offset: usize, scale: f32) !void {
    if (captured_cadence or mlx.getShape(q)[0] != 2048) return;
    const path = std.c.getenv("SUSHI_GLM_PREFILL_CADENCE_CAPTURE") orelse return;
    const history_target = if (std.c.getenv("SUSHI_GLM_PREFILL_CADENCE_CAPTURE_HISTORY")) |raw|
        try std.fmt.parseInt(usize, std.mem.span(raw), 10)
    else
        8192;
    if (history_target != 8192 and history_target != 16384) return error.InvalidCadenceCaptureHistory;
    if (state.processed != history_target) return;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const off: u32 = @intCast(offset);
    const history: u32 = @intCast(state.processed);
    const values = [_]Arr{ q, iq, weights, state.latent, state.pooled, try ops.own(mlx.mlx_array_new_data(&off, &.{}, 0, .uint32)), try ops.own(mlx.mlx_array_new_data(&history, &.{}, 0, .uint32)), try ops.own(mlx.mlx_array_new_data(&scale, &.{}, 0, .float32)) };
    const ev = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    const arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    const metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    for ([_][*:0]const u8{ "q", "index_q", "weights", "latent", "pooled", "offset", "processed", "scale" }, values) |name, value|
        try mlx.check(mlx.mlx_map_string_to_array_insert(arrays, name, value));
    try mlx.check(mlx.mlx_map_string_to_string_insert(metadata, "provenance", "first real MLA T2048 at selected history; capture forces evaluation; not a throughput run"));
    try mlx.check(mlx.mlx_save_safetensors(path, arrays, metadata));
    captured_cadence = true;
}

const Scope = struct {
    s: mlx.mlx_stream,
    values: [96]Arr = undefined,
    count: usize = 0,
    fn deinit(self: *Scope) void {
        for (self.values[0..self.count]) |a| _ = mlx.mlx_array_free(a);
    }
    fn slot(self: *Scope) !*Arr {
        if (self.count == self.values.len) return error.AttentionGraphTooLarge;
        self.values[self.count] = mlx.mlx_array_new();
        self.count += 1;
        return &self.values[self.count - 1];
    }
    fn own(self: *Scope, a: Arr) !Arr {
        const out = self.slot() catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        _ = mlx.mlx_array_free(out.*);
        out.* = a;
        return a;
    }
    fn result(_: *Scope, a: Arr) !Arr {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_set(&out, a));
        return out;
    }
    fn cast(self: *Scope, a: Arr, dtype: mlx.mlx_dtype) !Arr {
        if (mlx.mlx_array_dtype(a) == dtype) return a;
        const out = try self.slot();
        try mlx.check(mlx.mlx_astype(out, a, dtype, self.s));
        return out.*;
    }
    fn shape(self: *Scope, a: Arr, dims: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_reshape(out, a, dims.ptr, dims.len, self.s));
        return out.*;
    }
    fn cut(self: *Scope, a: Arr, start: c_int, end: c_int) !Arr {
        const sh = mlx.getShape(a);
        var lo: [4]c_int = @splat(0);
        var hi: [4]c_int = @splat(0);
        const step: [4]c_int = @splat(1);
        if (sh.len > 4 or start < 0 or end < start or end > sh[0]) return error.InvalidGlmAttentionShape;
        @memcpy(hi[0..sh.len], sh);
        lo[0] = start;
        hi[0] = end;
        const out = try self.slot();
        try mlx.check(mlx.mlx_slice(out, a, &lo, sh.len, &hi, sh.len, &step, sh.len, self.s));
        return out.*;
    }
    fn join(self: *Scope, a: Arr, b: Arr) !Arr {
        if (a.ctx == null or mlx.getShape(a)[0] == 0) return b;
        const parts = [_]Arr{ a, b };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        const out = try self.slot();
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, 0, self.s));
        return out.*;
    }
    fn zeros(self: *Scope, dims: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_zeros(out, dims.ptr, dims.len, dtype, self.s));
        return out.*;
    }
    fn copy(self: *Scope, a: Arr) !Arr {
        const zero = try self.zeros(&.{}, mlx.mlx_array_dtype(a));
        const out = try self.slot();
        try mlx.check(mlx.mlx_add(out, a, zero, self.s));
        return out.*;
    }
    fn appendRows(self: *Scope, buffer: Arr, used: usize, chunk: Arr) !Arr {
        const sh = mlx.getShape(chunk);
        const needed = try std.math.add(usize, used, @intCast(sh[0]));
        if (needed > std.math.maxInt(c_int) - 255) return error.InvalidGlmAttentionShape;
        const current: usize = if (buffer.ctx == null) 0 else @intCast(mlx.getShape(buffer)[0]);
        var target = buffer;
        if (needed > current) {
            const capacity = (needed + 255) / 256 * 256;
            const padding = try self.zeros(&.{ @intCast(capacity - current), sh[1] }, mlx.mlx_array_dtype(chunk));
            target = try self.join(buffer, padding);
        }
        const out = try self.slot();
        try mlx.check(mlx.mlx_slice_update(out, target, chunk, &.{ @intCast(used), 0 }, 2, &.{ @intCast(needed), sh[1] }, 2, &.{ 1, 1 }, 2, self.s));
        return out.*;
    }
};

fn supported(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float32;
}

fn compress(scope: *Scope, keys: Arr, gates: Arr, ape: Arr, ready: c_int) !Arr {
    const width = mlx.getShape(keys)[1];
    const k = try scope.shape(try scope.cut(keys, 0, ready), &.{ @divExact(ready, 4), 4, width });
    const g = try scope.shape(try scope.cut(gates, 0, ready), &.{ @divExact(ready, 4), 4, width });
    const logits = try scope.slot();
    try mlx.check(mlx.mlx_add(logits, g, ape, scope.s));
    const probs = try scope.slot();
    try mlx.check(mlx.mlx_softmax_axis(probs, logits.*, 1, false, scope.s));
    const weighted = try scope.slot();
    try mlx.check(mlx.mlx_multiply(weighted, probs.*, k, scope.s));
    const pooled = try scope.slot();
    try mlx.check(mlx.mlx_sum_axis(pooled, weighted.*, 1, false, scope.s));
    return scope.cast(pooled.*, mlx.mlx_array_dtype(keys));
}

pub const State = struct {
    latent: Arr = nil,
    pooled: Arr = nil,
    tail_keys: Arr = nil,
    tail_gates: Arr = nil,
    processed: usize = 0,

    pub fn init() State {
        return .{};
    }
    pub fn deinit(self: *State) void {
        for (self.arrays()) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        self.* = .{};
    }
    pub fn reset(self: *State) void {
        self.deinit();
    }
    pub fn arrays(self: *const State) [4]Arr {
        return .{ self.latent, self.pooled, self.tail_keys, self.tail_gates };
    }
    pub fn evaluate(self: *const State) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (self.arrays()) |a| if (a.ctx != null) try mlx.check(mlx.mlx_vector_array_append_value(vec, a));
        try mlx.check(mlx.mlx_eval(vec));
    }
    pub fn append(self: *State, latent: Arr, keys: Arr, gates: Arr, ape: Arr, s: mlx.mlx_stream) !usize {
        return self.appendImpl(latent, keys, gates, ape, s, true);
    }
    /// Verification forks retain their latent prefix; attention reads the supplied overlay.
    pub fn appendIndexOnly(self: *State, latent: Arr, keys: Arr, gates: Arr, ape: Arr, s: mlx.mlx_stream) !usize {
        return self.appendImpl(latent, keys, gates, ape, s, false);
    }
    fn appendImpl(self: *State, latent: Arr, keys: Arr, gates: Arr, ape: Arr, s: mlx.mlx_stream, store_latent: bool) !usize {
        if (latent.ctx == null or keys.ctx == null or gates.ctx == null or ape.ctx == null) return error.InvalidGlmAttentionShape;
        const ls = mlx.getShape(latent);
        const ks = mlx.getShape(keys);
        if (ls.len != 2 or ks.len != 2 or ls[0] <= 0 or ls[1] <= 0 or ks[1] <= 0 or ls[0] != ks[0] or
            !std.mem.eql(c_int, ks, mlx.getShape(gates)) or !std.mem.eql(c_int, &.{ 4, ks[1] }, mlx.getShape(ape)) or
            !supported(mlx.mlx_array_dtype(latent)) or !supported(mlx.mlx_array_dtype(keys)) or
            mlx.mlx_array_dtype(gates) != mlx.mlx_array_dtype(keys) or mlx.mlx_array_dtype(ape) != mlx.mlx_array_dtype(keys)) return error.InvalidGlmAttentionShape;
        if (self.processed != 0 and (mlx.getShape(self.latent)[1] != ls[1] or mlx.mlx_array_dtype(self.latent) != mlx.mlx_array_dtype(latent))) return error.InvalidGlmAttentionShape;
        if (self.tail_keys.ctx != null and (mlx.getShape(self.tail_keys)[1] != ks[1] or mlx.mlx_array_dtype(self.tail_keys) != mlx.mlx_array_dtype(keys))) return error.InvalidGlmAttentionShape;
        if (self.pooled.ctx != null and (mlx.getShape(self.pooled)[1] != ks[1] or mlx.mlx_array_dtype(self.pooled) != mlx.mlx_array_dtype(keys))) return error.InvalidGlmAttentionShape;
        const next = try std.math.add(usize, self.processed, @intCast(ls[0]));
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const l = if (store_latent) try scope.appendRows(self.latent, self.processed, latent) else self.latent;
        const k = try scope.join(self.tail_keys, keys);
        const g = try scope.join(self.tail_gates, gates);
        const ready = @divTrunc(mlx.getShape(k)[0], 4) * 4;
        const p = if (ready > 0) try scope.appendRows(self.pooled, self.processed / 4, try compress(&scope, k, g, ape, ready)) else self.pooled;
        const tail_k = try scope.copy(try scope.cut(k, ready, mlx.getShape(k)[0]));
        const tail_g = try scope.copy(try scope.cut(g, ready, mlx.getShape(g)[0]));
        var new = State{ .processed = next };
        errdefer new.deinit();
        if (l.ctx != null) new.latent = try scope.result(l);
        if (p.ctx != null) new.pooled = try scope.result(p);
        new.tail_keys = try scope.result(tail_k);
        new.tail_gates = try scope.result(tail_g);
        const before = self.processed;
        self.deinit();
        self.* = new;
        return before;
    }
};

var score_kernel: ?mlx.mlx_fast_metal_kernel = null;
var expand_kernel: ?mlx.mlx_fast_metal_kernel = null;
var attention_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}
fn apply(k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) !mlx.mlx_vector_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    return ov;
}
fn kernelOutput(scope: *Scope, ov: mlx.mlx_vector_array, i: usize) !Arr {
    const out = try scope.slot();
    try mlx.check(mlx.mlx_vector_array_get(out, ov, i));
    return out.*;
}
fn template(cfg: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
}

const SCORE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane=thread_position_in_threadgroup.x;
    \\const uint p=threadgroup_position_in_grid.y;
    \\const uint row=threadgroup_position_in_grid.z;
    \\const uint pools=uint(count);
    \\if ((p+1u)*4u > uint(offset)+row+1u) { if(lane==0) out[row*pools+p]=-INFINITY; return; }
    \\float total=0.0f;
    \\for(uint h=0;h<uint(J);++h) {
    \\  float dot=0.0f;
    \\  for(uint d=lane;d<uint(I);d+=32u) dot+=float(q[(row*uint(J)+h)*uint(I)+d])*float(keys[p*uint(I)+d]);
    \\  dot=simd_sum(dot);
    \\  const float rounded=float(InT(dot));
    \\  total+=float(InT(max(rounded,0.0f)*float(weights[row*uint(J)+h])));
    \\}
    \\if(lane==0) out[row*pools+p]=float(InT(total));
;
const EXPAND: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*2051u) return;
    \\const uint row=i/2051u;
    \\const uint col=i%2051u;
    \\const uint pos=uint(offset)+row;
    \\int token=-1;
    \\if(col<uint(K)*4u) {
    \\ const int p=int(selected[row*uint(K)+col/4u]);
    \\ if(p>=0 && (uint(p)+1u)*4u<=pos+1u) token=p*4+int(col%4u);
    \\} else if(col>=2048u) {
    \\ const uint rem=(pos+1u)%4u;
    \\ const uint j=col-2048u;
    \\ if(j<rem) token=int(pos+1u-rem+j);
    \\}
    \\out[i]=token;
;
const ATTENTION: [:0]const u8 = prefill_direct.common ++ prefill_direct.partial_tail;
const MERGE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*uint(H)*uint(D)) return;
    \\const uint rowhead=i/uint(D),d=i%uint(D);
    \\float maximum=-INFINITY;
    \\for(uint p=0;p<uint(SPLITS);++p) maximum=max(maximum,stats[(rowhead*uint(SPLITS)+p)*2u]);
    \\float sum=0.0f,denom=0.0f;
    \\for(uint p=0;p<uint(SPLITS);++p) {
    \\ const uint base=rowhead*uint(SPLITS)+p;
    \\ const float n=stats[base*2u+1u];
    \\ if(n>0.0f) {const float w=precise::exp(stats[base*2u]-maximum);sum+=w*partial[base*uint(D)+d];denom+=w*n;}
    \\}
    \\out[i]=OutT(denom>0.0f?sum/denom:0.0f);
;

fn indexScores(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const sh = mlx.getShape(index_q);
    const rows = sh[0];
    const pools: c_int = @intCast(state.processed / 4);
    if (try @import("glm5_indexpool_nax.zig").tryScores(index_q, state.pooled, weights, offset, @intCast(pools), scope.s)) |out|
        return scope.own(out);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, pools }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, pools, rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try template(cfg, "J", sh[1]);
    try template(cfg, "I", sh[2]);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx.mlx_array_dtype(index_q)));
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const count = try scope.own(mlx.mlx_array_new_int(pools));
    const k = try kernel(&score_kernel, "sushi_glm_index_scores", &.{ "q", "keys", "weights", "offset", "count" }, &.{"out"}, SCORE);
    const ov = try apply(k, &.{ index_q, state.pooled, weights, off, count }, cfg, scope.s);
    defer _ = mlx.mlx_vector_array_free(ov);
    return kernelOutput(scope, ov, 0);
}

fn selectChunk(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const rows = mlx.getShape(index_q)[0];
    const pools: c_int = @intCast(state.processed / 4);
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const scores = try indexScores(scope, state, index_q, weights, offset);
    const neg = try scope.slot();
    try mlx.check(mlx.mlx_negative(neg, scores, scope.s));
    const partition = try scope.slot();
    try mlx.check(mlx.mlx_argpartition_axis(partition, neg.*, @min(pools, pool_budget) - 1, -1, scope.s));
    const selected = try scope.slot();
    try mlx.check(mlx.mlx_slice(selected, partition.*, &.{ 0, 0 }, 2, &.{ rows, @min(pools, pool_budget) }, 2, &.{ 1, 1 }, 2, scope.s));
    const ec = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(ec);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(ec, &.{ rows, selected_width }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(ec, rows * selected_width, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(ec, 256, 1, 1));
    try template(ec, "ROWS", rows);
    try template(ec, "K", @min(pools, pool_budget));
    const ek = try kernel(&expand_kernel, "sushi_glm_expand_pools", &.{ "selected", "offset" }, &.{"out"}, EXPAND);
    const ev = try apply(ek, &.{ selected.*, off }, ec, scope.s);
    defer _ = mlx.mlx_vector_array_free(ev);
    return kernelOutput(scope, ev, 0);
}

fn attentionChunk(scope: *Scope, state: *const State, q: Arr, selected: ?Arr, offset: usize, scale: f32, splits: c_int, direct: bool, headpack: bool, overlay: ?latent_overlay.View) !Arr {
    if (headpack) {
        var ops = @import("glm5_model.zig").Ops{ .s = scope.s };
        defer ops.deinit();
        if (try packed_nax.run(&ops, q, state.latent, selected.?, offset, state.processed, scale)) |out| {
            // Materialize the small result before releasing the gathered bank.
            try mlx.check(mlx.mlx_array_eval(out));
            return scope.own(try ops.result(out));
        }
    }
    const sh = mlx.getShape(q);
    const rows = sh[0];
    const heads = sh[1];
    const dim = sh[2];
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const length = try scope.own(mlx.mlx_array_new_int(@intCast(state.processed)));
    const scaling = try scope.own(mlx.mlx_array_new_float(scale));
    const indices = selected orelse try scope.zeros(&.{1}, .int32);
    if (direct) return scope.own(try prefill_direct.attend(q, state.latent, indices, off, length, scaling, selected != null, scope.s));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, heads, splits, dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, heads, splits, 2 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, heads, rows * splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try template(cfg, "H", heads);
    try template(cfg, "D", dim);
    try template(cfg, "SPLITS", splits);
    try template(cfg, "SELECTED", @intFromBool(selected != null));
    var partial: Arr = undefined;
    var stats: Arr = undefined;
    if (overlay) |view| {
        const result = try latent_overlay.partials(q, view, indices, off, length, scaling, splits, selected != null, scope.s);
        defer result.deinit();
        partial = try scope.own(try scope.result(result.partial));
        stats = try scope.own(try scope.result(result.stats));
    } else {
        const k = try kernel(&attention_kernel, "sushi_glm_latent_partial", &.{ "q", "cache", "selected", "offset", "length", "scale" }, &.{ "partial", "stats" }, ATTENTION);
        const ov = try apply(k, &.{ q, state.latent, indices, off, length, scaling }, cfg, scope.s);
        defer _ = mlx.mlx_vector_array_free(ov);
        partial = try kernelOutput(scope, ov, 0);
        stats = try kernelOutput(scope, ov, 1);
    }
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, sh.ptr, 3, mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, rows * heads * dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, 256, 1, 1));
    try template(mc, "ROWS", rows);
    try template(mc, "H", heads);
    try template(mc, "D", dim);
    try template(mc, "SPLITS", splits);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OutT", mlx.mlx_array_dtype(q)));
    const mk = try kernel(&merge_kernel, "sushi_glm_latent_merge", &.{ "partial", "stats" }, &.{"out"}, MERGE);
    const mv = try apply(mk, &.{ partial, stats }, mc, scope.s);
    defer _ = mlx.mlx_vector_array_free(mv);
    return kernelOutput(scope, mv, 0);
}

pub fn attend(state: *const State, q: Arr, index_q: ?Arr, weights: ?Arr, offset: usize, scale: f32, s: mlx.mlx_stream) !Arr {
    return attendImpl(state, q, index_q, weights, offset, scale, null, s);
}
pub fn attendOverlay(state: *const State, q: Arr, index_q: ?Arr, weights: ?Arr, offset: usize, scale: f32, view: latent_overlay.View, s: mlx.mlx_stream) !Arr {
    try view.validate(q);
    if (state.processed != view.length() or offset + 1 != state.processed) return error.InvalidGlmOverlay;
    return attendImpl(state, q, index_q, weights, offset, scale, view, s);
}

const PackedTile = struct {
    scope: Scope,
    ops: Ops,
    out: Arr = nil,
    pending: bool = false,
    fn deinit(self: *PackedTile) void {
        // Error paths must also settle submitted work before releasing its bank.
        if (self.pending) _ = mlx.mlx_array_eval(self.out);
        self.ops.deinit();
        self.scope.deinit();
    }
};

fn attendPackedPairs(state: *const State, q: Arr, iq: Arr, weights: Arr, offset: usize, scale: f32, max_rows: usize, direct: bool, s: mlx.mlx_stream) !Arr {
    const rows: usize = @intCast(mlx.getShape(q)[0]);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var start: usize = 0;
    while (start < rows) {
        var tiles = [_]PackedTile{
            .{ .scope = .{ .s = s }, .ops = .{ .s = s } },
            .{ .scope = .{ .s = s }, .ops = .{ .s = s } },
        };
        defer for (&tiles) |*tile| tile.deinit();
        const outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        var count: usize = 0;
        for (&tiles) |*tile| {
            if (start == rows) break;
            const end = @min(rows, start + max_rows);
            const qc = try tile.scope.cut(q, @intCast(start), @intCast(end));
            const selected = try selectChunk(&tile.scope, state, try tile.scope.cut(iq, @intCast(start), @intCast(end)), try tile.scope.cut(weights, @intCast(start), @intCast(end)), offset + start);
            tile.out = (try packed_nax.run(&tile.ops, qc, state.latent, selected, offset + start, state.processed, scale)) orelse
                try attentionChunk(&tile.scope, state, qc, selected, offset + start, scale, 1, direct, false, null);
            const submit = mlx.mlx_vector_array_new_data(&.{tile.out}, 1);
            defer _ = mlx.mlx_vector_array_free(submit);
            tile.pending = true;
            try mlx.check(mlx.mlx_async_eval(submit));
            try mlx.check(mlx.mlx_vector_array_append_value(outputs, tile.out));
            count += 1;
            start = end;
        }
        try mlx.check(mlx.mlx_eval(outputs));
        for (tiles[0..count]) |*tile| {
            tile.pending = false;
            try mlx.check(mlx.mlx_vector_array_append_value(parts, tile.out));
        }
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 0, s));
    return out;
}
fn attendImpl(state: *const State, q: Arr, index_q: ?Arr, weights: ?Arr, offset: usize, scale: f32, overlay: ?latent_overlay.View, s: mlx.mlx_stream) !Arr {
    if (!mlx.streamIsGpu(s)) return error.GlmAttentionGpuRequired;
    if (q.ctx == null) return error.InvalidGlmAttentionShape;
    const sh = mlx.getShape(q);
    const cache = if (overlay) |view| view.storage() else state.latent;
    if (sh.len != 3 or sh[0] <= 0 or sh[1] <= 0 or sh[2] <= 0 or !supported(mlx.mlx_array_dtype(q)) or
        !std.math.isFinite(scale) or scale <= 0 or cache.ctx == null or
        sh[2] != mlx.getShape(cache)[1] or mlx.mlx_array_dtype(q) != mlx.mlx_array_dtype(cache) or
        offset > state.processed or @as(usize, @intCast(sh[0])) > state.processed - offset) return error.InvalidGlmAttentionShape;
    const sparse = offset + @as(usize, @intCast(sh[0])) > pool_size * (pool_budget + 1) - 1;
    if (sparse) {
        if (state.pooled.ctx == null) return error.InvalidGlmAttentionShape;
        const iq = index_q orelse return error.MissingGlmIndexQuery;
        const w = weights orelse return error.MissingGlmIndexQuery;
        const is = mlx.getShape(iq);
        if (is.len != 3 or is[0] != sh[0] or is[1] <= 0 or is[2] != mlx.getShape(state.pooled)[1] or
            !std.mem.eql(c_int, is[0..2], mlx.getShape(w)) or mlx.mlx_array_dtype(iq) != mlx.mlx_array_dtype(state.pooled) or
            mlx.mlx_array_dtype(w) != mlx.mlx_array_dtype(iq)) return error.InvalidGlmAttentionShape;
    }
    const splits: c_int = if (sh[0] <= 8) 8 else 1;
    const headpack = overlay == null and sparse and splits == 1 and sh[1] == 64 and sh[2] == 512 and
        mlx.mlx_array_dtype(q) == .bfloat16 and packed_nax.enabled();
    const direct = splits == 1 and prefill_direct.enabled();
    const per_row = if (direct) try prefill_direct.rowBytes(@intCast(sh[1]), @intCast(sh[2]), mlx.mlx_array_itemsize(q)) else try std.math.mul(usize, @intCast(sh[1]), try std.math.mul(usize, @intCast(splits), (@as(usize, @intCast(sh[2])) + 2) * 4));
    const pool_bytes = @max(@as(usize, 4), state.processed / 4 * 4);
    if (per_row > attention_scratch_bytes or (sparse and pool_bytes > score_scratch_bytes)) return error.GlmAttentionScratchBudget;
    const max_rows = @max(@as(usize, 1), @min(@min(attention_scratch_bytes / per_row, if (sparse) score_scratch_bytes / pool_bytes else std.math.maxInt(usize)), if (headpack) packed_nax.max_rows else 128));
    if (headpack) {
        try captureCadence(state, q, index_q.?, weights.?, offset, scale);
        if (packedCadenceEnabled() and @as(usize, @intCast(sh[0])) > max_rows) {
            const out = try attendPackedPairs(state, q, index_q.?, weights.?, offset, scale, max_rows, direct, s);
            cadence_calls += 1;
            return out;
        }
    }
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var start: usize = 0;
    while (start < sh[0]) {
        const end = @min(@as(usize, @intCast(sh[0])), start + max_rows);
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const qc = try scope.cut(q, @intCast(start), @intCast(end));
        const selected = if (sparse) try selectChunk(&scope, state, try scope.cut(index_q.?, @intCast(start), @intCast(end)), try scope.cut(weights.?, @intCast(start), @intCast(end)), offset + start) else null;
        const out = try attentionChunk(&scope, state, qc, selected, offset + start, scale, splits, direct, headpack, overlay);
        // Settle bounded chunks before dropping their score/partial buffers.
        if (@as(usize, @intCast(sh[0])) > max_rows) try mlx.check(mlx.mlx_array_eval(out));
        try mlx.check(mlx.mlx_vector_array_append_value(parts, out));
        start = end;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 0, s));
    return out;
}

fn array(data: []const f32, shape: []const c_int) Arr {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

test "GLM IndexPool append matches scalar pooling across irregular chunks" {
    const s = mlx.gpuStream();
    const n = 9;
    const d = 4;
    var keys: [n * d]f32 = undefined;
    var gates: [n * d]f32 = undefined;
    var bias: [4 * d]f32 = undefined;
    for (&keys, &gates, 0..) |*k, *g, i| {
        k.* = @as(f32, @floatFromInt(i)) / 16 - 1;
        g.* = @as(f32, @floatFromInt(i % 7)) / 8 - 0.5;
    }
    for (&bias, 0..) |*b, i| b.* = @as(f32, @floatFromInt(i % 5)) / 16;
    const ape = array(&bias, &.{ 4, d });
    defer _ = mlx.mlx_array_free(ape);
    var state = State.init();
    defer state.deinit();
    var offset: usize = 0;
    for ([_]usize{ 1, 3, 1, 4 }) |rows| {
        const k = array(keys[offset * d ..][0 .. rows * d], &.{ @intCast(rows), d });
        defer _ = mlx.mlx_array_free(k);
        const g = array(gates[offset * d ..][0 .. rows * d], &.{ @intCast(rows), d });
        defer _ = mlx.mlx_array_free(g);
        try std.testing.expectEqual(offset, try state.append(k, k, g, ape, s));
        offset += rows;
        try std.testing.expectEqual(offset, state.processed);
        try state.evaluate();
        if (offset >= 4) {
            const got = mlx.mlx_array_data_float32(state.pooled).?;
            for (0..offset / 4) |p| for (0..d) |c| {
                var sum: f64 = 0;
                var value: f64 = 0;
                for (0..4) |j| {
                    const e = @exp(@as(f64, gates[(p * 4 + j) * d + c] + bias[j * d + c]));
                    sum += e;
                    value += e * keys[(p * 4 + j) * d + c];
                }
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(value / sum)), got[p * d + c], 1e-6);
            };
        }
        try std.testing.expectEqual(@as(usize, offset % 4 * d), if (state.tail_keys.ctx != null) mlx.mlx_array_size(state.tail_keys) else 0);
    }
    state.reset();
    try std.testing.expectEqual(@as(usize, 0), state.processed);
    for (state.arrays()) |a| try std.testing.expect(a.ctx == null);
}

test "GLM latent attention matches scalar causal attention at pool boundaries" {
    const s = mlx.gpuStream();
    for ([_]usize{ 1, 3, 4, 5, 2048, 2049 }) |n| {
        const a = std.testing.allocator;
        const d = 4;
        const keys = try a.alloc(f32, n * d);
        defer a.free(keys);
        for (keys, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) / 32 - 0.25;
        const latent = array(keys, &.{ @intCast(n), d });
        defer _ = mlx.mlx_array_free(latent);
        const bias: [16]f32 = @splat(0);
        const ape = array(&bias, &.{ 4, d });
        defer _ = mlx.mlx_array_free(ape);
        var state = State.init();
        defer state.deinit();
        _ = try state.append(latent, latent, latent, ape, s);
        const qv = [_]f32{ 0.5, -0.25, 0.75, 0.125 };
        const q = array(&qv, &.{ 1, 1, d });
        defer _ = mlx.mlx_array_free(q);
        const output = try attend(&state, q, null, null, n - 1, 1.0 / 16.0, s);
        defer _ = mlx.mlx_array_free(output);
        try mlx.check(mlx.mlx_array_eval(output));
        const got = mlx.mlx_array_data_float32(output).?;
        var denominator: f64 = 0;
        var want: [d]f64 = @splat(0);
        for (0..n) |i| {
            var dot: f64 = 0;
            for (qv, keys[i * d ..][0..d]) |x, k| dot += @as(f64, x) * k / 16;
            const e = @exp(dot);
            denominator += e;
            for (&want, keys[i * d ..][0..d]) |*v, k| v.* += e * k;
        }
        for (want, 0..) |v, c| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(v / denominator)), got[c], 2e-6);
    }
}

test "GLM sparse pool selection excludes future pools and appends tail exactly once" {
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    const n = 2055;
    const d = 4;
    const heads = 2;
    const rows = 8;
    const offset = n - rows;
    const lat = try a.alloc(f32, n * d);
    defer a.free(lat);
    const key = try a.alloc(f32, n * d);
    defer a.free(key);
    const zeros = try a.alloc(f32, n * d);
    defer a.free(zeros);
    @memset(zeros, 0);
    for (lat, key, 0..) |*v, *k, i| {
        v.* = @as(f32, @floatFromInt(i % 23)) / 32 - 0.25;
        k.* = if (i % d == 0) @as(f32, @floatFromInt(i / d)) / 2048 else 0;
    }
    const la = array(lat, &.{ n, d });
    defer _ = mlx.mlx_array_free(la);
    const ka = array(key, &.{ n, d });
    defer _ = mlx.mlx_array_free(ka);
    const ga = array(zeros, &.{ n, d });
    defer _ = mlx.mlx_array_free(ga);
    const av: [16]f32 = @splat(0);
    const ape = array(&av, &.{ 4, d });
    defer _ = mlx.mlx_array_free(ape);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(la, ka, ga, ape, s);
    const query: [rows * heads * d]f32 = @splat(0.125);
    const q = array(&query, &.{ rows, heads, d });
    defer _ = mlx.mlx_array_free(q);
    const weight: [rows * heads]f32 = @splat(0.5);
    const w = array(&weight, &.{ rows, heads });
    defer _ = mlx.mlx_array_free(w);
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const selected = try selectChunk(&scope, &state, q, w, offset);
    try mlx.check(mlx.mlx_array_eval(selected));
    const ids = mlx.mlx_array_data_int32(selected).?;
    for (0..rows) |r| {
        const pos = offset + r;
        const completed = (pos + 1) / 4;
        const first = if (completed > 512) (completed - 512) * 4 else 0;
        var seen: [n]bool = @splat(false);
        var count: usize = 0;
        for (ids[r * selected_width ..][0..selected_width]) |id| {
            if (id < 0) continue;
            try std.testing.expect(id >= first and id <= pos);
            try std.testing.expect(!seen[@intCast(id)]);
            seen[@intCast(id)] = true;
            count += 1;
        }
        try std.testing.expectEqual(pos + 1 - first, count);
        for (seen[first .. pos + 1]) |present| try std.testing.expect(present);
    }
    const y = try attend(&state, q, q, w, offset, 1.0 / 16.0, s);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    const got = mlx.mlx_array_data_float32(y).?;
    for (0..rows) |r| {
        const pos = offset + r;
        const completed = (pos + 1) / 4;
        const first = if (completed > 512) (completed - 512) * 4 else 0;
        var sum: f64 = 0;
        var wanted: [d]f64 = @splat(0);
        for (first..pos + 1) |token| {
            var dot: f64 = 0;
            for (lat[token * d ..][0..d]) |v| dot += @as(f64, v) * 0.125 / 16;
            const p = @exp(dot);
            sum += p;
            for (&wanted, lat[token * d ..][0..d]) |*v, k| v.* += p * k;
        }
        for (0..heads) |h| for (wanted, 0..) |v, c| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(v / sum)), got[(r * heads + h) * d + c], 2e-6);
    }
}

test "GLM latent BF16 attention covers width512 and prefill chunk continuation" {
    const s = mlx.gpuStream();
    const n = 5;
    const d = 512;
    const h = 2;
    var data: [n * d]f32 = undefined;
    var queries: [n * h * d]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 32 - 0.125;
    for (&queries, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 32 - 0.0625;
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const lat = try scope.cast(try scope.own(array(&data, &.{ n, d })), .bfloat16);
    const q = try scope.cast(try scope.own(array(&queries, &.{ n, h, d })), .bfloat16);
    const key = try scope.zeros(&.{ n, 4 }, .bfloat16);
    const ape = try scope.zeros(&.{ 4, 4 }, .bfloat16);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(lat, key, key, ape, s);
    const whole = try scope.own(try attend(&state, q, null, null, 0, 1.0 / 16.0, s));
    const full = try scope.cast(whole, .float32);
    try mlx.check(mlx.mlx_array_eval(full));
    var streamed = State.init();
    defer streamed.deinit();
    var start: c_int = 0;
    for ([_]c_int{ 1, 3, 1 }) |len| {
        const end = start + len;
        _ = try streamed.append(try scope.cut(lat, start, end), try scope.cut(key, start, end), try scope.cut(key, start, end), ape, s);
        const out = try scope.own(try attend(&streamed, try scope.cut(q, start, end), null, null, @intCast(start), 1.0 / 16.0, s));
        const got = try scope.cast(out, .float32);
        try mlx.check(mlx.mlx_array_eval(got));
        try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(full).?[@as(usize, @intCast(start)) * h * d ..][0 .. @as(usize, @intCast(len)) * h * d], mlx.mlx_array_data_float32(got).?[0 .. @as(usize, @intCast(len)) * h * d]);
        start = end;
    }
}

fn bf16Round(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}

test "GLM index score matches rounded per-head scalar math with negative weights" {
    const s = mlx.gpuStream();
    const rows = 2;
    const heads = 32;
    const dim = 128;
    const pools = 7;
    var kv: [pools * dim]f32 = undefined;
    var query: [rows * heads * dim]f32 = undefined;
    var weight: [rows * heads]f32 = undefined;
    for (&kv, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 7)) - 3) / 8;
    for (&query, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 11)) - 5) / 16;
    for (&weight, 0..) |*v, i| v.* = if (i % 2 == 0) 0.25 else -0.5;
    inline for (.{ mlx.mlx_dtype.float32, mlx.mlx_dtype.bfloat16 }) |dtype| {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const k = try scope.cast(try scope.own(array(&kv, &.{ pools, dim })), dtype);
        const q = try scope.cast(try scope.own(array(&query, &.{ rows, heads, dim })), dtype);
        const w = try scope.cast(try scope.own(array(&weight, &.{ rows, heads })), dtype);
        const state = State{ .pooled = k, .processed = pools * 4 };
        const scores = try indexScores(&scope, &state, q, w, pools * 4 - 2);
        try mlx.check(mlx.mlx_array_eval(scores));
        const got = mlx.mlx_array_data_float32(scores).?;
        for (0..rows) |r| for (0..pools) |p| {
            if ((p + 1) * 4 > pools * 4 - 1 + r) {
                try std.testing.expect(got[r * pools + p] == -std.math.inf(f32));
                continue;
            }
            var total: f32 = 0;
            for (0..heads) |h| {
                var dot: f32 = 0;
                for (0..dim) |d| dot += query[(r * heads + h) * dim + d] * kv[p * dim + d];
                if (dtype == .bfloat16) dot = bf16Round(dot);
                var product = @max(dot, 0) * weight[r * heads + h];
                if (dtype == .bfloat16) product = bf16Round(product);
                total += product;
            }
            if (dtype == .bfloat16) total = bf16Round(total);
            try std.testing.expectEqual(total, got[r * pools + p]);
        };
    }
}

test "GLM attention rejects invalid append without advancing request state" {
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const lat = try scope.zeros(&.{ 3, 4 }, .bfloat16);
    const key = try scope.zeros(&.{ 3, 4 }, .bfloat16);
    const ape = try scope.zeros(&.{ 4, 4 }, .bfloat16);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(lat, key, key, ape, s);
    const old = state.arrays();
    const bad = try scope.zeros(&.{ 3, 5 }, .bfloat16);
    try std.testing.expectError(error.InvalidGlmAttentionShape, state.append(bad, key, key, ape, s));
    try std.testing.expectEqual(@as(usize, 3), state.processed);
    for (old, state.arrays()) |before, after| try std.testing.expect(before.ctx == after.ctx);
    const q = try scope.zeros(&.{ 1, 1, 4 }, .bfloat16);
    try std.testing.expectError(error.InvalidGlmAttentionShape, attend(&state, q, null, null, 3, 1.0 / 16.0, s));
    var other = State.init();
    defer other.deinit();
    try std.testing.expectEqual(@as(usize, 0), other.processed);
}
