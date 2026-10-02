//! Partial GLM layer assembly; not yet connected to the served model forward.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const primitive = @import("glm5_next.zig");
const exl3 = @import("sushi_exl3");
const Arr = mlx.mlx_array;

pub const Ops = struct {
    s: mlx.mlx_stream,
    values: [768]Arr = undefined,
    count: usize = 0,

    pub fn deinit(self: *Ops) void {
        for (self.values[0..self.count]) |value| _ = mlx.mlx_array_free(value);
    }

    pub fn slot(self: *Ops) !*Arr {
        if (self.count == self.values.len) return error.GlmGraphTooLarge;
        self.values[self.count] = mlx.mlx_array_new();
        self.count += 1;
        return &self.values[self.count - 1];
    }

    pub fn own(self: *Ops, value: Arr) !Arr {
        if (self.count == self.values.len) {
            _ = mlx.mlx_array_free(value);
            return error.GlmGraphTooLarge;
        }
        self.values[self.count] = value;
        self.count += 1;
        return value;
    }

    pub fn cast(self: *Ops, x: Arr, dtype: mlx.mlx_dtype) !Arr {
        if (mlx.mlx_array_dtype(x) == dtype) return x;
        const out = try self.slot();
        try mlx.check(mlx.mlx_astype(out, x, dtype, self.s));
        return out.*;
    }

    pub fn reshape(self: *Ops, x: Arr, shape: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_reshape(out, x, shape.ptr, shape.len, self.s));
        return out.*;
    }

    pub fn transpose(self: *Ops, x: Arr, axes: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_transpose_axes(out, x, axes.ptr, axes.len, self.s));
        return out.*;
    }

    pub fn contiguous(self: *Ops, x: Arr) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_contiguous(out, x, false, self.s));
        return out.*;
    }

    pub fn slice(self: *Ops, x: Arr, axis: usize, start: c_int, end: c_int) !Arr {
        const shape = mlx.getShape(x);
        if (shape.len > 4 or axis >= shape.len or start < 0 or end < start or end > shape[axis]) return error.InvalidGlmShape;
        var lo: [4]c_int = @splat(0);
        var hi: [4]c_int = @splat(0);
        const step: [4]c_int = @splat(1);
        @memcpy(hi[0..shape.len], shape);
        lo[axis] = start;
        hi[axis] = end;
        const out = try self.slot();
        try mlx.check(mlx.mlx_slice(out, x, &lo, shape.len, &hi, shape.len, &step, shape.len, self.s));
        return out.*;
    }

    pub fn concat(self: *Ops, parts: []const Arr, axis: c_int) !Arr {
        const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        const out = try self.slot();
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, axis, self.s));
        return out.*;
    }

    pub fn scalar(self: *Ops, value: f32, dtype: mlx.mlx_dtype) !Arr {
        return self.cast(try self.own(mlx.mlx_array_new_float(value)), dtype);
    }

    pub fn ones(self: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_ones(out, shape.ptr, shape.len, dtype, self.s));
        return out.*;
    }

    pub fn zeros(self: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_zeros(out, shape.ptr, shape.len, dtype, self.s));
        return out.*;
    }

    const Binary = enum { add, sub, mul, div, min, max, mm, less, le, land, lor, floor_div };
    pub fn binary(self: *Ops, comptime operation: Binary, a: Arr, b: Arr) !Arr {
        const function = switch (operation) {
            .add => mlx.mlx_add,
            .sub => mlx.mlx_subtract,
            .mul => mlx.mlx_multiply,
            .div => mlx.mlx_divide,
            .min => mlx.mlx_minimum,
            .max => mlx.mlx_maximum,
            .mm => mlx.mlx_matmul,
            .less => mlx.mlx_less,
            .le => mlx.mlx_less_equal,
            .land => mlx.mlx_logical_and,
            .lor => mlx.mlx_logical_or,
            .floor_div => mlx.mlx_floor_divide,
        };
        const out = try self.slot();
        try mlx.check(function(out, a, b, self.s));
        return out.*;
    }

    const Unary = enum { exp, sigmoid, rsqrt, negative };
    pub fn unary(self: *Ops, comptime operation: Unary, x: Arr) !Arr {
        const function = switch (operation) {
            .exp => mlx.mlx_exp,
            .sigmoid => mlx.mlx_sigmoid,
            .rsqrt => mlx.mlx_rsqrt,
            .negative => mlx.mlx_negative,
        };
        const out = try self.slot();
        try mlx.check(function(out, x, self.s));
        return out.*;
    }

    pub fn reduce(self: *Ops, x: Arr, axis: c_int, mean: bool, keep: bool) !Arr {
        const out = try self.slot();
        if (mean) try mlx.check(mlx.mlx_mean_axis(out, x, axis, keep, self.s)) else try mlx.check(mlx.mlx_sum_axis(out, x, axis, keep, self.s));
        return out.*;
    }

    pub fn rms(self: *Ops, x: Arr, weight: Arr, eps: f32) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_fast_rms_norm(out, x, weight, eps, self.s));
        return out.*;
    }

    pub fn layerNorm(self: *Ops, x: Arr, weight: Arr, bias: Arr, eps: f32) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_fast_layer_norm(out, x, weight, bias, eps, self.s));
        return out.*;
    }

    pub fn softmax(self: *Ops, x: Arr, axis: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_softmax_axis(out, x, axis, true, self.s));
        return out.*;
    }

    pub fn take(self: *Ops, x: Arr, indices: Arr, axis: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_take_axis(out, x, indices, axis, self.s));
        return out.*;
    }

    pub fn broadcast(self: *Ops, x: Arr, shape: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_broadcast_to(out, x, shape.ptr, shape.len, self.s));
        return out.*;
    }

    pub fn qmm(self: *Ops, x: Arr, w: Arr, scales: Arr, biases: Arr, transposed: bool) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_quantized_matmul(out, x, w, scales, biases, transposed, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", self.s));
        return out.*;
    }

    pub fn dequant(self: *Ops, w: Arr, scales: Arr, biases: Arr) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_dequantize(out, w, scales, biases, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, self.s));
        return out.*;
    }

    pub fn silu(self: *Ops, x: Arr) !Arr {
        // The source rounds sigmoid to the activation dtype before multiplying.
        return self.binary(.mul, x, try self.unary(.sigmoid, x));
    }

    pub fn conv(self: *Ops, x: Arr, weight: Arr, groups: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_conv1d(out, x, weight, 1, 0, 1, groups, self.s));
        return out.*;
    }

    pub fn result(_: *Ops, x: Arr) !Arr {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_set(&out, x));
        return out;
    }
};

pub const Linear = struct {
    w: Arr,
    scales: Arr = .{ .ctx = null },
    biases: Arr = .{ .ctx = null },
    input: c_int,
    output: c_int,

    pub fn load(weights: *const model.Weights, base: []const u8, input: u32) !Linear {
        var buf: [256]u8 = undefined;
        const w = weights.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{base})) orelse return error.MissingGlmWeight;
        const shape = mlx.getShape(w);
        if (shape.len != 2 or input == 0 or shape[0] <= 0) return error.InvalidGlmLinear;
        if (mlx.mlx_array_dtype(w) == .uint32) {
            const sc = weights.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{base})) orelse return error.MissingGlmWeight;
            const bias = weights.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{base})) orelse return error.MissingGlmWeight;
            const grid = [_]c_int{ shape[0], @intCast(input / 128) };
            if (input % 128 != 0 or shape[1] != @as(c_int, @intCast(input / 4)) or
                !std.mem.eql(c_int, &grid, mlx.getShape(sc)) or !std.mem.eql(c_int, &grid, mlx.getShape(bias)) or
                mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bias) != .bfloat16) return error.InvalidGlmLinear;
            return .{ .w = w, .scales = sc, .biases = bias, .input = @intCast(input), .output = shape[0] };
        }
        if (shape[1] != input or (mlx.mlx_array_dtype(w) != .bfloat16 and mlx.mlx_array_dtype(w) != .float32)) return error.InvalidGlmLinear;
        return .{ .w = w, .input = @intCast(input), .output = shape[0] };
    }

    pub fn apply(self: Linear, ops: *Ops, x: Arr) !Arr {
        if (self.scales.ctx != null) return ops.qmm(x, self.w, self.scales, self.biases, true);
        return ops.binary(.mm, x, try ops.transpose(self.w, &.{ 1, 0 }));
    }
};

fn named(weights: *const model.Weights, prefix: []const u8, leaf: []const u8) !Arr {
    var buf: [256]u8 = undefined;
    return weights.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, leaf })) orelse error.MissingGlmWeight;
}

fn projection(weights: *const model.Weights, prefix: []const u8, leaf: []const u8, inputs: u32, outputs: u32) !Linear {
    var buf: [256]u8 = undefined;
    const value = try Linear.load(weights, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, leaf }), inputs);
    if (value.output != outputs) return error.InvalidGlmLinear;
    return value;
}

pub const DenseMlp = struct {
    gate: Linear,
    up: Linear,
    down: Linear,

    pub fn load(weights: *const model.Weights, prefix: []const u8, hidden: u32, intermediate: u32) !DenseMlp {
        return .{
            .gate = try projection(weights, prefix, "gate_proj", hidden, intermediate),
            .up = try projection(weights, prefix, "up_proj", hidden, intermediate),
            .down = try projection(weights, prefix, "down_proj", intermediate, hidden),
        };
    }

    pub fn apply(self: DenseMlp, ops: *Ops, x: Arr, limit: f32) !Arr {
        const gate = try self.gate.apply(ops, x);
        const up = try self.up.apply(ops, x);
        if (try @import("glm5_activation.zig").apply(ops.s, gate, up, limit)) |middle|
            return self.down.apply(ops, try ops.own(middle));
        const hi = try ops.scalar(limit, mlx.mlx_array_dtype(gate));
        const lo = try ops.scalar(-limit, mlx.mlx_array_dtype(up));
        const cg = try ops.binary(.min, gate, hi);
        const cu = try ops.binary(.max, try ops.binary(.min, up, hi), lo);
        return self.down.apply(ops, try ops.binary(.mul, try ops.silu(cg), cu));
    }
};

var hc_mix_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn hcMixExact(ops: *Ops, x: Arr, w: Arr) !Arr {
    const shape = mlx.getShape(x);
    const rows = shape[0] * shape[1];
    const width = shape[2];
    if (hc_mix_kernel == null) {
        const inputs = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{ "x", "w" }, 2);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outputs);
        const source: [:0]const u8 =
            \\const uint group = threadgroup_position_in_grid.x;
            \\const uint row = group / 24u;
            \\const uint output = group % 24u;
            \\const uint lane = thread_position_in_threadgroup.x;
            \\threadgroup float partial[4];
            \\float value = 0.0f;
            \\for (uint d = lane; d < uint(WIDTH); d += 128u)
            \\  value += x[row * uint(WIDTH) + d] * float(w[output * uint(WIDTH) + d]);
            \\value = simd_sum(value);
            \\if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = value;
            \\threadgroup_barrier(mem_flags::mem_threadgroup);
            \\if (lane == 0) out[row * 24u + output] = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        ;
        const kernel = mlx.mlx_fast_metal_kernel_new("sushi_glm_hc_mix_fp32", inputs, outputs, source, "", true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        hc_mix_kernel = kernel;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ shape[0], shape[1], 24 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 24 * 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIDTH", width));
    const inputs = mlx.mlx_vector_array_new_data(&[_]Arr{ x, w }, 2);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, hc_mix_kernel.?, inputs, cfg, ops.s));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(out, outputs, 0));
    return out.*;
}

pub const Hc = struct {
    w: Arr,
    scale: Arr,
    base: Arr,

    pub fn load(weights: *const model.Weights, prefix: []const u8, label: []const u8, hidden: u32) !Hc {
        var buf: [256]u8 = undefined;
        const value = Hc{
            .w = try named(weights, prefix, try std.fmt.bufPrint(&buf, "{s}_fn", .{label})),
            .scale = try named(weights, prefix, try std.fmt.bufPrint(&buf, "{s}_scale", .{label})),
            .base = try named(weights, prefix, try std.fmt.bufPrint(&buf, "{s}_base", .{label})),
        };
        if (!std.mem.eql(c_int, &.{ 24, @intCast(hidden * 4) }, mlx.getShape(value.w)) or
            mlx.mlx_array_size(value.scale) != 3 or mlx.mlx_array_size(value.base) != 24 or
            (mlx.mlx_array_dtype(value.w) != .float32 and mlx.mlx_array_dtype(value.w) != .bfloat16) or mlx.mlx_array_dtype(value.scale) != .float32 or mlx.mlx_array_dtype(value.base) != .float32) return error.InvalidGlmHc;
        return value;
    }

    pub fn collapse(self: Hc, ops: *Ops, x: Arr, cfg: *const model.ModelConfig) !primitive.HcResult {
        if (try @import("glm5_hc_fused.zig").mix(ops.s, x, self.w, cfg.rms_norm_eps)) |candidate| {
            const mixes = try ops.own(candidate);
            return primitive.hcCollapse(x, mixes, self.scale, self.base, @intCast(cfg.glm_hc_sinkhorn_iters), cfg.glm_hc_eps, ops.s);
        }
        return self.collapseReference(ops, x, cfg);
    }

    pub fn collapseReference(self: Hc, ops: *Ops, x: Arr, cfg: *const model.ModelConfig) !primitive.HcResult {
        const sh = mlx.getShape(x);
        const flat = try ops.reshape(try ops.cast(x, .float32), &.{ sh[0], sh[1], sh[2] * sh[3] });
        const normalized = try ops.rms(flat, .{ .ctx = null }, cfg.rms_norm_eps);
        // This sensitive FP32 projection must not take the backend's TF32 path.
        const mixes = try hcMixExact(ops, normalized, self.w);
        return primitive.hcCollapse(x, mixes, try ops.cast(self.scale, .float32), try ops.cast(self.base, .float32), @intCast(cfg.glm_hc_sinkhorn_iters), cfg.glm_hc_eps, ops.s);
    }
};

fn compactConvTail(ops: *Ops, input: Arr, rows: c_int, keep: c_int) !Arr {
    const tail = try ops.slice(input, 1, rows, rows + keep);
    // A batch-one tail is already contiguous but still owns its whole prompt parent.
    if (rows > 1) return ops.own(try @import("transformer.zig").materializedOwnedCopy(ops.s, tail));
    return ops.contiguous(tail);
}

pub const KdaLayer = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    fa: Linear,
    fb: Linear,
    ga: Linear,
    gb: Linear,
    beta: Linear,
    out: Linear,
    conv_q: Arr,
    conv_k: Arr,
    conv_v: Arr,
    a_log: Arr,
    dt_bias: Arr,
    out_norm: Arr,
    prepared_conv: Arr = .{ .ctx = null },
    prepared_decay: Arr = .{ .ctx = null },

    pub fn prepare(self: *KdaLayer, stream: mlx.mlx_stream) !void {
        if (self.prepared_conv.ctx != null and self.prepared_decay.ctx != null) return;
        var ops = Ops{ .s = stream };
        defer ops.deinit();
        const conv = try ops.contiguous(try ops.transpose(try ops.concat(&.{ self.conv_q, self.conv_k, self.conv_v }, 0), &.{ 0, 2, 1 }));
        const decay = try ops.unary(.exp, self.a_log);
        try mlx.check(mlx.mlx_array_eval(conv));
        try mlx.check(mlx.mlx_array_eval(decay));
        const owned_conv = try ops.result(conv);
        errdefer _ = mlx.mlx_array_free(owned_conv);
        const owned_decay = try ops.result(decay);
        self.deinit();
        self.prepared_conv = owned_conv;
        self.prepared_decay = owned_decay;
    }

    pub fn deinit(self: *KdaLayer) void {
        if (self.prepared_conv.ctx != null) _ = mlx.mlx_array_free(self.prepared_conv);
        if (self.prepared_decay.ctx != null) _ = mlx.mlx_array_free(self.prepared_decay);
        self.prepared_conv = .{ .ctx = null };
        self.prepared_decay = .{ .ctx = null };
    }

    pub fn load(weights: *const model.Weights, prefix: []const u8, cfg: *const model.ModelConfig) !KdaLayer {
        const h = cfg.hidden_size;
        const d = cfg.linear_key_head_dim;
        const width = cfg.linear_num_value_heads * d;
        const result = KdaLayer{
            .q = try projection(weights, prefix, "q_proj", h, width),
            .k = try projection(weights, prefix, "k_proj", h, width),
            .v = try projection(weights, prefix, "v_proj", h, width),
            .fa = try projection(weights, prefix, "f_a_proj", h, d),
            .fb = try projection(weights, prefix, "f_b_proj", d, width),
            .ga = try projection(weights, prefix, "g_a_proj", h, d),
            .gb = try projection(weights, prefix, "g_b_proj", d, width),
            .beta = try projection(weights, prefix, "b_proj", h, cfg.linear_num_value_heads),
            .out = try projection(weights, prefix, "o_proj", width, h),
            .conv_q = try named(weights, prefix, "q_conv1d.weight"),
            .conv_k = try named(weights, prefix, "k_conv1d.weight"),
            .conv_v = try named(weights, prefix, "v_conv1d.weight"),
            .a_log = try named(weights, prefix, "A_log"),
            .dt_bias = try named(weights, prefix, "dt_bias"),
            .out_norm = try named(weights, prefix, "o_norm.weight"),
        };
        const conv_shape = [_]c_int{ @intCast(width), 1, @intCast(cfg.linear_conv_kernel_dim) };
        for ([_]Arr{ result.conv_q, result.conv_k, result.conv_v }) |w| {
            if (!std.mem.eql(c_int, &conv_shape, mlx.getShape(w)) or mlx.mlx_array_dtype(w) != .bfloat16) return error.InvalidGlmKdaWeight;
        }
        if (!std.mem.eql(c_int, &.{@intCast(cfg.linear_num_value_heads)}, mlx.getShape(result.a_log)) or
            !std.mem.eql(c_int, &.{@intCast(width)}, mlx.getShape(result.dt_bias)) or
            !std.mem.eql(c_int, &.{@intCast(d)}, mlx.getShape(result.out_norm)) or
            mlx.mlx_array_dtype(result.a_log) != .float32 or mlx.mlx_array_dtype(result.dt_bias) != .float32 or
            mlx.mlx_array_dtype(result.out_norm) != .bfloat16) return error.InvalidGlmKdaWeight;
        return result;
    }

    fn projectQkv(self: KdaLayer, ops: *Ops, x: Arr) !Arr {
        const projections = if (try @import("glm5_decode.zig").qkv(ops.s, x, .{
            .{ .weight = self.q.w, .scales = self.q.scales, .biases = self.q.biases },
            .{ .weight = self.k.w, .scales = self.k.scales, .biases = self.k.biases },
            .{ .weight = self.v.w, .scales = self.v.scales, .biases = self.v.biases },
        })) |group| blk: {
            var owned: usize = 0;
            errdefer for (group[owned..]) |value| {
                _ = mlx.mlx_array_free(value);
            };
            var result: [3]Arr = undefined;
            for (group, 0..) |value, i| {
                owned += 1; // Ops.own also frees its argument if ownership fails.
                result[i] = try ops.own(value);
            }
            break :blk result;
        } else [_]Arr{ try self.q.apply(ops, x), try self.k.apply(ops, x), try self.v.apply(ops, x) };
        return ops.concat(&projections, -1);
    }

    pub fn applyFused(self: KdaLayer, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, state: *@import("transformer.zig").SSMCacheEntry) !?Arr {
        const sh = mlx.getShape(x);
        if (sh.len != 3 or sh[0] != 1 or sh[1] != 1 or mlx.mlx_array_dtype(x) != .bfloat16 or cfg.linear_key_head_dim != 128 or cfg.linear_conv_kernel_dim != 4 or cfg.kda_gate_lower_bound != -5 or self.prepared_conv.ctx == null or self.prepared_decay.ctx == null or !@import("glm5_kda_fused.zig").hardwareSupported()) return null;
        const heads: c_int = @intCast(cfg.linear_num_value_heads);
        const width = heads * 128;
        const joined = try self.projectQkv(ops, x);
        const previous = if (state.initialized) state.conv_state else try ops.zeros(&.{ 1, 3, width * 3 }, .bfloat16);
        const recurrent = if (state.initialized) state.ssm_state else try ops.zeros(&.{ 1, heads, 128, 128 }, .float32);
        const a = try self.fb.apply(ops, try self.fa.apply(ops, x));
        const gate = try self.gb.apply(ops, try self.ga.apply(ops, x));
        const beta = try self.beta.apply(ops, x);
        const result = (try @import("glm5_kda_fused.zig").step(ops.s, .{ .qkv = joined, .beta = beta, .a = a, .gate = gate, .conv_weight = self.prepared_conv, .exp_a = self.prepared_decay, .dt_bias = self.dt_bias, .norm = self.out_norm, .conv_state = previous, .state = recurrent, .heads = heads, .norm_eps = cfg.rms_norm_eps, .lower = cfg.kda_gate_lower_bound })) orelse return null;
        defer result.deinit();
        try mlx.check(mlx.mlx_array_set(&state.conv_state, result.conv));
        try mlx.check(mlx.mlx_array_set(&state.ssm_state, result.state));
        state.initialized = true;
        return try self.out.apply(ops, result.y);
    }

    pub fn apply(self: KdaLayer, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, state: *@import("transformer.zig").SSMCacheEntry) !Arr {
        if (try self.applyFused(ops, x, cfg, state)) |result| return result;
        return self.applyReference(ops, x, cfg, state);
    }

    pub fn applyReference(self: KdaLayer, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, state: *@import("transformer.zig").SSMCacheEntry) !Arr {
        const sh = mlx.getShape(x);
        const heads: c_int = @intCast(cfg.linear_num_value_heads);
        const dim: c_int = @intCast(cfg.linear_key_head_dim);
        const width = heads * dim;
        const keep: c_int = @intCast(cfg.linear_conv_kernel_dim - 1);
        const joined = try self.projectQkv(ops, x);
        const previous = if (state.initialized) state.conv_state else try ops.zeros(&.{ sh[0], keep, width * 3 }, mlx.mlx_array_dtype(x));
        const conv_input = try ops.concat(&.{ previous, joined }, 1);
        const conv_weight = if (self.prepared_conv.ctx != null) self.prepared_conv else try ops.contiguous(try ops.transpose(try ops.concat(&.{ self.conv_q, self.conv_k, self.conv_v }, 0), &.{ 0, 2, 1 }));
        const convolved = try ops.silu(try ops.conv(conv_input, conv_weight, width * 3));
        const dims = [_]c_int{ sh[0], sh[1], heads, dim };
        const raw_q = try ops.cast(try ops.reshape(try ops.slice(convolved, 2, 0, width), &dims), .float32);
        const raw_k = try ops.cast(try ops.reshape(try ops.slice(convolved, 2, width, width * 2), &dims), .float32);
        const values = try ops.reshape(try ops.slice(convolved, 2, width * 2, width * 3), &dims);
        const epsilon = try ops.scalar(1e-6, .float32);
        const qnorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, raw_q, raw_q), -1, false, true), epsilon));
        const knorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, raw_k, raw_k), -1, false, true), epsilon));
        const q = try ops.cast(try ops.binary(.mul, try ops.binary(.mul, raw_q, qnorm), try ops.scalar(1 / @sqrt(@as(f32, @floatFromInt(dim))), .float32)), mlx.mlx_array_dtype(x));
        const k = try ops.cast(try ops.binary(.mul, raw_k, knorm), mlx.mlx_array_dtype(x));
        const a = try ops.reshape(try ops.cast(try self.fb.apply(ops, try self.fa.apply(ops, x)), .float32), &dims);
        const shift = try ops.reshape(try ops.cast(self.dt_bias, .float32), &.{ 1, 1, heads, dim });
        const exp_decay = if (self.prepared_decay.ctx != null) self.prepared_decay else try ops.unary(.exp, self.a_log);
        const magnitude = try ops.reshape(exp_decay, &.{ 1, 1, heads, 1 });
        const forget = try ops.unary(.sigmoid, try ops.binary(.mul, magnitude, try ops.binary(.add, a, shift)));
        const decay = try ops.unary(.exp, try ops.binary(.mul, forget, try ops.scalar(cfg.kda_gate_lower_bound, .float32)));
        const beta = try ops.unary(.sigmoid, try self.beta.apply(ops, x));
        const recurrent = if (state.initialized) state.ssm_state else try ops.zeros(&.{ sh[0], heads, dim, dim }, .float32);
        const result = try primitive.kda(.{ .q = q, .k = k, .v = values, .decay = decay, .beta = beta, .state = recurrent }, ops.s);
        defer result.deinit();
        const new_conv = try compactConvTail(ops, conv_input, sh[1], keep);
        try mlx.check(mlx.mlx_array_set(&state.conv_state, new_conv));
        try mlx.check(mlx.mlx_array_set(&state.ssm_state, result.state));
        state.initialized = true;
        const gate = try ops.reshape(try self.gb.apply(ops, try self.ga.apply(ops, x)), &dims);
        const candidate = if (sh[1] > 1) try @import("glm5_kda_fused.zig").post(ops.s, result.y, gate, self.out_norm, cfg.rms_norm_eps) else null;
        const gated = if (candidate) |value| try ops.own(value) else blk: {
            const y = try ops.cast(result.y, .float32);
            const variance = try ops.reduce(try ops.binary(.mul, y, y), -1, true, true);
            const normalization = try ops.unary(.rsqrt, try ops.binary(.add, variance, try ops.scalar(cfg.rms_norm_eps, .float32)));
            const normalized = try ops.binary(.mul, try ops.binary(.mul, y, normalization), try ops.cast(self.out_norm, .float32));
            break :blk try ops.cast(try ops.binary(.mul, normalized, try ops.unary(.sigmoid, try ops.cast(gate, .float32))), mlx.mlx_array_dtype(x));
        };
        return self.out.apply(ops, try ops.reshape(gated, &.{ sh[0], sh[1], width }));
    }
};

test "GLM model affine projection and transpose preserve stored grids" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const codes: [4 * 32]u32 = @splat(0x01010101);
    const scales: [4]u16 = @splat(0x3f00);
    const biases: [4]u16 = @splat(0);
    try weights.map.put(try a.dupe(u8, "p.weight"), mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 32 }, 2, .uint32));
    try weights.map.put(try a.dupe(u8, "p.scales"), mlx.mlx_array_new_data(&scales, &[_]c_int{ 4, 1 }, 2, .bfloat16));
    try weights.map.put(try a.dupe(u8, "p.biases"), mlx.mlx_array_new_data(&biases, &[_]c_int{ 4, 1 }, 2, .bfloat16));
    const linear = try Linear.load(&weights, "p", 128);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.ones(&.{ 1, 2, 128 }, .bfloat16);
    const y = try ops.cast(try linear.apply(&ops, x), .float32);
    try mlx.check(mlx.mlx_array_eval(y));
    for (mlx.mlx_array_data_float32(y).?[0..8]) |v| try std.testing.expectEqual(@as(f32, 64), v);
    const xt = try ops.ones(&.{ 1, 4 }, .bfloat16);
    const yt = try ops.cast(try ops.qmm(xt, linear.w, linear.scales, linear.biases, false), .float32);
    try mlx.check(mlx.mlx_array_eval(yt));
    for (mlx.mlx_array_data_float32(yt).?[0..128]) |v| try std.testing.expectEqual(@as(f32, 2), v);
    try std.testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(weights.get("p.weight").?));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("p.scales").?));
}

test "GLM prefill convolution tail owns its compact storage" {
    const s = mlx.gpuStream();
    const rows = 64;
    const keep = 3;
    const width = 12;
    var data: [(rows + keep) * width]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 16;
    const parent = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, rows + keep, width }, 3, .float32);
    defer _ = mlx.mlx_array_free(parent);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const tail = try compactConvTail(&ops, parent, rows, keep);
    try mlx.check(mlx.mlx_array_eval(parent));
    try mlx.check(mlx.mlx_array_eval(tail));
    const pp = mlx.mlx_array_data_float32(parent).?;
    const tp = mlx.mlx_array_data_float32(tail).?;
    try std.testing.expectEqualSlices(f32, data[rows * width ..], tp[0 .. keep * width]);
    const address = @intFromPtr(tp);
    try std.testing.expect(address < @intFromPtr(pp) or address >= @intFromPtr(pp) + @sizeOf(@TypeOf(data)));
}

fn putValidationWeight(weights: *model.Weights, key: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype) !void {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_zeros(&out, shape.ptr, shape.len, dtype, mlx.gpuStream()));
    try weights.map.put(try std.testing.allocator.dupe(u8, key), out);
}

test "GLM KDA loader refuses broadcast norms and malformed preserved tensors" {
    for (0..5) |bad| {
        var weights = model.Weights.init(std.testing.allocator);
        defer weights.deinit();
        for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "o_proj", "b_proj" }) |p| {
            var buf: [96]u8 = undefined;
            try putValidationWeight(&weights, try std.fmt.bufPrint(&buf, "a.{s}.weight", .{p}), &.{ if (std.mem.eql(u8, p, "b_proj")) 1 else 128, 128 }, .bfloat16);
        }
        for ([_][]const u8{ "a.q_conv1d.weight", "a.k_conv1d.weight", "a.v_conv1d.weight" }) |p|
            try putValidationWeight(&weights, p, if (bad == 2) &.{ 128, 4, 1 } else &.{ 128, 1, 4 }, .bfloat16);
        try putValidationWeight(&weights, "a.A_log", &.{1}, if (bad == 1) .bfloat16 else .float32);
        try putValidationWeight(&weights, "a.dt_bias", if (bad == 3) &.{127} else &.{128}, .float32);
        try putValidationWeight(&weights, "a.o_norm.weight", if (bad == 0) &.{1} else &.{128}, .bfloat16);
        const cfg = model.ModelConfig{ .hidden_size = 128, .linear_key_head_dim = 128, .linear_num_value_heads = 1, .linear_conv_kernel_dim = 4 };
        if (bad == 4) {
            _ = try KdaLayer.load(&weights, "a", &cfg);
        } else try std.testing.expectError(error.InvalidGlmKdaWeight, KdaLayer.load(&weights, "a", &cfg));
    }
}

test "GLM HC loader preserves BF16 matrix and requires FP32 coefficients" {
    for ([_]bool{ false, true }) |rounded| {
        var weights = model.Weights.init(std.testing.allocator);
        defer weights.deinit();
        try putValidationWeight(&weights, "a.hc_attn_fn", &.{ 24, 512 }, .bfloat16);
        try putValidationWeight(&weights, "a.hc_attn_scale", &.{3}, if (rounded) .bfloat16 else .float32);
        try putValidationWeight(&weights, "a.hc_attn_base", &.{24}, .float32);
        if (rounded) {
            try std.testing.expectError(error.InvalidGlmHc, Hc.load(&weights, "a", "hc_attn", 128));
        } else {
            const hc = try Hc.load(&weights, "a", "hc_attn", 128);
            try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(hc.w));
        }
    }
}

fn loadLayerFixture() !model.Weights {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "layers.safetensors", .data = @embedFile("fixtures/glm5_layers.safetensors") });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path_buf);
    const path = try std.fmt.allocPrintSentinel(t.allocator, "{s}/layers.safetensors", .{path_buf[0..n]}, 0);
    defer t.allocator.free(path);
    var tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&tensors, &metadata, path.ptr, cpu));
    const iter = mlx.mlx_map_string_to_array_iterator_new(tensors);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    var weights = model.Weights.init(t.allocator);
    errdefer weights.deinit();
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        try mlx.check(mlx.mlx_array_eval(value));
        try weights.map.put(try t.allocator.dupe(u8, std.mem.span(key.?)), value);
    }
    return weights;
}

fn expectLayerReference(actual: Arr, expected: Arr, abs: f32, rel: f32) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(actual));
    try std.testing.expectEqual(mlx.mlx_array_dtype(expected), mlx.mlx_array_dtype(actual));
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const a = try ops.contiguous(try ops.cast(actual, .float32));
    const e = try ops.contiguous(try ops.cast(expected, .float32));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(e));
    const size = mlx.mlx_array_size(a);
    var maximum: f32 = 0;
    var relative: f32 = 0;
    for (mlx.mlx_array_data_float32(a).?[0..size], mlx.mlx_array_data_float32(e).?[0..size]) |v, want| {
        maximum = @max(maximum, @abs(v - want));
        relative = @max(relative, @abs(v - want) / @max(@abs(want), 1e-8));
        try std.testing.expectApproxEqAbs(want, v, abs + rel * @abs(want));
    }
    if (@import("transformer.zig").diagEnvOn("SUSHI_GLM_REFERENCE_STATS"))
        std.debug.print("GLM reference {s} {any}: max_abs={e} max_rel={e}\n", .{ @tagName(mlx.mlx_array_dtype(actual)), mlx.getShape(actual), maximum, relative });
}

test "GLM reference full KDA apply matches oMLX serial and irregular chunks" {
    var fixture = try loadLayerFixture();
    defer fixture.deinit();
    const cfg = model.ModelConfig{ .hidden_size = 128, .linear_key_head_dim = 128, .linear_num_value_heads = 1, .linear_conv_kernel_dim = 4, .kda_gate_lower_bound = -5, .rms_norm_eps = 1e-5 };
    var layer = try KdaLayer.load(&fixture, "a", &cfg);
    defer layer.deinit();
    try layer.prepare(mlx.gpuStream());
    const Case = struct { label: []const u8, chunks: []const c_int };
    for ([_]Case{ .{ .label = "full", .chunks = &.{5} }, .{ .label = "serial", .chunks = &.{ 1, 1, 1, 1, 1 } }, .{ .label = "irregular", .chunks = &.{ 2, 1, 2 } }, .{ .label = "cold", .chunks = &.{5} } }) |case| {
        var state = @import("transformer.zig").SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = !std.mem.eql(u8, case.label, "cold") };
        defer _ = mlx.mlx_array_free(state.conv_state);
        defer _ = mlx.mlx_array_free(state.ssm_state);
        try mlx.check(mlx.mlx_array_set(&state.conv_state, fixture.get("initial.conv").?));
        try mlx.check(mlx.mlx_array_set(&state.ssm_state, fixture.get("initial.state").?));
        var pos: c_int = 0;
        for (case.chunks) |count| {
            var ops = Ops{ .s = mlx.gpuStream() };
            defer ops.deinit();
            const input = try ops.slice(fixture.get("input").?, 1, pos, pos + count);
            const output = try layer.apply(&ops, input, &cfg, &state);
            var name: [48]u8 = undefined;
            const reference = fixture.get(try std.fmt.bufPrint(&name, "{s}.output", .{case.label})).?;
            // One BF16 rounding unit plus a small cancellation allowance.
            try expectLayerReference(output, try ops.slice(reference, 1, pos, pos + count), 0.001, 0.01);
            try mlx.check(mlx.mlx_array_eval(state.conv_state));
            try mlx.check(mlx.mlx_array_eval(state.ssm_state));
            pos += count;
        }
        var name: [48]u8 = undefined;
        try expectLayerReference(state.conv_state, fixture.get(try std.fmt.bufPrint(&name, "{s}.conv", .{case.label})).?, 0.001, 0.01);
        try expectLayerReference(state.ssm_state, fixture.get(try std.fmt.bufPrint(&name, "{s}.state", .{case.label})).?, 1e-5, 0.001);
    }
}

test "GLM reference mHC assembled collapse and expand match oMLX" {
    var fixture = try loadLayerFixture();
    defer fixture.deinit();
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const hc = try Hc.load(&fixture, "h", "hc_attn", 128);
    const cfg = model.ModelConfig{ .rms_norm_eps = 1e-5 };
    const input = fixture.get("hc.input").?;
    const collapsed = try hc.collapse(&ops, input, &cfg);
    defer collapsed.deinit();
    try expectLayerReference(collapsed.mixed, fixture.get("hc.mixed").?, 0.002, 0.008);
    try expectLayerReference(collapsed.post, fixture.get("hc.post").?, 2e-5, 0);
    try expectLayerReference(collapsed.comb, fixture.get("hc.comb").?, 2e-5, 0);
    const expanded = try primitive.hcExpand(input, collapsed.mixed, collapsed.post, collapsed.comb, ops.s);
    defer _ = mlx.mlx_array_free(expanded);
    try expectLayerReference(expanded, fixture.get("hc.expanded").?, 0.002, 0.008);
}

test "GLM reference SiLU preserves source BF16 convolution and dense FFN rounding" {
    var fixture = try loadLayerFixture();
    defer fixture.deinit();
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const input = fixture.get("silu.input").?;
    try expectLayerReference(try ops.silu(input), fixture.get("silu.conv").?, 0, 0);
    const hi = try ops.scalar(10, .bfloat16);
    const lo = try ops.scalar(-10, .bfloat16);
    const gate = try ops.binary(.min, input, hi);
    const up = try ops.binary(.max, try ops.binary(.min, fixture.get("silu.up").?, hi), lo);
    try expectLayerReference(try ops.binary(.mul, try ops.silu(gate), up), fixture.get("silu.dense").?, 0, 0);
}

test "GLM prepared KDA constants preserve results and have idempotent ownership" {
    var fixture = try loadLayerFixture();
    defer fixture.deinit();
    const cfg = model.ModelConfig{ .hidden_size = 128, .linear_key_head_dim = 128, .linear_num_value_heads = 1, .linear_conv_kernel_dim = 4, .kda_gate_lower_bound = -5, .rms_norm_eps = 1e-5 };
    var layer = try KdaLayer.load(&fixture, "a", &cfg);
    defer layer.deinit();
    var results: [2]Arr = undefined;
    var states: [2]Arr = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        _ = mlx.mlx_array_free(results[i]);
        _ = mlx.mlx_array_free(states[i]);
    };
    for (0..2) |pass| {
        if (pass == 1) {
            try layer.prepare(mlx.gpuStream());
            const owned = layer.prepared_conv.ctx;
            try layer.prepare(mlx.gpuStream());
            try std.testing.expectEqual(owned, layer.prepared_conv.ctx);
        }
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        var state = @import("transformer.zig").SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer _ = mlx.mlx_array_free(state.conv_state);
        defer _ = mlx.mlx_array_free(state.ssm_state);
        const out = try layer.apply(&ops, fixture.get("input").?, &cfg, &state);
        try mlx.check(mlx.mlx_array_eval(out));
        try mlx.check(mlx.mlx_array_eval(state.ssm_state));
        results[pass] = try ops.result(out);
        states[pass] = try ops.result(state.ssm_state);
        made += 1;
    }
    try expectLayerReference(results[0], results[1], 0, 0);
    try expectLayerReference(states[0], states[1], 0, 0);
    layer.deinit();
    layer.deinit();
    try std.testing.expect(layer.prepared_conv.ctx == null and layer.prepared_decay.ctx == null);
}

fn expectLayerBits(actual: Arr, expected: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(actual));
    try std.testing.expectEqual(mlx.mlx_array_dtype(expected), mlx.mlx_array_dtype(actual));
    try mlx.check(mlx.mlx_array_eval(actual));
    try mlx.check(mlx.mlx_array_eval(expected));
    const n = mlx.mlx_array_size(actual);
    if (mlx.mlx_array_dtype(actual) == .bfloat16) {
        for (mlx.mlx_array_data_bfloat16(actual).?[0..n], mlx.mlx_array_data_bfloat16(expected).?[0..n]) |x, y| try std.testing.expectEqual(y, x);
    } else {
        for (mlx.mlx_array_data_float32(actual).?[0..n], mlx.mlx_array_data_float32(expected).?[0..n]) |x, y| try std.testing.expectEqual(@as(u32, @bitCast(y)), @as(u32, @bitCast(x)));
    }
}

test "GLM fused KDA decode preserves output and FP32 state exactly" {
    if (!@import("glm5_kda_fused.zig").hardwareSupported()) return error.SkipZigTest;
    for ([_]u32{ 1, 3, 64 }) |heads| {
        var fixture = try loadLayerFixture();
        defer fixture.deinit();
        if (heads > 1) {
            const Repeated = struct { key: []const u8, axis: c_int };
            for ([_]Repeated{
                .{ .key = "a.q_proj.weight", .axis = 0 },   .{ .key = "a.k_proj.weight", .axis = 0 },   .{ .key = "a.v_proj.weight", .axis = 0 },
                .{ .key = "a.f_b_proj.weight", .axis = 0 }, .{ .key = "a.g_b_proj.weight", .axis = 0 }, .{ .key = "a.b_proj.weight", .axis = 0 },
                .{ .key = "a.o_proj.weight", .axis = 1 },   .{ .key = "a.q_conv1d.weight", .axis = 0 }, .{ .key = "a.k_conv1d.weight", .axis = 0 },
                .{ .key = "a.v_conv1d.weight", .axis = 0 }, .{ .key = "a.A_log", .axis = 0 },           .{ .key = "a.dt_bias", .axis = 0 },
                .{ .key = "initial.conv", .axis = 2 },      .{ .key = "initial.state", .axis = 1 },
            }) |spec| {
                const value = fixture.map.getPtr(spec.key).?;
                var repeated = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_repeat_axis(&repeated, value.*, @intCast(heads), spec.axis, mlx.gpuStream()));
                _ = mlx.mlx_array_free(value.*);
                value.* = repeated;
            }
        }
        {
            var setup = Ops{ .s = mlx.gpuStream() };
            defer setup.deinit();
            var head_scale: [64]f32 = undefined;
            var logs: [64]f32 = undefined;
            for (head_scale[0..heads], logs[0..heads], 0..) |*scale, *value, h| {
                scale.* = 0.5 + @as(f32, @floatFromInt(h)) / 64;
                value.* = -0.5 + @as(f32, @floatFromInt(h)) / 32;
            }
            const decay = fixture.map.getPtr("a.A_log").?;
            const decay_new = mlx.mlx_array_new_data(&logs, &[_]c_int{@intCast(heads)}, 1, .float32);
            _ = mlx.mlx_array_free(decay.*);
            decay.* = decay_new;
            const factors = try setup.own(mlx.mlx_array_new_data(&head_scale, &[_]c_int{ @intCast(heads), 1 }, 2, .float32));
            const beta = fixture.map.getPtr("a.b_proj.weight").?;
            const beta_new = try setup.result(try setup.cast(try setup.binary(.mul, beta.*, factors), .bfloat16));
            _ = mlx.mlx_array_free(beta.*);
            beta.* = beta_new;
            const initial = fixture.map.getPtr("initial.state").?;
            const initial_new = try setup.result(try setup.binary(.mul, initial.*, try setup.reshape(factors, &.{ 1, @intCast(heads), 1, 1 })));
            _ = mlx.mlx_array_free(initial.*);
            initial.* = initial_new;
        }
        const cfg = model.ModelConfig{ .hidden_size = 128, .linear_key_head_dim = 128, .linear_num_value_heads = heads, .linear_conv_kernel_dim = 4, .kda_gate_lower_bound = -5, .rms_norm_eps = 1e-5 };
        var layer = try KdaLayer.load(&fixture, "a", &cfg);
        try layer.prepare(mlx.gpuStream());
        defer layer.deinit();
        for ([_]bool{ false, true }) |initial| {
            var reference = @import("transformer.zig").SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = initial };
            defer _ = mlx.mlx_array_free(reference.conv_state);
            defer _ = mlx.mlx_array_free(reference.ssm_state);
            var fused = @import("transformer.zig").SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = initial };
            defer _ = mlx.mlx_array_free(fused.conv_state);
            defer _ = mlx.mlx_array_free(fused.ssm_state);
            if (initial) {
                var ops = Ops{ .s = mlx.gpuStream() };
                defer ops.deinit();
                const conv = try ops.slice(fixture.get("initial.conv").?, 0, 0, 1);
                const recurrent = try ops.slice(fixture.get("initial.state").?, 0, 0, 1);
                try mlx.check(mlx.mlx_array_set(&reference.conv_state, conv));
                try mlx.check(mlx.mlx_array_set(&fused.conv_state, conv));
                try mlx.check(mlx.mlx_array_set(&reference.ssm_state, recurrent));
                try mlx.check(mlx.mlx_array_set(&fused.ssm_state, recurrent));
            }
            for (0..5) |i| {
                var ops = Ops{ .s = mlx.gpuStream() };
                defer ops.deinit();
                const x = try ops.slice(try ops.slice(fixture.get("input").?, 0, 0, 1), 1, @intCast(i), @intCast(i + 1));
                const expected = try layer.applyReference(&ops, x, &cfg, &reference);
                const got = (try layer.applyFused(&ops, x, &cfg, &fused)) orelse return error.TestExpectedFusedKda;
                try expectLayerBits(got, expected);
                try expectLayerBits(fused.conv_state, reference.conv_state);
                try expectLayerBits(fused.ssm_state, reference.ssm_state);
            }
        }
    }
}
