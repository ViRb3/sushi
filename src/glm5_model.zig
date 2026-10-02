//! Partial GLM layer assembly; not yet connected to the served model forward.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const primitive = @import("glm5_next.zig");
const exl3 = @import("sushi_exl3");
const Arr = mlx.mlx_array;

const Ops = struct {
    s: mlx.mlx_stream,
    values: [768]Arr = undefined,
    count: usize = 0,

    fn deinit(self: *Ops) void {
        for (self.values[0..self.count]) |value| _ = mlx.mlx_array_free(value);
    }

    fn slot(self: *Ops) !*Arr {
        if (self.count == self.values.len) return error.GlmGraphTooLarge;
        self.values[self.count] = mlx.mlx_array_new();
        self.count += 1;
        return &self.values[self.count - 1];
    }

    fn own(self: *Ops, value: Arr) !Arr {
        if (self.count == self.values.len) {
            _ = mlx.mlx_array_free(value);
            return error.GlmGraphTooLarge;
        }
        self.values[self.count] = value;
        self.count += 1;
        return value;
    }

    fn cast(self: *Ops, x: Arr, dtype: mlx.mlx_dtype) !Arr {
        if (mlx.mlx_array_dtype(x) == dtype) return x;
        const out = try self.slot();
        try mlx.check(mlx.mlx_astype(out, x, dtype, self.s));
        return out.*;
    }

    fn reshape(self: *Ops, x: Arr, shape: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_reshape(out, x, shape.ptr, shape.len, self.s));
        return out.*;
    }

    fn transpose(self: *Ops, x: Arr, axes: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_transpose_axes(out, x, axes.ptr, axes.len, self.s));
        return out.*;
    }

    fn contiguous(self: *Ops, x: Arr) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_contiguous(out, x, false, self.s));
        return out.*;
    }

    fn slice(self: *Ops, x: Arr, axis: usize, start: c_int, end: c_int) !Arr {
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

    fn concat(self: *Ops, parts: []const Arr, axis: c_int) !Arr {
        const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        const out = try self.slot();
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, axis, self.s));
        return out.*;
    }

    fn scalar(self: *Ops, value: f32, dtype: mlx.mlx_dtype) !Arr {
        return self.cast(try self.own(mlx.mlx_array_new_float(value)), dtype);
    }

    fn ones(self: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_ones(out, shape.ptr, shape.len, dtype, self.s));
        return out.*;
    }

    fn zeros(self: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_zeros(out, shape.ptr, shape.len, dtype, self.s));
        return out.*;
    }

    const Binary = enum { add, sub, mul, div, min, max, mm, less, le, land, lor, floor_div };
    fn binary(self: *Ops, comptime operation: Binary, a: Arr, b: Arr) !Arr {
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
    fn unary(self: *Ops, comptime operation: Unary, x: Arr) !Arr {
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

    fn reduce(self: *Ops, x: Arr, axis: c_int, mean: bool, keep: bool) !Arr {
        const out = try self.slot();
        if (mean) try mlx.check(mlx.mlx_mean_axis(out, x, axis, keep, self.s)) else try mlx.check(mlx.mlx_sum_axis(out, x, axis, keep, self.s));
        return out.*;
    }

    fn rms(self: *Ops, x: Arr, weight: Arr, eps: f32) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_fast_rms_norm(out, x, weight, eps, self.s));
        return out.*;
    }

    fn layerNorm(self: *Ops, x: Arr, weight: Arr, bias: Arr, eps: f32) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_fast_layer_norm(out, x, weight, bias, eps, self.s));
        return out.*;
    }

    fn softmax(self: *Ops, x: Arr, axis: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_softmax_axis(out, x, axis, true, self.s));
        return out.*;
    }

    fn take(self: *Ops, x: Arr, indices: Arr, axis: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_take_axis(out, x, indices, axis, self.s));
        return out.*;
    }

    fn broadcast(self: *Ops, x: Arr, shape: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_broadcast_to(out, x, shape.ptr, shape.len, self.s));
        return out.*;
    }

    fn qmm(self: *Ops, x: Arr, w: Arr, scales: Arr, biases: Arr, transposed: bool) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_quantized_matmul(out, x, w, scales, biases, transposed, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", self.s));
        return out.*;
    }

    fn dequant(self: *Ops, w: Arr, scales: Arr, biases: Arr) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_dequantize(out, w, scales, biases, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype.some(.bfloat16), self.s));
        return out.*;
    }

    fn silu(self: *Ops, x: Arr) !Arr {
        const f = try self.cast(x, .float32);
        const product = try self.binary(.mul, f, try self.unary(.sigmoid, f));
        return self.cast(product, mlx.mlx_array_dtype(x));
    }

    fn conv(self: *Ops, x: Arr, weight: Arr, groups: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_conv1d(out, x, weight, 1, 0, 1, groups, self.s));
        return out.*;
    }

    fn result(_: *Ops, x: Arr) !Arr {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_set(&out, x));
        return out;
    }
};

const Linear = struct {
    w: Arr,
    scales: Arr = .{ .ctx = null },
    biases: Arr = .{ .ctx = null },
    input: c_int,
    output: c_int,

    fn load(weights: *const model.Weights, base: []const u8, input: u32) !Linear {
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

    fn apply(self: Linear, ops: *Ops, x: Arr) !Arr {
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

const DenseMlp = struct {
    gate: Linear,
    up: Linear,
    down: Linear,

    fn load(weights: *const model.Weights, prefix: []const u8, hidden: u32, intermediate: u32) !DenseMlp {
        return .{
            .gate = try projection(weights, prefix, "gate_proj", hidden, intermediate),
            .up = try projection(weights, prefix, "up_proj", hidden, intermediate),
            .down = try projection(weights, prefix, "down_proj", intermediate, hidden),
        };
    }

    fn apply(self: DenseMlp, ops: *Ops, x: Arr, limit: f32) !Arr {
        const gate = try self.gate.apply(ops, x);
        const up = try self.up.apply(ops, x);
        const hi = try ops.scalar(limit, mlx.mlx_array_dtype(gate));
        const lo = try ops.scalar(-limit, mlx.mlx_array_dtype(up));
        const cg = try ops.binary(.min, gate, hi);
        const cu = try ops.binary(.max, try ops.binary(.min, up, hi), lo);
        return self.down.apply(ops, try ops.binary(.mul, try ops.silu(cg), cu));
    }
};

const Hc = struct {
    w: Arr,
    scale: Arr,
    base: Arr,

    fn load(weights: *const model.Weights, prefix: []const u8, label: []const u8, hidden: u32) !Hc {
        var buf: [256]u8 = undefined;
        const value = Hc{
            .w = try named(weights, prefix, try std.fmt.bufPrint(&buf, "{s}_fn", .{label})),
            .scale = try named(weights, prefix, try std.fmt.bufPrint(&buf, "{s}_scale", .{label})),
            .base = try named(weights, prefix, try std.fmt.bufPrint(&buf, "{s}_base", .{label})),
        };
        if (!std.mem.eql(c_int, &.{ 24, @intCast(hidden * 4) }, mlx.getShape(value.w)) or
            mlx.mlx_array_size(value.scale) != 3 or mlx.mlx_array_size(value.base) != 24 or
            mlx.mlx_array_dtype(value.w) != .float32 or mlx.mlx_array_dtype(value.scale) != .float32 or mlx.mlx_array_dtype(value.base) != .float32) return error.InvalidGlmHc;
        return value;
    }

    fn collapse(self: Hc, ops: *Ops, x: Arr, cfg: *const model.ModelConfig) !primitive.HcResult {
        const sh = mlx.getShape(x);
        const flat = try ops.reshape(try ops.cast(x, .float32), &.{ sh[0], sh[1], sh[2] * sh[3] });
        const normalized = try ops.rms(flat, .{ .ctx = null }, cfg.rms_norm_eps);
        const mixes = try ops.binary(.mm, normalized, try ops.transpose(try ops.cast(self.w, .float32), &.{ 1, 0 }));
        return primitive.hcCollapse(x, mixes, try ops.cast(self.scale, .float32), try ops.cast(self.base, .float32), @intCast(cfg.glm_hc_sinkhorn_iters), cfg.glm_hc_eps, ops.s);
    }
};

fn compactConvTail(ops: *Ops, input: Arr, rows: c_int, keep: c_int) !Arr {
    const tail = try ops.slice(input, 1, rows, rows + keep);
    // A batch-one tail is already contiguous but still owns its whole prompt parent.
    if (rows > 1) return ops.own(try @import("transformer.zig").materializedOwnedCopy(ops.s, tail));
    return ops.contiguous(tail);
}

const KdaLayer = struct {
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

    fn load(weights: *const model.Weights, prefix: []const u8, cfg: *const model.ModelConfig) !KdaLayer {
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

    fn apply(self: KdaLayer, ops: *Ops, x: Arr, cfg: *const model.ModelConfig, state: *@import("transformer.zig").SSMCacheEntry) !Arr {
        const sh = mlx.getShape(x);
        const heads: c_int = @intCast(cfg.linear_num_value_heads);
        const dim: c_int = @intCast(cfg.linear_key_head_dim);
        const width = heads * dim;
        const keep: c_int = @intCast(cfg.linear_conv_kernel_dim - 1);
        const projections = [_]Arr{ try self.q.apply(ops, x), try self.k.apply(ops, x), try self.v.apply(ops, x) };
        const joined = try ops.concat(&projections, -1);
        const previous = if (state.initialized) state.conv_state else try ops.zeros(&.{ sh[0], keep, width * 3 }, mlx.mlx_array_dtype(x));
        const conv_input = try ops.concat(&.{ previous, joined }, 1);
        const conv_weight = try ops.contiguous(try ops.transpose(try ops.concat(&.{ self.conv_q, self.conv_k, self.conv_v }, 0), &.{ 0, 2, 1 }));
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
        const magnitude = try ops.reshape(try ops.unary(.exp, try ops.cast(self.a_log, .float32)), &.{ 1, 1, heads, 1 });
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
        const y = try ops.cast(result.y, .float32);
        const variance = try ops.reduce(try ops.binary(.mul, y, y), -1, true, true);
        const normalization = try ops.unary(.rsqrt, try ops.binary(.add, variance, try ops.scalar(cfg.rms_norm_eps, .float32)));
        const normalized = try ops.binary(.mul, try ops.binary(.mul, y, normalization), try ops.cast(self.out_norm, .float32));
        const gate = try ops.reshape(try ops.cast(try self.gb.apply(ops, try self.ga.apply(ops, x)), .float32), &dims);
        const gated = try ops.cast(try ops.binary(.mul, normalized, try ops.unary(.sigmoid, gate)), mlx.mlx_array_dtype(x));
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

test "GLM HC loader refuses rounded FP32 coefficients" {
    var weights = model.Weights.init(std.testing.allocator);
    defer weights.deinit();
    try putValidationWeight(&weights, "a.hc_attn_fn", &.{ 24, 512 }, .bfloat16);
    try putValidationWeight(&weights, "a.hc_attn_scale", &.{3}, .float32);
    try putValidationWeight(&weights, "a.hc_attn_base", &.{24}, .float32);
    try std.testing.expectError(error.InvalidGlmHc, Hc.load(&weights, "a", "hc_attn", 128));
}
