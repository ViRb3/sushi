//! Vector-gated KDA recurrence over parent-indexed verification rows.
const std = @import("std");
const mlx = @import("mlx.zig");
const primitive = @import("glm5_next.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

const SOURCE =
    \\constexpr int N = Dk / 32;
    \\const int head = int(thread_position_in_grid.z);
    \\const int dv = int(thread_position_in_grid.y);
    \\const int lane = int(thread_position_in_threadgroup.x);
    \\float saved[W][N];
    \\for (int row = 0; row < W; ++row) {
    \\  const int parent = parents[row];
    \\  const device InT* q_ = q + (row * H + head) * Dk;
    \\  const device InT* k_ = k + (row * H + head) * Dk;
    \\  float state[N];
    \\  float memory = 0.0f;
    \\  for (int i = 0; i < N; ++i) {
    \\    const int key = N * lane + i;
    \\    state[i] = parent < 0 ? state_in[(head * Dv + dv) * Dk + key] : saved[parent][i];
    \\    state[i] = state[i] * decay[(row * H + head) * Dk + key];
    \\    memory += state[i] * k_[key];
    \\  }
    \\  memory = simd_sum(memory);
    \\  const auto delta = (v[(row * H + head) * Dv + dv] - memory) * beta[row * H + head];
    \\  float output = 0.0f;
    \\  for (int i = 0; i < N; ++i) {
    \\    const int key = N * lane + i;
    \\    state[i] = state[i] + k_[key] * delta;
    \\    output += state[i] * q_[key];
    \\    saved[row][i] = state[i];
    \\  }
    \\  output = simd_sum(output);
    \\  if (thread_index_in_simdgroup == 0) y[(row * H + head) * Dv + dv] = OutT(output);
    \\}
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;

pub fn recurrent(input: primitive.KdaInputs, parents: []const i32, stream: mlx.mlx_stream) !Arr {
    if (!mlx.streamIsGpu(stream)) return error.KdaGpuRequired;
    const q = mlx.getShape(input.q);
    const v = mlx.getShape(input.v);
    if (q.len != 4 or v.len != 4 or q[0] != 1 or q[1] < 1 or q[1] > 16 or q[2] < 1 or q[3] < 32 or @mod(q[3], 32) != 0 or v[3] < 4 or @mod(v[3], 4) != 0 or
        !std.mem.eql(c_int, q[0..3], v[0..3]) or !std.mem.eql(c_int, q, mlx.getShape(input.k)) or !std.mem.eql(c_int, q, mlx.getShape(input.decay)) or
        !std.mem.eql(c_int, q[0..3], mlx.getShape(input.beta)) or !std.mem.eql(c_int, &.{ 1, q[2], v[3], q[3] }, mlx.getShape(input.state)) or parents.len != q[1]) return error.InvalidKdaShape;
    const dtype = mlx.mlx_array_dtype(input.q);
    const beta_type = mlx.mlx_array_dtype(input.beta);
    if ((dtype != .bfloat16 and dtype != .float32) or mlx.mlx_array_dtype(input.k) != dtype or mlx.mlx_array_dtype(input.v) != dtype or
        mlx.mlx_array_dtype(input.state) != .float32 or mlx.mlx_array_dtype(input.decay) != .float32 or (beta_type != .bfloat16 and beta_type != .float32)) return error.InvalidKdaDtype;
    if (parents[0] != -1) return error.InvalidGlmDraftTree;
    for (parents[1..], 1..) |parent, row| if (parent < 0 or parent >= row) return error.InvalidGlmDraftTree;
    if (kernel == null) {
        const names = [_][*:0]const u8{ "q", "k", "v", "decay", "beta", "state_in", "parents" };
        const outputs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        kernel = mlx.mlx_fast_metal_kernel_new("sushi_glm_kda_tree", iv, ov, SOURCE, "", true, false);
        if (kernel.?.ctx == null) {
            kernel = null;
            return error.MetalKernelCompileFailed;
        }
    }
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, v.ptr, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, v[3], q[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", q[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", v[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "H", q[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "W", q[1]));
    const par = mlx.mlx_array_new_data(parents.ptr, &[_]c_int{q[1]}, 1, .int32);
    defer _ = mlx.mlx_array_free(par);
    const arrays = [_]Arr{ input.q, input.k, input.v, input.decay, input.beta, input.state, par };
    const inputs = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.?, inputs, config, stream));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, outputs, 0));
    return result;
}

test "GLM DFlash KDA tree follows per-channel parent state exactly" {
    const s = mlx.gpuStream();
    inline for (.{ .{ 5, 2, 4 }, .{ 16, 64, 128 } }) |geometry| {
        const R = geometry[0];
        const H = geometry[1];
        const DV = geometry[2];
        var parents: [R]i32 = undefined;
        parents[0] = -1;
        for (1..R) |row| parents[row] = @intCast((row - 1) / 2);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
            var ops = Ops{ .s = s };
            defer ops.deinit();
            var q_data: [R * H * 128]f32 = undefined;
            var k_data: @TypeOf(q_data) = undefined;
            var g_data: @TypeOf(q_data) = undefined;
            var v_data: [R * H * DV]f32 = undefined;
            var beta_data: [R * H]f32 = undefined;
            for (&q_data, &k_data, &g_data, 0..) |*q, *k, *g, i| {
                q.* = @as(f32, @floatFromInt(i % 13)) / 64 - 0.1;
                k.* = @as(f32, @floatFromInt(i % 17)) / 96 - 0.08;
                g.* = 0.1 + @as(f32, @floatFromInt(i % 29)) / 36;
            }
            for (&v_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) / 32 - 0.2;
            for (&beta_data, 0..) |*v, i| v.* = 0.1 + @as(f32, @floatFromInt(i % 13)) / 16;
            const q = try ops.cast(try ops.own(mlx.mlx_array_new_data(&q_data, &[_]c_int{ 1, R, H, 128 }, 4, .float32)), dtype);
            const k = try ops.cast(try ops.own(mlx.mlx_array_new_data(&k_data, &[_]c_int{ 1, R, H, 128 }, 4, .float32)), dtype);
            const v = try ops.cast(try ops.own(mlx.mlx_array_new_data(&v_data, &[_]c_int{ 1, R, H, DV }, 4, .float32)), dtype);
            const decay = try ops.own(mlx.mlx_array_new_data(&g_data, &[_]c_int{ 1, R, H, 128 }, 4, .float32));
            const beta = try ops.cast(try ops.own(mlx.mlx_array_new_data(&beta_data, &[_]c_int{ 1, R, H }, 3, .float32)), dtype);
            const initial = try ops.binary(.mul, try ops.ones(&.{ 1, H, DV, 128 }, .float32), try ops.scalar(0.07, .float32));
            const all = try ops.own(try recurrent(.{ .q = q, .k = k, .v = v, .decay = decay, .beta = beta, .state = initial }, &parents, s));
            var states: [R]Arr = undefined;
            for (parents, 0..) |parent, row| {
                const start: c_int = @intCast(row);
                const one = try primitive.kda(.{ .q = try ops.slice(q, 1, start, start + 1), .k = try ops.slice(k, 1, start, start + 1), .v = try ops.slice(v, 1, start, start + 1), .decay = try ops.slice(decay, 1, start, start + 1), .beta = try ops.slice(beta, 1, start, start + 1), .state = if (parent < 0) initial else states[@intCast(parent)] }, s);
                states[row] = try ops.own(one.state);
                const y = try ops.cast(try ops.own(one.y), .float32);
                const actual = try ops.cast(try ops.slice(all, 1, start, start + 1), .float32);
                try mlx.check(mlx.mlx_array_eval(y));
                try mlx.check(mlx.mlx_array_eval(actual));
                try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(y).?[0 .. H * DV], mlx.mlx_array_data_float32(actual).?[0 .. H * DV]);
            }
        }
    }
}

pub const ProjectionMode = enum { serial_rows, affine_rows, affine_rows_ffn, batched };

pub fn linearRows(ops: *Ops, linear: @import("glm5_model.zig").Linear, x: Arr, mode: ProjectionMode) !Arr {
    const shape = mlx.getShape(x);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > 16) return error.InvalidGlmDraftShape;
    if (mode == .batched or shape[1] == 1) return linear.apply(ops, x);
    if (mode == .affine_rows or mode == .affine_rows_ffn) if (try @import("glm5_dflash_qmm.zig").project(ops.s, x, linear)) |output| return ops.own(output);
    var rows: [16]Arr = undefined;
    var made: usize = 0;
    defer for (rows[0..made]) |value| {
        _ = mlx.mlx_array_free(value);
    };
    for (0..@intCast(shape[1])) |i| {
        var row = Ops{ .s = ops.s };
        defer row.deinit();
        rows[i] = try row.result(try linear.apply(&row, try row.slice(x, 1, @intCast(i), @intCast(i + 1))));
        made += 1;
    }
    return ops.concat(rows[0..made], 1);
}

pub const Tape = struct {
    inputs: primitive.KdaInputs,
    conv_input: Arr,
    parents: [16]i32 = @splat(-1),
    count: usize = 0,
    pub fn deinit(self: *Tape) void {
        for ([_]Arr{ self.inputs.q, self.inputs.k, self.inputs.v, self.inputs.decay, self.inputs.beta, self.inputs.state, self.conv_input }) |value| _ = mlx.mlx_array_free(value);
    }
    pub fn replay(self: *const Tape, path: []const u32, s: mlx.mlx_stream) !@import("transformer.zig").SSMCacheEntry {
        const rows = mlx.getShape(self.inputs.q)[1];
        if (path.len == 0 or path.len > 16 or path[0] != 0 or self.count != rows) return error.InvalidGlmDraftTree;
        for (path, 0..) |row, i| {
            if (row >= rows or (i > 0 and self.parents[row] != path[i - 1])) return error.InvalidGlmDraftTree;
        }
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const indices = try ops.own(mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .uint32));
        const output = try primitive.kda(.{ .q = try ops.take(self.inputs.q, indices, 1), .k = try ops.take(self.inputs.k, indices, 1), .v = try ops.take(self.inputs.v, indices, 1), .decay = try ops.take(self.inputs.decay, indices, 1), .beta = try ops.take(self.inputs.beta, indices, 1), .state = self.inputs.state }, s);
        defer output.deinit();
        var tail: [3]u32 = undefined;
        for (&tail, 0..) |*row, i| {
            const position = @as(i32, @intCast(path.len)) - 3 + @as(i32, @intCast(i));
            row.* = if (position < 0) @intCast(3 + position) else path[@intCast(position)] + 3;
        }
        const tail_ids = try ops.own(mlx.mlx_array_new_data(&tail, &[_]c_int{3}, 1, .uint32));
        const conv = try ops.result(try ops.contiguous(try ops.take(self.conv_input, tail_ids, 1)));
        errdefer _ = mlx.mlx_array_free(conv);
        return .{ .conv_state = conv, .ssm_state = try ops.result(output.state), .initialized = true };
    }
};

pub const LayerResult = struct { output: Arr, tape: Tape };

/// Projections may be tested in batches; the default mode preserves one-row projection geometry.
pub fn applyLayer(layer: @import("glm5_model.zig").KdaLayer, ops: *Ops, x: Arr, cfg: *const @import("model.zig").ModelConfig, state: *const @import("transformer.zig").SSMCacheEntry, parents: []const i32, mode: ProjectionMode) !LayerResult {
    const sh = mlx.getShape(x);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 1 or sh[1] > 16 or sh[1] != parents.len or cfg.linear_conv_kernel_dim != 4) return error.InvalidGlmDraftShape;
    if (parents[0] != -1) return error.InvalidGlmDraftTree;
    for (parents[1..], 1..) |parent, i| if (parent < 0 or parent >= i) return error.InvalidGlmDraftTree;
    const heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dim: c_int = @intCast(cfg.linear_key_head_dim);
    const width = heads * dim;
    const dtype = mlx.mlx_array_dtype(x);
    const qraw = try linearRows(ops, layer.q, x, mode);
    const kraw = try linearRows(ops, layer.k, x, mode);
    const vraw = try linearRows(ops, layer.v, x, mode);
    const raw = try ops.concat(&.{ qraw, kraw, vraw }, -1);
    const old = if (state.initialized) state.conv_state else try ops.zeros(&.{ 1, 3, width * 3 }, dtype);
    const conv_input = try ops.concat(&.{ old, raw }, 1);
    var window: [16 * 4]i32 = undefined;
    for (0..parents.len) |row| for (0..4) |j| {
        var back = 3 - j;
        var at: i32 = @intCast(row);
        while (back > 0 and at >= 0) {
            at = parents[@intCast(at)];
            back -= 1;
        }
        window[row * 4 + j] = if (at >= 0) 3 + at else 2 - @as(i32, @intCast(back));
    };
    const indices = try ops.own(mlx.mlx_array_new_data(&window, &[_]c_int{@intCast(parents.len * 4)}, 1, .int32));
    const conv_x = try ops.reshape(try ops.take(conv_input, indices, 1), &.{ sh[1], 4, width * 3 });
    const conv_w = if (layer.prepared_conv.ctx != null) layer.prepared_conv else try ops.contiguous(try ops.transpose(try ops.concat(&.{ layer.conv_q, layer.conv_k, layer.conv_v }, 0), &.{ 0, 2, 1 }));
    const convolved = try ops.reshape(try ops.silu(try ops.conv(conv_x, conv_w, width * 3)), &.{ 1, sh[1], width * 3 });
    const dims = [_]c_int{ 1, sh[1], heads, dim };
    const rq = try ops.cast(try ops.reshape(try ops.slice(convolved, 2, 0, width), &dims), .float32);
    const rk = try ops.cast(try ops.reshape(try ops.slice(convolved, 2, width, 2 * width), &dims), .float32);
    const values = try ops.reshape(try ops.slice(convolved, 2, 2 * width, 3 * width), &dims);
    const eps = try ops.scalar(1e-6, .float32);
    const qnorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, rq, rq), -1, false, true), eps));
    const knorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, rk, rk), -1, false, true), eps));
    const q = try ops.cast(try ops.binary(.mul, try ops.binary(.mul, rq, qnorm), try ops.scalar(1 / @sqrt(@as(f32, @floatFromInt(dim))), .float32)), dtype);
    const k = try ops.cast(try ops.binary(.mul, rk, knorm), dtype);
    const a = try ops.reshape(try ops.cast(try linearRows(ops, layer.fb, try linearRows(ops, layer.fa, x, mode), mode), .float32), &dims);
    const shift = try ops.reshape(try ops.cast(layer.dt_bias, .float32), &.{ 1, 1, heads, dim });
    const exp_decay = if (layer.prepared_decay.ctx != null) layer.prepared_decay else try ops.unary(.exp, layer.a_log);
    const magnitude = try ops.reshape(exp_decay, &.{ 1, 1, heads, 1 });
    const decay = try ops.unary(.exp, try ops.binary(.mul, try ops.unary(.sigmoid, try ops.binary(.mul, magnitude, try ops.binary(.add, a, shift))), try ops.scalar(cfg.kda_gate_lower_bound, .float32)));
    const beta = try ops.unary(.sigmoid, try linearRows(ops, layer.beta, x, mode));
    const initial = if (state.initialized) state.ssm_state else try ops.zeros(&.{ 1, heads, dim, dim }, .float32);
    const inputs = primitive.KdaInputs{ .q = q, .k = k, .v = values, .decay = decay, .beta = beta, .state = initial };
    const y = try ops.cast(try ops.own(try recurrent(inputs, parents, ops.s)), .float32);
    const variance = try ops.reduce(try ops.binary(.mul, y, y), -1, true, true);
    const normalization = try ops.unary(.rsqrt, try ops.binary(.add, variance, try ops.scalar(cfg.rms_norm_eps, .float32)));
    const normalized = try ops.binary(.mul, try ops.binary(.mul, y, normalization), try ops.cast(layer.out_norm, .float32));
    const gate = try ops.reshape(try ops.cast(try linearRows(ops, layer.gb, try linearRows(ops, layer.ga, x, mode), mode), .float32), &dims);
    const gated = try ops.cast(try ops.binary(.mul, normalized, try ops.unary(.sigmoid, gate)), dtype);
    const output = try linearRows(ops, layer.out, try ops.reshape(gated, &.{ 1, sh[1], width }), mode);
    var tape = Tape{ .inputs = .{ .q = .{ .ctx = null }, .k = .{ .ctx = null }, .v = .{ .ctx = null }, .decay = .{ .ctx = null }, .beta = .{ .ctx = null }, .state = .{ .ctx = null } }, .conv_input = .{ .ctx = null } };
    errdefer tape.deinit();
    inline for (.{ "q", "k", "v", "decay", "beta", "state" }) |name| @field(tape.inputs, name) = try ops.result(@field(inputs, name));
    tape.conv_input = try ops.result(conv_input);
    tape.count = parents.len;
    @memcpy(tape.parents[0..parents.len], parents);
    return .{ .output = output, .tape = tape };
}

test "GLM DFlash KDA layer tree and replay equal serial ancestor forwards" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    for ([_]u32{ 1, 64 }) |heads| {
        var weights = @import("model.zig").Weights.init(a);
        defer weights.deinit();
        var cfg = try @import("glm5_forward.zig").completeFixture(&weights);
        cfg.linear_num_value_heads = heads;
        var iter = weights.map.iterator();
        var seed: usize = 75;
        while (iter.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, "model.language_model.layers.0.self_attn.")) continue;
            const value = entry.value_ptr;
            const original_shape = mlx.getShape(value.*);
            const name = entry.key_ptr.*["model.language_model.layers.0.self_attn.".len..];
            var dimensions: [3]c_int = undefined;
            @memcpy(dimensions[0..original_shape.len], original_shape);
            const width: c_int = @intCast(heads * 128);
            if (std.mem.eql(u8, name, "A_log") or std.mem.eql(u8, name, "dt_bias")) {
                const replacement = mlx.mlx_array_new();
                var mutable = replacement;
                try mlx.check(mlx.mlx_zeros(&mutable, &[_]c_int{if (std.mem.eql(u8, name, "A_log")) @intCast(heads) else width}, 1, .float32, s));
                _ = mlx.mlx_array_free(value.*);
                value.* = mutable;
                continue;
            }
            if (original_shape.len < 2) continue;
            if (std.mem.eql(u8, name, "o_proj.weight")) dimensions[1] = width else if (std.mem.eql(u8, name, "b_proj.weight")) dimensions[0] = @intCast(heads) else if (!std.mem.eql(u8, name, "f_a_proj.weight") and !std.mem.eql(u8, name, "g_a_proj.weight")) dimensions[0] = width;
            const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(dimensions[0..original_shape.len], seed, s);
            _ = mlx.mlx_array_free(value.*);
            value.* = replacement;
            seed += 1;
        }
        var layer = try @import("glm5_model.zig").KdaLayer.load(&weights, "model.language_model.layers.0.self_attn", &cfg);
        defer layer.deinit();
        try layer.prepare(s);
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const x = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 5, 128 }, 37, s));
        const initial = @import("transformer.zig").SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer _ = mlx.mlx_array_free(initial.conv_state);
        defer _ = mlx.mlx_array_free(initial.ssm_state);
        const parents = [_]i32{ -1, 0, 0, 1, 2 };
        var all = try applyLayer(layer, &ops, x, &cfg, &initial, &parents, .serial_rows);
        defer all.tape.deinit();
        var states: [5]@import("transformer.zig").SSMCacheEntry = undefined;
        var made: usize = 0;
        defer for (states[0..made]) |state| {
            _ = mlx.mlx_array_free(state.conv_state);
            _ = mlx.mlx_array_free(state.ssm_state);
        };
        for (parents, 0..) |parent, row| {
            states[row] = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = parent >= 0 };
            made += 1;
            if (parent >= 0) {
                try mlx.check(mlx.mlx_array_set(&states[row].conv_state, states[@intCast(parent)].conv_state));
                try mlx.check(mlx.mlx_array_set(&states[row].ssm_state, states[@intCast(parent)].ssm_state));
            }
            const expected = try layer.apply(&ops, try ops.slice(x, 1, @intCast(row), @intCast(row + 1)), &cfg, &states[row]);
            try equalArray(expected, try ops.slice(all.output, 1, @intCast(row), @intCast(row + 1)), s);
        }
        try std.testing.expectError(error.InvalidGlmDraftTree, all.tape.replay(&.{ 0, 1, 4 }, s));
        const replay = try all.tape.replay(&.{ 0, 2, 4 }, s);
        defer _ = mlx.mlx_array_free(replay.conv_state);
        defer _ = mlx.mlx_array_free(replay.ssm_state);
        try equalArray(replay.conv_state, states[4].conv_state, s);
        try equalArray(replay.ssm_state, states[4].ssm_state, s);
    }
}

fn equalArray(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    const x = try @import("dflash.zig").TinyFix.readF32(a, std.testing.allocator, s);
    defer std.testing.allocator.free(x);
    const y = try @import("dflash.zig").TinyFix.readF32(b, std.testing.allocator, s);
    defer std.testing.allocator.free(y);
    try std.testing.expectEqualSlices(f32, x, y);
}
