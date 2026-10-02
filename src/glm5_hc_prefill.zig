//! Experimental exact multi-output HC prefill dot, with optional exact RMS fusion.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const SOURCE: [:0]const u8 =
    \\const uint row=threadgroup_position_in_grid.y;
    \\const uint first=threadgroup_position_in_grid.x*uint(COLS);
    \\const uint tid=thread_position_in_threadgroup.x;
    \\threadgroup float partial[4*COLS];
    \\float values[COLS];
    \\for(uint o=0;o<uint(COLS);++o)values[o]=0.0f;
    \\for(uint d=tid;d<uint(WIDTH);d+=128u) {
    \\ const float v=x[row*uint(WIDTH)+d];
    \\ for(uint o=0;o<uint(COLS);++o)values[o]+=v*float(w[(first+o)*uint(WIDTH)+d]);
    \\}
    \\for(uint o=0;o<uint(COLS);++o) {
    \\ values[o]=simd_sum(values[o]);
    \\ if(thread_index_in_simdgroup==0)partial[o*4u+simdgroup_index_in_threadgroup]=values[o];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(tid==0)for(uint o=0;o<uint(COLS);++o)out[row*24u+first+o]=(partial[o*4u]+partial[o*4u+1u])+(partial[o*4u+2u]+partial[o*4u+3u]);
;
// RMS mapping reused from glm5_hc_fused; MLX MIT and oMLX Apache-2.0 attribution in NOTICE.
const RMS_SOURCE: [:0]const u8 =
    \\const uint row=threadgroup_position_in_grid.y;
    \\const uint first=threadgroup_position_in_grid.x*uint(COLS);
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    \\threadgroup float sums[32];
    \\threadgroup float inverse[1];
    \\for(uint virtual_group=0;virtual_group<8;++virtual_group) {
    \\  const uint logical=virtual_group*128u+tid;
    \\  float acc=0.0f;
    \\  for(uint r=0;r<16384u;r+=4096u) {
    \\    for(uint i=0;i<4u;++i) {float xi=float(x[row*16384u+r+logical*4u+i]);acc+=xi*xi;}
    \\  }
    \\  acc=simd_sum(acc);
    \\  if(lane==0) sums[virtual_group*4u+sg]=acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(sg==0) {float acc=simd_sum(sums[lane]);if(lane==0) inverse[0]=metal::precise::rsqrt(acc/16384.0f+float(eps));}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\threadgroup float partial[4*COLS];
    \\float values[COLS];
    \\for(uint o=0;o<uint(COLS);++o)values[o]=0.0f;
    \\for(uint d=tid;d<uint(WIDTH);d+=128u) {
    \\ volatile float v=float(x[row*uint(WIDTH)+d])*inverse[0];
    \\ for(uint o=0;o<uint(COLS);++o)values[o]+=v*float(w[(first+o)*uint(WIDTH)+d]);
    \\}
    \\for(uint o=0;o<uint(COLS);++o) {
    \\ values[o]=simd_sum(values[o]);
    \\ if(thread_index_in_simdgroup==0)partial[o*4u+simdgroup_index_in_threadgroup]=values[o];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(tid==0)for(uint o=0;o<uint(COLS);++o)out[row*24u+first+o]=(partial[o*4u]+partial[o*4u+1u])+(partial[o*4u+2u]+partial[o*4u+3u]);
;
const REFERENCE: [:0]const u8 =
    \\const uint group=threadgroup_position_in_grid.x;
    \\const uint row=group/24u,output=group%24u,tid=thread_position_in_threadgroup.x;
    \\threadgroup float partial[4];
    \\float value=0.0f;
    \\for(uint d=tid;d<uint(WIDTH);d+=128u)value+=x[row*uint(WIDTH)+d]*float(w[output*uint(WIDTH)+d]);
    \\value=simd_sum(value);
    \\if(thread_index_in_simdgroup==0)partial[simdgroup_index_in_threadgroup]=value;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(tid==0)out[row*24u+output]=(partial[0]+partial[1])+(partial[2]+partial[3]);
;
var calls: usize = 0;
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}

var kernels: [3]?mlx.mlx_fast_metal_kernel = @splat(null);
fn kernel(index: usize) !mlx.mlx_fast_metal_kernel {
    if (kernels[index]) |k| return k;
    const iv = if (index == 2) mlx.mlx_vector_string_new_data(&.{ "x", "w", "eps" }, 3) else mlx.mlx_vector_string_new_data(&.{ "x", "w" }, 2);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(if (index == 2) "sushi_glm_hc_rms_multi_output" else if (index == 1) "sushi_glm_hc_multi_output" else "sushi_glm_hc_single_output_reference", iv, ov, if (index == 2) RMS_SOURCE else if (index == 1) SOURCE else REFERENCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[index] = k;
    return k;
}
fn run(s: mlx.mlx_stream, x: Arr, w: Arr, cols: c_int) !Arr {
    return runMode(s, x, w, cols, null);
}
fn runMode(s: mlx.mlx_stream, x: Arr, w: Arr, cols: c_int, epsilon: ?f32) !Arr {
    const sh = mlx.getShape(x);
    const rows = sh[1];
    const width: c_int = if (epsilon != null) 16384 else sh[2];
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, 24 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIDTH", width));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "COLS", cols));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, if (cols == 1) rows * 24 * 128 else @divExact(24, cols) * 128, if (cols == 1) 1 else rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    const eps = mlx.mlx_array_new_float(epsilon orelse 0);
    defer _ = mlx.mlx_array_free(eps);
    const iv = if (epsilon != null) mlx.mlx_vector_array_new_data(&.{ x, w, eps }, 3) else mlx.mlx_vector_array_new_data(&.{ x, w }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(if (epsilon != null) 2 else @intFromBool(cols != 1)), iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}
pub fn experimentalMix(s: mlx.mlx_stream, x: Arr, w: Arr, cols: c_int) !?Arr {
    if (!mlx.streamIsGpu(s) or x.ctx == null or w.ctx == null or mlx.mlx_array_dtype(x) != .float32 or
        (mlx.mlx_array_dtype(w) != .float32 and mlx.mlx_array_dtype(w) != .bfloat16)) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 16 or sh[1] > @divTrunc(std.math.maxInt(c_int), 3072) or sh[2] < 128 or sh[2] > 16384 or @rem(sh[2], 128) != 0 or
        !std.mem.eql(c_int, &.{ 24, sh[2] }, mlx.getShape(w)) or (cols != 2 and cols != 4 and cols != 8 and cols != 24)) return null;
    if (@as(u64, @intCast(sh[1])) * @as(u64, @intCast(sh[2])) > std.math.maxInt(u32)) return null;
    const out = try run(s, x, w, cols);
    calls += 1;
    return out;
}
pub fn experimentalRmsMix(s: mlx.mlx_stream, x: Arr, w: Arr, epsilon: f32, cols: c_int) !?Arr {
    if (!mlx.streamIsGpu(s) or x.ctx == null or w.ctx == null or !std.math.isFinite(epsilon) or epsilon <= 0 or mlx.mlx_array_dtype(x) != .bfloat16 or
        (mlx.mlx_array_dtype(w) != .float32 and mlx.mlx_array_dtype(w) != .bfloat16)) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 4 or sh[0] != 1 or sh[1] < 128 or sh[1] > 65535 or sh[2] != 4 or sh[3] != 4096 or
        !std.mem.eql(c_int, &.{ 24, 16384 }, mlx.getShape(w)) or (cols != 8 and cols != 24)) return null;
    const out = try runMode(s, x, w, cols, epsilon);
    calls += 1;
    return out;
}
fn expectBits(a: Arr, b: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    try std.testing.expectEqual(n, mlx.mlx_array_size(b));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n]));
}

test "GLM HC multi-output dot preserves original FP32 reduction bits" {
    const s = mlx.gpuStream();
    const alloc = std.testing.allocator;
    var random = std.Random.DefaultPrng.init(71357);
    const rng = random.random();
    for ([_]c_int{ 128, 512, 16384 }) |width| {
        const weights = try alloc.alloc(f32, @as(usize, @intCast(width)) * 24);
        defer alloc.free(weights);
        for (weights) |*v| v.* = (rng.float(f32) - 0.5) * 0.0625;
        const wf = mlx.mlx_array_new_data(weights.ptr, &[_]c_int{ 24, width }, 2, .float32);
        defer _ = mlx.mlx_array_free(wf);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, wf, .bfloat16, s));
        for ([_]c_int{ 17, 128, 512 }) |rows| {
            const input = try alloc.alloc(f32, @as(usize, @intCast(rows)) * @as(usize, @intCast(width)));
            defer alloc.free(input);
            for (input, 0..) |*v, i| v.* = if (i % 97 == 0) 0 else (rng.float(f32) - 0.5) * 4;
            const x = mlx.mlx_array_new_data(input.ptr, &[_]c_int{ 1, rows, width }, 3, .float32);
            defer _ = mlx.mlx_array_free(x);
            for ([_]Arr{ wf, wb }) |w| {
                const reference = try run(s, x, w, 1);
                defer _ = mlx.mlx_array_free(reference);
                for ([_]c_int{ 2, 4, 8, 24 }) |cols| {
                    const got = (try experimentalMix(s, x, w, cols)) orelse return error.UnsupportedHcMultiOutput;
                    defer _ = mlx.mlx_array_free(got);
                    expectBits(reference, got) catch |err| {
                        std.debug.print("HCdot mismatch width={d} rows={d} cols={d} weightdtype={}\\n", .{ width, rows, cols, mlx.mlx_array_dtype(w) });
                        return err;
                    };
                }
            }
        }
    }
}

test "GLM HC multi-output keeps cancellation and magnitude behavior unchanged" {
    const s = mlx.gpuStream();
    const alloc = std.testing.allocator;
    const width = 16384;
    const rows = 17;
    const data = try alloc.alloc(f32, rows * width);
    defer alloc.free(data);
    const wb = try alloc.alloc(u16, 24 * width);
    defer alloc.free(wb);
    for (wb, 0..) |*w, i| {
        const v: f32 = if (i % width < width / 2) 128 else -128;
        w.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
    for (0..24) |o| {
        wb[o * width + width / 2 - 1] = 0;
        wb[o * width + width - 1] = 0x3b00;
    }
    const w = mlx.mlx_array_new_data(wb.ptr, &[_]c_int{ 24, width }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    for ([_]f32{ 0, 0.0009765625, 0.9999949932098389, 128 }) |v| {
        for (data, 0..) |*x, i| x.* = if (i / width % 2 == 0) v else -v;
        const x = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ 1, rows, width }, 3, .float32);
        defer _ = mlx.mlx_array_free(x);
        const expected = try run(s, x, w, 1);
        defer _ = mlx.mlx_array_free(expected);
        for ([_]c_int{ 2, 4, 8, 24 }) |cols| {
            const actual = (try experimentalMix(s, x, w, cols)).?;
            defer _ = mlx.mlx_array_free(actual);
            try expectBits(expected, actual);
        }
    }
    const one = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ 1, 1, width }, 3, .float32);
    defer _ = mlx.mlx_array_free(one);
    try std.testing.expect((try experimentalMix(s, one, w, 4)) == null);
}

fn fixtureValue(map: mlx.mlx_map_string_to_array, key: [:0]const u8) !Arr {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_map_string_to_array_get(&a, map, key));
    return a;
}
fn timed(s: mlx.mlx_stream, x: Arr, w: Arr, cols: c_int) !u64 {
    var timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const out = try run(s, x, w, cols);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return timer.read();
}
fn timedMode(s: mlx.mlx_stream, x: Arr, w: Arr, cols: c_int, epsilon: ?f32) !u64 {
    if (epsilon == null) return timed(s, x, w, cols);
    var timer = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const out = if (cols == 1) try staged(s, x, w, epsilon.?) else (try experimentalRmsMix(s, x, w, epsilon.?, cols)).?;
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return timer.read();
}
fn staged(s: mlx.mlx_stream, x: Arr, w: Arr, epsilon: f32) !Arr {
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_astype(&wide, x, .float32, s));
    try mlx.check(mlx.mlx_reshape(&flat, wide, &.{ 1, mlx.getShape(x)[1], 16384 }, 3, s));
    try mlx.check(mlx.mlx_fast_rms_norm(&norm, flat, .{ .ctx = null }, epsilon, s));
    return run(s, norm, w, 1);
}
test "GLM HC multi-output actual fixture parity and paired timing" {
    const file = std.c.getenv("SUSHI_GLM_HC_PREFILL_FIXTURE") orelse return error.SkipZigTest;
    const out = std.c.getenv("SUSHI_GLM_HC_PREFILL_REPORT") orelse return error.MissingHcReport;
    const rms_mode = std.c.getenv("SUSHI_GLM_HC_PREFILL_RMS") != null;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&tensors, &metadata, file, cpu));
    const params = try fixtureValue(tensors, "rms_eps_hc_eps_sinkhorn_iters");
    defer _ = mlx.mlx_array_free(params);
    try mlx.check(mlx.mlx_array_eval(params));
    const eps = mlx.mlx_array_data_float32(params).?[0];
    const Record = struct { layer: u32, ffn: bool, rows: c_int, cols: c_int, reference_ns: [24]u64, candidate_ns: [24]u64 };
    var records: [32]Record = undefined;
    var count: usize = 0;
    for ([_]u32{ 0, 3, 23, 44 }) |layer| for ([_]bool{ false, true }) |ffn| {
        var xb: [80]u8 = undefined;
        var wb: [80]u8 = undefined;
        var mb: [80]u8 = undefined;
        const kind = if (ffn) "ffn" else "attn";
        var x = try fixtureValue(tensors, try std.fmt.bufPrintSentinel(&xb, "layer{d:0>2}.{s}.x", .{ layer, kind }, 0));
        defer _ = mlx.mlx_array_free(x);
        const w = try fixtureValue(tensors, try std.fmt.bufPrintSentinel(&wb, "layer{d:0>2}.{s}.w", .{ layer, kind }, 0));
        defer _ = mlx.mlx_array_free(w);
        var expected = try fixtureValue(tensors, try std.fmt.bufPrintSentinel(&mb, "layer{d:0>2}.{s}.mix", .{ layer, kind }, 0));
        defer _ = mlx.mlx_array_free(expected);
        if (std.c.getenv("SUSHI_GLM_HC_PREFILL_ROWS128") != null and mlx.getShape(x)[1] >= 128) {
            var slice = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&slice, x, &.{ 0, 0, 0, 0 }, 4, &.{ 1, 128, 4, 4096 }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            _ = mlx.mlx_array_free(x);
            x = slice;
            slice = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&slice, expected, &.{ 0, 0, 0 }, 3, &.{ 1, 128, 24 }, 3, &.{ 1, 1, 1 }, 3, s));
            _ = mlx.mlx_array_free(expected);
            expected = slice;
        }
        const sh = mlx.getShape(x);
        if (sh.len != 4 or sh[0] != 1 or sh[2] != 4 or sh[3] != 4096 or mlx.mlx_array_dtype(x) != .bfloat16 or
            !std.mem.eql(c_int, &.{ 24, 16384 }, mlx.getShape(w)) or !std.mem.eql(c_int, &.{ 1, sh[1], 24 }, mlx.getShape(expected))) return error.InvalidHcFixture;
        const rows = sh[1];
        var wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wide);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        var normalized = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normalized);
        try mlx.check(mlx.mlx_astype(&wide, x, .float32, s));
        try mlx.check(mlx.mlx_reshape(&flat, wide, &.{ 1, rows, sh[2] * sh[3] }, 3, s));
        try mlx.check(mlx.mlx_fast_rms_norm(&normalized, flat, .{ .ctx = null }, eps, s));
        const old = try run(s, normalized, w, 1);
        defer _ = mlx.mlx_array_free(old);
        try expectBits(expected, old);
        for ([_]c_int{ 2, 4, 8, 24 }) |cols| {
            if (rms_mode and cols < 8) continue;
            if (rows < 16) {
                try std.testing.expect((try experimentalMix(s, normalized, w, cols)) == null);
                continue;
            }
            const check = if (rms_mode) (try experimentalRmsMix(s, x, w, eps, cols)).? else (try experimentalMix(s, normalized, w, cols)).?;
            defer _ = mlx.mlx_array_free(check);
            try expectBits(expected, check);
            for (0..4) |_| {
                _ = try timedMode(s, if (rms_mode) x else normalized, w, 1, if (rms_mode) eps else null);
                _ = try timedMode(s, if (rms_mode) x else normalized, w, cols, if (rms_mode) eps else null);
            }
            var record = Record{ .layer = layer, .ffn = ffn, .rows = rows, .cols = cols, .reference_ns = undefined, .candidate_ns = undefined };
            for (0..24) |i| {
                if (i % 2 == 0) {
                    record.reference_ns[i] = try timedMode(s, if (rms_mode) x else normalized, w, 1, if (rms_mode) eps else null);
                    record.candidate_ns[i] = try timedMode(s, if (rms_mode) x else normalized, w, cols, if (rms_mode) eps else null);
                } else {
                    record.candidate_ns[i] = try timedMode(s, if (rms_mode) x else normalized, w, cols, if (rms_mode) eps else null);
                    record.reference_ns[i] = try timedMode(s, if (rms_mode) x else normalized, w, 1, if (rms_mode) eps else null);
                }
            }
            records[count] = record;
            count += 1;
        }
    };
    const json = try std.json.Stringify.valueAlloc(alloc, .{ .method = if (rms_mode) "captured raw BF16 inputs, RMS+dot, 4warm/24ABBApairs, all outputs bit-exact" else "captured normalized FP32 inputs, dot only, 4warm/24ABBApairs, all outputs bit-exact", .records = records[0..count] }, .{ .whitespace = .indent_2 });
    defer alloc.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(out), .data = json });
}

test "GLM HC multi-output RMS preserves staged bits and rejects small rows" {
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    const weights = try a.alloc(f32, 24 * 16384);
    defer a.free(weights);
    for (weights, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 41)) - 20)) / 128;
    const wf = mlx.mlx_array_new_data(weights.ptr, &.{ 24, 16384 }, 2, .float32);
    defer _ = mlx.mlx_array_free(wf);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, wf, .bfloat16, s));
    for ([_]c_int{ 128, 512 }) |rows| {
        const bits = try a.alloc(u16, @as(usize, @intCast(rows)) * 16384);
        defer a.free(bits);
        for ([_]f32{ 0, 0.0001, 1, 128 }) |magnitude| {
            for (bits, 0..) |*v, i| {
                const f = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 101)) - 50)) * magnitude;
                v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
            }
            const x = mlx.mlx_array_new_data(bits.ptr, &.{ 1, rows, 4, 4096 }, 4, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            for ([_]Arr{ wf, wb }) |w| for ([_]f32{ 1e-6, 0.1 }) |eps| {
                const expected = try staged(s, x, w, eps);
                defer _ = mlx.mlx_array_free(expected);
                for ([_]c_int{ 8, 24 }) |cols| {
                    const actual = (try experimentalRmsMix(s, x, w, eps, cols)).?;
                    defer _ = mlx.mlx_array_free(actual);
                    try expectBits(expected, actual);
                }
            };
        }
        const small = mlx.mlx_array_new_data(bits.ptr, &.{ 1, 17, 4, 4096 }, 4, .bfloat16);
        defer _ = mlx.mlx_array_free(small);
        try std.testing.expect((try experimentalRmsMix(s, small, wb, 1e-6, 24)) == null);
    }
    try std.testing.expect((try experimentalRmsMix(s, .{ .ctx = null }, wb, 1e-6, 24)) == null);
    try std.testing.expect((try experimentalRmsMix(s, wf, wb, 0, 24)) == null);
}
