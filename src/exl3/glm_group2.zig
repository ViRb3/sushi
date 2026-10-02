//! Unintegrated exact two-member reuse of GLM's cooperative F16 expert dot.
//! Routing IDs must be valid for the supplied expert bank, as in the baseline API.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const base = @import("expert_exl3_kernels.zig");
const support = base.Group2Support;
const Arr = mlx.mlx_array;
pub const Reduction = enum { parallel, serial };
pub const Layout = enum { natural, lane };

fn replace(comptime source: []const u8, comptime old: []const u8, comptime value: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const at = comptime std.mem.indexOf(u8, source, old) orelse return source ++ "";
    return source[0..at] ++ value ++ replace(source[at + old.len ..], old, value);
}
const MEMBERS =
    \\const uint first=uint(threadgroup_position_in_grid.y);
    \\const uint group_lane=uint(thread_index_in_simdgroup);
    \\const uint group_eid=uint(slots[first]);
    \\uint rank=0u, partner=first;
    \\for(uint b=0u;b<uint(NSLOTS);b+=32u) {
    \\ const bool hit=b+group_lane<uint(NSLOTS) && uint(slots[b+group_lane])==group_eid;
    \\ const uint mask=uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    \\ if(b+32u<=first) rank+=popcount(mask);
    \\ else if(b<=first) rank+=popcount(mask & ((1u<<(first-b))-1u));
    \\ const uint low=uint(max(int(first+1u)-int(b),0));
    \\ const uint after=low>=32u?0u:(mask & (0xffffffffu<<low));
    \\ if(partner==first && after!=0u) partner=b+ctz(after);
    \\}
    \\if((rank&1u)!=0u) return;
    \\threadgroup float partial[SERIAL_REDUCTION?1024:2048];
;
const EPILOGUE =
    \\if constexpr(SERIAL_REDUCTION) {
    \\ for(uint j=0u;j<2u;++j) {
    \\  for(uint si=0u;si<8u;++si) partial[sg*256u+pos[si]]=acc[j][si];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if(lid<16u) {
    \\   float sum=0.0f;
    \\   for(uint r=0u;r<16u;++r) for(uint g=0u;g<SGS;++g) sum+=partial[g*256u+r*16u+lid];
    \\   const uint member=j==0u?first:partner;
    \\   y[size_t(member)*uint(ODIM)+ot*TILE+lid]=half(sum);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ }
    \\} else {
    \\ for(uint j=0u;j<2u;++j) for(uint si=0u;si<8u;++si) partial[j*1024u+sg*256u+pos[si]]=acc[j][si];
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ if(lid<32u) {
    \\  const uint j=lid/16u,col=lid%16u;
    \\  float sum=0.0f;
    \\  for(uint r=0u;r<16u;++r) for(uint g=0u;g<SGS;++g) sum+=partial[j*1024u+g*256u+r*16u+col];
    \\  const uint member=j==0u?first:partner;
    \\  y[size_t(member)*uint(ODIM)+ot*TILE+col]=half(sum);
    \\ }
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;
const SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(1000000);
    const original = replace(support.cooperative_source, "threadgroup float partial[4 * 256];", "");
    const cut = std.mem.indexOf(u8, original, "for (uint si = 0u; si < 8u; si++)").?;
    var paired: []const u8 = original[0..cut];
    paired = replace(paired, "float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};", "float acc[2][8] = {};");
    paired = replace(paired, "const size_t xb = (size_t)slot * (size_t)(IDIM);", "const size_t xb = size_t(first)*uint(IDIM), xb1=size_t(partner)*uint(IDIM);");
    for (0..4) |r| {
        const old = std.fmt.comptimePrint("const float in{d} = float(x[xb + tk * TILE + row{d}]);", .{ r, r });
        const new = std.fmt.comptimePrint("const float2 in{d} = float2(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]));", .{ r, r, r });
        paired = replace(paired, old, new);
    }
    paired = replace(paired, "const float ins[8]", "const float2 ins[8]");
    paired = replace(paired, "acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);", "acc[0][p*2u]=fma(ins[p*2u].x,w.x,acc[0][p*2u]); acc[1][p*2u]=fma(ins[p*2u].y,w.x,acc[1][p*2u]);");
    paired = replace(paired, "acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);", "acc[0][p*2u+1u]=fma(ins[p*2u+1u].x,w.y,acc[0][p*2u+1u]); acc[1][p*2u+1u]=fma(ins[p*2u+1u].y,w.y,acc[1][p*2u+1u]);");
    break :blk MEMBERS ++ "\nif(partner==first) {\n" ++ original ++ "\n} else {\n" ++ paired ++ EPILOGUE ++ "\n}\n";
};
const PAIR_SOURCE: [:0]const u8 =
    \\const bool upper=threadgroup_position_in_grid.z!=0u;
    \\const device half* x=upper?xu:xg;
    \\const device ushort* trellis=upper?tu:tg;
    \\device half* y=upper?yu:yg;
++ replace(SOURCE, "uint split = uint(threadgroup_position_in_grid.z);", "uint split=0u;");
fn laneSource(comptime source: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    var result: []const u8 = source;
    const names = [_][]const u8{ "x", "y", "z", "w" };
    for (0..4) |r| {
        const single_old = std.fmt.comptimePrint("const float in{d} = float(x[xb + tk * TILE + row{d}]);", .{ r, r });
        const single_new = (if (r == 0) "const float4 in4=float4(*((const device half4*)(x+xb+tk*TILE+(lane&3u)*4u)));\n" else "") ++ std.fmt.comptimePrint("const float in{d}=in4.{s};", .{ r, names[r] });
        result = replace(result, single_old, single_new);
        const pair_old = std.fmt.comptimePrint("const float2 in{d} = float2(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]));", .{ r, r, r });
        const pair_new = (if (r == 0) "const float4 in4a=float4(*((const device half4*)(x+xb+tk*TILE+(lane&3u)*4u)));\nconst float4 in4b=float4(*((const device half4*)(x+xb1+tk*TILE+(lane&3u)*4u)));\n" else "") ++ std.fmt.comptimePrint("const float2 in{d}=float2(in4a.{s},in4b.{s});", .{ r, names[r], names[r] });
        result = replace(result, pair_old, pair_new);
    }
    return result ++ "";
}
const LANE_SOURCE = laneSource(SOURCE);
const LANE_PAIR_SOURCE = laneSource(PAIR_SOURCE);
var single_lane_kernels: support.Slots = support.empty;
var pair_lane_kernels: support.Slots = support.empty;
var single_kernels: support.Slots = support.empty;
var pair_kernels: support.Slots = support.empty;
const Key = struct { input: c_int, output: c_int, slots: c_int, rate: c_int, paired: bool, reduction: Reduction };
const Entry = struct { key: Key, config: mlx.mlx_fast_metal_kernel_config };
var cache: [64]?Entry = @splat(null);

fn config(key: Key) !struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool } {
    for (cache) |entry| if (entry) |e| if (std.meta.eql(e.key, key)) return .{ .value = e.config, .cached = true };
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sh = [_]c_int{ key.slots, key.output };
    for (0..if (key.paired) @as(usize, 2) else 1) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(key.output, 16) * 128, key.slots, if (key.paired) 2 else 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
    inline for (.{ "IDIM", "ODIM", "NSLOTS", "NHW", "SERIAL_REDUCTION" }, .{ key.input, key.output, key.slots, key.rate, @as(c_int, @intFromBool(key.reduction == .serial)) }) |name, v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
    for (&cache) |*entry| if (entry.* == null) {
        entry.* = .{ .key = key, .config = c };
        return .{ .value = c, .cached = true };
    };
    return .{ .value = c, .cached = false };
}
fn eligible(x: Arr, bank: Arr, ids: Arr) bool {
    if (x.ctx == null or bank.ctx == null or ids.ctx == null) return false;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(bank);
    const is = mlx.getShape(ids);
    return xs.len == 2 and ws.len == 4 and is.len == 1 and xs[0] == is[0] and xs[0] >= 2 and xs[0] <= 128 and
        xs[1] > 0 and @mod(xs[1], 128) == 0 and ws[0] > 0 and ws[1] == @divExact(xs[1], 16) and ws[2] > 0 and ws[2] <= @divTrunc(std.math.maxInt(c_int), 128) and @mod(ws[2], 8) == 0 and
        ws[3] >= 32 and ws[3] <= 64 and @mod(ws[3], 2) == 0 and mlx.mlx_array_dtype(x) == .float16 and mlx.mlx_array_dtype(bank) == .uint16 and
        (mlx.mlx_array_dtype(ids) == .uint32 or mlx.mlx_array_dtype(ids) == .int32);
}
pub fn project(s: mlx.mlx_stream, x: Arr, bank: Arr, ids: Arr, reduction: Reduction) !?Arr {
    return projectLayout(s, x, bank, ids, reduction, .natural);
}
pub fn projectLayout(s: mlx.mlx_stream, x: Arr, bank: Arr, ids: Arr, reduction: Reduction, layout: Layout) !?Arr {
    if (!mlx.streamIsGpu(s) or !eligible(x, bank, ids)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(bank);
    const cfg = try config(.{ .input = xs[1], .output = ws[2] * 16, .slots = xs[0], .rate = ws[3], .paired = false, .reduction = reduction });
    defer if (!cfg.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg.value);
    };
    const kernel = if (layout == .lane) try support.makeKernel(&single_lane_kernels, "sushi_glm_exl3_group2_lane", &.{ "x", "trellis", "slots" }, &.{"y"}, LANE_SOURCE) else try support.makeKernel(&single_kernels, "sushi_glm_exl3_group2", &.{ "x", "trellis", "slots" }, &.{"y"}, SOURCE);
    const iv = mlx.mlx_vector_array_new_data(&.{ x, bank, ids }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg.value, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}
pub fn pair(s: mlx.mlx_stream, xg: Arr, xu: Arr, tg: Arr, tu: Arr, ids: Arr, reduction: Reduction) !?[2]Arr {
    return pairLayout(s, xg, xu, tg, tu, ids, reduction, .natural);
}
pub fn pairLayout(s: mlx.mlx_stream, xg: Arr, xu: Arr, tg: Arr, tu: Arr, ids: Arr, reduction: Reduction, layout: Layout) !?[2]Arr {
    if (!mlx.streamIsGpu(s) or !eligible(xg, tg, ids) or !eligible(xu, tu, ids) or !std.mem.eql(c_int, mlx.getShape(tg), mlx.getShape(tu)) or !std.mem.eql(c_int, mlx.getShape(xg), mlx.getShape(xu))) return null;
    const xs = mlx.getShape(xg);
    const ws = mlx.getShape(tg);
    const cfg = try config(.{ .input = xs[1], .output = ws[2] * 16, .slots = xs[0], .rate = ws[3], .paired = true, .reduction = reduction });
    defer if (!cfg.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg.value);
    };
    const kernel = if (layout == .lane) try support.makeKernel(&pair_lane_kernels, "sushi_glm_exl3_pair_group2_lane", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, LANE_PAIR_SOURCE) else try support.makeKernel(&pair_kernels, "sushi_glm_exl3_pair_group2", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, PAIR_SOURCE);
    const iv = mlx.mlx_vector_array_new_data(&.{ xg, xu, tg, tu, ids }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg.value, s));
    var output = [2]Arr{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (output) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&output, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    return output;
}

pub const Down = enum { baseline, grouped };
/// Research-only full routed chain. Unsupported inputs return null for baseline fallback.
/// groupDown deliberately compares separate middle+grouped down with the native fused-middle path.
pub fn moe(s: mlx.mlx_stream, x: Arr, bank: @import("root.zig").Bank, indices: Arr, scores: Arr, dec: @import("expert_exl3.zig").Decode, reduction: Reduction, down: Down) !?Arr {
    return moeLayout(s, x, bank, indices, scores, dec, reduction, down, .natural);
}
pub fn moeLayout(s: mlx.mlx_stream, x: Arr, bank: @import("root.zig").Bank, indices: Arr, scores: Arr, dec: @import("expert_exl3.zig").Decode, reduction: Reduction, down: Down, layout: Layout) !?Arr {
    if (!mlx.streamIsGpu(s)) return null;
    for ([_]Arr{ x, indices, scores, bank.gate.trellis, bank.gate.suh, bank.gate.svh, bank.up.trellis, bank.up.suh, bank.up.svh, bank.down.trellis, bank.down.suh, bank.down.svh }) |value| if (value.ctx == null) return null;
    const shape = mlx.getShape(x);
    const ids_shape = mlx.getShape(indices);
    const gs = mlx.getShape(bank.gate.trellis);
    if (shape.len != 3 or ids_shape.len != 3 or gs.len != 4 or shape[0] != 1 or shape[1] < 2 or shape[1] > 16 or ids_shape[0] != 1 or ids_shape[1] != shape[1] or ids_shape[2] != 8 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const r = shape[1];
    const h = shape[2];
    const top: c_int = 8;
    const nslots = r * top;
    const inter = std.math.mul(c_int, gs[2], 16) catch return null;
    if (h <= 0 or @mod(h, 128) != 0 or inter <= 0 or !std.mem.eql(c_int, ids_shape, mlx.getShape(scores))) return null;
    const idtype = mlx.mlx_array_dtype(indices);
    const scoretype = mlx.mlx_array_dtype(scores);
    if ((idtype != .uint32 and idtype != .int32) or (scoretype != .float32 and scoretype != .float16 and scoretype != .bfloat16)) return null;
    const api = @import("root.zig");
    api.validateClampedProjection(bank.gate, gs[0], h, inter) catch return null;
    api.validateClampedProjection(bank.up, gs[0], h, inter) catch return null;
    api.validateClampedProjection(bank.down, gs[0], inter, h) catch return null;
    base.setDecodeParams(dec);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ r, h }, 2, s));
    try mlx.check(mlx.mlx_reshape(&ids, indices, &.{nslots}, 1, s));
    try mlx.check(mlx.mlx_reshape(&sc, scores, &.{nslots}, 1, s));
    const prep = if (layout == .lane) try base.lanePairPrepare(s, flat, bank.gate.suh, bank.up.suh, ids, null, h, nslots, top) else try support.prepare(s, flat, bank.gate.suh, bank.up.suh, ids, null, h, nslots, top);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    const gu = (try pairLayout(s, prep[0], prep[1], bank.gate.trellis, bank.up.trellis, ids, reduction, layout)) orelse return null;
    defer for (gu) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const d = blk: {
        if (layout == .lane) {
            const middle = try support.lane_middle(s, gu[0], gu[1], bank.gate.svh, bank.up.svh, bank.down.suh, ids, inter, nslots, 10);
            defer _ = mlx.mlx_array_free(middle);
            break :blk if (down == .grouped) (try projectLayout(s, middle, bank.down.trellis, ids, reduction, .lane)) orelse return null else try support.lane_down(s, middle, bank.down.trellis, ids);
        }
        if (down == .baseline and h == 4096 and inter == 2048 and gs[3] == 36 and dec.codebook == .mcg and dec.window == .w12)
            if (try support.fused_middle_down(s, gu[0], gu[1], bank.down.trellis, bank.gate.svh, bank.up.svh, bank.down.suh, ids, 10, 8)) |v| break :blk v;
        const middle = try support.middle(s, gu[0], gu[1], bank.gate.svh, bank.up.svh, bank.down.suh, ids, inter, nslots, 10);
        defer _ = mlx.mlx_array_free(middle);
        break :blk if (down == .grouped) (try project(s, middle, bank.down.trellis, ids, reduction)) orelse return null else try base.indexedGemvCoopF16(s, middle, bank.down.trellis, ids);
    };
    defer _ = mlx.mlx_array_free(d);
    const out = try base.downFinishReduce(s, d, bank.down.svh, ids, sc, h, r, top, .bfloat16);
    defer _ = mlx.mlx_array_free(out);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_reshape(&result, out, shape.ptr, shape.len, s));
    return result;
}

const Owned = struct {
    values: std.ArrayList(Arr) = .empty,
    fn deinit(self: *Owned) void {
        for (self.values.items) |a| {
            _ = mlx.mlx_array_free(a);
        }
        self.values.deinit(std.testing.allocator);
    }
    fn own(self: *Owned, a: Arr) !Arr {
        self.values.append(std.testing.allocator, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        return a;
    }
    fn weights(self: *Owned, e: c_int, k: c_int, n: c_int, rate: c_int, seed: u64) !Arr {
        const len: usize = @intCast(e * @divExact(k, 16) * @divExact(n, 16) * rate);
        const data = try std.testing.allocator.alloc(u16, len);
        defer std.testing.allocator.free(data);
        var rng = std.Random.DefaultPrng.init(seed);
        for (data) |*v| v.* = rng.random().int(u16);
        return self.own(mlx.mlx_array_new_data(data.ptr, &.{ e, @divExact(k, 16), @divExact(n, 16), rate }, 4, .uint16));
    }
    fn floats(self: *Owned, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64, scale: f32, s: mlx.mlx_stream) !Arr {
        var count: usize = 1;
        for (shape) |v| count *= @intCast(v);
        const data = try std.testing.allocator.alloc(f32, count);
        defer std.testing.allocator.free(data);
        var rng = std.Random.DefaultPrng.init(seed);
        for (data) |*v| v.* = (rng.random().float(f32) - 0.5) * scale;
        const input = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(input);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_astype(&out, input, dtype, s));
        return self.own(out);
    }
};
fn exact(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .float16) try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float16(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float16(b).?[0..n])) else try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
}

test "GLM group2 cooperative projections preserve F16 bits at all rates and ballot boundaries" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    for (0..18) |case| {
        const production = case == 17;
        const k: c_int = if (production) 4096 else 128;
        const n: c_int = if (production) 2048 else 128;
        const e: c_int = if (production) 32 else 128;
        const rate: c_int = if (production) 36 else @intCast(32 + case * 2);
        var owned: Owned = .{};
        defer owned.deinit();
        const tg = try owned.weights(e, k, n, rate, 132);
        const tu = try owned.weights(e, k, n, rate, 175);
        for ([_]c_int{ 3, 7, 32, 65, 128 }) |count| {
            const xg = try owned.floats(&.{ count, k }, .float16, 821, 0.3, s);
            const xu = try owned.floats(&.{ count, k }, .float16, 823, 0.5, s);
            var data: [128]u32 = undefined;
            for (data[0..@intCast(count)], 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(e)));
            for ([_]usize{ 0, 2, 6, 31, 32, 63, 64, 127 }) |i| if (i < count) {
                data[i] = 0;
            };
            const ids = try owned.own(mlx.mlx_array_new_data(&data, &.{count}, 1, .uint32));
            const rg = try owned.own(try base.indexedGemvCoopF16(s, xg, tg, ids));
            const ru = try owned.own(try base.indexedGemvCoopF16(s, xu, tu, ids));
            for ([_]Reduction{ .parallel, .serial }) |reduction| {
                const got = (try project(s, xg, tg, ids, reduction)) orelse return error.TestExpectedGroup2;
                defer _ = mlx.mlx_array_free(got);
                try exact(rg, got);
                const both = (try pair(s, xg, xu, tg, tu, ids, reduction)) orelse return error.TestExpectedGroup2;
                defer for (both) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                try exact(rg, both[0]);
                try exact(ru, both[1]);
            }
        }
    }
}

fn tensor(owned: *Owned, value: std.json.Value) !Arr {
    const file = value.object.get("file").?.string;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, file, std.testing.allocator, .limited(256 * 1024 * 1024));
    defer std.testing.allocator.free(data);
    const dims = value.object.get("shape").?.array.items;
    if (dims.len == 0 or dims.len > 4) return error.BadGroup2Fixture;
    var shape: [4]c_int = undefined;
    var bytes: usize = 2;
    for (dims, 0..) |dim, i| {
        if (dim.integer <= 0 or dim.integer > std.math.maxInt(c_int)) return error.BadGroup2Fixture;
        shape[i] = @intCast(dim.integer);
        bytes = try std.math.mul(usize, bytes, @intCast(dim.integer));
    }
    if (data.len != bytes) return error.BadGroup2Fixture;
    const dtype_name = value.object.get("dtype").?.string;
    const dtype: mlx.mlx_dtype = if (std.mem.eql(u8, dtype_name, "U16")) .uint16 else if (std.mem.eql(u8, dtype_name, "F16")) .float16 else return error.BadGroup2Fixture;
    return owned.own(mlx.mlx_array_new_data(data.ptr, shape[0..dims.len].ptr, @intCast(dims.len), dtype));
}
const Real = struct { bank: @import("root.zig").Bank, x: Arr, ids: Arr, scores: Arr, layer: i64, saved: i64 };
fn realFixture(owned: *Owned, value: std.json.Value, s: mlx.mlx_stream) !Real {
    var bank: @import("root.zig").Bank = undefined;
    const banks = value.object.get("banks").?;
    inline for (.{ "gate", "up", "down" }) |name| {
        const desc = banks.object.get(name).?;
        @field(bank, name) = .{ .trellis = try tensor(owned, desc.object.get("trellis").?), .suh = try tensor(owned, desc.object.get("suh").?), .svh = try tensor(owned, desc.object.get("svh").?) };
    }
    const r: c_int = @intCast(value.object.get("rows").?.integer);
    const top: c_int = @intCast(value.object.get("topk").?.integer);
    if (r < 2 or r > 16 or top != 8) return error.BadGroup2Fixture;
    const raw = value.object.get("local_ids").?.array.items;
    if (raw.len != @as(usize, @intCast(r * top))) return error.BadGroup2Fixture;
    var indices: [128]u32 = undefined;
    var scores: [128]f32 = undefined;
    for (raw, 0..) |id, i| {
        if (id.integer < 0 or id.integer >= mlx.getShape(bank.gate.trellis)[0]) return error.BadGroup2Fixture;
        indices[i] = @intCast(id.integer);
        scores[i] = @as(f32, @floatFromInt(i % 8 + 1)) / 36 * 2.5;
    }
    const x = try owned.floats(&.{ 1, r, 4096 }, .bfloat16, @intCast(717 + value.object.get("layer").?.integer), 2, s);
    try mlx.check(mlx.mlx_array_eval(x));
    return .{ .bank = bank, .x = x, .ids = try owned.own(mlx.mlx_array_new_data(&indices, &.{ 1, r, top }, 3, .uint32)), .scores = try owned.own(mlx.mlx_array_new_data(&scores, &.{ 1, r, top }, 3, .float32)), .layer = value.object.get("layer").?.integer, .saved = value.object.get("saved_slots").?.integer };
}
fn realArm(s: mlx.mlx_stream, f: Real, id: usize) !Arr {
    const dec = @import("expert_exl3.zig").Decode{ .codebook = .mcg, .window = .w12 };
    if (id == 0) return @import("root.zig").moeClamped(s, f.x, f.bank, f.ids, f.scores, dec, 10);
    return (try moe(s, f.x, f.bank, f.ids, f.scores, dec, if (id == 1 or id == 3) .parallel else .serial, if (id <= 2) .baseline else .grouped)) orelse error.TestExpectedGroup2;
}

test "GLM group2 actual checkpoint weights and captured routes preserve fullchain BF16 output" {
    const path = std.c.getenv("SUSHI_GLM_GROUP2_FIXTURE") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(path), a, .limited(1024 * 1024));
    defer a.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, data, .{});
    defer parsed.deinit();
    const s = mlx.gpuStream();
    for (parsed.value.object.get("cases").?.array.items) |case| {
        var owned: Owned = .{};
        defer owned.deinit();
        const f = try realFixture(&owned, case, s);
        const expected = try realArm(s, f, 0);
        defer _ = mlx.mlx_array_free(expected);
        for (1..5) |id| {
            const actual = try realArm(s, f, id);
            defer _ = mlx.mlx_array_free(actual);
            try exact(expected, actual);
        }
    }
}
fn timeReal(s: mlx.mlx_stream, f: Real, id: usize, count: usize) !u64 {
    const clock = @import("mlx_host").io_util.Stopwatch.init(std.testing.io);
    for (0..count) |_| {
        const y = try realArm(s, f, id);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    return clock.read() / count;
}

test "GLM group2 isolated actual-route timing" {
    const output = std.c.getenv("SUSHI_GLM_GROUP2_BENCH_OUT") orelse return error.SkipZigTest;
    const path = std.c.getenv("SUSHI_GLM_GROUP2_FIXTURE") orelse return error.MissingGroup2Fixture;
    const a = std.testing.allocator;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(path), a, .limited(1024 * 1024));
    defer a.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, data, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases").?.array.items;
    if (cases.len != 3) return error.BadGroup2Fixture;
    var samples: [3][11][5]u64 = undefined;
    var layers: [3]i64 = undefined;
    var saved: [3]i64 = undefined;
    const s = mlx.gpuStream();
    for (cases, 0..) |case, ci| {
        var owned: Owned = .{};
        defer owned.deinit();
        const f = try realFixture(&owned, case, s);
        layers[ci] = f.layer;
        saved[ci] = f.saved;
        const expected = try realArm(s, f, 0);
        defer _ = mlx.mlx_array_free(expected);
        for (0..5) |id| {
            const actual = try realArm(s, f, id);
            defer _ = mlx.mlx_array_free(actual);
            try exact(expected, actual);
            _ = try timeReal(s, f, id, 5);
        }
        for (0..11) |round| for (0..5) |step| {
            const id = if (round % 2 == 0) step else 4 - step;
            samples[ci][round][id] = try timeReal(s, f, id, 3);
        };
    }
    const json = try std.json.Stringify.valueAlloc(a, .{ .exact = true, .arms = .{ "baseline", "group_pair_parallel", "group_pair_serial", "group_all_parallel", "group_all_serial" }, .layers = layers, .saved_slots = saved, .slots = 32, .warmup = 5, .repetitions = 3, .timing = "warm selected real banks, synthetic activations, host apply+eval+free, interleaved arms", .nanoseconds = samples }, .{ .whitespace = .indent_2 });
    defer a.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}

test "GLM group2 singleton and all-shared routes preserve strided input bits" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    var owned: Owned = .{};
    defer owned.deinit();
    const source = try owned.floats(&.{ 65, 256 }, .float16, 481, 0.2, s);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_slice(&x, source, &.{ 0, 0 }, 2, &.{ 65, 256 }, 2, &.{ 1, 2 }, 2, s));
    const bank = try owned.weights(128, 128, 128, 36, 876);
    var transposed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed);
    try mlx.check(mlx.mlx_transpose_axes(&transposed, bank, &.{ 0, 2, 1, 3 }, 4, s));
    for ([_]bool{ false, true }) |shared| {
        var raw: [130]u32 = undefined;
        for (&raw, 0..) |*v, i| v.* = if (shared) 0 else @intCast(i / 2);
        const all = try owned.own(mlx.mlx_array_new_data(&raw, &.{130}, 1, .uint32));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_slice(&ids, all, &.{1}, 1, &.{130}, 1, &.{2}, 1, s));
        const reference = try base.indexedGemvCoopF16(s, x, transposed, ids);
        defer _ = mlx.mlx_array_free(reference);
        for ([_]Reduction{ .parallel, .serial }) |reduction| {
            const output = (try project(s, x, transposed, ids, reduction)) orelse return error.TestExpectedGroup2;
            defer _ = mlx.mlx_array_free(output);
            try exact(reference, output);
        }
    }
}

test "GLM group2 routed-chain guards precede any preparation" {
    const s = mlx.gpuStream();
    var owned: Owned = .{};
    defer owned.deinit();
    const proj = @import("root.zig").Proj{ .trellis = try owned.weights(8, 128, 128, 36, 71), .suh = try owned.floats(&.{ 8, 128 }, .float16, 73, 0.1, s), .svh = try owned.floats(&.{ 8, 128 }, .float16, 77, 0.1, s) };
    const valid = @import("root.zig").Bank{ .gate = proj, .up = proj, .down = proj };
    const x = try owned.floats(&.{ 1, 2, 128 }, .bfloat16, 79, 2, s);
    var ids_data: [16]u32 = undefined;
    for (&ids_data, 0..) |*v, i| v.* = @intCast(i % 8);
    const ids = try owned.own(mlx.mlx_array_new_data(&ids_data, &.{ 1, 2, 8 }, 3, .uint32));
    const scores = try owned.floats(&.{ 1, 2, 8 }, .float32, 83, 1, s);
    var malformed = valid;
    malformed.gate.suh = try owned.floats(&.{ 8, 1 }, .float16, 81, 0.1, s);
    const dec = @import("expert_exl3.zig").Decode{ .codebook = .mcg, .window = .w12 };
    try std.testing.expect((try moe(s, x, malformed, ids, scores, dec, .parallel, .grouped)) == null);
    const bad_scores = try owned.floats(&.{ 1, 2, 7 }, .float32, 91, 1, s);
    try std.testing.expect((try moe(s, x, valid, ids, bad_scores, dec, .parallel, .grouped)) == null);
    try std.testing.expect((try moe(s, x, valid, ids, ids, dec, .parallel, .grouped)) == null);
    try std.testing.expect((try moe(s, x, valid, scores, scores, dec, .parallel, .grouped)) == null);
    var huge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(huge);
    const scalar = try owned.own(mlx.mlx_array_new_data(&[_]u16{0}, &.{1}, 1, .uint16));
    try mlx.check(mlx.mlx_broadcast_to(&huge, scalar, &.{ 8, 8, @divTrunc(std.math.maxInt(c_int), 16) + 1, 36 }, 4, s));
    malformed = valid;
    malformed.gate.trellis = huge;
    try std.testing.expect((try moe(s, x, malformed, ids, scores, dec, .parallel, .grouped)) == null);
}

fn laneInput(owned: *Owned, x: Arr, s: mlx.mlx_stream) !Arr {
    const width: usize = @intCast(mlx.getShape(x)[1]);
    const indices = try std.testing.allocator.alloc(u32, width);
    defer std.testing.allocator.free(indices);
    for (0..width / 16) |tile| for (0..4) |q| for ([_]usize{ 0, 1, 8, 9 }, 0..) |offset, j| {
        indices[tile * 16 + q * 4 + j] = @intCast(tile * 16 + q * 2 + offset);
    };
    const ids = mlx.mlx_array_new_data(indices.ptr, &[_]c_int{@intCast(width)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_take_axis(&result, x, ids, 1, s));
    return owned.own(result);
}

test "GLM group2 half4 composition preserves all-rate cooperative projection bits" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    for (0..18) |case| {
        const production = case == 17;
        const k: c_int = if (production) 4096 else 128;
        const n: c_int = if (production) 2048 else 128;
        const rate: c_int = if (production) 36 else @intCast(32 + 2 * case);
        var owned: Owned = .{};
        defer owned.deinit();
        const g = try owned.weights(32, k, n, rate, 987);
        const u = try owned.weights(32, k, n, rate, 977);
        for ([_]c_int{ 3, 32, 65, 128 }) |count| {
            const xg = try owned.floats(&.{ count, k }, .float16, 877, 0.3, s);
            const xu = try owned.floats(&.{ count, k }, .float16, 857, 0.5, s);
            const pg = try laneInput(&owned, xg, s);
            const pu = try laneInput(&owned, xu, s);
            var data: [128]u32 = undefined;
            for (data[0..@intCast(count)], 0..) |*id, i| id.* = @intCast(i % 17);
            const ids = try owned.own(mlx.mlx_array_new_data(&data, &.{count}, 1, .uint32));
            const rg = try owned.own(try base.indexedGemvCoopF16(s, xg, g, ids));
            const ru = try owned.own(try base.indexedGemvCoopF16(s, xu, u, ids));
            for ([_]Reduction{ .parallel, .serial }) |reduction| {
                const got = (try projectLayout(s, pg, g, ids, reduction, .lane)) orelse return error.TestExpectedGroup2;
                defer _ = mlx.mlx_array_free(got);
                try exact(rg, got);
                const pair_out = (try pairLayout(s, pg, pu, g, u, ids, reduction, .lane)) orelse return error.TestExpectedGroup2;
                defer for (pair_out) |v| {
                    _ = mlx.mlx_array_free(v);
                };
                try exact(rg, pair_out[0]);
                try exact(ru, pair_out[1]);
            }
        }
    }
}
fn realLaneArm(s: mlx.mlx_stream, f: Real, id: usize) !Arr {
    const dec = @import("expert_exl3.zig").Decode{ .codebook = .mcg, .window = .w12 };
    if (id == 0) return @import("root.zig").moeClamped(s, f.x, f.bank, f.ids, f.scores, dec, 10);
    return (try moeLayout(s, f.x, f.bank, f.ids, f.scores, dec, if (id == 1 or id == 3) .parallel else .serial, if (id <= 2) .baseline else .grouped, .lane)) orelse error.TestExpectedGroup2;
}
fn requireLaneBaseline() !void {
    const pair_on = std.c.getenv("SUSHI_GLM_LANE_PAIR") orelse return error.MissingLaneBaseline;
    const down_on = std.c.getenv("SUSHI_GLM_DOWN_LANE") orelse return error.MissingLaneBaseline;
    if (!std.mem.eql(u8, std.mem.span(pair_on), "1") or !std.mem.eql(u8, std.mem.span(down_on), "1")) return error.MissingLaneBaseline;
}

test "GLM group2 half4 actual weights preserve lane-plus-down fullchain bits" {
    _ = std.c.getenv("SUSHI_GLM_GROUP2_LANE_TEST") orelse return error.SkipZigTest;
    const path = std.c.getenv("SUSHI_GLM_GROUP2_FIXTURE") orelse return error.MissingGroup2Fixture;
    try requireLaneBaseline();
    const a = std.testing.allocator;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(path), a, .limited(1024 * 1024));
    defer a.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, data, .{});
    defer parsed.deinit();
    const s = mlx.gpuStream();
    for (parsed.value.object.get("cases").?.array.items) |case| {
        var owned: Owned = .{};
        defer owned.deinit();
        const f = try realFixture(&owned, case, s);
        const pair_before = base.lanePairChainCalls();
        const down_before = base.downLaneCalls();
        const expected = try realLaneArm(s, f, 0);
        defer _ = mlx.mlx_array_free(expected);
        try std.testing.expectEqual(pair_before + 1, base.lanePairChainCalls());
        try std.testing.expectEqual(down_before + 1, base.downLaneCalls());
        for (1..5) |id| {
            const actual = try realLaneArm(s, f, id);
            defer _ = mlx.mlx_array_free(actual);
            try exact(expected, actual);
        }
    }
}
fn timeLane(s: mlx.mlx_stream, f: Real, id: usize, count: usize) !u64 {
    const clock = @import("mlx_host").io_util.Stopwatch.init(std.testing.io);
    for (0..count) |_| {
        const y = try realLaneArm(s, f, id);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    return clock.read() / count;
}

test "GLM group2 isolated lane-route timing" {
    const output = std.c.getenv("SUSHI_GLM_GROUP2_LANE_BENCH_OUT") orelse return error.SkipZigTest;
    const path = std.c.getenv("SUSHI_GLM_GROUP2_FIXTURE") orelse return error.MissingGroup2Fixture;
    try requireLaneBaseline();
    const a = std.testing.allocator;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(path), a, .limited(1024 * 1024));
    defer a.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, data, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases").?.array.items;
    if (cases.len != 6) return error.BadGroup2Fixture;
    var samples: [6][11][5]u64 = undefined;
    var layers: [6]i64 = undefined;
    var saved: [6]i64 = undefined;
    var row_counts: [6]i64 = undefined;
    const s = mlx.gpuStream();
    for (cases, 0..) |case, ci| {
        var owned: Owned = .{};
        defer owned.deinit();
        const f = try realFixture(&owned, case, s);
        layers[ci] = f.layer;
        row_counts[ci] = mlx.getShape(f.x)[1];
        saved[ci] = f.saved;
        const pair_before = base.lanePairChainCalls();
        const down_before = base.downLaneCalls();
        const expected = try realLaneArm(s, f, 0);
        defer _ = mlx.mlx_array_free(expected);
        try std.testing.expectEqual(pair_before + 1, base.lanePairChainCalls());
        try std.testing.expectEqual(down_before + 1, base.downLaneCalls());
        for (0..5) |id| {
            const actual = try realLaneArm(s, f, id);
            defer _ = mlx.mlx_array_free(actual);
            try exact(expected, actual);
            _ = try timeLane(s, f, id, 5);
        }
        for (0..11) |round| for (0..5) |step| {
            const id = if (round % 2 == 0) step else 4 - step;
            samples[ci][round][id] = try timeLane(s, f, id, 3);
        };
    }
    const json = try std.json.Stringify.valueAlloc(a, .{ .exact = true, .arms = .{ "lane_down_baseline", "group_lane_pair_parallel", "group_lane_pair_serial", "group_lane_all_parallel", "group_lane_all_serial" }, .layers = layers, .saved_slots = saved, .rows = row_counts, .topk = 8, .route_source = "three-row cases are prefixes of the captured N3 routes", .warmup = 5, .repetitions = 3, .timing = "warm selected real banks, synthetic activations, host apply+eval+free, interleaved arms", .nanoseconds = samples }, .{ .whitespace = .indent_2 });
    defer a.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
