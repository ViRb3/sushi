//! EXL3 trellis experts: routed MoE banks in turboderp's EXL3 format, decoded
//! and multiplied by Metal kernels. A module, so another MLX host (mlx-serve)
//! can serve EXL3 packs through the same code: it needs an `mlx_host` import
//! whose root exposes `pub const mlx`, `log` and `io_util`.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
pub const format = @import("expert_exl3.zig");
pub const kernels = @import("expert_exl3_kernels.zig");

/// One projection's bank, expert e on axis 0: `trellis` U16 [E, in/16, out/16, n],
/// `suh` F16 [E, in], `svh` F16 [E, out].
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const group_layout = @import("layout.zig");
pub const GroupLayout = group_layout.Layout;
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K],
/// returned in x's shape and dtype. Up to `kernels.DECODE_ROWS_MAX` rows (or any
/// verify rows) take the decode chain, wider the prefill GEMM.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    return moeOutput(s, x, bank, inds, scores, dec, verify_rows, mlx.mlx_array_dtype(x));
}

fn moeOutput(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    // Which kernel a dispatch picks is read off ONE process-global codebook,
    // and several EXL3 packs can be resident at once: set it per call.
    kernels.setDecodeParams(dec);
    const xsh = mlx.getShape(x);
    const xd = out_dtype;
    const B = xsh[0];
    const S = xsh[1];
    const D = xsh[xsh.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ B * S, D }, 2, s));
    const ish = mlx.getShape(inds);
    const K = ish[ish.len - 1];
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_reshape(&slots, inds, &[_]c_int{B * S * K}, 1, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{B * S * K}, 1, s));
    var slots_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_u);
    try mlx.check(mlx.mlx_astype(&slots_u, slots, .uint32, s));
    const rows: usize = @intCast(B * S);
    if (rows >= 2 and rows <= kernels.DECODE_ROWS_MAX) {
        kernels.dumpUnionHist(slots_u, rows, @intCast(K)) catch {};
    }
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const y = if (rows <= kernels.DECODE_ROWS_MAX or verify_rows)
        if (mlx.getShape(g.trellis)[3] == mlx.getShape(u.trellis)[3])
            try kernels.moeSwigluFused(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, xd)
        else
            try moeMixedDecode(s, x2, bank, slots_u, sc, K, xd)
    else
        try kernels.moePrefill(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, K);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xsh.ptr, @intCast(xsh.len), s));
    if (mlx.mlx_array_dtype(out) != xd) {
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, out, xd, s));
        _ = mlx.mlx_array_free(out);
        return cast;
    }
    return out;
}

pub const Spec = struct {
    rate: format.Rate,
    codebook: format.Codebook,
    window: format.Window = .w16,
};

/// `k` is a rate, integer or fractional: admitted only when 16k is an even
/// whole number of halfwords per tile, which is what every reader indexes by.
fn rateFromConfigK(k_v: std.json.Value) ?format.Rate {
    const scaled: f64 = switch (k_v) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |f| f * 16.0,
        else => return null,
    };
    const rounded = @round(scaled);
    if (@abs(scaled - rounded) > 1e-6) return null;
    if (rounded < 0 or rounded > 1024) return null;
    return format.kFromPackedDim(@intFromFloat(rounded));
}

/// `window` is the codeword width the pack's search hashed. Absent means 16,
/// the whole sliding window; a width this build cannot decode is refused, not
/// rounded — the same bitstream decodes to different weights at each width.
fn windowFromConfig(v: ?std.json.Value) ?format.Window {
    const raw = v orelse return format.Window.w16;
    if (raw != .integer) return null;
    return format.Window.fromBits(raw.integer);
}

pub fn parseExpertQuant(obj: std.json.ObjectMap) !Spec {
    const block = obj.get("expert_quant") orelse return error.ExpertLayoutUnsupported;
    if (block != .object) return error.ExpertLayoutUnsupported;
    const fmt_v = block.object.get("format") orelse return error.ExpertLayoutUnsupported;
    if (fmt_v != .string or !std.mem.eql(u8, fmt_v.string, "exl3")) return error.ExpertLayoutUnsupported;
    const k_v = block.object.get("k") orelse return error.ExpertLayoutUnsupported;
    const rate = rateFromConfigK(k_v) orelse return error.ExpertLayoutUnsupported;
    const cb_v = block.object.get("codebook") orelse return error.ExpertLayoutUnsupported;
    if (cb_v != .string) return error.ExpertLayoutUnsupported;
    const codebook = format.Codebook.fromName(cb_v.string) orelse return error.ExpertLayoutUnsupported;
    const window = windowFromConfig(block.object.get("window")) orelse return error.Exl3WindowUnsupported;
    return .{ .rate = rate, .codebook = codebook, .window = window };
}

fn specFromConfigJson(allocator: std.mem.Allocator, raw: []const u8) !Spec {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    return parseExpertQuant(parsed.value.object);
}

test "an EXL3 pack's codeword window is 16 unless its config names one this build decodes" {
    const t = std.testing;
    const base = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\"";
    try t.expectEqual(format.Window.w16, (try specFromConfigJson(t.allocator, base ++ "}}")).window);
    try t.expectEqual(format.Window.w16, (try specFromConfigJson(t.allocator, base ++ ",\"window\":16}}")).window);
    try t.expectEqual(format.Window.w12, (try specFromConfigJson(t.allocator, base ++ ",\"window\":12}}")).window);
    try t.expectEqual(format.Window.w11, (try specFromConfigJson(t.allocator, base ++ ",\"window\":11}}")).window);
    try t.expectEqual(format.Window.w10, (try specFromConfigJson(t.allocator, base ++ ",\"window\":10}}")).window);
    try t.expectEqual(format.Window.w8, (try specFromConfigJson(t.allocator, base ++ ",\"window\":8}}")).window);
    // The same bitstream decodes to different weights at each width, so an
    // unreadable one is a refusal, never a fallback to 16.
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":17}}"));
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":7}}"));
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":\"12\"}}"));
}

test "an EXL3 pack naming a codebook this build does not decode is refused" {
    const t = std.testing;
    try t.expectError(error.ExpertLayoutUnsupported, specFromConfigJson(t.allocator,
        \\{"expert_quant":{"format":"exl3","k":2.5,"codebook":"mul2","window":12}}
    ));
}

pub fn admitTopK(topk: u32) !void {
    if (topk > kernels.REDUCE_MAX_TOPK) return error.Exl3TopKExceedsReduceBank;
}

pub fn trellisAdmitted(shape: []const c_int, experts: u32, in_dim: u32, out_dim: u32, rate: format.Rate) bool {
    if (shape.len != 4) return false;
    for (shape) |d| if (d <= 0) return false;
    if (in_dim % 16 != 0 or out_dim % 16 != 0) return false;
    const want = [3]u32{ experts, in_dim / 16, out_dim / 16 };
    for (want, 0..) |w, i| if (w != @as(u32, @intCast(shape[i]))) return false;
    const packed_rate = format.kFromPackedDim(@intCast(shape[3])) orelse return false;
    return packed_rate.n <= rate.n;
}

test "exl3 top-k above reduce-bank is a named refusal" {
    const t = std.testing;
    try admitTopK(10);
    try admitTopK(16);
    try admitTopK(32);
    try t.expectError(error.Exl3TopKExceedsReduceBank, admitTopK(33));
}

test "exl3 expert_quant admits integer and fractional K under every served codebook and refuses the rest" {
    const t = std.testing;
    const cases = [_]struct { text: []const u8, n: u32 }{
        .{ .text = "2", .n = 32 },
        .{ .text = "2.5", .n = 40 },
        .{ .text = "2.75", .n = 44 },
        .{ .text = "3", .n = 48 },
        .{ .text = "3.5", .n = 56 },
        .{ .text = "4", .n = 64 },
    };
    for (cases) |c| {
        for ([_]format.Codebook{ .mul1, .mcg }) |cb| {
            var buf: [96]u8 = undefined;
            const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"{s}\"}}}}", .{ c.text, @tagName(cb) });
            const ok = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
            defer ok.deinit();
            const spec = try parseExpertQuant(ok.value.object);
            try t.expectEqual(c.n, spec.rate.n);
            try t.expectEqual(cb, spec.codebook);
        }
    }
    // 2.3 is not a multiple of 1/16; 4.5 and 6 are off the served range.
    for ([_][]const u8{ "2.3", "8.125", "9", "0.875", "2.0625" }) |bad| {
        var buf: [96]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"mul1\"}}}}", .{bad});
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
        defer parsed.deinit();
        try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(parsed.value.object));
    }
    const bad_cb = try std.json.parseFromSlice(std.json.Value, t.allocator,
        \\{"expert_quant":{"format":"exl3","k":4,"codebook":"mul2"}}
    , .{});
    defer bad_cb.deinit();
    try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(bad_cb.value.object));
    const missing = try std.json.parseFromSlice(std.json.Value, t.allocator, "{}", .{});
    defer missing.deinit();
    try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(missing.value.object));
}

test "exl3 a layer may pack below the rate the config bills, never above it" {
    const t = std.testing;
    const E: u32 = 256;
    const h: u32 = 4096;
    const i: u32 = 2048;
    const k4: format.Rate = .{ .n = 64 };
    const shape = struct {
        fn at(n: c_int) [4]c_int {
            return .{ @intCast(E), @intCast(h / 16), @intCast(i / 16), n };
        }
    }.at;
    // A pack whose tail layers carry more bits than its body: the config bills
    // the widest, so a narrower layer admits and a wider one cannot.
    try t.expect(trellisAdmitted(&shape(64), E, h, i, k4));
    try t.expect(trellisAdmitted(&shape(40), E, h, i, k4));
    try t.expect(!trellisAdmitted(&shape(64), E, h, i, .{ .n = 40 }));
    try t.expect(!trellisAdmitted(&shape(41), E, h, i, k4));
    try t.expect(!trellisAdmitted(&shape(64), E, h, h, k4));
}

test {
    _ = format;
    _ = kernels;
}

test "exl3 Sushi CPU config and packed shapes admit K1 through K8" {
    const t = std.testing;
    for (0..145) |n| {
        const k = @as(f64, @floatFromInt(n)) / 16;
        const admitted = n >= 16 and n <= 128 and n % 2 == 0;
        try t.expectEqual(admitted, rateFromConfigK(.{ .float = k }) != null);
        try t.expectEqual(admitted, trellisAdmitted(&.{ 2, 8, 16, @intCast(n) }, 2, 128, 256, .{ .n = 128 }));
        if (!admitted) continue;
        for ([_][]const u8{ "mul1", "mcg" }) |cb| {
            const raw = try std.fmt.allocPrint(t.allocator, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{d},\"codebook\":\"{s}\"}}}}", .{ k, cb });
            defer t.allocator.free(raw);
            try t.expectEqual(@as(u32, @intCast(n)), (try specFromConfigJson(t.allocator, raw)).rate.n);
        }
        if (n > 16) try t.expect(!trellisAdmitted(&.{ 2, 8, 16, @intCast(n) }, 2, 128, 256, .{ .n = @intCast(n - 2) }));
    }
    for ([_]i64{ 1, 2, 3, 4, 5, 6, 7, 8 }) |k| try t.expectEqual(@as(u32, @intCast(k * 16)), rateFromConfigK(.{ .integer = k }).?.n);
}

test "sushi coder layouts validate groups route slots and bill stored bytes" {
    const t = std.testing;
    var plan = GroupLayout.init(128, 128, .{ .n = 64 });
    for ([_]u32{ 2, 1 }, 0..) |e, g| {
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
            var buf: [96]u8 = undefined;
            const n: u32 = @intCast(32 + 16 * ((g + p) % 3));
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.trellis", .{ proj, g }), &.{ e, 8, 8, n }, .u16);
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.suh", .{ proj, g }), &.{ e, 128 }, .f16);
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.svh", .{ proj, g }), &.{ e, 128 }, .f16);
        }
    }
    try plan.finish(3, 2, false, 512);
    try t.expectEqual(@as(usize, 2), plan.count);
    try t.expectEqual(@as(u64, 3 * (8 * 8 * (32 + 48 + 64) * 2 + 3 * 256 * 2)), plan.bytes);
    try t.expectEqual(@as(u32, 1), (try plan.locate(2)).group);
    try t.expectEqual(@as(u32, 0), (try plan.locate(2)).local);
    try t.expectEqual(@as(u32, 1), (try plan.locate(1)).local);
    try t.expectError(error.ExpertOutOfRange, plan.locate(3));
    try t.expectError(error.Exl3RouterWidthMismatch, plan.finish(4, 2, false, 512));
    try t.expectError(error.Exl3TopKExceedsExperts, plan.finish(3, 4, false, 512));
    try t.expectError(error.Exl3RaggedStreamingUnsupported, plan.finish(3, 2, true, 512));
}

test "sushi coder layouts refuse malformed missing mixed and unsupported groups" {
    const t = std.testing;
    var plan = GroupLayout.init(128, 128, .{ .n = 64 });
    try t.expectError(error.Exl3GroupNameInvalid, plan.add("gate_proj.g01.trellis", &.{ 1, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3TrellisGeometry, plan.add("gate_proj.g0.trellis", &.{ 1, 8, 8, 33 }, .u16));
    try t.expectError(error.Exl3GroupGeometry, plan.add("gate_proj.g0.trellis", &.{ 0, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3GroupGeometry, plan.add("gate_proj.g0.suh", &.{ 1, 64 }, .f16));
    _ = try plan.add("gate_proj.g1.trellis", &.{ 1, 8, 8, 32 }, .u16);
    try t.expectError(error.Exl3GroupMissing, plan.finish(1, 1, false, 512));
    try t.expectError(error.Exl3GroupGeometry, plan.add("up_proj.g1.trellis", &.{ 2, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3MixedGroupLayout, plan.add("gate_proj.trellis", &.{ 1, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3GroupDtype, plan.add("up_proj.g1.suh", &.{ 1, 128 }, .u16));
    try t.expectError(error.Exl3GroupNameInvalid, plan.add("gate_proj.g32.trellis", &.{ 1, 8, 8, 32 }, .u16));
}

test "sushi coder GPU two groups equal independently routed uniform banks" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var banks: [2]Bank = undefined;
    var held: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (held.items) |a| _ = mlx.mlx_array_free(a);
        held.deinit(t.allocator);
    }
    for (&banks, 0..) |*bank, group| {
        for ([_]*Proj{ &bank.gate, &bank.up, &bank.down }, 0..) |p, projection| {
            const n: c_int = @intCast(32 + 16 * ((group + projection) % 3));
            const words = try t.allocator.alloc(u16, @intCast(8 * 8 * n));
            defer t.allocator.free(words);
            for (words, 0..) |*v, i| v.* = @truncate(i *% 7919 +% 23);
            const signs: [128]u16 = @splat(format.f32ToF16Bits(0.125));
            p.trellis = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, 8, 8, n }, 4, .uint16);
            try held.append(t.allocator, p.trellis);
            p.suh = mlx.mlx_array_new_data(&signs, &[_]c_int{ 1, 128 }, 2, .float16);
            try held.append(t.allocator, p.suh);
            p.svh = mlx.mlx_array_new_data(&signs, &[_]c_int{ 1, 128 }, 2, .float16);
            try held.append(t.allocator, p.svh);
        }
    }
    const xh: [128]u16 = @splat(format.f32ToF16Bits(0.25));
    const x = mlx.mlx_array_new_data(&xh, &[_]c_int{ 1, 1, 128 }, 3, .float16);
    defer _ = mlx.mlx_array_free(x);
    const ids = mlx.mlx_array_new_data(&[_]i32{ 1, 0 }, &[_]c_int{ 1, 1, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const scores = mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75 }, &[_]c_int{ 1, 1, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const got = try moeGroups(s, x, &banks, ids, scores, .mcg, false);
    defer _ = mlx.mlx_array_free(got);
    const one_id = mlx.mlx_array_new_data(&[_]i32{0}, &[_]c_int{ 1, 1, 1 }, 3, .int32);
    defer _ = mlx.mlx_array_free(one_id);
    var outputs: [2]mlx.mlx_array = undefined;
    for (&outputs, 0..) |*out, i| {
        const score = mlx.mlx_array_new_data(&[_]f32{if (i == 0) 0.75 else 0.25}, &[_]c_int{ 1, 1, 1 }, 3, .float32);
        defer _ = mlx.mlx_array_free(score);
        out.* = try moe(s, x, banks[i], one_id, score, .mcg, false);
    }
    defer for (outputs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_add(&want, outputs[0], outputs[1], s));
    var actual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(actual);
    var expected = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expected);
    try mlx.check(mlx.mlx_astype(&actual, got, .float32, s));
    try mlx.check(mlx.mlx_astype(&expected, want, .float32, s));
    try mlx.check(mlx.mlx_array_eval(actual));
    try mlx.check(mlx.mlx_array_eval(expected));
    const a = mlx.mlx_array_data_float32(actual).?;
    const e = mlx.mlx_array_data_float32(expected).?;
    for (0..128) |i| try t.expectApproxEqAbs(e[i], a[i], 0.0001 + @abs(e[i]) * 0.01);
}

pub fn moeGroups(s: mlx.mlx_stream, x: mlx.mlx_array, banks: []const Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    if (banks.len == 0) return error.Exl3GroupMissing;
    if (banks.len == 1) return moe(s, x, banks[0], inds, scores, dec, verify_rows);
    var total = mlx.mlx_array{ .ctx = null };
    defer if (total.ctx != null) {
        _ = mlx.mlx_array_free(total);
    };
    var offset: c_int = 0;
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    for (banks) |bank| {
        const count = mlx.getShape(bank.gate.trellis)[0];
        const lo = mlx.mlx_array_new_int(offset);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_int(offset + count);
        defer _ = mlx.mlx_array_free(hi);
        var lower = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lower);
        try mlx.check(mlx.mlx_greater_equal(&lower, inds, lo, s));
        var upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upper);
        try mlx.check(mlx.mlx_less(&upper, inds, hi, s));
        var mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_logical_and(&mask, lower, upper, s));
        var shifted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shifted);
        try mlx.check(mlx.mlx_subtract(&shifted, inds, lo, s));
        var local = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local);
        try mlx.check(mlx.mlx_where(&local, mask, shifted, zero, s));
        var weight = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weight);
        try mlx.check(mlx.mlx_where(&weight, mask, scores, zero, s));
        const partial = try moeOutput(s, x, bank, local, weight, dec, verify_rows, .float32);
        defer _ = mlx.mlx_array_free(partial);
        var wide = mlx.mlx_array_new();
        defer if (wide.ctx != null) {
            _ = mlx.mlx_array_free(wide);
        };
        try mlx.check(mlx.mlx_astype(&wide, partial, .float32, s));
        if (total.ctx == null) {
            total = wide;
            wide = .{ .ctx = null };
        } else {
            var sum = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_add(&sum, total, wide, s));
            _ = mlx.mlx_array_free(total);
            total = sum;
        }
        offset += count;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, total, mlx.mlx_array_dtype(x), s));
    return out;
}

fn moeMixedDecode(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const hidden = mlx.getShape(x)[1];
    const rows = mlx.getShape(x)[0];
    const inter = mlx.getShape(bank.gate.trellis)[2] * 16;
    const nslots = rows * topk;
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const gate = try kernels.pairGemv(s, x, g.suh, g.suh, g.trellis, g.trellis, slots, hidden, inter, nslots, topk, 0);
    defer _ = mlx.mlx_array_free(gate[0]);
    defer _ = mlx.mlx_array_free(gate[1]);
    const up = try kernels.pairGemv(s, x, u.suh, u.suh, u.trellis, u.trellis, slots, hidden, inter, nslots, topk, 0);
    defer _ = mlx.mlx_array_free(up[0]);
    defer _ = mlx.mlx_array_free(up[1]);
    const inner = try kernels.downGemvFusedMid(s, gate[0], up[0], d.trellis, g.svh, u.svh, d.suh, slots, inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(inner);
    return kernels.downFinishReduce(s, inner, d.svh, slots, scores, hidden, rows, topk, dtype);
}

test "imatrix GPU capture K2 K3 K4 parity and sorted slot alignment" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const HostProj = struct { trellis: []u16, suh: [256]u16, svh: [256]u16 };
    const Probe = struct {
        calls: usize = 0,
        host: [3]HostProj,
        input: []const f32,
        global_ids: []const u32,
        rate: format.Rate,
        fn expectStageBound(self: *const @This(), row: usize, expert: usize, projection: []const u8, stage: []const u8, values: []const f32) !void {
            var peak: f32 = 0;
            for (values) |v| {
                if (!std.math.isFinite(v)) {
                    peak = std.math.inf(f32);
                    break;
                }
                peak = @max(peak, @abs(v));
            }
            if (peak >= 1e4) {
                std.debug.print("imatrix host oracle exceeds f16 safety bound: K{d} row={d} expert={d} {s}.{s} max_abs={d}, required < 10000\n", .{ @divExact(self.rate.n, 16), row, expert, projection, stage, peak });
                return error.ImatrixFixtureExceedsF16Bound;
            }
        }
        fn boundedHostProject(self: *const @This(), row: usize, expert: usize, projection: usize, input: []const f32, output: *[128]f32) !void {
            const name = ([_][]const u8{ "gate", "up", "down" })[projection];
            const bank = self.host[projection];
            const stride = 8 * 8 * self.rate.n;
            var prepared: [128]f32 = undefined;
            try self.expectStageBound(row, expert, name, "input", input);
            for (input, &prepared, 0..) |v, *scaled, channel| scaled.* = v * format.f16BitsToF32(bank.suh[expert * 128 + channel]);
            try self.expectStageBound(row, expert, name, "input_times_suh", &prepared);
            format.hadamard128(&prepared);
            try self.expectStageBound(row, expert, name, "input_hadamard", &prepared);
            var weights: [128 * 128]u16 = undefined;
            format.reconstructInner(bank.trellis[expert * stride ..][0..stride], 128, 128, self.rate, .mcg, &weights);
            for (output, 0..) |*v, column| {
                v.* = 0;
                for (prepared, 0..) |x, k| v.* += x * format.f16BitsToF32(weights[k * 128 + column]);
            }
            try self.expectStageBound(row, expert, name, "inner", output);
            format.hadamard128(output);
            try self.expectStageBound(row, expert, name, "output_hadamard", output);
            for (output, 0..) |*v, channel| v.* *= format.f16BitsToF32(bank.svh[expert * 128 + channel]);
            try self.expectStageBound(row, expert, name, "output", output);
        }
        fn expectFiniteHostOracle(self: *const @This()) !void {
            for (0..self.input.len / 128) |row| {
                const input = self.input[row * 128 ..][0..128];
                var outputs: [2][128]f32 = undefined;
                for (&outputs, 0..) |*expected, expert| {
                    var gate: [128]f32 = undefined;
                    var up: [128]f32 = undefined;
                    var sigmoid: [128]f32 = undefined;
                    var silu: [128]f32 = undefined;
                    var mid: [128]f32 = undefined;
                    try self.boundedHostProject(row, expert, 0, input, &gate);
                    try self.boundedHostProject(row, expert, 1, input, &up);
                    for (&sigmoid, &silu, &mid, gate, up) |*sig, *si, *v, g, u| {
                        sig.* = 1 / (1 + @exp(-g));
                        si.* = g * sig.*;
                        v.* = si.* * u;
                    }
                    try self.expectStageBound(row, expert, "swiglu", "sigmoid", &sigmoid);
                    try self.expectStageBound(row, expert, "swiglu", "silu", &silu);
                    try self.expectStageBound(row, expert, "swiglu", "activation", &mid);
                    try self.boundedHostProject(row, expert, 2, &mid, expected);
                    for ([_][]const f32{ &gate, &up, &mid, expected }) |values| for (values) |v| try t.expect(std.math.isFinite(v));
                }
                var max_difference: f32 = 0;
                var norm: [2]f32 = .{ 0, 0 };
                for (outputs[0], outputs[1]) |a, b| {
                    max_difference = @max(max_difference, @abs(a - b));
                    norm[0] += a * a;
                    norm[1] += b * b;
                }
                const tolerance = 0.02 * @sqrt(@max(norm[0], norm[1]) / 128) + 1e-6;
                if (max_difference <= 2 * tolerance) {
                    std.debug.print("imatrix host experts are not distinguishable for the same input: K{d} row={d} max_difference={d}, required > {d}\n", .{ @divExact(self.rate.n, 16), row, max_difference, 2 * tolerance });
                    return error.ImatrixFixtureExpertsIndistinguishable;
                }
            }
        }
        fn observe(raw: *anyopaque, activation: mlx.mlx_array, outputs: mlx.mlx_array, ids: mlx.mlx_array, scores: mlx.mlx_array) !void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
            try mlx.check(mlx.mlx_array_eval(ids));
            try mlx.check(mlx.mlx_array_eval(scores));
            try mlx.check(mlx.mlx_array_eval(activation));
            try mlx.check(mlx.mlx_array_eval(outputs));
            const p = mlx.mlx_array_data_uint32(ids).?;
            const w = mlx.mlx_array_data_float32(scores).?;
            const act = mlx.mlx_array_data_float32(activation).?;
            const out = mlx.mlx_array_data_float32(outputs).?;
            const stride = 8 * 8 * self.rate.n;
            for (0..mlx.mlx_array_size(ids)) |i| {
                const original: usize = @intFromFloat((w[i] - 0.125) * 512);
                try t.expect(original < self.global_ids.len);
                try t.expectEqual(self.global_ids[original], p[i]);
                const e: usize = p[i];
                const input = self.input[original / 2 * 128 ..][0..128];
                var transformed: [128]f32 = undefined;
                var inner: [128]f32 = undefined;
                var gate: [128]f32 = undefined;
                var up: [128]f32 = undefined;
                var mid: [128]f32 = undefined;
                var expected: [128]f32 = undefined;
                for ([_]HostProj{ self.host[0], self.host[1] }, [_]*[128]f32{ &gate, &up }) |projection, dest| {
                    format.project(input, projection.trellis[e * stride ..][0..stride], projection.suh[e * 128 ..][0..128], projection.svh[e * 128 ..][0..128], 128, 128, self.rate, .mcg, &transformed, &inner, dest);
                }
                for (&mid, gate, up) |*v, g, u| v.* = (g / (1 + @exp(-g))) * u;
                const down = self.host[2];
                format.project(&mid, down.trellis[e * stride ..][0..stride], down.suh[e * 128 ..][0..128], down.svh[e * 128 ..][0..128], 128, 128, self.rate, .mcg, &transformed, &inner, &expected);
                for ([_][]const f32{ &mid, &expected }, [_][]const f32{ act[i * 128 ..][0..128], out[i * 128 ..][0..128] }) |truth, actual| {
                    var norm: f32 = 0;
                    for (truth) |v| norm += v * v;
                    const tolerance = 0.02 * @sqrt(norm / 128) + 1e-6;
                    for (truth, actual) |want, got| {
                        try t.expect(std.math.isFinite(want) and std.math.isFinite(got));
                        try t.expectApproxEqAbs(want, got, tolerance);
                    }
                }
            }
        }
    };
    for ([_]c_int{ 32, 48, 64 }) |n| {
        var bank: Bank = undefined;
        var host: [3]HostProj = undefined;
        var word_buffers: std.ArrayList([]u16) = .empty;
        defer {
            for (word_buffers.items) |words| t.allocator.free(words);
            word_buffers.deinit(t.allocator);
        }
        var held: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (held.items) |a| _ = mlx.mlx_array_free(a);
            held.deinit(t.allocator);
        }
        for ([_]*Proj{ &bank.gate, &bank.up, &bank.down }, 0..) |p, projection| {
            const words = try t.allocator.alloc(u16, @intCast(2 * 8 * 8 * n));
            try word_buffers.append(t.allocator, words);
            for (words, 0..) |*v, i| v.* = @truncate(i *% 7919 +% 23 +% projection * 103);
            const input_scale: [256]u16 = @splat(format.f32ToF16Bits(if (projection == 2) 0.03125 else 0.0078125));
            const output_scale: [256]u16 = @splat(format.f32ToF16Bits(0.5));
            host[projection] = .{ .trellis = words, .suh = input_scale, .svh = output_scale };
            p.trellis = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 2, 8, 8, n }, 4, .uint16);
            try held.append(t.allocator, p.trellis);
            p.suh = mlx.mlx_array_new_data(&input_scale, &[_]c_int{ 2, 128 }, 2, .float16);
            try held.append(t.allocator, p.suh);
            p.svh = mlx.mlx_array_new_data(&output_scale, &[_]c_int{ 2, 128 }, 2, .float16);
            try held.append(t.allocator, p.svh);
        }
        for (host) |projection| {
            const stride = projection.trellis.len / 2;
            try t.expect(!std.mem.eql(u16, projection.trellis[0..stride], projection.trellis[stride..]));
        }
        for (0..host.len) |i| for (i + 1..host.len) |j| {
            try t.expect(!std.mem.eql(u16, host[i].trellis, host[j].trellis));
            const stride = host[i].trellis.len / 2;
            for (0..2) |expert| {
                const start = expert * stride;
                try t.expect(!std.mem.eql(u16, host[i].trellis[start..][0..stride], host[j].trellis[start..][0..stride]));
            }
        };
        for ([_]usize{ 1, 33 }) |rows| {
            for ([_]f32{ 1, 128 }) |magnitude| {
                for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
                    const xh = try t.allocator.alloc(f32, rows * 128);
                    defer t.allocator.free(xh);
                    for (xh, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) * 0.25 * magnitude;
                    const ih = try t.allocator.alloc(u32, rows * 2);
                    defer t.allocator.free(ih);
                    const wh = try t.allocator.alloc(f32, rows * 2);
                    defer t.allocator.free(wh);
                    for (ih, wh, 0..) |*id, *w, i| {
                        id.* = @intCast((i + i / 2) % 2);
                        w.* = 0.125 + @as(f32, @floatFromInt(i)) / 512;
                    }
                    const raw_x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, @intCast(rows), 128 }, 3, .float32);
                    defer _ = mlx.mlx_array_free(raw_x);
                    var x = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&x, raw_x, dtype, s));
                    defer _ = mlx.mlx_array_free(x);
                    const ids = mlx.mlx_array_new_data(ih.ptr, &[_]c_int{ 1, @intCast(rows), 2 }, 3, .uint32);
                    defer _ = mlx.mlx_array_free(ids);
                    const weights = mlx.mlx_array_new_data(wh.ptr, &[_]c_int{ 1, @intCast(rows), 2 }, 3, .float32);
                    defer _ = mlx.mlx_array_free(weights);
                    var probe = Probe{ .host = host, .input = xh, .global_ids = ih, .rate = format.kFromPackedDim(@intCast(n)).? };
                    try probe.expectFiniteHostOracle();
                    const normal = try moe(s, x, bank, ids, weights, .mcg, false);
                    defer _ = mlx.mlx_array_free(normal);
                    try mlx.check(mlx.mlx_array_eval(normal));
                    var normal32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(normal32);
                    try mlx.check(mlx.mlx_astype(&normal32, normal, .float32, s));
                    try mlx.check(mlx.mlx_array_eval(normal32));
                    for (mlx.mlx_array_data_float32(normal32).?[0..xh.len], 0..) |v, i| {
                        if (!std.math.isFinite(v)) {
                            std.debug.print("imatrix normal arm is non-finite before capture-off parity: K{d} rows={d} magnitude={d} dtype={s} index={d} value={d}\\n", .{ @divExact(n, 16), rows, magnitude, @tagName(dtype), i, v });
                            return error.TestExpectedEqual;
                        }
                    }
                    const off = try moeWithCapture(s, x, bank, ids, weights, .mcg, false, null);
                    defer _ = mlx.mlx_array_free(off);
                    try mlx.check(mlx.mlx_array_eval(off));
                    if (dtype == .float32) {
                        try t.expectEqualSlices(f32, mlx.mlx_array_data_float32(normal).?[0..xh.len], mlx.mlx_array_data_float32(off).?[0..xh.len]);
                    } else {
                        try t.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(normal).?[0..xh.len], mlx.mlx_array_data_bfloat16(off).?[0..xh.len]);
                    }
                    const captured = try moeWithCapture(s, x, bank, ids, weights, .mcg, false, .{ .context = &probe, .observe = Probe.observe });
                    defer _ = mlx.mlx_array_free(captured);
                    try mlx.check(mlx.mlx_array_eval(captured));
                    try t.expectEqual(@as(usize, 1), probe.calls);
                    var actual32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(actual32);
                    var expected32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(expected32);
                    try mlx.check(mlx.mlx_astype(&actual32, captured, .float32, s));
                    try mlx.check(mlx.mlx_astype(&expected32, normal, .float32, s));
                    try mlx.check(mlx.mlx_array_eval(actual32));
                    try mlx.check(mlx.mlx_array_eval(expected32));
                    const actual = mlx.mlx_array_data_float32(actual32).?;
                    const expected = mlx.mlx_array_data_float32(expected32).?;
                    var norm: f32 = 0;
                    for (expected[0..xh.len]) |v| norm += v * v;
                    const tolerance = 0.02 * @sqrt(norm / @as(f32, @floatFromInt(xh.len))) + 1e-6;
                    for (0..xh.len) |i| {
                        try t.expect(std.math.isFinite(actual[i]) and std.math.isFinite(expected[i]));
                        try t.expectApproxEqAbs(expected[i], actual[i], tolerance);
                    }
                }
            }
        }
    }
}

pub const Capture = struct {
    context: *anyopaque,
    observe: *const fn (*anyopaque, mlx.mlx_array, mlx.mlx_array, mlx.mlx_array, mlx.mlx_array) anyerror!void,
};

var capture_engaged = false;

pub fn moeWithCapture(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, capture: ?Capture) !mlx.mlx_array {
    const tap = capture orelse return moe(s, x, bank, inds, scores, dec, verify_rows);
    kernels.setDecodeParams(dec);
    const shape = mlx.getShape(x);
    const rows = shape[0] * shape[1];
    const hidden = shape[2];
    const topk = mlx.getShape(inds)[2];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, hidden }, 2, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_reshape(&ids, inds, &[_]c_int{rows * topk}, 1, s));
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_astype(&slots, ids, .uint32, s));
    var weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights);
    try mlx.check(mlx.mlx_reshape(&weights, scores, &[_]c_int{rows * topk}, 1, s));
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const captured = try kernels.moeCapture(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots, weights, topk, verify_rows);
    defer captured.deinit();
    try tap.observe(tap.context, captured.activation, captured.outputs, captured.slots, captured.scores);
    if (!capture_engaged) {
        capture_engaged = true;
        @import("mlx_host").log.info("[exl3] imatrix capture engaged: unfused SwiGLU and unweighted f32 down outputs\n", .{});
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, captured.output, shape.ptr, @intCast(shape.len), s));
    return out;
}
