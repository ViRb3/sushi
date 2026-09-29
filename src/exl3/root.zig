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
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K],
/// returned in x's shape and dtype. Up to `kernels.DECODE_ROWS_MAX` rows (or any
/// verify rows) take the decode chain, wider the prefill GEMM.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    // Which kernel a dispatch picks is read off ONE process-global codebook,
    // and several EXL3 packs can be resident at once: set it per call.
    kernels.setDecodeParams(dec);
    const xsh = mlx.getShape(x);
    const xd = mlx.mlx_array_dtype(x);
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
        try kernels.moeSwigluFused(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, xd)
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
