//! Small, deliberately shuffled source shards for the GLM BF16 and FP8 stream tests.
const std = @import("std");

pub const Fault = enum { none, missing, fp16, wrong_shape, truncated, dense_prefix, mixed, fp8_scale };
pub const Storage = enum { bf16, fp8 };

pub fn value(expert: usize, projection: usize) u16 {
    return @intCast(0x3f00 + expert * 16 + projection * 4);
}

/// A finite e4m3 code (never the 0x7f/0xff NaNs) that differs per expert, projection and element.
pub fn fp8Code(expert: usize, projection: usize, i: usize) u8 {
    var h: u32 = @intCast((expert * 131 + projection * 31 + i) & 0xffffffff);
    h = (h ^ (h >> 7)) *% 0x9e3779b1;
    const code: u8 = @truncate(h >> 13);
    return if (code & 0x7f == 0x7f) code ^ 1 else code;
}

/// A block-128 scale large enough that one routed expert moves a fixture model's logits.
pub fn fp8Scale(expert: usize, projection: usize, block: usize) f32 {
    return @as(f32, @floatFromInt(17 + expert * 5 + projection * 3 + block)) * 1.0e-4;
}

pub fn write(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault) !void {
    return writeSized(allocator, dir, fault, 32, 16);
}

pub fn writeSized(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault, hidden: usize, intermediate: usize) !void {
    return writeStorage(allocator, dir, fault, hidden, intermediate, .bf16);
}

pub fn writeStorage(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault, hidden: usize, intermediate: usize, storage: Storage) !void {
    const io = std.testing.io;
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var entries: usize = 0;
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |projection, pi| {
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(allocator);
        try header.append(allocator, '{');
        const payload = try allocator.alloc(u8, 4 * (hidden * intermediate * 2 + hidden * intermediate / (128 * 128) * 4 + 4));
        defer allocator.free(payload);
        var offset: usize = 0;
        var count: usize = 0;
        for ([_]usize{ 3, 1, 0, 2 }) |expert| {
            if (fault == .missing and expert == 2 and pi == 1) continue;
            const layer: usize = if (fault == .dense_prefix and expert == 2 and pi == 1) 2 else 3;
            const rows: usize = if (pi == 2) hidden else intermediate;
            const cols: usize = if (fault == .wrong_shape and expert == 2 and pi == 1) hidden - 1 else if (pi == 2) intermediate else hidden;
            const parts: usize = if (storage == .fp8) 2 else 1;
            for (0..parts) |part| {
                if (part == 1 and fault == .fp8_scale and expert == 2 and pi == 1) continue;
                const key = try std.fmt.allocPrint(allocator, "model.language_model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ layer, expert, projection, if (part == 0) "weight" else "weight_scale_inv" });
                defer allocator.free(key);
                const shape: [2]usize = if (part == 0) .{ rows, cols } else .{ rows / 128, cols / 128 };
                const elem: usize = if (part == 1) 4 else if (storage == .fp8) 1 else 2;
                const bytes = shape[0] * shape[1] * elem;
                const dtype: []const u8 = if (part == 1) "F32" else if (storage == .fp8) "F8_E4M3" else if (fault == .fp16 and expert == 2 and pi == 1) "F16" else "BF16";
                const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ if (count == 0) "" else ",", key, dtype, shape[0], shape[1], offset, offset + bytes });
                defer allocator.free(entry);
                try header.appendSlice(allocator, entry);
                const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"{s}.safetensors\"", .{ if (entries == 0) "" else ",", key, projection });
                defer allocator.free(mapping);
                try index.appendSlice(allocator, mapping);
                for (0..shape[0] * shape[1]) |i| {
                    if (part == 1) {
                        std.mem.writeInt(u32, payload[offset + i * 4 ..][0..4], @bitCast(fp8Scale(expert, pi, i)), .little);
                    } else if (storage == .fp8) {
                        payload[offset + i] = fp8Code(expert, pi, i);
                    } else std.mem.writeInt(u16, payload[offset + i * 2 ..][0..2], value(expert, pi), .little);
                }
                offset += bytes;
                count += 1;
                entries += 1;
            }
        }
        try header.append(allocator, '}');
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(allocator);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.items.len, .little);
        try bytes.appendSlice(allocator, &len);
        try bytes.appendSlice(allocator, header.items);
        try bytes.appendSlice(allocator, payload[0..offset]);
        const filename = try std.fmt.allocPrint(allocator, "{s}.safetensors", .{projection});
        defer allocator.free(filename);
        const trim: usize = if (fault == .truncated and pi == 1) 1 else 0;
        try dir.writeFile(io, .{ .sub_path = filename, .data = bytes.items[0 .. bytes.items.len - trim] });
    }
    // Dense/shared weights are not opened by the expert store; nor is the MTP shard.
    try index.appendSlice(allocator, ",\"model.language_model.layers.0.mlp.gate_proj.weight\":\"dense-unopened.safetensors\",\"model.language_model.layers.3.mlp.shared_experts.gate_proj.weight\":\"shared-unopened.safetensors\",\"model.language_model.layers.4.mlp.experts.0.gate_proj.weight\":\"mtp-unopened.safetensors\"");
    if (fault == .mixed) try index.appendSlice(allocator, ",\"model.language_model.layers.3.mlp.switch_mlp.gate_proj.weight\":\"mixed.safetensors\"");
    if (fault == .fp8_scale and storage == .bf16) try index.appendSlice(allocator, ",\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv\":\"fp8.safetensors\"");
    try index.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}
