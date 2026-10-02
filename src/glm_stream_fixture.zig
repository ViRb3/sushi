//! Small, deliberately shuffled source shards for the GLM BF16 stream tests.
const std = @import("std");

pub const Fault = enum { none, missing, fp16, wrong_shape, truncated, dense_prefix, mixed, fp8_scale };

pub fn value(expert: usize, projection: usize) u16 {
    return @intCast(0x3f00 + expert * 16 + projection * 4);
}

pub fn write(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault) !void {
    const io = std.testing.io;
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var entries: usize = 0;
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |projection, pi| {
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(allocator);
        try header.append(allocator, '{');
        var payload: [4 * 32 * 16 * 2]u8 = undefined;
        var offset: usize = 0;
        var count: usize = 0;
        for ([_]usize{ 3, 1, 0, 2 }) |expert| {
            if (fault == .missing and expert == 2 and pi == 1) continue;
            const layer: usize = if (fault == .dense_prefix and expert == 2 and pi == 1) 2 else 3;
            const key = try std.fmt.allocPrint(allocator, "model.language_model.layers.{d}.mlp.experts.{d}.{s}_proj.weight", .{ layer, expert, projection });
            defer allocator.free(key);
            const rows: usize = if (pi == 2) 32 else 16;
            const cols: usize = if (fault == .wrong_shape and expert == 2 and pi == 1) 31 else if (pi == 2) 16 else 32;
            const bytes = rows * cols * 2;
            const dtype: []const u8 = if (fault == .fp16 and expert == 2 and pi == 1) "F16" else "BF16";
            const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ if (count == 0) "" else ",", key, dtype, rows, cols, offset, offset + bytes });
            defer allocator.free(entry);
            try header.appendSlice(allocator, entry);
            const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"{s}.safetensors\"", .{ if (entries == 0) "" else ",", key, projection });
            defer allocator.free(mapping);
            try index.appendSlice(allocator, mapping);
            for (0..rows * cols) |i| std.mem.writeInt(u16, payload[offset + i * 2 ..][0..2], value(expert, pi), .little);
            offset += bytes;
            count += 1;
            entries += 1;
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
    if (fault == .fp8_scale) try index.appendSlice(allocator, ",\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv\":\"fp8.safetensors\"");
    try index.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}
