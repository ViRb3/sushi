//! Loopback-only native GLM HTTP benchmark bridge; public serving gates stay unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const chat = @import("chat.zig");
const forward = @import("glm5_forward.zig");
const tokenizer = @import("tokenizer.zig");
const native = @import("glm5_diagnostic.zig");
const dflash = @import("dflash.zig");
const adapter = @import("glm5_dflash.zig");
const net = @import("server.zig");
const timing = @import("io_util.zig");
const Arr = mlx.mlx_array;

const Completion = struct {
    parsed: std.json.Parsed(std.json.Value),
    messages: []chat.Message,
    allocator: std.mem.Allocator,
    stream: bool,
    include_usage: bool,
    max_tokens: usize,
    ignore_eos: bool,
    thinking: bool,
    effort: ?[]const u8,
    fn deinit(self: *Completion) void {
        self.allocator.free(self.messages);
        self.parsed.deinit();
    }
};
fn boolField(root: std.json.ObjectMap, name: []const u8, fallback: bool) !bool {
    const value = root.get(name) orelse return fallback;
    if (value != .bool) return error.BenchmarkInvalidRequest;
    return value.bool;
}
fn intField(root: std.json.ObjectMap, name: []const u8, fallback: usize) !usize {
    const value = root.get(name) orelse return fallback;
    if (value != .integer or value.integer < 0) return error.BenchmarkInvalidRequest;
    return std.math.cast(usize, value.integer) orelse error.BenchmarkInvalidRequest;
}
fn parseCompletion(a: std.mem.Allocator, body: []const u8, expected_model: []const u8, max_tokens: usize, allow_ignore_eos: bool) !Completion {
    var parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    errdefer parsed.deinit();
    if (parsed.value != .object) return error.BenchmarkInvalidRequest;
    const root = parsed.value.object;
    if (root.get("tools")) |value| if (value != .array or value.array.items.len != 0) return error.BenchmarkToolsUnsupported;
    if (root.get("tool_choice")) |value| if (value != .string or !std.mem.eql(u8, value.string, "none")) return error.BenchmarkToolsUnsupported;
    if (root.get("response_format")) |_| return error.BenchmarkStructuredOutputUnsupported;
    if (root.get("stop")) |value| if (value != .array or value.array.items.len != 0) return error.BenchmarkStopUnsupported;
    if (root.get("model")) |value| {
        if (value != .string) return error.BenchmarkInvalidRequest;
        if (!std.mem.eql(u8, expected_model, value.string)) return error.BenchmarkModelNotFound;
    }
    if (root.get("temperature")) |value| {
        const temperature = switch (value) {
            .integer => @as(f64, @floatFromInt(value.integer)),
            .float => value.float,
            else => return error.BenchmarkInvalidRequest,
        };
        if (temperature != 0) return error.BenchmarkGreedyOnly;
    }
    if (try intField(root, "n", 1) != 1) return error.BenchmarkMultipleChoicesUnsupported;
    const limit = if (root.get("max_completion_tokens") != null) try intField(root, "max_completion_tokens", 256) else try intField(root, "max_tokens", 256);
    if (limit == 0 or limit > max_tokens) return error.BenchmarkInvalidCompletionBudget;
    const ignore = try boolField(root, "ignore_eos", false);
    if (ignore and !allow_ignore_eos) return error.BenchmarkIgnoreEosDisabled;
    const minimum = try intField(root, "min_tokens", 0);
    if (minimum > limit or (minimum > 0 and !ignore)) return error.BenchmarkInvalidCompletionBudget;
    var effort: ?[]const u8 = null;
    var thinking = false;
    if (root.get("reasoning_effort")) |value| {
        if (value != .string) return error.BenchmarkInvalidReasoningEffort;
        if (!std.mem.eql(u8, value.string, "none") and !std.mem.eql(u8, value.string, "off") and !std.mem.eql(u8, value.string, "low") and !std.mem.eql(u8, value.string, "medium") and !std.mem.eql(u8, value.string, "high")) return error.BenchmarkInvalidReasoningEffort;
        thinking = !std.mem.eql(u8, value.string, "none") and !std.mem.eql(u8, value.string, "off");
        if (thinking) effort = value.string;
    }
    thinking = try boolField(root, "enable_thinking", thinking);
    if (root.get("chat_template_kwargs")) |kwargs| {
        if (kwargs != .object) return error.BenchmarkInvalidRequest;
        thinking = try boolField(kwargs.object, "enable_thinking", thinking);
    }
    const items = root.get("messages") orelse return error.BenchmarkMessagesRequired;
    if (items != .array or items.array.items.len == 0 or items.array.items.len > 128) return error.BenchmarkInvalidMessages;
    const messages = try a.alloc(chat.Message, items.array.items.len);
    errdefer a.free(messages);
    for (messages, items.array.items) |*message, item| {
        if (item != .object) return error.BenchmarkInvalidMessages;
        if (item.object.get("tool_calls")) |value| if (value != .array or value.array.items.len != 0) return error.BenchmarkToolsUnsupported;
        const role = item.object.get("role") orelse return error.BenchmarkInvalidMessages;
        const content = item.object.get("content") orelse return error.BenchmarkInvalidMessages;
        if (role != .string or content != .string) return error.BenchmarkTextMessagesOnly;
        if (!std.mem.eql(u8, role.string, "system") and !std.mem.eql(u8, role.string, "user") and !std.mem.eql(u8, role.string, "assistant")) return error.BenchmarkToolsUnsupported;
        message.* = .{ .role = role.string, .content = content.string };
    }
    var include_usage = false;
    if (root.get("stream_options")) |value| {
        if (value != .object) return error.BenchmarkInvalidRequest;
        include_usage = try boolField(value.object, "include_usage", false);
    }
    return .{ .parsed = parsed, .include_usage = include_usage, .messages = messages, .allocator = a, .stream = try boolField(root, "stream", false), .max_tokens = limit, .ignore_eos = ignore, .thinking = thinking, .effort = effort };
}

test "GLM benchmark HTTP rejects tools and preserves exact completion token bound" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.BenchmarkToolsUnsupported, parseCompletion(a, "{\"model\":\"glm\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\"}]}", "glm", 4096, false));
    var value = try parseCompletion(a, "{\"model\":\"glm\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true,\"max_completion_tokens\":256,\"ignore_eos\":true,\"min_tokens\":256}", "glm", 4096, true);
    defer value.deinit();
    try std.testing.expectEqual(@as(usize, 256), value.max_tokens);
    try std.testing.expect(value.stream and value.ignore_eos);
    try std.testing.expectEqualStrings("hello", value.messages[0].content);
}

const Options = struct {
    directory: []const u8,
    assistant: ?[]const u8 = null,
    port: u16 = 8094,
    context: usize = 132096,
    chunk: usize = 2048,
    memory_gib: usize = 110,
    max_tokens: usize = 4096,
    allow_ignore_eos: bool = false,
};
fn parseOptions(args: []const []const u8) !Options {
    if (args.len == 0 or args[0].len == 0 or args[0][0] == '-') return error.BenchmarkModelRequired;
    var result = Options{ .directory = args[0] };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--benchmark-ignore-eos")) {
            result.allow_ignore_eos = true;
            continue;
        }
        i += 1;
        if (i >= args.len) return error.BenchmarkArgumentValueRequired;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--assistant")) result.assistant = value else if (std.mem.eql(u8, arg, "--port")) result.port = try std.fmt.parseInt(u16, value, 10) else if (std.mem.eql(u8, arg, "--ctx-size")) result.context = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--prefill-chunk")) result.chunk = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--memory-gib")) result.memory_gib = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--max-tokens")) result.max_tokens = try std.fmt.parseInt(usize, value, 10) else return error.BenchmarkUnknownOption;
    }
    if (result.context == 0 or result.chunk == 0 or result.chunk > 2048 or result.memory_gib == 0 or result.max_tokens == 0 or result.max_tokens > 4096) return error.BenchmarkInvalidBudget;
    return result;
}
fn readyUtf8(bytes: []const u8, final: bool) usize {
    var i: usize = 0;
    while (i < bytes.len) {
        const width = std.unicode.utf8ByteSequenceLength(bytes[i]) catch {
            i += 1;
            continue;
        };
        if (!final and i + width > bytes.len) {
            var continuation = true;
            for (bytes[i + 1 ..]) |b| if (b & 0xc0 != 0x80) {
                continuation = false;
            };
            if (continuation) return i;
        }
        i = chat.utf8Next(bytes, i).end;
    }
    return i;
}
const Utf8Buffer = struct {
    pending: std.ArrayList(u8) = .empty,
    fn deinit(self: *Utf8Buffer, a: std.mem.Allocator) void {
        self.pending.deinit(a);
    }
    fn append(self: *Utf8Buffer, a: std.mem.Allocator, bytes: []const u8, final: bool) ![]u8 {
        try self.pending.appendSlice(a, bytes);
        const n = readyUtf8(self.pending.items, final);
        const output = try chat.utf8Sanitize(a, self.pending.items[0..n]);
        std.mem.copyForwards(u8, self.pending.items, self.pending.items[n..]);
        self.pending.items.len -= n;
        return output;
    }
};
fn contextFits(prompt: usize, output: usize, context: usize) bool {
    return prompt > 0 and output > 0 and prompt <= context and output <= context - prompt;
}
fn capacity(rows: usize) !usize {
    return (try std.math.add(usize, rows, 255)) / 256 * 256;
}
fn scratchAdmits(cfg: *const model.ModelConfig, total: usize) bool {
    const reserved = capacity(total +| 3) catch return false;
    _ = @import("glm5_dflash_memory.zig").plan(total, reserved, capacity((total +| 3) / 4) catch return false, cfg.mla_kv_lora_rank, cfg.indexer_head_dim, cfg.num_attention_heads, 3, 2) catch return false;
    return true;
}
fn scratchLimit(cfg: *const model.ModelConfig, limit: usize) usize {
    var lo: usize = 1;
    var hi = limit;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (scratchAdmits(cfg, mid)) lo = mid else hi = mid - 1;
    }
    return lo;
}
fn requestReserve(cfg: *const model.ModelConfig, draft_cfg: ?*const dflash.DflashConfig, tokens: usize, chunk: usize) !usize {
    if (cfg.full_attention_interval == 0) return error.BenchmarkInvalidModel;
    const rows: u128 = try capacity(try std.math.add(usize, tokens, 3));
    const mla: u128 = cfg.num_hidden_layers / cfg.full_attention_interval;
    const kda_layers: u128 = cfg.num_hidden_layers - cfg.num_hidden_layers / cfg.full_attention_interval;
    const width: u128 = @as(u128, cfg.linear_num_value_heads) * cfg.linear_key_head_dim;
    const recurrent = kda_layers * width * (@as(u128, cfg.linear_key_head_dim) * 4 + 3 * 3 * 2) * 4;
    // Immutable cache growth can retain old, concatenated, updated and branch views.
    const latent = mla * rows * (@as(u128, cfg.mla_kv_lora_rank) * 2 + @as(u128, cfg.indexer_head_dim) * 2 / 4) * 4;
    var assistant_cache: u128 = 0;
    if (draft_cfg) |dc| assistant_cache = rows * dc.num_hidden_layers * dc.num_key_value_heads * dc.head_dim * 2 * 2 * 4;
    const activations = @as(u128, chunk) * (@as(u128, cfg.hidden_size) * 64 + width * 32 + @as(u128, cfg.num_experts_per_tok) * (@as(u128, cfg.moe_intermediate_size) * 8 + @as(u128, cfg.hidden_size) * 4)) * 2;
    const scratch = @as(u128, @import("glm5_attention.zig").score_scratch_bytes + @import("glm5_attention.zig").attention_scratch_bytes) * 2 + @import("glm5_dflash_memory.zig").limit_bytes;
    const expanded = try @import("glm5_a6_dense_once.zig").transientBudget(chunk, 2);
    const mla_batch = try @import("glm5_mla_prefill_batch.zig").transientBudget(chunk, 2);
    const packed_bytes = try @import("glm5_attention_nax_packed.zig").transientBudget(chunk, 2);
    const index_bytes = try @import("glm5_indexpool_nax.zig").transientBudget(chunk, 2);
    return std.math.cast(usize, recurrent + latent + assistant_cache + activations + scratch + expanded + mla_batch + packed_bytes + index_bytes + 256 * 1024 * 1024) orelse error.BenchmarkMemoryOverflow;
}

fn plannedGrowth(cfg: *const model.ModelConfig, prompt: usize, output: usize) !usize {
    const reserve = @import("glm5_dflash_reserve.zig");
    const total = try std.math.add(usize, try std.math.add(usize, prompt, output), 3);
    const p = try reserve.plan(prompt, try reserve.capacity(prompt), try reserve.capacity(prompt / 4), cfg.mla_kv_lora_rank, cfg.indexer_head_dim, 2, 2, total);
    return std.math.mul(usize, p.additional_peak_bytes, cfg.num_hidden_layers / cfg.full_attention_interval);
}

const Frame = struct {
    id: []const u8,
    object: []const u8 = "chat.completion.chunk",
    created: i64,
    model: []const u8,
    choices: []const Choice,
    usage: ?Usage = null,
    const Choice = struct {
        index: usize = 0,
        delta: Delta,
        finish_reason: ?[]const u8 = null,
        pub fn jsonStringify(self: Choice, writer: anytype) !void {
            try writer.beginObject();
            try writer.objectField("index");
            try writer.write(self.index);
            try writer.objectField("delta");
            try writer.write(self.delta);
            try writer.objectField("finish_reason");
            try writer.write(self.finish_reason);
            try writer.endObject();
        }
    };
    const Delta = struct { role: ?[]const u8 = null, content: ?[]const u8 = null };
};
const Usage = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    prompt_tokens_details: struct { cached_tokens: usize = 0 } = .{},
};
fn usage(prompt: usize, output: usize) Usage {
    return .{ .prompt_tokens = prompt, .completion_tokens = output, .total_tokens = prompt + output };
}
fn frameJson(a: std.mem.Allocator, id: []const u8, name: []const u8, created: i64, text: ?[]const u8, finish: ?[]const u8, role: bool) ![]u8 {
    const choice = [_]Frame.Choice{.{ .delta = .{ .role = if (role) "assistant" else null, .content = text }, .finish_reason = finish }};
    return std.json.Stringify.valueAlloc(a, Frame{ .id = id, .model = name, .created = created, .choices = &choice }, .{ .emit_null_optional_fields = false });
}

test "GLM benchmark HTTP incomplete UTF8 streams without replacing a split character" {
    const a = std.testing.allocator;
    var bytes: Utf8Buffer = .{};
    defer bytes.deinit(a);
    const first = try bytes.append(a, "x\xe4", false);
    defer a.free(first);
    try std.testing.expectEqualStrings("x", first);
    const second = try bytes.append(a, "\xb8\xad", false);
    defer a.free(second);
    try std.testing.expectEqualStrings("中", second);
    const last = try bytes.append(a, "\xff\xe4", true);
    defer a.free(last);
    try std.testing.expectEqualStrings("\u{FFFD}\u{FFFD}", last);
}

test "GLM benchmark HTTP SSE usage counts emitted IDs and finish exactly once" {
    const a = std.testing.allocator;
    const text = try frameJson(a, "id", "real-model", 42, "a\n中", null, false);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("a\n中", parsed.value.object.get("choices").?.array.items[0].object.get("delta").?.object.get("content").?.string);
    try std.testing.expectEqualStrings("real-model", parsed.value.object.get("model").?.string);
    try std.testing.expect(parsed.value.object.get("choices").?.array.items[0].object.get("finish_reason").? == .null);
    const u = usage(2048, 256);
    try std.testing.expectEqual(@as(usize, 2304), u.total_tokens);
    try std.testing.expectEqual(@as(usize, 256), u.completion_tokens);
}

test "GLM benchmark HTTP context admission never truncates and preserves DFlash scratch limit" {
    try std.testing.expect(contextFits(2048, 256, 2304));
    try std.testing.expect(!contextFits(2048, 257, 2304));
    try std.testing.expect(!contextFits(std.math.maxInt(usize), 1, 2304));
    const cfg = model.ModelConfig{ .num_hidden_layers = 45, .full_attention_interval = 4, .hidden_size = 4096, .mla_kv_lora_rank = 512, .indexer_head_dim = 128, .num_attention_heads = 64, .linear_num_value_heads = 64, .linear_key_head_dim = 128 };
    try std.testing.expect(scratchAdmits(&cfg, 2048));
    try std.testing.expect(scratchAdmits(&cfg, 131072));
    try std.testing.expectEqual(@as(usize, 132096), scratchLimit(&cfg, 132096));
    try std.testing.expectError(error.GlmTreeScratchLimit, @import("glm5_dflash_memory.zig").plan(131072, 131072, 32768, 512, 128, 64, 3, 2));
    try std.testing.expect((try requestReserve(&cfg, null, 131072, 2048)) > (try requestReserve(&cfg, null, 65536, 2048)));
}

test "GLM benchmark HTTP admission includes head-batched MLA permutation buffers" {
    const batch = @import("glm5_mla_prefill_batch.zig");
    const cfg = model.ModelConfig{ .num_hidden_layers = 45, .full_attention_interval = 4, .hidden_size = 4096, .mla_kv_lora_rank = 512, .indexer_head_dim = 128, .num_attention_heads = 64, .linear_num_value_heads = 64, .linear_key_head_dim = 128 };
    const baseline = blk: {
        const binding = batch.bind(false);
        defer binding.restore();
        break :blk try requestReserve(&cfg, null, 32768, 2048);
    };
    const binding = batch.bind(true);
    defer binding.restore();
    try std.testing.expectEqual(try batch.transientBudget(2048, 2), (try requestReserve(&cfg, null, 32768, 2048)) - baseline);
}

test "GLM benchmark HTTP admission includes bounded packed attention banks" {
    const headpack = @import("glm5_attention_nax_packed.zig");
    const cfg = model.ModelConfig{ .num_hidden_layers = 45, .full_attention_interval = 4, .hidden_size = 4096, .mla_kv_lora_rank = 512, .indexer_head_dim = 128, .num_attention_heads = 64, .linear_num_value_heads = 64, .linear_key_head_dim = 128 };
    const baseline = blk: {
        const binding = headpack.bind(false);
        defer binding.restore();
        break :blk try requestReserve(&cfg, null, 32768, 2048);
    };
    const binding = headpack.bind(true);
    defer binding.restore();
    try std.testing.expectEqual(2 * headpack.scratch_limit, (try requestReserve(&cfg, null, 32768, 2048)) - baseline);
}

test "GLM benchmark HTTP admission includes bounded NAX index score planes" {
    const indexer = @import("glm5_indexpool_nax.zig");
    const cfg = model.ModelConfig{ .num_hidden_layers = 45, .full_attention_interval = 4, .hidden_size = 4096, .mla_kv_lora_rank = 512, .indexer_head_dim = 128, .num_attention_heads = 64, .linear_num_value_heads = 64, .linear_key_head_dim = 128 };
    const baseline = blk: {
        const binding = indexer.bind(false);
        defer binding.restore();
        break :blk try requestReserve(&cfg, null, 32768, 2048);
    };
    const binding = indexer.bind(true);
    defer binding.restore();
    try std.testing.expectEqual(try indexer.transientBudget(2048, 2), (try requestReserve(&cfg, null, 32768, 2048)) - baseline);
}

test "GLM benchmark HTTP CLI is explicit assistant and benchmark EOS mode" {
    const opts = try parseOptions(&.{ "/model", "--assistant", "/assistant", "--ctx-size", "132096", "--benchmark-ignore-eos" });
    try std.testing.expectEqualStrings("/assistant", opts.assistant.?);
    try std.testing.expect(opts.allow_ignore_eos);
    try std.testing.expectError(error.BenchmarkUnknownOption, parseOptions(&.{ "/model", "--host", "0.0.0.0" }));
    try std.testing.expectError(error.BenchmarkInvalidBudget, parseOptions(&.{ "/model", "--prefill-chunk", "0" }));
}

const HttpRequest = struct { method: []const u8, path: []const u8, body: []const u8 };
const Head = struct { method: []const u8, path: []const u8, content_length: usize };
fn parseHead(bytes: []const u8) !Head {
    var lines = std.mem.splitSequence(u8, bytes, "\r\n");
    const first = lines.next() orelse return error.BenchmarkInvalidHttp;
    var words = std.mem.splitScalar(u8, first, ' ');
    const method = words.next() orelse return error.BenchmarkInvalidHttp;
    const target = words.next() orelse return error.BenchmarkInvalidHttp;
    const version = words.next() orelse return error.BenchmarkInvalidHttp;
    if (words.next() != null or (!std.mem.eql(u8, version, "HTTP/1.1") and !std.mem.eql(u8, version, "HTTP/1.0"))) return error.BenchmarkInvalidHttp;
    const path = target[0..(std.mem.indexOfScalar(u8, target, '?') orelse target.len)];
    var length: ?usize = null;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BenchmarkInvalidHttp;
        const name = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "Transfer-Encoding")) return error.BenchmarkChunkedRequestUnsupported;
        if (std.ascii.eqlIgnoreCase(name, "Content-Length")) {
            if (length != null) return error.BenchmarkDuplicateContentLength;
            length = std.fmt.parseInt(usize, value, 10) catch return error.BenchmarkInvalidContentLength;
        }
    }
    if (std.mem.eql(u8, method, "POST") and length == null) return error.BenchmarkContentLengthRequired;
    return .{ .method = method, .path = path, .content_length = length orelse 0 };
}
fn readHttp(a: std.mem.Allocator, conn: *net.Conn) !HttpRequest {
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(a);
    var buffer: [4096]u8 = undefined;
    var head_end: ?usize = null;
    while (head_end == null) {
        const count = try conn.read(&buffer);
        if (count == 0) return error.BenchmarkIncompleteHttp;
        try data.appendSlice(a, buffer[0..count]);
        if (std.mem.indexOf(u8, data.items, "\r\n\r\n")) |end| head_end = end + 4;
        if ((head_end != null and head_end.? > 16 * 1024) or (head_end == null and data.items.len > 16 * 1024)) return error.BenchmarkHeadersTooLarge;
    }
    const head = try parseHead(data.items[0 .. head_end.? - 4]);
    if (head.content_length > 8 * 1024 * 1024) return error.BenchmarkBodyTooLarge;
    const total = try std.math.add(usize, head_end.?, head.content_length);
    while (data.items.len < total) {
        const count = try conn.read(buffer[0..@min(buffer.len, total - data.items.len)]);
        if (count == 0) return error.BenchmarkIncompleteBody;
        try data.appendSlice(a, buffer[0..count]);
    }
    const saved = try a.dupe(u8, data.items[0..total]);
    const saved_head = try parseHead(saved[0 .. head_end.? - 4]);
    return .{ .method = saved_head.method, .path = saved_head.path, .body = saved[head_end.?..] };
}
fn sendJson(a: std.mem.Allocator, conn: *net.Conn, status: []const u8, value: anytype) !void {
    const body = try std.json.Stringify.valueAlloc(a, value, .{ .emit_null_optional_fields = false });
    defer a.free(body);
    const head = try std.fmt.allocPrint(a, "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ status, body.len });
    defer a.free(head);
    conn.length_framed = true;
    try conn.writeAllNoFlush(head);
    try conn.writeAll(body);
}
fn sendEvent(a: std.mem.Allocator, conn: *net.Conn, value: anytype) !void {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{ .emit_null_optional_fields = false });
    defer a.free(bytes);
    try conn.writeAllNoFlush("data: ");
    try conn.writeAllNoFlush(bytes);
    try conn.writeAll("\n\n");
}
fn sendRawEvent(conn: *net.Conn, bytes: []const u8) !void {
    try conn.writeAllNoFlush("data: ");
    try conn.writeAllNoFlush(bytes);
    try conn.writeAll("\n\n");
}
fn failure(a: std.mem.Allocator, conn: *net.Conn, code: []const u8, message: []const u8, maximum: usize) !void {
    const value = .{ .@"error" = .{ .message = message, .type = "diagnostic_benchmark_error", .code = code, .max_allowed_context = maximum }, .diagnostic_only = true };
    if (conn.sse_headers_sent) try sendEvent(a, conn, value) else try sendJson(a, conn, if (std.mem.eql(u8, code, "BenchmarkModelNotFound")) "404 Not Found" else "400 Bad Request", value);
}
fn streamHeaders(conn: *net.Conn) !void {
    try conn.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
    conn.sse_headers_sent = true;
}
fn verifiedTail(tokens: []const u32, pending: u32) ![]const u32 {
    if (tokens.len == 0 or tokens[0] != pending) return error.GlmBenchmarkPendingMismatch;
    return tokens[1..];
}

test "GLM benchmark HTTP frame parser rejects ambiguous body bounds" {
    const head = try parseHead("POST /v1/chat/completions?x=1 HTTP/1.1\r\nhost: localhost\r\ncontent-length: 123");
    try std.testing.expectEqual(@as(usize, 123), head.content_length);
    try std.testing.expectEqualStrings("/v1/chat/completions", head.path);
    try std.testing.expectError(error.BenchmarkDuplicateContentLength, parseHead("POST / HTTP/1.1\r\nContent-Length: 1\r\ncontent-length: 2"));
    try std.testing.expectError(error.BenchmarkChunkedRequestUnsupported, parseHead("POST / HTTP/1.1\r\nTransfer-Encoding: chunked"));
    try std.testing.expectError(error.BenchmarkContentLengthRequired, parseHead("POST / HTTP/1.1\r\nHost: localhost"));
}

test "GLM benchmark HTTP DFlash emits pending once and counts verified native IDs" {
    try std.testing.expectEqualSlices(u32, &.{ 8, 9 }, try verifiedTail(&.{ 7, 8, 9 }, 7));
    try std.testing.expectEqual(@as(usize, 0), (try verifiedTail(&.{7}, 7)).len);
    try std.testing.expectError(error.GlmBenchmarkPendingMismatch, verifiedTail(&.{ 8, 9 }, 7));
    try std.testing.expectError(error.GlmBenchmarkPendingMismatch, verifiedTail(&.{}, 7));
}

const Emission = struct {
    a: std.mem.Allocator,
    conn: *net.Conn,
    tok: *const tokenizer.Tokenizer,
    id: []const u8,
    name: []const u8,
    created: i64,
    stream: bool,
    text: std.ArrayList(u8) = .empty,
    ids: std.ArrayList(u32) = .empty,
    utf8: Utf8Buffer = .{},
    fn deinit(self: *Emission) void {
        self.text.deinit(self.a);
        self.ids.deinit(self.a);
        self.utf8.deinit(self.a);
    }
    fn textPart(self: *Emission, part: []const u8) !void {
        if (part.len == 0) return;
        try self.text.appendSlice(self.a, part);
        if (self.stream) {
            const frame = try frameJson(self.a, self.id, self.name, self.created, part, null, false);
            defer self.a.free(frame);
            try sendRawEvent(self.conn, frame);
        }
    }
    fn token(self: *Emission, id: u32) !void {
        try self.ids.append(self.a, id);
        const bytes = try self.tok.decode(self.a, &.{id}, false);
        defer self.a.free(bytes);
        const part = try self.utf8.append(self.a, bytes, false);
        defer self.a.free(part);
        try self.textPart(part);
    }
    fn flushText(self: *Emission) !void {
        const part = try self.utf8.append(self.a, "", true);
        defer self.a.free(part);
        try self.textPart(part);
    }
};

var shutdown = std.atomic.Value(bool).init(false);
fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    shutdown.store(true, .release);
}
const Runtime = struct {
    options: Options,
    cfg: *const model.ModelConfig,
    target: *const forward.Model,
    assistant: ?*dflash.DflashModel,
    tok: *const tokenizer.Tokenizer,
    chat_config: *const chat.ChatConfig,
    name: []const u8,
    version: []const u8,
    budget: usize,
    resident: usize,
    sequence: usize = 0,
    fn backend(self: *const Runtime) []const u8 {
        return if (self.assistant != null) "native-glm-dflash2-n2c4-async4" else "native-glm-serial-async4";
    }
    fn metadata(self: *const Runtime) struct {
        diagnostic_only: bool = true,
        benchmark_only: bool = true,
        public_serving_enabled: bool = false,
        engine: []const u8 = "sushi-native-glm-diagnostic",
        backend: []const u8,
        mlx_version: []const u8,
        assistant: ?[]const u8,
        sampling: []const u8 = "greedy",
        prefix_cache: bool = false,
        max_parallel: usize = 1,
        kv: []const u8 = "BF16 compressed MLA cache; FP32 KDA state",
        prefill_chunk: usize,
        prefill_async_layers: usize = 2,
        verify_nodes: usize,
        verify_children: usize,
        verify_async_layers: usize,
        group2: bool,
        lane_pair: bool,
        down_lane: bool,
        dense_rows: bool,
        mini_head: bool,
        a6_dense_prefill: bool,
        mla_headbatch: bool,
        packed_attention: bool,
        nax_index_scores: bool,
        verify_mla_batch: bool,
        bounded_draft_readout: bool,
        kda_keep_leaf: bool,
        kda_value_rows: usize,
        benchmark_ignore_eos_enabled: bool,
    } {
        const diag = @import("transformer.zig");
        return .{
            .backend = self.backend(),
            .mlx_version = self.version,
            .assistant = self.options.assistant,
            .prefill_chunk = self.options.chunk,
            .verify_nodes = if (self.assistant != null) 2 else 0,
            .verify_children = if (self.assistant != null) 4 else 0,
            .verify_async_layers = if (self.assistant != null) 4 else 0,
            .group2 = diag.diagEnvOn("SUSHI_GLM_DFLASH_GROUP2"),
            .lane_pair = diag.diagEnvOn("SUSHI_GLM_LANE_PAIR"),
            .down_lane = diag.diagEnvOn("SUSHI_GLM_DOWN_LANE"),
            .dense_rows = @import("glm5_dflash_dense_rows.zig").enabled(),
            .mini_head = if (self.assistant) |draft_model| draft_model.draft_head != null else false,
            .a6_dense_prefill = @import("glm5_a6_dense_once.zig").enabled(),
            .mla_headbatch = @import("glm5_mla_prefill_batch.zig").enabled(),
            .packed_attention = @import("glm5_attention_nax_packed.zig").enabled(),
            .nax_index_scores = @import("glm5_indexpool_nax.zig").enabled(),
            .verify_mla_batch = @import("glm5_mla_verify_batch.zig").enabled(),
            .bounded_draft_readout = diag.diagEnvOn("SUSHI_GLM_DFLASH_READOUT_HORIZON"),
            .kda_keep_leaf = diag.diagEnvOn("SUSHI_GLM_KDA_KEEP_LEAF"),
            .kda_value_rows = (@import("glm5_kda_value_rows.zig").configuredRows() catch null) orelse 0,
            .benchmark_ignore_eos_enabled = self.options.allow_ignore_eos,
        };
    }
    fn models(self: *const Runtime, a: std.mem.Allocator, conn: *net.Conn) !void {
        try sendJson(a, conn, "200 OK", .{ .object = "list", .data = [_]@TypeOf(.{
            .id = self.name,
            .object = "model",
            .created = @as(i64, 0),
            .owned_by = "sushi-diagnostic",
            .context_length = self.options.context,
            .max_model_len = self.options.context,
            .has_tools = false,
            .has_reasoning = true,
            .benchmark_only = true,
            .diagnostic_only = true,
            .meta = self.metadata(),
        }){.{
            .id = self.name,
            .object = "model",
            .created = @as(i64, 0),
            .owned_by = "sushi-diagnostic",
            .context_length = self.options.context,
            .max_model_len = self.options.context,
            .has_tools = false,
            .has_reasoning = true,
            .benchmark_only = true,
            .diagnostic_only = true,
            .meta = self.metadata(),
        }} });
    }
    fn props(self: *const Runtime, a: std.mem.Allocator, conn: *net.Conn) !void {
        var active: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&active));
        try sendJson(a, conn, "200 OK", .{
            .settings = self.metadata(),
            .default_generation_settings = .{ .n_ctx = self.options.context, .temperature = @as(usize, 0) },
            .model_info = .{ .id = self.name, .max_position_embeddings = self.cfg.max_position_embeddings },
            .memory_limit_bytes = self.budget,
            .active_bytes = active,
            .admission = "checked context, cache growth, assistant KV and DFlash branch scratch",
        });
    }
    fn memoryContextLimit(self: *const Runtime, output: usize) usize {
        if (output >= self.options.context) return 0;
        var lo: usize = output + 1;
        var hi = self.options.context;
        const dc = if (self.assistant) |draft_model| &draft_model.config else null;
        while (lo < hi) {
            const mid = lo + (hi - lo + 1) / 2;
            const base_bill = requestReserve(self.cfg, dc, mid, self.options.chunk) catch return 0;
            const growth = if (self.assistant != null) plannedGrowth(self.cfg, mid - output, output) catch return 0 else 0;
            const bill = std.math.add(usize, base_bill, growth) catch return 0;
            if (@import("glm5_a6_dense_once.zig").budgetFits(self.resident, bill, self.budget)) lo = mid else hi = mid - 1;
        }
        return lo;
    }
    fn complete(self: *Runtime, a: std.mem.Allocator, conn: *net.Conn, body: []const u8) !void {
        var completion = try parseCompletion(a, body, self.name, self.options.max_tokens, self.options.allow_ignore_eos);
        defer completion.deinit();
        const ids = try chat.formatChat(a, self.tok, completion.messages, self.chat_config, null, null, completion.thinking, completion.effort, false);
        defer a.free(ids);
        if (!contextFits(ids.len, completion.max_tokens, self.options.context)) {
            const message = try std.fmt.allocPrint(a, "Prompt {d} + completion {d} exceeds configured context {d}; input was not truncated", .{ ids.len, completion.max_tokens, self.options.context });
            defer a.free(message);
            return failure(a, conn, "context_length_exceeded", message, self.options.context);
        }
        const total = try std.math.add(usize, ids.len, completion.max_tokens);
        const draft_cfg = if (self.assistant) |draft_model| &draft_model.config else null;
        const reserve = try std.math.add(usize, try requestReserve(self.cfg, draft_cfg, total, self.options.chunk), if (self.assistant != null) try plannedGrowth(self.cfg, ids.len, completion.max_tokens) else 0);
        if (!@import("glm5_a6_dense_once.zig").budgetFits(self.resident, reserve, self.budget)) {
            const message = try std.fmt.allocPrint(a, "Context {d} requires {d} request bytes above {d} resident bytes; budget {d}; input was not truncated", .{ total, reserve, self.resident, self.budget });
            defer a.free(message);
            return failure(a, conn, "glm_context_memory_exceeded", message, self.memoryContextLimit(completion.max_tokens));
        }
        if (self.assistant != null and !scratchAdmits(self.cfg, total)) {
            const maximum = scratchLimit(self.cfg, self.options.context);
            const message = try std.fmt.allocPrint(a, "DFlash context {d} exceeds unchanged branch scratch policy; maximum total context with reserved cache capacity {d}", .{ total, maximum });
            defer a.free(message);
            return failure(a, conn, "glm_dflash_scratch_limit", message, maximum);
        }
        self.sequence += 1;
        const id = try std.fmt.allocPrint(a, "chatcmpl-glmbench-{d}", .{self.sequence});
        defer a.free(id);
        var emit = Emission{ .a = a, .conn = conn, .tok = self.tok, .id = id, .name = self.name, .created = timing.nowSecs(conn.io), .stream = completion.stream };
        defer emit.deinit();
        if (completion.stream) {
            try streamHeaders(conn);
            const role = try frameJson(a, id, self.name, emit.created, null, null, true);
            defer a.free(role);
            try sendRawEvent(conn, role);
        }
        var request = try forward.Request.init(a, self.cfg.num_hidden_layers);
        defer request.deinit();
        @import("glm5_attention_nax_packed.zig").resetDispatchCount();
        @import("glm5_indexpool_nax.zig").resetDispatchCount();
        @import("glm5_mla_verify_batch.zig").resetDispatchCount();
        @import("glm5_dflash_kda.zig").resetLeafStats();
        const horizon_before = adapter.readoutHorizonCalls();
        request.dense_prefill = true;
        request.prefill_async = true;
        request.prefill_sync_layers = 2;
        request.decode_async = true;
        var context: ?dflash.DflashCtx = if (self.assistant) |draft_model| try dflash.DflashCtx.init(a, draft_model, 0) else null;
        defer if (context) |*ctx| ctx.deinit();
        const schedule = try @import("glm5_dflash_model.zig").bindSchedule(if (self.assistant != null) 4 else 0);
        defer schedule.restore();
        var timer = timing.Stopwatch.init(conn.io);
        var cursor: usize = 0;
        var pending: u32 = 0;
        while (cursor < ids.len) {
            if (conn.peerClosed() or shutdown.load(.acquire)) return error.BenchmarkClientDisconnected;
            const end = @min(ids.len, cursor + self.options.chunk);
            const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &.{ 1, @intCast(end - cursor) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            if (self.assistant) |draft_model| pending = try adapter.prefill(draft_model, &context.?, self.target, &request, input) else {
                const logits = try self.target.forwardLast(&request, input, true);
                defer _ = mlx.mlx_array_free(logits);
                pending = try native.greedy(logits, self.target.s);
            }
            cursor = end;
            if (completion.stream and conn.keepaliveDue()) try conn.writeAll(": keepalive\n\n");
        }
        var reserve_bytes: usize = 0;
        if (self.assistant != null) {
            var active: usize = 0;
            try mlx.check(mlx.mlx_get_active_memory(&active));
            const future = try std.math.add(usize, try std.math.add(usize, ids.len, completion.max_tokens), 3);
            reserve_bytes = try @import("glm5_dflash_reserve.zig").reserve(&request, future, self.budget -| active, self.target.s);
        }
        const prefill_ns = timer.read();
        try emit.token(pending);
        timer.reset();
        var stopped = !completion.ignore_eos and self.cfg.isEosToken(pending);
        var rounds: usize = 0;
        var accepted: usize = 0;
        var draft_ns: u64 = 0;
        var verify_ns: u64 = 0;
        var replay_ns: u64 = 0;
        var commit_ns: u64 = 0;
        const eos: []const u32 = if (completion.ignore_eos) &.{} else self.cfg.eosTokenSlice();
        while (!stopped and emit.ids.items.len < completion.max_tokens) {
            if (conn.peerClosed() or shutdown.load(.acquire)) return error.BenchmarkClientDisconnected;
            if (self.assistant) |draft_model| {
                const remaining = completion.max_tokens - emit.ids.items.len;
                const round = try adapter.roundTreeLayerwiseConfigured(conn.io, draft_model, &context.?, self.target, &request, pending, 2, remaining + 1, eos, .affine_rows_ffn, 4);
                rounds += 1;
                accepted += round.accepted_drafts;
                draft_ns += round.draft_ns;
                verify_ns += round.verify_ns;
                replay_ns += round.replay_ns;
                commit_ns += round.commit_ns;
                const tail = try verifiedTail(round.tokens[0..round.count], pending);
                for (tail) |token_id| try emit.token(token_id);
                stopped = round.stopped;
                if (!stopped and emit.ids.items.len < completion.max_tokens) {
                    pending = round.pending orelse return error.MissingGlmPendingToken;
                    try emit.token(pending);
                    stopped = !completion.ignore_eos and self.cfg.isEosToken(pending);
                }
            } else {
                const input = mlx.mlx_array_new_data(&pending, &.{ 1, 1 }, 2, .uint32);
                defer _ = mlx.mlx_array_free(input);
                const logits = try self.target.forwardLast(&request, input, true);
                defer _ = mlx.mlx_array_free(logits);
                pending = try native.greedy(logits, self.target.s);
                try emit.token(pending);
                stopped = !completion.ignore_eos and self.cfg.isEosToken(pending);
            }
            if (completion.stream and conn.keepaliveDue()) try conn.writeAll(": keepalive\n\n");
        }
        const decode_ns = timer.read();
        try emit.flushText();
        const finish = if (stopped) "stop" else "length";
        const counts = usage(ids.len, emit.ids.items.len);
        const stats = .{ .prompt_n = ids.len, .prompt_ms = @as(f64, @floatFromInt(prefill_ns)) / 1e6, .predicted_n = emit.ids.items.len, .predicted_ms = @as(f64, @floatFromInt(decode_ns)) / 1e6 };
        const diagnostic = .{ .reserved_cache_growth_bytes = reserve_bytes, .settings = self.metadata(), .packed_attention_calls = @import("glm5_attention_nax_packed.zig").dispatchCount(), .nax_index_score_calls = @import("glm5_indexpool_nax.zig").dispatchCount(), .verify_mla_query_calls = @import("glm5_mla_verify_batch.zig").dispatchCount(.query), .verify_mla_value_calls = @import("glm5_mla_verify_batch.zig").dispatchCount(.value), .bounded_draft_readout_calls = adapter.readoutHorizonCalls() - horizon_before, .kda_leaf_hits = @import("glm5_dflash_kda.zig").leafHits(), .kda_leaf_misses = @import("glm5_dflash_kda.zig").leafMisses(), .output_ids = emit.ids.items, .ignore_eos = completion.ignore_eos, .speculative_rounds = rounds, .accepted_drafts = accepted, .draft_ns = draft_ns, .verify_ns = verify_ns, .replay_ns = replay_ns, .commit_ns = commit_ns };
        if (completion.stream) {
            const last = try frameJson(a, id, self.name, emit.created, null, finish, false);
            defer a.free(last);
            try sendRawEvent(conn, last);
            if (completion.include_usage) try sendEvent(a, conn, .{ .id = id, .object = "chat.completion.chunk", .created = emit.created, .model = self.name, .choices = [_]Frame.Choice{}, .usage = counts, .timings = stats, .sushi_diagnostic = diagnostic });
            try conn.writeAll("data: [DONE]\n\n");
        } else try sendJson(a, conn, "200 OK", .{ .id = id, .object = "chat.completion", .created = emit.created, .model = self.name, .choices = [_]@TypeOf(.{ .index = @as(usize, 0), .message = .{ .role = "assistant", .content = emit.text.items }, .finish_reason = finish }){.{ .index = 0, .message = .{ .role = "assistant", .content = emit.text.items }, .finish_reason = finish }}, .usage = counts, .timings = stats, .sushi_diagnostic = diagnostic });
        @import("log.zig").info("[glm-bench] request={d} backend={s} prompt={d} output={d} prefill_ms={d:.2} decode_ms={d:.2} rounds={d} ignore_eos={} packed={d} index_nax={d} mla_q={d} mla_v={d} horizon={d} kda_hit={d} kda_miss={d} draft_ns={d} verify_ns={d} replay_ns={d} commit_ns={d}\n", .{ self.sequence, self.backend(), ids.len, emit.ids.items.len, stats.prompt_ms, stats.predicted_ms, rounds, completion.ignore_eos, diagnostic.packed_attention_calls, diagnostic.nax_index_score_calls, diagnostic.verify_mla_query_calls, diagnostic.verify_mla_value_calls, diagnostic.bounded_draft_readout_calls, diagnostic.kda_leaf_hits, diagnostic.kda_leaf_misses, draft_ns, verify_ns, replay_ns, commit_ns });
    }
    fn handle(self: *Runtime, a: std.mem.Allocator, conn: *net.Conn) !void {
        const http = try readHttp(a, conn);
        if (std.mem.eql(u8, http.method, "GET")) {
            if (std.mem.eql(u8, http.path, "/health")) return sendJson(a, conn, "200 OK", .{ .status = "ok", .diagnostic_only = true, .benchmark_only = true });
            if (std.mem.eql(u8, http.path, "/v1/models")) return self.models(a, conn);
            if (std.mem.eql(u8, http.path, "/props")) return self.props(a, conn);
        }
        if (std.mem.eql(u8, http.method, "POST") and std.mem.eql(u8, http.path, "/tokenize")) {
            var completion = try parseCompletion(a, http.body, self.name, self.options.max_tokens, self.options.allow_ignore_eos);
            defer completion.deinit();
            const ids = try chat.formatChat(a, self.tok, completion.messages, self.chat_config, null, null, completion.thinking, completion.effort, false);
            defer a.free(ids);
            return sendJson(a, conn, "200 OK", .{ .model = self.name, .prompt_tokens = ids.len, .diagnostic_only = true });
        }
        if (std.mem.eql(u8, http.method, "POST") and std.mem.eql(u8, http.path, "/v1/chat/completions")) return self.complete(a, conn, http.body);
        return sendJson(a, conn, "404 Not Found", .{ .@"error" = .{ .code = "diagnostic_route_not_found", .message = "Diagnostic benchmark supports /health, /props, /v1/models and /v1/chat/completions" } });
    }
};

pub fn run(a: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    if (args.len == 1 and std.mem.eql(u8, args[0], "--help")) {
        var bytes: [1024]u8 = undefined;
        var output = std.Io.File.stdout().writer(io, &bytes);
        try output.interface.writeAll("usage: sushi glm-bench <model-dir> [--assistant <dir>] [--port 8094] [--ctx-size 132096] [--prefill-chunk 2048] [--memory-gib 110] [--max-tokens 4096] [--benchmark-ignore-eos]\nLoopback-only diagnostic benchmark; greedy text, no tools or public serving dispatch.\n");
        try output.interface.flush();
        return;
    }
    const options = try parseOptions(args);
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.GlmBenchmarkGpuRequired;
    const memory = try std.math.mul(usize, options.memory_gib, 1024 * 1024 * 1024);
    const recommended = mlx.maxRecommendedWorkingSet();
    if (recommended == 0) return error.GlmBenchmarkWorkingSetUnavailable;
    const budget = @min(memory, recommended);
    var old_memory: usize = 0;
    var old_cache: usize = 0;
    var old_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_memory, budget));
    defer _ = mlx.mlx_set_memory_limit(&ignored, old_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, 2 * 1024 * 1024 * 1024));
    defer _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    try mlx.check(mlx.mlx_set_wired_limit(&old_wired, budget));
    defer _ = mlx.mlx_set_wired_limit(&ignored, old_wired);
    var cfg = try model.parseConfig(io, a, options.directory);
    defer cfg.deinit(a);
    if (!cfg.isGlm5() or cfg.expert_layout != .exl3_k4) return error.UnsupportedGlmBenchmarkLayout;
    _ = try @import("glm5_kda_value_rows.zig").configuredRows();
    if (options.context > cfg.max_position_embeddings) return error.GlmBenchmarkContextExceedsModel;
    try @import("mimo_source.zig").validateExl3Pack(io, a, options.directory, &cfg);
    @import("log.zig").info("[glm-bench] loading native target; loopback diagnostic only\n", .{});
    var weights = try native.loadWeightsBounded(io, a, options.directory, s, false, budget);
    defer weights.deinit();
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    var assistant: ?dflash.DflashModel = if (options.assistant) |path| try adapter.loadAssistantStored(io, a, path, &target) else null;
    defer if (assistant) |*draft_model| draft_model.deinit();
    var tok = try tokenizer.loadTokenizer(io, a, options.directory);
    defer tok.deinit();
    var template = try chat.loadChatConfig(io, a, options.directory);
    defer template.deinit();
    var version = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(version);
    try mlx.check(mlx.mlx_version(&version));
    var resident: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&resident));
    if (resident >= budget) return error.GlmBenchmarkResidentBudgetExceeded;
    var runtime = Runtime{ .options = options, .cfg = &cfg, .target = &target, .assistant = if (assistant) |*draft_model| draft_model else null, .tok = &tok, .chat_config = &template, .name = std.fs.path.basename(options.directory), .version = std.mem.span(mlx.mlx_string_data(version)), .budget = budget, .resident = resident };
    var listener = try net.startListener("127.0.0.1", options.port);
    defer listener.deinit(io);
    shutdown.store(false, .release);
    const sigact = std.posix.Sigaction{ .handler = .{ .handler = signalHandler }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
    std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);
    @import("log.zig").info("[glm-bench] READY http://127.0.0.1:{d} model={s} backend={s} ctx={d} mlx={s}; benchmark-only, tools unsupported\n", .{ listener.socket.address.ip4.port, runtime.name, runtime.backend(), options.context, runtime.version });
    while (!shutdown.load(.acquire)) {
        var fds = [_]std.posix.pollfd{.{ .fd = listener.socket.handle, .events = std.posix.POLL.IN, .revents = 0 }};
        if ((std.posix.poll(&fds, 1000) catch continue) == 0) continue;
        const stream = listener.accept(io) catch {
            if (shutdown.load(.acquire)) break;
            continue;
        };
        var conn: net.Conn = undefined;
        net.Conn.init(&conn, stream, io);
        defer conn.close();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const request_a = arena.allocator();
        const had_error = mlx.errorPending();
        runtime.handle(request_a, &conn) catch |err| {
            mlx.dropLatchedErrorUnless(had_error);
            if (err != error.BenchmarkClientDisconnected) failure(request_a, &conn, @errorName(err), @errorName(err), options.context) catch {};
        };
    }
}

test "GLM benchmark HTTP gated native server" {
    const directory = std.c.getenv("SUSHI_GLM_BENCH_MODEL") orelse return error.SkipZigTest;
    const port = std.c.getenv("SUSHI_GLM_BENCH_PORT") orelse "8094";
    if (std.c.getenv("SUSHI_GLM_BENCH_ASSISTANT")) |assistant| {
        try run(std.testing.allocator, std.testing.io, &.{ std.mem.span(directory), "--assistant", std.mem.span(assistant), "--port", std.mem.span(port), "--benchmark-ignore-eos" });
    } else try run(std.testing.allocator, std.testing.io, &.{ std.mem.span(directory), "--port", std.mem.span(port), "--benchmark-ignore-eos" });
}

test "GLM benchmark HTTP reserve forecast bills every MLA full replacement" {
    const cfg = model.ModelConfig{ .num_hidden_layers = 45, .full_attention_interval = 4, .mla_kv_lora_rank = 512, .indexer_head_dim = 128 };
    try std.testing.expectEqual(@as(usize, 143785984 * 11), try plannedGrowth(&cfg, 131072, 256));
    try std.testing.expectError(error.Overflow, plannedGrowth(&cfg, std.math.maxInt(usize), 1));
}
