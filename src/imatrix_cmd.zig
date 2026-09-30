const std = @import("std");
const kld = @import("kld.zig");
const mlx = @import("mlx.zig");
const imatrix = @import("imatrix.zig");
const log = @import("log.zig");

pub const MAX_CHUNK: u32 = 512;

pub const Options = struct {
    model_dir: []const u8 = "",
    prompts: []const u8 = "",
    out: []const u8 = "",
    ctx_size: u32 = 0,
    prefill_chunk: u32 = MAX_CHUNK,
    ssd_budget_bytes: u64 = 0,
    expert_cache_bytes: u64 = 0,
    help: bool = false,
};

pub fn parseArgs(args: []const []const u8) !Options {
    var opts = Options{};
    if (args.len == 0) return error.ImatrixBadArguments;
    if (std.mem.eql(u8, args[0], "--help")) return .{ .help = true };
    if (!std.mem.eql(u8, args[0], "capture")) return error.ImatrixBadArguments;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const name = args[i];
        if (std.mem.eql(u8, name, "--help")) return .{ .help = true };
        if (i + 1 == args.len) return error.ImatrixBadArguments;
        const value = args[i + 1];
        if (std.mem.eql(u8, name, "--model")) {
            opts.model_dir = value;
        } else if (std.mem.eql(u8, name, "--prompts")) {
            opts.prompts = value;
        } else if (std.mem.eql(u8, name, "--out")) {
            opts.out = value;
        } else if (std.mem.eql(u8, name, "--ctx-size")) {
            opts.ctx_size = std.fmt.parseInt(u32, value, 10) catch return error.ImatrixBadArguments;
            if (opts.ctx_size == 0) return error.ImatrixBadArguments;
        } else if (std.mem.eql(u8, name, "--prefill-chunk")) {
            opts.prefill_chunk = std.fmt.parseInt(u32, value, 10) catch return error.ImatrixBadArguments;
            if (opts.prefill_chunk == 0 or opts.prefill_chunk > MAX_CHUNK) return error.ImatrixBadArguments;
        } else if (std.mem.eql(u8, name, "--ssd-budget-gb") or std.mem.eql(u8, name, "--expert-cache-gb")) {
            const gb = std.fmt.parseInt(u64, value, 10) catch return error.ImatrixBadArguments;
            const bytes = std.math.mul(u64, gb, 1_000_000_000) catch return error.ImatrixBadArguments;
            if (std.mem.eql(u8, name, "--ssd-budget-gb")) opts.ssd_budget_bytes = bytes else opts.expert_cache_bytes = bytes;
        } else return error.ImatrixBadArguments;
    }
    if (opts.model_dir.len == 0 or opts.prompts.len == 0 or !std.fs.path.isAbsolute(opts.out) or !std.mem.endsWith(u8, opts.out, ".safetensors")) return error.ImatrixBadArguments;
    return opts;
}

pub const Chunks = struct {
    ids: []const u32,
    size: usize,
    offset: usize = 0,

    pub fn next(self: *Chunks) ?[]const u32 {
        if (self.offset == self.ids.len) return null;
        const end = self.offset + @min(self.size, self.ids.len - self.offset);
        const part = self.ids[self.offset..end];
        self.offset = end;
        return part;
    }
};

fn loadPrompts(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !kld.PromptList {
    if (try kld.classifySource(io, path) != .jsonl) return error.ImatrixRequiresTokenJsonl;
    var list = try kld.loadPrompts(allocator, io, path, 0);
    errdefer list.deinit();
    if (list.items.len == 0) return error.ImatrixEmptyPrompts;
    for (list.items) |p| {
        const ids = p.ids orelse return error.ImatrixRequiresTokenJsonl;
        if (ids.len == 0) return error.ImatrixEmptyPrompt;
        for (ids) |id| if (id > std.math.maxInt(i32)) return error.ImatrixBadToken;
    }
    return list;
}

fn progress(io: std.Io, start: std.Io.Timestamp, tokens: u64, prompts: usize) void {
    const seconds = @as(f64, @floatFromInt(@max(start.untilNow(io, .awake).nanoseconds, 1))) / 1e9;
    log.info("[imatrix] {d} prompts, {d} tokens, {d:.1} tokens/s\n", .{ prompts, tokens, @as(f64, @floatFromInt(tokens)) / seconds });
}

pub fn cmdCapture(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    const opts = try parseArgs(args);
    if (opts.help) {
        log.info("sushi imatrix capture --model DIR --prompts JSONL --out /ABS.safetensors [--ctx-size N] [--prefill-chunk 1..512] [--ssd-budget-gb N] [--expert-cache-gb N]\n", .{});
        return;
    }
    const partial = try std.fmt.allocPrint(allocator, "{s}.partial", .{opts.out});
    defer allocator.free(partial);
    const marker = try std.Io.Dir.cwd().createFile(io, partial, .{});
    marker.close(io);
    var prompts = try loadPrompts(allocator, io, opts.prompts);
    defer prompts.deinit();
    if (opts.ctx_size > 0) for (prompts.items) |p| {
        if (p.ids.?.len > opts.ctx_size) return error.ImatrixPromptExceedsContext;
    };
    const loaded = try kld.loadModel(io, allocator, .{
        .model_dir = opts.model_dir,
        .ctx_size = opts.ctx_size,
        .ssd_budget_bytes = opts.ssd_budget_bytes,
        .expert_cache_bytes = opts.expert_cache_bytes,
        .enable_mtp = false,
        .mtp_explicit = true,
        .no_template = true,
    });
    defer loaded.deinit();
    const arch = imatrix.Arch.fromModelType(loaded.config.model_type) orelse return error.ImatrixUnsupportedModel;
    if (loaded.xfm.imatrix) |existing| existing.deinit();
    loaded.xfm.imatrix = null;
    const collector = try imatrix.Collector.init(allocator, loaded.xfm.s, partial, loaded.config.num_hidden_layers, @intCast(loaded.config.num_experts), arch);
    loaded.xfm.imatrix = collector;
    defer {
        loaded.xfm.imatrix = null;
        collector.deinit();
    }
    const started = std.Io.Timestamp.now(io, .awake);
    var tokens: u64 = 0;
    for (prompts.items, 0..) |p, index| {
        const ids = p.ids.?;
        const context = if (opts.ctx_size > 0) opts.ctx_size else loaded.config.contextCap();
        if (context > 0 and ids.len > context) return error.ImatrixPromptExceedsContext;
        for (ids) |id| if (id >= loaded.config.vocab_size) return error.ImatrixBadToken;
        try loaded.xfm.resetCache();
        var ctx = loaded.xfm.defaultCtx();
        ctx.skip_lm_head = true;
        var chunks = Chunks{ .ids = ids, .size = opts.prefill_chunk };
        while (chunks.next()) |chunk| {
            const input = mlx.mlx_array_new_data(chunk.ptr, &[_]c_int{ 1, @intCast(chunk.len) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const hidden = try loaded.xfm.forwardWith(&ctx, input);
            defer _ = mlx.mlx_array_free(hidden);
            try mlx.check(mlx.mlx_array_eval(hidden));
            tokens += chunk.len;
        }
        if ((index + 1) % 10 == 0) progress(io, started, tokens, index + 1);
    }
    _ = try collector.flush();
    const first: usize = if (arch == .mimo_v2) loaded.config.first_k_dense_replace else 0;
    for (collector.layers[first..]) |layer| if (layer.tokens != tokens) return error.ImatrixIncompleteCapture;
    try selfCheck(collector, loaded.config.num_experts_per_tok);
    const from = try allocator.dupeSentinel(u8, partial, 0);
    defer allocator.free(from);
    const to = try allocator.dupeSentinel(u8, opts.out, 0);
    defer allocator.free(to);
    if (std.c.rename(from, to) != 0) return error.ImatrixRenameFailed;
    progress(io, started, tokens, prompts.items.len);
}

fn selfCheck(collector: *imatrix.Collector, topk: u32) !void {
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const path = try collector.allocator.dupeSentinel(u8, collector.path, 0);
    defer collector.allocator.free(path);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &metadata, path, cpu));
    var key: [192]u8 = undefined;
    var layers: usize = 0;
    for (collector.layers, 0..) |layer, i| {
        if (layer.tokens == 0) continue;
        layers += 1;
        for ([_][]const u8{ "gate_up_proj", "down_proj", "gate_up_proj.rows", "gate_mass", "reap" }, 0..) |suffix, kind| {
            if (kind >= 3 and layer.gate_mass.ctx == null) continue;
            const name = try std.fmt.bufPrintSentinel(&key, "{s}{d}.mlp.experts.{s}", .{ collector.arch.layerPrefix(), i, suffix }, 0);
            var array = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(array);
            try mlx.check(mlx.mlx_map_string_to_array_get(&array, arrays, name));
            if (mlx.mlx_array_dtype(array) != .float32 or mlx.getShape(array).len != 1) return error.ImatrixSelfCheckFailed;
            const expected = switch (kind) {
                0 => mlx.mlx_array_size(layer.gu),
                1 => mlx.mlx_array_size(layer.down),
                else => @as(usize, @intCast(collector.experts)),
            };
            if (mlx.mlx_array_size(array) != expected) return error.ImatrixSelfCheckFailed;
            try mlx.check(mlx.mlx_array_eval(array));
            const values = mlx.mlx_array_data_float32(array) orelse return error.ImatrixSelfCheckFailed;
            for (values[0..expected]) |v| if (!std.math.isFinite(v) or v < 0) return error.ImatrixSelfCheckFailed;
            if (kind == 2) try validateRows(values[0..expected], layer.tokens, topk);
        }
    }
    if (layers == 0) return error.ImatrixNothingCaptured;
}

test "imatrix command CPU routed row self-check refuses missing or fractional slots" {
    try validateRows(&.{ 2, 4, 0 }, 3, 2);
    try std.testing.expectError(error.ImatrixSelfCheckFailed, validateRows(&.{ 2, 3, 0 }, 3, 2));
    try std.testing.expectError(error.ImatrixSelfCheckFailed, validateRows(&.{ 2.5, 3.5, 0 }, 3, 2));
    try std.testing.expectError(error.ImatrixSelfCheckFailed, validateRows(&.{ std.math.nan(f32), 6 }, 3, 2));
}

fn validateRows(values: []const f32, tokens: u64, topk: u32) !void {
    var sum: f64 = 0;
    for (values) |v| {
        if (!std.math.isFinite(v) or v < 0 or @floor(v) != v) return error.ImatrixSelfCheckFailed;
        sum += v;
    }
    if (sum != @as(f64, @floatFromInt(tokens)) * @as(f64, @floatFromInt(topk))) return error.ImatrixSelfCheckFailed;
}

test "imatrix command CPU options validate paths and bounded chunks" {
    const t = std.testing;
    const opts = try parseArgs(&.{ "capture", "--model", "pack", "--prompts", "p.jsonl", "--out", "/tmp/a.safetensors", "--prefill-chunk", "32", "--ctx-size", "64", "--ssd-budget-gb", "80", "--expert-cache-gb", "2" });
    try t.expectEqual(@as(u32, 32), opts.prefill_chunk);
    try t.expectEqual(@as(u32, 64), opts.ctx_size);
    try t.expectEqual(@as(u64, 80_000_000_000), opts.ssd_budget_bytes);
    try t.expectEqual(@as(u64, 2_000_000_000), opts.expert_cache_bytes);
    try t.expectError(error.ImatrixBadArguments, parseArgs(&.{"capture"}));
    try t.expectError(error.ImatrixBadArguments, parseArgs(&.{ "capture", "--decode" }));
    try t.expectError(error.ImatrixBadArguments, parseArgs(&.{ "capture", "--model", "p", "--prompts", "p", "--out", "relative.safetensors" }));
    try t.expectError(error.ImatrixBadArguments, parseArgs(&.{ "capture", "--prefill-chunk", "0" }));
    try t.expectError(error.ImatrixBadArguments, parseArgs(&.{ "capture", "--prefill-chunk", "16384" }));
}

test "imatrix command CPU prompts preserve ids and chunk every token" {
    const t = std.testing;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(io, "prompts.jsonl", .{});
    try file.writeStreamingAll(io, "{\"id\":7,\"prompt_ids\":[1,2,3,4,5]}\n");
    file.close(io);
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const path = try std.fs.path.join(t.allocator, &.{ root, "prompts.jsonl" });
    defer t.allocator.free(path);
    var prompts = try loadPrompts(t.allocator, io, path);
    defer prompts.deinit();
    try t.expectEqualStrings("7", prompts.items[0].id);
    try t.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5 }, prompts.items[0].ids.?);
    var chunks = Chunks{ .ids = prompts.items[0].ids.?, .size = 2 };
    try t.expectEqualSlices(u32, &.{ 1, 2 }, chunks.next().?);
    try t.expectEqualSlices(u32, &.{ 3, 4 }, chunks.next().?);
    try t.expectEqualSlices(u32, &.{5}, chunks.next().?);
    try t.expect(chunks.next() == null);
    const empty = try tmp.dir.createFile(io, "prompts.jsonl", .{});
    empty.close(io);
    try t.expectError(error.ImatrixEmptyPrompts, loadPrompts(t.allocator, io, path));
    const out = try std.fs.path.join(t.allocator, &.{ root, "failed.safetensors" });
    defer t.allocator.free(out);
    try t.expectError(error.ImatrixEmptyPrompts, cmdCapture(t.allocator, io, &.{ "capture", "--model", "missing", "--prompts", path, "--out", out }));
    const partial = try std.fmt.allocPrint(t.allocator, "{s}.partial", .{out});
    defer t.allocator.free(partial);
    _ = try std.Io.Dir.cwd().statFile(io, partial, .{});
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, out, .{}));
    try tmp.dir.writeFile(io, .{ .sub_path = "failed.safetensors", .data = "existing artifact" });
    try t.expectError(error.ImatrixEmptyPrompts, cmdCapture(t.allocator, io, &.{ "capture", "--model", "missing", "--prompts", path, "--out", out }));
    const previous = try tmp.dir.readFileAlloc(io, "failed.safetensors", t.allocator, .limited(64));
    defer t.allocator.free(previous);
    try t.expectEqualStrings("existing artifact", previous);
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data = "{\"prompt\":\"text is not token IDs\"}\n" });
    try t.expectError(error.ImatrixRequiresTokenJsonl, loadPrompts(t.allocator, io, path));
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data = "{\"prompt_ids\":[-1]}\n" });
    try t.expectError(error.BadPromptJsonl, loadPrompts(t.allocator, io, path));
}
