//! Opt-in, synchronization-perturbed verifier timings. Disabled markers do not evaluate arrays.
const std = @import("std");
const mlx = @import("mlx.zig");
const Clock = @import("io_util.zig").Stopwatch;
const Arr = mlx.mlx_array;
pub const Sample = struct { ns: u64 = 0, calls: u64 = 0, rows: u64 = 0 };
pub const Totals = struct {
    embedding: Sample = .{},
    hc_attn: Sample = .{},
    attn_norm: Sample = .{},
    kda_qkv: Sample = .{},
    kda_lowrank_beta: Sample = .{},
    kda_prework: Sample = .{},
    kda_recurrence: Sample = .{},
    kda_gate_post: Sample = .{},
    kda_out: Sample = .{},
    mla_common: Sample = .{},
    mla_branches: Sample = .{},
    mla_out: Sample = .{},
    expand_attn: Sample = .{},
    hc_ffn: Sample = .{},
    ffn_norm: Sample = .{},
    ffn_overall: Sample = .{},
    ffn_router: Sample = .{},
    ffn_routed: Sample = .{},
    ffn_shared: Sample = .{},
    ffn_combine: Sample = .{},
    expand_ffn: Sample = .{},
    layer_settle: Sample = .{},
    head: Sample = .{},
};
pub const Profile = struct {
    layers: [128]Totals = @splat(.{}),
    global: Totals = .{},
};
pub const max_route_records = 8192;
pub const max_route_assignments = 128;
pub const max_route_experts = 512;
pub const RouteStats = struct {
    counts: [max_route_experts]u16 = @splat(0),
    unique: usize = 0,
    groups2: usize = 0,
};
pub const RouteRecord = struct {
    round_index: u32,
    layer_index: u16,
    rows: u8,
    top_k: u8,
    experts: u16,
    ordered_ids: [max_route_assignments]u32,
    pub fn stats(self: *const RouteRecord) RouteStats {
        var result: RouteStats = .{};
        for (self.ordered_ids[0 .. @as(usize, self.rows) * self.top_k]) |id| result.counts[id] += 1;
        for (result.counts[0..self.experts]) |count| {
            result.unique += @intFromBool(count != 0);
            result.groups2 += (count + 1) / 2;
        }
        return result;
    }
    pub fn jsonStringify(self: RouteRecord, writer: anytype) !void {
        const n = @as(usize, self.rows) * self.top_k;
        const summary = self.stats();
        try writer.write(.{
            .round_index = self.round_index,
            .layer_index = self.layer_index,
            .rows = self.rows,
            .top_k = self.top_k,
            .experts = self.experts,
            .ordered_ids = self.ordered_ids[0..n],
            .multiplicities = summary.counts[0..self.experts],
            .unique = summary.unique,
            .repeated_slots = n - summary.unique,
            .group2_groups = summary.groups2,
            .group2_saved_slots = n - summary.groups2,
        });
    }
};
pub const RouteCapture = struct {
    allocator: std.mem.Allocator,
    records: []RouteRecord,
    count: usize = 0,
    round_index: ?u32 = null,
    skipped_single_calls: usize = 0,
    pub fn init(allocator: std.mem.Allocator, limit: usize) !RouteCapture {
        if (limit == 0 or limit > max_route_records) return error.InvalidGlmRouteCapacity;
        return .{ .allocator = allocator, .records = try allocator.alloc(RouteRecord, limit) };
    }
    pub fn deinit(self: *RouteCapture) void {
        self.allocator.free(self.records);
        self.* = undefined;
    }
};
threadlocal var active_routes: ?*RouteCapture = null;
pub const RouteBinding = struct {
    previous: ?*RouteCapture,
    pub fn restore(self: RouteBinding) void {
        active_routes = self.previous;
    }
};
pub fn bindRoutes(capture: ?*RouteCapture) RouteBinding {
    const previous = active_routes;
    active_routes = capture;
    return .{ .previous = previous };
}
pub fn beginRouteRound(index: usize) !void {
    const capture = active_routes orelse return;
    if (index >= 4096) return error.InvalidGlmRouteRound;
    capture.round_index = @intCast(index);
}
pub fn skipSingleRoute() void {
    if (active_routes) |capture| capture.skipped_single_calls += 1;
}
pub fn captureRoutes(s: mlx.mlx_stream, layer: usize, rows: usize, top_k: usize, experts: usize, ids: Arr) !void {
    const capture = active_routes orelse return;
    const round = capture.round_index orelse return error.GlmRouteRoundMissing;
    if (layer >= 128 or rows < 2 or rows > 16 or top_k == 0 or top_k > 8 or experts < top_k or experts > max_route_experts or ids.ctx == null) return error.InvalidGlmRouteCapture;
    if (capture.count >= capture.records.len) return error.GlmRouteCaptureFull;
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(top_k) };
    const dtype = mlx.mlx_array_dtype(ids);
    if (!std.mem.eql(c_int, &shape, mlx.getShape(ids)) or (dtype != .uint32 and dtype != .int32)) return error.InvalidGlmRouteCapture;
    var compact = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(compact);
    try mlx.check(mlx.mlx_contiguous(&compact, ids, false, s));
    try mlx.check(mlx.mlx_array_eval(compact));
    var record = RouteRecord{ .round_index = round, .layer_index = @intCast(layer), .rows = @intCast(rows), .top_k = @intCast(top_k), .experts = @intCast(experts), .ordered_ids = undefined };
    const raw: [*]const u32 = if (dtype == .uint32) mlx.mlx_array_data_uint32(compact) orelse return error.MlxArrayDataNull else @ptrCast(mlx.mlx_array_data_int32(compact) orelse return error.MlxArrayDataNull);
    for (raw[0 .. rows * top_k], 0..) |id, i| {
        if (id >= experts) return error.InvalidGlmRouteId;
        // A top-k row must not list one expert twice. Repetition across rows is expected.
        for (record.ordered_ids[i / top_k * top_k .. i]) |previous| if (previous == id) return error.DuplicateGlmRouteId;
        record.ordered_ids[i] = id;
    }
    capture.records[capture.count] = record;
    capture.count += 1;
}
threadlocal var active: ?*Profile = null;
threadlocal var current_layer: ?usize = null;

pub const Binding = struct {
    previous: ?*Profile,
    previous_layer: ?usize,
    pub fn restore(self: Binding) void {
        active = self.previous;
        current_layer = self.previous_layer;
    }
};
/// The caller owns Profile for this host-thread scope. No global process enable flag.
pub fn bind(profile: ?*Profile) Binding {
    const old = Binding{ .previous = active, .previous_layer = current_layer };
    active = profile;
    current_layer = null;
    return old;
}
pub const LayerScope = struct {
    previous: ?usize,
    changed: bool,
    pub fn restore(self: LayerScope) void {
        if (self.changed) current_layer = self.previous;
    }
};
pub fn enterLayer(index: usize) !LayerScope {
    const old = LayerScope{ .previous = current_layer, .changed = active != null };
    if (active == null) return old;
    if (index >= 128) return error.InvalidGlmProfileLayer;
    current_layer = index;
    return old;
}
pub const Timer = struct {
    profile: ?*Profile,
    layer: ?usize,
    rows: usize,
    clock: Clock,
    pub fn start(rows: usize) Timer {
        const p = active orelse return .{ .profile = null, .layer = null, .rows = 0, .clock = undefined };
        return .{ .profile = p, .layer = current_layer, .rows = rows, .clock = Clock.init(std.Io.Threaded.global_single_threaded.io()) };
    }
    /// Settle needed outputs, accumulate this interval, then reset the clock.
    /// Nested overall timers are inclusive; do not add their child fields again.
    pub fn finish(self: *Timer, comptime field: []const u8, outputs: []const Arr) !void {
        if (self.profile == null) return;
        const values = mlx.mlx_vector_array_new_data(outputs.ptr, outputs.len);
        defer _ = mlx.mlx_vector_array_free(values);
        try mlx.check(mlx.mlx_eval(values));
        self.record(field);
    }
    /// Use after an existing evaluation boundary; does not introduce another eval.
    pub fn record(self: *Timer, comptime field: []const u8) void {
        const p = self.profile orelse return;
        const totals = if (self.layer) |index| &p.layers[index] else &p.global;
        const sample = &@field(totals, field);
        sample.ns +|= self.clock.read();
        sample.calls +|= 1;
        sample.rows +|= self.rows;
        self.clock.reset();
    }
};

test "GLM DFlash disabled profiler leaves lazy output unevaluated" {
    const s = mlx.gpuStream();
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const input = try ops.ones(&.{ 64, 128 }, .float32);
    const result = try ops.binary(.mul, input, try ops.scalar(3, .float32));
    var available = true;
    try mlx.check(mlx._mlx_array_is_available(&available, result));
    try std.testing.expect(!available);
    const disabled = bind(null);
    defer disabled.restore();
    const layer = try enterLayer(std.math.maxInt(usize));
    defer layer.restore();
    var timer = Timer.start(64);
    try timer.finish("head", &.{result});
    try mlx.check(mlx._mlx_array_is_available(&available, result));
    try std.testing.expect(!available);
    var measured: Profile = .{};
    const enabled = bind(&measured);
    defer enabled.restore();
    timer = Timer.start(64);
    try timer.finish("head", &.{result});
    try std.testing.expectEqual(@as(u64, 1), measured.global.head.calls);
    try std.testing.expectEqual(@as(u64, 64), measured.global.head.rows);
    try std.testing.expect(measured.global.head.ns > 0);
    try std.testing.expectEqual(@as(f32, 3), mlx.mlx_array_data_float32(result).?[0]);
    try std.testing.expectError(error.InvalidGlmProfileLayer, enterLayer(128));
    const local = try enterLayer(3);
    timer = Timer.start(2);
    try timer.finish("ffn_router", &.{result});
    local.restore();
    try std.testing.expectEqual(@as(u64, 1), measured.layers[3].ffn_router.calls);
    try std.testing.expectEqual(@as(u64, 0), measured.global.ffn_router.calls);
}

test "GLM DFlash route capture is bounded ordered and disabled without evaluation" {
    const s = mlx.gpuStream();
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const data = [_]u32{ 0, 287, 0, 1, 0, 287 };
    const ids = try ops.own(mlx.mlx_array_new_data(&data, &.{ 1, 3, 2 }, 3, .uint32));
    const lazy = try ops.binary(.add, ids, try ops.scalar(1, .uint32));
    var available = true;
    try mlx.check(mlx._mlx_array_is_available(&available, lazy));
    try std.testing.expect(!available);
    const disabled = bindRoutes(null);
    defer disabled.restore();
    try captureRoutes(s, std.math.maxInt(usize), 0, 0, 0, lazy);
    try mlx.check(mlx._mlx_array_is_available(&available, lazy));
    try std.testing.expect(!available);
    try std.testing.expectError(error.InvalidGlmRouteCapacity, RouteCapture.init(std.testing.allocator, 0));
    try std.testing.expectError(error.InvalidGlmRouteCapacity, RouteCapture.init(std.testing.allocator, max_route_records + 1));
    var capture = try RouteCapture.init(std.testing.allocator, 2);
    defer capture.deinit();
    const enabled = bindRoutes(&capture);
    defer enabled.restore();
    try std.testing.expectError(error.GlmRouteRoundMissing, captureRoutes(s, 3, 3, 2, 288, ids));
    try beginRouteRound(7);
    try std.testing.expectError(error.InvalidGlmRouteRound, beginRouteRound(4096));
    try captureRoutes(s, 3, 3, 2, 288, ids);
    try std.testing.expectEqual(@as(usize, 1), capture.count);
    try std.testing.expectEqual(@as(u32, 7), capture.records[0].round_index);
    try std.testing.expectEqual(@as(u16, 3), capture.records[0].layer_index);
    try std.testing.expectEqualSlices(u32, &data, capture.records[0].ordered_ids[0..6]);
    const stats = capture.records[0].stats();
    try std.testing.expectEqual(@as(usize, 3), stats.unique);
    try std.testing.expectEqual(@as(usize, 4), stats.groups2);
    try std.testing.expectEqual(@as(u16, 3), stats.counts[0]);
    try std.testing.expectEqual(@as(u16, 2), stats.counts[287]);
    try std.testing.expectError(error.InvalidGlmRouteId, captureRoutes(s, 3, 3, 2, 288, lazy));
    const duplicate = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 0, 0, 1, 2, 3, 4 }, &.{ 1, 3, 2 }, 3, .uint32));
    try std.testing.expectError(error.DuplicateGlmRouteId, captureRoutes(s, 3, 3, 2, 288, duplicate));
    const negative = try ops.own(mlx.mlx_array_new_data(&[_]i32{ -1, 0, 1, 2, 3, 4 }, &.{ 1, 3, 2 }, 3, .int32));
    try std.testing.expectError(error.InvalidGlmRouteId, captureRoutes(s, 3, 3, 2, 288, negative));
    try std.testing.expectError(error.InvalidGlmRouteCapture, captureRoutes(s, 3, 16, 9, 288, ids));
    try std.testing.expectEqual(@as(usize, 1), capture.count);
    try beginRouteRound(8);
    try captureRoutes(s, 4, 3, 2, 288, ids);
    try std.testing.expectError(error.GlmRouteCaptureFull, captureRoutes(s, 5, 3, 2, 288, ids));
    try std.testing.expectEqual(@as(usize, 2), capture.count);
    skipSingleRoute();
    try std.testing.expectEqual(@as(usize, 1), capture.skipped_single_calls);
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, capture.records[0], .{});
    defer std.testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("group2_saved_slots").?.integer);
    try std.testing.expectEqual(@as(usize, 6), parsed.value.object.get("ordered_ids").?.array.items.len);
    try std.testing.expectEqual(@as(usize, 288), parsed.value.object.get("multiplicities").?.array.items.len);
}
