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
