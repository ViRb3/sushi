//! One exact R3 A6 hoist test on a production 8192x4096 q_proj bank.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("glm5_model.zig");
const Ops = model.Ops;
const Arr = mlx.mlx_array;
const original = @import("glm5_dflash_qmm.zig");
const hoisted = @import("glm5_dflash_a6_hoist.zig");
fn load(ops: *Ops, desc: std.json.Value) !Arr {
    const path = desc.object.get("file").?.string;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(32 * 1024 * 1024));
    defer std.testing.allocator.free(data);
    const dims = desc.object.get("shape").?.array.items;
    if (dims.len != 2) return error.BadHoistFixture;
    const shape: [2]c_int = .{ @intCast(dims[0].integer), @intCast(dims[1].integer) };
    const dtype_name = desc.object.get("dtype").?.string;
    const dtype: mlx.mlx_dtype = if (std.mem.eql(u8, dtype_name, "U32")) .uint32 else if (std.mem.eql(u8, dtype_name, "BF16")) .bfloat16 else return error.BadHoistFixture;
    const expected = try std.math.mul(usize, try std.math.mul(usize, @intCast(shape[0]), @intCast(shape[1])), if (dtype == .uint32) 4 else 2);
    if (data.len != expected) return error.BadHoistFixture;
    return ops.own(mlx.mlx_array_new_data(data.ptr, &shape, 2, dtype));
}
fn timed(linear: model.Linear, input: Arr, variant: bool) !u64 {
    const watch = @import("io_util.zig").Stopwatch.init(std.testing.io);
    const y = (if (variant) try hoisted.project(mlx.gpuStream(), input, linear) else try original.project(mlx.gpuStream(), input, linear)) orelse return error.ExpectedHoistProjection;
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    _ = mlx.mlx_array_free(y);
    return watch.read();
}
test "GLM A6 R3 masked coefficient hoist production bits and timing" {
    const output = std.c.getenv("SUSHI_GLM_A6_HOIST_OUT") orelse return error.SkipZigTest;
    const fixture = std.c.getenv("SUSHI_GLM_A6_HOIST_FIXTURE") orelse return error.MissingHoistFixture;
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(fixture), std.testing.allocator, .limited(64 * 1024));
    defer std.testing.allocator.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, data, .{});
    defer parsed.deinit();
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const records = parsed.value.object.get("records").?;
    const linear = model.Linear{ .input = 4096, .output = 8192, .w = try load(&ops, records.object.get("weight").?), .scales = try load(&ops, records.object.get("scales").?), .biases = try load(&ops, records.object.get("biases").?) };
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 5303));
    const input = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(input, &.{ 1, 3, 4096 }, 3, .bfloat16, 0, 1, key.*, ops.s));
    const values = mlx.mlx_vector_array_new_data(&.{ input.*, linear.w, linear.scales, linear.biases }, 4);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    const a = try ops.own((try original.project(ops.s, input.*, linear)) orelse return error.ExpectedOriginalProjection);
    const b = try ops.own((try hoisted.project(ops.s, input.*, linear)) orelse return error.ExpectedHoistProjection);
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..3 * 8192], mlx.mlx_array_data_bfloat16(b).?[0..3 * 8192]);
    var samples: [2][11]u64 = undefined;
    for (0..3) |_| for ([_]bool{ false, true }) |variant| {
        _ = try timed(linear, input.*, variant);
    };
    for (0..11) |round| for (0..2) |position| {
        const arm = if (round % 2 == 0) position else 1 - position;
        samples[arm][round] = try timed(linear, input.*, arm == 1);
    };
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .model = parsed.value.object.get("model").?.string, .layer = 0, .projection = "q_proj", .shape = .{ 3, 4096, 8192 }, .fixture = records, .exact_bf16_values = 3 * 8192, .arms = .{ "current A6 rowtile", "hoisted12maskedfloatcoeffs" }, .nanoseconds = samples, .warmups = 3, .rounds = 11, .timing = "hostapply/eval/free; resident original production U32A6/BF16scale/bias; syntheticBF16RMSinput", .runtime_hook = false }, .{ .whitespace = .indent_2 });
    defer std.testing.allocator.free(json);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = json });
}
