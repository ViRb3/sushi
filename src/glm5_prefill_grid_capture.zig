//! One opt-in actual L20 routed-prefill fixture; ordinary calls do no GPU work.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
threadlocal var captured = false;
pub fn capture(layer: u16, x: Arr, indices: Arr, scores: Arr) !void {
    if (captured or layer != 20) return;
    const path = std.c.getenv("SUSHI_GLM_PREFILL_GRID_CAPTURE") orelse return;
    if (!std.mem.eql(c_int, mlx.getShape(x), &.{ 1, 2048, 4096 })) return;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(indices), &.{ 1, 2048, 8 }) or !std.mem.eql(c_int, mlx.getShape(scores), &.{ 1, 2048, 8 }) or mlx.mlx_array_dtype(indices) != .uint32 or mlx.mlx_array_dtype(scores) != .float32) return error.InvalidGlmGridCapture;
    const values = [_]Arr{ x, indices, scores };
    const ev = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    const arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    const metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    for ([_][*:0]const u8{ "x", "indices", "scores" }, values) |name, value| try mlx.check(mlx.mlx_map_string_to_array_insert(arrays, name, value));
    try mlx.check(mlx.mlx_map_string_to_string_insert(metadata, "provenance", "first actual normal L20 T2048 routed input/IDs/scores; capture forces evaluation; not a throughput run"));
    try mlx.check(mlx.mlx_save_safetensors(path, arrays, metadata));
    captured = true;
}
