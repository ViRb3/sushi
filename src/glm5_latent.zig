//! GLM MLA latent rows: BF16, or kv8 (affine 8-bit, group 64, BF16 scales and biases).
//! Kernels read either form through `SUSHI_LATENT` from `header`; host code through `Latent.dense`.
const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const Arr = mlx.mlx_array;
const nil = Arr{ .ctx = null };

pub const group_size: u32 = 64;
pub const kv8_bits: u8 = 8;

/// Bytes one stored latent row of `width` values occupies at `bits` (0 = BF16).
pub fn rowBytes(width: usize, bits: u8) usize {
    if (bits == 0) return width * 2;
    return width * bits / 8 + width / group_size * 4;
}

/// kv8 scratch over `pending` layers: dense prefill dequantizes at most 2051 history rows, and
/// one chunk's quantizer output lives beside the stored rows.
pub fn kv8ScratchBytes(width: usize, chunk: usize, pending: usize) usize {
    return pending * (2051 * rowBytes(width, 0) + chunk * rowBytes(width, kv8_bits));
}

/// Borrowed storage: BF16 `data [rows, D]`, or kv8 `data [rows, D/4]` u32 codes with
/// `[rows, D/64]` BF16 scales and biases.
pub const Latent = struct {
    data: Arr,
    scales: Arr = nil,
    biases: Arr = nil,

    pub fn quantized(self: Latent) bool {
        return self.scales.ctx != null;
    }
    pub fn rows(self: Latent) c_int {
        return mlx.getShape(self.data)[0];
    }
    pub fn width(self: Latent) c_int {
        const w = mlx.getShape(self.data)[1];
        return if (self.quantized()) w * 4 else w;
    }
    pub fn dtype(self: Latent) mlx.mlx_dtype {
        return mlx.mlx_array_dtype(if (self.quantized()) self.scales else self.data);
    }
    /// Kernels index rows by stride, so every part must be dense row-major.
    pub fn rowMajor(self: Latent) bool {
        if (self.data.ctx == null or mlx.getShape(self.data).len != 2) return false;
        const parts = [_]Arr{ self.data, self.scales, self.biases };
        for (parts[0..if (self.quantized()) 3 else 1]) |part| {
            const shape = mlx.getShape(part);
            const strides = mlx.mlx_array_strides(part);
            if (shape.len != 2 or shape[0] != self.rows() or strides[0] != @as(usize, @intCast(shape[1])) or strides[1] != 1) return false;
        }
        if (!self.quantized()) return true;
        return mlx.mlx_array_dtype(self.data) == .uint32 and mlx.mlx_array_dtype(self.biases) == self.dtype() and
            mlx.getShape(self.scales)[1] * @as(c_int, group_size) == self.width() and
            mlx.getShape(self.biases)[1] == mlx.getShape(self.scales)[1];
    }
    /// Dense rows `[start, end)`; kv8 runs MLX's affine dequantizer, the reference every kernel matches.
    pub fn dense(self: Latent, start: c_int, end: c_int, s: mlx.mlx_stream) !Arr {
        var parts: [3]Arr = .{ nil, nil, nil };
        defer for (&parts) |*p| if (p.ctx != null) {
            _ = mlx.mlx_array_free(p.*);
        };
        const sources = [_]Arr{ self.data, self.scales, self.biases };
        const count: usize = if (self.quantized()) 3 else 1;
        for (sources[0..count], parts[0..count]) |source, *part| {
            const shape = mlx.getShape(source);
            part.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(part, source, &.{ start, 0 }, 2, &.{ end, shape[1] }, 2, &.{ 1, 1 }, 2, s));
        }
        if (!self.quantized()) {
            const out = parts[0];
            parts[0] = nil;
            return out;
        }
        return kv_quant.dequantizeAffine(s, parts[0], parts[1], parts[2], group_size, kv8_bits);
    }
};

/// Quantizes new rows once, at append.
pub fn quantize(rows: Arr, s: mlx.mlx_stream) !kv_quant.QuantizedKV {
    return kv_quant.quantizeAffine(s, rows, group_size, kv8_bits);
}

/// What attention reads for rows not yet stored: their kv8 round trip, or the rows themselves.
pub fn readable(rows: Arr, bits: u8, s: mlx.mlx_stream) !Arr {
    if (bits == 0) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, rows));
        return out;
    }
    var q = try quantize(rows, s);
    defer q.deinit();
    return kv_quant.dequantizeAffine(s, q.q, q.scales, q.biases, group_size, kv8_bits);
}

const bf16_header: [:0]const u8 = "#define SUSHI_LATENT(src,row,d,width) (src[size_t(row)*size_t(width)+size_t(d)])\n";
// Same expression and types as MLX's affine_dequantize, so kernels match `Latent.dense` bit for bit.
const kv8_header: [:0]const u8 =
    \\template <typename T>
    \\inline T sushi_latent8(const device uint32_t* codes, const device T* scales, const device T* biases, size_t row, uint d, uint width) {
    \\  const uint8_t q = reinterpret_cast<const device uint8_t*>(codes)[row*width+d];
    \\  const size_t g = row*(width/64u)+d/64u;
    \\  const T scale = scales[g];
    \\  const T bias = biases[g];
    \\  return scale * q + bias;
    \\}
    \\#define SUSHI_LATENT(src,row,d,width) sushi_latent8(src, src##_scales, src##_biases, size_t(row), uint(d), uint(width))
    \\
;
/// Kernel header defining `SUSHI_LATENT(src, row, d, width)`; a kv8 source `src` also takes
/// inputs `src_scales` and `src_biases`.
pub fn header(quantized: bool) [:0]const u8 {
    return if (quantized) kv8_header else bf16_header;
}

pub fn randomRows(rows: c_int, cols: c_int, seed: u64, s: mlx.mlx_stream) !Arr {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var normal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normal);
    try mlx.check(mlx.mlx_random_normal(&normal, &.{ rows, cols }, 2, .float32, 0, 1, key, s));
    // Row magnitudes from 2^-12 to 2^11 exercise every scale exponent the cache can meet.
    var exponent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exponent);
    try mlx.check(mlx.mlx_arange(&exponent, -12, 12, 24.0 / @as(f64, @floatFromInt(rows)), .float32, s));
    var magnitude = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(magnitude);
    const two = mlx.mlx_array_new_float(2);
    defer _ = mlx.mlx_array_free(two);
    try mlx.check(mlx.mlx_power(&magnitude, two, exponent, s));
    var column = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(column);
    try mlx.check(mlx.mlx_reshape(&column, magnitude, &.{ rows, 1 }, 2, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, normal, column, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, scaled, .bfloat16, s));
    return out;
}

var probe_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
fn probe(latent: Latent, s: mlx.mlx_stream) !Arr {
    const quantized = latent.quantized();
    const slot = &probe_kernels[@intFromBool(quantized)];
    if (slot.* == null) {
        const names: []const [*:0]const u8 = if (quantized) &.{ "cache", "cache_scales", "cache_biases" } else &.{"cache"};
        const ins = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const source: [:0]const u8 =
            \\const uint i=thread_position_in_grid.x;
            \\if(i>=uint(ROWS)*uint(W)) return;
            \\out[i]=SUSHI_LATENT(cache,i/uint(W),i%uint(W),uint(W));
        ;
        const k = mlx.mlx_fast_metal_kernel_new(if (quantized) "sushi_glm_latent8_probe" else "sushi_glm_latent_probe", ins, outs, source, header(quantized), false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ latent.rows(), latent.width() }, 2, latent.dtype()));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, latent.rows() * latent.width(), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", latent.rows()));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "W", latent.width()));
    const arrays = [_]Arr{ latent.data, latent.scales, latent.biases };
    const iv = mlx.mlx_vector_array_new_data(&arrays, if (quantized) 3 else 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, slot.*.?, iv, cfg, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

fn expectBits(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
}

test "GLM kv8 latent kernel helper matches MLX affine dequantization bit for bit" {
    const s = mlx.gpuStream();
    const random = try randomRows(4094, 512, 7, s);
    defer _ = mlx.mlx_array_free(random);
    // An all-zero row and a constant row take the quantizer's degenerate-scale branches.
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &.{ 1, 512 }, 2, .bfloat16, s));
    var constant = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(constant);
    try mlx.check(mlx.mlx_ones(&constant, &.{ 1, 512 }, 2, .bfloat16, s));
    var source = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(source);
    const parts = [_]Arr{ random, zero, constant };
    const vec = mlx.mlx_vector_array_new_data(&parts, 3);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&source, vec, 0, s));
    var q = try quantize(source, s);
    defer q.deinit();
    const latent = Latent{ .data = q.q, .scales = q.scales, .biases = q.biases };
    try std.testing.expect(latent.rowMajor());
    try std.testing.expectEqual(@as(c_int, 512), latent.width());
    const want = try latent.dense(0, 4096, s);
    defer _ = mlx.mlx_array_free(want);
    const got = try probe(latent, s);
    defer _ = mlx.mlx_array_free(got);
    try expectBits(want, got);
    const plain = try probe(.{ .data = want }, s);
    defer _ = mlx.mlx_array_free(plain);
    try expectBits(want, plain);
}

test "GLM latent row bytes bill kv8 at 544 bytes per 512-wide row" {
    try std.testing.expectEqual(@as(usize, 1024), rowBytes(512, 0));
    try std.testing.expectEqual(@as(usize, 544), rowBytes(512, 8));
}
