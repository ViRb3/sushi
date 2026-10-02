# GLM serial decode projection kernels

`glm5_decode.qkv` combines three independent affine8/group128 projections into one dispatch for a single
BF16 token. It accepts existing packed weight, scale and bias arrays and returns three separate outputs.
The kernel selects each bank by its output-row range; it never concatenates or dequantizes resident
weights into another buffer. This avoids the extra 3.287 GiB that duplicating all loaded GLM KDA QKV
banks would require.

Eligibility requires already-materialized weight grids, a contiguous BF16 input vector, an input width divisible by 256, output widths
divisible by eight, contiguous U32 packed weights, and matching contiguous BF16 scale/bias grids.
Unsupported inputs return no candidate so the caller can use ordinary MLX quantized matmul. Prefill,
dense projections, other rates/groups and strided weight banks use that fallback.

The implementation is a restricted port of oMLX `6745c39c`'s `multi_qmv`. It retains the MLX-derived
qmv-fast lane mapping, eight-value per-lane blocks, four output rows per SIMD group, accumulation order
and BF16 output rounding. Provenance and licenses are recorded in `NOTICE`. Cached dispatch configurations
hold only small metadata; they do not retain or copy model tensors.

The numerical check compares all output bits with three independent calls to the pinned MLX affine
matmul. It uses distinct random packed codes, scales and biases and nonzero BF16 inputs, including unequal small output banks and the
actual 4096-input/8192-output KDA geometry. These comparisons establish the tested kernel's arithmetic
parity. A full-model timing comparison must separately establish whether removing two projection
dispatches per KDA layer improves decode; no throughput gain is implied here.

`dispatchCount()` and `resetDispatchCount()` expose successful graph dispatches for the diagnostic
runner; they are used on the sole MLX inference thread and do not synchronize the GPU.

Larger KDA convolution, normalization, gate and recurrence fusions remain separate work. They have more
rounding boundaries and must preserve the established layer reference before integration. This module
adds no speculative decoding path.
