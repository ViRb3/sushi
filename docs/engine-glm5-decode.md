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


## Fused one-token KDA body

The mixed checkpoint retains BF16 low-rank gate and beta projections. The native path computes
those projections in their stored format, then adapts oMLX's precomputed-gate KDA body. One Metal
dispatch performs depthwise convolution, BF16 SiLU, query/key L2 normalization, per-channel forget
gates, delta recurrence, and gated output normalization. The output projection remains separate.
Prepared convolution weights and exp(A_log) are reused; recurrent state stays FP32.

This path is restricted to the tested M5-class backend, one BF16 token, head width128, convolution
width4 and lower bound-5. Unsupported shapes retain the reference implementation. Before first use,
small probes select the unary-math variants matching this MLX build; unsupported math does not enable
the fusion. Probe errors do not poison the capability cache. Input shapes/dtypes and launch products
are checked, and immutable launch configurations are cached by head count.

Strict regressions compare output, convolution history and every FP32 recurrent-state bit across
five consecutive decode steps, cold and nonzero initial states, and1/3/64 heads with different head
parameters. The complete GLM-filtered suite also passed with this path enabled, including the draft
adapter's branch-state tests. The diagnostic records successful KDA-body dispatches independently
from QKV dispatches. Full-checkpoint throughput is a separate measurement, not inferred from these
fusion or parity results.


## One-token sigmoid router

Eligible one-token routers use two dispatches: the source-aligned FP32 GEMV with sigmoid/correction,
then stable selection and unbiased score normalization. The correction bias affects selection only.
The existing path remains for other geometry, strides or storage. The fused path accepts stored BF16 or FP32 router matrices and FP32 correction
biases. BF16 weights widen locally during FP32 multiplication; no resident FP32 copy is created. Production288-expert/4096-input regressions compare
selected order and every normalized score bit, including tied scores, normalized/unnormalized modes,
and BF16/FP32 inputs. This optimization does not change prefill routing.

A real-run engagement check found that the checkpoint stores its router matrices as BF16 despite
requesting FP32 router arithmetic. The first FP32-storage-only prototype therefore did not engage.
The corrected storage-aware path passes exact score/order tests for both stored dtypes and both
activation dtypes; configuration compute precision must not be mistaken for checkpoint storage.
