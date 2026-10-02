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


## Dense/shared BF16 activation

`glm5_activation.apply` combines gate upper-clamping, up-clamping to [-10,10], SiLU and
multiplication into one dispatch. It uses the existing exhaustive BF16 sigmoid table and rounds
both products separately to BF16, matching `DenseMlp` and the source fixtures. It applies only to
matching BF16 arrays with the checkpoint's limit 10; other inputs use the original operations.
The EXL3 expert middle stage has a different arithmetic contract and does not use this helper.

Tests sweep all 65,536 BF16 encodings in both gate and up, crossed with values at and adjacent to
clamp boundaries, signed values and zero. Every non-NaN result bit matches the original operations;
NaN results remain NaN. Broadcast inputs exercise the kernel's contiguous-input preparation.
The diagnostic reports activation dispatches, excluding warmup. This adds a small shared 128KiB
sigmoid table when not already present, rather than changing any stored model weight.


## Prefill KDA output fusion

The prefill reference path uses a separate `glm5_kda_fused.post` kernel for 128-wide BF16
heads. It fuses FP32 mean-square normalization, the stored norm weight and FP32 sigmoid gating,
then stores BF16 output. Other geometry/storage retains the staged implementation. The exact
reduction order and unary modes match the tested one-token body; recurrent state is unchanged.
Tests compare every output bit at 1, 17 and 512 rows, including the production 64-head geometry.
The diagnostic separately counts successful prefill epilogue dispatches, excluding warmup.

A warmed alternating AB/BA component experiment (100 pairs per shape) measured median host
construction plus synchronous evaluation of 673.08 microseconds staged versus203.92 fused
at 512x64x128. This is an epilogue result, not whole-model throughput. The same experiment at
one/17 rows measured277.02/318.44 versus163.23/187.63 microseconds. One-token model decode
already has its own larger fusion; this separate helper is integrated only for multiple rows.


Eligible one-token HC calls can opt into the normalization/mix fusion with `SUSHI_GLM_HC_FUSED=1`. `Hc.collapseReference`
retains the staged path for unsupported geometry and independent tests/benchmarks. The diagnostic
counts fused HC calls separately. It is off by default: the full-model attribution run
measured 26.49 tok/s with HC alone versus 26.54 with both decode candidates disabled, so the
queued component gain did not establish an end-to-end benefit. This fusion leaves Sinkhorn iterations and stream collapse unchanged.

## Batched verification routing

DFlash's short-row FFN path can route up to 16 rows using the same two kernels as serial
routing. The logits grid adds a row dimension; selection/normalization stays independent
and keeps the serial order for each row. Normal model routing remains one-row-only, and
unsupported verification inputs retain the per-row fallback. Tests cover production
288-expert/4096-wide geometry, BF16/FP32 inputs and weights, normalized/unnormalized scores,
stable ties and integrated FFN output parity. The diagnostic reports router_batch_calls.

A warmed 100-pair alternating AB/BA microbenchmark, with one evaluation for the whole routing
set in both arms, measured four rows at 282.98 microseconds serial versus232.75 batched.
This is a routing-component result, not end-to-end speculative speed. The timing-only binary
avoids filling the configuration cache with unrelated correctness-test cases before measuring.


## Stored A6 and A8 trunk grids

Native linear loading and `Ops` affine matmul/dequantization infer 6 or 8 bits
from packed weight and BF16 group128 grids. They validate matrix/grid geometry
against the declared input width. Selected embedding rows and sliced per-head
MLA key/value banks retain their packed storage; MLA reshape uses the stored
packed width instead of assuming four weights per U32. Retained BF16/FP32
tensors keep the dense path. No weights are requantized or expanded at load time.

A6 regressions independently pack nonzero six-bit coefficients, verify dequantized
values, forward/transposed matmul and gathered embedding rows, and reject malformed
grids or unsupported widths. MLA tests cover both absorbed-key and value projection
orientations with nonzero A6 banks. Runtime measurements for the A6 trunk are
separate from historical A8 trunk results.
