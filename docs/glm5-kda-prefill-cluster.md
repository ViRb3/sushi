# Retained BF16 KDA prefill projection clustering

The opt-in `SUSHI_GLM_KDA_PREFILL_CLUSTER=1` path combines the retained FA128, GA128
and beta64 weight banks into one BF16 `[320,4096]` bank. It does not change the
original stored tensors. The input is `[1,2048,4096]`; outputs retain
their existing BF16 boundaries and compact `[1,2048,128/128/64]` layout.

[`KdaLayer.applyReference`](../src/glm5_model.zig) applies all three projections
to the same normalized input, with no projection bias or intervening activation.
FA then feeds FB and GA feeds GB. Concatenating the first-stage output channels
therefore preserves their independent linear products. The probe uses actual
layer-zero retained BF16 weights and a materialized synthetic BF16 input, compares
all first-stage and downstream products, and times both scopes.

## Native dispatch and cost

MLX 0.32.3 `64ea011cb` regular native NAX GEMM uses BF16 operands with its default
FP32 accumulator ([GEMM loop](../lib/mlx-src/mlx/backend/metal/kernels/steel/gemm/gemm_nax.h)).
At M2048/K4096, the native split-K predicate fails: K is less than three times
max(M,N), and max(M,N) exceeds 1024. On the `d` architecture branch, regular NAX
chooses BM64/BN128/BK512
([dispatch](../lib/mlx-src/mlx/backend/metal/matmul.cpp)). Three 32-group commands
become one 96-group command. Total padded output width is still 384; this is a
launch/concurrency experiment, not a reduction in padded multiply work. Native
NAX rounding is permitted; no operand cast or precision restoration is added.

The prepared copy costs 2,621,440 bytes per layer, or 85 MiB across 34 KDA layers if
kept for every layer. At 2048 rows the joined output costs 1,310,720 bytes, and the
three compact outputs together cost another 1,310,720 bytes. Compacting is necessary
for the existing fused prework's direct beta indexing. Native matmul itself can
read a last-dimension slice with row stride 320, but that does not establish the
layout contract of all downstream kernels.

The probe includes fresh graph creation, three slice copies, evaluation and graph
teardown in each candidate sample. Prepared weight concatenation is a one-time
cost, measured separately. No full-model throughput claim follows from this
component experiment.

Normal model loading prepares eligible banks only when the flag is enabled, the
stream is GPU, and native NAX hardware support is available. The normal KDA layer
owns the prepared copy and frees it with the model. `KdaLayer.applyReference` uses
the copy only at exactly 2048 BF16 rows; other shapes and disabled calls use the
existing projections. The tree verifier and recurrence implementation are unchanged.

`Model.kdaPrefillClusterBytes()` reports actual retained storage, including a bank
whose flag was subsequently disabled. `transientBudget(chunk,pending_layers)` bills
the additional joined output (1,310,720 bytes per pending layer) for enabled 2048-row
chunks. Existing compact output banks replace the baseline banks; their space is
already required by the original path. Evaluated banks are already included in
actual resident memory; admission must add the transient premium without billing
the persistent bank again. The getter exposes the resident premium for metadata.
The module exports a dispatch count for qualification.

## Component qualification

One exclusive, foreground GPU job used maximum fans and a 10-second idle before
running the probe. The focused wrapper and production qualification passed 2 tests.
All 655,360 first-stage BF16 values and all 33,685,504 downstream FA→FB/GA→GB/beta
values matched the three-call baseline bit for bit; every output was finite.

| scope | original median | cluster median | change | paired wins |
|---|---:|---:|---:|---:|
| FA/GA/beta projections | 883.041 µs | 822.333 µs | −6.87% | 17/21 |
| FA/GA/beta plus FB/GB | 953.583 µs | 925.333 µs | −2.96% | 18/21 |

Each scope used 5 warmup pairs and 21 alternating AB/BA sample pairs, in one process.
The one-time prepared bank took 8.151 ms including cold setup; that cost is excluded
from the per-chunk table. Input and source weights were already materialized. The
component savings do not establish a full-model gain, and source memory increases
by 85 MiB if all 34 KDA layers retain a bank. The narrow experiment remains separate
from the three-row decode/verification projection experiment.

The subsequent focused caller-only job passed 2 tests without repeating timings.
It exercised owned `KdaLayer.preparePrefillCluster`, the exact first-stage helper
used by `applyReference`, and `deinit` against the same production banks. Outputs
matched raw BF16 bits. Repeated preparation retained the same bank; freeing it
cleared the handle. The model getter reported 2,621,440 bytes before free and zero
afterwards. Enabled two-pending-layer transient admission reported 2,621,440 bytes;
disabled calls and T3 inputs fell back, without incrementing the dispatch count.
This check covers the helper and ownership seam; it is not a full-model or full
recurrence qualification.
