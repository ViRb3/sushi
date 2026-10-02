# GLM KDA prefill: recurrence schedules and projection dispatch

Research status, 2026-10-03: staged recurrence schedules were rejected. A small raw
threadgroup change remains an unwired experiment. The large affine projections
already qualify for MLX NAX dispatch; missing NAX does not explain their cost.
No runtime defaults were changed by this investigation.

## Exact recurrence experiment

The baseline is `glm5_next.kda` → `transformer.getGdnKernel(true)`. It retains one
FP32 state vector per value row, applies FP32 decay separately for every key
channel, and uses 32 SIMD lanes with four key elements per lane. Both dot products
retain the existing scalar accumulation followed by `simd_sum`.

Two schedule changes were tested independently:

- Raw groups contain 4, 8 or 16 independent value rows. The Metal kernel source
  remains unchanged; only threadgroup height changes.
- Staged groups copy Q/K, FP32 vector decay, V and beta into threadgroup memory in
  blocks of 4, 8 or 16 tokens. The nine row/block combinations retain state
  precision and arithmetic order, including the original BF16 output boundary.

The initial seven schedules passed BF16/FP32 input tests with nonzero FP32 state,
batch 2, three heads, and sequence lengths 3/17/65/128. Production geometry
(batch 1, 64 heads, 128 key/value dimensions) also matched all output and final
state bits for a 512-token sequence and continuation chunks 65/63/127/257.
All twelve timed schedules matched output and state bits at production geometry
for lengths 128 and 512. The five subsequently added staged schedules did not
receive the broader mixed-dtype continuation suite; no adoption depends on them.

Timing used already evaluated BF16 inputs, BF16 beta and nonzero FP32 state;
three warmups per arm, nine alternating forward/reverse arm rounds, and three
evaluations per sample. Numbers include host apply/evaluate/free overhead. The
GPU was exclusive, fans were held at maximum after a ten-second idle, and the
process used interactive QoS. No model was loaded.

| Schedule | 128-token median ms | 512-token median ms | Change at 512 |
|---|---:|---:|---:|
| Raw, 4 rows (baseline) | 0.437597 | 1.303291 | — |
| Raw, 8 rows | 0.435972 | 1.250000 | −4.09% |
| Raw, 16 rows | 0.405625 | 1.262791 | −3.11% |
| Staged, 4 rows / 4 tokens | 0.562611 | 1.947278 | +49.41% |
| Staged, 8 rows / 4 tokens | 0.508000 | 1.606194 | +23.24% |
| Staged, 16 rows / 4 tokens | 0.463486 | 1.552500 | +19.12% |
| Staged, 4 rows / 8 tokens | 0.536889 | 1.723500 | +32.24% |
| Staged, 8 rows / 8 tokens | 0.495708 | 1.503486 | +15.36% |
| Staged, 16 rows / 8 tokens | 0.454625 | 1.431388 | +9.83% |
| Staged, 4 rows / 16 tokens | 0.628833 | 2.156055 | +65.43% |
| Staged, 8 rows / 16 tokens | 0.478208 | 1.404958 | +7.80% |
| Staged, 16 rows / 16 tokens | 0.444528 | 1.414764 | +8.55% |

Raw eight-row groups beat the paired baseline in eight of nine 512-token rounds;
the paired median improvement was 4.78%. The 128-token samples contain large
outliers and need more evidence before interpreting their small differences.
The 512-token saving is only about 0.053 ms per recurrence, or 1.8 ms across 34
layers if it transfers unchanged. This is not a measured full-model saving.
All staged schedules lose and are rejected. The private run artifact
`glm53-kda-prefill-20261003` retains the complete candidate, exact build recipe,
raw samples and provenance; no experiment is imported into the engine.

The artifact records Sushi base `cbf79b2b4bd4957cd74dacc1fc5f5d51dafe0270`, candidate
SHA-256 `95b084f0e337bc983cca0fccba2588d6912f3547ec953bdb3932bcb1eebfa918`,
and executable SHA-256
`e3a7229acd076380d3dacf694c621f19ac225ecdd8c2938e9c12ef90c4830c30`.

## Actual MLX projection dispatch

The staged runtime identifies MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85`,
while the Sushi gitlink pins `d73eb752ef2e6288fd95b032c0bff0a15a4a9e93`.
`mlx/backend/metal/quantized.cpp` and `matmul.cpp` have no differences between
those revisions. The installed metallib contains the required BF16 affine8g128
NAX kernel. `metal::is_nax_available` requires an enabled NAX build, macOS 26.2+
and an eligible GPU generation; the measured M5 Max host meets the OS/hardware
conditions. These are source-derived dispatch conclusions, not an individual
Metal command trace.

Checkpoint headers confirm Q/K/V packed weights `[8192,1024]` U32 and
scales/biases `[8192,32]` BF16. FA/GA weights are BF16 `[128,4096]`, FB/GB are
BF16 `[8192,128]`, and beta is BF16 `[64,4096]`. `KdaLayer.projectQkv` falls back
to three ordinary `Linear.apply` calls at 512 tokens; the fused decode projection
requires one row. The ordinary linear path preserves the BF16 activations and
uses transposed affine8/group128 weights or BF16 dense matmul as applicable.

For batch 1 and 512 tokens, flattening produces the following dispatches:

| Projection | M / N / K | Source-selected path | Tile / grid detail |
|---|---|---|---|
| Q, K, V individually | 512 / 8192 / 4096 | Affine NAX QMM | 64×64×64; 128×8 groups |
| Output | 512 / 4096 / 8192 | Affine NAX QMM | 64×64×64; 64×8 groups |
| FA, GA individually | 512 / 128 / 4096 | BF16 NAX split-K | 64×64×256; 2 partitions, 32 groups |
| Beta | 512 / 64 / 4096 | BF16 NAX split-K | 64×64×256; 2 partitions, 16 groups |
| FB, GB individually | 512 / 8192 / 128 | BF16 regular NAX | 64×128×256; K tail, 512 groups |

For affine Q/K/V, `QuantizedMatmul::eval_gpu` selects matrix mode above its
13-row vector limit. `qmm_splitk` targets approximately 512 threadgroups using
32×32 tiles. The provisional grid has 4096 groups, so split-K is one and calls
`qmm`. BF16 input, transposed weights and K divisible by 64 qualify for `qmm_nax`.
The 128-token Q/K/V case also reaches NAX: its provisional grid has 1024 groups.
An actual split-K greater than one instead selects legacy `qmm_t_splitk`; this
is a real route elsewhere, but not the audited Q/K/V cases.

For dense BF16 projections, `steel_matmul_axpby` enables NAX and selects split-K
when K is sufficiently larger than M/N. `steel_gemm_splitk_axpby_nax` uses
2048-element partitions at K4096, FP32 partial buffers and a separate reduction.
FA/GA therefore launch only 32 groups and beta only 16 on a 40-core GPU. FB/GB
instead select `steel_matmul_regular_axpby_nax`; its large-device tile uses
K256, leaving an unaligned K128 tail.

Primary source symbols at the runtime revision are:

- `mlx/backend/metal/quantized.cpp`: `get_qmv_batch_limit`,
  `QuantizedMatmul::eval_gpu`, `qmm_splitk`, `qmm`, `qmm_nax`.
- `mlx/backend/metal/matmul.cpp`: `Matmul::eval_gpu`, `steel_matmul_axpby`,
  `steel_gemm_splitk_axpby_nax`, `steel_matmul_regular_axpby_nax`.
- `mlx/backend/metal/device.cpp`: `metal::is_nax_available`.
- Sushi `src/glm5_model.zig`: `Linear.apply`, `KdaLayer.projectQkv`,
  `KdaLayer.applyReference`; `src/glm5_decode.zig`: `qkv`.

## Next experiments and limits

The source audit suggests testing the two skinny first-stage projections plus
beta together to increase group count, and evaluating a smaller K tile for the
two shallow second-stage projections. These are hypotheses, not measured wins.
Any joining must preserve each output's BF16 storage boundary; concatenating
unrelated lowrank inputs changes the mathematical operation and is invalid.
More split-K partitions change reduction order and require explicit drift tests.

Arithmetic volume also limits how much these small projections can explain.
At 512 tokens, Q/K/V/output together represent approximately 137.44 GFLOP per
layer, versus 3.49 GFLOP for all five lowrank/beta projections. Component timing
should separate these projections, convolution/normalization/decay preparation,
recurrence and output epilogue before assigning the unmeasured attention cost.
The earlier 203.46 ms whole-KDA attention profile does not provide that split;
subtracting isolated recurrence timings is only a rough prioritization aid.

The oMLX recurrence is another possible experiment, not a drop-in exact schedule.
At oMLX revision `6745c39cb66ba5ec130a761149fc1b8b832fcb07`,
`omlx/patches/glm53_kda_recurrence.py` uses blocked 8-lane dot products with
16 key elements per lane, float4 accumulation and a different reduction tree.
It retains FP32 state and vector gates but changes both K·state and Q·state
rounding. Its default per-core schedule also does not fit 8192 value rows within
128 rows/core on this 40-core host; the blocked fallback is the relevant design.

An isolated port should first consume the current precomputed FP32 decay, then
measure output and final-state maximum error and normalized RMS/Frobenius error
against the current checkpoint's runtime. Include nonzero state, nearly unit
decay, irregular continuation, and actual layer captures. Longer-context logits
and varied-prompt KL against the current runtime are subsequent gates. This is
kernel drift validation, distinct from quantization KLD against a BF16 teacher.
Keep any changed-order candidate opt-in until those gates pass, and ensure
DFlash verification agrees with whichever serial arithmetic is selected.

## Affine NAX tile ordering audit

A subsequent CPU audit examined the actual coordinate mapping for
M512/K4096/N8192. The host `qmm_nax` launch is `(128,8,1)` threadgroups, with
`(32,2,2)` threads per group. `affine_qmm_t_nax` passes its threadgroup ID unchanged
to `qmm_t_nax_tgp_impl`, which computes `y_row=tid.y*64` and `y_col=tid.x*64`.
`CommandEncoder::dispatch_threadgroups` forwards the grid directly to Metal.
There is **no explicit M-tile swizzle on this affine path**.

The pinned and runtime `quantized_nax.h` revisions differ only in a later gather
RHS tail-shape expression; the audited ordinary affine wrapper/body are unchanged.
Relevant primary sources are `qmm_nax` in `quantized.cpp`,
`affine_qmm_t_nax`, `qmm_t_nax_tgp_impl` and `QuantizedBlockLoader` in
`kernels/quantized_nax.h`, and `CommandEncoder::dispatch_threadgroups` in
`device.cpp`, at the runtime revision above.

Each logical tile owns a 64-output-column stripe. Its loader iterates over the
4096 input columns in blocks of 64, dequantizing weights into a private
`Ws[64][72]` BF16 threadgroup buffer (9 KiB). The two M-direction simdgroups reuse
this staged weight tile within the group. Across the eight M tiles, independent
groups load and dequantize the same stripe again. One entire projection bank is
32 MiB packed codes plus 1 MiB BF16 scales/biases. Eight logical bank traversals
do **not** demonstrate eight DRAM reads: the GPU's scheduling/cache behavior is
not established by this source audit.

MLX's dense NAX path does have an explicit locality mapping.
`steel_matmul_regular_axpby_nax` in `matmul.cpp` selects `swizzle_log=2` for
architecture suffixes s/c/d. `steel_gemm_fused_nax` maps physical IDs to
`logical_n=x>>2`, `logical_m=(y<<2)+(x&3)` and guards padded tiles. Four adjacent
physical x IDs therefore refer to different M tiles of one N tile. This is
direct evidence of a grouping mechanism elsewhere in MLX; it is not evidence
that the affine path would benefit from the same mechanism. Metal does not
guarantee an x-major execution order for the original or remapped grid.

A contained experiment can keep the affine body and its arithmetic unchanged,
altering only the ID passed to it:

| M tiles grouped, G | Physical grid | Logical N tile | Logical M tile |
|---:|---|---|---|
| 1, control | 128×8 | x | y |
| 2 | 256×4 | x/2 | 2y + x%2 |
| 4 | 512×2 | x/4 | 4y + x%4 |
| 8 | 1024×1 | x/8 | 8y + x%8 |

All four are bijections of the same 1024 tiles. Swizzling leaves the number of
issued loads and dequantizations unchanged; the possible benefit is fewer cache
misses when neighboring groups read the same weight stripe. It also changes
activation locality: neighboring N tiles previously reused an M tile's inputs.
The 4 MiB activation matrix and 33 MiB bank have different reuse footprints, but
size alone cannot establish the best schedule.

No shared `lib/mlx` binary needs modification. A separate opt-in Sushi custom
Metal kernel can embed the pinned NAX helper/quantized loader source and call the
same `qmm_t_nax_tgp_impl` with remapped IDs. `get_qmm_nax_kernel` documents its
preamble composition: MLX utils, GEMM NAX helpers, quantized utils and quantized
NAX body. The custom-kernel backend already prepends MLX utils; avoid duplicating
those definitions. Vendor only the required MIT-licensed helpers with revision
and attribution, or generate a reproducible standalone header from the pinned
source. Do not rely on filesystem includes into an unrelated checkout or private
JIT C++ symbols; the inspected staged library does not export the four preamble
accessors as public dynamic symbols.

First establish that the unmodified G1 clone matches stock QMM output bits and
latency, since custom source packaging and compiler options can themselves
change code generation. Then validate each remap's tile coverage and raw BF16
outputs at the actual geometry. Keep the first guard narrow: contiguous BF16
input, U32 affine8/group128 transposed weights, BF16 scale/bias, batch one and
fully aligned dimensions. Any broader tail case needs an explicit out-of-range
return before barriers. Measure same-body G1/G2/G4/G8 with identical geometry,
fresh graph inputs, rotated banks and paired order, plus the unchanged library
reference. A schedule-only win would not establish a benefit from larger M
tiles or shared dequantization across threadgroups.

### Scheduling experiment result

`src/glm5_qmm_prefill.zig` now contains the isolated implementation, with the
required pinned MLX helpers in `src/kernels/glm5_qmm_prefill_header.metal` and MIT
attribution in NOTICE. It is not imported into the engine or shared test root.
The primitive declines unmaterialized, strided, wrong-dtype and unsupported-shape
inputs; it does not silently add a contiguous copy. The G1 clone passed exact
native BF16 parity at both real projection geometries before any remap was
tested. G2/G4/G8 then passed the same raw-bit comparisons. Focused guard/parity
tests passed (four tests including the temporary root, one gated timing skip).

The subsequent exclusive timing run checked every arm against native on all
four independent random banks, then measured one-bank and four-bank rotation.
Each arm received twelve warmup evaluations. Eleven rounds alternated forward
and reverse arm order, with eight apply/evaluate/free repetitions per sample.
Inputs were evaluated before timing; this includes host orchestration and wait
overhead and uses fresh output graphs on each repetition. Fans were requested
at maximum, initial temperature was 39.18°C, ten seconds of idle preceded the run,
and the process used interactive QoS. No full model was loaded.

| Projection / bank rotation | Native ms | G1 ms | G2 ms | G4 ms | G8 ms |
|---|---:|---:|---:|---:|---:|
| Q/K/V / one | 0.870046 | 0.871718 | 0.885661 | 0.865364 | 0.874739 |
| Q/K/V / four | 0.882921 | 0.883208 | 0.892979 | 0.882203 | 0.874291 |
| Output / one | 0.897052 | 0.910270 | 0.931505 | 0.914369 | 0.907968 |
| Output / four | 0.957546 | 0.941968 | 0.981072 | 0.952109 | 0.933562 |

There is no compelling adoption result. Relative to the same-body G1 control,
G8 improves median four-bank latency by only 1.01% for Q/K/V and 0.89% for output
(10/11 and 8/11 paired wins). Warm-bank results are nearly flat. G2 is slower
in every case, while G4 changes direction across shapes/bank conditions. The
native-versus-G1 difference also changes sign for the output projection, so it
would be misleading to attribute G8's 2.50% four-bank advantage over native
entirely to scheduling. Keep this research path unintegrated; no full-model gain
or reduced DRAM traffic has been demonstrated.

Private artifact `glm53-qmm-prefill-20261003` retains the focused build commands,
temporary test-root source, raw samples, summary and source hashes. The timing
binary SHA-256 is `fd1a4bbea566f4f6dae3b1d91c4991b5ed4be9509c54333dc853adfde4954f86`;
the base checkout was `0fe8ea710cf92f445d723b227bd5fd5faa4575d6` plus this isolated
module/header. The gated test filter is `GLM QMM prefill isolated scheduling timing`,
enabled by `SUSHI_GLM_QMM_PREFILL_BENCH_OUT` naming its JSON result file. A launcher
path error occurred before the first attempted binary execution; no sample came
from that attempt, and its lock/fan cleanup completed before the recorded run.
