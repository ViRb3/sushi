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

### Tile aspect experiment

The same isolated body was subsequently parameterized for BM/BN/WM/WN while
retaining BK64 and G1 scheduling. The explicit staging allocation is
`BN*(64+8)*sizeof(BF16)`. Fragment counts below describe live logical tensors,
not compiler register allocation or measured occupancy.

| BM / BN; WM / WN | Shared weights | Per-SIMD M / N | FP32 D elements/lane | BF16 A / B elements/lane |
|---|---:|---|---:|---|
| 64 / 64; 2 / 2 (native) | 9 KiB | 32 / 32 | 32 | 32 / 32 |
| 128 / 32; 2 / 2 | 4.5 KiB | 64 / 16 | 32 | 64 / 16 |
| 128 / 32; 4 / 1 | 4.5 KiB | 32 / 32 | 32 | 32 / 32 |
| 64 / 128; 2 / 2 | 18 KiB | 32 / 64 | 64 | 32 / 64 |

All use 128 threads. Shared-memory fit does not establish residency: cooperative
tensor temporaries, compiler register allocation and spills can dominate it.
The WM4/WN1 tall tile retains the native per-SIMD fragment geometry and shares
each weight tile across four distinct M simdgroups. It halves the number of
M-tile bank/dequantization traversals. It doubles the number of distinct N tile
groups, but removes the native WN2 duplication of activation loads between
simdgroups; source-level issued activation element counts are therefore not
simply doubled. Cache reuse and residency still change.

The WM2/WN2 tall candidate failed its first native-parity gate, with zero values
starting at output row 16. It selects `tile_matmad_nax`'s TN1 paired-M overload;
the pinned overload uses a 16×32×16 MPP descriptor while loading two M fragments.
That branch differs from the native TN2 paired-N operation. It was rejected and
removed from the callable aspect enum without changing upstream arithmetic.
The failure log and candidate snapshot are preserved privately.

Native64/64, tall128/32 WM4/WN1 and wide64/128 WM2/WN2 all passed every native BF16
output bit at both real shapes. The timing run rechecked all supported arms on
four independent banks. It used the same exclusive, interleaved protocol as the
scheduling experiment: twelve warmups, eleven alternating rounds, eight
apply/evaluate/free evaluations per sample, one-bank and four-bank rotation.

| Projection / bank rotation | Native library ms | Same-body 64/64 ms | Tall 128/32 ms | Wide 64/128 ms |
|---|---:|---:|---:|---:|
| Q/K/V / one | 0.871395 | 0.875073 | 0.874625 | 1.040260 |
| Q/K/V / four | 0.882625 | 0.874083 | 0.878192 | 1.054390 |
| Output / one | 0.934260 | 0.939963 | 0.899296 | 1.121791 |
| Output / four | 0.962312 | 0.951948 | 0.958046 | 1.107265 |

The wide tile loses 16–21% against the same-body control and is not an adoption
candidate. The tall tile saves 4.33% on the warm output projection (11/11 paired
wins), but loses 0.64% with four-bank rotation (3/11 wins); Q/K/V is effectively
flat. No robust improvement justifies engine integration. The isolated aspect
options remain available for reproducing the result, while normal execution
continues through MLX.

Private artifact `glm53-qmm-prefill-aspect-20261003` contains the rejected-case
log, supported parity log, raw samples, summaries and exact provenance. Its gated
filter is `GLM QMM prefill isolated aspect timing`, enabled by
`SUSHI_GLM_QMM_ASPECT_BENCH_OUT`. This experiment demonstrates why fewer logical
weight traversals or smaller explicit shared-memory storage alone do not predict
an end-to-end win. No full-model timing was performed for either tile candidate.

## A6/group128 audit of the newer trunk

CPU-only follow-up, 2026-10-03. The runtime remains MLX
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`. New checkpoint headers independently
confirm KDA Q/K/V `[8192,768]` U32 and output `[4096,1536]` U32, with BF16 scales
and biases at group128: these are six-bit weights. `storedAffineBits` infers six
from the packed/scaling dimensions and `Ops.qmm` passes it to MLX. The installed
metallib contains the BF16/group128/bits6 NAX kernel for both batch0 and batch1;
this conclusion does not assume that A8 support implies A6 support.

The table reports source-selected dispatch at cold 512- and 2048-row prefill.
The recorded newer-target native benchmarks actually use chunks of 512 and 2048
respectively, with dense MLA prefill enabled. M below is the actual matrix row
count, not merely the number of tokens in a higher-rank tensor.

| Projection | Logical K → N | 512 rows | 2048 rows |
|---|---|---|---|
| KDA Q/K/V individually | 4096 → 8192 | A6 NAX | A6 NAX |
| KDA output | 8192 → 4096 | A6 NAX | A6 NAX |
| MLA query A | 4096 → 1536 | A6 NAX | A6 NAX |
| MLA query B | 1536 → 16384 | A6 NAX | A6 NAX |
| MLA latent A | 4096 → 512 | Legacy A6 split-K, two partitions | A6 NAX |
| MLA output | 16384 → 4096 | A6 NAX | A6 NAX |
| Dense MLA K/V expansion, each | 512 → 256, 64 heads | Batched A6 NAX | Batched A6 NAX |
| Index query, if consumed | 1536 → 4096 | A6 NAX | A6 NAX |

All listed NAX calls use BM64/BN64/BK64, four simdgroups and 9 KiB of explicit
BF16 weight staging. For MLA latent A at M512, `qmm_splitk` computes
`ceil(512/32)*ceil(512/32)=256` provisional groups, so it selects two K partitions.
That branch uses legacy `qmm_t_splitk`, a 1 MiB BF16 partial plane and a separate
sum. At M2048 it computes 1024 groups and falls through to NAX. This is a newly
identified shape-specific edge, **not an A6-only regression**. Forcing NAX there
would change the partial rounding/reduction order, and cannot be presented as
an exact schedule change. It does not explain the remaining 2K prefill gap,
because the 2K call already takes NAX.

Dense MLA expansion broadcasts the latent cache over 64 head-batched K/V banks;
its effective batch count is 64 and M is the cached-token count. It bypasses the
single-batch split-K heuristic and reaches the b6 NAX path directly. Cached
chunks use the accumulated cache length for this expansion, rather than the
current chunk length.

Absorbed prefill is different: queries/values are shaped `[T,64,1,D]`, so M=1 at
both token counts. Query absorption uses non-transposed `qvm` (K256,N512), and
value unembedding uses `qmv` (K512,N256). Neither becomes a large NAX matmul merely
because T is 2048. Moreover the ordinary `qmm` host path only selects NAX for
transposed weights, despite a non-transposed NAX body existing in the library.
Rebatching absorbed attention is an independent arithmetic/layout change, not a
missing A6 specialization. Native dense prefill already avoids this path here.
Index queries and index weights remain lazy and unconsumed for these cold
prefixes; forcing them during profiling would overstate the normal work.

Small retained tensors remain BF16: KDA FA/GA `[128,4096]`, FB/GB `[8192,128]`,
beta `[64,4096]`; MLA index keys/compression `[128,4096]` and index weights
`[32,4096]`. At M2048 the skinny K4096 projections no longer satisfy dense MLX's
split-K heuristic, so regular NAX uses only 32 groups (and a partial N tile for
N32/64). This group count alone does not establish idle hardware: independent
commands and backend scheduling must also be considered.

A6 does have a distinct unpack cost. `QuantizedBlockLoader` gives each thread
32 weights, read as eight 3-byte packs through `dequantize<...,4,6>`; A8 uses 32
single-byte packs. A6 saves packed bytes but introduces cross-byte masks/shifts.
A source-grounded secondary experiment would compare the unchanged A6 NAX body
with word-wise unpacking of those same 24 bytes (six aligned U32 loads), preserving
identical integer codes and the exact scale/bias/BF16 boundary. First inspect
compiler output: the compiler may already combine byte reads, so no performance
claim follows from the C++ spelling. Avoid padded uint3/ulong3 loads that overread
or assume stronger alignment. No unpack kernel was written in this audit, and
rejected A8 swizzle/aspect experiments are not being repeated.

Primary symbols are `Linear.apply`/`storedAffineBits` and `Mla.applyMode`/
`densePrefill` in Sushi; `QuantizedMatmul::eval_gpu`, `qmm`, `qmm_splitk`,
`qmm_nax` in MLX `quantized.cpp`; and `QuantizedBlockLoader`, `dequantize` plus
`qmm_t_nax_tgp_impl` in `quantized_nax.h`, at the runtime revision above.

## Register-resident reuse across KDA value rows

This exact recurrence experiment is different from both rejected staging
and the small raw threadgroup-height change. One 32-lane SIMD group can carry
R=2 or 4 independent value rows, retaining each lane's four **contiguous** keys
`4*lane+i`. It loads a common K/decay value once, applies it to each member's
state and dot accumulator in i=0..3 order, performs a separate unchanged
`simd_sum` for each member, then similarly reuses K/Q during the state/output
update. Beta is shared; V remains member-specific. State stays FP32 throughout.

Persistent state per lane rises from four FP32 values to eight or sixteen.
Common K and per-member accumulators/deltas add further registers; compiler
spills and occupancy are unknown. No threadgroup storage or barriers are needed.
Keeping four SIMD groups per threadgroup gives 2048/1024/512 groups for R1/R2/R4
at B1,H64,Dv128. This removes repeated source loads/instructions and may improve
independent instruction scheduling; it does not reduce the mathematical FMAs or
prove proportional DRAM savings.

`src/glm5_kda_value_rows.zig` contains isolated R1/R2/R4 candidates. Explicitly
unrolled member/key loops preserve independent accumulations and avoid dynamic
private-array indexing. The reference is `getGdnKernel(true)` through
`glm5_next.kda`; the blocked GDN kernel's eight-lane/sixteen-key contraction and
scalar gate are not substitutes. All candidates passed exact raw output and
FP32 final-state comparisons with nonzero state: BF16/FP32 inputs, BF16/FP32
beta, varying channel decay/beta, short B2/H3 geometries and B1/H64 at 512 and
2048 tokens. Both production lengths/dtypes also passed 17/63/rest continuation,
including concatenated output and final state. Four focused tests passed; the
gated timing test was skipped in the initial parity run and passed when enabled.

The exclusive microbenchmark used interactive QoS (`taskpolicy -a`), lock owner
`glm53-kda-value-rows-v61`, maximum fans requested, initial temperature 80.31°C
and ten seconds of idle. No model was loaded. Inputs were materialized before
timing, both outputs were evaluated, and every timed arm built fresh graphs.
Twelve warmups per arm preceded eleven alternating forward/reverse rounds;
each sample contains four apply/evaluate/free repetitions. The timing run
rechecked exact parity before sampling. This is a same-process comparison
against native and an R1 control, not an old-binary baseline rerun.

| Tokens, B1/H64/D128 | Native ms | R1 ms | R2 ms | R4 ms | R4 vs native |
|---:|---:|---:|---:|---:|---:|
| 512 | 1.330771 | 1.329448 | 1.133229 | 1.039438 | −21.89% |
| 2048 | 5.480125 | 5.555490 | 4.391063 | 3.616761 | −34.00% |

R2 and R4 won all eleven paired comparisons at both lengths. R1 differs from
native by −0.10% at 512 and +1.38% at 2048, so the larger R4 gains survive the
source/control packaging change. Keep the candidate isolated or opt-in until a
full-model measurement establishes its effect. At the earlier approximate 10%
recurrence share, even removing recurrence entirely would not close a 22.7%
latency gap from 1159.73 to 1500 tok/s; the current share must be measured again.

The base checkout was `bd955e12ccf981adf20dbcb5cc99180e9991eacf` plus this
isolated candidate and focused test root. The timing binary SHA-256 is
`1de4316f3f2ffca46b725d32b960056ca27a56e43c0c40a17f6aff715e146045`.
The private `glm53-kda-value-rows-qualified-20261003` artifact retains exact
source, build recipe, parity/timing logs, raw samples and provenance. The gated
filter is `GLM KDA value-row isolated timing`, enabled by
`SUSHI_GLM_VALUE_ROWS_BENCH_OUT` naming its JSON result file. The earlier
CPU-only `glm53-kda-value-rows-20261003` snapshot is also preserved.


## CPU-only A6 unpack inspection

An isolated Metal 3.2 probe compiled with Apple metal 32023.921, `-O3 -S
-emit-llvm`, compares the existing byte spelling, an explicitly unrolled byte
control, and six aligned U32 reads for the same 32 six-bit coefficients. Each
arm uses the same scalar scale/bias arithmetic and BF16 store boundary. The
word spelling uses six scalar reads rather than a padded three-component vector;
it consumes exactly 24 bytes and requires only four-byte alignment. The audited
64-by-64 weight-loader geometry supplies that alignment.

| Isolated arm | AIR weight loads | Unpack structure | Bytes per thread |
|---|---|---|---:|
| Byte spelling | Three i8 loads in an eight-iteration loop | Loop | 24 |
| Unrolled byte control | 24 i8 loads | Straight-line | 24 |
| Word spelling | Six i32 loads | Straight-line | 24 |

The common scale/bias reads are excluded from the table. The unrolled control
separates changing load width from removing a loop. Integer reconstruction matched
10,000 deterministic random 24-byte inputs (320,000 coefficients), including
word-crossing positions. No tensor values or weights were modified.

This is AIR, **not final GPU instructions or memory transactions**. The GPU backend
may still combine byte loads; the complete NAX kernel also has different register
pressure from this probe. Therefore neither fewer device transactions nor a speed
gain is established. It justifies an isolated full-QMM parity/timing experiment
after the teacher releases the GPU, with both byte controls retained. No runtime
patch or production dispatch change has been made. The private artifact
`glm53-a6-unpack-audit-20261003` retains source, AIR, counts, compiler identity and
hashes; all work in this investigation was CPU-only.

### Complete A6 NAX qualification

The follow-up `src/glm5_a6_prefill.zig` keeps the native BM64/BN64/BK64,
WM2/WN2 tile and all NAX arithmetic. It embeds the existing attributed MLX
header, changing only its unsafe weight-loader operation. Its controls are
unchanged byte unpacking and the same byte loop with explicit full unrolling.
The word arm consumes exactly six scalar U32 values for each thread's 32
coefficients, reconstructs identical integers and preserves scale/bias arithmetic
and the BF16 staging boundary. No shared MLX library or vendor header changed.
The native loader's row, thread and K-tile offsets supply four-byte alignment;
there is no padded vector load or overread. The guard accepts only materialized,
contiguous BF16 input and A6/group128 banks at the actual 512/2048-row QKV and
output geometries. This remains an isolated experiment with no runtime import.

All three custom arms passed every native BF16 output bit at all four shapes.
The timing run rechecked that parity on four independent banks. Guard tests also
passed. The initial parity run had three passing tests, including its temporary
root, and one gated timing skip; the enabled timing run passed all four tests.

Timing used lock owner `glm53-a6-unpack-v61`, interactive QoS (`taskpolicy -a`),
maximum fans requested, 52.12°C initial temperature and ten seconds of idle.
No model was loaded. Inputs were evaluated before timing; twelve warmups per
arm preceded eleven alternating forward/reverse rounds with eight fresh
apply/evaluate/free evaluations per sample. One-bank and four-bank rotation
were measured separately. The four arms shared each input geometry and bank.

| Rows / projection | Banks | Native ms | Byte clone ms | Unrolled byte ms | Word ms |
|---|---:|---:|---:|---:|---:|
| 512 / QKV | 1 | 0.903520 | 0.899765 | 0.904260 | 0.883562 |
| 512 / QKV | 4 | 0.912244 | 0.913578 | 0.934817 | 0.881479 |
| 512 / output | 1 | 0.928750 | 0.927432 | 0.939609 | 0.889692 |
| 512 / output | 4 | 0.960401 | 0.954708 | 0.955619 | 0.920849 |
| 2048 / QKV | 1 | 2.726270 | 2.752114 | 2.749083 | 2.650989 |
| 2048 / QKV | 4 | 2.740838 | 2.762437 | 2.754406 | 2.662109 |
| 2048 / output | 1 | 2.924453 | 2.936890 | 2.910781 | 2.855145 |
| 2048 / output | 4 | 2.852390 | 2.871614 | 2.873328 | 2.796416 |

Word unpacking beats the unrolled-byte control in all eleven paired samples of
all eight cells. Against the unchanged byte clone, four-bank gains are 3.51%
(QKV) and 3.55% (output) at 512, and 3.63% / 2.62% at 2048. The clone/native
packaging difference reduces the corresponding 2048 advantage over native to
2.87% / 1.96%. This is a modest consistent primitive gain. Applying those two
native differences to three QKV projections plus one output across 34 KDA
layers suggests about 10 ms saved, before interactions with the rest of the
model; no end-to-end benefit or reduction in DRAM transactions is established.
Further integration needs a full-model gate and support for lazy graph inputs,
which the isolated materialization guard deliberately declines.

The base checkout was `a40d6cd73a3faef22c653e296e46a3d1da317f2b` plus the isolated
module/test root. The timing binary SHA-256 is
`446c6b8cbda6defd9c8f76b382d592cf3e42fa90ce159abdd478dfd0bd823662`.
The private `glm53-a6-unpack-qualified-20261003` artifact retains exact source,
vendor header, build command, parity/timing logs, raw samples, telemetry and
provenance. Its gated filter is `GLM A6 unpack isolated timing`, enabled by
`SUSHI_GLM_A6_UNPACK_BENCH_OUT` naming its JSON result file.

### Eight-SIMD A6 tile: 128×64 with unchanged per-SIMD geometry

The next isolated variant doubles BM64→128 and WM2→4 while retaining BN64,
BK64 and WN2. Each SIMD still computes SM32/SN32 with the same accumulator,
fragment and K order. The 256-thread group shares one decoded weight tile
across four M SIMD groups instead of two. The explicit shared buffer remains
9 KiB (`64*72*BF16`); this quantized body has no separate threadgroup input tile.
`QuantizedBlockLoader.n_reads` falls from eight to four, so each thread decodes
16 coefficients rather than 32. Thread count across the projection stays the
same, while threadgroup count and logical weight-tile decodes halve. Register
allocation, residency and actual memory transactions still require measurement.
This differs from the earlier rejected BM128/BN32/WM4/WN1 aspect experiment.

`glm5_a6_prefill.applyTall` admits only materialized contiguous A6/group128 BF16
inputs at the actual 2048-row QKV/output shapes. The unchanged header compiled
and every native BF16 output bit matched at both shapes. Timing rechecked exact
output across four independent banks for native, the BM64 byte clone and the
BM128 byte candidate. No word-unpack arm was stacked into this experiment.

The exclusive run used lock owner `glm-a6-tall8-v61`, interactive QoS,
maximum fans requested, 49.60°C initial temperature and ten seconds idle.
Twelve warmups per arm preceded eleven alternating forward/reverse rounds,
each with eight fresh apply/evaluate/free evaluations and four-bank rotation.
Inputs were evaluated before timing; no model was loaded. The focused parity
run had two passes including its temporary root and one timing skip; the
subsequent timing run passed all three tests.

| 2048-row projection | Native ms | Same-body BM64 ms | BM128/WM4 ms | vs same-body |
|---|---:|---:|---:|---:|
| QKV 4096→8192 | 2.867203 | 2.846067 | 2.716942 | −4.54% |
| Output 8192→4096 | 2.891796 | 2.936109 | 2.809880 | −4.30% |

The candidate won all eleven paired comparisons against the same-body control
for each shape. Its advantages against the native library were 5.24% and 2.83%,
so the packaging/control difference matters. The gain is consistent but modest;
fewer logical weight decodes did not produce a proportional latency reduction.
The isolated candidate remains available without a production import. It has no
full-model qualification, and its gains cannot be added to the word-unpack
numbers without a combined comparison.

The private `glm53-a6-tall8-20261003` artifact preserves exact candidate source,
test root, build command, raw parity/timing, summary, telemetry and binary/source
hashes. The base was `ae12b393` plus this isolated extension. Its gated filter is
`GLM A6 eight SIMD tall tile isolated timing`, enabled by
`SUSHI_GLM_A6_TALL_BENCH_OUT` naming the output JSON.

### Runtime BF16 dequantization before dense NAX

`src/glm5_a6_dense_once.zig` tests dequantizing one complete A6 bank to BF16 per
projection and then running ordinary dense matmul. It uses existing
`Ops.dequant`, transpose and matmul, with no new shader or persistent expanded
weight tensor. The guard for any later integration must stay at B1/T2048,
BF16/GPU/A6/group128 and the actual QKV/output widths. Each temporary bank is
67,108,864 bytes, so an async2 bound of four KDA projections per layer is
536,870,912 additional bytes before reclamation; a runtime hook must account
for that transient memory and keep original stored tensors unchanged.

The initial qualification compared every decoded BF16 coefficient against an
independent implementation of the native NAX integer unpack, FP32 scale/bias
arithmetic and BF16 store boundary. The finite fixture scales times a six-bit
integer have an exact FP32 product, so FMA contraction cannot change this oracle.
Both shapes had zero mismatches across 33,554,432 weights each. Affine and dense
outputs also matched every BF16 bit: 16,777,216 QKV values and 8,388,608 output
values. A sampled FP64 dot reference checked 256 dispersed output positions per
shape; both arms had identical RMS/max errors, without tolerance relaxation.

The inclusive microbenchmark reconstructed, evaluated and freed a fresh BF16
weight/matmul graph on every repetition. Dequantization, graph construction,
allocation, transpose view, matmul, evaluation wait and frees were all timed.
The qualification's expanded bank was released before sampling, and no expanded
bank was reused by the timed path. Inputs were materialized; each shape used one
synthetic bank, with different qualification and timing seeds. The timing
fixtures rechecked all decoded weights and recorded zero output mismatches.

Twelve warmups per arm preceded eleven alternating AB/BA rounds with four
repetitions per sample. The run used interactive QoS, exclusive GPU lock owner
`glm-a6-dense-once-v61`, maximum fans requested, initial temperature 47.06°C
and ten seconds idle. Other builds paused for timing; no model was loaded.

| 2048-row projection | Affine NAX ms | Dequant + dense inclusive ms | Reduction | Paired wins |
|---|---:|---:|---:|---:|
| QKV 4096→8192 | 2.732989 | 2.530552 | 7.41% | 11/11 |
| Output 8192→4096 | 2.863562 | 2.712552 | 5.27% | 11/11 |

The result is a consistent primitive win with exact fixture arithmetic. Three
QKV projections plus one output across 34 KDA layers suggest about 25.8 ms saved,
before full-model interactions and memory effects. It does not establish a
1500 tok/s workload or justify a default change; a narrow opt-in full-model
counter/peak-memory gate is next. The focused qualification and timing runs
each had two passing tests including their temporary root and one gated skip.

The base checkout was `e4be4673ff8b3f0cfad231818b48dcbb00ccae3b` plus the isolated
module. The timing binary SHA-256 was
`0313f20af242e39140396f065a0d15aade52b4307322b03b52f1f76afc71ff7a`.
The private `glm53-a6-dense-once-20261003` artifact retains source, explicit build
command, test root, complete weight/output statistics, FP64 sample errors,
raw timing, summary, telemetry and hashes. Its gates are
`SUSHI_GLM_A6_DENSE_PARITY_OUT` and `SUSHI_GLM_A6_DENSE_BENCH_OUT` for the
`GLM A6 dense once` test filter.

The full-model gate is now exposed through `SUSHI_GLM_A6_DENSE_PREFILL=1`.
`Linear.apply` uses the candidate only on NAX-capable GPU streams with BF16
B1/T2048 inputs and the two admitted A6/group128 bank shapes. Decode rows,
F32/B2/A8/group64 and retained small tensors keep their original path. The
native diagnostic resets/reports `a6_dense_prefill_dispatches`, reports the
per-matrix and pending-layer transient bill, and checks enabled-only memory
headroom before warmup. Async warmup is included in the conservative bill even
when the timed profile is synchronous. With async2 the bound is 512 MiB, without
changing stored/resident tensor bytes. Focused opt-in Linear, guard and bill
tests passed (four including their test root, two gated research skips);
eligible outputs were exact and three-row decode did not increment the counter.
The subsequent full native checkpoint run qualified the hook at2K. The fixed
ReleaseFast binary was built from `bcc89c6e` plus the hook subsequently committed
unchanged as `63bfbba3`. It used the2.3bpw A6-trunk target, R4 recurrence, lane/down,
dense SDPA prefill, async2 prefill/async4 decode, chunk2048,64 output tokens and
one warmup with eight decode forwards. Components, route histograms and profiling
were disabled. Foreground QoS, exclusive GPU lock, maximum fans and ten seconds
idle below90°C were recorded; no compiler/GPU job overlapped.

Prefill measured1,219.30194 tok/s versus inherited R4's1,193.70359 (+2.14%). All64
output IDs were unchanged. The hook engaged136 times (four projections across
34 KDA layers); serial decode measured24.6266 tok/s. Peak was95,311,918,132 bytes,
160 MiB above the prior arm, within the declared512 MiB additional transient
bound. This single-run full-model qualification is consistent with the isolated
paired win, but does not establish broader contexts, a default policy or the
1,500 tok/s goal. The option remains off by default. Private artifact
`glm53-a6-dense-full-20261003` retains source diff, binary hash, settings, raw
result, ID comparison and telemetry.

## Head-batched absorbed MLA projections on MLX v0.32.3

The new runtime (MLX `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`) changes an earlier
audit conclusion: `qmm` now admits NAX for both transposed weights and affine
non-transposed weights. The cached absorbed path still supplies `[T,64,1,D]`,
so its matrix row count is one and it selects qvm/qmv. The isolated
`glm5_mla_prefill_batch` helper permutes this to contiguous `[64,T,D]`, calls the
same native affine QMM with the original banks, then restores contiguous
`[T,64,1,N]`. Query uses K256→N512/non-transposed; value uses K512→N256/transposed.
For T2048 the two dispatches drop from 5,242,880 vector groups to 24,576 NAX groups.
This is a scheduling/implementation change, not a claim of identical arithmetic.

No weights are expanded or repacked. Input/output permutation planes add
384 MiB across both projections per T2048 MLA layer. The conservative async2
copy bill is 768 MiB; it is separate from and added to the A6 dense-expansion bill.
The compressed MLA cache remains BF16 and KDA state remains FP32. The public
helper guard is NAX-capable GPU, BF16, H64, T128–2048 and the actual A6/group128
U32 bank plus BF16 scale/bias grids. F32 is researched for drift but declined
by the runtime guard.

Qualification recorded full raw output differences at T128/512/2048 for both
orientations and BF16/F32 inputs, without a relaxed tolerance. At T2048 the BF16
query comparison differed in 42.7% of bits, relative L2 0.2576% and maximum
absolute difference 0.03125; value differed in 52.5%, relative L2 0.3193% and
maximum absolute difference 0.03125. Very large ULP distances occurred around
zero/sign crossings; raw ULP bins remain in the artifact and are not concealed
by an output-magnitude tolerance.

A 256-position FP64 dot reference separates two weight interpretations. Native
vector arithmetic is closer to the original FP32 decoded coefficients; NAX is
closer to coefficients rounded to BF16 before multiplication. This weight
boundary changes along with the contraction order. F32 research differences
had relative L2 about 0.0654%; native reference RMS was around 1e-7 while NAX RMS
was 5e-4–7e-4. It is not an exact-layout optimization.

The exclusive paired microbenchmark included native QMM, all contiguous copies,
construction, evaluation and frees. Six warmups per arm preceded eleven AB/BA
rounds with two repetitions each. Inputs were evaluated first. Interactive QoS,
lock owner `glm-mla-headbatch-v61`, maximum-fan request, 47.81°C initial temperature
and ten seconds idle isolated the run; no model was loaded. Every cell had 11/11
paired wins. This compares one synthetic bank per geometry, not full-checkpoint
traffic.

| T / projection | Original M1 ms | Head-batched ms | Reduction |
|---|---:|---:|---:|
| 128 / query | 1.033062 | 0.254646 | 75.35% |
| 512 / query | 3.593270 | 0.538916 | 85.00% |
| 2048 / query | 13.689042 | 1.685020 | 87.69% |
| 128 / value | 1.407166 | 0.264021 | 81.24% |
| 512 / value | 5.020729 | 0.537646 | 89.29% |
| 2048 / value | 19.445646 | 1.639083 | 91.57% |

At T2048 the combined primitive cost is 33.135→3.324 ms, suggesting roughly
328 ms across eleven MLA layers before pipeline/attention effects. The whole
wide-prefill cliff also includes index selection and latent attention; this
table does not assign their cost or establish a whole-model gain.

`SUSHI_GLM_MLA_PREFILL_BATCH=1` enables the default-off qualification hook only
in the absorbed branch of `Mla.applyMode`; the existing cold dense-SDPA branch
is unchanged. `bind(false|true)` returns a scoped binding with `restore()`, so
one process can compare independent Requests using the same loaded weights.
Query/value dispatch counters and the native diagnostic copy bill identify
engagement. Focused guard/binding/bill tests passed, with research tests gated.
A meaningful next gate is old/new same-model logits after long 4K prefill plus
the same teacher-forced continuation. That kernel-drift comparison is distinct
from quantization KLD against a lossless BF16 teacher. The earlier 512-token
teacher capture does not exercise this changed prefill path and cannot qualify
it. Defaults must stay unchanged until the long-prefill drift/quality gates pass.

The private `glm53-mla-headbatch-20261003` artifact retains exact source, test
root/command, complete BF16 ULP/F32 differences, both FP64 reference errors,
timing samples/summary, telemetry and source/binary/runtime hashes. Its gates
are `SUSHI_GLM_MLA_BATCH_PARITY_OUT` and `SUSHI_GLM_MLA_BATCH_BENCH_OUT` under
`GLM MLA headbatch prefill`. No model quality or throughput claim is made here.

The subsequent4K/64 diagnostic used the same loaded2.3bpw weights with the
original projections as reference, then head batching for the same long prefix
and teacher-forced continuation. Eleven query and eleven value calls engaged.
First-position KL was0.00028924; mean KL across64 positions was0.01160392,
top1 matched63/64 and cosine was0.9991044. No EOS occurred in the64 reference
tokens. The largest position KL was0.395861. This is material kernel drift;
the ordinary BF16 head-batch option remains unpromoted and default off.
It is not quantization KLD against a lossless teacher.

The fixed ReleaseFast diagnostic and settings are retained in private artifact
`glm53-long-prefill-drift-20261003`. Its source snapshot matches `c57f317b` plus
`glm5_prefill_drift.zig`, with direct attention disabled to isolate projections.
Peak was95,500,874,808 bytes. Reference/candidate prefill was10.192/8.969 seconds,
but cold reference followed by candidate is not an interleaved throughput gate.
The user subsequently accepted ordinary NAX compound rounding with BF16
operands/intermediates and FP32 accumulators, and requested no precision
restoration. The measurement above remains the numerical drift record.


### Archived MLA precision boundary audit

A stopped, isolated experiment on MLX v0.32.3 split FP32 affine weights into
BF16 high/low parts and used two FP32-output NAX GEMMs. All 8,388,608 weights per
direction reconstructed exactly. Query output relative L2 difference versus
native M=1 fell to 0.00002153. Value required restoring the native classic-qmv
BF16 four-input sum before group-bias multiplication; that reduced its relative
L2 difference from 0.00251971 to 0.00002420. Neither result was bit-exact.

At 2,048 rows / 64 heads, quiet inclusive medians were query 13.926770 to
6.360979 ms and value 19.757021 to 15.079562 ms, with 11/11 paired wins each.
Decoding, splitting, copying, both GEMMs, the value bias correction, evaluation,
and free were included. The ordinary BF16 head-batched path is substantially
faster. The user explicitly accepted BF16 operands and intermediates with FP32
accumulators and requested no precision restoration. The prototype and research
seam were therefore removed and archived; no model hook or default changed.
These are component findings, with no model-level quality or throughput claim.
