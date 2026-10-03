# Engine: EXL3 trellis experts (`expert_layout == .exl3_k4`)

How routed experts in turboderp's EXL3 trellis format are decoded and multiplied: the rate, codebook and window a
pack names, the prefill GEMM and the four-dispatch decode chain, and the parity bars their tests hold. Read this
before touching `src/exl3/`, `src/expert_quant.zig` or `moeExl3`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [pack-format](pack-format.md) (the on-disk contract),
[engine-kernels](engine-kernels.md), [engine-expert-streaming](engine-expert-streaming.md),
[perf-baselines](perf-baselines.md#exl3), [quality-kld](quality-kld.md).

## Code map

| File | Role |
|---|---|
| `src/expert_quant.zig` | Expert layout detection from PACKED shapes: `.quantized_split` (affine banks) vs `.exl3_k4` (trellis); affine (bits, group_size) solved from geometry |
| `src/exl3/root.zig` | The `sushi_exl3` module's host API: `expert_quant` parse (`parseExpertQuant`), `admitTopK`, `trellisAdmitted`, `moe` (the one dispatch) |
| `src/exl3/expert_exl3.zig` | Host reference decoders (MUL1, MCG), `Rate`, `Window`, `Decode`, fixtures |
| `src/exl3/expert_exl3_kernels.zig` | Prefill run-aligned 32-row window GEMM (NAX body, K4 fast branch; simdgroup-matrix body off NAX; scalar body), decode chain (`moeSwigluFused`), `DECODE_ROWS_MAX` (16), `usesPrefillArm` |
| `src/expert_bf16_kernels.zig` | bf16 selected-expert kernels over a slab (`gateUpSwiglu`; `downReduce`) for the unquantized HF checkpoint |

**`src/exl3` is a module** (`sushi_exl3`), so another MLX host (mlx-serve) can serve EXL3 packs through the same code.
It reaches `mlx`, `log` and `io_util` through an `mlx_host` import whose root file exposes them as `pub const`; here that
root is `src/main.zig` (and `src/tests.zig` for tests), and it can never import a Sushi file by path. Its tests run as
their own artifact (`exl3-test`) on `zig build test`.

The NAX compile-probe regression test checks hardware capability independently
of the runtime failure latch. It deliberately latches dispatch off before
probing the real kernel, so an earlier SIMD fallback cannot hide a broken NAX
kernel on supported hardware.

## Format as the engine sees it

- Routed experts are stacked per layer as `[E, ...]` so gather kernels index expert e on axis 0; 16x16 tiles,
  `suh`/`svh` with the H128 Hadamard; `config.json` carries `expert_quant = {format: exl3, k, codebook: mul1|mcg}`
  (plus `window`); per-tensor rate read from the trellis shape. Every other module stays the affine pack's.
- **A rate is K = n/16**, n the packed halfwords per 256-weight tile (36 = K2.25, 48 = K3, 64 = K4): weight t's
  codeword is the 16-bit window ending at `((t+1)*n)>>4`, so its fresh bits follow from n and the pattern is never
  stored. Even n in [16, 128] admits (K1 to K8); `expert_quant.k` may be fractional JSON.
- **Every reader keys on n, never on an integer K** (`exl3.Rate`, kernel template `NHW`, cache keys,
  `exl3ExpertBytes`); a K printed anywhere reads 2.25, not 36.
- **Every fast path serves every admitted n; a guard test enumerates them** (`every admitted rate takes the fast
  arms`). The funnel readers take their word indices and shifts from n at compile time: eight weights span n/2 whole
  bits, and 16 weights span n. Below n64 every codebook reads through them. n64 keeps its packed K4 branch, which does
  the same reads at a word-aligned rate, one output tile per threadgroup:
  - decode GEMV lane: 64 bits ending at the lane's last bit. A third word is read only when the first codeword can
    start before the two words (every n from 42 to 62 but 48).
  - simdgroup-matrix group: one or two 32-bit funnels, split at the widest weight whose first codeword still fits.
  - NAX fragment: one funnel per quad of weights; above n50 a quad's codewords pass 32 bits, and the funnel reads 64
    bits from three words.
- **The window is a pack field** (`expert_quant.window`, absent = 16, 8..16 admitted): the codeword is masked to the
  window in the one helper every weight kernel inlines, and kernel slots are keyed by codebook AND window. A w16
  bitstream decodes to different weights at every other window, so a window can never come from a flag.
- **The codebook follows the MODEL at every dispatch**: `exl3.moe` calls `kernels.setDecodeParams`
  (codebook + window) before each dispatch because several EXL3 packs can be resident at once; every weight kernel
  inlines `exl3_pairh` from `codebookHelpers`, built per (codebook, window). A codebook name this build does not decode is refused at load
  ([pack-format](pack-format.md#configjson)). A/B lever:
  `SUSHI_EXL3_CODEBOOK_AB=1` on the `codebook A/B` test.
- **A shard's `__metadata__` stamp is CHECKED against `expert_quant` before upload**
  (`mimo_source.validateShardStamps`): see [pack-format](pack-format.md#the-shard-stamp).
- `num_experts_per_tok` above 32 refuses by name (`Exl3TopKExceedsReduceBank`).

<a id="mimo"></a>
## MiMo EXL3 packs

A MiMo EXL3 pack serves RESIDENT: its banks nest under `model.layers.` (qwen4's under
`language_model.model.layers.`), `expertStreamingRequired` excepts `.exl3_k4`, and the trunk still takes the
source FP8→bf16 loader (`usesMimoSourceTrunk`), billed dense by `mimoSourceResidentBytes`. See
[arch-mimo-v2](arch-mimo-v2.md).

## GLM clamped experts

`moeClamped` applies the gate upper bound and symmetric up bound in FP32 before SwiGLU. Decode keeps
slots in their original top-k order. Prefill prepares gate/up directly from token rows, shares one window
table across all three projections, and uses its GPU inverse routing to finish directly from the sorted
down plane. The reducer still visits the original top-k order; stride-table and unsupported-bank
fallbacks retain scatter-plus-reduce. It does not materialize repeated and sorted token planes or sort
the inverse permutation.

Decode also prepares both natural-order input planes directly from token rows in one kernel. The two
expert-specific Hadamards remain necessary because gate/up scales differ, and their F16 stores and all
subsequent GEMV, clamp and reduction arithmetic are unchanged. This removes the repeated token plane
(64 KiB for one GLM token, top-k eight) and replaces two prepare dispatches with one, without resident
weight copies. Direct plane tests cover distinct signed gate/up scales, expert IDs through 287, widths
128/4096 and rows 1/8/16; end-to-end rate coverage includes rows 1/2/8/16/17.

Matching gate/up bank shapes and rates also share one cooperative GEMV dispatch. Grid Z selects the
projection's original input and weight pointers; the original K split remains zero and the four-simdgroup
reduction and F16 stores are unchanged. Mixed-rate banks retain the separate-call fallback. Direct tests
check both output planes byte-for-byte at every supported 2–4 bpw rate with distinct inputs and banks,
and at 4096/2048 widths with top-k eight and one/sixteen rows. The new path does not concatenate or
repack weights. Prefill continues to use the existing sorted GEMM path.

The optimized routing matches the staged path bit-for-bit at every even packed width n32–64 (2–4 bpw),
with BF16 and FP32 inputs/outputs, top-k eight, and decode/prefill rows. A K2.25/W12 case also checks the
GLM hidden/intermediate widths 4096/2048. These are arithmetic and dispatch-structure checks; they do not
establish a measured full-model speedup.

Aligned GLM prefill also avoids the scattered F16 down intermediate (32 MiB at 512 rows, top-k eight
and hidden width 4096) and one dispatch. Exact staged-output tests cover all 17 rates at 288 experts,
real 4096/2048 projection widths, sparse/all-expert/skewed routing, and the retained stride fallback.

### Cooperative clamped middle/down fusion

The GLM K2.25/W12 MCG decode arm can fuse the clamped middle transform with the cooperative down
projection. It uses the original middle arithmetic, stores the transformed input as F16 in threadgroup
memory, and then runs the original four-simdgroup cooperative reduction and F16 output store. It does
not substitute the normal EXL3 FP32 split-K chain. Four output tiles per group are used for one row;
eight are used for two through sixteen rows.

The cached diagnostic switch `SUSHI_EXL3_CLAMPED_MIDDLE` accepts `auto` (also the absent/default
setting), `0`/`off`, and `1`/`on`. Auto retains the separate one-row path and permits the validated
2–16-row candidate. Off disables fusion; on explicitly permits the one-row research arm as well.
Full-model attribution did not reproduce the warm-bank serial improvement, so one-row fusion is no
longer enabled by default. The multirow component gains still require separate DFlash/full-model
measurement before broader throughput claims.

The served guard is BF16 output, hidden/intermediate widths 4096/2048, top-k eight and matching K2.25
projection rates. Other widths, storage formats, codebooks and mixed rates retain the separate path.
The kernel's direct F16 parity tests cover all 17 rates from 2 through 4 bpw with both tile choices;
production-width tests cover 1/2/4/8/16 rows. Whole-chain staged-byte tests remain required. The runtime
engagement counter is `clampedMiddleDispatchCount`, with a matching reset function.

A warmed M5 Max component comparison used ReleaseFast, foreground QoS, an exclusive GPU lock,
20 warmup A/B pairs and 100 alternating A/B–B/A pairs. Workers paused GPU work and builds. Max fans
were requested and a 10-second cooldown applied; the controller did not confirm spin-up on the cool
machine. Numbers include graph construction and evaluation wait, not just GPU execution. The banks
were synthetic eight-expert K2.25/W12 banks; no full-model speedup is inferred.

| Rows | Output tiles/group | Separate middle + down | Fused | Stage latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 4 | 241.625 µs | 236.083 µs | 2.29% |
| 2 | 8 | 294.875 µs | 287.167 µs | 2.61% |
| 4 | 8 | 465.584 µs | 438.333 µs | 5.85% |
| 8 | 8 | 788.708 µs | 686.584 µs | 12.95% |
| 16 | 8 | 1387.833 µs | 1169.042 µs | 15.77% |

The earlier clean paired sample showed a 4.55% one-row reduction, so the serial gain is small and
context-dependent. One output tile/group was slower and is not selected. An initial timing that may
have overlapped another GPU test was discarded. Measurements were built from `06422dd4` plus this
change; raw records include binary/source hashes. Real-checkpoint throughput and unchanged output IDs
must be checked on the combined final build.

### GLM opt-in lane-ordered gate/up input

`SUSHI_GLM_LANE_PAIR=1` enables a separate exact gate/up layout for BF16-input GLM clamped decode,
rows 1–16, equal-shaped MCG/W12 banks and all supported 2–4 bpw rates. The default is off. Unsupported
or mixed-rate inputs retain the original path. The pair preparation stores each 16-element tile in
four groups `(2q, 2q+1, 2q+8, 2q+9)`; each cooperative lane then loads one aligned `half4`. The
Hadamard arithmetic, eight accumulators, K iteration order, final row/group reduction and F16 stores
are unchanged. This does not substitute MiMo's different reduction tree or split-K arithmetic.

The matched component experiment included both pair preparation and gate/up GEMV, with distinct
scale planes and bank contents. All 17 rates passed exact F16 output parity, including production
4096→2048 geometry. Three warmup ABBA rounds preceded forty samples per arm; the GPU lock, foreground
QoS and paused workers isolated timing. Fan maximum was requested, but actual maximum RPM was not
confirmed (reported RPM zero); initial maximum temperature was about 54°C, followed by ten seconds idle.

| Rows | Original | Lane `half4` | Reduction |
|---:|---:|---:|---:|
| 1 | 437.459 µs | 361.791 µs | 17.3% |
| 3 | 670.875 µs | 515.042 µs | 23.2% |
| 7 | 1239.959 µs | 911.166 µs | 26.5% |
| 16 | 2567.083 µs | 1824.833 µs | 28.9% |

These are warm synthetic eight-expert component measurements, not full-checkpoint throughput.
The gate remains off pending a real-model comparison. Pair-kernel and routed-chain engagement have
separate `lanePairCalls` and `lanePairChainCalls` counters. The standalone parity and timing tests
are gated by `SUSHI_GLM_LANE_PAIR_TEST` and `SUSHI_GLM_LANE_PAIR_BENCH` respectively.

### GLM opt-in lane-ordered down input

`SUSHI_GLM_DOWN_LANE=1` selects an experimental separate clamped middle preparation followed by a
lane-ordered cooperative down projection. It is off by default. It is restricted to BF16-output
clamped decode with equal-rate MCG/W12 gate/up/down banks; all unsupported cases retain the existing
path. The middle keeps its original clamps, Hadamard arithmetic and F16 store. Only the stored lane
layout and the down kernel's four input loads change. Accumulator and final reduction order remain
unchanged. The comparison also tested an uncalled version that permutes the fused middle's
threadgroup buffer and uses threadgroup `half4` loads, preserving its single dispatch.

All 17 rates and production 2048→4096 geometry passed exact F16 comparisons between all four arms.
The integrated full clamped MoE chain also matched BF16 bits at every rate and production rows
1, 3, 4, 7 and 16.
The warmed experiment used three palindrome warmup rounds and forty samples per arm, an exclusive
GPU lock, foreground QoS, ten seconds idle and paused workers. Fan maximum was requested but RPM
confirmation was unavailable; initial maximum temperature was about 50°C.

| Rows | Original separate | Original fused | Lane separate | Lane fused |
|---:|---:|---:|---:|---:|
| 1 | 247.416 µs | 237.708 µs | 221.750 µs | 224.125 µs |
| 4 | 478.834 µs | 421.417 µs | 359.167 µs | 408.208 µs |
| 8 | 803.750 µs | 691.667 µs | 619.708 µs | 654.458 µs |
| 16 | 1393.000 µs | 1173.542 µs | 1031.334 µs | 1104.625 µs |

The separate lane layout won this component experiment, reducing time by about 10–15% versus the
current policy (separate at one row, fused at multiple rows). It remains default-off until a real
checkpoint confirms the benefit. These synthetic eight-expert component timings cannot predict cold
bank traffic or full-model throughput. `downLaneCalls` reports integration engagement. Tests and
benchmark are gated separately by `SUSHI_GLM_DOWN_LANE_TEST` and `SUSHI_GLM_DOWN_LANE_BENCH`.

### GLM 512-token window-height study: retain 32

A separate untimed capture provided all 42 routed layers' 512-token/top-8 histograms. Every histogram
had 4096 assignments, 288 possible experts and zero counts in the remaining diagnostic slots. Median
active experts were 230 (range 197–250). The current kernel already skips its second 16-row matrix
operation for runs of at most 16 rows, so a 32-row window does not imply 32 rows of computation.

The real distributions required a median 287.5 live 32-row windows and 398.5 16-row matrix tiles.
Minimum-tile padding was 35.76% (range 33.33–37.56%). Changing the window to 16 leaves that matrix-tile
count unchanged while increasing weight-decode windows by a median 37.95% (range 31.49–43.38%).

A rejected prototype retained one 16-row accumulator and split longer runs internally, preserving the
per-output K order and inverse routing. Exact F16 parity passed all 17 rates on boundary-sized runs,
and all production replay arms were byte-identical. The replay used captured layers 7, 15 and 32,
synthetic 288-expert K2.25/W12 banks, production matrix dimensions and 4096 prepared rows. Metadata
was precomputed. Twenty samples per arm were interleaved forward/backward after warmup, with an
exclusive GPU lock, foreground QoS, max-fan request and ten-second cooldown. Other workers paused
builds and GPU work. This measured individual GEMMs, not full-model throughput.

| Layer | Projection | Current, window 32 | Current, window 16 | One accumulator, window 32 | One accumulator, window 16 |
|---:|---|---:|---:|---:|---:|
| 7 | 4096→2048 | 2.541 ms | 2.760 ms | 3.030 ms | 2.739 ms |
| 15 | 4096→2048 | 2.620 ms | 2.840 ms | 3.072 ms | 2.820 ms |
| 32 | 4096→2048 | 2.679 ms | 2.875 ms | 3.079 ms | 2.872 ms |
| 7 | 2048→4096 | 2.379 ms | 2.650 ms | 2.781 ms | 2.630 ms |
| 15 | 2048→4096 | 2.472 ms | 2.735 ms | 2.844 ms | 2.715 ms |
| 32 | 2048→4096 | 2.533 ms | 2.769 ms | 2.868 ms | 2.743 ms |

The existing 32-row implementation won every case; alternatives were approximately 7–19% slower.
The occupancy hypothesis did not compensate for losing weight-decode reuse and two-tile scheduling.
The prototype was archived and removed, and the validated production selector was left unchanged.
The replay was built from `99107a48` plus the isolated prototype; raw records retain binary, source and
histogram hashes. Reducing the actual padding would require a different minimum matrix tile or another
validated short-run strategy, not simply changing the window-height setting.

### GLM narrower output groups: retain 128

The served NAX body has four independent SIMD groups per 128-thread group. Each SIMD group computes
16×32×16 matrix tiles and optionally a second 16-row tile. There is no explicit threadgroup weight or
input tile in this body. Narrowing the output group therefore changes scheduling, not shared-memory
allocation or the number of SIMD matrix operations. No per-core occupancy claim follows from it.

An uncalled prototype used 64- or 32-thread groups, changing only the output-group base. Each output
kept the original SIMD arithmetic, K order and F16 store. All 17 rates passed exact F16 parity across
run lengths 1, 7, 15, 16, 17, 31, 32, 33 and 64. Production replay used the same captured routing
histograms and synthetic 288-expert banks as the window-height study, with precomputed metadata,
two warmup palindromes and twenty interleaved samples per arm under an exclusive quiet GPU window.

| Layer | Projection | Group 128 | Group 64 | Group 32 |
|---:|---|---:|---:|---:|
| 7 | 4096→2048 | 2.537 ms | 2.494 ms | 2.536 ms |
| 15 | 4096→2048 | 2.615 ms | 2.579 ms | 2.616 ms |
| 32 | 4096→2048 | 2.678 ms | 2.629 ms | 2.675 ms |
| 7 | 2048→4096 | 2.375 ms | 2.349 ms | 2.388 ms |
| 15 | 2048→4096 | 2.476 ms | 2.457 ms | 2.490 ms |
| 32 | 2048→4096 | 2.528 ms | 2.510 ms | 2.555 ms |

All six production cases were byte-identical. Group 64 improved these isolated GEMMs by only
0.71–1.81%; group 32 was effectively flat or slower. This single component run does not establish
full-model benefit. The prototype and reproducibility hashes were archived, the prototype was removed,
and production remains on group 128. The fan maximum was requested but spin-up was not confirmed;
initial temperature was below 38°C, followed by ten seconds idle. No full-model arm was run.


### GLM n36 SIMD word sharing: retain direct reads

An isolated reader changed only how the existing NAX body obtains its packed
words. Each SIMD/K16 step consumes two adjacent 72-byte n36 tiles, or 36 U32
values. The original four funnels issue eight overlapping U32 reads per lane.
The candidate loaded words 0–31 once across the SIMD and words 32–35 in lanes
0–3, then shuffled the original indices into the unchanged funnel/codebook
operations. Tile shape 16×32×16, K order, two-accumulator WIN32 reuse and F16
stores stayed unchanged. No threadgroup memory or barrier was added.

Exact native F16 parity passed at 4096→2048 and 2048→4096 with E288/n36/MCG/W12,
run lengths 1/7/15/16/17/31/32/33/65 and an active expert287. The subsequent
replay also passed exact output bits. It used the captured 512-token layer7
counts multiplied by four, **synthetic 2K routing**, and synthetic banks:
16384 assignments, 210 active experts, 633 live WIN32 windows, 1128 M16 tiles
and a routing-independent capacity of 800. This is not an actual 2K capture.

The exclusive run used interactive QoS (`taskpolicy -a`), GPU lock owner
`glm-prefill-shuffle-v61`, maximum fans requested, 53.67°C initial temperature
and ten seconds idle. Twelve warmups per arm preceded eleven alternating AB/BA
rounds with four fresh apply/evaluate/free repetitions per sample. Inputs and
metadata were materialized before timing. No model was loaded.

| Projection | Native NAX ms | Shuffle ms | Change | Paired wins |
|---|---:|---:|---:|---:|
| Gate/up 4096→2048 | 5.752270 | 10.670541 | +85.50% | 0/11 |
| Down 2048→4096 | 5.722843 | 10.461010 | +82.79% | 0/11 |

The candidate was rejected and removed, including its research seam and module
import. Fewer source reads do not establish fewer physical transactions; cache
coalescing can serve overlapping addresses, while shuffles add instructions
and register dependencies. No full-model arm was warranted. Both explicit
qualification/timing tests passed. An earlier host-root-only run executed no
dependency-module tests and was excluded from the qualification evidence.

The private `glm53-prefill-shuffle-20261003` artifact preserves exact source,
seam/import snapshots, explicit test root, build command, replay counts, logs,
samples and provenance. The base was `99b62c52` plus the isolated module/seam;
no production dispatch was changed. The timing binary SHA-256 was
`2f3ec51565414fe61b87f4a048a9063f2b5babd9216be9b55f17c31b601f7d08`.
Its gated output variable was
`SUSHI_GLM_PREFILL_SHUFFLE_BENCH_OUT`, with replay counts supplied through
`SUSHI_GLM_PREFILL_SHUFFLE_COUNTS`.

## Rejected GLM three-row forced NAX chain

On 2026-10-03, an isolated experiment sent GLM verification's three rows through
the existing sorted prefill NAX body instead of the group2 serial lane pair and
grouped lane down chain. The existing body already executes M16/N32/K16 for an
expert run of one to three rows: clamped input pointers keep dummy reads valid,
and masked stores emit only live rows. No physical padding buffer or expanded
weight bank was needed. Operands and original coefficient storage remained F16;
destination accumulators remained FP32.

The component replay used selected original K2.25/MCG/w12 bank bytes and the
three-row prefixes of captured four-row routes at layers 3/20/34, hidden 4096,
intermediate 2048, top-8, clamp 10. Activations and scores were fixed synthetic
values, identical to the earlier lane replay. This was not a model request.
Although the fixture banks compact selected experts, GPU metadata retained
the production logical E=288 count and 289-window capacity. All sorting,
metadata, preparation, GEMMs, middle, finish, evaluation and free were included.
Empty windows return before any bank read. No dispatch default changed.

There were 23/17/15 active experts across 24 assignments, giving 1472/1088/960 live
GEMM threadgroups across the three projections and 18496 total dispatched groups.
Implicit M16 work was 15.33/11.33/10 times the live assignment count. Compared
with the current chain, BF16 output relative L2 differences were
0.00083037/0.00074590/0.00082764; maximum absolute differences were
0.00390625/0.001953125/0.00390625. The independent scalar FP32 format reference,
including clamping and routed reduction, gave nearly equal RMS error for both
arms (roughly 0.00037–0.00046). No bit-exact claim was made.

| Layer | Group2 serial lane chain, µs | Forced NAX, µs | Latency change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| 3 | 661.180 | 868.583 | +31.37% | 0/11 |
| 20 | 574.611 | 759.888 | +32.24% | 0/11 |
| 34 | 589.430 | 655.194 | +11.16% | 0/11 |

The ReleaseFast binary used MLX v0.32.3 (`64ea011c`), base `59149d82` plus the
isolated candidate/research exports. The exclusive run used interactive QoS,
maximum fans, a 50.76°C initial temperature and ten seconds idle. Five warmups
preceded eleven alternating AB/BA pairs, with three fresh executions per sample.
The candidate was rejected and removed, including its exports and test wrapper.
Raw source, full parity/reference errors, samples, build command, binary hashes
and telemetry are archived privately. No full-model run was warranted.
This experiment does not distinguish the cost of M16 padding from the cost of
empty metadata windows, so it establishes no gain for a different capacity.

A single capacity-only retry retained all 288 expert threads and changed the
window allocation from 289 to 25. For each positive count c and WIN≥1,
ceil(c/WIN)≤c; summing across experts bounds the live windows by the 24 slots.
The extra entry remains zero. The actual cases had 23/17/15 live windows.
All start/live-count prefixes and inverse indices matched the original table,
and all 36,864 BF16 chain output bits matched the 289-entry NAX arm. The NAX
math, original stored banks and implicit padding were unchanged.

| Layer | Group2 control, µs | NAX 289, µs | NAX 25, µs | NAX 25 vs control | Wins vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 667.250 | 811.847 | 758.027 | +13.60% | 2/11 |
| 20 | 596.791 | 733.708 | 709.180 | +18.83% | 1/11 |
| 34 | 602.639 | 678.819 | 643.528 | +6.78% | 2/11 |

This quiet retry used the same runtime and replay, base `b0d5644e` plus the
isolated capacity variant, maximum fans, a 50.42°C initial temperature and ten
seconds idle. Five warmups preceded eleven alternating ABC/CBA rounds, with
three fresh executions per sample. The smaller capacity reduced NAX latency
by 3.34–6.63%, but all three medians still lost to the current chain. Both
variants, the research helpers and wrapper were archived and removed.
No further NAX padding experiment or model-level run was warranted.

## Kernels

- **Prefill**: run-aligned 32-row windows, K-generic cooperative readers, the NAX 16x32x16 GEMM body with a K4 fast
  branch; ONE GEMM config reused across window counts (a per-row-count JIT compiled per novel prompt length).
- **At the two served geometries (`mimoPrefillOn`: MiMo, Flash-Next) the routing never leaves the GPU**: one
  threadgroup (a thread per expert, at most 512) builds the window table and the inverse sort order, and the finish
  reduce reads the sorted down plane through that inverse, bytes equal to un-sorting it first. Any other geometry
  builds the table on the host (a sync) and un-sorts with a copy ([perf-baselines](perf-baselines.md#exl3-gpu-routing-meta)).
- **The NAX body's x loads carry no bounds branch**: a lane's row pointers are clamped into the input once per run
  (a padded row reads a live neighbour whose product is never stored), the k loop is unswitched on the window's
  second 16-row block and unrolled by two. Every row's products are the branch-guarded body's, so its bytes are
  `GEMM_NAX_REFERENCE_SOURCE`'s at every admitted rate (the test's reference); -25% per GEMM at MiMo geometry
  ([perf-baselines](perf-baselines.md#mimo-prefill-nax-body)).
- **Prefill off NAX** (M1–M4, or NAX declined): the 8x8 `simdgroup_matrix` body computes D = W^T X^T so each lane's
  eight-weight slot group lands straight in its A fragments (the tile layout is the MMA fragment layout). f16 x and
  128-multiple widths only; anything else takes the scalar body. Not byte-identical to the scalar body (sum order).
- The funnel readers' codewords and decoded weights equal the host tile decode's at every n, so the GEMM
  accumulation order and output bytes are unchanged (n32..36 read a group through one funnel, wider rates two).
- **Its block count is compile-time** (`(WIN+7)/8`), never the run's: a data-dependent bound over the
  `simdgroup_matrix` arrays spilled them, 2.6x slower. Short runs pay the padding and still win
  ([perf-baselines](perf-baselines.md#m2max-64gb)).
- **Decode**: four dispatches per MoE layer — pair prepare, split-K pair GEMV with f32 inner planes, fused mid+down
  GEMV, f32 finish reduce (`moeSwigluFused`; top-k ≤ 32, named refusal above). On MiMo geometry the SwiGLU mid is
  prepared once per (row, expert) (`preparedMidOn`, disabled for Qwen; like `mimoPrefillOn` it keys on geometry,
  never on the rate), and at two or more rows the pair input too, in its own dispatch (`pairPrepare`): five
  dispatches. A pair threadgroup would otherwise re-derive its K span of x (64 times per slot); one row keeps that
  fused prepare, where the extra dispatch costs more than it saves. Rows ≤ `DECODE_ROWS_MAX` or verify rows take this
  chain; wider takes `moePrefill`. The MTP head's MoE rows ride the decode chain and refuse wider
  (`Exl3MtpRowsExceedDecode`).
- **The prepared pair input is stored in GEMV lane order** (`LANE_ORDER_SLOT`: each 16-row tile keeps rows 2q,
  2q+1, 2q+8, 2q+9 at 4q..4q+3), so a lane reads its four as one half4 (`laneQuadReads`). Same values in the same
  order: the pair planes keep the self-preparing kernel's bytes. The same layout for the prepared middle measured no
  gain on the down (mid + down 52.1 vs 51.7 us at 1 row, 145.1 vs 145.9 at 4) and is not taken.
- **Streamed serial decode** can add its already gated shared expert in the finish reduce, removing a dependent
  elementwise dispatch. The routed sum is rounded to its output dtype before the shared addition, matching the
  separate store and add bit for bit. Other widths, mixed gate/up rates and dtype mismatches retain the separate add.
- **The decode GEMVs are bound by fixed per-tile work, not DRAM** (64-bit index math, two word loads and a 64-bit
  shift, four input reads, loop control). The lane-funnel arms (`gemvLayout`: every n below 64) carry two output
  tiles per threadgroup, load both k-tiles of an iteration before decoding, and bump pointers; the per-tile
  accumulation order is unchanged, so the bytes equal the one-tile generic reader's (`FUNNEL=0`, the test's
  reference). A layout that changes which simdgroup sums which k-tile (8 simdgroups) is NOT bit-identical.
- A rate on the generic reader decodes ~40% slower per GEMV than on the funnel, with no other symptom. The engagement
  line `[exl3] n<n> funnel engaged arm=<arm>` names the rate and arm in a live log.
- **MiMo verify rows share an expert's weight reads** (`PAIR_GEMV_GROUPED_SOURCE`, `DOWN_PREPARED_GROUPED_SOURCE`;
  prepared-mid geometry, 2+ rows): among an expert's slots, each even-ranked slot leads itself and the next one,
  decodes each weight once and feeds both members in the single-slot order, so every row's bytes are its one-row
  decode tick's. Two members only: four spill their accumulators. Not on Flash-Next, whose rows share too few experts.
- **A decode GEMV slot is bound by its own FMA and input path**, not the weight decode or DRAM, so deduplicating
  shared experts recovers only 4-7% of the expert kernels at 3-4 rows.
- Dead for the decode GEMVs (microbenched): 4 or 8 tiles per threadgroup, software prefetch, 2 simdgroups, a
  threadgroup LUT decode, a 24-bit multiply split, half2 input reads, bitfield extracts.
- Dead for the prefill GEMM on the NAX body (MiMo and Flash-Next, outputs bit-identical, all slower): 64-row windows
  with one decode feeding 4 MMAs (+7-18%), a threadgroup-shared double-buffered decode (+30%), decoding tile k+1
  before tile k's MMA (+27%), 256- or 64-thread groups (+12% at 2048 rows); on the branch-free body: a threadgroup
  LUT decode of the w12 codebook (+32%), a per-k-step threadgroup barrier (+9%), unroll 4 (+20% over unroll 2),
  64-row windows again (+9-13% on gate/up). The kernel is register/occupancy bound: added live state loses.
- **The SwiGLU chain is f32**: gate, up, sigmoid, SiLU and their product stay in f32 registers through the multiply
  by the down suh. In f16, MiMo's activations put gate and up near 400 each and the product past 65504, so a whole
  routed row became inf. The next ceiling is the f16 down inner plane (about 2x above the measured peak).
- **The shared-expert add must free the routed output it consumed**: it once retained 1920 MiB per 8192-token chunk
  (the 48k prefill cliff). Owned-copy hidden captures at the chunk boundary; kernel configs dropped on their error
  paths.
- **Levers**: `SUSHI_EXL3_GEMM_WIN`, `SUSHI_EXL3_WIN_ALIGN` (window geometry A/B); diagnostics
  `SUSHI_EXL3_LAYER_UBENCH`, `SUSHI_EXL3_UNION_HIST`, `SUSHI_EXL3_SWIGLU_MAXABS`, `SUSHI_EXL3_GEMM_ARMS` (served vs
  reference NAX body, interleaved, at MiMo geometry; `SUSHI_EXL3_GEMM_COUNTS` replays the per-layer `[exl3-counts]`
  lines `SUSHI_EXL3_UNION_HIST` logs on a prefill).

## Parity bars

- **Quality bar**: KLD vs the bf16 teacher (`sushi kld capture|compare`), never bytes against the affine pack.
  The EXL3 kernel arms are not byte-identical to any composite (they round once). MTP: EXL3 packs take the chip's
  generic depth row ([engine-mtp](engine-mtp.md#round-cost-table)).
- **A GEMM/GEMV parity bar is relative to the SUMMANDS, never the result** (`Exl3GemmParity`): a trellis dot product
  cancels orders below sum|w·x|, so a result-magnitude floor is seed-locked. Element ceiling = one f16 store + an
  f32 accumulation `in_dim` deep; whole-tensor RMS no worse than 3x mlx's own f16 matmul over the decoded weights
  (`measureInnerGemmParity`). A parity case sweeps `PARITY_SEEDS`, never one chosen seed.
- **Test every arm at REAL magnitudes against a TRUE f32 oracle** with a finiteness assert (outlier residual
  channels, products in the 1e4..1e5 range). A host oracle that mirrors the kernel's own f16 stores cannot see a
  saturation. When a pack "mostly works", capture per-layer max|x|, max|gate*up|, max|down inner| on a real prompt
  first.
- **Score the Metal arms on a real pack's own bytes** (real suh vectors span four decades), not only synthetic
  trellises. The w12 fixture (`exl3_k2p5_mcg_w12_linear.safetensors`) certifies the window convention against the
  converter's own decode.
- A "systematically wrong but not garbage" pack whose reference decoders agree points at live-path numerics OR at
  the pack's own weights along real activations (a converter-side defect; converter details live in the private
  repo), not the bit layout.

### GLM 2K grid transpose: exact component win

The isolated `glm_prefill_grid` helper transposes physical threadgroup axes for
original sorted WIN32 NAX projections: physical X visits routing windows and
physical Y visits 128-column output stripes. Logical window/output IDs, dot
body, accumulation order, F16 stores and metadata remain unchanged. It uses
no new dispatch, weight copy or precision conversion. The candidate admits only
BF16 B1/T2048/H4096/I2048, top-eight, E288 and n36/MCG/W12. The component
qualification used an isolated call.

An opt-in normal-FFN capture records the first actual L20 T2048 input, indices
and scores as BF16/U32/F32 arrays, totaling 16.125 MiB plus headers. The capture
forces evaluation and is not a throughput run. The probe loads the original nine
L20 bank tensors lazily from the three checkpoint shards, preserving all 288
experts' physical spacing. It does not compact banks or scale older routes.

This actual fixture contained 16384 assignments, 265 active experts, 671 live
WIN32 windows and 1174 M16 tiles; the maximum expert received 571 rows. Capacity
remained 800, with 12800/12800/25600 dispatched gate/up/down threadgroups per
layer in both arms. All 8388608 final BF16 output values matched the unchanged
native routed chain bit for bit. Six focused tests passed.

Three warmup pairs preceded eleven alternating AB/BA pairs. Timing included
sorting, metadata/inverse, preparation, all three GEMMs, middle transform,
finish, allocation, endpoint evaluation and frees on fresh graphs. Original
full expert banks remained resident; input, routes and scores were actual
captured arrays.

| Whole L20 routed chain | Median ms | Change | Paired wins |
|---|---:|---:|---:|
| Native grid | 19.640459 | Reference | — |
| Transposed grid | 17.957667 | −8.568% | 11/11 |

This qualifies one inclusive component comparison, not full-model throughput
or a default change. Improved cache locality is a hypothesis: Metal execution
order and physical traffic were not traced. The arithmetic and issued tile count
are unchanged.

The source baseline was `49192c0b` plus the isolated helper/probe and capture
seam. The ReleaseFast run used MLX 0.32.3, interactive `taskpolicy -a`, exclusive
lock `glm-prefill-grid-transpose`, confirmed maximum fans near 5346/5780 RPM,
54.68°C initial temperature and ten seconds idle. Private measurement key
`glm53-prefill-grid-transpose-20261003` retains the actual fixture, raw samples,
source/build/run commands and provenance. Binary SHA-256:
`e17997486ed69723e1f368dc8a7b6fdf2422e8bb994deb4ce9423ed80f67dc7d`.


`SUSHI_GLM_PREFILL_GRID_TRANSPOSE=1` now exposes a default-off normal-FFN
qualification hook with the same strict geometry, clamp ten and MCG/W12 guards.
Unsupported shapes or a declined native NAX capability/probe use the existing
routed path. Both native and candidate kernel probes run before preparing a
candidate graph. `bind(bool)` provides scoped control, and `dispatchCount()` /
`resetDispatchCount()` count successful qualified whole-MoE constructions.
One count represents three transposed GEMMs. Guard/binding tests and the actual
fixture probe assert that unsupported calls leave this counter unchanged.
The chain's arrays and bounds are the original chain's, so no new transient
buffer or resident weight bill is required. This source integration still
requires the combined full-model qualification; the option remains off.
