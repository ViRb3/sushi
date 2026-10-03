# Near-full GLM prefill eligibility research

Read-only researcher 1 report at `c7e648a2`, 2026-10-03. Goal: 1500 prefill /
60 DFlash2 decode tokens/s with stable approximately 2K–32K operation. The last
implementation round accepted no runtime change: pool-score cadence was exact
but only 0.45% faster in a drifting real-model ABBA; canonical KDA and shared
scoring did not have repeatable component wins. Do not repeat those candidates.
This report recommends two existing-path eligibility extensions, with no new
shader, precision restoration, padding, model capture or source implementation.

## What the real 2K row actually executes

The completed predictable 2K request in `glm53-prefill-wave-llmprobe-20261003`
had **2037 input IDs**, not 2048. It recorded 2186.30 ms prefill, 931.71 tok/s,
192 output IDs / 191 decode forwards and 47.08 decode tok/s. Its logged retained
cluster, routed-grid and packed-cadence counts were all zero. The HTTP metadata
reported the A6 expansion flag enabled, but that bridge does not currently reset
or serialize the A6 expansion counter. Zero A6 calls at 2037 is therefore a
source-derived eligibility fact, not an observed HTTP counter.

The separate exact-T2048 synchronized component diagnostic recorded 136 A6
expansion calls, 34 retained clusters and 1788.124 ms total prefill. Its disjoint
routed FFN and KDA-attention totals were 810.624 and 500.232 ms. This profile
forces evaluation and changes scheduling; **1145.3 profile tok/s versus 931.7
HTTP tok/s is not a measured guard-extension speedup**. Both length and execution
method differ. The KDA subtotal includes projections, prework, recurrence, gate
and output; it is not all recurrence.

`Mla.densePrefillEligible` already accepts any BF16 row count above eight with
`offset + rows <= 2051`, at the qualified D256/D256 head geometry. Thus cold
2037, 2011 and 1938 inputs already use the dense MLA SDPA branch. The 2051
cutoff represents complete top-512 pools plus the raw tail; widening it would
change attention semantics once another completed pool can be excluded. Leave
that guard unchanged. The eligibility problem is elsewhere.

## Candidate A: existing A6 expansion at 1536–2048 rows

**Recommend worker 1.** `glm5_a6_dense_once.geometry` currently requires
B1/T2048 and either K4096→N8192 or K8192→N4096. Extend only its row admission
to **1536 through 2048 inclusive**. Keep its original U32 A6/group128 weights,
BF16 scales/biases and BF16 activation guard. Do not admit other bank shapes,
smaller rows, decode, multi-batch input or another quantization format.

The implementation remains `Ops.dequant` to a fresh BF16 weight bank followed
by native dense matmul. Both operations already accept dynamic row counts.
No new kernel, coefficient formula, permanent decoded weight or padding is needed.
For cold 2037 this would enable the same four KDA banks across 34 layers,
**136 dispatches**, while retained projection clustering remains unchanged at
T2048 only. Original small BF16/FP32 tensors and resident embeddings remain intact.

**Native dispatch and partial rows:** throughout this band, both wide shapes
fail the native NAX split-K condition `K >= 3*max(M,N)`. They use regular NAX;
on the M5 `d` branch N exceeds M, selecting BM128/BN64/BK512 and WM4/WN2.
At M2037/2011/1938 there are sixteen M tiles, as at M2048. The existing
`align_M=false`, `sgp_sm=min(SM,M-row)` and safe load/store branches handle
partial final rows. At M1536 there are twelve complete M tiles. These are
source eligibility conclusions; every valid output at the partial edge still
requires numerical proof. Keep original FP32 accumulators and BF16 stores.

**Memory:** each decoded bank is still 64 MiB. Four banks may remain alive in
one layer's Ops scope, so the premium is 256 MiB per pending KDA layer and
512 MiB at async2. HTTP uses configured chunk2048 and already reserves this
premium even when a shorter actual input does not engage the hook. A configured
chunk1536–2047 currently receives zero expansion premium and must receive the
same bound after admission is broadened. Normal diagnostics also allow chunk
sizes above 2048: they can produce an eligible final remainder. Budget by the
maximum possible call—when enabled, configured max chunk at least1536 can
produce an eligible call—even though actual helper eligibility stays at most2048.
Do not underbill those remainders or bill newly decoded persistent banks.

**Measured prior and realistic ceiling:** the qualified T2048 primitive measured
2.732989→2.530552 ms QKV and 2.863562→2.712552 ms output, with expansion,
allocation, construction, evaluation and frees included. Its roughly 25.8 ms
projection across 34 KDA layers is a prioritization estimate, not a 2037 result.
The earlier native T2048 whole-model qualification improved 1193.70→1219.30
prefill tok/s, +2.14%, preserving 64 IDs. A similarly modest gain is plausible
at near-full rows; there is no basis for assigning the entire profile/HTTP gap
to the guard or promising 1500. Full 2048 chunks already use this path, so long
prompts benefit only when a remainder lies inside the band.

**Ownership:** worker owns `src/glm5_a6_dense_once.zig`, its row/budget guard tests
and one isolated near-full probe. Coordinator owns HTTP counter reset/metadata
and admission integration. Keep `glm5_kda_prefill_cluster.zig`, MLA and routed-grid
behavior fixed for this candidate's measurements. Existing A6 opt-in remains the
qualification mechanism; scoped binding may support an in-process comparison.

**Focused proof:** one actual layer-zero QKV/output bank pair, inclusive native
versus expansion/native GEMM at representative M2037, with boundary checks at
1536/2048 and rejection at1535/2049. Check complete output bits, finite values,
actual dispatch and the configured-chunk remainder bill. Ordinary native BF16
NAX compound rounding is owner-permitted if a real difference occurs; record
raw difference/quality evidence and keep it opt-in, without restoration math.

**One real-model gate:** load the target once; use the same exact 2037 input IDs
and fresh Requests for current versus candidate in **ABBA** with profiling off,
foreground QoS and an exclusive quiet GPU. Include prefill endpoint evaluation
and cleanup in timing. Preserve one reference for all logits and complete cache
state, plus the same 64-token continuation; record raw drift if not exact.
Confirm 136 A6 calls in each candidate arm and zero in control. The control at
2037 is the currently recorded native path, so an A6 binding-off control does
not disable any already-engaged T2048 call. Keep other flags identical. This is
one loaded-model job, not a repeated ladder. Only a real repeatable gain proceeds
with its precision/state proof; root owns eventual serving/HTTP qualification.

## Candidate B: existing routed-grid transpose at 1536–2048 rows

**Recommend worker 2.** This reaches the larger cold routed lead without a new
GEMM body. Keep B1/H4096/I2048, top-eight, E288, packed n36/MCG/W12, WIN32
aligned expert windows and clamp ten. Admit only T1536–2048. Source equality,
accumulator order, F16 intermediate stores, original slot ordering and final
BF16 output remain the current accepted grid-transpose algorithm.

This is **not merely a guard edit**. `glm_prefill_grid.zig` also hardcodes
2048 and16384 in reshape, preparation, metadata, middle, finish and result
shapes; its projection configs currently emit16384 output rows. Replace those
with validated T and **S=8*T**, consistently through the complete chain.
Do not keep a padded16384 output or fabricate token rows. At T2037, S16296;
at2011, S16088; at1938, S15504. All shapes and inverse routing must use actual S.

**Native window capacity and cache:** keep the original bound
`ceil(S/32)+288`: T2048→800, T2037→798, T2011→791 and T1938→773.
It includes empty windows as before; live windows retain native starts/counts.
Gate/up/down physical axes are the existing transpose; dispatched groups derive
from actual capacity and 128-wide output stripes. Keep only the existing two
projection config geometries, gate/up4096→2048 and down2048→4096. When S changes,
free/rebuild that geometry's config and retain at most two entries, not a cache
entry for every prompt length. Reuse a config only when its output S matches.

The API ownership contract supports replacement after apply: `mlx/c/fast.cpp`
uses `auto config_ctx = config_get(config)`, copying the C++ configuration; the
kernel function constructs owned output shapes and a `CustomKernel` whose grid/
threadgroup tuples are stored by value. The pending primitive does not borrow
the C config. A behavioral test must still construct two different-S lazy graphs,
replace the cached config before evaluating the first, then evaluate both and
compare their complete shapes/bytes. The native metadata cache is already bounded
and should not be redesigned.

**Memory and ceiling:** the candidate uses the same S-sized sorted/prepared/F16
planes, metadata and inverse as the current native chain. It adds no weight copy,
activation plane, dispatch or resident memory. Existing original-path admission
therefore suffices; smaller S reduces those original planes. The actual T2048 L20
fixture's inclusive chain measured19.640→17.958 ms, 8.57% faster with11/11 wins.
Scaling that percentage to the old810.624 ms routed subtotal suggests about69 ms,
approximately3.9% of its synchronized prompt, before interactions. This is a
loose projection and not a measured 2037 gain; a cold prompt's expert distribution
can differ substantially from the saved later-prefix L20 fixture.

**Ownership:** worker owns `src/exl3/glm_prefill_grid.zig` and a unique near-full
probe/capture adapter. Leave the existing native expert kernels/decoder intact.
Coordinator owns any capture metadata changes and real-model integration. There
is no shared source ownership with candidate A.

**Focused proof:** slice the existing actual T2048 L20 fixture to2037 rows and
slice its indices/scores in lockstep, labeling it as a sliced fixture. Compare
all complete routed outputs against the current native chain, including sorting,
window/inverse, preparation, gate/up/down, activation, finish and frees. Verify
capacity798, no padded rows and cache replacement across2037/2011/1938/2048.
Three warmups and eleven paired fresh whole-chain samples test one candidate.
This fixture proof does not establish cold2037 throughput.

**One real-model gate:** after candidate A's disposition, load the selected stack
once and run a fixed exact2037 cold prompt/current-versus-near-full-grid ABBA with
fresh Requests, profiling off and identical A6 settings in both arms. Confirm
42 grid calls in candidate and zero in control, with complete logits/cache-state
and 64-token continuation equality. Optionally capture actual cold L20 input,
indices and scores during this same job for its artifact; do not add another
model load. Only a repeatable actual-model gain is accepted or pushed. Do not
attribute the combined A6/grid difference to this grid arm alone.

## Shared/dense MLP census: deferred, not a third candidate

Checkpoint headers and config identify the excluded families:

| Family | Count | Gate/up K→N; down K→N | BF16 decoded bank each |
|---|---:|---|---:|
| Shared expert | 42 MLPs | 4096→2048;2048→4096 | 16 MiB |
| First dense FFN | 3 MLPs | 4096→12288;12288→4096 | 96 MiB |

The existing A6 helper accepts neither family. Shared-only expansion would retain
three16 MiB banks after attention, so a KDA-plus-shared layer's premium would be
4×64+3×16=304 MiB, **608 MiB at async2**, rather than current512 MiB. Its old
synchronized shared subtotal is91.919 ms, only5.1% of prompt time, and no current
inclusive actual shared-bank dequant/dense win exists. Do not silently include
shared geometry under the old four-bank bill.

Dense gate/up require96 MiB each, and dense down K12288 meets the native NAX
split-K condition at N4096: three4096-K partitions, another96 MiB FP32 partial
plane at T2048 and a changed reduction path. This is neither the current regular
NAX KDA qualification nor the shared16 MiB case. Its old36.411 ms total is small.
Defer these geometry expansions and the previously unmeasured dense-swizzle4 port.

## Concurrency and acceptance

The diagnostic bridge still deliberately serves one Request at a time with one
MLX-owning thread; MLA admits B1. Eligibility extensions shorten queued service
without claiming independent-request batching or parallel throughput. They add
no new model copy and preserve BF16 compressed MLA and FP32 KDA persistent state.
Keep actual token counts visible; no input truncation or padding is part of either
candidate. Researcher2 supplies the third, decode/speculative implementer.
Only accepted measured changes enter the real combined performance evaluation and
push. A noisy component or real-model arm is archived without another variant sweep.
