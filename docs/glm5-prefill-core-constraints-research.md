# Current prefill core: no supported new kernel yet

Do not assign another prefill kernel implementation from the present evidence.
The L20 attribution narrows the observed staged work to the three GEMMs, but
cannot identify a removable reader/decode bottleneck. No consequential fresh
mechanism survives the source, loser and memory constraints below. Runtime stays
`4fcb541e`; this is source-only research after `5929c9d6`, with no new measurement,
prototype, build or GPU job. The 1500/60 goals remain unmet.

## What the current GEMM already reuses

`glm_prefill_grid` uses the accepted physical window/output-axis transposition,
original aligned WIN32 metadata and native MPP 16×32×16 operation. Four SIMD groups
own disjoint 32-column output stripes inside a 128-column threadgroup. In each K16
step, `nax_wfrag_k<36>` extracts the two adjacent output tiles once for that
SIMD, forms the right cooperative tensor, and feeds the original first and
second M16 accumulators. There is no duplicate local RHS fragment between those
two row blocks or the four disjoint output stripes to eliminate.

The n36 funnel reader uses rate-derived codeword ends/shifts and the stored W12
mask. MCG decodes each pair with fixed integer operations and original rounded
F16 arithmetic; it does not fetch a codebook tensor. K order, descriptor geometry,
FP32 accumulation and F16 stores matter to exactness. Replacing any of them is
a separate numerical mode, not an obviously redundant load removal.

Cross-window reuse of an expert's decoded RHS needs more live destinations,
shared inter-threadgroup storage, or a materialized decoded bank. These are not
free extensions of the current two-accumulator loop.

## Existing measured exclusions

- WIN64/four-MMA reuse, both original and branch-free bodies, was exact but
  slower. Threadgroup count/group changes and unroll 4 were also slower.
- Threadgroup-shared double buffering, next-K decode prefetch, LUT decode and
  per-K barriers all lost. Current GLM SIMD word sharing retained exact data
  but slowed projections about 83–85%; source read counts were not physical
  transaction evidence.
- Grouped middle/down and singleton splitting did not win the complete
  original routed replay. A4/mini2 cannot be a new prefill-kernel justification.
- Route6's 20.23% complete component win failed both fixed quality inputs.
  Another route count or threshold is not a supported retry. Cold absorbed MLA,
  BF16 retention and HC expansion also failed their model/cost gates.

A device/constant lookup instead of the failed threadgroup LUT would move
integer arithmetic into dependent memory reads. The failure does not prove
shared allocation was the cause, so changing address space alone has no supported
benefit ceiling. Do not invent occupancy/spill or cache-hit evidence from source
register counts or the available pipeline thread limits.

## Memory/layout boundaries

One 16×16 n36 tile occupies 72 packed bytes; its decoded F16 values occupy 512
bytes, a 7.111× expansion. A complete E288 projection contains 4,831,838,208 decoded
F16 bytes (4.5 GiB); all three projections contain 14,495,514,624 bytes (13.5 GiB),
compared with 2,038,431,744 packed bytes. Keeping original compressed decode
weights and all small tensors makes this additional storage, not a replacement.

At the recorded ordinary 33631+192 workload, resident 94,548,731,128 plus all
old request/growth reserves 11,132,338,176 leaves 9,767,656,200 bytes under actual
wired/admission 115,448,725,504. A full three-bank decoded layer already exceeds
that headroom; two pending layers are further beyond it. Even one decoded bank
per pending layer uses 9,663,676,416 bytes, leaving only 103,979,784 before any
new preparation/storage allowance, and requires proving a different restricted
lifetime. No larger wire limit, A4 default switch or old-reserve credit is allowed.
All 42 routed layers would add 567 GiB if retained together.

Rebuilding decoded banks one at a time avoids that full residency but charges
extra decode/store/read passes and possible waits on every chunk. Wider/padded
packed pitches similarly require a second immutable consumer representation,
separate logical rate from physical pitch, preparation/teardown and a measured
addressing benefit. Current 4-byte-aligned funnel reads and successful native
codeword parity provide no evidence that such repacking helps. Neither proposal
isolates decoder ALU cost; both change data traffic and working set substantially.

## Evidence limit and bounded decision

The fixed staged L20 gate/up and down clocks were 10.797 and 5.512833 ms, 90.3% of
the 18.061958 ms staged summary; normal complete chain was 17.352958 ms. The inserted
endpoints raised the complete clock 4.09%. Those groups include packed reads,
codeword extraction, MCG arithmetic, prepared input loads and native MMA together.
They do not split those costs or establish DRAM bandwidth, occupancy or spills.
The old 810.624 ms routed subtotal was forced evaluation, not current HTTP latency.
There is no honest source-derived percentage for a new reader kernel.

The bounded implementation decision for this round is **none**. Keep the current
reader/descriptor/stores and avoid another pointwise, route, window, group,
lookup or fusion sweep. Do not rerun the same endpoint/profile job or collect a
broad trace to fill missing counters. A future proposal must supply a materially
new mechanism with evidence separating its claimed work from native MMA/load
cost and a checked preparation/lifetime bill before consuming a GPU slot.

If such evidence appears, use one fixed actual L20 full-E288/top-eight proof,
all codewords/retained F16 intermediates and final BF16 output against the current
transposed source, followed by three warmups and eleven inclusive complete-chain
pairs with equal held references and all preparation/evaluation/frees. Stop
mismatch/noise/loss without a variant. Only a clear winner receives matched
long-prefill model ABBA and strict logits/valid cache/state/continuation, then
root's HTTP qualification. Any rounding change must be opt-in and first pass the
unchanged two nonrepeated 16K code/prose screen: 24 late plus 192 forced rows per
input, mean KL≤.01, max≤.15, top1≥95%, NLL increase≤.02, new NF=0. No smaller corpus,
threshold adjustment or precision restoration is permitted.

Ownership remains root for a later decision; no prefill runtime/helper seam is
requested now. [Expert history](engine-exl3-experts.md),
[wide attribution](glm5-wide-prefill-attribution-result.md),
[resource limits](glm5-routed-resource-result.md) and
[route policy rejection](glm5-route6-quality-result.md) contain the measurements.
