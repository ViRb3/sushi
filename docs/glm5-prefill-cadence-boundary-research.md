# Pair only routed experts; preserve the original attention cadence

Recommend one bounded scheduling experiment: a layer-major pair of original
2048-row halves, joining **only routed expert work** into one 4096-row top8
chain. This differs materially from rejected steady4096, which widened every
layer operation and appended all 4096 attention rows before selecting keys.
It is not an automatic retry or a scalar-scoring override. Accepted runtime
remains `4fcb541e`, qualified CLI `102cb8d5`; no implementation accompanies this
source report.

## The history boundary is real; the cause of KL is unmeasured

`Mla.applyMode` chooses its dense predicate using the pre-append state, then
appends the entire input and calls `attention.attend`. `indexScores` supplies
`State.processed/4` to `tryScores`, which requires at least 3584 pools. The
pool count also determines scalar-score output and argpartition widths.

| Query offsets | Original appended history/pools | Rejected wide history/pools |
| --- | --- | --- |
| 10240–12287 | 12288 / 3072: scalar | 14336 / 3584: NAX |
| 12288–14335 | 14336 / 3584: NAX | 14336 / 3584: NAX |

The [failed numerical screen](glm5-steady4096-quality-result.md) recorded
2816→4224 NAX calls. This source boundary explains engagement; it does not
establish the cause of measured KL. Merely slicing queries after a wide append
does not preserve the original path: history, dense admission and partition
width already changed. Masking future pools alone is insufficient, including
for equal-score partition ordering. Do not change the 3584/8192 eligibility
bounds, disable NAX, restore precision or relax quality rules.

## One fixed work boundary

Keep the cold 2048 forward unchanged. For each subsequent complete pair,
process two 2048 halves at each layer before moving both to the next layer:

1. Construct embedding, HC, normalization, KDA/MLA projections and attention
   independently at the original 2048 shapes. At an MLA layer, append/select/
   attend the first half before appending the second, with each half's original
   offset, `processed`, pool count, cold predicate, argpartition width and
   packed32/two-graph policy. Preserve KDA recurrence and convolution order.
2. Keep router/top8 selection and shared/dense FFNs at their original 2048
   shapes. Concatenate only the two routed inputs, IDs and FP32 scores for one
   current WIN32/MCG/W12/n36 expert chain. Inverse routing and weighted finish
   retain original slot order. Split the BF16 result before the original
   per-half shared add and HC expansion.
3. Retain both next residuals, then advance the layer. Never send 4096 into
   dense, HC, router, A6, MLA or clustered projection kernels. Final unpaired
   chunks and decode use the original forward. No streaming/profile variant.

These operations are causally separable across layers; the source's state is
per layer. Nevertheless, a scheduler must retain intermediate state handles
correctly and preserve failure/offset semantics. Reordering graph construction
does not itself prove numerical parity or improve overlap. Keep stored small
BF16/F32 tensors, BF16 compressed MLA caches and FP32 KDA state/accumulators.

`forwardLast` currently interleaves attention/FFN and owns async2 settlement,
cache side-output evaluation and Capture publication. Root must own a bounded
paired-layer seam and characterize it before extraction. DFlash Capture must
retain both halves' original per-token HC reductions in input order. A paired
prefill transaction must call assistant `appendContext` twice at the original
2048 shapes/offsets, settle both, and publish target/context only on success;
one 4096 assistant append would introduce another numerical change. Keep the
last-row head calculation and pending prediction from the second half.

## Honest cost and conservative lifetime bill

The prior replicated L20 expert chain was 8.36% faster with exact stages,
11/11 pairs. That is suggestive for expert-only joining, not a normal model
share or a promised gain on real adjacent halves. This proposal retains the
original counts of every other projection and attention call. Extra joins,
ownership and waits can erase its benefit. Charge all settlement; no overlap
or 1500-token/s forecast follows dispatch/window counts.

Assume both halves' graphs remain live at two pending layers. Each half can
retain four 64 MiB A6 expansions: charge 1 GiB total, rather than the old
512 MiB. MLA permutation allowance becomes 1610612736 bytes, clustered output
allowance 5242880; original resident cluster banks remain shared/unchanged.
Do not infer that evaluating a view releases these graphs.

Retaining every original 32K reserve, the prospective increments are:

| Increment over original async2 reserve | Bytes |
| --- | ---: |
| Two halves' generic activation allowance | 3221225472 |
| Additional half-specific A6 expansions | 536870912 |
| Additional MLA permutations | 805306368 |
| Additional clustered outputs | 2621440 |
| Fixed wider route/config/sort allowance | 4194304 |
| Explicit routed input/output joins and ID/score joins, pending2 | 134742016 |

This yields 15837298688 reserved bytes above the recorded 94548731128 active
baseline: 110386029816 total, 5062695688 below the fixed 115448725504 limit.
It is prospective accounting, not executed admission or a measured peak.
Proof/model reference owners and DFlash retained feature joins are additional;
the latter cost two BF16 `[4096,4096]` capture owners per tapped layer if both
per-half originals and joined storage coexist. Root must bill the actual tap
count and lifetimes, without reserve credits or a wired-limit increase.

## Bounded decision and ownership

Root owns the paired-layer/transaction/capture seams, admission and acceptance.
One implementation worker owns the expert join; another independently proves
per-half history/selector boundaries and Capture/assistant ordering. No new
reader, policy, width sweep or precision mode is proposed.

Require all complete paired-layer outputs, original route IDs/scores, expert
stages, convolution and FP32 states against two original forwards. Include
directed original MLA histories on both sides of the 3584-pool boundary and
the 8192-pool ceiling: exact scores, ordered selected IDs, attended BF16 values,
cache/tail and nonfinite/causal behavior. New source scheduling must pass these
checks; do not reuse the rejected whole-wide model proof as a pass.

Only then run one three-warm/eleven-pair complete layer gate, charging original
2048 projections/shared work, joins, all endpoints/side-state waits and frees,
with equal references and actual window counts. Stop mismatch, noise or loss.
A clear winner receives the same frozen nonrepeated 16K actual-model exact
prefix/valid-state/native64 gate before any ABBA. Any mismatch stops this exact
proposal; no numerical guard toggle or threshold retry follows. Capture and
assistant-context parity must also pass before matched full-request timing.
