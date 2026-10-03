# Exact-retrieval full-history prefill research

Research only at accepted runtime `e1597cc2`, 2026-10-03. One constrained
experiment is plausible: original per-query selected IDs, one dense bool
membership plane and one **head-batched native T2048/N16384 SDPA**. Speedup is
unmeasured and a loss is plausible. No source, prototype, build or GPU run.

## Existing evidence and distinction

The shared group4 bank won its component53.27% but failed both fixed actual-
model quality inputs: code mean/max KL0.512904/23.547608 and prose
0.028818/2.180192. It is restored, with no bank/group sweep. Preserve independent
retrieval100%; do not reuse the rejected anchor approximation.

The existing `glm5_attention_nax_mask` already uses this head-batched layout,
original IDs and chronological full-history membership. Its2MiB mask cap
limits16K to127 real queries. Recorded synthetic31-query/16K attention took
3.518ms versus packed1.852ms; at16 queries it took2.785 versus0.687ms.
Thus this is a **single T2048 dispatch/layout experiment**, not a new native
algorithm or evidence that full-history arithmetic beats gathered attention.
The older tests did not include the newly demonstrated masked-nonfinite safety
failure or the proposed all-history finite guard.

The actual T2048/16K gathered control is130.207ms inclusive; the earlier cadence
artifact was129.888ms. Exact indexed original-cache loads lost85.15%.
No existing profile isolates gather/command savings. The cold synchronized
profile still attributes45.3% to routed FFN,28.0% to KDA and8.3% to dense MLA.
This sparse candidate does not remove those cold/routed costs. The entire
130ms late attention call is its cost ceiling, not a removable budget or a
forecast of1500tok/s. Eleven MLA layers give at most about1.43s for one late
T2048 block if attention were free; it will not be free.

## Verified pinned-backend layout

Use BF16 Q `[1,64,2048,512]`, a view of original `[2048,64,512]` with head
stride512, query stride32768 and final stride1. KV is one immutable valid-history
view `[1,1,16384,512]`, aliased for K/V. In MLX0.32.3 full mode,
`ScaledDotProductAttention::eval_gpu` copies Q/K/V only if their final stride
is not1. These admitted views avoid query/head/history copies. Native output
uses physical query-major storage, so the inverse head/query transpose can
return `[2048,64,512]` as a view.

The API accepts GQA64; `attention_nax_dsplit` selects
`kv_head_idx=tid.y/64`, sharing that same KV bank across all heads. Bool
membership `[1,1,2048,16384]` broadcasts to64 heads with head stride0.
Full-mode mask copy also checks only final stride1, so this remains a physical
32MiB plane, not a2GiB head-expanded allocation. Do not call contiguous on
the head-broadcast mask. Use array mode, scale1/16 and `force_fused=true`:
the normal D512 default declines array masks even at large Q.

Native D512 uses BQ32/BK32, WM2/WN4 and FP32 accumulators. Grid is
`(64,64,1)`:4096 tensor groups, the same total as128 current B16/Q64 calls.
Full N16384 needs512 K blocks versus65 for gathered K2051: **7.88× K-loop
work**, with no array-mask skip of empty tiles. At32K it would be15.75×;
do not extend this trial to32K. A single dispatch/shared contiguous history
may reduce command and gather/cache costs, but does not reduce dot work or
tensor-group count. Existing small-query losses are an adverse signal.

## One bounded component

One worker owns an isolated BF16-only helper/probe; coordinator owns any later
delegation. Strict T2048/N16384/H64/D512, immutable cache strides `[512,1]`,
existing original selector/NAX flags and no new retrieval policy. Reuse the
actual16K captured planes. Construct original IDs using the same128 T16 selector
calls. Keep every valid ID/causal set, including unpooled tails. Membership
uses zero initialization and scatter to a separate dummy column for invalid
IDs, as the existing prototype does; invalid writes cannot clear valid key0.
Prove valid-ID uniqueness, since collapsing duplicate keys changes weighting.

Ordinary bool masking is insufficient: shared-native proof showed future
NaN/Inf V poisoning earlier outputs. Before building the full-history graph,
scan **all valid16384×512 cache values**, then settle one finite predicate.
If any value is nonfinite, use the original safe gathered path. Never inspect
uninitialized capacity or zero/sanitize valid historical values. This16MiB
read plus bounded8MiB finite predicate and host wait must be included in timing.
No reuse of a stale finite result after append/mutation. A full-call test must
show actual fallback/unchanged output for masked-future and unselected
NaN/Inf, with valid nonfinite behavior still observable. Empty membership
requires the existing explicit positive-zero output guard.

Memory: about16.02MiB IDs,32MiB physical membership, bounded predicates/safe
indices/dummy-column scatter storage,8MiB finite plane and128MiB native output;
empty-row finalization may allocate another128MiB output. Reserve a conservative
**512MiB per pending MLA layer** for this experiment, without assuming donation
or a free finalization. No F32 cache, floating Q×H×N scores, full-history copy
or64-head mask materialization. Measure actual peak before setting any runtime
bill; fixed layout/count/stride observations must confirm those copy claims.

Prove every selected ID and dense membership bit against original retrieval,
causal/invalid/key0/tail handling, Q/head order and nonfinite fallback. Native
chronological K order and M32 per-head query tiling differ from the current
argpartition-order/fake-head M32 body. Report BF16 mismatch counts, relative L2,
maximum error and nonfinite counts; ordinary BF16/FP32 NAX rounding may be
investigated under owner policy, without restoration or claiming bit parity.

Then one inclusive3-warmup/11-pair actual T2048 comparison against current
two-tile packed attention includes finite scan/wait, unchanged selection,
membership construction, native SDPA, empty handling, endpoint evaluation and
all frees. Stop a noisy/losing arm. No chunk/layout/precision sweep.

A winner needs the declared nonrepeated16K code/prose long-prefix forced-logit
screen, including late-prefill samples and192 baseline-forced continuation
rows; retain the fixed meanKL≤0.01/maxKL≤0.15/top1≥95%/NLL+≤0.02/zero-new-
nonfinite bounds. Exact retrieval is not proof of acceptable rounding drift.
Then one actual-model prefill ABBA holds equal reference memory and matches
flags/admission. Strict serial/spec token and complete valid BF16 MLA/FP32 KDA
state equality must start from the **same full-history-prefill snapshot**;
baseline-prefill state equality is a separate drift question. Decoder/selector
policy stays unchanged. Acceptance precedes any opt-in integration/commit.
