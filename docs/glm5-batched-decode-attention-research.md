# True B3 native packed attention: next decode component

Source-only research at `c7e648a2`, 2026-10-03. Recommend one decode worker:
qualify true B3 packed native attention against the current complete three-node
scalar attention path. **No B3 speedup is measured.** N3/T4 policy work remains
the second documented candidate, parked for a later round. No source, build or
GPU work was performed during this audit.

## Verified native dispatch

MLX 0.32.3's `mlx/fast.cpp::scaled_dot_product_attention` requires rank-four
Q/K/V and equal batch dimensions. For Q `[B,1,64,512]`, KV
`[B,1,2051,512]` and a bool array mask, neither the API nor full-attention
backend flattens B into query rows. The mask broadcasts over Q64 with zero
query stride; K/V can alias one contiguous gathered bank.

In `mlx/backend/metal/scaled_dot_product_attention.cpp`,
`sdpa_full_self_attention_nax` selects BQ32/BK32/WM2/WN4 for D512. Its grid is
`(ceil(64/32),1,B)` and its threadgroup is `(32,2,4)`, or 256 threads:

| Shape | Native tensor groups |
|---|---:|
| B1/Q64/K2051/D512 | 2 |
| B3/Q64/K2051/D512 | 6 |

The kernel name/function constants depend on dtype, tile dimensions, alignment,
mask/causal/sink flags, not B. In
`kernels/steel/attn/kernels/steel_attention_nax.h::attention_nax_dsplit`, batch
only offsets Q/K/V/O and mask pointers using `tid.z`; accumulator types are
FP32. This supports a B1/B3 arithmetic-equivalence hypothesis, which still
requires every output bit to be compared. Group count 6 remains too low to
claim GPU saturation. Default D512 `use_fallback` explicitly requires at least
1024 query blocks and causal Q length at least 1024 to prefer its NAX route.
The proposed Q64 array-mask path therefore needs `force_fused=true`, as the
existing packed helper does.

The archived `glm53-decode-attention-nax-20261003` T1/T2/T3 probe was NOT a B3
experiment: `timed()` built and evaluated one B1 Ops scope per node. At 16K its
three-node scalar and native medians were 846.000 and 1347.104 µs, with 0/6 wins.
Its selected keys were common inputs and IndexPool selection was excluded.
That rejects three separately settled B1 native calls, not one B3 gather/SDPA.
The new control must mirror current layer cadence: build all three scalar
outputs and settle their output vector once, not reproduce artificial per-node
waits. Batching may reduce command/allocation/wait overhead; its six tensor
groups may still lose. No source-derived latency saving is assigned.

## One coupled native target mode

Keep original ordered selected IDs and IndexPool arithmetic. A single B3 gather
reads the immutable committed latent prefix plus each branch's ancestry from
the original three-row latent tape. It writes one BF16 `[3,2051,512]` bank and
bool `[3,2051]` mask. Use actual per-branch offsets/lengths and an ancestry map:
fork paths `[0]`, `[0,1]`, `[0,2]` must not use `offset+row`. Invalid/future IDs
produce zero without reading source data; valid key zero remains valid. K/V
alias the bank. Native SDPA sees `[3,1,64,512]` queries and array-mask mode,
with explicit scale 1/16. Zero all-invalid outputs after SDPA.

An opt-in mode must use matching native B1 math for normal one-token target
forwards and independent serial/replay oracle execution. B3-unavailable cases
within that mode fall back to three native B1 calls, never to scalar attention
for just verification. Device/dtype/model admission declines the WHOLE mode
when matching B1 cannot be supported. Dense/first-token and sparse-boundary
cases must use a consistent per-node rule; a batch-size threshold must not make
serial and verification different target models. The ordinary scalar mode stays
available and unchanged.

BF16 Q/K/V/operands/output with native FP32 score/output accumulators fit the
user's accepted ordinary NAX rounding policy. The new mode can differ from the
old scalar target: its measured component drift and later teacher-forced logits
must be reported, with no precision restoration. Speculative correctness is
tested against mode-matched B1 serial execution; numerical drift against the
old scalar target is a separate record, not a relaxed state-parity tolerance.

## Narrow source, memory and fallback plan

Worker owns one isolated batch helper and probe, reusing native SDPA and a
small overlay-aware gather. No new attention algorithm or shared-prefix scorer
is introduced. Before a component win, add no `mlaTree`, normal-serial or HTTP
production hook. Coordinator owns the later coupled mode/counter/admission
delegation. Preserve current packed-prefill and cadence code.

One gathered B3 KV bank is 6300672 bytes, about 6.01 MiB. The existing packed
helper's conservative three-row temporary formula is 7536321 bytes, about
7.19MiB, including query/results/copies, masks and bookkeeping. Plan an 8 MiB
per-pending-layer bound for the isolated coupled mode; async4 would reserve
32 MiB. Count retained arrays/actual peak in the probe and model gate. No
full-history latent copy, expanded floating score plane, weight copy or cache
donation is allowed. Existing three-branch scratch planning must admit the
batch plus its native temporary bound; otherwise use mode-matched B1 calls.

The native helper returns lazy outputs whose dependencies retain the gathered
bank until the enclosing settlement. Do not reuse a wrapper that evaluates
every B1 result immediately. Failure cleanup releases each owned bank/output
while preserving committed source arrays. Prefix/cache strides and masks need
explicit guards to prevent hidden full-cache copies.

## One component decision, then a model gate only for a winner

Reuse actual 16K capture planes: Q `[2048,64,512]`, index Q `[2048,32,128]`,
weights `[2048,32]`, pooled keys `[4096,128]` and latent cache. Label constructed
chain/fork ancestry and suffix arrangements as constructed fixtures, not actual
speculative-tree captures. Compute each branch's ordered selection using the
unchanged scalar selector, and materialize common input planes before timing.

1. Compare all B3 BF16 output bits with three B1 native calls on identical
   selections/views. Cover chain/fork, odd offsets, distinct suffix values,
   sole valid key zero, all-invalid rows and masked nonfinite source data.
2. Record relative L2/max error and finite outputs versus current scalar split8
   plus original merge, without precision restoration or tolerance changes.
3. Measure fresh complete three-branch attention construction, ancestry mapping,
   gather/mask, SDPA or scalar partial/merge, endpoint settlement and frees.
   Use three warmups and eleven inclusive alternating pairs. Selector work is
   common and remains unchanged; state explicitly whether it is outside timing.
4. Stop a noisy/losing component. A consistent winner receives the coordinator's
   coupled B1/B3 integration, mode-matched token/full-state oracle, old-target
   teacher-forced drift record and one real performance arm. No implementation
   commit or push precedes that real-performance acceptance.

## Parked candidate: fully optimized N3/T4 policy

Current N3 is not a fair optimized wider-policy comparison. Required gaps are:

- Bounded readout: generalize horizon2 to horizon3 while computing the same
  complete eight-row assistant block; lattice hidden retains anchor plus 3 rows.
- Overlay4: both `mlaTree`'s `parents.len<=3` and overlay tail/view guards need
  four ancestry rows, preserving immutable-prefix access.
- Exact MLA broadcast4: widen both query/value native M1-broadcast guards and
  derive batch dimensions; do not use changed-rounding head-M4 projection.
- A6 QKV hoist4: existing R-template/config supports four rows, but its guard
  admits only three; qualify T4 bits without broadening output projection.
- KDA retained leaf4: `recurrentLeaf` and `applyLayer` cap retention at three.
  Without this extension N3 pays replay that current N2 commonly avoids.
- Policy/admission: HTTP hard-codes N2, `+3` reservation/growth and rows3
  scratch. Derive four verification rows and clip actual proposal budget by
  remaining output/context space. Router/dense rows and group2 already admit T4.

Source arithmetic at the actual 33595+192 frontier gives reserved latent 33792
and pooled 8448 rows. The unchanged 256 MiB planner admits all four branches:
149.735 MiB for W4 versus 112.295 MiB for W3. At 64K it admits only three W4
branches; at 128K only one. These are CPU ledger results, not measured peaks.

Historical N2/N3-children4 verifier cost was 53.988/65.736 ms per round, a 21.76%
increase, while emitted tokens/round rose only 9.09% (64/24 to 64/22). The chain
policy did not improve acceptance. Those samples predate the optimized T4
guards above, so a complete policy deserves one later end-to-end comparison,
not a width sweep. Even ideal four-token acceptance at zero extra round cost
would scale current 42.39 tok/s only to roughly 56.5 tok/s; N3 alone cannot promise 60.

For this round choose the small, distinct B3 component worker alongside
researcher 1's two near-full prefill extensions. Keep N3 infrastructure, joint-QKV
launch savings and further selector/recurrence variants parked.
