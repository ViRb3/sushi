# Fixed horizon2 proposal mini-head

Source-only during `4fcb541e` qualification. Recommend one opt-in **N2/horizon2
3-bit top32 shortlist with original A6 re-score**. This revisits a different
caller geometry from the old mini experiment; it does not replace the assistant,
target head or greedy verification. No prototype, build or GPU work is included.

## Why the existing switch is not the candidate

`glm5_dflash.proposeTreeWithChildren` now narrows ordinary N2 readout to hidden
positions 1–2 and supplies anchor plus those positions to the selector. However,
`readoutHorizonEnabled` explicitly excludes an assistant with a mini-head.
Existing `mini.project` therefore still reads seven coarse positions, and for
each top32 set re-scores all eight hidden rows to preserve native batch math.
Enabling that old switch would repeat the old workload. Its prefix512/64 model
result was only 0.31% within noise, with unchanged acceptance; it is not the new
acceptance or performance baseline. The baseline is **current full2 readout**.

## One bounded implementation

Keep the complete trained eight-row assistant forward, all stored A6g128
projections/small tensors, N2/children4 and current HC/native verification.
For eligible horizon2 only, use hidden `[1,2,4096]` at positions 1–2:

1. Project both rows through the existing separate affine3/gs64 coarse head.
2. Obtain two top32 sets; gather their original A6 weight/scale/bias rows.
3. For each set, evaluate **both draft rows** through that selected head, then
   take its corresponding row. Do not re-score M1 or join the two head banks.
4. Mask outside each set and keep original transforms, selector top16,
   conditional normalization, two-depth lattice and best-first tree.

The actual target head is U32 `[154880,768]` plus BF16 scale/bias
`[154880,32]`: A6/gs128, 495616000 bytes. The supported coarse format is
U32 `[154880,384]` plus BF16 grids `[154880,64]`: A3/gs64, **277544960 additional
resident bytes**. Target/assistant stored tensors remain unchanged; reuse the
existing consumer implementation, with no conversion recipe or precision
restoration. The actual A6 configuration uses positive default multiplier 1,
no softcap and selector top16; preserve existing unsupported-transform refusal.
Other policies/partial budgets retain their explicitly recorded fallback.

Pinned MLX dispatches affine M2 through `qmv_wide`: vecs2, eight K lanes and two
SIMD groups, thread group 32×2. Both full/coarse vocabulary calls have grid
1×19360×1; each selected32 head has 1×4×1. Full2 and selected-M2 share per-row
reduction geometry, but selected-logit bits still require proof. The source
packed footprint changes from 495616000 to 277544960 plus 204800 selected-row bytes;
that is logical address accounting, **not DRAM traffic or a bandwidth saving**.
Top32 passes, gathers, dense masks/scatters, transforms and selector work may
outweigh it. Do not claim the old full7/full8 saving against current full2.

Current matched drafting occupies about 10% of complete decode time. Even deleting
all drafting cannot close the 60 goal from that run, and this removes only part
of drafting. Acceptance/round counts can erase a small readout gain. There is no
GPU-family attribution or expected throughput percentage here.

## Proof, complete gate and ownership

One worker owns an isolated bounded helper/probe. Root owns proposal dispatch,
mode/counters, resident/preparation admission and model gates. Control must call
current full2 even while the same coarse object remains resident in both arms;
otherwise the comparison silently becomes old mini7 versus new mini2. Default
mini mode and caller behavior remain unchanged until acceptance.

Use real assistant block activations from fixed ordinary/predictable requests.
The old fixed-seed head fixture is synthetic and cannot establish real retention;
confirm an actual activation artifact or schedule one bounded capture of complete
eight-row blocks before testing. Never label random hidden vectors as real drafts.
Prove selected A6 logit bits against full2, head/source handle/dtype preservation,
ordered top32/ties, mask/nonfinite behavior and unsupported geometry. Report
original top1/top16 retention on those real rows; shortlist misses may alter unary
support, conditional probabilities, proposals and acceptance. No target-quality
claim follows retained-row equality alone.

One three-warmup/eleven-pair gate includes coarse projection, both top32 passes,
all gathers, same-M2 re-scoring, mask/scatter/concat, transforms, complete selector/
lattice/proposal construction, settlement and frees. Include all cold preparation
and release time separately, coarse residency, temporary peak and every existing
head/cache/scratch bill. No shortlist/bit/group/batch sweep or readout-only speed
claim substitutes for this complete operation. Stop loss/noise without retuning.

A clear winner gets current HC_ON/N2/A6/native full2-versus-bounded-mini ABBA,
192 outputs on both frozen ordinary and predictable inputs. Hold coarse, assistant
and exact references equally before all arms; include clones, all phases and
cleanup. Require exact serial IDs/valid target states, acceptance/round counts
and retention diagnostics, and total gain exceeding control drift on both inputs.
Worse acceptance must be reported and cannot be hidden by faster drafting.
Then existing-admission HTTP 2K–16K and selected 32K qualification decides adoption.
All cold/coarse retained-head bills remain; no reserve credit or default switch.
