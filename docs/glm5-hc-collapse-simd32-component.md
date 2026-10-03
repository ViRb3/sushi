# Fixed SIMD32 HC collapse component

The fixed complete eight-block gate passed exactly and won all eleven pairs.
Median operation latency fell 27.64%; the later [model gate](glm5-hc-collapse-model-result.md)
passed with 2.6475% complete decode gain and is accepted for HTTP qualification. The candidate
follows the [round plan](glm5-hc-a4-round-plan.md). One uniform subgroup computes
ordered FP32 coefficients and 20 Sinkhorn iterations; the original mixed loop,
256-thread group and BF16 mixed/FP32 post/comb boundaries remain. There is no
subgroup maximum/sum, reassociation, reciprocal substitution or precision change.

The isolated API is `collapse(x, mixes, scale, base, iters, epsilon, stream)`.
It returns an optional owned `HcResult`, with the same three-array cleanup as the
original. Strict eligibility is B1/T3/four streams/H4096, BF16 X, FP32
mix/scale/base and finite positive epsilon with 20 iterations. The null behavioral-red stub failed at the intended missing-helper check before
the single fixed green body was applied. Scoped opt-in policy/counters are now
prepared for the root-owned model seam.
No runtime delegation was present in the numerical component binary. A later
default-off primitive seam is root-owned WIP for the model gate; this worker
made no shared runtime or public test-runner edit.

All four files in the actual HC manifest matched their byte counts and SHA256.
Both prefill files have X `[1,512,4,4096]`, mix `[1,512,24]`, scale `[3]`, base
`[24]`, BF16 mixed and FP32 post/comb. HC epsilon is the stored FP32 value near
1e-6; iterations are exactly 20. The probe uses three adjacent rows starting at
509 from layers 0/3/23/44, attention and FFN. These are constructed T3 blocks
from real code/prose inputs, not captured speculative rows.

Tests were written first. Strict proof compares every mixed/post/comb
bit against current collapse and the stored capture, then downstream RMS/expand
bits. One special-value case includes signed zero, extreme values, NaN/Inf;
row permutation and unsupported T2 check ordering/fallback. All SIMD shuffles
execute uniformly across the first 32 lanes, with four entries read in original
order and the same maximum operand order/NaN semantics.

Only the code fixture's complete eight-block collapse receives three warmups
and eleven alternating pairs. Config/metadata, all three outputs, endpoint
evaluation and every free are included. Inputs are resident equally; proof
copies stay outside timing. Candidate peak is recorded above the held code/prose
fixtures. No target/assistant load, variant or model speed claim is part of this
component. Root owns all delegation/admission/model gates after a clear win.

Evidence key `glm53-hc-collapse-simd32-20261003` holds verified fixture contracts,
prepared stub/green/probe hashes and exact focused ReleaseFast build recipes.


## Recorded component gate

ReleaseFast red compiled after a probe-only pinned Zig formatting API repair,
then failed at `ExpectedHcSimd32` as intended. Green compiled and passed both
focused tests, exit 0. Exact evidence:

| Evidence | Values |
| --- | ---: |
| Current reference versus stored capture | 197568 |
| Candidate mixed BF16 bits | 221184 |
| Candidate post/comb FP32 bits | 1080 |
| Downstream BF16 RMS/expand bits | 1105920 |

The 1080 coefficient values are post/comb only. Pre remains the unchanged
scalar formula executed by four lanes, verified by source inspection; it was
not separately materialized or directly compared. The mixed/downstream proofs
exercise its effect. These include code/prose, special values, row permutation
and fallback guards.
Three warmups per arm preceded eleven fresh alternating complete-operation
pairs. Original median was 0.369375 ms and candidate 0.267292 ms, 27.6367% less.
All 11 pairs won; paired median reduction was 26.7588%, range 19.6950–34.4382%.
All raw pairs are retained in the artifact. Peak above the held fixtures was
264288 bytes. This is eight constructed T3 collapse operations, not 90 actual
verifier calls or a decode-throughput result.

Foreground QoS, per-job lock, confirmed maximum fans and quiet idle were used.
Red and green processes ended; lock was released and fans restored. Helper
SHA256 at the numerical gate was
`f5425a2579b33222752798386bccd48308c8672f30a5728eaecf390fc8338015`.
Binary/source/build/thermal records preserve the original stub, formatting
failure and fixed green. Later scoped policy/counter additions have separate
hashes and remain uncommitted pending root's strict actual-model gate.
