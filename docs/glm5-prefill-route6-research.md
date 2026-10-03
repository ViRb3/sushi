# Fixed six-route wide-prefill experiment

Recommend one bounded **opt-in lossy prefill policy**, conditional on the fixed
quality screen: retain six of the original eight routed experts at each full
T2048 chunk. Keep decode, verification and assistant proposals at eight routes.
This targets the main routed consumer rather than another small HC kernel.
Current baseline is the accepted native/packed32/HC stack at `4fcb541e`.
No prototype, runtime edit, build or GPU job accompanied this research.

## Mechanism and measured source evidence

`glm5_forward.routeReference` computes original FP32 sigmoid router scores,
adds stored correction for top-eight selection, and returns output weights in
an unsorted selected-slot order. Therefore the first six slots are not a valid
truncation. The proposed policy starts from these exact eight IDs/weights:
select the six largest **actual output weights**, break ties by earlier original
slot, then compact survivors in original slot order. Renormalize their FP32
weights by `sum(original eight) / sum(retained six)`, with sums in original slot
order. Do not rerun a different corrected-router selection or use score slots
as expert ranks. This targets the original total mass subject to FP32 rounding; it does not
preserve expert contributions.
Expert outputs can differ greatly; routing mass does not bound logit error.

CPU-only analysis of the existing actual L20 T2048 capture establishes:

| Existing / derived policy | Eight | Six |
| --- | ---: | ---: |
| Assignments | 16384 | 12288 |
| Active experts | 265 | 245 |
| Live aligned WIN32 windows | 671 | 536 |
| M16 matrix tiles | 1174 | 906 |
| Routing-independent window capacity | 800 | 672 |

The six-route column is a hypothetical derivation from that one stored capture,
not a new target measurement. Retained original weight mass averages **82.1275%**,
median 81.7750%, minimum 76.4231% and maximum 94.9430%. Renormalization therefore
amplifies surviving terms materially. This is a substantial semantic risk, not
lossless math or a precision improvement.

Use strict BF16 B1/T2048/H4096/I2048, original E288/n36/MCG/W12/clamp10 and
normalized top-eight routing only. Keep accepted physical grid transposition,
WIN32, native 16×32×16 descriptor, original K order, F16 intermediate stores,
SwiGLU/Hadamard stages and FP32 finish. Parameterize only assignment count/top-k
for this separate S12288 consumer; do not fall back to the old output-stripe
scheduling and compare that as if it were the accepted baseline. Every other
shape, partial chunk or configuration retains the original top-eight path.
Original compressed banks, small BF16/F32 tensors, BF16 MLA cache and FP32 KDA
state/accumulators remain unchanged.

## Whole-work opportunity and limits

The old forced T2048 profile assigned 810.624 ms of 1788.124 ms to complete routed
FFNs (45.3%). Even eliminating that entire old phase is an unrealizable 45.3%
ceiling within that diagnostic; the candidate affects only part of it. It is
perturbed attribution, not current HTTP latency. A purely
proportional 25% routed reduction would remove roughly 11.3% of that old total;
that is an optimistic scenario, not a forecast or strict latency bound. Sorting,
metadata, preparation and finish remain, and only 20.1% of live windows and
22.8% of M16 tiles disappear in the derived L20 fixture. Cache effects, fewer
rows per expert, compaction/renormalization and any validation work can erase
part of the opportunity. Fewer assignments never establish a speedup by themselves.
This does not promise 1500 prefill or affect decode's 60 tok/s gap.

Explicit existing expert planes fall from 592 to 448 MiB at the fixed dimensions:
paired prepared inputs 192 MiB, gate/up 48 MiB each, middle 48 MiB, down 96 MiB,
final output 16 MiB. New compact IDs and weights total 98,304 bytes; original
router outputs remain live. Metadata/inverse/sort storage stays bounded. Do not
credit these source savings out of any accepted request reserve before actual
lifetime/peak proof; charge all extra policy arrays and keep current wired policy.
No persistent decoded weights or expanded checkpoint is proposed.

## Fixed proof and gates

Freeze policy, corpus, positions and quality rules before implementation. First
prove survivor masks, uniqueness/range, stable tie/slot order, scaling, no
manufactured routes and fallback guards. Zero/nonfinite source scores must not
be silently sanitized into finite accepted results. Compare policy values to an
independent FP32-order oracle and every retained gate/up/down F16 value against
the original consumer on those same six routes. Native final BF16 results must
match the independent six-route chain. Report drift versus original eight-route
outputs separately; cross-policy bit equality is not expected.

One actual L20 complete-chain gate uses the saved X/IDs/scores and original full
E288 banks, current transposed eight-route control versus fixed six-route policy.
Three warmups and eleven alternating pairs include compaction/renormalization,
sort, metadata, prepare, all three unchanged GEMMs, middle, finish, allocations,
settlement and frees. Hold reference memory equally, prove peak/cleanup and
report every pair. Stop a noisy/loss outcome with no width or kernel variant.

A component winner then receives **quality-first actual-model evaluation before
throughput acceptance**. Reuse the declared shared-bank screen's two nonrepeated
16384-ID archives: consumer code with an early retry-limit record, and engine
prose with an early lantern record. Preserve all 24 declared late-prefill positions
and 192 baseline-forced continuation predictions per prompt on the current
native/HC target. Require per prompt mean KL≤0.01 nats, max KL≤0.15, top1≥95%,
mean forced NLL increase≤0.02 nats and zero new nonfinite values. These same-pack
runtime drift bounds do not replace lossless-teacher checkpoint KLD. A failed
screen rejects the policy without changing thresholds, routes or corpus.

Only a passing screen receives one matched long-prefill model ABBA, equal held
references, inclusive preparation/cleanup, unchanged admission and strict
candidate serial/spec IDs/valid state from the same approximate prefix. Original
and approximate prefix caches are not expected to match. Final runtime acceptance
still requires root's HTTP qualification; no default change follows a component.

Ownership: one worker owns isolated prune/six-route consumer helper and actual
L20 probe; another owns the frozen existing quality harness adaptation; root owns
Moe delegation, checked billing, model/HTTP scheduling and acceptance. No generic
dispatcher, template sweep, model conversion or precision restoration.

Evidence: [accepted routed grid](engine-exl3-experts.md),
[current profile](glm5-next-wave-performance-plan.md),
[fixed quality corpus/rules](glm5-shared-bank-quality-result.md),
[retention rejection](glm5-prefill-weight-retention-result.md) and
[HC expansion model rejection](glm5-hc-expand-prefill-result.md).
The documented 64-row expert windows and other failed load/decode variants are
excluded; this experiment changes the routing policy while preserving the kernel.
