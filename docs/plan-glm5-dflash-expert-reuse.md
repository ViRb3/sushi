# GLM DFlash: reuse EXL3 weights across short verification rows

CPU study, 2026-10-03. This is an implementation proposal, not a measured GLM
bottleneck or speedup. No EXL3 code or runtime setting changed during this study.
The source baseline is Sushi `99107a48`; concurrent prefill kernel work is outside
this proposal. See [DFlash measurements](plan-glm5-dflash2.md) for the existing
exact 512/64 comparisons.

## Existing paths and the arithmetic constraint

`glm5_dflash_ffn.apply` obtains native router IDs/scores separately for each tree
row, concatenates them in original row/top-k order, and calls `moeClamped` once.
For R=2–16 verification rows, GLM uses `moeSwigluClamped`'s decode-width branch:

1. `pairPrepareFromTokens` writes two expert-scaled/Hadamard F16 input planes.
2. `indexedPairCoopF16` dispatches gate/up together. Grid Z selects the projection;
   it does not combine different experts or reuse weights between token rows.
3. The clamped middle transform and down projection use either the separate
   F16 middle plane or the eligible `clampedMiddleDownCoop` fusion.
4. `downFinishReduce` visits each token's experts in the original top-k order.

For H=4096, I=2048, K=8, E=288 and S=8R slots, the planes are gate/up inputs
`[S,4096]` F16 each, gate/up inner outputs `[S,2048]` F16 each, optional prepared
middle `[S,2048]` F16, and down inner `[S,4096]` F16. K2.25/W12 banks use packed
gate/up `[288,256,128,36]` U16 and down `[288,128,256,36]` U16. Scores remain
attached to their original slot throughout.

There is already an expert-reuse implementation for MiMo:
`GROUP_MEMBERS_SOURCE`, `groupFunnelStep`, `pairPreparedGroupedBody` and
`downGroupedBody` group two slots of the same expert, decode weights once, and
maintain separate per-member accumulators. A 64-bit ballot finds matching slots;
only every second matching slot's threadgroup performs the work. The group size
is deliberately two because wider accumulator groups previously spilled.

That implementation is **not the GLM oracle**. MiMo's `groupEpilogue` combines
lane accumulators pairwise, uses XOR shuffles, then adds simdgroup contributions.
Its gate/up path may also preserve FP32 K-split planes. GLM's
`INDEXED_COOP_SOURCE` instead writes all lane partials, then for each output adds
`partial[g*256 + r*16 + column]` with **r=0..15 outside g=0..3**, and stores F16.
Replacing that with the existing MiMo body can change both reduction order and
rounding boundaries. Relaxing the MiMo architecture guard would not fix this.

The current sorted prefill GEMM similarly provides no proof of serial GLM
equivalence. Use its routing/inverse ideas where useful, not its dot-product
body, for an exact DFlash verifier.

## Proposed first candidate: unsorted two-member cooperative GEMV

Implement a GLM-specific group-two version of `INDEXED_COOP_SOURCE`. Keep the
four simdgroups, original key-tile traversal (`tk=sg; tk<IT; tk+=4`), eight FP32
accumulators per member, original codebook/window decode and exact epilogue.
For every decoded `float2` weight, update member 0 and member 1 independently
with the same per-member FMA sequence as the existing kernel. Extra independent
FMAs can be interleaved without changing either dependency chain.

Membership is ordered by original slot number. Rank 0/2/4/... among slots sharing
an expert is a leader, paired with the next matching slot when present. A group
must contain only equal expert IDs; grouping does not depend on adjacent tree
rows or accepted ancestry. Each member reads its own prepared input and writes
its original output slot. Rejected tree rows remain independent computations.

Start with the gate/up paired dispatch. Grid Z still selects only gate versus
up; group membership is identical for both. Allocate `partial[2][4][256]` FP32,
8 KiB total, versus the current 4 KiB. Each lane needs sixteen FP32 accumulators
instead of eight. After a uniform barrier, the first sixteen lanes can finalize
each member sequentially using the original r-then-g loop and F16 store. Give a
singleton its own single-member loop so it does not execute dummy FMAs.

Two ways to obtain membership should be compared:

- **Inline ballots:** retain the existing fixed S-sized grid and determine
  membership within each threadgroup. R≤8 fits the current two 32-bit ballots;
  R≤16 needs four 32-bit words, with explicit rank/next-set-bit logic. Do not
  shift a 64-bit integer by 64 or truncate slots 64–127. Membership decisions
  must be uniform across all four simdgroups before an early return or barrier.
- **One GPU metadata dispatch:** produce `[S,2]` U32 members and `[S]` U32 count,
  indexed by original leader slot; nonleaders have count zero. This is at most
  1536 bytes for S128 and is reusable by gate/up/down. Launch the same fixed S
  grid and return on zero count, avoiding host readback or a CPU-sized compact
  grid. A separate uniform leader list can be explored later if empty group
  launches matter. The cost is one dispatch plus tiny allocation/reads.

The inline version has no added graph operation but repeats membership discovery
for every output tile. At R4, the current paired projection launches
128 output tiles ×32 slots ×2 projections =8192 groups. The metadata version
avoids rescanning all route IDs in each of those groups. Which wins is unknown.
Neither version needs to gather or reorder activation planes.

After direct gate/up parity and timing, extend the same reduction to down.
For the separate middle path this needs only a grouped cooperative down body.
For the existing multirow middle/down fusion, stage **each member's** exact
clamped middle result as F16: two 2048-wide planes consume 8 KiB, plus the 8 KiB
partial array, about 16 KiB before small metadata. Traverse its eight output
tiles sequentially as before, sharing each decoded weight across the members.
This option preserves the current multirow fusion benefit and avoids introducing
a global middle plane merely to enable grouping. Compare whole chains against
the current auto-fused path, not only against the older separate-down arm.

## Sorting alternative and costs

Sorting slots alone may improve cache locality, but it still decodes each weight
per slot. Existing helpers can avoid explicit activation-plane copies:
`pairPrepareFromTokens` accepts an order vector and reads the original token row;
`finishMimoSorted` follows an inverse map while retaining original top-k order.
The extra work is argsort, order conversion if needed, sorted-ID gather and
inverse construction. At S32 the metadata is tiny; launch costs may still exceed
the saved bank reads. A sorted cooperative body must retain GLM's reduction.

If grouped kernels already access original member slots directly, sorting adds
no necessary correctness function. Use it as a separate locality experiment,
not a prerequisite. Avoid an explicit gather of the two `[S,4096]` input planes:
that adds reads/writes to planes totaling 512 KiB at R4 and 2 MiB at R16. Avoid
the `[S,4096]` output scatter too; direct original-slot stores eliminate it.

## Evidence needed before spending GPU time

Capture real DFlash routing for every layer and round at NODES1/3/7 (R2/4/8,
plus terminal short rounds), prompt and layer IDs, and original ordered slot IDs.
Do this in an untimed run or defer host extraction until after timing. Preserve
the actual tree rows, including rejected siblings; serial-token routing alone
does not estimate verification overlap.

Existing `SUSHI_EXL3_UNION_HIST` logs short-row assignments, unique experts, number
of repeated experts and maximum multiplicity, but emits full counts only above
16 rows. Existing warmed DFlash result artifacts did not enable this capture.
For group two, collect the complete multiplicity histogram `c_e` and compute:

- baseline decoded-bank visits: `S = sum(c_e)`;
- grouped visits: `G2 = sum(ceil(c_e/2))`;
- potential eliminated visits: `S-G2` and fraction `1-G2/S`;
- singleton frequency, pairs per layer, and upper-group alternatives G4/G8.

Unique count alone overstates realizable two-member reuse for long runs. Record
original-slot distances and pairwise row intersections too, to evaluate the
sorting-only hypothesis. Weight all aggregate statistics by actual calls and
component duration, rather than averaging per-layer percentages equally.

One packed 4096×2048 projection at 2.25 bits/weight is 2,359,296 bytes. Three banks
therefore represent about 7.08 MB per slot before scales. At R4/S32, a theoretical
all-paired case removes about 113.25 MB of logical repeated bank visits per layer.
This is **not saved DRAM traffic**: concurrent groups may already share cache
lines, and grouping also changes occupancy, register pressure and instruction
mix. No end-to-end speedup follows from this byte estimate.

Historical [Qwen results](perf-baselines.md) are a counterexample: repeated slots
were 20/28/31% at 2/3/4 verification rows, yet byte-exact MiMo grouping lost
0.8–1.3% end-to-end in controlled ABBA runs. MiMo's own positive component results
used different geometry and arithmetic. GLM needs its own route replay and
whole-model test.

## Correctness and performance gates

First compare raw F16 gate, up and down output bits against the unchanged
cooperative kernels. Cover all seventeen 2–4bpw rates, supported codebook/window
pairs, distinct gate/up inputs, rows 1/2/3/4/8/16, expert IDs 0 and 287, and slots
31/32/63/64/127. Include no overlap, full row overlap, odd multiplicities, widely
separated matches and singleton tails. Mixed bank rates retain an explicit safe
fallback. Check noncontiguous inputs against the existing layout contract.

Then compare full clamped MoE output bits, scores/top-k ordering, and the fused
middle option at 4096/2048. The original BF16 input boundary, F16 preparation,
FP32 per-lane FMA sequence, r-then-g reduction, F16 inner stores, clamp/SwiGLU
sequence and final score reduction all remain invariants. Finally rerun the
real DFlash token/full-state oracle, including rejected siblings, EOS and budget.

Timing should replay captured routing on representative layer banks with fresh
activations and both warm-bank and rotated-bank conditions. Compare current,
sorting-only, inline grouping and metadata grouping; include all prepare,
metadata, middle, finish and allocation work. After a component win, use the
same warmed 512/64 committed-token denominator and matched serial reference.
Report grouping engagement, paired/singleton counts, draft/verify/replay times
and acceptance. Keep this opt-in until whole-model measurements show a gain;
the current evidence establishes only a plausible implementation path.

## Actual N3 route capture

A separate opt-in capture at `489c5832` on 2026-10-03 collected actual consumed
routes for the same warmed 512-prefix/64-committed-token N3 workload. All output
IDs and complete final state matched the serial oracle and earlier profile-off
run. Capture is synchronization-perturbed and provides no throughput result.
The collector recorded 966 eligible FFNs: 23 rounds ×42 routed layers, each with
four verification rows and top-k eight. No one-row calls were skipped in this run.

| Quantity | Measured route statistic |
|---|---:|
| Slot assignments S | 30,912 |
| Unique expert visits, summed per call | 20,737 |
| Repeated slots | 10,175 (32.92%) |
| Group-two visits G2 | 23,466 |
| Potentially eliminated visits S−G2 | 7,446 (24.09%) |
| Two-member groups / singleton groups | 7,446 / 16,020 |
| Mean original-slot distance within a pair | 9.43 |

Multiplicity-one/two/three/four expert-call counts were
14,684 / 3,324 / 1,336 / 1,393. Consequently 51.82% of assignments remain unpaired:
a group-two shader must keep singleton work cheap, and its maximum register/shared
memory footprint can still reduce occupancy even on a singleton branch.

Reuse varies substantially by layer. The table below aggregates the 23 calls
for each zero-based layer; every layer had 736 assignments.

| Layer | Potential group-two eliminated visits | Fraction |
|---:|---:|---:|
| 3 | 49 | 6.66% |
| 4 | 65 | 8.83% |
| 5 | 65 | 8.83% |
| 6 | 69 | 9.38% |
| 24 | 189 | 25.68% |
| 20 | 194 | 26.36% |
| 27 | 244 | 33.15% |
| 26 | 251 | 34.10% |
| 25 | 253 | 34.38% |
| 34 | 260 | 35.33% |

This supports a controlled replay experiment covering low, median and high
overlap. It does not justify enabling grouping everywhere or fixing a layer policy
from one prompt. The 24.09% figure describes logical decoded-bank visits;
unchanged per-member FMAs, existing cache reuse, membership overhead and occupancy
can erase the theoretical benefit. The earlier negative Qwen result remains a
relevant caution at a similar repeated-slot fraction.

The [bounded collector](glm5-dflash-profile.md#bounded-actual-route-capture) is
off by default, preserves original slot order, and fails rather than truncating.
Private artifact `glm53-dflash-routecapture-20261003/capture` contains all ordered
IDs, full per-expert multiplicities, per-layer summaries and provenance. Every
serialized multiplicity/G2 count was independently recomputed from the ordered
IDs. The binary SHA-256 is
`ee78945e31ac04f08b01418d5730b6494b333acd294653f63c4af2236a9f7314`.
The exclusive capture used interactive QoS, a max-fan request and ten seconds idle
at 48.73°C before load; its GPU lock was released and fans restored afterward.
No group-two kernel has been implemented or benchmarked by this capture step.

## Standalone exact group-two candidate

`src/exl3/glm_group2.zig` now contains an unintegrated cooperative candidate.
Inline ballots pair occurrences in original slot order, including masks spanning
slots 31/32/63/64/127. Followers return uniformly; singleton leaders execute the
unchanged baseline body. Paired leaders decode each weight once, then update two
independent FP32 accumulator sets with the original FMA sequence. Both epilogues
preserve the original r-then-simdgroup addition order and F16 stores: parallel
partials use 8 KiB, while sequential member reduction reuses a 4 KiB plane.
The candidate has no production callsite, and incompatible shapes return no
candidate. Its routed-chain guard validates all bank/scaling/score shapes and
dtypes before preparing inputs, and checks output-width multiplication.

Tests passed all seventeen 2–4bpw rates, 4096/2048 dimensions, sparse/odd/full
sharing, singletons, ballot boundaries, noncontiguous inputs, and malformed input
refusal. Actual checkpoint trellis/Suh/Svh slices were then loaded for three
captured route sets; all four candidate chains matched every baseline BF16 output
bit. Expert IDs were compactly remapped with their corresponding original weights.
The real-weight subsets are exact copies; activations and scores are synthetic.
This proves these component outputs, not full-model speculative parity.

A quiet warm-bank comparison used the natural-layout baseline, with lane-pair
and lane-down switches explicitly off. Five warmups per arm preceded eleven
alternating forward/reverse rounds, three evaluations per sample. Times include
host construction/evaluation/free and all routed-chain preparation/finish work.
The gate/up-only arms retain the native fused middle/down. The all-projection
arms use separate middle preparation followed by grouped down, so they include
the cost of losing that fusion.

| Captured layer / potential saved visits | Baseline µs | Gate/up parallel µs | Gate/up serial µs | All parallel µs | All serial µs |
|---|---:|---:|---:|---:|---:|
| 3 / 6.25% | 1073.63 | 1044.17 | 1045.04 | 1048.01 | 1042.99 |
| 20 / 28.13% | 1151.21 | 1062.75 | 1063.06 | 1042.26 | 1045.64 |
| 34 / 34.38% | 1126.65 | 1011.57 | 1040.00 | 994.58 | 990.18 |

The best routed-chain medians improve by 2.85%, 9.46% and 12.11% respectively.
Both all-projection arms won all eleven paired rounds in the middle-overlap
case; the high-overlap case won ten or eleven. This is sufficient to retain the
candidate for composition experiments. It does **not** establish a gain over the
newer half4 lane paths, nor show that those gains add together. Compare a composed
candidate directly against lane-pair plus lane-down before integration.

Compacting selected banks changes physical address spacing/cache behavior, and
only three route records were timed. Full-model confirmation remains required.
The nine sampled expert shards (gate/up/down at these layers) in the newer
Sushi-2.3bpw target have matching size and sampled SHA-256 data with the original
Sushi-2.4bpw target; this is a sample check, not a whole-file hash proof. Its A6
trunk can still change activation/routing distributions.

Private artifact `glm53-group2-20261003` contains the exact source/build recipe,
actual-weight fixture exporter/manifest, parity logs, samples and provenance.
Timing binary SHA-256:
`a38386598412ab7e1420196fb1f95bcd7c1f7c807cac3048fa4918e99abdb674`.
The run held an exclusive GPU lock, used interactive QoS, requested maximum fans,
and idled ten seconds at 49.84°C before timing; lock/fan cleanup completed.

## Half4 composition and N2 qualification

The candidate now composes with the qualified lane-ordered gate/up preparation
and staged lane-down preparation. Singleton and paired paths load the same four
F16 values through half4 reads, then retain the original independent FMAs and
r-then-simdgroup reduction. All seventeen rates and production-width projection
tests passed, followed by exact real-weight routed-chain comparisons.

Three-row cases below replay the first three rows of the existing captured N3
routes. They are not a fresh route capture from the newer N2/2.3bpw configuration.
The comparison explicitly enabled and verified the lane-pair and lane-down
baseline. Both variants group all three projections; numbers are warm-bank
routed-chain medians in microseconds.

| Layer | Rows | Baseline lane+down | Grouped 8 KiB parallel | Grouped 4 KiB serial |
|---:|---:|---:|---:|---:|
| 3 | 3 | 639.19 | 640.40 | 631.89 |
| 20 | 3 | 716.30 | 580.65 | 573.12 |
| 34 | 3 | 726.15 | 606.08 | 577.51 |
| 3 | 4 | 784.05 | 782.55 | 776.12 |
| 20 | 4 | 847.38 | 726.05 | 725.92 |
| 34 | 4 | 846.28 | 676.64 | 708.97 |

The three-row serial-member variant reduces the middle/high cases by 19.99% and
20.47%, with ten or eleven paired wins out of eleven; the low-overlap case is
essentially flat. Another worker reported a brief approximately 0.1-second
CPU-only tokenizer hash during this microbenchmark window. No GPU/build overlap
was reported, but treat the component timing as provisional rather than fully
isolated evidence. The subsequent full-checkpoint run provides the stronger gate.

`SUSHI_GLM_DFLASH_GROUP2=1` opts the DFlash FFN adapter into the serial-member,
4 KiB, half4 variant. It is **off by default**. Eligibility is deliberately narrow:
3–4 BF16 verification rows, hidden/intermediate widths 4096/2048, top-k eight,
clamp ten, MCG/W12 and K2.25 gate rate. Complete bank validation remains required;
unsupported cases use the existing routed path. One-row dispatch remains unchanged.
The diagnostic reports `group2_batches` to prove engagement. A production-width
integration smoke matched the baseline and independent serial-row FFN outputs.

One full N2 qualification used the newer Sushi-2.3bpw target, A6/group128 assistant,
lane-pair/down enabled, async layer group four, the same 512-token captured prefix,
chunk 128 and 64 committed-token accounting as the preceding N2 baseline.

| Measure | Group-two qualification |
|---|---:|
| Speculative committed-token rate | **45.4485 tok/s** |
| Prior N2 baseline sample | 42.4344 tok/s |
| Change versus prior sample | +7.10% |
| Same-run matched serial | 31.4159 tok/s |
| Speculative / matched serial | 1.44667 |
| Rounds / accepted drafts / verified rows | 24 / 40 / 72 |
| Group-two / FFN / batched router calls | 1008 / 1008 / 1008 |
| Decode-phase peak | 94.979 GB |

All 64 output IDs and complete final target state matched the serial oracle;
IDs also matched the preceding N2 sample. Draft/verify/replay/commit totals were
146.25 / 1201.90 / 42.32 / 16.73 ms. This crosses the revised 45 tok/s decode goal
in one short run, with limited margin. Repeat and broader-prompt qualification
remain necessary before treating it as a stable floor or enabling it by default.
Captured chunk-128 prefill was 366.81 tok/s; this is separate from native dense
prefill, and this run does not establish the revised 1,200 tok/s prefill goal.

Implementation commits are `89632167` (standalone) and `ba106e5e` (composition and
opt-in FFN adapter). Private artifact `glm53-group2-20261003` retains composition
samples and `qualification-n2` settings/results. Qualification binary SHA-256:
`af011353940b06945df3a2e99d045af873392d94a68b614a58fc71fcb736f071`.
The build/run provenance records concurrent unrelated KLD source work rather
than claiming a wholly clean checkout. The exclusive full-model process passed
its gate, restored fan auto and released the GPU directly to teacher capture.
