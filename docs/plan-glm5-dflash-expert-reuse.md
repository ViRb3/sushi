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
