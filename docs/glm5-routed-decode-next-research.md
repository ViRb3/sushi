# Current routed-verifier decision measurement

Research only, accepted runtime `e1597cc2`, 2026-10-03. Recommend **one small
current N2 route/component capture** before another expert kernel. Source does
not establish a fresh weight-traffic or scheduling win beyond accepted group2.
No prototype, build, GPU run or default change accompanies this recommendation.

## Current path and evidence

`glm5_dflash_model.verify` collapses HC, computes FFN RMS and calls
`glm5_dflash_ffn.apply` in affine-FFN mode. The batched router preserves each
row's top8 slots/scores. Qualified BF16 T3/T4, H4096/I2048, E288, MCG/W12,
n36/clamp10 enters `glm_group2.moeLayout(.serial,.grouped,.lane)`:
lane gate/up preparation → grouped lane gate/up → separate lane middle →
grouped lane down → original weighted finish. T1 and unsupported shapes use
the existing native route. The selected run flags include GROUP2, LANE_PAIR
and DOWN_LANE, async4, N2/children4 and A6 assistant. Native B1/B3 attention
is the optional target mode in the latest matched decode evidence.

The group2 path calls lane routines directly; legacy lane-pair/down switches
and counters alone do not prove this branch. Require actual group2 engagement,
qualified shape and the selected serial/grouped/lane call. It already shares
each decoded weight across two matching slots. Singleton leaders execute the
original eight-accumulator body; pairs keep independent accumulators and the
same r-before-simdgroup reduction. Serial epilogues reuse4KiB threadgroup
partials. Original slot stores, F16 boundaries and top8 reduction remain exact.

Latest matched8192/192 N2 control cost50.897ms verification per round, with
81 rounds and40.22 delivered tok/s. This bounds the **entire** verifier,
including MLA/KDA/HC/shared work; it is not routed cost. The old145.49ms
six-round child profile forced waits, predates later optimizations, and its
marker tax is visible in tiny norms. Do not divide it into a current removable
latency budget.

Actual old N3 capture:30912 assignments,23466 group2 leaders,7446 paired
groups and16020 singleton groups. Thus51.82% of assignments, or68.27% of
active leaders, were singleton work. But this was512-prefix/T4 routing. The
L20 T3 prefix fixture has24 assignments,17 unique experts and18 leaders:
six pairs/twelve singleton leaders. It uses old captured routing, compact
bank subsets and synthetic activations/scores, not current N2 activations.
Its current inline whole chain measured590.385µs; partner prepass589.416µs
won only5/11 and was removed. That rules out claiming repeated ballots as a
large demonstrated cost. One projection is2.25MiB; this fixture's three-bank
logical visits are162MiB before grouping and121.5MiB after. Neither number
measures DRAM traffic or the cache benefit still available.

## Why measurement comes first

A conceivable remaining issue is singleton work inheriting the compiled
paired branch's register/resource footprint. Source branch structure does
not prove spilling or occupancy loss. Separating singleton/pair dispatches
would add classification and projection commands, and metadata prepass alone
already failed its inclusive gate. **Do not implement that split without new
current evidence.** No group3 replacement, word sharing, window/grid variant,
precision restoration or KDA projection work is proposed.

Owner: one routed-component worker with a tiny default-off capture/helper;
coordinator owns the actual-model run. Piggyback on the next current proof:
Sushi2.3bpw target, A6g128 assistant, the same fixed8192 IDs as the matched N2
control,192 outputs/ignoreEOS/greedy, N2/children4, nativeB1/B3 ON, async4,
group2/lane/dense-row/QKV-hoist/leaf accepted flags, prefill chunk2048 and
prefill async2. Stamp source/binary/flags and actual engagement. Collect ordered route IDs and
shape/engagement for three complete T3 rounds across42 routed layers, plus
post-norm BF16 X, scores and routed BF16 output for **one** of those rounds.
Keep captures outside throughput evidence. IDs need about12KiB; one42-layer
X/output/ID/score record is about2MiB. No weight copies, widened row policy,
prefix clone or general capture framework. Use full original E288 bank
addresses already resident in that same model load; compact-bank replay can
change locality and is not the decisive control.

Recompute multiplicities, G2, singleton/pair fractions and original pair-slot
distances from those IDs. Replay the captured42 whole routed chains with fresh
scopes, current four-layer settlement cadence, original bank order and all
prepare/gate-up/middle/down/finish/eval/free costs. A few warm baseline samples
provide a current component cost/variance, not a speedup; no A/B variants.
Check every replay BF16 bit against captured routed output. Do not insert
per-child waits and call their sum normal verifier latency. If existing GPU
resource telemetry is available, retain singleton/pair resource evidence;
do not build a profiler merely to support the hypothesis.

This one measurement decides whether routed work still has sufficient whole-
chain headroom and whether singleton resource behavior warrants one bounded
follow-up. Without it, neither a new kernel nor a60tok/s forecast is justified.
Original temporary planes remain under1MiB per T3 call, with4KiB on-chip
partials; capture memory is diagnostic and requires no persistent runtime bill.

A subsequently evidenced exact candidate must preserve every gate/up/down
F16 bit, final BF16 MoE output, score/slot order and independent tree rows,
then win an inclusive captured whole-chain comparison. Coordinator's one
fixed8192/192 model ABBA must show repeatable total decode gain with identical
assistant/target flags and complete serial token/valid-state parity; equal
reference allocations and cleanup/all rounds remain included. Routing geometry
has no2K–32K prefix-dependent switch, but actual routes/acceptance can change
with context. Do not extrapolate the8K overlap to32K; selected32K qualification
is a final winner gate. This proposal does not improve cold prefill or alter
the accepted MLA/KDA precision/cache policy.
