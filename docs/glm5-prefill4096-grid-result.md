# Fixed 4096 expert grid: component gate passed

One complete 4096 routed chain took 31.178834 ms versus 34.024125 ms for two
2048 chains by arm medians: 8.3626% lower, 11/11 favorable pairs and 8.3815%
median paired reduction. All retained F16 stages and final BF16 values matched
exactly. This qualifies the fixed component for a coordinator-owned model gate;
it is not runtime acceptance, actual 4096 activation evidence or a throughput
claim. See the [round plan](glm5-steady4096-tree-core-round-plan.md).

## Fixed workload and proof

The original actual L20 T2048 fixture supplied BF16 X, top8 IDs and FP32 scores,
with all three original full E288 banks. Concatenating each input with itself
constructed T4096; neither routes nor stored tensors changed. This replication
is explicit and can favor repeated expert membership. It does not predict the
window distribution of a real next 4096 chunk.

Only fixed 2048/4096 eligibility and route counts changed in
`glm_prefill_grid`: reshape, prepare, window metadata, middle, finish and
output shapes. The configuration cache has four bounded K/N/route-row entries
for the two existing bank geometries and two assignment counts. Physical grid
transpose, WIN32, MCG/W12/n36, K16 native MMA and retained F16/BF16 boundaries
are unchanged. No trunk helper, reader, route policy, window or group variant
participated. The temporary projection alias exposes the existing body only.

After inverse-routing sorted rows to original route slots, 603979776 F16 values
matched across prepared gate/up, gate/up outputs, middle and down. Final output
comparison covered 16777216 BF16 values. The private reconstruction also
matched production `tryMoe` outputs for both sizes. Independent CPU checks proved
sorted IDs/order, every inverse mapping, expert segments, starts, live counts
and unused zero windows. Mixed-size lazy graphs settled correctly with both
configuration sizes pending. Unsupported T2047/T2049/T4095 declined.

| Scope | Live assignments | Expert segments | Live windows | Padded rows | Window capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| One constructed 4096 | 32768 | 265 | 1174 | 4800 | 1312 |
| First original 2048 | 16384 | 265 | 671 | 5088 | 800 |
| Second replicated 2048 | 16384 | 265 | 671 | 5088 | 800 |

Both paths retain original routing-independent capacity. Live window counts
and padding differ as expected; no measured traffic or isolated GEMM share is
inferred from those counts.

## Inclusive timing

Three warmups per arm preceded eleven alternating pairs. Each control created
two original 2048 graphs and evaluated both together; the candidate created
one 4096 graph. Clocks included input views, sorting, metadata, preparation,
all projections, middle, weighted finish, endpoint evaluation and every free.
The same materialized fixture, banks and narrow/wide output and stage references
remained held throughout both arms. No forced intermediate endpoints were
inserted into timed execution. Engagement was exactly two calls versus one in
every timed sample.

| Pair | Two 2048 ms | One 4096 ms | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 34.038833 | 31.179542 | 8.4001% |
| 2 | 34.002791 | 31.197416 | 8.2504% |
| 3 | 34.031166 | 31.178834 | 8.3815% |
| 4 | 34.016000 | 31.160000 | 8.3960% |
| 5 | 34.006584 | 31.155083 | 8.3851% |
| 6 | 34.010292 | 31.160875 | 8.3781% |
| 7 | 34.008875 | 31.167042 | 8.3562% |
| 8 | 34.086417 | 31.200291 | 8.4671% |
| 9 | 34.027250 | 31.160667 | 8.4244% |
| 10 | 34.066416 | 31.221792 | 8.3502% |
| 11 | 34.024125 | 31.191334 | 8.3258% |

Maximum timing peak increments above the equally retained reference baseline
were 1074016772 bytes for control and 1074735432 for candidate. These are scoped
component peaks, not full-model admission bills or persistent storage costs.
No reserve credit follows from this measurement.

## Provenance and next decision

Artifact `glm53-prefill4096-grid-20261004` retains all pairs, proof counts,
fixture/bank identities, commands, exact compiled source snapshots,
binary/runtime hashes, process exits and telemetry. Source checkpoint was
`5de9c08d` plus hashed WIP. Original fixture SHA256 starts `1a2b0335ac281927`;
green grid source starts `60a88f99e609eb4b`, binary `211524fdd5eeecd2`.

The genuine red returned `MissingGrid4096` before implementation. Two earlier
private packaging failures—missing output directory and an error-return syntax
error—are preserved separately and are not feature-red evidence. Red compiler
27121 exited 0 and proof 27298 exited 1 as expected. Green compiler 27998 and
proof 28149 exited 0; the filtered process reported both tests passed.
ReleaseFast, foreground `taskpolicy -a`, explicit accepted libraries and an
exclusive per-job lock were used. Fans reached 5349/5767 RPM after ten idle
seconds, with hottest reported sensor 43.43°C. The job ended, lock released
and fans returned automatic.

## Actual-model exact proof stopped

The subsequent source-frozen model job used the original nonrepeated 16384-ID
`consumer_code_far_retrieval` prompt, one target and unchanged A6 assistant
resident. It compared eight 2048 chunks with the fixed
`2048,4096,4096,4096,2048` schedule under current HC/native/packed32 flags and
T3 core off. Final prefix logits and valid MLA/KDA state both differed.
`Steady4096UntimedPrefixParityFailed` stopped before the candidate native64
continuation, equal-reference warmups or any ABBA arm. There is no actual-model
throughput result, numeric-error estimate or attribution to an individual stage.

Both memory and wired limits were 115448725504 bytes. The checked conservative
bill was 14106482688 above active memory: full existing 4096 request reserve,
normal 64-token growth, four immutable state owners with separately rounded
latent/pool capacities and maximum tails, four logit arrays and input/preparation
allowance. No reserve credits were taken. The failing prefix endpoint did not
record a scoped peak, so no model peak claim is made.

Artifact `glm53-steady4096-model-16k-20261004` preserves the frozen corpus/input
hashes, reviewed bill and protocol, all 148 compiled source snapshots and
hashes, logs and failure output. Build PID 31998 exited 0; binary SHA256 starts
`d7fb8dea9edf7159`. Actual PID 32385 exited 1, with seven filtered tests passing
and this proof failing. Source and binary hashes stayed unchanged. Foreground
QoS, confirmed maximum fans/idle and the exclusive per-job lock were used;
the same process handle ended, lock released and fans returned automatic.

Exact runtime acceptance failed. The declared unchanged code/prose numerical
gate must pass before any performance acceptance; no quality, HTTP, retry or
row-policy variant was run by this worker. The exact expert-component win does
not establish exactness of the complete widened model schedule.

The subsequent [fixed numerical screen](glm5-steady4096-quality-result.md)
rejected both prompt populations. The component win was never an actual-model
throughput win; the entire batching candidate was archived and removed.
