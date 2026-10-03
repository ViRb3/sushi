# Fixed route6 quality rejection

Route6 is rejected by the unchanged two-prompt long-quality gate. Its 20.23%
complete L20 component win did not establish acceptable target behavior.
No throughput model ABBA, threshold change, corpus variant or numeric retry
follows this rejection. Accepted runtime remains `4fcb541e`.

One target-only load used the original nonrepeated 16384-ID code/prose corpus,
current HC collapse/native B1/packed32 flags, chunk2048/async2 and mini off.
Only the route6 binding changed between fresh requests. Both independent
prefixes remained held consistently through teacher-forced continuation.
The original 24 declared positions in the last prefix chunk plus 192 predictions
were scored for each prompt; every row is retained even though code rejected
before prose began. Last-only actual forward logits supply the first generation
prediction. Late labels are actual next input IDs except the final baseline
argmax; continuation labels are baseline greedy IDs and the candidate consumes
those exact IDs. Each sampled head runs one row at a time, with no full
2048×vocabulary head or per-position teacher-cache clones.

Per-prompt bounds stayed meanKL≤0.01, maxKL≤0.15, top1≥95%, mean forced-NLL
increase≤0.02 and new nonfinite=0. The declared population is **all 216 rows**;
late/tail groups below are diagnostic subdivisions, not replacement gates.

| Prompt | Population | Mean KL | Max KL | Top1 agreement | Mean NLL delta | New nonfinite |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Code | All 216 | 0.33400158 | 18.89898256 | 197/216 (91.20%) | -0.09397863 | 0 |
| Code | Late 24 | 2.78637546 | 18.89898256 | 14/24 (58.33%) | -0.87725940 | 0 |
| Code | Forced 192 | 0.02745484 | 0.71579242 | 183/192 (95.31%) | +0.00393147 | 0 |
| Prose | All 216 | 0.05216336 | 2.40919189 | 207/216 (95.83%) | +0.06618885 | 0 |
| Prose | Late 24 | 0.32181646 | 2.40919189 | 17/24 (70.83%) | +0.55104865 | 0 |
| Prose | Forced 192 | 0.01845672 | 0.44074699 | 190/192 (98.96%) | +0.00558137 | 0 |

Code fails aggregate mean/max KL and top1. Prose fails mean/max KL and NLL
increase, despite passing aggregate top1. Both isolated forced-continuation
subdivisions also exceed the fixed KL bounds; finite outputs or high tail top1
do not rescue the declared gate. All 432 rows were scored, with zero invalid,
baseline nonfinite, candidate nonfinite or new-nonfinite entries. CPU f64 stable
log-sum-exp/KL arithmetic and the original thresholds were unchanged.

Each control made 0 route6 calls and each candidate made 336 during full prefill.
Continuation made 0 route6 calls in both arms, as required for single-row fallback
to the original eight routes. Native B1 calls were 2101 per arm and both requests
ended at offset 16575. Sampled last-capture logits were bit-identical to actual
last-only forward logits in both arms for both prompts. Approximate and original
prefix states are not required to match; no speculative acceptance gate was run.

Loaded/final active memory was 93,535,640,312 /93,535,771,384 bytes, measured peak
95,924,383,756. Memory and wired limits both stayed 115,448,725,504. The complete
conservative bill was 12,177,650,688 bytes above active memory, including every
original reserve, two independent four-version state owners, 256KiB route-policy
arrays and 149,304,320 retained CPU-logit bytes. No reserve credit was taken and
all staged admission checks passed. This screen makes no performance claim.

Artifact `glm53-route6-quality-20261004` retains the original corpus identity,
every row/group/teacher ID, counts, source/binary/flags hashes and telemetry.
It reuses frozen corpus key `glm53-shared-bank-quality-corpus-20261003` without
changing its arrays. The post-hook private binary SHA256 is
`cf2d80c24ad572e58a95c0c5702862d8abb49658efecef4b2c838cce6dad6e91`.
Focused ReleaseFast build and CPU score/bounds/nonfinite fixture passed. Loaded
job exit 1 is `Route6QualityRejected`, with result complete and both prompts fully
scored. Foreground QoS, explicit accepted libraries, maximum fans/idle and a
per-job lock were used. Process ended, lock released and fans auto; binary hash
was unchanged. This numeric failure is retained as decisive evidence.

The two private quality sources and compiled helper dependency were hash-verified
and archived; only those owned private quality files were removed from the active
source tree. Root restored the runtime seams and the helper owner removed its
prototype. All 432 rows remain available in the artifact.
