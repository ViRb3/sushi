# Current routed capture and full-bank replay

The fixed current-baseline job passed all 192 greedy output IDs and
247,959,552 valid state bytes against the matching native B1 serial oracle.
Replay matched all 516,096 recorded BF16 routed-output values. This is a
measurement of one saved T3 workload, with no kernel arm or speedup claim.

One Sushi 2.3 target and stored A6g128 assistant load used N2/children 4,
native B1/B3, accepted group2/lane/hoist/leaf flags and async4 verification.
The exact original 2048-ID official-template 512 task plus neutral WATER
filler was repeated four times to 8192 IDs; it is not pure code. Dense prefill
used chunk 2048/async2. Generation ignored EOS. All 81 speculative rounds were
retained. The final request committed 192 inputs; MLA latent/pooled comparisons
excluded uninitialized capacity tails and included valid pool tails plus full
initialized FP32 KDA/conv state. No teacher precision restoration was used.

The scoped producer captured complete natural T3 model rounds 0, 1, 2 across
routed layers 3–44: 126 ordered-ID/engagement records. Only round 0 retained
post-norm BF16 X, FP32 scores and BF16 routed Y for all 42 layers. CPU capture
storage was 2,085,720 bytes; the saved 168-array fixture was 2,085,637 bytes.
T1 replay, prefill and serial oracle were unbound. Every captured layer engaged
the current group2 path. Readback forced evaluation: this captured request and
its round phase clocks are **not normal throughput evidence**.

| Captured model round | Assignments | G2 leaders | Pair leaders | Singleton leaders |
|---|---:|---:|---:|---:|
| 0 | 1008 | 878 | 130 | 748 |
| 1 | 1008 | 837 | 171 | 666 |
| 2 | 1008 | 848 | 160 | 688 |
| Total | 3024 | 2563 | 461 | 2102 |

Singletons are 69.51% of assignments and 82.01% of leaders. Multiplicities were
1999 singleton experts, 358 pairs and 103 triples across layer/round records.
Original paired-slot distances had mean 9.52 and median 9; the complete histogram
and all IDs are preserved. These are current routes rather than compact-bank
or synthetic-activation proxies. Group counts are logical bank visits, not
measured traffic, occupancy, spills or an attributed removable latency budget.

Replay used each original target layer's full E288 banks directly, with saved
operands resident, original layer order and current
`api.glm_group2.moeLayout(serial, grouped, lane)`. It included preparation,
gate/up, middle, down and weighted finish. Eleven async submissions matched
absolute layer boundaries 3/7/…/43; one endpoint settled all 42 outputs, followed
by output/vector cleanup. Error paths settle pending work. No per-child forced
waits or timing markers were inserted.

After two warmups, three baseline complete 42-chain samples were
17.837334, 17.769000 and 17.823292 ms: median 17.823292 ms, range 0.383% of median.
The workload contains saved independent routed chains; it does not measure
routing, HC, shared FFN, KDA/MLA, assistant or policy/commit work. This warm
fixture cost is **not whole-verifier attribution**, and no ratio to a captured
or older verifier total establishes its share. No A/B comparison, new kernel
or throughput forecast follows this measurement alone.

Current captured-run counters were native B3=880 / B1=22, group2=3360,
routed=3402, QKV hoist=8160, MLA query/value=880 each and leaf hits/misses=
1632/1122. Captured-model peak was 96,315,441,696 bytes; replay high-water was
7,700,480 bytes above its resident target/inputs/held-reference baseline.
Both scopes retained the serial references consistently. The raw replay
increment is not a standalone persistent-runtime bill.

Artifact `glm53-current-routed-8k-20261003` retains exact input/output IDs,
all round phases, complete route/multiplicity/distance records, first-round
X/IDs/scores/Y fixture, all baseline samples, source/binary/flags hashes and
thermal/progress logs. Binary SHA256 starts `7a962df9d28b3231`, source checkpoint
`fe37527c`; input u32-LE SHA256 starts `f31a5905dfbbf824`. Eight filtered tests
passed, exit 0. Foreground `taskpolicy -a`, exclusive per-job lock, confirmed
5355/5783 RPM fans and 38.89°C after ten idle seconds were recorded. The process
ended, lock is free and fan terminal state is auto (manual false, TTL 0). No further
runs occurred. Private harness/replay source and the producer's diagnostic tap
remain uncommitted pending coordinator-owned archive/disposition.


After measurement, all private replay/capture sources and the one diagnostic
FFN tap were archived with hashes and removed from the active tree. Accepted
runtime is unchanged. The next candidate must beat this current full-bank
workload and then a matched actual-model gate; these samples do not predict60tok/s.
