# Four-subgroup scalar IndexPool scoring

Research only at accepted runtime `e1597cc2`, 2026-10-03. Source warrants one
bounded **four-SIMD-group head-parallel scalar scorer** component. It shortens
the serial32-head chain without changing dot arithmetic or selected sets.
Speedup is unmeasured; added threads, shared traffic and a barrier may lose.
No source/prototype/build/GPU work accompanies this recommendation.

## Current path and evidence limits

Normal/replay `attention.decodeSelected` and verifier `mlaTree` → per-node
`decodeSelected` → `selectChunk/indexScores` still use scalar SCORE for short
rows. Prefill NAX declines T1–3. Each pool/query currently uses32 threads,
serially visiting32 heads. Each lane accumulates D128 at d=lane,lane+32,
lane+64,lane+96, then `simd_sum`. The dot is rounded BF16, rectified and
multiplied by its original head weight, rounded BF16 again, accumulated h0..31
in FP32, then rounded BF16 into the FP32 score output.

The fixed padded M128 short-row NAX scorer lost14.03%,0/11 complete attention
pairs; shared-prefix scalar scoring lost18.23%,1/22 complete selector samples.
Neither qualifies a new scoring arm. This candidate changes only head
parallelism within one original query/pool; no padding, native matmul, common
branch reuse, widened query rows or modified partition/expansion.

Existing actual16K three-branch native-attention controls measured0.770ms in
the earlier B3 qualification and1.153ms in the later short-scorer gate, with
different scopes/boots. Do not combine them or divide prefill T16 timings by16.
The complete selection+attention call, not an unmeasured scoring fraction, is
the only ceiling. Current system trace lacked kernel names and cannot attribute
an IndexPool share; long-context verifier growth is not a removable score budget.

## One exact layout

Strict BF16 J32/I128. One128-thread group handles each existing pool/query.
Subgroup s=0..3 handles heads8*s..8*s+7. Use lane **within its SIMD group**,
not thread_position.x, so each dot retains exactly the original four key
indices/update order and32-lane `simd_sum`. Retain original contraction-off
pragma and each BF16 dot/product boundary.

Lane0 of each subgroup stores its eight **already BF16-rounded, converted-FP32**
contributions into `threadgroup float contribution[32]` (128 bytes), avoiding
another cast/rounding boundary. After one uniform threadgroup barrier, global
thread0 adds contribution[h] sequentially h0..31 in FP32 and performs the
original final BF16/FP32 store. No subgroup total or parallel head reduction.
Future-pool eligibility is uniform: global thread0 writes negative infinity,
then all128 threads return before reads/barrier, preserving current masking.

B1 normal/replay and B3 verification retain their original per-node pooled
arrays, P dimensions, offsets and actual fork suffixes. Initially use the
helper for each existing node selection independently; no shared-prefix
plane or new three-query selection batching. Original negative/argpartition/
top512/2051 expansion and tie order stay unchanged. Unsupported shapes/dtypes
fall back. Below sparse eligibility no score call is introduced.

The head critical chain falls32→8 iterations, but group width rises32→128.
Total head dots/products are unchanged. Key expressions are evaluated for all
same32 heads; compiler hoisting could differ, so four subgroups may reload
the128-element key more often. Shared stores/barrier and one serial final sum
are new. No cache, occupancy, spill or4× latency claim follows source counts.
Global score/sort/attention buffers remain unchanged; only128 bytes on-chip
per group are added. Keep current native-attention reservation until measured.

## One decisive complete gate

One worker owns an isolated helper/probe and minimal original SCORE export;
root owns later ordinary/verifier seams and counters. Reuse actual16K captured
index-Q/weights/pooled/latent planes. Construct and explicitly label ancestry
frontier/suffix guards rather than call them captured trees. Prove every FP32
score bit versus current scalar, ordered512 pools and2051 IDs, then every
native B1/B3 attention BF16 output. Include negative weights, both zeros,
infinities/NaNs, cutoff ties, all four pool phases, odd history, distinct fork
suffix and P>8192. Storing/summing contributions must preserve these boundaries;
any mismatch rejects the candidate without a rounding-restoration arm.

One actual16K complete three-branch selector+native-attention comparison uses
three warmups and eleven fresh alternating pairs, including all metadata,
scores, negative/partition/expansion, gathered KV, native attention, final
settlement and frees. Keep three original branch dimensions/offsets. Measure
peak and engagement; do not time only SCORE or claim saved shader instructions.
Stop noisy/slower results without changing subgroup/head count.

A clear winner gets one root-owned matched8192/192 N2/A6/native model ABBA,
equal references, all rounds/cleanup, strict serial output and valid-state bytes,
unchanged acceptance and measured peak. No numeric target change is intended;
score drift is a failure. Preserve dense/sparse eligibility at2K, and do not
inherit the prefill NAX8192-pool cap:32K plus outputs must remain eligible under
existing limits. Selected32K follows only a real winner; no context sweep or
throughput forecast precedes acceptance.
