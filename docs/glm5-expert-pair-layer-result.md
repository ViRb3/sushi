# Expert-only pairing: complete original L20 result

The fixed constructed complete-layer gate passed exactness and reduced median
latency by 2.47%, with 11/11 paired wins. This qualifies further history/Model
proof; it is not a whole-model or HTTP acceptance result. Starting runtime is
`4fcb541e`, round plan `35666a7a`.

The test loaded original layer 20 stored tensors and full E288 banks only. Two
independent original 2048 BF16 residual halves used seeds 47101/47102 and nonzero
convolution/FP32 history 44/45. HC, KDA, router and shared operations retained
original 2048 shapes/arithmetic. Only routed expert X/ID/score arrays were joined,
then routed outputs split before the original shared add/HC expansion.

All 86,196,224 compared route/score/full BF16 layer/mean-HC capture/convolution/FP32
state values matched. The independent expert stage proof checked 603,979,776 F16
prepare/gate/up/middle/down values, both 8,388,608 BF16 routed output values,
original-order metadata, and output-view survival after owner release.

Three warmup pairs preceded 11 alternating inclusive pairs. Fresh state aliases,
every HC/KDA/projection/router/shared operation, preparation/join/split, final
output/state/capture settlement and frees were timed. Both control and candidate
references were held equally; CPU comparisons and loading were outside clocks.

| Pair | Two original experts, ms | Paired experts, ms |
| --- | ---: | ---: |
| 1 | 72.876500 | 71.950708 |
| 2 | 73.522208 | 71.625875 |
| 3 | 73.756625 | 71.747083 |
| 4 | 73.459667 | 71.775666 |
| 5 | 72.977875 | 70.738042 |
| 6 | 73.347833 | 71.620042 |
| 7 | 74.188958 | 71.842042 |
| 8 | 73.558875 | 71.703041 |
| 9 | 73.741375 | 71.888292 |
| 10 | 73.368125 | 71.332708 |
| 11 | 73.834959 | 71.493334 |

Complete-layer medians were 73.522208→71.703041ms (2.47431% reduction); median
per-pair reduction was 2.57927%. All samples remain. Successful pair calls were 1
and grid calls 2→1 per measured arm. Constructed halves used 229/231 expert
segments; the join used 240. Valid windows were 647+645→1157 and reserved launched
rows 51,200→41,984. These counts do not measure physical traffic or whole-engine
savings.

Absolute peak active allocations were 8,802,237,080 control and 9,080,959,452
candidate bytes, including equal fixture/reference owners. Candidate peak plus
all original prospective paired reserve 15,837,298,688 was 24,918,258,140 under
fixed memory/wired limit 115,448,725,504. This component scope does not establish
actual-model admission or net memory overhead.

The first private adapter used a short-row fused router API and declined 2048
before complete characterization, candidate comparison or timing. A temporary
read-only alias then exposed the existing dispatcher/reference fallback; its
Ops-owned results were retained directly. This packaging-only repair changed no
input, bank, route policy, arithmetic, timing or threshold. First failure and
corrected completion are both preserved; there was no numeric retry.

The corrected ReleaseFast build and all 16 dependency hash checks passed before
launch. Foreground QoS, confirmed maximum fans, idle and an exclusive per-job lock
were used. PID 45789 completed with exit 0; its lock released and fans returned to
auto. No history, target Model, assistant, new capture or benchmark ladder ran.

[Prepared history/capture protocol](glm5-expert-pair-layer-protocol.md) describes
remaining pre-Model and assistant lifecycle gates. Artifact key
`glm53-expert-pair-layer-20261004` preserves every pair/proof/metadata/result,
source/dependency/binary hash, library check, command and telemetry in both
attempts. Corrected binary SHA256 is
`56198a828435f42bb341012bd08edfa6b25682aec196a989e20bb64cdd84abe0`.
