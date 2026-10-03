# Indexed prefill and optimized N3 round

Start from accepted optional runtime checkpoint `e1597cc2`. The A6 assistant
remains default; native decode attention remains opt-in. Latest native 32K
performance is 608.77/614.65 prefill and 43.00/38.08 decode tok/s for predictable/
ordinary inputs. These are separate-boot qualification values, not a matched
native speedup claim. The 1500/60 goals remain open.

Two research reports define this round: [indexed prefill](glm5-prefill-next-wave-research.md)
and [optimized verification policy](glm5-decode-next-wave-research.md).

| Worker | Candidate | Owned source |
|---|---|---|
| 1 | Direct indexed NAX prefill KV loads | New helper/probe and minimal pinned native Metal subset, NOTICE |
| 2 | N3 readout and T4 MLA path | Assistant horizon3, overlay4, native B4, exact M1 query/value broadcast4; MLA verifier seam |
| 3 | T4 KDA path | A6 QKV hoist4 and retained first-path KDA leaf4 plus focused probe |

Worker1 first packages unchanged native D512 math and proves its output against
existing SDPA. Only then change selected K/V fragment addressing. Keep ordered
selectors, two-tile cadence, BF16 operands and FP32 accumulators. One complete
T2048 actual 16K-fixture proof and eleven inclusive pairs decide this component.
Do not integrate a losing/noisy loader, approximate selectors or precision restoration.
Memory admission stays conservatively unchanged until actual live bytes are proved.

Workers2/3 form one optimized N3/children4 bundle, not independent speed claims.
Keep N2 behavior exact. T4 output/state proofs cover chain and fork ancestry,
partial pool boundaries and shortened final trees; no full-prefix clone or
shared mutable cache. Native B4 must match mode-matched B1 outputs exactly,
with conservative 16 MiB/layer and 64 MiB/async4 temporary bounds and fallback when branches cannot fit. KDA
retains only the first-path FP32 leaf, with the existing hit/miss replay rule.

Coordinator owns HTTP opt-in node policy, clipping by remaining output/context,
reservation/admission/counters, aggregate model gate and integration. Decode
microbench uses whole four-row work, not isolated dispatch savings. One target/
assistant load with a fixed 8K prefix and 192 outputs compares fully optimized
N2 versus N3 in ABBA, with identical flags/assistant and explicit numerical
target mode. Exact serial output and all valid final state are checked outside
timing; retain equal reference memory and all rounds in throughput. Stop N3
if extra verifier work outweighs acceptance. No policy width sweep.

All builds and quiet GPU/full-model runs are scheduled sequentially; take and
release the GPU lock per job, foreground QoS and thermal protocol. Production
source stays WIP until component and real-model acceptance, then commit/push
only accepted runtime. Routine HTTP remains 2K–16K and selected32K once.


## Outcome

Both candidates were rejected and all owned runtime source was restored.
[Indexed prefill](glm5-indexed-nax-prefill.md) was85.15% slower in its inclusive
component, despite exact outputs and selected IDs.
[Optimized N3](glm5-optimized-n3-result.md) passed complete target token/state
proof, but gained0.56% against1.46% control timing drift. Component wins did
not overcome higher whole-model verifier cost. No further variants or ladders
followed. Runtime remains `e1597cc2`: optional A4 format and native B1/B3
attention, with A6/default N2 unchanged. Goals remain open.
