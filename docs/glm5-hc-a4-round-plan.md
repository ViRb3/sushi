# Fixed HC candidates and current-native A4 gate

Start from accepted runtime `af51e72f`, documented outcome `7bb5d32b`.
Two source-only reports identify distinct exact HC candidates:
[prefill expansion](glm5-hc-expand-prefill-research.md) and
[verifier coefficient preparation](glm5-hc-collapse-next-research.md).
A separate [optional A4 comparison](glm5-a4-current-native-gate-plan.md)
addresses the remaining current-native evidence gap. None is a speed promise.

| Worker | Fixed scope | Ownership |
| --- | --- | --- |
| Prefill | Four-output T2048 HC expansion | Isolated helper and complete original L0 probe |
| Decode | T3/20-iteration SIMD32 HC collapse | Isolated helper and eight-block complete-collapse probe |
| Assistant | Matched ordinary/predictable nominal8K A6/A4 | Private frozen corpus and evaluator |

Root alone owns `glm5_next` delegation seams, admission and model gates. Workers
prepare sources/tests in separate files; no shared runtime edits or test-runner
imports. Build and GPU grants remain sequential and explicit. Tests precede
implementation; ReleaseFast builds precede each GPU lock. Foreground QoS,
maximum fans/idle, recorded hashes and complete per-job cleanup apply.

Prefill retains original per-stream four-term FP32 arithmetic and BF16 boundaries.
Its complete original L0 layer includes both HC endpoints and unchanged KDA/FFN,
initial state handling, allocations, endpoint settlement and frees. Three warmups
and eleven alternating pairs follow strict output/all-state proof. No shader
variant follows noise/loss; only a clear complete-layer winner gets matched
long-prefill model ABBA with exact logits/cache/continuation and then HTTP.

Decode retains original ordered four-term maximum/sums, precise exp, division,
20-iteration policy and mixed-output loop. Existing real eight-block512-row
fixtures are present; three adjacent rows form a constructed T3 component, not
captured verifier rows. Check source hashes and actual tensor contract before
use. Prove coefficient/mixed/downstream bits and special values before one
complete eight-block three-warm/eleven-pair gate. No arithmetic tolerance or
subgroup sweep. A clear winner gets root's current-stack8192/192 N2/A6/native
model ABBA and exact serial IDs/valid state before any runtime acceptance.

The assistant worker freezes complete templated bodies and exact IDs before
launch and reuses the archived matched evaluator. Both assistants and all held
references remain resident equally. One shared target prefill per prompt,
independent assistant preparation, complete warm requests and A6/A4/A4/A6
use the current native/packed32 stack. Exact target IDs and valid state plus
gain greater than A6 control drift on both prompts are mandatory. Preparation,
clones, cleanup, acceptance, phases and measured peak are recorded separately.
The115448725504-byte limit and every existing reserve remain; stop on admission
failure. No new conversion or default switch follows this two-prompt gate.

Do not stage private probe sources or unrelated primary changes. Archive/remove
rejected helpers and restore seams. Accepted runtime gains receive appropriate
HTTP2K–16K qualification and selected32K once, then commit/push to Sushi main.
A4 evidence stays consumer-only, with no unmeasured whole-engine memory claim.
Keep A6 default, BF16 compressed MLA, FP32 KDA state/accumulators, original small
tensors and resident embeddings. No precision restoration or64K/128K.
Goals1500 prefill/60 decode remain active and unmet.


## Recorded round outcomes

- **SIMD32 HC collapse:** exact eight-block component, 27.6367% median reduction
  and 11/11 wins. The one matched current-stack 8K/192 model gate preserved all
  IDs and valid state, with 7200 candidate calls per enabled arm and a 2.6475%
  clone/decode/cleanup reduction versus 1.9891% control drift. The optional default-off consumer feature retains
  original outputs/declared shared arrays and adds no runtime tensor plane. The full ReleaseFast suite, quiet-output
  guard and CLI build passed; HTTP qualification remains pending; this is not a new 2K–32K rate table.
  [Component proof](glm5-hc-collapse-simd32-component.md) and
  [model proof](glm5-hc-collapse-model-result.md).
- **Four-output HC expansion:** exact complete-L0 component won 0.4328% and
  11/11 pairs, but the 16K model ABBA was 0.3361% slower amid 1.8959% control
  drift. Rejected; helper/private harnesses removed and expansion delegation
  restored. [Complete result](glm5-hc-expand-prefill-result.md).
- **A4 current-native comparison:** every target ID/valid state matched at the
  fixed ordinary/predictable inputs of 8756/8720 IDs. Ordinary total decode was
  slightly slower; predictable gain was below control drift. Replacement gate
  failed, with no broader ladder or default change. A6 remains default; A4 remains
  an optional consumer format. [Consumer result](glm5-a4-current-native-result.md).

HTTP mode metadata now names `hc_collapse_simd32` and its actual
`hc_collapse_simd32_calls` counter for forthcoming qualification. These model
and component clocks have distinct scopes; none establishes the 1500/60 goals,
new admission limits, or an unmeasured context result.
