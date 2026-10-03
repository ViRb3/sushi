# Optional A4 current-native 8K consumer gate

A6g128 remains the default. The previous matched consumer gate used only
1176/1140 input IDs with native decode off; its modest gains were below control
drift. The separate-boot side ladder cannot resolve current-stack adoption.
Run one bounded matched gate using the existing assistants and archived
`glm53-a4g64-matched-2k-20261003` evaluator. This plan changes no runtime,
checkpoint or default and authorizes no build/model launch by itself.

## Fixed inputs and stack

Use the archived pinned llmprobe 0.6.13 corpus generator: fixed
`[probe fixed-a4-comparison]` nonce, `buildCodeContextWithConstant(32768)`,
ordinary `withRetryBudget`/`RETRY_BUDGET_MS` instruction and predictable
count-from-1-to-200 instruction. Apply the current official chat template once.
Before a model job, freeze both complete message bodies, body hashes, actual
U32 input arrays, ID hashes and counts. These are nominal 8K inputs, with no
repeated WATER fixture or per-arm nonce. Their lengths are not yet measured;
do not claim exactly 8192 or inherit HTTP counts. Confirm the frozen nominal
8K scope before launch; do not retune inputs after performance results.

Use accepted runtime `af51e72f`, native B1/B3 and packed32 on, N2/children4/
verification async4, dense prefill chunk2048/async2, and all other accepted
flags fixed. Greedy sampling explicitly ignores EOS for 192 outputs. No HTTP
ladder, extra prefix, assistant format variant or capture dump is included.

## One loaded matched job

Load the target once and both assistants once. Process one prompt at a time.
Prefill its target prefix once and prepare/settle each assistant context from
identical normal target features. Record target prefill and each assistant's
context-preparation time separately. Retain the same immutable target prefix,
both assistant weights/contexts and native-B1 serial references throughout.
Build the serial 192-ID reference and valid states at 191/192 consumed inputs.

Warm one complete 192-output request per assistant. Then run exactly
A6/A4/A4/A6, cloning the same target prefix and the corresponding prepared
assistant context for every arm. Delivery decode rate is **191 / decode seconds**,
because the first output comes from prefill. Record actual committed-input
counts separately. Timing includes decode endpoint settlement; report clone
and cleanup clocks separately and include them in composed request cost.
The composed cost also includes shared target prefill and that assistant's
preparation; it is not HTTP end-to-end latency. Oracle work stays outside timing.

## Frozen gates and evidence limits

Every measured arm must match all 192 serial target IDs and every valid
MLA latent/pooled/tail plus initialized FP32 KDA/convolution state at its actual
191/192 committed offset. Compare no uninitialized capacity tails. Require
packed32 engagement in shared prefill, native B1 in the serial oracle and native
B3 in measured verification; preserve all output IDs, counters and failures.
Target correctness allows zero differing IDs or valid-state bytes.

For each prompt, let A be the mean of the two A6 decode intervals and B the
mean of the two A4 intervals. Define gain `g = 1 - B/A` and control drift
`d = abs(A6_last - A6_first)/A`. Evidence for a current-stack speed benefit
requires **g > 0 and g > d separately on both prompts**, plus exactness and
admission. Otherwise classify the gate inconclusive or rejected, without a
ladder or threshold change. Report both A4 intervals too; one ABBA is not a
statistical confidence claim. Record every round's draft/verify/replay/commit
cost, accepted drafts, verified rows and committed inputs. Faster drafting
alone cannot establish a whole-engine gain; extra verification rounds count.

Keep the 115448725504-byte memory/wired limit and every existing reserve.
Charge both resident assistants, both prepared contexts, held serial references,
actual cache growth and preparation peaks, native async4's 32MiB bill and
packed32's async2 increment. Record active/peak memory per phase/arm; stop if
the equal-resident setup does not admit. Stored payload differences are not a
measured whole-engine memory saving. Use the accepted libraries, foreground
QoS, maximum fans/idle and one exclusive per-job lock; restore auto/release.

This gate can qualify optional A4 on two fixed current-native workloads.
It cannot switch the default, establish general acceptance quality or quantify
unmeasured resident/whole-engine savings. No precision restoration, conversion,
64K/128K test or 1500/60 claim follows from it.
