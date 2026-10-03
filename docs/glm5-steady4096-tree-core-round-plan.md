# Fixed steady-4096 prefill and T3 KDA core round

Start from accepted runtime `4fcb541e`, documentation closure `7a104c81`.
The [prefill source study](glm5-prefill-next-research.md) supports one batching
experiment; the [decode source study](glm5-decode-next-research.md) supports
one full KDA-core fusion. Neither has measured performance. Targets remain
1500 prefill and 60 DFlash2 decode tok/s at 2K–32K.

| Worker | Owned implementation | Fixed proof |
| --- | --- | --- |
| Expert grid | Fixed 2048/4096 route geometry and configuration | Every stage/output versus two original 2048 calls; inclusive complete chain |
| Prefill trunk | A6 expansion, MLA batching and KDA cluster at fixed 4096 | Stored coefficient/output consistency and exact geometry bills |
| Tree core | New T3/64-head/128-dimension KDA core helper | Every replay input, chain/fork output and FP32 committed state |

Root owns shared caller/HTTP scheduling seams, independent review, model
harnesses, admission, test exposure, grants and acceptance. Separate file
ownership avoids collisions. Each candidate gets a genuine failing behavioral
test before implementation. ReleaseFast builds precede exclusive GPU jobs;
foreground QoS, confirmed maximum fans, idle and cleanup are mandatory.

## Steady 4096 batching

Keep the first cold chunk at most 2048, preserving the existing early dense
attention boundary. Thereafter use full 4096 only while at least 4096 IDs remain;
use original 2048 and final partial chunks for the remainder. Keep WIN32,
native MMA shape, K order, stored grids and all BF16/F16/FP32 boundaries.
Preserve packed attention's B32/two-inflight cadence and async2 settlement.
Extend only the existing fixed tuned shapes, not arbitrary chunk sizes.
2K/4K requests have no new full chunk; no speed improvement there is assumed.

Combining routing segments can reduce padding and repeated setup, but source
counts do not prove physical traffic or a speed gain. The component uses one
original complete L20 bank/fixture and an explicitly derived 4096-row input;
record segment/window/live/padded counts and label any replicated rows. Compare
one candidate complete chain to two original chains, holding equal input and
reference arrays. Require every retained F16 stage and final BF16 bit, three
warmups and eleven alternating inclusive graph/eval/free pairs. Stop mismatch,
noise or loss without another chunk/window/reader variant. No new route policy.

Retain every original reserve. At the recorded 32K resident baseline, the
prospective 4096 increment is 3,221,225,472 activation bytes, 805,306,368 MLA
permutation bytes and 2,621,440 cluster bytes. The existing full reserve becomes
15,165,685,760 bytes including a fixed 4,194,304-byte route/config/sort
allowance; resident plus reserve
is 109,714,416,888 under the fixed 115,448,725,504 limit. This is source-only
prospective accounting, not executed admission or a measured peak. Charge
additional metadata/config storage and any held model references separately;
no old scratch, cache, expansion or margin credit is permitted.

Only a component winner receives a fixed nonrepeated 16K actual-model prefill
comparison with all tuned path counters. Prove final logits and every valid
BF16/FP32 cache/state against original scheduling. If scheduling changes rounding,
stop exact acceptance and apply the unchanged two-prompt code/prose 16K screen before
performance: 24 declared late-prefix plus 192 baseline-forced predictions per
prompt, mean KL<=.01, max KL<=.15, top1>=95%, mean NLL increase<=.02, new NF=0.
Any such numerical mode remains explicit. Both arms include complete prefill,
settlement and cleanup, equal resident inputs/references, matched ABBA. A win
must exceed control drift before HTTP 2K–16K and selected 32K qualification.

## T3 tree-aware KDA core

Fuse current prework, retained-leaf recurrence and post gating for T3 only,
using the qualified serial head-local organization with tree ancestry. Keep raw
Q/K/V, FA/FB, GA/GB, beta and output projections unchanged. Emit the same five
replay arrays and selected 4 MiB FP32 leaf; commit/replay and retained-state
policy remain unchanged. Preserve original convolution, BF16 normalization/
beta/y boundaries, sigmoid/exp modes, lane/key accumulation, SIMD reductions,
FP32 state and final gated BF16 output. T1/T2 and unsupported inputs stay ordinary.
No canonical recurrence, precision restoration or numerical tolerance is added.

One original complete L0 fixture with declared synthetic activations/nonzero
history proves all intermediate replay arrays, chain/fork output, convolution
and committed FP32 states versus original and serial ancestry, including leaf
hit/miss. Directed finite/nonfinite and cancellation checks follow existing
primitive conventions. New global storage, output handles and retained state
must have explicit ownership; charge all traffic even where outputs stay for
replay. Three warmups and eleven complete-layer/normal-commit alternating pairs
include every projection, graph/eval/free and equal references. Stop mismatch,
new nonfinite, noise or loss; no group-size/core variant.

Only an exact component winner receives frozen ordinary 8756/predictable 8720
192-output N2/A6/native ABBA, current HC and all accepted flags. Every arm must
match serial IDs and valid committed state. Include clone/decode/cleanup and
all rounds; require improvement beyond control drift on both workloads before
HTTP qualification. Do not combine unproven prefill and decode candidates in
one model acceptance run.

## Landing

No teacher recapture, default A4 switch, embedding offload, precision restoration
or 64K/128K. BF16 compressed MLA cache and FP32 persistent KDA state/accumulators
remain. Actual model winners receive full required ReleaseFast checks, final CLI
build, documents and commit/push to Sushi main. Archive rejected compiled sources
before removing only owned changes. Run the next two-researcher round after closure.

## Component decisions

The [expert component](glm5-prefill4096-grid-result.md) passed every retained
stage/output bit and reduced the constructed complete-chain median 8.3626%,
with 11/11 wins. The [trunk proof](glm5-trunk4096-result.md) passed original
L0 coefficients/projections and complete 4096-versus-two-2048 KDA output,
convolution and FP32 state; MLA projection banks were explicitly synthetic.
These qualify the combined schedule for one actual-model gate, not acceptance.

The [T3 core](glm5-t3-kda-core-result.md) passed direct/complete special-value,
serial/tree and ownership proofs, but its 0.6867% median/0.6610% paired gain
and 6/11 wins were inconclusive. No model or repeat followed; compiled sources
were verified/archived before the helper/private probes and caller were removed.
All prefill changes remain separate; accepted runtime is still `4fcb541e`.

## Actual-model exactness stop

The frozen 16384-ID code gate changed final logits and valid prefix states.
The evaluator stopped before candidate continuation, warmups or ABBA; no actual
model speed was measured. All compiled sources and binary remained unchanged.
The existing numerical screen above is the next gate for the same fixed
schedule, after checking for any missed enabled path. No input, threshold or
precision variant is permitted. A numerical pass alone would not establish
performance; matched own-mode references and model timing would still be needed.

Source inspection identifies one numerical boundary: IndexPool NAX eligibility
uses total appended pools, with a minimum of 3584. The 4096 chunk beginning
at offset 10240 appends through 14336, enabling NAX for queries 10240–12287;
the original 2048 chunk at that offset has only 3072 pools and uses scalar
scores. Causal masks and guard code remain unchanged. This is a plausible
mechanism, not a measured cause or a quality result. Preserve the threshold
and existing flags in the fixed screen; no scalar restoration follows.
