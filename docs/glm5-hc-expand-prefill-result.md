# Fixed four-output HC expansion rejection

The strict prefill candidate passed exact proofs and a small complete-L0 paired
win, then failed to establish a model speed gain. Reject the runtime candidate;
no further variant, rerun or HTTP qualification follows.

At B1/T2048/D4096, one thread owns four independent output streams and shares
residual/branch scalar loads. Original ordered j=0..3 FP32 contractions, separate
post products, final product-plus-value additions and BF16 stores remain intact.
Native GEMMs, KDA recurrence, dense FFN, small tensors, cache precision, output
storage and async policy are unchanged. Grid is T2048*4096, group256, with no
threadgroup allocation or extra tensor plane. Other geometries decline.

The fixture contains all 40 original L0 tensors: reused KDA banks plus 17 original
HC/norm/dense-MLP tensors, with stored dtypes/header offsets/payload hashes.
Input is a constructed frozen BF16 `[1,2048,4,4096]` tensor, seed 46201. Initialized
BF16 convolution/FP32 KDA state reuses seeds 44/45. It is a component fixture,
not captured full-model activations.

All 67,108,864 raw BF16 values matched, including original HC coefficients,
full BF16 classes, cancellation/signed zero/nonfinite values and the independent
rounded-contraction regression. The complete L0 endpoint matched 33,554,432 BF16
values, 73,728 convolution-tail values and 1,048,576 FP32 state values. Strict
geometry, F32 fallback and scoped policy passed. Red failed at the intended
missing endpoint; green passed all 3 tests, exit 0.

Complete timing includes both HC collapses/norms, original KDA projections,
prework/R4/state/gate/output, original dense FFN, both HC expansions, state
handle construction, endpoint/state settlement and every free. Fixture loading,
immutable parameter preparation and both held reference Results precede all
arms equally. Three warmups per arm preceded 11 alternating pairs. Median
28.850333→28.725459 ms is 0.4328% lower; paired median is 0.3676% lower. Every pair
favored the candidate; reductions ranged 0.0545–0.8570%. Expansion engagement
was 0/2 per arm. No isolated expansion time is substituted for complete work.

| Pair | Original ms | Four-output ms | Change |
| --- | ---: | ---: | ---: |
| 1 | 28.832583 | 28.688208 | -0.5007% |
| 2 | 28.817958 | 28.580125 | -0.8253% |
| 3 | 28.867541 | 28.656000 | -0.7328% |
| 4 | 28.925042 | 28.818709 | -0.3676% |
| 5 | 28.816708 | 28.725459 | -0.3167% |
| 6 | 28.798833 | 28.783125 | -0.0545% |
| 7 | 28.883167 | 28.751375 | -0.4563% |
| 8 | 28.884292 | 28.813417 | -0.2454% |
| 9 | 28.961666 | 28.713458 | -0.8570% |
| 10 | 28.850333 | 28.767250 | -0.2880% |
| 11 | 28.784666 | 28.721583 | -0.2192% |

Candidate proof peak was 1,031,192,612 bytes above the loaded fixture plus held
control output/state. It includes all complete-layer owner arrays, candidate
output/state and proof operations; it is not an isolated kernel footprint.
The 2 GiB directed guard passed. Existing model reserves remain necessary.

Artifact `glm53-hc-expand-prefill-20261003` records commands, fixture headers,
source/binary/runtime hashes, all proofs/pairs/peaks and fan/thermal records.
Checkpoint 99547a26, helper SHA256
`83dd1a5c08d1531edf032981b3c0860e8cf3170073664884f039babd0a57ed04`;
binary SHA256 `75b22ac4165a13f733143e5fbfe8c6853839e789ea3950ef8f99bcfa898dc517`.
Red PID 74765 and green PID 75359 both terminated; foreground QoS, explicit
library environment, maximum fans/idle and per-job lock were used. GPU locks
were released and fans restored. The subsequently completed matched model gate is recorded below.


## Matched 16K model disposition

The one-loaded-target ABBA kept packed32 enabled in every arm and explicitly
disabled the separate HC collapse candidate. It toggled only HC expansion.
Frozen original 2048 IDs were repeated eight times, with chunk 2048/async2,
no assistant, original BF16 cache/FP32 KDA state, and native 64 continuation.
Both HC-off prefix and native 64 endpoint references were established and held
before every arm. Timed construction, all input/forward/output work and measured
prefix cleanup stayed inclusive; strict bit oracles and continuation fork/work/
cleanup stayed outside prefill timing.

| Arm | Expansion | Prefill including cleanup (s) | New calls |
| --- | --- | ---: | ---: |
| 1 | HC off | 21.158409959 | 0 |
| 2 | HC on | 21.373151625 | 720 |
| 3 | HC on | 21.488398500 | 720 |
| 4 | HC off | 21.559556542 | 0 |

HC-off mean was 21.358983251 s; HC-on mean was 21.430775062 s: **0.3361% slower**,
versus 1.8959% control drift. The small component gain did not establish a model
benefit. No causal attribution to one stage is inferred from this ABBA.

All four last-logit arrays, every valid MLA prefix/pool/tail and initialized
FP32 KDA/BF16 convolution state matched exactly. All 64 native continuation IDs
and final states matched at offset 16448. New expansion calls were 0/720/720/0;
packed32 calls were 4928 in every arm, with 704 native B1 and zero B3 calls per
continuation. This is functional evidence, not runtime performance acceptance.

Active-start memory was exactly 94,227,208,952 bytes in all four arms. Prefill
peaks were 96,428,241,976 /96,428,225,592 /96,428,241,976 /96,428,241,976 bytes.
The private checked no-assistant ledger retained 7,301,431,296 request/growth
bytes and every old reserve. Memory and wired limits were both 115,448,725,504
bytes; no reserve reduction or limit increase was used. Cleanup times were
1.108042 /1.005542 /1.031750 /1.041000 ms and were included above.

Artifact `glm53-hc-expand-model-16k-20261003` retains all arms, progress,
engagement, fixed input IDs, complete source/runtime/binary hashes, limits,
proofs and fan records. Model binary SHA256
`53cc0c9796433f1d97d06ec41b8ee89669bed047cac3bc560afb1a73405c8493`.
ReleaseFast build passed. PID 79033 exited 0; GPU lock was released and fans
restored to automatic. Root removed only the expansion delegation. The helper,
private component probe/root and model evaluator/root were hash-verified,
archived and removed; other HC candidate files remain under their owner's control.
No runtime expansion change is accepted.
