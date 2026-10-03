# Fixed route-six prefill: rejected after quality screen

The fixed six-route policy passed its independent policy/retained-arithmetic
proofs and reduced one actual L20 complete-chain median by 20.23%, with 11/11
wins. The subsequent fixed quality screen rejected both inputs. No runtime
policy, default change or throughput trial is accepted.

Eligible input is original BF16 B1/T2048/H4096, E288/I2048/n36/MCG/W12,
normalized top-eight routing and clamp10. Select the six greatest actual output
weights, with earlier-original-slot ties; emit survivors in original slot order.
Original-order FP32 sums, division and multiplication target original total mass.
The declared invalid/zero/nonfinite/overflow whole-batch predicate is freshly
evaluated before use; a failed predicate returns null for original eight-route
fallback, without changing original inputs. This adds a GPU wait and is charged;
there is no claim that the old async2 behavior is preserved.

`tryMoe` takes original eight-route arrays and internally prunes/consumes six.
The six-route consumer reuses the accepted transposed WIN32 NAX source, descriptor,
K order, F16 intermediate boundaries, Hadamard/SwiGLU and FP32 finish. No stored
weight, cache precision, small tensor, decoded-bank cache or kernel geometry
changes. Partial chunks, decode/verification and unsupported configurations
keep the original eight-route path. New policy allowance is 128 KiB per pending
layer; all old reservations remain, with no source-savings credit.

The fixture is the existing actual T2048 L20 X/indices/scores with all nine
original full-E288 bank tensors. CPU derivation retains 82.1275% original routing
weight mass on average (minimum 76.4231%); that semantic risk remains. Independent
FP32-order policy tests matched all 12,288 selected IDs and scores, including ties
and scaling, and directed invalid/zero/nonfinite/fallback tests passed. Retained
gate/up/middle/down/final stages matched 134,217,728 F16/BF16 values against the
original consumer at the same six routes. All 8,388,608 final BF16 values matched
an independent generic six-route oracle. Four focused tests passed, exit 0.

Drift from the current eight-route result is large: 8,347,116 of 8,388,608 BF16 output
values differ, relative L2 is 0.129162978 and maximum absolute difference 2.0.
Exact retained arithmetic does not make this policy lossless or establish quality.

Three warmups per arm preceded eleven alternating pairs on fresh graphs.
Timing includes validity wait, policy compaction/normalization, sorting,
metadata/inverse, preparation, all three unchanged transposed GEMMs, middle,
finish, allocation, endpoint evaluation and frees. Current eight-route median
was 17.351583 ms; six-route median 13.841458 ms, 20.2294% lower. Paired median reduction
was 20.2743%, range 19.5443–20.9834%. Candidate engagement was 0/1 per control/candidate.
Both output references and original full banks remained held equally.

| Pair | Current eight ms | Fixed six ms | Change |
| --- | ---: | ---: | ---: |
| 1 | 17.369000 | 13.879875 | -20.0882% |
| 2 | 17.328958 | 13.841458 | -20.1253% |
| 3 | 17.359791 | 13.806458 | -20.4688% |
| 4 | 17.485083 | 13.816125 | -20.9834% |
| 5 | 17.405583 | 13.841250 | -20.4781% |
| 6 | 17.410292 | 13.853834 | -20.4273% |
| 7 | 17.351583 | 13.830750 | -20.2911% |
| 8 | 17.342709 | 13.852792 | -20.1233% |
| 9 | 17.305959 | 13.923625 | -19.5443% |
| 10 | 17.323083 | 13.875416 | -19.9022% |
| 11 | 17.326959 | 13.814042 | -20.2743% |

Warm/timed-window peak was 621,287,720 bytes above the equally-held fixture and
reference/proof outputs. It includes the fresh chains' owner arrays and results;
it is not a model peak or a proof that existing reservations can be reduced.

The first green attempt compared against an uninitialized reference dispatch:
its log explicitly showed Mul1/W16 instead of this pack's MCG/W12. It failed the
first retained gate stage before timing. That is invalid reference evidence,
not a quality result. The authorized repair added only private
`setDecodeParams(MCG,W12)` before the first reference call; helper/shader/policy
hash remained identical. The corrected log explicitly records MCG/W12 and all
strict stages passed. No tolerance or candidate arithmetic changed.

Artifact `glm53-prefill-route6-20261004` preserves red stub, original compile
packaging failure (tuple cleanup), failed wrong-reference source/log/hashes,
corrected command/binary/runtime/source hashes, policy/stage proofs, every pair,
TOP8 drift, peak and thermal/fan records. Helper SHA256
`f2fbf140b6bb508cbdba536b3d7bb15c29e159bff7c93911a934200502768ec0`;
qualified binary SHA256 `a5812792c9c7dd8959ad4ba4f3ed4a1a5fec25a2ff21e25d46b9fe194884eea4`.
Red PID 92071 exited 1 as intended; failed green PID 93609 exited 1; qualified green
PID 94247 exited 0. Foreground QoS, explicit accepted-library environment, exclusive
per-job lock and maximum fans/idle were used. All jobs ended, locks released and
fans automatic. The final quality disposition is recorded below.


## Final quality rejection

The [fixed quality result](glm5-route6-quality-result.md) evaluated both original
nonrepeated 16384-ID code/prose inputs, all 24 declared late-prefix samples and
192 baseline-forced continuation predictions per input on the current native/
HC stack. Every one of 432 rows was scored; invalid rows and new nonfinite values
were zero. Per-input limits remained mean KL≤0.01, max KL≤0.15, top1≥95% and
mean forced NLL increase≤0.02.

| Input /216 rows | Mean KL | Maximum KL | Top1 matches | Mean NLL delta |
| --- | ---: | ---: | ---: | ---: |
| Code | 0.3340016 | 18.89898 | 197/216 (91.20%) | −0.09398 |
| Prose | 0.05216336 | 2.40919 | 207/216 (95.83%) | +0.06619 |

Both inputs failed the fixed KL bounds; code also failed top1 and prose failed
NLL. Negative aggregate code NLL does not offset the distribution/tail failures.
There was no filtering, threshold relaxation or route-policy variant.
Candidate engagement was 336 wide-prefill calls per prompt, controls 0, and both
continuation tails 0. Last capture and actual last-only logits matched within
each mode; native B1 engagement was 2101 per arm. These confirm evaluated scope,
not an approximation-quality pass or model speedup.

Evidence `glm53-route6-quality-20261004` retains every scored row and both prompt
summaries. The job ended and released the GPU/fans. Root restored Moe delegation,
HTTP bill and the reference-project proof alias. This worker hash-verified,
archived and removed only its helper, private actual-L20 probe and private root;
failed reference evidence and every component pair remain. No model throughput
or HTTP follow-up is justified. Accepted runtime is unchanged.
