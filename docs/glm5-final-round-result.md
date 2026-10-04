# Final bounded optimization round

The final round closed with no optimization adopted. Long-pool scoring is rejected
by the frozen code quality bound. Expert pairing passed tiny scheduling/ownership
characterization, but real E288 Model, assistant-lifetime and performance gates
remain unproved and are deferred. The previously qualified runtime is retained.
No additional model job, timing, benchmark ladder or candidate search followed
these two gates.

At the owner's final request, the previously qualified fast flags became
default-on with explicit `0` opt-outs. This changes default selection, not the
measured kernel recipe. See [current defaults](glm5-benchmark-http.md#current-qualified-defaults).

## Long-pool quality

The recorded real P8394/T811 component gained 9.65% with 11/11 paired wins. Its
subsequent full-target screen kept the two frozen nonrepeated 33,579-ID code/prose
inputs, original 2048 chunks and final 811 at 32,768. Only prefix cap 8192→8448 changed;
pairing was off and qualified TF32=1/nativeB1/B32/HC/BF16-cache/FP32-state math stayed.
Each prompt scored 24 declared late rows and 192 baseline-forced predictions; the 191
continuation forwards used original scalar scoring. All 432 scores were finite.
This is same-pack runtime drift, not teacher or quant-pack KLD.

The unchanged per-prompt population gate is meanKL<=.01,maxKL<=.15,top1>=95%,
mean forcedNLL increase<=.02,newNF=0 across all 216 rows. Code fails maximum KL;
prose passes. Late/tail subgroups do not replace that gate.

| Prompt/group | Rows | Mean KL | Max KL | Top1 | Mean NLL delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code, all | 216 | 0.00870123 | 0.34538801 | 211/216 (97.685%) | -0.02377797 |
| Code, late | 24 | 0.04871707 | 0.34538801 | 21/24 (87.500%) | -0.20867744 |
| Code, continuation | 192 | 0.00369925 | 0.05818241 | 190/192 (98.958%) | -0.00066554 |
| Prose, all | 216 | 0.00526153 | 0.08421481 | 209/216 (96.759%) | -0.00357062 |
| Prose, late | 24 | 0.00843683 | 0.05288641 | 21/24 (87.500%) | -0.05339314 |
| Prose, continuation | 192 | 0.00486461 | 0.08421481 | 188/192 (97.917%) | +0.00265719 |

Both prompts recorded old/new long calls 0/561, total IndexPool calls 14,080/14,641,
and 2,101 nativeB1 calls per arm; forced offset ended 33,770. One target loaded,
without an assistant. Loaded active 93,535,640,312 and peak 96,484,963,900 bytes plus
full no-credit bill 15,972,578,304 (including 2 GiB allocator cache and all old
reference/scratch/CPU-logit allowances) reached 112,457,542,204 under fixed
115,448,725,504 memory/wired limits. The last-layer capture was diagnostic.
PID 93941 completed with expected exit 1; all 432 rows were retained. No tail/full
prefill timing, relaxed threshold, precision restoration or numeric retry followed.

## Tiny expert-pair characterization

The existing nonzero dense tiny Model reached offset 6144 after original 2048 cold
prefill and two 2048 halves. Last logits and every valid state matched separate
forwards; ordered captures at layers 0/3 matched all 1,048,576 nonzero values.
Initial persistent state had 32,970 nonzero words. Unsupported invocations declined
before mutation. Injected checked-call 1/late 2641 errors marked the child failed,
blocked reuse and left offset 2048 unpublished while the source state/hash stayed
immutable. One mechanical parameter-shadow rename preceded a single rebuild;
no operator, constant, input or allocation changed. PID 96812 passed all 8 tests,
exit 0. Expert helper calls 0 are correct for this fixture, which does not exercise
E288 joining or an assistant.

[Complete original L20 evidence](glm5-expert-pair-layer-result.md) remains an exact
2.47% component win. It does not close the real 16K Model helper/capture/state/native64
gate, five ordered production captures, two original assistant context appends,
actual memory admission or whole-request performance. That larger adapter is
explicitly deferred at the user-requested stopping boundary, not adopted.

## Preservation and closure

Both jobs used ReleaseFast, foreground taskpolicy with explicit accepted libraries
and TF32=1, confirmed maximum fans/idle and separate per-job locks. Each returned
fans to auto, released its lock and left no live child. Both 193 before/after source,
library and binary checks matched. Artifact `glm53-final-round-20261004` retains
complete results, every declared row, protocols, source/binary/library hashes,
packaging failures, telemetry and cleanup. Both complete archives hold 188 verified
compiled-source copies; tiny's `closure-archive-manifest.json` additionally verifies
its result/provenance/posthash/cleanup and binary before runtime-source cleanup.

Long binary SHA256:
`36a1b429e5426c923f22df195e56737a39287c700439ca86d2452afe948c7d1b`.
Tiny binary SHA256:
`d1fc9c5aeb00e63e0da585fee24daa0ac8f341d632fcb900498028b5b7158ebe`.
No accepted runtime gain follows from these results. The larger 1500/60 throughput
goal remains unmet; the bounded final round ends here.
