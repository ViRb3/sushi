# Singleton expert resources and KDA endpoint round

Start from accepted runtime `e1597cc2`, documentation checkpoint `597fc5b9`.
[Singleton research](glm5-current-singleton-resource-research.md) and
[endpoint research](glm5-kda-all-endpoint-research.md) define two bounded
candidates. Neither predicts a whole-model gain.

| Worker | Scope | Ownership |
| --- | --- | --- |
| 1 | Gate/up singleton resource split | Isolated helper and behavioral probe; minimal original source exports |
| 2 | All T3 FP32 KDA endpoints | Isolated helper and complete verify/commit probe |
| 3 | Current full-bank comparison harness | Private replay/model orchestration and artifact evidence |

The expert candidate keeps original inline ballot pairing and separate singleton
and pair output buffers. A fully written 24-slot mask selects the family inside
the unchanged middle arithmetic. Down and finish remain current. Prove masks,
written gate/up and middle values, then all 516096 saved BF16 routed outputs.
Use the actual first-round inputs and all original E288 banks, original 42-layer
order and four-layer asynchronous settlement. Include preparation, allocation,
classification, all kernels, evaluation and frees in one three-warmup,
eleven-pair gate. Extra buffers are included in the memory bound. No compact-bank
substitute, down split or routing metadata prepass.

The KDA candidate stores three separate 4 MiB FP32 endpoint buffers from the
existing recurrence values. Preserve arithmetic, convolution-tail construction
and partial T1/T2 fallback. Prove chain/fork outputs and all endpoints, every
accepted tail, alias lifetime and release of unselected buffers. Include verify,
normal commit and frees with the recorded hit/miss and length distribution in
one three-warmup, eleven-pair gate. A constructed topology mix is a surrogate,
not captured topology. Add 272 MiB across all 34 tapes to admission if integrated.

Root owns runtime delegation, admission and scheduling. Workers prepare tests
before implementation and do not start GPU work without a sequential grant.
Build ReleaseFast before acquiring the GPU lock. Timed work uses foreground QoS,
maximum fans and the thermal protocol. No concurrent heavy loads.

Reject mismatches, noisy results or losses without another variant. Only a clear
component winner receives one matched 8192-input/192-output N2/A6/native-attention
model ABBA with equal retained references, all rounds and cleanup. Require exact
serial output IDs and valid final states, unchanged acceptance and measured peak.
Then qualify an accepted winner with HTTP bench-only 2K–16K and selected 32K.
Commit and push only accepted runtime; archive and remove rejected prototypes.

A4/group64 remains optional. The independent ladder supports smaller weights and
faster drafting, but inherited A6 controls and unequal 32K inputs do not establish
a replacement. The existing matched comparison is also within control drift;
see [stored assistant evidence](glm5-dflash-affine-storage.md). Keep A6 default
for this round. BF16 compressed MLA, FP32 KDA state and resident embeddings remain
unchanged. Prefill 1500 and decode 60 tok/s remain open.
