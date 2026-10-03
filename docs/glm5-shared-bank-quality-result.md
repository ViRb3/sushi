# Four-query shared bank: rejected by fixed quality screen

The candidate reduced inclusive T2048/16K MLA attention from130.207 to60.846ms
(53.27%,11/11 paired wins), and its native math/production seam were exact
within the chosen retrieval policy. The actual-model quality screen rejected
that policy. All shared-bank source, port/NOTICE and production delegation
were archived and removed; runtime remains `e1597cc2`.

One target load evaluated two fixed nonrepeated16384-ID archives: consumer
code with an early retry-limit record, and engine prose with an early lantern
record explicitly requested first. Native Sushi chat/tokenization produced
the IDs; both arms used exactly those IDs and the same stored A6/EXL3 target.
Native B1 decoding was enabled in both. BF16 compressed MLA/cache/operands,
FP32 KDA state/accumulators and all other accepted flags were unchanged.

Each prompt scored24 declared last-chunk positions and192 baseline-forced
continuation predictions. Layer44 mean-HC captures were normalized/projected
one row at a time; the sampled final row matched actual last-only prefill
logits bitwise in both arms. Continuation row0 used those actual logits,
followed by191 forwards consuming identical baseline greedy token IDs.
CPU f64 stable log-sum-exp/KL retained bounded149,304,320-byte logit storage.
This is same-pack runtime drift, not checkpoint KLD against a lossless teacher.

| Prompt /216 rows | Mean KL | Max KL | Top1 agreement | Mean forced NLL delta |
|---|---:|---:|---:|---:|
| Code | 0.512904 | 23.547608 | 199/216 (92.13%) | −0.042313 |
| Prose | 0.028818 | 2.180192 | 210/216 (97.22%) | +0.027436 |

The predeclared per-prompt limits were meanKL≤0.01, maximumKL≤0.15,
top1≥95%, mean forcedNLL increase≤0.02 and zero new nonfinite values.
Neither prompt passed. Negative code NLL delta on the combined population
does not cancel its distribution/tail failures or establish downstream quality.

| Subpopulation | Mean KL | Max KL | Top1 | NLL delta |
|---|---:|---:|---:|---:|
| Code late24 | 4.502304 | 23.547608 | 15/24 | −0.655222 |
| Code continuation192 | 0.014229 | 0.164406 | 184/192 | +0.034300 |
| Prose late24 | 0.215185 | 2.180192 | 20/24 | +0.186923 |
| Prose continuation192 | 0.005522 | 0.171507 | 190/192 | +0.007500 |

Code continuation alone exceeded mean/max KL and NLL bounds, so removing
late positions would still fail. Late drift appeared at all four pool phases,
including anchors; code phase means were2.333/4.662/3.594/7.419 nats and prose
0.090/0.232/0.419/0.120. This does not isolate one boundary or prove a root cause.
Independent component pool recall was69.91% on1536 nonanchor rows, with21.89%
missed positive index-score mass; recall was diagnostic rather than a quality gate.

Both inputs recorded shared calls0/704 and2101 native B1 calls per arm.
Final offsets were16575, and every one of432 declared rows was preserved.
No invalid/nonfinite rows occurred. Whole-process GPU peak was96,080,475,196
bytes (89.48GiB), under110GiB memory/2GiB cache limits and recommended wiring.
No assistant/speculative or matched prefill performance follow-up was justified.
No threshold adjustment, group-size/bank-policy variant or numeric rerun followed.

Evidence key `glm53-shared-bank-quality-20261003` records HEAD `2e7786a2` plus
hashed uncommitted source, binary `fc7f8068f4ae2dfff9149dfbe85cea8bfb7fe24b7b18392f0f1cef288b18ccfe`,
MLX0.32.3, fixed corpus hash, complete rows/phase summaries and archived prototype.
An earlier technical attempt stopped before scored rows because a private capture
slot was null; its initializer was corrected to the existing capture contract.
Only one complete numeric screen was run. Foreground QoS, exclusive per-job GPU
lock, confirmed maximum fans and cool ten-second idle were used. The final exit1
was `SharedBankQualityRejected`; lock released and fans restored automatic.
