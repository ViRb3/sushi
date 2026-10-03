# Temporal KDA and grouped expert fusion round

Start from accepted runtime `e1597cc2`, outcome checkpoint `f06a035c`.
[Prefill research](glm5-next-prefill-research.md) recommends one fixed temporal
working-set experiment; [decode research](glm5-next-decode-research.md) recommends
one grouped middle/down fusion. Both have substantial adverse cost evidence.
Neither is a throughput prediction or a retry of the previous layouts.

| Worker | Implementation | Ownership |
| --- | --- | --- |
| 1 | Grouped middle/down fusion | Isolated helper/probe and minimal original-source exports |
| 2 | T2048 KDA temporal tiles | Isolated helper/probe, fixed 128-token pieces and two pending graphs |
| 3 | Full-bank expert gate and model orchestration | Private replay/model harness using existing fixtures |

Root owns runtime seams, admission and sequential heavy-job grants. Tests precede
implementation. Build ReleaseFast before taking the GPU lock. Quiet jobs use
foreground QoS, maximum confirmed fans and the thermal protocol; no concurrent
heavy load. Capture fixtures already exist: do not recapture or use compact banks.

The expert candidate keeps current gate/up, original ballot pairing, F16 middle
boundary, grouped down reduction and weighted finish. Stage both members on-chip
and use the existing fixed eight output tiles per group. Charge repeated middle
construction and 12 KiB shared storage. Prove middle/down and all 516096 saved
BF16 outputs, including singleton/pair/triple, signed-zero and barrier cases.
Use full original E288 banks, all 42 layers and original async4 settlement in one
three-warmup, eleven-pair complete-chain gate. No tile tuning or extra split.

The KDA candidate keeps all full-T2048 projections, retained cluster, gate/post
and output GEMMs unchanged. Each 128-token piece uses original prework and R4
recurrence. Carry FP32 state; use original raw preceding-three-row convolution
history at interior boundaries. Keep at most two tile graphs with explicit
dependencies and cleanup. Prove prepared pieces, Y, complete layer output and
final state/tail for cold/nonzero state and signed-zero boundaries. One original
layer-zero full-T2048 complete-layer gate includes projections, all additional
launches/waits/state traffic, concat, evaluation and frees. Do not credit a
reservation reduction without lifetime evidence; include any added output
concat and transient state in a conservative bound.

Stop mismatches, noisy results or losses without a new schedule/layout variant.
Only a clear component winner gets a matched whole-model comparison: identical
inputs, equal reference memory, full prefill/decode work and cleanup, original
chunk2048/async2 outside the candidate, and strict logits/token/valid-state
proof. Expert winner uses the frozen 8192/192 N2/A6/native-attention ABBA.
Prefill winner uses matched long-prefill ABBA plus continuation state proof.
Then run HTTP bench-only 2K–16K and selected 32K for an accepted winner.

Commit and push only accepted runtime. Archive and remove rejected prototypes.
A6 remains default; A4/group64 stays optional. Keep BF16 compressed MLA, FP32
KDA state/accumulators, original small tensors and resident embeddings.
Prefill1500/decode60 and stable 2K–32K remain unproven and active.

## Outcome

Both candidates were exact and slower. [Grouped fusion](glm5-grouped-fusion-current-replay.md)
matched all stages and 516096 routed BF16 outputs, but lost every pair:
17.953958 to 20.023042 ms, 11.52% slower. [Temporal KDA](glm5-kda-temporal-result.md)
passed all four final tests, including complete cold/nonzero layer state parity,
but also lost every pair: 14.527666 to 16.246584 ms, 11.83% slower.
The first temporal green attempt stopped on empty cold-state cloning; the
harness-only repair preserved the candidate schedule and arithmetic.
No model arm, retuning or variant followed. Sources were archived and removed,
root seams/bills restored, and accepted runtime remains `e1597cc2`.
