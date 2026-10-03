# Optimized N3 bundle: rejected after matched model gate

The complete four-row bundle passed exact native-target token and valid-state
checks, but its 0.56% aggregate decode gain was smaller than 1.46% control
drift. No wider policy, T4 helper/probe, admission change or default was retained.
Source was archived and restored to accepted runtime `e1597cc2`.

| Fixed 8192-ID workload | N2 | Optimized N3 |
|---|---:|---:|
| Delivered decode tok/s, aggregate | 40.22 | 40.44 |
| Rounds per run | 81 | 68 |
| Accepted drafts per run | 111 | 123 |
| Delivered outputs per round | 2.358 | 2.809 |
| Draft ms/round | 5.614 | 5.710 |
| Verify ms/round | 50.897 | 61.405 |
| Replay ms/round | 1.248 | 1.435 |
| Commit ms/round | 0.820 | 0.844 |

The two N2 decode intervals were 4.714993 and 4.783933 s; N3 intervals were
4.698340 and 4.747285 s, in N2/N3/N3/N2 order. Both policies delivered
192 output IDs, scored as 191 delivered forwards after the prefill token.
N2 committed 192 target inputs and N3 committed 191; these are separately
recorded state offsets, not different delivered output budgets. Fewer N3
rounds did not overcome its roughly 20.6% higher verification cost per round.
The small total difference does not justify another variant or qualification ladder.

One target and A6g128 assistant were loaded, with the same immutable prefix and
assistant context. The fixed input repeats the existing official-template
512-token prompt plus neutral WATER filler four times to reach 8192 IDs; it
is a constructed context workload, not pure code or a representative corpus.
Native attention was enabled in both policies, BF16 compressed MLA cache and
FP32 KDA state/accumulators retained, with accepted prefill/verification flags.

Before warmups and timed arms, one native B1 serial execution produced all
192 reference IDs and valid final-state references for 191/192 committed
inputs. All four measured arms matched every output ID and every valid MLA,
pooled/tail, BF16 convolution and FP32 recurrent state byte at their actual
offsets. Oracles ran outside timing; endpoint request/context frees were
measured separately and included in decode intervals. Warmup covered both
policies; no cold or losing rounds were excluded.

Both N2 arms engaged 880 B3 calls, both N3 arms 748 B4 calls. QKV hoist, MLA
query/value broadcast and retained-leaf counters were positive in every arm.
All starting active allocations were exactly 95,548,572,664 bytes; two serial
reference states remained held throughout. Peaks were 96.315 GB for N2 and
96.319 GB for N3. Native B4 reserved 16 MiB/layer and 64 MiB at async4 under
the unchanged 256 MiB branch limit.

The component gates preceded this model decision: [T4 KDA](glm5-dflash-t4-kda.md)
was exact and 6.92% faster with 11/11 paired wins; [T4 MLA](glm5-dflash-t4-mla.md)
proved exact B1/B4, M1 broadcasts and horizon3, with a modest/noisy 4.14%
B4 component reduction. Those wins did not establish a whole-policy gain.

Evidence key `glm53-n3-matched-8k-20261003` records `88f19e22` plus hashed WIP,
ReleaseFast evaluator/runtime provenance, flags, exact IDs, phase rows, ABBA
intervals, oracle results and archived source. MLX0.32.3, foreground
`taskpolicy -a`, exclusive GPU owner `glm53-n3-matched-8k`, verified maximum
fans and ten-second idle from 47.63°C. Seven tests passed, exit0; lock released
and fans automatic. Full ReleaseFast suite/CLI and quiet/local-path guards
also passed before the model gate. No 32K/64K/128K follow-up was run.
