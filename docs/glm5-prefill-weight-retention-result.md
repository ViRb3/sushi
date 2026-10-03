# Four-bank retention attribution result

Do not implement the fixed 8.5 GiB runtime cohort from this evidence. Materialized
BF16 banks preserved arithmetic but did not produce repeatable four-projection
savings. No target load, runtime cache, geometry variant or model gate follows.

The private probe used the original layer-zero Q/K/V/output A6/group128 banks
from the existing 12-tensor fixture. Frozen synthetic BF16 activations were
`[1,2048,4096]` shared by Q/K/V and `[1,2048,8192]` for output, seeds 61201 and
61202. This is a four-projection attribution experiment, not a recurrent KDA
layer or HTTP workload. Both arms held the same 268,435,456-byte materialized
reference cohort before all timed samples. Current fresh expansion was compared
with the same BF16 GEMMs using those held banks; original transpose views,
operands, geometry and native arithmetic stayed unchanged.

All 134,217,728 coefficients matched independent A6 reconstruction/FP32 scale
and bias arithmetic/BF16 rounding. All 58,720,256 projection output BF16 bits
matched. The zero-output red stub was rejected before timing; qualified green
passed both tests, exit 0. Fresh expansion counters were four per sample and
resident counters zero. No original stored or small tensor changed.

Three warmups per arm preceded eleven alternating pairs. Each timed sample
included all four projection graphs, current dequantization where applicable,
allocation, native GEMMs, evaluation and frees. Fresh median was 9.765625 ms;
resident median was 9.801000 ms, **0.362% slower**, with only 4/11 wins. Paired
median was 0.306% slower, or negative 0.029625 ms saved. Individual pair swings
exceed the median movement.

| Pair | Fresh ms | Held-bank ms | Held-bank change |
| --- | ---: | ---: | ---: |
| 1 | 9.703084 | 9.968667 | +2.737% |
| 2 | 9.682000 | 9.711625 | +0.306% |
| 3 | 9.871958 | 9.882542 | +0.107% |
| 4 | 9.765625 | 9.702542 | -0.646% |
| 5 | 9.741583 | 9.821375 | +0.819% |
| 6 | 9.665000 | 9.780875 | +1.199% |
| 7 | 10.127625 | 9.609250 | -5.118% |
| 8 | 9.797667 | 9.874041 | +0.780% |
| 9 | 9.905667 | 9.839917 | -0.664% |
| 10 | 9.690917 | 9.748334 | +0.592% |
| 11 | 10.102084 | 9.801000 | -2.980% |

Separately, the first four-bank preparation measured 0.017959 ms graph
construction, 11.940458 ms evaluation and 0.003750 ms release: 11.962167 ms total.
Its extra peak was exactly 268,435,456 bytes. This cold first invocation may
include kernel compilation and initial source access; it is perturbed standalone
attribution, not the recurring dequantization cost within normal prefill. Do not
multiply it across chunks or subtract it from HTTP times. The full warm/timed
window recorded 385,875,968 extra peak bytes above the equally-held banks, original
compressed weights and frozen inputs. That scope includes fresh expansions and
projection results and does not describe a 136-bank model cohort.

Artifact `glm53-a6-retention-attribution-20261003` retains the original manifest
and source tensor hashes, compiler commands, qualified red/green source and
binary/runtime provenance, cold preparation, every pair/counter and fan records.
The first red compiler invocation found a loader import alias typo; that log is
preserved. Red PID 61394 exited 1 at its intended output parity failure. Green
PID 61620 exited 0. Both jobs used foreground QoS, explicit library environment,
maximum fans/required idle and exclusive GPU lock, then released/restored fans.
Probe SHA256 `1770f4bcaa7bf0ea0ba6d1727294687dd0e8728054cd9a136e6094ab2de10c62`.

The saved-work counts in [the research](glm5-prefill-weight-retention-research.md)
remain valid, but fewer expansion calls did not establish material complete-chain
savings here. The measured cause of that result is not isolated further. No
smaller-cohort search, preparation discount or memory-limit increase is justified.

The private probe/root are archived and removed. Runtime code remains unchanged.
