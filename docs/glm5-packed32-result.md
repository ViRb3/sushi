# Fixed packed B32 component result

The fixed B32 candidate passed the directed fixture gate and reduced complete
attention time by 15.38%. The matched 16K model gate then reduced prefill latency
by 4.93%, with exact logits, valid state and continuation. It is accepted as an
opt-in with a larger conservative memory allowance. HTTP qualification follows;
the component percentage is not a prefill-throughput percentage.

The candidate keeps two original T16 selector calls per B32 call, concatenates
ordered IDs, and changes the native batch dimension only. The native
Q64/head1/D512/K2051 arithmetic and gather body are unchanged. B17–B31 decline;
remainders use original fragments no larger than 16. Two graphs remain pending.

| Evidence | Result |
| --- | --- |
| Ordered selected IDs, actual T2048 at 16K history | 4,200,448 exact |
| BF16 outputs at T33, T49 and T2048 | 69,795,840 exact |
| Directed B32 versus two B16 safety outputs | 1,048,576 exact |
| Empty, causal, invalid/future nonfinite, head order and final slot | Passed |
| Valid selected NaN/Inf remains observable | Passed |
| Scoped policy, geometry and ragged scheduling | Passed |
| Full fixture suite | 5 tests passed, exit 0 |

After three warmups per arm, eleven alternating pairs measured the entire call:
original selectors, ID concat, gather, native attention, two-graph settlement,
result concat, final evaluation and frees. Current B16 median was 130.189583 ms;
B32 median was 110.162709 ms. All eleven pairs favored B32, with paired median
15.24% faster and individual reductions 13.72–16.45%. Each arm still issued 128
T16 NAX selector calls; native packed calls fell from 128 to 64. The warm/timed
window recorded 3,584 original selector calls.

The directed candidate peak above the loaded fixture plus held control output
was 145,785,324 bytes, including candidate output and owner graphs. Reserve
128 MiB per B32 graph, 256 MiB per pending attention layer and 512 MiB for two
pending layers. Root admission red observed the expected 256 MiB missing
increment; green passed with the increment. Original reserves remain present.

Provenance: artifact `glm53-packed32-20261003`, fixture
`glm53-prefill-cadence-16k-20261003`, source checkpoint `2749c30d` plus recorded
WIP hashes. Helper SHA256
`2edc80784af62259e7bb75e1af5c9ff985915d802ea3372991b9d4b164389e37`;
probe SHA256
`d98deed09262f89f46fd9e0e3571ddfc9f886fff8609f05d63cea6c8ae5c1c08`.
The artifact retains exact compiler command, binary/runtime/source hashes,
flags, raw pairs, proof counts and fan/thermal records. Red compiled and failed
at the intended absent B32 behavior; green compiled and passed. GPU lock was
released and fans restored after each job. No model capture, new scorer,
precision change or batch sweep was used.

## Matched model gate

One target load, without an assistant, retained equal 16K prefix and native
64-token endpoint references before all A/B/B/A arms. Original 2048-token
official-template smoke plus neutral WATER filler IDs were repeated eight
times to exactly 16384; this is not a pure code workload. Outer chunk2048,
async2, accepted projection/cluster/grid flags and native decode mode were fixed.

| Arm | Complete prefill seconds |
| --- | ---: |
| B16 A | 22.379832042 |
| B32 B | 21.459642251 |
| B32 B | 21.510691333 |
| B16 A | 22.816540708 |

Mean latency fell 22.598186375 to 21.485166792 seconds, 4.9253% lower, versus
1.9513% control drift. Mean input rates were 725.01 and 762.57 tok/s. Timing
includes request/input construction, all forward work, endpoint evaluation,
input/output frees and unchanged-prefix cleanup. Bitwise oracles and the
continuation fork/work/frees remain outside prefill timing.

All four arms matched last logits and every valid MLA prefix/pool/tail plus
initialized FP32 KDA/convolution state. All 64 continuation IDs and final state
matched, after consuming every emitted ID to offset16448. Each continuation
engaged 704 native B1 calls and zero B3 calls. Packed32 calls were0/4928/4928/0.
Seven focused tests passed, exit0. This prefill gate does not measure speculative
decode speed or qualify 32K by extrapolation.

Active-start memory was exactly94227159800 bytes in all four arms. Prefill peaks
were96428110904/96428160056/96428176440/96428129080 bytes. Those allocator peaks
do not remove the conservative 256 MiB async2 increment. No original tensor,
cache precision or small weight changed. Artifact
`glm53-packed32-model-16k-20261003` retains every source/binary/flag hash, arm,
oracle, continuation ID and thermal record. Foreground QoS, per-job lock,
confirmed maximum fans and required idle were used; process ended, lock released
and fans automatic. Full ReleaseFast suite and local-path/quiet-runner guards
also passed before the model gate.
