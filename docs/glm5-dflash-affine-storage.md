# GLM DFlash stored affine assistant formats

The native diagnostic adapter accepts one uniform stored affine format per
assistant: **A4/group 64, A6/group 128 or A8/group 128**. Its labels are
`A4g64`, `A6g128` and `A8g128`. The existing BF16 assistant remains supported,
and small retained BF16 linear matrices may accompany an affine assistant.
A6/group 128 remains the selected default. A4 support is an explicit optional
consumer capability, not an adopted precision or performance replacement.

## Consumer contract

Each affine linear uses a two-dimensional U32 packed weight array and BF16 scale
and bias grids. The grids must have identical shapes and match the weight's
output rows. Packed geometry must satisfy:

`packed_columns * 32 == scale_columns * stored_group_size * stored_bits`

Every affine linear must match the assistant's stored bit rate and group size.
Only the three named tuples are admitted; mixed rates/groups, missing biases,
invalid grid shapes and inconsistent packed widths fail with
`UnsupportedGlmDraftStorage`. Retained dense linears keep their existing BF16
weights and empty scale/bias handles. Validation and labeling do not rewrite,
cast or replace any tensor. Other retained tensors are unchanged.

Existing inference kernels already consume A4/group 64. The adapter extension
changes only stored-format validation, the geometry equation and the reported
label. No kernel, target computation, cache precision or assistant default is
changed. BF16 compressed MLA and FP32 KDA state remain the target contract.

The behavioral test first failed on the old validator's rejection of A4/group
64, then passed after the extension. It covers all three tuples, correct labels,
unchanged weight/scale/bias handles, mixed-format rejection and mismatched or
inconsistent grids. The focused ReleaseFast filter passed eight tests. Evidence
key: `glm53-a4g64-consumer-storage-20261003`.

## Matched drafter comparison

One loaded target and both assistants were resident in a single process. Two
fixed inputs used the pinned llmprobe 0.6.13 archive corpus with a fixed nonce:
an ordinary code-writing instruction and a predictable counting instruction.
Tokenization produced **1176 and 1140 input IDs** respectively. These are the
actual tested lengths, smaller than the planned approximate 2K scope, and are
not a 2K–16K ladder or a replacement for the side study's context table.

Each prompt's target prefill and captured features were computed once. Both
assistant contexts used those same captures; every arm cloned the same target
prefix and its corresponding prepared assistant context. Both assistants and
serial reference states remained resident throughout the A6/A4/A4/A6 sequence.
The target's native decode-attention mode was explicitly off in all arms; actual
native B1/B3 counters were zero. Sampling was greedy and EOS was ignored to
produce equal work: 192 output IDs per arm. Delivery decode rates use **191
outputs after the first prefill token**; all arms actually committed 192 target
input tokens, recorded separately.

Aggregate decode rates below divide total delivered tokens by the sum of the
two measured decode intervals for each assistant. Phase means are divided by
actual round counts.

| Fixed workload | A6 / A4 decode tok/s | Aggregate gain | Rounds per run A6 / A4 | Accepted drafts per run A6 / A4 |
|---|---:|---:|---:|---:|
| Ordinary, 1176 input IDs | 48.19 / 48.92 | 1.53% | 68 / 68 | 124 / 124 |
| Predictable, 1140 input IDs | 50.79 / 51.28 | 0.96% | 64 / 64 | 128 / 128 |

| Fixed workload | A6 / A4 draft ms/round | Draft reduction | A6 / A4 verify ms/round | Committed inputs/round |
|---|---:|---:|---:|---:|
| Ordinary | 5.492 / 4.857 | 11.56% | 50.999 / 50.871 | 2.824 / 2.824 |
| Predictable | 5.566 / 4.938 | 11.28% | 51.504 / 51.651 | 3.000 / 3.000 |

Ordinary ABBA decode intervals were 3.889957 / 3.884421 / 3.923539 / 4.037258
seconds; predictable intervals were 3.618096 / 3.584056 / 3.865959 / 3.903339.
A6 controls drifted 3.79% and 7.88%, larger than the aggregate gains. Equal
acceptance and faster drafting establish a functional smaller drafter on these
inputs, but these modest total-rate differences do not establish a robust
replacement for A6. No broader A4 context ladder or default adoption followed.

The shared target prefill was 2059.21 ms ordinary and 1296.29 ms predictable.
Assistant context preparation was A6/A4 8.02/6.97 ms and 9.21/7.62 ms.
The evaluator's composed request-generation rates, including that shared target
prefill, assistant preparation, clone and decode, were 31.84/32.16 and
37.90/38.18 output tok/s. Those composed rates are not HTTP end-to-end latency:
networking, token-text serialization and client delivery are outside this test.

All eight measured output sequences matched the same target serial 192-token
reference, and every valid final MLA/pooled/tail and initialized KDA convolution/
FP32 recurrent cache matched at the actual committed offset. Oracle work was
outside the timing intervals. The standalone run passed seven tests with exit
zero. Source baseline `39f8693d` plus the uncommitted consumer extension is
recorded by exact adapter/evaluator/runtime/binary hashes; binary SHA-256 is
`04a2b41ff8b7577e8626a9e03e2bafc5ff10d7e9ff1fc0fafa434693916565f0`.
Foreground QoS, exclusive GPU lock, verified maximum fans and ten-second idle
from 47.12°C were recorded. The lock was released and fans returned automatic.
Evidence key: `glm53-a4g64-matched-2k-20261003`.

## Memory interpretation

| Stored assistant tensor payload | Bytes | Decimal MB |
|---|---:|---:|
| A6/group 128 | 1,013,090,816 | 1013.1 |
| A4/group 64 | 774,539,776 | 774.5 |
| Payload difference | 238,551,040 | 238.6 |

A4's stored payload is 23.55% smaller. Payload excludes headers, caches,
prepared constants and allocator overhead; it is not a measured whole-engine
resident saving. This matched test deliberately held the target and **both**
assistants: loaded active memory was 95.323 GB, equal starting active memory was
95.924 GB ordinary and 95.922 GB predictable, and measured peaks were
96.344–96.482 GB. A6 and A4 arms had essentially the same peak because both
weight sets were resident. No separate per-assistant resident delta was measured.
