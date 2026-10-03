# Rejected near-full A6 prefill expansion

The isolated candidate extended the existing opt-in A6/group128 prefill
expansion from exactly2048 to1536–2048 input rows. It retains only the two
qualified KDA shapes, 4096→8192 and8192→4096, B1 BF16 activations, U32 packed
weights and BF16 scales/biases. Other rows, shapes and formats keep the native
quantized path. No tokens are padded, and no stored tensor or persistent cache
precision changes. `Ops.dequant` plus native dense matmul is unchanged.

A scoped `bind(bool)` controls the existing opt-in within one process and restores
its previous state. Existing successful-dispatch/reset APIs remain available.
The premium remains64 MiB per decoded matrix, four matrices per pending layer,
512 MiB at async2. `transientBudget` covers any configured maximum chunk at
least1536 because larger chunks can emit an eligible final remainder; actual
kernel admission still stops at2048. HTTP already reserves this premium for its
configured2048 chunk. The actual per-request HTTP counter belongs to coordinator
integration.

## Production-bank component, 2026-10-03

The ReleaseFast probe used the original layer-zero QKV and output weight/scale/
bias banks from the selected2.3bpw target. Inputs were materialized synthetic
BF16 normal arrays, sigma0.2, at their actual row counts. All69,070,848 BF16
output values checked across1536/2037/2048 and both bank shapes matched the
current affine NAX path bit for bit, with zero relative L2/max error and no
nonfinite output. Rows1535/2049 declined without incrementing dispatch counts.
Guard/dtype/shape, remainder billing, scoped restoration and the existing caller
checks passed. The regression first failed on the old exact2048 guard.

Only2037 rows were timed. Three warmup pairs preceded eleven alternating AB/BA
pairs per bank, each averaging four fresh apply/evaluate/free graphs. Expansion,
copies, allocation, construction, native GEMM, endpoint evaluation and graph
teardown were included; no decoded bank was reused between timed calls.

| Projection | Affine NAX ms | Expansion + dense ms | Median arms | Median paired | Wins |
|---|---:|---:|---:|---:|---:|
| 4096→8192 | 2.756375 | 2.542375 | −7.76% | −7.90% | 11/11 |
| 8192→4096 | 2.811385 | 2.708875 | −3.65% | −3.94% | 11/11 |

The component had seven passes and two unrelated gated research skips. It used
foreground QoS, exclusive GPU lock, maximum fans requested and ten-second idle;
initial/end temperatures were54.71/50.29°C. The owned lock was released and fans
returned automatic. Source baseline `19ab4f9b` plus the candidate is captured by
exact helper/probe, native shader, compiled runtime, binary and original tensor
payload hashes. Measurement key: `glm53-a6-nearfull-2037-20261003`.

These are repeatable inclusive primitive wins, not model throughput. The first whole-model gate is recorded below. Ordinary native BF16 NAX rounding
was permitted, but no restoration math was introduced. The candidate was rejected after the equal-memory real check below.

## First actual2037 model gate: exact, performance not yet accepted

One loaded target ran current/candidate ABBA with grid transposition scoped off
and all other selected settings retained, async2 prefill, profiling off. The native test input was the first2037 IDs of the original2048-token fixture.
The candidate processes those exact2037 IDs without padding.
The measured intervals were1.805824 /1.889118 /1.892869 /2.073496 seconds.
Control average1.939660 versus candidate1.890993 seconds suggests2.51% less
time, but the controls drifted14.8%, and reference snapshots were retained
incrementally after timed arms. This does not yet establish a real-model win.

Actual A6 calls were0/136/136/0; grid and retained-cluster calls were zero. All
logits and every initialized KDA/valid MLA cache array matched exactly across
arms. All64 reference-forced continuation positions had matching greedy IDs,
zero KL and exact final logits/full valid cache state at offset2101. Peak Metal
memory was95.23–95.74 GB, including held reference/candidate caches. The job
used ReleaseFast, foreground QoS, exclusive lock, verified maximum fans and
ten-second cooldown from52.01°C; it completed with exit zero, released the
lock and restored automatic fans. Raw key:
`glm53-a6-nearfull-real-model-2037-20261003`.

One performance-only ABBA check then retained both warmed reference snapshots
before every timed arm, with equal memory and no repeated quality/continuation
loop. It addressed that specific harness/drift concern, without a new kernel
variant or parameter sweep.

## Equal-memory real check: rejected

Every timed arm began with exactly93,880,665,848 active bytes. Both warmed
control and candidate Requests/logits were held before A0, grid remained off,
and actual A6 dispatches were again0/136/136/0. ABBA prefill intervals were
1.708875 /1.829852 /1.945202 /1.952121 seconds. Control average1.830498 versus
candidate1.887527 seconds made the candidate **3.115% slower**. Controls still
drifted14.23%; this check did not establish a repeatable real-model benefit.
Peak memory was95.57 GB in controls and95.74 GB in candidates. Foreground QoS,
exclusive lock, verified maximum fans and ten-second idle from54.76°C were
recorded. The check completed with exit zero, released the lock and restored
automatic fans. Raw key:
`glm53-a6-nearfull-equal-memory-perfcheck-20261003`.

The 1536–2048 guard, enlarged configured-remainder bill and scoped binding were
restored to the prior exact2048 implementation. The isolated probe/private
evaluators/roots and their binaries were archived and removed from the checkout.
Only this measurement lesson lands. The generic HTTP A6 engagement counter is
coordinator-owned and independent of the rejected range change. No further check,
nearby threshold or alternate kernel was tried. Exact output and a strong
primitive win did not justify pushing a real-model performance regression.
