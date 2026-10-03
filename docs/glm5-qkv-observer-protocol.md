# Frozen raw-QKV observer protocol

The bounded diagnostic holder passed its fixture-free protocol gate and the
[completed one-tree actual-model attribution](glm5-qkv-attribution-result.md).
Exact raw outputs, all three-row
logits/decisions and valid committed states passed; complete median latency was
48.868083 → 42.276333 ms (13.4889%, 11/11 pairs). This is a one-case perfect-reuse
ceiling, without a cache, kernel-time or production-throughput claim. The
[round plan](glm5-prefill-qkv-attribution-round-plan.md) defines that scope.

A private red copy retained the normal capture/observe behavior but declined
reuse. ReleaseFast build passed; the proof failed at `ExpectedQkvReuse` after
the disabled/identity test passed. The unchanged real holder then built and
passed both tests. No packaging retry or arithmetic change was needed.

Directed checks cover unbound/off no-ops, immutable epoch/tree/prefix identity,
layer order, BF16 input/raw/head geometry, incomplete and repeated passes,
all 34 raw slots, capture/ordinary-observe/reuse counters, and caller-owned reuse
handles. Retained raw/head aliases preserve exact synthetic BF16 values after
the source handles are freed. One vector evaluation occurs outside hooks;
the hooks retain aliases without evaluation or CPU copies. This synthetic
protocol proof does not establish actual-model raw/logit/state equality.

The actual payload is 5,013,504 bytes for 34 BF16 `[1,3,24576]` raw arrays. Reference
and current BF16 `[1,3,154880]` logits add 1,858,560 logical bytes; fixed holder/pass
metadata keeps the total below 8 MiB. No inputs, weights or prefix states are
retained by this helper. The synthetic proof aliases one raw buffer across
slots, so its physical allocation is not an actual-model peak measurement.

The six root-owned seams retain the original endpoint and cadence: validated
verify entry, actual KDA layer identity, before/after raw projection, existing
head output and completed decision settlement. Reuse aliases enter normal
`Ops` ownership. Default-off handling performs no validation or allocation.

Baseline is accepted `4fcb541e`, round plan `a17382a5`; source was WIP.
Helper SHA256: `4925d167fd356ef9b50aa4dd261b57746d758a345e4f54ea7634c832119bd5af`.
Green binary SHA256:
`c61d14ec0f75d05135cdbe9a41605885c19d39f18ddc798b15a3f6dca9a1dcaf`.
Accepted staged libraries, ReleaseFast, foreground `taskpolicy -a`, separate
per-job locks, max fans/idle and automatic fan cleanup were used. Both runs
are terminal; raw source, commands, hashes, exits and logs are preserved
privately under artifact key `glm53-qkv-observer-proof-20261004`. The protocol
gate itself loaded no model and measured no shader or throughput performance.

After the actual-model job, root removed the shared diagnostic seams. The
helper and private protocol probe were hash-verified against their compiled
sources, archived privately and removed. Their exact protocol/source evidence
remains under `glm53-qkv-observer-proof-20261004`; actual-model evidence is under
`glm53-qkv-attribution-20261004`. No runtime optimization landed from this gate.
