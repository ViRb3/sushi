# Frozen raw-QKV observer protocol

The bounded diagnostic holder passes its fixture-free protocol gate. This is
preparation for the [one-tree counterfactual](glm5-prefill-qkv-attribution-round-plan.md),
not a numerical kernel, cache or performance result.

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
privately under artifact key `glm53-qkv-observer-proof-20261004`. There is no
model load, shader dispatch benchmark or throughput claim in this gate.
