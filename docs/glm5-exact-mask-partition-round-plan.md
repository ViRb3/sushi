# Exact mask and partition round

Start from accepted runtime `e1597cc2`, rejected-round checkpoint `b552ff49`.
[Full-history mask research](glm5-full-history-mask-prefill-research.md) and
[exact partition research](glm5-exact-pool-partition-research.md) define two
bounded candidates. No new retrieval approximation, bank-size sweep or
precision restoration belongs in this round.1500/60 remain unverified.

| Worker | Scope | Owned source |
|---|---|---|
| 1 | Exact dense membership from original selection | New full-history mask builder/probe, original T16 selector delegation |
| 2 | Safe large native SDPA | New head-batched wrapper/probe, timed all-history finite check and safe fallback |
| 3 | Exact top512 cutoff | New BF16-key selector/probe with stable compaction/512 survivor sort |

Workers1/2 share one strict T2048/N16384/H64/D512 component. Agree a small
interface for original2051 IDs and bool[T,N] membership before editing.
Invalid IDs target a separate dummy column, never key0. Preserve100% original
key membership, uniqueness, causality and tails. No full-history/head copies
or head-expanded bool plane. Native wrapper calls current MLX API and admits
verified strides/GQA64; no Metal body port is planned.

Include a fresh finite scan of all valid history and endpoint predicate wait.
Any nonfinite value uses the original safe gathered path, preserving valid
nonfinite propagation rather than sanitizing it. Include this work in timing.
Native full-history loops7.88× more key tiles; earlier small-query masks lost.
One complete actual16K T2048 comparison with three warmups/eleven pairs decides
this large-dispatch hypothesis; stop a noisy or slower result without another
layout/count variant. Record selected/membership bits, finite arithmetic drift,
fallback/head/empty-mask behavior, strides, actual peak and conservative512MiB
per pending-layer allowance. Native key order/rounding can differ under existing
owner policy, without retrieval loss or restoration.

Worker3 keeps unchanged BF16-rounded FP32 scores and ordered GPU result.
The pinned Metal ArgPartition ignores kth and sorts the entire axis. Use one
two-byte radix cutoff, stable ID compaction and native512-survivor sort, with
exact score order and ascending original IDs on ties. Handle negative values,
both zeros, infinities, NaNs and partial pool tiles; CPU/unsupported precision
or geometry uses the original path. No scoring changes or axis padding.
Prove512 IDs/2051 expansion/all attention bits, then one inclusive T2048
3-warmup/11-pair test. Keep2MiB/graph and two graphs initially; no radix-width
variants. A mismatch or noisy/slower component is removed.

Root owns all production attention seams, bills/counters, actual-model drift/
state and performance gates. A winning full-history arm gets fixed long-prefix
quality/drift and mode-matched serial/spec state with identical prefix/assistant.
An exact selector winner requires zero logits/cache/state drift and a matched
model performance gain. Hold reference memory equal, include all rounds and
cleanup. No actual-model run for a failed component. Source remains WIP until
acceptance; commit/push only accepted runtime.

CPU builds, quiet fixture jobs and model loads are sequentially granted by root,
foreground QoS/thermal protocol/per-job GPU lock. RoutineHTTP2K–16K; selected32K
once for an accepted combined winner. Workers prepare source/tests first.
