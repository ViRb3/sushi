# Selector and recurrence performance round

Start from the accepted exact prefill stack `f9f4c8d2`, recorded and pushed
through `18b27d76`. Its measured 2K/4K/8K/16K prefill rates are
931.71/802.82/727.81/655.06 tok/s; decode is47.08/45.48/44.44/43.04.
The selected 32K cell has33595 input IDs,604.18 prefill and42.39 decode.
It is2.59% longer than the inherited32K input; rates are not identical-input
comparisons. Both8K/32-output and32K/64-output complete serial-state gates pass.
The1500 prefill/60 decode targets remain open.

Two researchers supplied the starting evidence:
[prefill, long context and concurrency](glm5-round-next-prefill-research.md)
and [decode, speculative verification and concurrency](glm5-round-next-decode-research.md).
They distinguish forced-evaluation profiles from normal throughput and document
rejected paths. No new lossy arithmetic is planned in this round.

| Worker | Candidate | Owned source |
|---|---|---|
|1|Bounded overlap of NAX selector pool tiles|`glm5_indexpool_nax.zig` and focused probe|
|2|Canonical three-node KDA chain/fork recurrence|New canonical helper/probe; tiny source export in `glm5_dflash_kda.zig`|
|3|Exact shared-prefix scoring across three verification queries|New shared-prefix helper/probe; tiny SCORE source export in `glm5_attention.zig`|

Worker1 retains at most two2048-pool dot graphs, each at most2MiB, before
settling the pair. Keep query tile16,32 heads, key dimension128, native dot
geometry and every BF16 rounding boundary. Conservative extra admission is
8MiB per pending MLA layer,16MiB at async2, above the existing scorer bill.
The existing real16K attention fixture supplies the inclusive paired test,
with accepted packed cadence on in both arms. Check all scores, selected IDs,
attention outputs, ragged queries and the final partial pool tile.

Worker2 handles only parents[-1,0,1] and[-1,0,0] at actual W3/H64/D128
geometry. Replace runtime private-array indexing with explicit chain state or
root state for a fork. Keep FP32 state and accumulator, original FMA/simd_sum
order, BF16 output, cached first-path leaf and general fallback. Prove all tree
outputs and retained state, plus replay hit/miss and convolution tail. Benchmark
the complete recurrence/leaf/replay component for both topologies. No additional
persistent state or matrix is admitted.

Worker3 scores the immutable completed pool prefix for all three queries in
one kernel. Share key loads and retain three independent original arithmetic
chains. Each branch may have one new completed pool; handle that suffix using
the original scalar operation and its actual offset. Preserve three separate
argpartition calls and each original pool dimension/tie ordering. Do not pad
fork branches into a common total pool count. A32K request plus generated
tokens exceeds8192 pools, so the guard must cover that real workload. Compile
one shader with runtime prefix count. Reuse captured BF16 query/weight/pool
data for exact score/selected-set and inclusive three-selector comparison.
Do not copy full pooled histories.

The coordinator owns production MLA delegation, request admission, engagement
counters and real-model evaluation. Workers do not overlap those edits.
Builds and measured GPU runs are scheduled; no build or other heavy job runs
beside a95GB model load or quiet timing. Three warmups and eleven alternating
fresh paired samples suffice for each focused component. Stop a noisy or losing
arm; do not start another variant sweep.

Keep implementation source uncommitted until acceptance. Stamp dirty-source
hashes and binaries for experiments. A component winner advances to one actual
model arm, including serial-token/complete-state correctness and matched
performance evidence. Evaluate each candidate before accepting its runtime hook,
then commit and push only accepted changes. Rejection findings may be documented
without retaining a runtime path. Subsequent HTTP iterations use2K–16K; one
selected32K qualification closes the round. Then two researchers start the next
round. Keep sensible lossy experiments opt-in if a later round proposes them.
