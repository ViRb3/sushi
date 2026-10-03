# Exact IndexPool top-512 selection

Research only, 2026-10-03, accepted runtime `e1597cc2`. Recommend one isolated
**BF16-key radix cutoff, stable compaction, then sort only 512 survivors**
experiment for T16 prefill. No prototype, build, GPU run or speed forecast.

## Verified local behavior

Local MLX is `v0.32.3`, commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`.
`mlx/backend/metal/sort.cpp::ArgPartition::eval_gpu` calls
`gpu_merge_sort(..., true)` without using `kth_`: full argsort. FP32 axes above
2048 use sorted 2048-element blocks and full merges: two blocks at 4096 pools,
four at 8192. It allocates two complete value planes, two complete index planes,
block partitions and complete output indices although Sushi keeps only 512.

`kernels/sort.h::LessThan` uses `<`, placing NaNs last. Thread sorts swap strict
inversions; block/global merges take the left input on equality. Initial IDs
follow pool position. The inspected GPU therefore returns decreasing scores
after Sushi's negation, with ascending pool IDs on equal scores, including
signed zeros and NaNs. This is a pinned implementation property requiring
behavioral proof, not an argpartition API guarantee. CPU uses `std::nth_element`
with an ID tie-breaker; its prefix is unsorted. Keep CPU unchanged.

`glm5_attention.selectChunk` expands those ordered IDs directly. Identical
selected sets with different order can change bank layout and attention rounding.

## One bounded selector

Both BF16 scalar SCORE and accepted NAX epilogue store `float(BF16(total))` in
FP32 after original dot/product rounding and sequential 32-head accumulation.
ReLU dots do not imply nonnegative scores: weights can be negative. Future pools
receive `−inf`; finite, positive or unique scores cannot be assumed.

Admit qualified GPU, contiguous FP32 scores from the unchanged BF16 scorer,
exactly 16 queries and 3584–8192 pools. Other geometry/precision/history and CPU
use the original selector; decode/verification stays unchanged. Never pad the
partition axis. Derive ordered 16-bit keys from widened BF16 bits: complement
negative encodings, flip the sign bit otherwise, fold both zeros together and
place all NaNs below `−inf`. Infinities retain numeric order. These keys affect
selection only; scores, operands, cache precision and accumulators stay intact.

Use a 256-bin high-byte histogram, then a 256-bin low-byte histogram within
the cutoff high byte, to locate rank 512 exactly. Keep higher scores and the
required earliest pool IDs tied at the cutoff. GPU tile counts/prefix scans
compact stably in original ID order. No CPU score readback, atomic-arrival
ordering, oversized survivor list or 65536-bin histogram. Sort the exactly 512
original score values with the pinned GPU comparator, mapping its permutation
back to original IDs. All-zero, all-masked and tie-heavy rows still produce 512
unique IDs. Existing expansion masks future IDs; do not sanitize valid caches.

Original merge planes alone occupy 2 MiB at T16/P8192; output indices/negation
add 1 MiB, excluding existing 512 KiB scores. P4096 halves those spans. The new
histogram/tile counts and survivor buffers are smaller, but extra dispatches
and scans may erase the gain. Start conservatively at 2 MiB per live selector
graph, two graphs maximum: 4 MiB/pending MLA layer and 8 MiB at async2, added to
existing attention/output bills. Confirm peak and failure cleanup before
admission. No retained history copy or T2048-wide score plane. Scoped default-off
binding and actual engaged/declined counters belong in the narrow caller seam.

## Cost comparison

Partition time is **unisolated** in recorded actual-input evidence.

| Recorded work | Cost | Limitation |
|---|---:|---|
| Historical synthetic T16 selector, 16K / 32K | 525.396 / 905.271 µs | Includes scoring, partition, expansion and frees |
| Accepted actual T2048/16K whole attention | 129.888 ms/MLA layer | Includes every selector, gather, SDPA and cleanup |
| Cold T2048 routed FFN | 810.624 ms (45.3%) | Forced-evaluation prompt total 1788.124 ms; no projection/preparation/sort split |

The cold profile is `glm53-next-wave-prefill-components-20261003`; router was
separately 23.183 ms. It uses dense MLA and cannot attribute late sparse sort
cost or rank directly against the late fixture. The entire late attention call
is about 1.43 s across eleven layers for one block; partition's removable share
is unknown. Routed FFN remains the largest established cold bucket, but no new
exact routed proposal is evidenced after failed near-full grid/model gates.
Choose this selector experiment because source establishes unnecessary sorting of the complete history and an exact bounded replacement. It cannot improve cold dense
attention or remove routed work. The rejected exact short-row scorer's 14.03%
whole-attention slowdown reinforces the need for an inclusive gate; the faster
approximate shared bank failed fixed model quality.

## Gates and ownership

Worker owns one new helper/probe/result doc; coordinator owns the narrow
`selectChunk` seam, billing, counters and model evaluation. Scorer, accepted
packed cadence, gather and native SDPA stay fixed. Behaviorally prove all 512
ordered pool IDs, all 2051 expanded token IDs and every attention output bit. Directed rows
cover negative values, both zeros, BF16 neighbors, cutoff/block-boundary ties,
`±inf`, future `−inf`, NaNs, partial pool tiles and declined geometries. No
source-scan tests, tolerance or precision restoration.

Reuse actual 16K Q/index-Q/weights/pooled/latent planes. One complete T2048
attention comparison uses three warmups and eleven fresh AB/BA pairs, including
scoring, cutoff/compaction/sort, expansion, gather, SDPA, collection, endpoint
settlement and frees. Record hashes, all pairs, actual engagement and peak.
Stop a noisy/losing result; no radix-width or geometry variants.

A clear win proceeds to coordinator-owned actual-model qualification: unchanged
accepted pack/flags, fresh 16K requests, exact logits/output IDs and every valid
prefix/cache/state byte, then native serial/spec state with identical assistant
and policy. One warmed matched 192-output actual-model comparison includes
complete prefill/cleanup and equal retained references. Exact selection must
introduce no target drift; a mismatch rejects it. Selected 32K qualification
follows model acceptance under the exclusive-resource protocol.
