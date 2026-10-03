# Exact B32 packed attention research

Source-only audit at `0e82efad`, accepted runtime `e1597cc2`. One fixed
proposal is materially distinct from pool-tile overlap: keep **two original
T16 selector calls**, concatenate their ordered IDs, and combine only gather
and native head-packed attention into one B32 call. Retain two pending packed
graphs. No prototype, build or GPU run was made; no speed is assigned yet.
Do not widen selector math, reuse pools across queries, change key loading or
attempt a row-count sweep.

The current helper reshapes Q to `[B,1,64,512]` and gathered KV to
`[B,1,2051,512]`. Native full D512 selects BQ32/BK32, WM2/WN4, group `(32,2,4)`,
GQA1, scale 1/16 and an array mask. B16→B32 changes grid `(2,1,16)`→`(2,1,32)`:
32→64 independent tensor groups per call. Q64, head 1, 65 ordered key blocks,
alignment/function constants, FP32 partial-score/softmax/value arithmetic and
BF16 stores remain unchanged. Batch index only advances Q/K/V/output/mask
pointers. This does not enlarge the cooperative tile or its live accumulators.
No native Metal port, new mask policy or precision restoration is needed.
Bit equality remains a required proof rather than an assumption from source.

Selector handling is decisive. `glm5_indexpool_nax.tryScores` currently admits
9–16 BF16 rows, 3584–8192 pools and at most a 2 MiB raw dot plane. Passing 32 rows
would decline it and select scalar `SCORE`; that is a numerical-mode crossover,
not an exact batching optimization. The proposal instead calls the original
selector at offsets `base` and `base+16`, each with exactly 16 queries and the
same pool/head geometry, waits and partition order. At 3584–8192 pools those calls retain NAX eligibility; at 8K or above
8192 pools they retain the current scalar mode. The nominal 32K label must not
be confused with an actual history above 32768 tokens. Concatenate
only the resulting two `[16,2051]` ID planes. All pool/tail completion rules,
causality, tie order, query-local membership and invalid-ID guards stay current.
No 4 MiB dot limit or M1024 scorer kernel is proposed.

A T2048 late block becomes 64 packed B32 calls instead of 128 B16 calls, and
32 pair settlements instead of 64. Native/gather/empty-row graphs and their host
construction decrease, while the original 128 T16 selections and every internal
pool-tile wait remain. At 16K this still includes 256 scorer tile waits. Total
native tensor groups stay 4096, dot work stays unchanged, and logical gathered
KV writes stay 4,301,258,752 bytes per MLA layer. The hypothesis is more work per
native dispatch plus fewer complete graph boundaries—not fewer dot products
or gathered bytes. The existing two B16 graphs already offer pending work;
actual utilization has not been measured and doubling per-call groups does
not imply a doubled execution rate.

| Per packed graph | B16 → B32 |
| --- | ---: |
| BF16 gathered KV | 33,603,584 → 67,207,168 bytes (32.047→64.094 MiB) |
| Existing conservative temporary formula | 40,193,712 → 80,387,424 bytes (38.332→76.663 MiB) |
| Joined original ID plane | new 262,528-byte concat at B32; original selector planes also retained |
| Scorer geometry and ledger | T16 /2 MiB dots /8 MiB reservation, unchanged |

The B32 KV bank alone exceeds the current 64 MiB helper cap. Reserve 128 MiB per
B32 graph, 256 MiB for the two-graph pair, and 512 MiB for two pending MLA layers;
the current B16 pair reserves 128 MiB/layer, 256 MiB at async2. The conservative
increment is therefore 256 MiB at async2, before keeping the existing selector
and activation bills. Observe ownership and actual peak, including both T16
selector results, their concat, native contiguity behavior, empty handling,
ragged cleanup and failures. No full-history/head copy, F32 cache or floating
Q×H×K score plane is introduced. Source bounds do not justify a lower bill.

The rejected [pool-tile cadence](glm5-indexpool-cadence.md) changed only overlap
inside each T16 selector. It was exact and reduced the 16K attention component
129.559→108.509 ms (16.25%, 11/11), but its loaded-model ABBA improved only 0.454%
amid monotonic drift; it was removed. B32 does not restore that scorer overlap.
It changes native dispatch population and whole packed-graph count, while
leaving its rejected candidate's scheduling lever unused. Still, the model
failure warns against treating saved boundaries or a component win as prompt-
wide gain. The current roughly 130 ms late-attention call is an inclusive cost,
not an identified removable budget; cold dense MLA and routed/KDA work are
outside this proposal.

One bounded gate uses the existing actual T2048/16K fixture. Compare B32 with
two original T16 selections against current B16/two-graph cadence. Prove all
ordered IDs and BF16 outputs, head/query order, key 0, slot 2050, unique valid
keys, future/unselected NaN protection, valid nonfinite propagation and empty
rows. Preserve original B16 fallback for a final fragment; include a 33-row
case to exercise B32 plus one-row cleanup. Then one three-warmup/eleven-pair
whole-attention test includes both selections/ID concat, gather, unchanged
SDPA, zeroing, pair settlement, output concat, evaluation and every free.
No isolated SDPA timing or overlap-only retry decides acceptance.

Stop mismatches, noisy results or losses without a wider tile. Only a clear
winner gets one same-input loaded-model long-prefill ABBA with equal held
references, unchanged outer chunk 2048/async2, exact logits and complete valid
MLA/KDA state through continuation, all work/cleanup, counters and measured
peak. The prior 0.454% model result and unchanged total arithmetic leave no
basis for predicting 1500 prefill tok/s from B32.
