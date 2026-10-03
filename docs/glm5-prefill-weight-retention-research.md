# Prefill BF16 weight retention research

Recommend attribution first, before implementing retention. Repeated work is
confirmed and one fixed cohort narrowly fits the recorded 32K admission ledger,
but no existing result isolates dequantization's removable time. There is no
basis to choose a smaller bank subset or promise a material throughput gain.
Research checkpoint `acb62ecd`; accepted packed32 runtime remains unchanged.

## Repeated work and one fixed cohort

`glm5_a6_dense_once.tryPrefill` admits only B1/T2048, BF16/GPU and the original
A6/group128 Q/K/V 4096→8192 and output 8192→4096 banks. `apply` creates a new
`Ops.dequant` output, transposes it as a view, and invokes the unchanged native
BF16 GEMM. The enclosing layer Ops later releases that 64 MiB expansion. There
is no owner-held decoded tensor, so every full chunk expands the same four
banks again. Decode and partial chunks continue through compressed affine paths.

Latest HTTP records confirm the count, rather than merely predicting it:

| Actual input IDs | Full chunks | A6 expansion calls |
| --- | --- | --- |
| 2037 | 0 | 0 |
| 2073 | 1 | 136 |
| 4061 / 4097 | 1 / 2 | 136 / 272 |
| 8225 / 8261 | 4 | 544 |
| 16274 / 16310 | 7 | 952 |
| 33595 / 33631 | 16 | 2176 |

The only proposed cohort is **all four banks of all 34 KDA layers**: 136 original
banks, each 67,108,864 bytes, total 9,126,805,504 bytes (8.5 GiB). No shape,
layer or prompt-length bank search is proposed. Keep it request-owned only
during multi-full-chunk prefill, fill lazily at each bank's first eligible use,
reuse the identical materialized BF16 buffer thereafter, settle the final
prefill result and release the entire cohort before decode. A request with fewer
than two full chunks uses the current path; it has no repeat expansion to save.
Original compressed weights and every small stored BF16/F32 tensor remain owned
by the model. Decode must never select a retained BF16 bank.

At seven full chunks this avoids 816 subsequent expansions and 51 GiB of BF16
bank stores; at sixteen it avoids 2040 and 127.5 GiB. Those are source/count
facts, not latency estimates. GEMM operands, shape, transposed strides, kernel
selection and arithmetic must remain identical. Verify every retained coefficient
and full-layer output/state bit against fresh expansion; do not assume a new
contiguous or transposed materialization is arithmetically harmless.

## Both limits and exact conservative ledger

The requested 110 GiB limit is 118,111,600,640 bytes. The latest actual memory
and wired/admission limit is **115,448,725,504 bytes**, because HTTP takes
`min(requested memory, recommended working set)` and uses that value for both
limits. Neither limit may be enlarged. Physical 128 GB is not admission evidence.

Use recorded startup active 94,548,731,128 bytes as the resident reference,
including accepted target, A6 assistant and cluster weights. Keep every existing
reserve, including the 536,870,912-byte A6 async2 transient reserve; do not
credit it away merely because decoded buffers now have a longer lifetime.
For ordinary 33631 IDs plus 192 outputs, the unchanged source ledger is:

| Request reserve term | Bytes |
| --- | ---: |
| KDA recurrent/convolution views | 590,479,360 |
| MLA cache views at capacity 34048 | 1,629,945,856 |
| Assistant cache forecast | 2,789,212,160 |
| Activations | 3,221,225,472 |
| Attention/branch scratch | 289,406,976 |
| Existing A6 expansion | 536,870,912 |
| MLA projection permutations | 805,306,368 |
| Packed bank plus second cadence bank | 536,870,912 |
| Index scoring | 16,777,216 |
| Cluster activations | 2,621,440 |
| Native decode | 33,554,432 |
| Fixed reserve | 268,435,456 |
| Planned immutable growth | 411,631,616 |
| **Total unchanged request reserve** | **11,132,338,176** |

Resident + cohort + that reserve is 114,807,874,808 bytes. Remaining headroom
is **640,850,696 bytes against the actual wired/admission limit**, and
3,303,725,832 bytes against requested 110 GiB. Predictable 33595 plus 192 leaves
1,085,709,064 bytes against the actual limit. These calculations apply to the
recorded inputs/output bound and settings only. Every actual request must use
checked additions with its real resident/cohort bytes, context capacity and
unchanged growth forecast before allocating. Larger requested outputs or another
resident configuration can fail; do not truncate or silently shrink the cohort.
Prepare/store each bank directly into its retained handle, preserve the original
transient bill, and measure preparation peak rather than assuming allocator reuse.
The latest HTTP records have no allocator peak; the matched 16K model's roughly
96.43 GB peak is a different scoped observation, not proof for this cohort.

## Missing cost and one decisive measurement

The original component measured 2.530552 ms for QKV and 2.712552 ms for output,
including fresh dequantization, allocation, GEMM, evaluation and frees on synthetic
banks. Those timings do not isolate dequantization. The difference from affine
NAX is the net implementation difference, not dequantization time. Current HTTP
counters identify repeated calls but supply no per-expansion duration; the old
forced component profile is not a current normal-throughput budget. All GEMMs
remain necessary, and the first expansion of every bank remains necessary.

Request one bounded attribution job using the existing actual layer-zero four
bank fixture and original T2048 geometry. Pair the current complete dequant-plus-
GEMM chain with the identical GEMMs supplied the already-materialized original
BF16 banks; record a separately measured four-bank preparation/evaluation/free
cost. Use one fixed fixture, three warmups and eleven pairs, inclusive copies,
waits, allocations and cleanup. Any forced preparation wait must be labeled as
perturbed attribution, not transplanted directly into HTTP latency. Do not alter
the kernel or cohort, load a target, build a profile framework or survey banks.
This decides whether the omitted expansion work is large enough to justify
request-lifetime retention despite its 8.5 GiB cost.

If attribution is material, the eventual complete-layer gate must time at least
two consecutive T2048 chunks with preparation at first use and teardown included,
matching all outputs and initialized FP32 KDA/convolution state. The model gate
then uses one exact long input, ABBA with both warmed control/candidate Requests
held before every timed arm for equal reference memory. The candidate cohort
must be created inside each timed prefix and freed before timing ends; startup
or warm preparation is not free. Invalidation is request completion/failure,
model teardown/reload and any original tensor replacement; keys must identify
actual immutable bank ownership, not an unbounded process-global pointer cache.
Record cold preparation, repeated prefix, final release, peak/wired headroom,
exact logits/valid cache-state and continuation. Stop noise/loss or failed
admission without selecting a smaller cohort or restoring precision.

Evidence: `glm53-packed32-llmprobe-20261003`,
`glm53-packed32-32k-20261003`, `glm53-a6-dense-once-20261003` and
`glm53-packed32-model-16k-20261003`. No implementation, build or GPU run was
performed for this report.
