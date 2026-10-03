# One fixed steady-4096 prefill candidate

Recommend a bounded **batching change**, not another expert reader: retain the
initial chunk of at most 2048, then use 4096 only while at least 4096 inputs remain;
otherwise use the original 2048 plus final partial. This preserves 2K/4K behavior
and the cold dense-attention boundary, while 8K–32K can combine steady work.
No current packed32/grid-transposed4096 measurement was found. Current HTTP
rejects chunk>2048 and several optimized helpers decline 4096; changing a CLI
number alone would measure a different fallback stack.

The actual L20 attribution measured gate/up 10.797000 ms and down 5.512833 ms,
90.2994% of the staged sum. Inserted endpoints made complete execution 4.09%
slower, so these are diagnostic groups, not unperturbed GPU/HTTP shares.
This candidate changes work in those GEMMs without revisiting excluded retention,
route pruning, gate/up splitting, middle/down fusion or WIN/group/reader variants.

## Changed work and fixed ownership

`buildMimoWindowTable` in `src/exl3/expert_exl3_kernels.zig` aligns WIN32 windows
to experts. For unchanged per-token routes,
`ceil((countA+countB)/32) <= ceil(countA/32)+ceil(countB/32)` per expert.
Combining two batches can remove partial expert windows and their repeated
full-K weight-decode/MMA loops, plus repeated chunk preparation. It does not
halve routed token work or guarantee a gain. Record actual window/padded-row
counts; router/native dispatch and sort cost can change.

| Required fixed support | Source |
| --- | --- |
| Only 2048/4096; route rows 16384/32768 throughout eligibility, reshape, prepare, window metadata, middle and finish; config key includes route rows | `src/exl3/glm_prefill_grid.zig` |
| Preserve current A6→BF16 expansion/GEMM behavior at 4096; same four-bank expansion reserve | `src/glm5_a6_dense_once.zig` |
| Explicit 4096 head-batch eligibility and doubled permutation bill | `src/glm5_mla_prefill_batch.zig` |
| Explicit 4096 clustered projection/output shape and bill; resident banks unchanged | `src/glm5_kda_prefill_cluster.zig` |
| Fixed schedule/parser/admission and diagnostic metadata | `src/glm5_bench_http.zig`, `src/glm5_diagnostic.zig` |

Keep WIN32, current physical grid, NAX 16×32×16/K16 accumulation, MCG/W12/n36,
top8, Hadamard/clamp/finish order and all stored tensors. Packed attention already
bounds work to B32/two graphs; do not widen its tile/cadence. Keep selector order,
causal offsets, invalid/future guards and IndexPool eligibility unchanged.
The cold 2048 exception prevents early queries from replacing original dense
attention with score-ordered sparse retrieval. KDA token-sequential FP32 state
handling has no declared 2048 ceiling, but full 4096 output/tail/state versus two
2048 calls requires proof. Wider native GEMM dispatch is not assumed bit-identical.

## Prospective admission

Retain all existing bills. Generic activation allowance already scales token
count, including expert planes; do not charge a fictitious new weight bank or
credit lower padding. Expanded A6 weights and bounded B32/selector scratch stay
fully charged; only row-dependent MLA permutations/cluster outputs double.

| 32K source ledger | Bytes |
| --- | ---: |
| Existing reserve |11132338176 |
| Added activations |3221225472 |
| Added MLA permutations |805306368 |
| Added cluster outputs |2621440 |
| Added fixed route metadata/config/sort allowance,2MiB per pending layer×2 |4194304 |
| Proposed full reserve |15165685760 |
| Recorded baseline active |94548731128 |
| Active plus proposed reserve |109714416888 |
| Remaining below 115448725504 limit |5734308616 |

Visible 4096 route metadata occupies 534788 bytes: four 32768-entry 32-bit arrays,
two 1312-entry window arrays and count scalar. The entire 4MiB addition is retained
above old reserves for pending 2 and sort/config headroom. Native sort scratch,
actual peak, failure cleanup and live async lifetimes still require qualification;
this arithmetic is prospective, not current admission. Current declined helpers
returning zero budget are not savings. Reference/fixture holders for proof/model
jobs are additional. No memory/wired limit increase or storage credit is allowed.

## One bounded gate

First prove fixed 4096 expert outputs/metadata against two current 2048 calls on
equal operands/routes/full original banks, and complete KDA output/convolution/
FP32 state at the chunk boundary. A repeated 2048 fixture is constructed evidence,
not a real 4096 activation capture. Then one three-warm/eleven-pair complete
layer gate includes sorting, all projections/pointwise work, endpoints and frees,
with equal references, actual padding counts and new-path engagement. Stop loss
or noise without a chunk/tile/reader sweep.

A winner receives one matched long-prefill ABBA with the fixed schedule, complete
last logits/valid state and continuation. If grouping changes old-target bits,
report it explicitly and require the original nonrepeated 16K code/prose 24+192
forced-prediction screen/bounds before acceptance; compare each prefix's own
serial/spec state, without precision restoration. Root owns bills/model/HTTP
qualification and default decisions. No helper, build, GPU/model/capture or
runtime change accompanies this research, and no 1500/60 prediction is made.
