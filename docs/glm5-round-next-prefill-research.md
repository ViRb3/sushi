# Next GLM prefill, long-context and concurrency research

Read-only research at `f9f4c8d2`, 2026-10-03. The active goal is 1500 prefill
and 60 DFlash2 decode tokens/s, stable from approximately 2K through 32K.
The current selected stack includes the exact routed-grid transpose and bounded
two-tile packed-attention cadence. Its normal 2K–16K HTTP evaluation is still
running at research time; no new throughput number is inferred from components.
No source edit, build, GPU run or model load was performed for this report.

## Existing measured evidence

The fresh exact-T2048, target-only synchronized component profile is
`glm53-next-wave-prefill-components-20261003` at `dfe17328` plus an import root.
It used the selected A6 dense-expansion, BF16 retained cluster and R4 recurrence
paths. Its 1788.124 ms prompt time is synchronization-perturbed and cannot be
compared with normal HTTP prefill rates.

| Disjoint component | ms | Share of synchronized prompt |
|---|---:|---:|
| Routed FFN | 810.624 | 45.3% |
| KDA attention, 34 layers | 500.232 | 28.0% |
| Cold dense MLA attention, 11 layers | 147.899 | 8.3% |
| Shared FFN | 91.919 | 5.1% |
| Three dense FFNs | 36.411 | 2.0% |

The KDA subtotal includes Q/K/V projection, joined-QKV construction, convolution/
normalization/decay preparation, R4 recurrence, gate/post and output projection.
It is not 500 ms of recurrence. The R8 comparison saved only about 71 microseconds
per recurrence, roughly 2.43 ms across 34 layers, and was archived. The T3 FA/GA
cluster lost its inclusive comparison and is also excluded.

The current two-packed-tile attention candidate already has exact component
proofs on actual T2048 checkpoint activations:

| History | Serial / two-tile ms | Reduction | Paired wins |
|---|---:|---:|---:|
| 8K, scalar index scores | 130.201 / 105.954 | 18.62% | 11/11 |
| 16K, NAX index scores | 139.507 / 129.888 | 6.89% | 11/11 |

These are inclusive whole-attention fixture timings, not model throughput.
The 16K fixture already contains actual Q, index Q/weights, BF16 latent and pooled
caches, offsets and scale. It can be reused without another full-model capture.
See [the cadence qualification](glm5-prefill-cadence.md).

The previous fixed-shape NAX-selector component included scoring, partition,
selection expansion and frees: 525.396 microseconds at 16K and 905.271 microseconds
at 32K for 16 queries. Those synthetic component numbers identify work magnitude;
they are not the current actual-input attention's decomposition. The scorer's
source still evaluates every 2048-pool dot tile separately.

## Recommended prefill implementation: pair the NAX scorer's pool tiles

**Select this as the next prefill worker.** It changes existing graph scheduling
without a new shader, copied NAX implementation, weight format or runtime patch.
The current `glm5_indexpool_nax.scores` loop constructs one dot/epilogue tile,
waits for it, appends the result, then constructs the next tile. Above about
14K, those internal waits can drain work submitted by the already accepted
packed-attention cadence. The smaller 16K cadence gain supports investigation,
but does not by itself attribute its difference entirely to those waits.

**First arm:** keep at most two original pool-tile graphs alive, submit their
score outputs asynchronously, settle the pair before releasing either graph,
then preserve the original concatenation and argpartition order. An odd final
pool tile follows the same cleanup. Do not also widen query rows, retile dot
GEMM, change top-512 selection, or cache a different coefficient interpretation.
All dot and epilogue arithmetic, BF16 boundaries and FP32 score/accumulator
contracts remain unchanged.

**Exact geometry:** 9–16 real queries, 32 index heads, D128, BF16 Q/pooled keys/
weights, 3584–8192 pools. Each original pool tile has at most 2048 columns.
At T16 the native matmul is M512/K128/N2048. A BF16 raw dot plane is 2 MiB;
its FP32 score plane is 128 KiB. At 16K there are two dot tiles, and at 32K four.
The change reduces two/four blocking pool-tile boundaries to one/two; it does
not reduce FLOPs or establish a proportional memory-traffic saving. Decode and
DFlash verification have at most eight rows and remain outside this selector.

**Memory:** retain only two raw planes at once, at most 4 MiB, plus their existing
small score outputs and normal temporary copies. Raise the conservative scorer
ledger from 8 to 16 MiB per pending MLA layer, +16 MiB at async2. Actual high-water
allocation must confirm this bound, including failures and the partial pool
column tile. Do not hold four dot planes or an entire T2048 query block's planes.
BF16 compressed cache and FP32 persistent KDA state remain unchanged.

**Expected ceiling:** a 16K T2048 block has 128 query tiles and about 256 blocking
pool-tile boundaries per MLA layer; pairing removes up to 128 boundaries. At
32K the corresponding counts are 512 and up to 256 removed. The current selected
whole-attention component costs approximately 129.9 ms per MLA layer at 16K.
A 5–10% inclusive gain would be useful, but is a hypothesis. The historical
525/905-microsecond selector totals are loose upper bounds on available work;
compute and partition remain. This candidate cannot improve the cold 2K dense
MLA branch and does not close the 1500/60 goals by itself.

**One minimal measurement:** load the existing actual 16K fixture; bind the
current accepted packed cadence in both arms; compare the current scorer against
two-pool-tile cadence in the entire `attend` call. Before timing, compare every
BF16 attention output bit and every score/selection value on representative
T16 slices, including a partial pool tile and final unpaired real query. Check
bounded peak memory. Three warmup pairs and eleven fresh AB/BA samples include
selection, gather, SDPA, concatenation, endpoint evaluation and frees. No model
load is needed. Stop after a noisy or losing inclusive result; no wider-row sweep.
A winning arm proceeds to the coordinator's selected 32K qualification.

**Ownership:** worker owns `src/glm5_indexpool_nax.zig` and a new isolated pool-
cadence probe. Coordinator owns admission/counters in normal/HTTP diagnostics.
`src/glm5_attention.zig` and the packed SDPA helper retain the accepted cadence;
use scoped scorer binding so both arms can run in one process. No overlap with
the decode workers' KDA-tree or three-row QKV files.

## Secondary lead: a larger M stripe in existing dense NAX projections

This is a concrete research alternative, not the chosen next implementer.
The current `glm5_a6_dense_once` path dequantizes one original A6 bank to BF16,
then runs native dense GEMM. On the M5 `d` architecture, regular NAX at
M2048/N8192/K4096 (Q/K/V) and M2048/N4096/K8192 (output) selects
BM128/BN64/BK512, WM4/WN2 and **swizzle_log2**. The current physical X groups
visit four M tiles for an N stripe. There are sixteen logical M tiles.

A fixed swizzle_log4 would put all sixteen M tiles in each physical X stripe,
with the same tile body, K iterations, FP32 accumulation and BF16 stores:
QKV grid 512×4 becomes 2048×1 (2048 groups); output grid 256×4 becomes
1024×1 (1024 groups). Threadgroup remains 32×2×4. Only independent logical
output tiles change physical coordinates. Better weight reuse is a hypothesis;
Metal's actual execution order and traffic were not traced. Stock swizzle2
already provides reuse, so this is not removal of an obviously wrong schedule.

**Implementation cost:** there is no MLX-C or Sushi API knob for native GEMM's
swizzle parameter. A Sushi JIT clone of the fixed native dense NAX kernel and
required helpers would be needed, with a same-body swizzle2 control. This is
larger than the scorer schedule change. Patching/rebuilding the shared MLX
runtime during the wave would change the baseline and is not recommended.
The existing affine6 research wrapper copies quantized GEMM and is not a dense
NAX scheduling control; substituting it silently would test another algorithm.

**Evidence and ceiling:** the selected dequant-plus-dense primitive previously
measured 2.530552 ms QKV and 2.712552 ms output at T2048. QKV performs about
137.44 GFLOP, already approximately 54 TFLOP/s inclusive of expansion overhead.
Those rates leave no basis for promising a large gain. Scaling the historical
three-QKV component cost across 34 layers gives about 258 ms, and output another
92 ms; these are rough isolated-component prioritization figures, not current
model attribution. A 10% QKV-only gain would suggest approximately 26 ms, about
1.5% of the old synchronized prompt total, before model interactions.
The cold KDA subtotal supplies a strict loose ceiling of 500.232 ms; it does
not all belong to these GEMMs. Shared and dense FFNs use different quantized
bank shapes and are not automatically covered by this proposed dense clone.

**Geometry and memory:** start with the actual layer-zero 4096→8192 A6/group128
bank and T2048 BF16 input. Keep original packed U32/BF16 scale/bias tensors.
Reuse the existing 64 MiB temporary decoded bank, 32 MiB output and current
async2 expansion bill; add no persistent bank or larger operand tile.
An input/weight clone that introduces extra copies must include them in timing.

**Minimum missing evidence:** one actual-bank inclusive primitive comparison of
current `dequant + native matmul` against `dequant + fixed dense clone`. First
prove the clone's swizzle2 output bits and tile geometry against native, then
compare only swizzle4, three warmups and eleven pairs. Include expansion, native
or automatic copies, construction, evaluation and frees. A clone packaging
penalty can erase the scheduling gain and must not be hidden. No layer/model
claim is warranted before this one new component result. Do not begin this
larger implementation ahead of the fixture-ready scorer candidate.

## Concurrency and final evaluation

The GLM diagnostic HTTP bridge deliberately serves requests sequentially with
one MLX-owning thread and `max_parallel=1`; its four-stream llmprobe result is
not an accidental missing-thread optimization. A second independent full-model
load would exceed this 128 GB host's memory. Batched independent requests would
need per-request MLA selection/offsets, KDA state and speculative transactions;
the current full model rejects batch sizes above one in MLA. That is a separate
serving design, not a small performance workstream for this round. Both proposed
kernel improvements reduce queued request service time without claiming parallel
request throughput. Preserve the single MLX caller invariant.

After researcher 2's decode/speculative recommendations arrive, dispatch two
decode implementers and the bounded-pool-cadence prefill implementer. Each uses
one inclusive paired component gate, without repeating old loaded-model baselines.
Only accepted candidates enter the combined model correctness gate and normal
HTTP evaluation; only accepted changes are pushed. Keep ordinary and predictable
prompt rows separate and record actual dispatch engagement: predictable 2K with
2036 IDs misses the exact-T2048 dense-expansion/retained-cluster guards.
The final selected stack needs one 32K token/complete-state qualification and
normal throughput cell, preserving BF16 compressed MLA and FP32 KDA state.
Normal HTTP numbers from the currently running ladder should be appended by the
coordinator before final performance comparisons are made.
