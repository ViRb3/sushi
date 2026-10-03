# Coupled short-row NAX IndexPool scoring

Research only on accepted runtime `e1597cc2` (documentation HEAD `88f 19e 22`),
2026-10-03. Recommend one isolated fixed-geometry decode/verify scorer, not a
shared-SCORE3 retry. No prototype, build, GPU job or runtime source change was
performed. BF16 cache/operands and FP32 accumulators/state remain; there is no
precision restoration or speed forecast.

## Why this is a distinct candidate

`glm5_indexpool_nax.tryScores` admits only T9–16 and at least 3584 pools, capped
at 8192. B1 serial/replay and B3 verification still use the scalar 32-head history
scan. Its dot is rounded to BF16, each clamped weighted term is rounded again,
heads accumulate sequentially in FP32, then the total is rounded to BF16 and
stored in FP32. Native matmul changes the dot reduction, so this must be a
coupled opt-in numerical target, with matching serial/replay math.

Recorded cost is limited evidence. The actual 16K complete three-branch native
attention fixture cost 769.667 µs/layer, including scalar selection, gather,
SDPA, one endpoint settle and frees (`glm53-decode-true-batch-20261003`,
foreground/exclusive GPU). That entire component, roughly 8.47 ms across 11 MLA
layers, is a loose ceiling; scoring alone is smaller and unmeasured. Synthetic
T16 scalar/native selection measured 775.709/525.396 µs at 16K and
1312.583/905.271 µs at 32K (`glm53-indexpool-nax-20261003`). Those M512 measurements
cannot be divided by 16 or projected onto this M128 arm. The accepted native
16K HTTP verifier averages 58.648 ms/round; its context growth is not an
indexer-only removable budget. Previous pool cadence and shared scalar scoring
were rejected and are not reopened.

## One fixed padded geometry

Pack up to three `[32,128]` index queries into BF16 `[128,128]`, padding unused
query/head rows with zero. Both B1 and B3 always call native matmul with
M128/K128/C2048, BF16 raw dots and native FP32 accumulation. Always pad the
final key tile to 2048 columns; changing only actual branch count or remaining
pool columns must not select another GEMM geometry. Fixed shape is necessary,
not proof: compare B1 row 0 with every B3 query slot and suffix-column placement.

Use the immutable common completed-pool prefix `P0=floor(prefix_rows/4)`.
Each at-most-three-node ancestry can contribute zero or one distinct completed
pool. Compute common-prefix dots once for the eligible queries. If a suffix
exists, one separately padded 2048-column plane holds the up-to-three original
branch suffix keys; extract each branch's own dot column. Do not substitute
prefix keys for newly compressed branch keys or copy the pooled history.
Keep the existing epilogue's BF16 boundaries, negative-weight handling and
sequential 32-head sum. Per-branch completion and causal masks use its actual
offset/length, never `offset+row` for forks.

First admission is B1/B3,32 index heads,D128, contiguous BF16 Q/keys/weights,
3584–8704 completed pools on the qualified GPU. The upper bound covers the
recorded approximately 32K frontier plus output growth; the prefill cap 8192
would miss late 32K decode. It is a guard, not another tuning axis. Eligibility
is evaluated per logical query. At a boundary where one branch qualifies and
another does not, qualifying slots use this same padded NAX path and others
use the original scalar path, exactly as independent B1 execution would.
Never flip all three queries according to batch size or maximum branch history.

After scoring, slice each branch to its exact `Pbranch` and retain its original
negative/argpartition/top 512/2051-expansion calls and order. Partition padding
is forbidden: extra −infinity entries can change tie ordering. Device/dtype/model
admission declines the complete optional mode when matching B1 is unavailable.

## Bound and one qualification

A padded query is 32 KiB, a 2048×128 BF16 key tile 512 KiB, and its 128×2048 BF16
raw dot plane 512 KiB (below the existing 2 MiB per-plane cap). Three FP32 score
rows at 8704 pools total 102 KiB. Start with the existing bounded tile settlement
pattern and an extra conservative 8 MiB/pending MLA-layer scorer bill,32 MiB
async 4, separate from native attention's existing 8/32 MiB. Retain no more than
the current tile's raw plane and bounded key/suffix scratch. Confirm actual
high-water and failure cleanup; no whole-history copies or expanded attention
score plane. New tile waits may erase the arithmetic benefit, so measure the
complete attention path before considering cadence changes.

Reuse actual 16K Q/index-Q/weights/pooled/latent captures, with explicitly
constructed chain/fork suffixes. Prove all B1/B3 score bits, ordered selected IDs,
attention outputs and shortened-tree fallbacks match within this mode. Include
negative weights, cutoff ties, dummy query lanes, odd pool boundaries, mixed
3584 eligibility, a frontier above 8192 and invalid/future suffix masks. Differences
from scalar scores/selection are recorded separately; no equality tolerance is
relaxed to hide a target change.

One whole-attention comparison uses the existing native-attention target in
both arms, old scalar scoring versus this scorer, including packing, suffix
preparation, scoring/partition/gather/SDPA, one endpoint settle and frees.
Three warmups and eleven fresh AB/BA pairs at one 16K fixture decide the arm;
stop a noisy/losing result, with no shared-SCORE or tile-width variant.

A winner receives one coordinator-owned native-B1 versus B3 token and complete
valid-state-byte gate. Then a meaningful 16K test starts from the same unchanged
16384-ID full-chunk prefix and teacher-forces 64 old-scorer greedy inputs through
the new B1 scorer. Record KL mean/max, top-1 against teacher next-output IDs,
nonfinite values, selected-ID drift and actual scorer engagement. The existing
8K attention drift test does not qualify this 14K-threshold scorer. This is
same-quantized-model kernel drift, not a lossless teacher. Only a matched
warmed 192-output actual-model result with identical assistant/other flags can
justify acceptance; root owns admission, integration and the final 32K gate.
