# Shared-bank sparse prefill research

Research only at accepted runtime `e1597cc2`, 2026-10-03. Recommend one bounded
**opt-in four-query anchor retrieval** experiment. Speed and quality are
unmeasured; adjacent selection overlap is not established by existing artifacts.
No prototype, build, GPU run or default change accompanies this recommendation.

## Evidence and ceiling

Current two-tile contiguous-bank attention took 129.888 ms per MLA layer on
the actual T2048/16K capture. The fresh indexed-loader control took 130.376 ms;
direct original-cache loads took 241.391 ms, 85.15% slower despite exact
selection/output. Preserve contiguous gathered banks rather than reopening
that loader. These complete calls include selection, gathering, attention,
collection, settlement and frees; neither artifact isolates gather time.

The cold T2048 synchronized profile charged routed FFN 810.624 ms (45.3%),
KDA 500.232 ms (28.0%) and dense MLA 147.899 ms (8.3%) out of 1788.124 ms.
It predates later scheduling changes and is not HTTP sparse attribution.
This candidate cannot improve the first dense chunk or eliminate routed/KDA
work. Near-full routed-grid/A6 and optimized N3 already failed model gates.
Even eliminating the whole measured late attention call would save at most
about 1.43 s across eleven MLA layers for one late T2048 block. The removable
fraction is unknown; this alone does not establish 1500 tok/s prompt-wide.

## One policy and its geometry

`Mla.applyMode` → `attention.attendPackedPairs` → `selectChunk` →
`glm5_attention_nax_packed.run` currently selects 512 four-token pools for each
real query, adds up to three unpooled tail tokens and gathers separate
`[16,2051,512]` BF16 banks. K/V alias each bank. Native Q64 represents the
query's original heads, with array masks and scale 1/16.

Use fixed groups of four **absolute** positions aligned to the pool boundary.
The first query selects its usual 512 completed pools using its own original
index query/head weights. Preserve the anchor's pool ordering. Gather those
2048 historical rows once, then append the four group-local raw rows once:
K2052. Each query masks local rows after its own position. Historical pools
finish before the group starts, so local IDs do not duplicate historical IDs.
The first anchor never depends on a future query or completed future pool.
At the fourth position the local four-token pool is always available, even
if that query's independent selector would omit it; this is part of the
explicit retrieval approximation. No score averaging or union/reranking.

Use original BF16 queries/cache/output and FP32 native accumulators. Pack four
real queries as Q256; their 64 heads remain consecutive. Sixteen such groups
use native `[16,1,256,512]` Q and `[16,1,2052,512]` KV. The native BQ32/BK32
D512 geometry has 128 tensor groups, exactly the original total for 64 real
queries. Dot/softmax/value arithmetic is not reduced: both K lengths need 65
BK32 blocks. Benefits must come from fewer selector rows, bank writes and
commands/cache working sets, not a claimed tensor-core group reduction.

Process sixteen anchors spanning 64 real queries per graph; retain the current
two-graph settlement bound. This is necessary because `tryScores` declines
eight-or-fewer rows: naive four-anchor/16-query tiles would fall back to scalar
scoring at 16K. Original NAX scoring can retain sixteen rows, but its epilogue
must use explicit anchor positions (`offset+4*row`), not `offset+row`.
Reference anchor scores/IDs against the original selector at those positions.
Do not silently change NAX/scalar numerical mode. Alignment fragments and a
final incomplete group use the existing path; only complete aligned groups
enter this fixed policy. Decode/verifier retrieval stays unchanged.

For T2048, shared-bank writes fall from 4301258752 to 1075838976 bytes,
about 4.006→1.002 GiB per MLA layer, and selector rows from 2048→512.
These are logical spans, not measured DRAM or a 4× latency forecast. One graph
has about 32.06 MiB KV, 4 MiB Q, 8.016 MiB explicit bool mask and 4 MiB output;
native temporaries and selector ownership require a measured peak. Keep the
conservative 64 MiB per graph/128 MiB pair bound only after proving it fits.
No full-history copy, persistent bank or floating Q×K plane is allowed.

## Quality and decision gates

One prefill worker owns an isolated helper/probe and fixed policy; coordinator
owns later integration, admission and quality/model runs. Reuse the existing
16K fixture (Q/index-Q/weights/pooled/latent) for component evidence; it contains
no saved overlap statistics. Before timing, report per-nonanchor retained-pool
recall, score-weighted missed mass and worst rows against independent selection.
Low overlap, rapid topic shifts, code identifiers, far-back exact retrieval and
head-weight differences can make an anchor a poor proxy. Similar adjacent
positions are not evidence of similar selected sets. Recall is diagnostic,
not a replacement for logits.

Prove original anchor score/ID bits, ID uniqueness, pool/tail boundaries,
causality, query/head order, partial-group fallback and empty masks. Shared
future local V rows introduce a specific hazard: native masked `0*NaN` can
poison earlier queries. Require a safe nonfinite-input decline/fallback or
prove the native operation protects this case; reject rather than silently
sanitizing valid values. Contiguous masked reference attention using precisely
the same approximate IDs is the component arithmetic oracle, not independent
baseline retrieval. No precision restoration.

Run one complete T2048 comparison on the actual 16K planes, three warmups and
eleven alternating pairs, including all anchor selection, position metadata,
gather/masks, SDPA, output collection, endpoint evaluation and frees. Record
actual peak and all fallback/engagement counts. Stop a noisy/slower component;
no group-size or bank-policy sweep follows.

A component winner needs **long-prefix forced-logit KL against current runtime**
before acceptance. Use fixed nonrepeated 16K natural/code and far-back retrieval
inputs, sampling a declared set of late-prefill positions plus 192 baseline-
forced continuation tokens. Both arms process identical IDs; report mean and
tail/worst KL, NLL delta, top1 agreement and retrieval-sensitive rows, with a
coordinator-declared quality bound fixed before running. Accumulate in bounded
logit chunks. Short 512-token dense-prefix KLD does not exercise this policy;
greedy agreement or attention RMS alone is insufficient. This paired runtime
drift test does not replace the checkpoint's lossless-teacher pack KLD.

Then one loaded-model matched long-prefix ABBA compares actual prefill time
and memory with equal held-reference allocations. From each approximate
prefill snapshot, strict serial/speculative token and complete valid BF16 MLA/
FP32 KDA state parity is required on that same approximate prefix. Baseline
versus approximate state equality is not expected. No future selector reuse
enters decode, and no extra precision or changed proposal policy is introduced.
Only a repeatable speed/quality winner can proceed to opt-in integration and
commit; cold/routed attribution and both throughput goals remain separate.
