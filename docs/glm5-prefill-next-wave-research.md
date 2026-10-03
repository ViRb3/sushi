# Next prefill candidate: remove the gathered KV bank

Research only, 2026-10-03. No production edit, prototype, build or GPU run.
Recommend one isolated **indexed NAX prefill attention** experiment: retain the
accepted native attention math but load selected BF16 latent rows directly,
eliminating the temporary gathered KV bank. Speedup is unmeasured.

## Recorded attribution and limits

The exact-T2048 synchronized profile `glm53-next-wave-prefill-components-20261003`
recorded 1788.124 ms prompt time: routed FFN 810.624 ms (45.3%), KDA attention
500.232 ms (28.0%), cold dense MLA 147.899 ms (8.3%), shared FFN 91.919 ms and
dense FFN 36.411 ms. This profile forces evaluation, predates later accepted
scheduling changes and does not split KDA projections from recurrence. It is
not interchangeable with HTTP throughput or a sparse-attention profile.

The remaining long-prefix path has direct actual-input evidence:
`glm53-prefill-cadence-16k-20261003` measured the complete T2048 MLA attention
call at 129.887917 ms per layer with current two-tile cadence and NAX selection.
The actual 8K counterpart was 105.953500 ms. Both include selector, gathered
bank, SDPA, collection, endpoint evaluation and frees. No recorded result
isolates the gather fraction. The inherited accepted HTTP 16K cell was 655.06
prefill tok/s; input counts and later optional native-decode modes must remain
separate in subsequent comparisons.

The largest cold profile bucket is still routed FFN, but near-full routed-grid
and A6 eligibility extensions failed their actual-model gates. Also exclude
word sharing, window64/output-group retuning, pool-tile cadence, R8 recurrence,
retained FA/GA joining and precision restoration. This recommendation attacks
large repeated work in a different, active long-prefix path; it cannot fix cold
2K dense attention or establish 1500 tok/s alone.

## Active call path and removable work

`Mla.applyMode` → `attention.attend` → `attendPackedCadence` →
`glm5_attention_nax_packed.run` → `gather` → native full D512 SDPA.
Every 16-query tile gathers `[16,2051,512]` BF16 KV, about 32.05 MiB, and a bool
mask. K/V alias this bank. A T2048 block has 128 such tiles: **4301258752 bytes
(4.006 GiB) of temporary KV writes per MLA layer**, or 44.064 GiB across eleven
layers, plus source reads and native reads of those banks. These are logical
issued spans, not measured DRAM traffic; cache/coalescing can serve repeats.
Two-tile cadence overlaps construction but retains these copies.

Native `attention_nax_dsplit` loads contiguous `Ktile` fragments before Q@K
and `Vtile` fragments before P@V. Its selected-slot traversal remains 65 BK32
blocks, with BQ32/WM2/WN4 and FP32 score/output accumulators. Replace only those
fragment loads with a bounded accessor using the existing ordered selected IDs
and original cache strides `[512,1]`. All fake query rows remain the original 64
heads. Keep scale 1/16, bool validity/causality, zero-before-load for invalid IDs,
last-slot 2050 behavior and all-invalid output zeroing. Do not scan full history,
change selection order, softmax/reduction order, query tile size or cadence.

## One bounded implementation and decision test

Owner: one prefill worker, isolated `glm5_attention_prefill_indexed` helper/probe
and a reproducible pinned MIT native D512 header subset, with required NOTICE
attribution. Coordinator owns the later narrow packed-cadence delegation.
Do not patch/rebuild shared MLX or touch the qualifying native B1/B3 decode mode.

Start with BF16 T16/H64/D512, K2051, contiguous immutable latent cache and the
existing selected-ID array. Native B16/Q64 uses 32 tensor groups; keep that grid.
First clone the unchanged contiguous-loader native body and prove its complete
BF16 outputs against native SDPA. This guards source packaging/compiler effects.
Then change only K/V address loading, retaining the existing bool mask and math.
No persistent weight/cache bank, full-history cast or floating Q×H×K plane.

Reuse the actual 16K capture; construct selections with the unchanged selector.
Prove every valid BF16 output bit against the accepted gathered path, including
future/invalid IDs, empty rows, valid key zero, masked nonfinite data and ragged
last slots. If compiler arithmetic differs, report raw L2/max/bit statistics;
do not claim exactness or add restoration math. The user's ordinary BF16/FP32
NAX policy permits investigation, but integration still needs explicit drift
and mode/state qualification.

After the loader proof, run ONE inclusive whole T2048 attention comparison on
that capture: current gathered path versus direct indexed loads, with the same
selector and accepted two-tile cadence. Three warmups and eleven fresh paired
samples include construction, selection, all cache accesses, attention, output
collection, endpoint evaluation and frees. Measure actual peak. Stop a noisy
or slower candidate without tile/loader variant sweeps.

Memory should decrease: remove the 32.05 MiB gathered bank per tile, retaining
selected IDs, small masks, queries/output and native register/threadgroup state.
Keep the existing 64 MiB per-tile/128 MiB two-tile admission bound initially; prove
the smaller live bound before reducing it. No hidden full-cache copy is allowed.

The measured 129.9 ms whole-attention call is the only component cost ceiling;
the gather share is unknown. A hypothetical 10% inclusive attention saving would
save roughly 143 ms across eleven layers for one late T2048 block, before pipeline
interactions. It is not a prompt-wide forecast. Indirect fragment loads may
lose contiguity or repeat source fetches across query blocks and erase the
copy saving, so byte counts alone cannot justify integration.

A component winner gets ONE coordinator-owned loaded-model late 16K-block arm
using fixed actual IDs, fresh Requests and equal retained-reference memory.
Require unchanged selectors, final logits and every valid BF16 MLA/FP32 KDA
state byte for an exact arm, plus one 64-token continuation; record old-target
drift separately for any accepted rounding difference. Real repeatable performance
acceptance precedes implementation commit/push. No cold 2K attribution or full
ladder is needed to decide this individual candidate.

Shared-selector approximation is deferred: one common pool bank for adjacent
queries could remove more gathers, but Q1024 has the same 32 native tensor
groups as B16/Q64 and changed per-query retrieval needs a meaningful long-prefix
forced-logit/valid-state consistency gate, not short dense-prefix KLD.
