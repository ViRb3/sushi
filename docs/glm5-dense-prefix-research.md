# Dense-prefix native packed research

The accepted HTTP cold prefix already uses native attention. A packed dense
prefix is a reassociation experiment, not a missing-native-dispatch fix. Recommend
at most one fixed T2048 complete-MLA component gate; do not enable it from source
counts alone. Research checkpoint `2749c30d`, accepted runtime `e1597cc2`; the
separate B32 candidate is still awaiting model acceptance.

## Current route

`glm5_bench_http.zig` sets `request.dense_prefill=true`. In
`glm5_forward.zig`, `Mla.densePrefillEligible` admits BF16 rows greater than eight
ending at or before 2051 with D256 keys/values. `applyMode` skips absorbed Q,
appends the original BF16 compressed cache, then calls `densePrefill`. It expands
that cache through the original per-head K/V banks and invokes native
`mlx_fast_scaled_dot_product_attention` with causal masking and `force_fused=true`:
Q `[1,64,T,256]`, K/V `[1,64,N,256]`. At cold T2048, expanded K and V each hold
64 MiB. The unchanged output projection follows.

The `sparse` gate in `glm5_attention.attendImpl` excludes head packing for an
absorbed dense fallback, but the accepted HTTP first chunk normally bypasses
that function. The existing direct-prefill option is the scalar online body
with exact one-part finalization; it is not another native dense kernel. The
isolated full-history mask helper is not delegated from the accepted runtime.
Native B1/B3 decode covers at most eight rows and is outside this proposal.

## One bounded proposal

For cold B1/T2048 only, H64, latent D512, original A6/group128 banks and BF16
inputs/cache, use the existing head-batched Q absorption and value projection.
Keep every key: construct explicit contiguous ascending IDs `[16,2051]` per
fixed B16 tile, then reuse the existing masked gather and native packed SDPA.
For real query r, live IDs are exactly `0..r`; future/out-of-history IDs become
zero before any cache load. Do not use a causal mask over fake Q positions:
those positions are heads. The gather's per-real-row mask supplies causality.

Native shapes remain Q `[16,1,64,512]`, shared K/V
`[16,1,2051,512]`, mask `[16,1,1,2051]`, `force_fused=true`, BF16 operands and
native FP32 accumulators. Keep exactly two pending graphs, settle their outputs,
release both banks/ID planes, retain only output handles for final concat. No
broadcasted cache expansion, selector calls, native port, new weight packing or
B32 dependency. All other chunks keep the accepted route.

This replaces expanded K/V projection plus native D256 attention with absorbed
Q projection, native D512 attention and value unembedding. It does not reduce
dot work by definition: the attention width doubles, and the fixed 2051 bank
still writes 4,301,258,752 bytes over the full call, including zeroed future slots.
At most two B16 banks remain live. Each bank is 33,603,584 bytes; each explicit
ID plane is 131,264 bytes. Use the existing 64 MiB graph reserve, hence 128 MiB
per pending MLA layer. The existing head-batched projection copy bill adds
384 MiB per T2048 layer; conservatively retain both bills for async2 (1 GiB)
alongside all other reserves until actual peak/lifetime proof. These are bills,
not an assertion that all copies are simultaneously live.

## Decision gate and risks

Expanded K/V and absorbed Q/V introduce different BF16 rounding points;
`glm5_mla_reference_test.zig` explicitly declines bitwise equivalence between
them. Accepted head-batched Q/V also changes native affine contraction rounding.
Keep original small BF16/F32 weights, BF16 cache and FP32 accumulators/state;
no precision restoration. Report every raw output mismatch, relative L2,
maximum error and nonfinite count against current expanded native MLA. Prove
complete ascending key membership, head order, future NaN/Inf zero-before-load,
valid historical nonfinite visibility, boundary 2051/2052 fallback and alias
release separately from finite numerical drift.

One actual cold T2048 complete MLA-layer gate must include Q absorption,
explicit IDs, all copies, gather, two-graph waits, SDPA, value projection,
original output GEMM, evaluation and frees. Use three warmups and eleven paired
arms against the current expanded-native path; stop mismatch/noise/loss with no
geometry variant. A clear winner then needs one loaded-model ABBA on the exact
same cold IDs with equal held reference memory. Before acceptance, test fixed
code/prose final-logit KL/top1 plus 64 teacher-forced continuation positions,
valid cache/state structure and candidate serial/spec token/state consistency.
Do not require cross-mode state bits if logits already differ; expose the drift.

The recorded synchronous T2048 profile assigns 147.899 ms to eleven complete
cold MLA layers out of 1788.124 ms, including projections and forced waits.
It neither isolates SDPA nor predicts normal throughput. Even removing that
entire subtotal would remove only 8.27% of that old diagnostic total. Existing
long-prefix packed wins therefore provide no evidence that this cold expanded
native path will win; the bounded whole-layer gate is the deciding evidence.
