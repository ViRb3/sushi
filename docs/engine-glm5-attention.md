# GLM IndexPool and latent attention

`glm5_attention.zig` implements request-local IndexPool state and NoPE attention over a single lossless
latent cache. Projection, normalization, stored-affine `kv_b` row splitting, query absorption and output
projection belong to the caller. It does not enable the architecture for production serving or claim
quantized KV-cache support.

## Interface and ownership

`State.init`, `append`, `arrays`, `evaluate`, `reset` and `deinit` manage one request's attention history.
`append` accepts latent rows `[T,D]`, normalized index keys and gate scores `[T,I]`, and positional bias
`[4,I]`. It returns the prior token offset. Latent and pooled storage grow in 256-row blocks; the raw
key/gate tail retains zero through three rows in compact owned arrays. There is no duplicate value cache.
The official latent/index widths are 512/128; BF16 and FP32 are supported for reference tests.

An append replaces state handles only after its graph is constructed successfully. Invalid shapes and
construction failures leave existing state intact. GPU execution is lazy: a later evaluation failure must
invalidate or reset the enclosing request. The model loop must evaluate final output and all non-null
`State.arrays()` together at its boundary, then release layer temporaries. `evaluate()` is a standalone
convenience when no enclosing output needs evaluation.

`attend` takes absorbed queries `[T,H,D]`, optional index queries `[T,J,I]`, optional pre-scaled index
weights `[T,J]`, the prior offset, and the attention scale. GLM's scale is **1/16**, based on its original
256-wide queries rather than the 512-wide latent. The caller applies the index weight factor
`1/sqrt(J*I)` and casts to the index-query dtype before dispatch.

## Pooling and causal selection

Pooling follows the reference tensor operations: add the four-row positional bias to gate scores,
softmax over the pool axis, multiply normalized keys, then sum the four rows. Partial windows carry
across calls and reset with the request.

Index scores sum weighted, nonnegative per-head query/key dots. For BF16 inputs, head dots and weighted
products round to BF16 at the corresponding reference boundaries before the final head reduction.
Weights may be negative. A pool is eligible only when all four tokens are at or before the query.
Selection retains at most 512 completed pools, expands them to token IDs, then adds the incomplete
zero-to-three-token tail once. Invalid slots carry `-1`.

With always-selected tails, every query through position 2050 has at most 512 completed pools, so its
selected set equals the dense causal prefix. The first selective row is position 2051, when the 513th
pool is complete. The state still accumulates pooling history throughout the dense prefix. A chunk
that crosses this boundary uses the same causal eligibility rule for each row.

## Bounded scratch

Attention uses FP32 online softmax over the latent cache and emits split partial numerators and
normalizers, then merges them into the query dtype. It does not construct a query-by-head-by-history
score tensor or a full-history boolean mask. Short blocks use eight splits; wider blocks use one.

Query chunks cap each index score plane at 2 MiB and attention partials at 8 MiB. Index scoring also
needs its negated scores and partition indices, about three such planes in total, plus at most roughly
1 MiB of expanded token IDs for the 128-row chunk cap. Allocator overhead and MLX partition-internal
workspace are additional; these figures describe explicit graph arrays rather than measured process
peak memory. Multi-chunk calls evaluate each output before releasing that chunk's scratch. Single-chunk
calls remain lazy for the enclosing layer boundary. A context that cannot fit even one row inside the
explicit scratch limits fails rather than silently exceeding them.

## Tests and limits

The focused ReleaseFast tests cover:

- Scalar pooling with irregular chunks of one, three, one and four tokens, including reset.
- Scalar causal attention at lengths 1, 3, 4, 5, 2048 and 2049.
- Exact selected token sets across positions 2047–2054, excluding future pools and duplicate tail tokens.
- BF16 width-512 attention with byte-identical whole versus irregular-chunk results in the tested case.
- Exact FP32/BF16 scalar index scores with positive and negative weights and causal masking.
- Rejection of invalid append/attention inputs without advancing request state.

The GLM-filtered unit suite passed after module integration. Separately, the diagnostic MLA binder's
nonzero two-head/two-token affine fixture verifies both stored-grid matrix orientations against an
independent dense dequantized calculation; malformed projection and normalization geometry is refused.
These checks do not establish full-model parity. The reference's dense prefill path expands keys and
values before attention; query absorption changes BF16 rounding order and requires independent
layer/logit validation. No decode/prefill throughput or quality result is implied by these primitive tests.
