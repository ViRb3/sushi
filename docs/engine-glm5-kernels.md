# Engine: GLM-5.3 native kernels

What the native `glm5_next` forward runs in prefill, decode and DFlash2 verification, the arithmetic contract of each
path, and the alternatives that lost. Every path here is always on for its eligible shape; anything else falls back to
the staged MLX chain. Read this before touching `src/glm5_*.zig`. Architecture, bills and numbers:
[arch-glm5-next](arch-glm5-next.md); routed experts: [engine-exl3-experts](engine-exl3-experts.md#glm).

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kernels](engine-kernels.md),
[engine-mlx-gotchas](engine-mlx-gotchas.md).

## Code map

| File | Role |
|---|---|
| `glm5_forward.zig` | `Model`/`Request`, layer loop, routing, async2 prefill and async4 decode schedules, captures |
| `glm5_model.zig` / `glm5_next.zig` | KDA layer, dense MLP, affine linears; KDA recurrence, HC collapse/expand primitives |
| `glm5_attention.zig` | IndexPool state, absorbed latent attention, packed prefill orchestration |
| `glm5_attention_nax_packed.zig` / `glm5_indexpool_nax.zig` | head-packed native sparse attention (B16/B32); NAX prefill index scores |
| `glm5_attention_decode_batch.zig` / `glm5_attention_overlay.zig` | native B1/B3 decode and verify attention; verify latent overlays |
| `glm5_mla_prefill_batch.zig` / `glm5_mla_verify_batch.zig` | head-batched MLA prefill projections; three-row verify projections |
| `glm5_kda_prework.zig` / `glm5_kda_value_rows.zig` / `glm5_kda_fused.zig` / `glm5_kda_prefill_cluster.zig` | KDA prework, R4 recurrence, one-token body and output epilogue, FA/GA/beta cluster |
| `glm5_a6_dense_once.zig` / `glm5_decode.zig` / `glm5_router.zig` / `glm5_activation.zig` | T2048 A6 expansion; copy-free QKV; router; dense/shared activation |
| `glm5_hc_prefill.zig` / `glm5_hc_collapse_simd32.zig` | RMS-fused HC prefill projection; SIMD32 verify collapse |
| `glm5_dflash*.zig` | assistant adapter, tree, layerwise verifier, KDA tape, FFN, row projections, A6 hoist, reserve, scratch, local cache |
| `glm5_stream.zig` / `glm5_vision.zig` | BF16 expert streaming (teacher); vision tower and processor |

## Contracts

- **Exact** means every output and state bit equals the staged chain (for verify: the serial row), proven on real
  checkpoint captures as well as synthetic fixtures. A dispatch counter proves engagement; a flag never does.
- Four paths change numerics (BF16 operands, FP32 accumulators, NAX reduction order): head-batched MLA prefill
  projections, NAX prefill index scores, packed prefill attention and native B1/B3 decode attention. The owner accepted
  plain NAX rounding on same-pack forced-logit drift checks that predate the frozen screen below (MLA projections at
  4K/64: mean KL 0.0116, max 0.396; the prefill trio at 16K/32: mean 0.0040, max 0.068; B1/B3 at 8K/64: mean 0.0032,
  max 0.048). B1 decode attention runs inside the 4x512 teacher KLD; the three prefill paths engage only past the
  KLD prompts' lengths, so none of them has a long-context teacher KLD.
- **Drift screen** for a numerics change: two frozen nonrepeated 16,384-ID prompts (code, prose), 24 late-prefix plus
  192 forced rows each; per prompt mean KL ≤ 0.01, max KL ≤ 0.15, top-1 ≥ 95%, mean NLL increase ≤ 0.02, no new
  nonfinite. Bounds are fixed before results; late-prefix rows are far more sensitive than forced continuation.
- One numerical target across serial (M1), verify (M3) and partial rounds; a native mode that cannot run raises a mode
  error instead of falling back to scalar math.
- An append swaps request state handles only after its graph builds; a lazy evaluation failure leaves the request
  unusable until reset, never half-advanced.
- Scalar fallback attention runs FP32 online softmax over the latent cache with split partials merged into the query
  dtype; no query×head×history score tensor or full mask exists, and per query chunk the index-score plane is capped at
  2 MiB and the partials at 8 MiB.
- **Model gate**: a component win counts only if a loaded-model ABBA gain exceeds control drift,
  |A_last − A_first| / mean(A), on each workload. Several 8–26% component wins below failed that gate.

## Prefill (chunk 2048, two layers pending)

- **Cold MLA** (BF16, more than 8 rows, ending at or before token 2051): latent expanded through the per-head K/V banks
  (64 MiB each at T2048) into native causal SDPA D256, `force_fused`.
- **Absorbed MLA projections**: query absorption (256→512) and value unembed (512→256) run head-batched (`[64,T,D]`)
  as native affine NAX QMM on A6 g128 banks, T128–2048. T2048 query 13.69 → 1.69 ms, value 19.45 → 1.64 ms; rel L2
  0.26%/0.32%; 4K/64 drift mean KL 0.0116. 768 MiB of copies at async2.
- **Index scores** (9–16 query rows, 3584–8192 completed pools): Q `[T·32,128]` × pooled-key tiles of at most 2048
  pools, each tile settled before the next (dot plane ≤ 2 MiB), scalar epilogue kept (BF16 dot, BF16 ReLU·weight,
  sequential FP32 32-head sum, BF16 total, −inf for future pools). 0/7/6 of 16K/65K/131K scores differ, every
  512-pool set kept; selector −32% at 16K/32K. Below 3584 pools full tiles waste the gain; above 8192 scalar runs.
- **Sparse attention**: per real query, gather its 512 pools plus raw tail (2051 latent rows) into one BF16 bank used
  as K and V, native SDPA at scale 1/16 for at most 16 queries per graph (16K: 0.69 vs 5.22 ms scalar). Two such graphs
  stay in flight (−18.6% at 8K, −6.9% at 16K), and exactly 32 rows join two unchanged T16 selections into one B32 call
  (whole attention −15.4%, 16K model prefill −4.9% vs 1.95% drift, exact against B16). Invalid and future slots are
  zeroed before load, all-invalid outputs after; MLX's automatic input copy is off for gathers.
- **KDA**: FA/GA/beta read the same normalized input, so at exactly 2048 rows they run as one prepared BF16
  `[320,4096]` NAX GEMM with compact outputs (exact, −6.9%; 85 MiB resident, built at load on a NAX GPU). A6 QKV and
  output banks are dequantized to temporary BF16 and multiplied by dense NAX at exactly 2048 rows (exact, −7.4%/−5.3%;
  512 MiB at async2). Prework is one dispatch: conv4, BF16 SiLU, Q/K L2, FP32 decay, BF16 beta (−74%). The recurrence
  carries four value rows per SIMD group sharing K/decay/Q loads, one `simd_sum` per member, FP32 state (−22% at 512,
  −34% at 2048 tokens). The output epilogue fuses FP32 RMS, norm weight and sigmoid gate. The conv history keeps raw
  BF16 bits (integer copies of the last three QKV rows: a float round trip flushes subnormals and signed zeros).
- **HC**: at 128+ BF16 rows one kernel fuses widening, RMS (MLX's 1024-logical-thread reduction on 128 threads) and
  the 24-output projection with shared accumulators (exact, −51% at 512 rows).

## Decode (one evaluation per token, submitted every four layers)

- Async4 is bit-identical to synchronous layers but may hold two cache generations.
- KDA QKV in one dispatch over the three resident A6/A8 banks (restricted port of oMLX `multi_qmv`; a concatenated
  bank copy would cost 3.29 GiB), then the fused one-token body: conv, BF16 SiLU, L2, per-channel forget gate, delta
  recurrence, gated output norm. Its unary math variants are chosen by probe at first use.
- Router: FP32 GEMV with sigmoid and correction, then stable top-8 and unbiased normalization (two dispatches; BF16
  weights widened locally). Dense/shared activation: one dispatch over the exhaustive BF16 sigmoid table (128 KiB).
- Attention: native B1 per decode row and B3 for three verify branches, from one gather source (`[B,2051,512]` bank,
  Q `[B,1,64,512]`). B3 equals three B1 bit for bit; against the old scalar split-8: rel L2 0.0013, 64-position mean KL
  0.0032, top-1 61/64. 32 MiB at async4.
- HC collapse on three verify rows: one SIMD32 subgroup runs the coefficients and 20 Sinkhorn iterations that thread 0
  ran alone while 255 threads waited (exact; −27.6% component, 8K model −2.65% vs 1.99% drift).

## DFlash2 verification

- **KDA**: parent-indexed prework over 1–16 nodes, then a tree recurrence holding FP32 parent states locally; the tape
  keeps projected prework and replays only the accepted path. The first-child leaf (chain row 2, fork row 1) is kept and
  aliased on a hit (4 MiB per layer; −25%; 59% hits at 8K, break-even 21%).
- **MLA**: trees of at most three nodes read the committed buffer plus a ≤3-row ancestry tail instead of a replaced
  latent buffer (1.97× at 32K); query and value projections broadcast the one-row geometry over three rows (exact,
  −3.6%). Accepted rows append at commit. Live branch scratch is capped at 256 MiB: three branches fit through 64K,
  one at 128K; branch groups that do not fit settle in turn and B3 falls back to per-node B1.
- **Projections**: affine row tiles reuse each weight group across up to four rows in serial qmv order; three-row A6
  QKV hoists coefficient decode out of the row loop (exact, −12.4%); the retained BF16 KDA projections run as column
  GEMVs with rows in the batch grid (exact; stock multi-row `Linear` is not); the router batches up to 16 rows.
- **Assistant**: only draft positions 1–2 reach the vocab head, since N2 visits depths 0–1 (exact, −58% readout);
  the temporary 8-row block attends a read-only slice of the last 2047 context rows (assistant forward 11.4 → 4.6 ms at
  32K; assistant rounding changes, target exact); the next context is cropped to 2047 rows before accepted captures
  append (50 MiB bound at any length; commit 4.27 → 0.73 ms at 32K; exact).
- After prefill, latent and pooled capacity for input + max output + 3 is reserved once, so verification never grows
  a buffer. Every array a replay needs is an async dispatch output.

## Ruled out

Prefill attention and index:
- Masked full-history NAX D512 attention: 2.2× scalar at 32K but visits every history tile; exact-membership variant
  37% slower at T2048/16K (7.9× the K tiles).
- Indexed K/V fragment loads instead of the gather: exact, 85% slower (130.4 → 241.4 ms).
- Exact radix top-512 pool selector: +1.6%, 2/11 (the pinned ArgPartition sorts the whole axis anyway).
- Two-tile cadence inside the NAX index scorer: −16.3% component, −0.45% at model level inside monotonic drift.
- Index-score NAX cap 8192 → 8448 pools for the 32K tail: −9.7% component, code max KL 0.345 on the 32K screen.
- Four-query shared-bank retrieval: −53% attention, code mean KL 0.513 (70% recall still fails).
- Cold absorbed D512 packed MLA instead of expanded K/V: +523%.
- Direct single-split prefill finalization: exact, no controlled win ever measured; removed.
- Steady 4096-row chunks after the first 2048: moved IndexPool NAX engagement, code mean KL 0.034.
- Head-batch hi/lo precision restoration: less drift, several times slower; plain NAX rounding accepted.

Prefill KDA and trunk:
- Staged (threadgroup) recurrence schedules: +7.8% to +65%; threadgroup height 8/16: −3–4%, superseded by R4.
- R8 value rows: −1.6% (8/11), about 2.4 ms per T2048 prompt.
- Temporal 128-token KDA tiles: +11.8%, 0/11.
- A6 dense expansion at T1536–2047: −7.9% component, +3.1% on the equal-memory model gate.
- Request-lifetime BF16 copies of all 136 A6 banks (8.5 GiB): +0.36%, 4/11.
- Affine QMM tile swizzle (≤1%), tile aspect (wide +16–21%; tall WM2/WN2 wrote zeros from row 16), A6 word unpack and
  BM128 tiles (−3–5% primitives, never model-gated).
- Four-output HC prefill expansion: −0.43% component, +0.34% at model level vs 1.9% drift.
- HC projection with 6 or 12 columns per group: +26–28% / +9%. Factored RMS after a BF16 NAX projection: failed the
  drift bound on cancellation inputs.

Decode and verify:
- One-token HC norm/mix fusion: −8% queued component, model 26.49 vs 26.54 tok/s; removed. Short-row HC collapse +
  RMS fusion: 0.4 µs per call, never integrated.
- Joined QKV dispatch (3–4 rows: +0.7–3.0%; later over the three hoisted A6 banks: −0.57%, 6/11): noise.
- Raw A6 four-product unpack: +1.3% (5/11). A6 hoist on the output projection: −2.8%, 7/11, drifting.
- Per-node packed NAX decode attention: +23–60%, 0/6 in all nine cases. Shared-factor split-8 merge: +2–5%.
- Short-row NAX index scorer (M128/K128/C2048): +14%, 0/11. Four-subgroup scalar scorer: 0.83% slower paired.
  Shared-prefix three-query scoring: +18%, 1/22.
- Fused T3 KDA core (prework + leaf recurrence + post): −0.7%, 6/11; removed. Canonical chain/fork recurrence:
  chain 6/11, fork slower. Keeping all three T3 endpoints: −1.5% (6/11) for +272 MiB.
- Retained KDA projections: native NAX batch +5.7%; FA+GA N256 column join +3.7% (N320 changes the GEMV kernel).
- Head-rebatched three-row MLA value QMM: switches to `qmv_wide`, not exact.
- In-place accepted latent append: a shared clone blocks MLX donation; at most 2.2 ms per round at 32K.
- Draft head shortlists (3-bit top-32 over 7 rows; A3 top-32 over 2 rows): −26% readout, decode gain inside drift,
  +265–278 MiB resident; removed.
- Wider trees: N3 with every T4 kernel optimized 40.44 vs N2 40.22 tok/s at 8192 IDs (`e1597cc2`, 1.46% drift);
  N4 31.4 vs N2 42.4 tok/s at 512/64. Verify per round grows faster than acceptance.
- Per-kernel attribution tools: synchronizing verifier markers (halve throughput), xctrace Metal System Trace (no
  shader names or intervals; 95.4% GPU busy overall) and private MLX timestamp hooks (only GPUTimestamp; overlapping
  command-buffer intervals). None attributes decode time by kernel.
