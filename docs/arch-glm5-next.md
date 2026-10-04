# GLM-5.3-Flash (`glm5_next`)

The native GLM path: checkpoint geometry, what `sushi serve`/`run` load and bill, DFlash2 speculation, recorded speed
and quality, and the lessons the bring-up paid for. Per-kernel contracts and the alternatives that lost are in
[engine-glm5-kernels](engine-glm5-kernels.md); the clamped EXL3 expert chain is in
[engine-exl3-experts](engine-exl3-experts.md#glm).

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-memory-admission](engine-memory-admission.md),
[quality-kld](quality-kld.md), [engine-expert-streaming](engine-expert-streaming.md).

## Scope

- Concurrent requests interleave; they do not batch yet ([concurrency](#concurrency)). MTP and RAM/disk prefix reuse
  are off: native KDA state has no prefix-cache restore and the checkpoint's MTP layer is not integrated.
- Cache: kv8 compressed MLA latent plus FP32 KDA state, the engine default; it passed its KLD gate
  ([quality-kld](quality-kld.md)). `--kv-quant 16` (or `kv_quant: 16` per request or in model-settings) keeps the
  latent BF16. The pooled index and KDA state stay lossless. kv4 is refused (`GlmKvQuantUnsupported`), and the KLD teacher capture refuses any kv-quant.
- Thinking: `low`, `high`, `max` (the template's `effective_reasoning_effort`); Sushi defaults to `high` (the HF
  template defaults to `max`); thinking off is refused. The effort words impose no token cap.
- Image and video input through the native tower, on when present; `--no-vision` drops its weights and buffers.
- DFlash2 speculation when an assistant is found ([below](#dflash2)), for greedy and sampled requests.

## Checkpoint geometry

| Part | Shape |
|---|---|
| Trunk | 45 layers, hidden 4096, vocab 154880, `max_position_embeddings` 1,048,576 |
| Attention mix | 34 KDA layers; 11 MLA layers at 3, 7, …, 43 (every fourth) |
| FFN | layers 0–2 dense (width 12288); 3–44 MoE: 288 routed experts, top-8, width 2048, one shared expert |
| Routing | sigmoid scores; the FP32 correction bias picks the top-8 only; unbiased scores normalized, times 2.5 |
| Expert activation | gate upper clamp and symmetric up clamp at 10 before SwiGLU |
| mHC | 4 residual streams; per layer two collapses (24 FP32 mix values from `[4×4096]`, Sinkhorn 20 iterations) and two expansions: 90 collapses per token |
| KDA | 64 heads × 128, short conv 4, per-key-channel FP32 decay `exp(−5·sigmoid(exp(A_log)·(a + dt_bias)))`, beta, FP32 state, gated output norm |
| MLA | NoPE; q LoRA 1536, one 512-wide compressed latent per token, 64 heads with 256-wide q/v; queries absorbed into latent space, scale 1/16 (from the 256-wide query, not the 512 latent) |
| IndexPool | 32 index heads × 128; keys pooled every 4 tokens (positional bias, softmax over the pool); each query keeps at most 512 completed causal pools (2048 tokens) plus the 0–3 token tail |
| Extras | MTP layer 45 (54 tensors under `layers.45.*`, unused); vision tower (24 layers, width 1024, patch 14, spatial merge 2, temporal patch 2) |

Selection is dense through position 2050: the first selective row is 2051, when the 513th pool completes. The
checkpoint stores the HC mixing matrices, router `[288,4096]` and the small KDA projections (FA/GA `[128,4096]`, FB/GB
`[8192,128]`, beta `[64,4096]`) in BF16 and the HC scale/base, decay parameters and correction bias in FP32;
`moe_router_dtype=float32` names compute precision, not storage.

## Packs and loading

A served pack carries MCG EXL3 routed experts (K2.25 or K2.5, search window 12) and a trunk stored as affine 6-bit group-128
(Sushi-2.3bpw, the served recipe), affine 8-bit group-128 (Sushi-2.4bpw), or the source FP8 E4M3FN block-128 trunk
kept raw (Sushi-2.45bpw). Small BF16/FP32 tensors keep their source precision. The consumer contract is
[pack-format](pack-format.md); how packs are made lives in the private converter repo.

- `model.loadWeightsForConfig` reads only indexed text tensors, plus the tower when vision is on. It uploads one shard
  at a time (a 566-shard pack once exhausted the 256-descriptor limit) and preserves every stored dtype.
- Affine 6/8-bit is inferred from the packed row width and the declared input width, never from a directory name.
- Raw FP8 projections run MiMo's `fp8_block` kernels: direct FP32-accumulating GEMV up to 16 rows, a temporary BF16
  expansion beyond it (billed for the widest pending layer: 576 MiB at two pending layers).
- The BF16 source checkpoint is the KLD teacher, run with SSD-streamed experts
  ([engine-expert-streaming](engine-expert-streaming.md)).
- `--ssd-budget-gb`/`--expert-cache-gb` stream any pack's EXL3 experts, or the BF16 source's, through the same engine
  for `serve`, `run` and `kld compare`: text only, DFlash2 off, output identical to the resident load.
- The FP8 release (E4M3FN block-128 trunk and experts) streams its experts as stored and runs its trunk on `fp8_block`
  like Sushi-2.45bpw; it is never a teacher. Hermetic proof only: the checkpoint is no longer on the box.

## Serving loop

- Prefill runs 2048-token chunks with two layers in flight; chunks ending at or before token 2051 use dense
  expanded-K/V attention, later chunks the absorbed sparse path. Decode submits every four layers and evaluates logits
  and every cache array once per token.
- Vision: padded CLIP preprocessing, temporal placement of video frames, visual embeddings spliced into the HC input.
  The tower weights join the load bill and the encoder scratch is checked before each image/video encode.
- Sampling, stop and output budgets run in the shared generator; a request that cannot speculate decodes serially.

<a id="dflash2"></a>
## DFlash2

The assistant is a separate 5-layer draft model, not the checkpoint's MTP layer: hidden 4096, block 8, mask token
154856, noncausal block attention, sliding window 2048, two-tap dynamic convolutions, selector rank 256 with top-16
lattice edges. It has no embedding or head and uses the target's. Its input is the mean of the four HC streams after
target layers 5, 14, 24, 33 and 42, before the final norm.

- **Discovery**: `--drafter <dir>` wins, `--no-drafter` disables; otherwise a valid `dflash2/` inside the pack, then
  legacy `drafter/`. One resolved path feeds both the bill and the loader.
- **First-load cache**: when only the shipped BF16 `GLM-5.3-Flash-DFlash2/` exists, `serve` and `run` quantize its
  matrices once to A4 group-64 with MLX's affine quantizer into `dflash2/` (selector codebooks, selector hidden
  projection and non-matrix tensors stay BF16), under a per-pack lock, staged and synced before publication, and
  invalidated by source/config identity. 2.18 GiB → 0.721 GiB. No space or no write permission
  falls back to the BF16 assistant and reruns preflight with its full size. The cache is local only: the assistant's
  CC BY-NC-ND 4.0 license is unchanged and the cache is no redistribution artifact.
- **Stored formats**: BF16, or one uniform affine format per assistant: A4 g64, A6 g128 or A8 g128 (anything else is
  `UnsupportedGlmDraftStorage`). A4 g64 is the first-load default (774.5 MB vs A6's 1013.1 MB; drafts ~11% faster, decode within drift, target
  output exact); an A6 cache from the earlier policy is regenerated.
- **Tree**: two draft nodes plus the root, up to four children per node; the verifier runs all rows layerwise.
  KDA replays only the accepted path from a prework tape; IndexPool builds branch-local pools from the committed prefix
  plus each node's ancestry (pooling flattened tree rows would pool siblings together); MLA reads the committed prefix
  plus the ancestry tail. Commit publishes target state and assistant context together; a commit that fails after
  taking over the request's MLA buffers leaves the request failed.
- **Decisions**: greedy follows the target argmax; sampled requests draw only the visited target path with the
  request's sampling parameters, advancing the RNG exactly as serial decoding does (budgets and EOS included). Both
  verify through the same batched rows, whose logits equal per-row serial projections bit for bit.
  Constrained, forced-tool-call, penalized, logprobs or explicitly budgeted-thinking requests decode serially.
- **Bills**: assistant weights at load; per request the sliding window ×4, captures per prefill row, three recurrent
  checkpoints, the 256 MiB verification-scratch cap and 64 MiB. The MLA reservation is input + max_tokens + 3 rows:
  a request without max_tokens reserves its whole context window (946K rows: 11.3 GB BF16, 6.3 GB kv8).
- **No yield gate**: `[spec-stats] gate_min` is the generic DFlash bar (1.80 here) and is never evaluated on the
  native path. N2 beats a serial step above ~1.1 accepted drafts per round at 1K–30K; measured requests ran 1.31–1.90.

Wider trees lost: N3 with every four-row kernel optimized measured 40.44 vs N2 40.22 tok/s at 8192 IDs (`e1597cc2`,
inside 1.46% drift) because verification per round grew 20.6%.

- **N2 saturates on copies, breaks even on prose** (no runtime yield gate): a ~2K-token verbatim copy and a rename
  edit accepted 2.00 of 2 drafts every round (41.4/40.8 vs 25.2 tok/s serial), low-effort prose 0.88 (26.1 vs 25.6);
  greedy bytes equal serial (`144f63db`, BF16 latent, Sushi-2.3bpw + A4 g64, `taskpolicy -a`, busy box, 2026-10-04).
- **No lookup drafting**: PLD never runs on GLM (`specInitWiring`'s module branch, cleared again in `Generator.init`),
  so `--no-drafter` decodes plain serial; the request log's `pld=enabled` then `drafter takes priority` is cosmetic
  and the 0.010 n-gram gate is inert (DFlash2 is exempt from it).

<a id="concurrency"></a>
## Concurrency

- Each slot owns its target state ([server-lifecycle](server-lifecycle.md#scheduler-and-batching)), so concurrent
  requests interleave one forward or round at a time and a prefill yields to the others' decode ticks; aggregate
  throughput stays one stream's until rows batch. A streamed GLM load (the teacher) still queues.
- **Rows are the currency**: a row costs ~10–12 ms over a ~20–27 ms fixed forward; draft rows pay only while their ms
  per accepted token (~19 ordinary, ~37 prose) beats the batch's own (fixed/B + row: ~26 at two requests, ~19 at four).
  Interleaving alone adds no throughput; batched plain rows do, and a B×3 verify crosses the four-row tile cliff.
- A multi-request forward can reuse the verifier's row-exact projections, router, HC and EXL3 FFN (1–16 rows); only
  the KDA recurrence and MLA attention hold per-request state.

## Memory

- Load bill: text weights, the enabled tower, the selected assistant and warmup. Request bill: BF16 latent 11,264
  plus pooled-index 704 bytes per token (11,968); under kv8 the latent is 5,984 (11 × 512 codes + 8 BF16 scale/bias
  pairs, 6,688 per token). Then capacity growth (256-row rounding, at the stored row width), the raw key/gate ring
  and FP32 KDA state (147,619,840 bytes), plus native kernel transients at two pending layers: A6 expansion 512 MiB,
  head-batched MLA copies 768 MiB, packed attention with its second tile and B32 512 MiB, index scores 8 MiB per
  pending layer, B1/B3 decode attention 32 MiB, KDA cluster 1.25 MiB per pending layer; kv8 adds its dense-prefill
  dequantization (≤ 2051 rows) and one chunk's quantizer output, 6.1 MiB.
- `max_safe_context` = (ceiling − active − transients) × 0.8 × 0.8 / per-token bill. The kv8 default drops the bill
  44%: Sushi-2.5bpw + vision + A4 assistant boots at 104.32 GB active with `max_safe_context` 1,048,576 (the position
  cap; about 1.36M by the bill), against 758,793 at `--kv-quant 16` (976a0dbb, auto context, margin 4 GiB).
- Measured: Sushi-2.3bpw plus the A6 assistant settles at 94.55 GB active (88.06 GiB) under a 115.45 GB limit on the
  128 GB box; a 16K prefill peaks at 96.43 GB without an assistant. Sushi-2.45bpw (raw FP8) is 101.75 GB resident,
  102.51 GB peak while scoring KLD. Sushi-2.5bpw with the A6 assistant and vision is 104.56 GB active, leaving
  `max_safe_context` 746,036 tokens (A4 assistant: 104.32 GB, 758,793; BF16 cache with no assistant, no vision and
  `--wired-margin-gib 2`: 955,781) under `iogpu.wired_limit_mb=120000` (margin 4 GiB): 1M context fits only at kv8.
- An explicit `--ctx-size` is not checked against that bill at load (GLM is outside the load-time serving bill), so
  `n_ctx` can advertise more than a request may use; request admission refuses past the affordable context.

## Recorded performance

llmprobe 0.6.13 `--bench-only --runs 1`, reasoning default, one request per cell, server timers. Runtime `4fcb541e`
(2026-10-04): Sushi-2.3bpw + A6 g128 assistant, N2/children 4, prefill chunk 2048, BF16 MLA, FP32 KDA, greedy,
prefix reuse off; decode = (outputs − 1) / decode time, 192 outputs (ordinary 16K stopped at 177 on EOS). The cells
ran through the loopback bench bridge that `sushi serve` has since replaced; no `sushi serve` ladder is recorded yet.

| Context | Ordinary prefill | Ordinary decode | Predictable prefill | Predictable decode |
|---|---:|---:|---:|---:|
| 2K | 869.79 | 46.18 | 918.33 | 50.61 |
| 4K | 844.65 | 39.39 | 844.02 | 49.33 |
| 8K | 769.25 | 46.05 | 779.09 | 49.64 |
| 16K | 730.25 | 42.47 | 721.95 | 48.08 |
| 32K | 660.77 | 42.99 | 663.04 | 47.56 |

Input IDs: ordinary 2072/4095/8261/16314/32783, predictable 2036/4059/8225/16278/32747. Without an assistant, serial
decode measured 31.4 tok/s at 512 context (2.3bpw, `ba106e5e`). The 1500 tok/s prefill / 60 tok/s decode goals are
open; verification dominates a speculative round (about 60 of 70 ms at 16K–32K).

## Quality

A pack is scored with `sushi kld compare --model <pack> --fixture <teacher>` against the native BF16 teacher
(`sushi kld capture --prompts standard4`, streamed BF16 experts, TF32 off). Four prompts × 512 (2026-10-04):
Sushi-2.3bpw 0.0930, Sushi-2.4bpw 0.0915, Sushi-2.45bpw 0.0913 mean KLD (code ~0.045, prose ~0.139); Sushi-2.5bpw
(K2.5 experts, A6 trunk) 0.0721. Not yet the 16x512
release reading; table and settings in [quality-kld](quality-kld.md#glm-53-flash-native-bf16-teacher-4x512-2026-10-04).

## Lessons

- mHC expansion contracts the residual streams first, then adds the separately rounded FP32 branch product; the
  reverse order changes BF16 results.
- SiLU rounds its sigmoid to BF16 before the multiply; an FP32 HC mix through generic matmul may pick TF32, so the
  mix uses an explicit FP32 dot product.
- Storage dtype is read from the headers, never from config compute precision (router and HC matrices are BF16).
- A contiguous batch-one slice still aliases its prompt buffer: the KDA convolution tail takes a materialized copy.
- Async scheduling may hold two cache generations at once; it is not a bill reduction.
- Absorbed and expanded MLA round at different BF16 boundaries; chunk width changes GEMM row shapes, and the chunk
  schedule moves IndexPool scoring between scalar and NAX. Prefill chunking is a numerics decision.
- The first generated token comes from prefill (64 outputs = 63 decode forwards); speculative harnesses that count
  committed input tokens are not comparable with server delivery rates.
- Count tokens through the tokenizer: the "2K" predictable prompt is 2036–2037 IDs, so exact-2048 guards never
  engage on it; 2048-token content repeated 4× is 8156 tokens, not 8192.
- Every fused path needs an engagement counter: a router fusion keyed on FP32 storage made zero calls behind a
  correct fallback.
- MLX masked SDPA does not protect against NaN/Inf in masked K/V rows (0·NaN poisoned every earlier query): zero
  invalid rows before load.
- Retaining a full KDA state per tree node would cost ~2.1 GiB for 16 nodes; replay the accepted path instead.
- A synchronizing per-stage profile changes overlap and allocation; it ranks stages, never times them.
- The pinned Metal backend cannot load safetensors on the GPU stream: load on the CPU stream, compute on the GPU.
