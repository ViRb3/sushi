# GLM external-runtime efficiency comparison

The strongest remaining serial-decode opportunity is to fuse KDA work around the recurrence, followed by
HC preprocessing. Long-context work should separately target index scoring and sparse latent attention.
The other runtimes demonstrate useful implementations, but their weight formats, rounding and dispatch
guards prevent treating them as drop-in faster versions of this engine.

## Scope and provenance

Read-only source study on 2026-10-02; storage/loading clarification on 2026-10-03. No external engine was executed, no model was loaded, and no GPU
benchmark was run for this report. Reachability below means that the caller and eligibility checks were
inspected; it is not an assertion that a particular external deployment logged kernel engagement.
The actual target's tensor headers were inspected to resolve format-dependent guards.

| Repository | Revision inspected | Relevant implementation |
|---|---|---|
| Sushi | `ddcf219f70f8786f7046cd4c142f6fb84e3637e0` | Native diagnostic GLM, affine8/group128 large trunk matrices, retained BF16/FP32 small tensors, EXL3 K2.25/W12 routed experts |
| oMLX | `6745c39cb66ba5ec130a761149fc1b8b832fcb07` | MLX/Metal GLM compatibility model, fused decode and prefill kernels, M5 NAX paths |
| ds4 | `110afdd8886586f18fc9b28bc5533152dd10e728` | C graph plus Metal GLM-5.3 KDA/DSA implementation; CUDA/ROCm alternatives have different dispatches |
| TensorFold | `bb4b4a35863af562fc4ccb2586300d8f94b5d6de` | GLM CUDA/Triton implementation with static buffers and optional CUDA graph replay |

The coordinator's approximately 748 token/s prefill and 24.6 token/s serial decode at 512/64 is the
starting point for prioritization, not a cross-engine comparison. No external throughput numbers are
used as evidence of a Sushi speedup. The directory's “2.4bpw” name does not change the stored expert
metadata: this target uses packed width 36, K2.25, window 12.

## What actually applies to this checkpoint

**KDA format mix matters.** The target's Q/K/V/O are affine8/group128. Its f_a, f_b, g_a, g_b and beta
projection weights remain BF16. In oMLX's `Glm5NextLinearAttention`, `_fused_in_proj` refuses to concatenate
mixed dense/quantized modules, and `_build_decode_groups` requires every input projection to be an affine
`QuantizedLinear`. Thus its complete `_decode_step` does **not** engage unchanged on this exact mix.
The lower-level `decode_kernels.kda_decode_step` can accept precomputed `a_pre` and `gate_pre`; adapting
that primitive to separate input buffers avoids both requantization and large resident copies.

**Stored dtype and loaded dtype differ.** The target stores routed `mlp.gate.weight` as BF16
`[288,4096]` and `e_score_correction_bias` as FP32 `[288]`; `moe_router_dtype=float32` describes compute,
not checkpoint storage. oMLX's `moe_router_logits` requires FP32 weight and bias, but its normal loading
path can satisfy that guard: container `Model.sanitize` calls `LanguageModel.sanitize`, which explicitly
upcasts router weights and correction biases to FP32. `glm5_next_cast_predicate` protects them from later
downcasting. Therefore the raw BF16 checkpoint does not by itself establish an oMLX router fallback.
Sushi preserves storage and must read BF16 weights with FP32 products/accumulation to reuse this fusion
without introducing a persistent widened weight array. A real diagnostic engagement counter caught the
initial FP32-storage-only adaptation doing zero fused router calls; successful numerical fallback alone
was not evidence of engagement.

**HC needs the same storage distinction.** `hc_mix` and `hc_defer_supported` require FP32 HC projection
weights. The target stores BF16 `hc_attn_fn`/`hc_ffn_fn`, with FP32 scales/bases, but oMLX's same sanitizer
upcasts both remapped HC namespaces before execution. Its deferred path also requires the NAX
relaxed-FP32-matmul policy. Direct reuse under Sushi's storage-preserving loader still needs BF16 reads
with explicit FP32 widening and parity with the current HC arithmetic; the guard is not evidence that
normal oMLX loading leaves this path disabled.

**Expert kernels are format-specific.** oMLX's two-dispatch MoE path accepts affine `SwitchGLU` weights,
including a compatible shared expert. It cannot decode this EXL3 bitstream. ds4's documented GLM Q2 pack
comes from the official FP8 source and uses IQ2_XXS gate/up plus Q2_K down experts. TensorFold's GLM loader
accepts affine4/group64, or EXL3 with BF16 non-expert weights; `exl3_mm.words` explicitly requires packed
width 64, K4. None is the same numerical or bandwidth workload as Sushi's target.

**Hardware differs.** TensorFold's inspected GLM implementation is CUDA, and its application wrapper
constructs a two-rank NCCL engine. CUDA graphs, Triton tensor-core operations and CUDA stream behavior
are architectural references, not portable Metal kernels or fair single-M5 comparisons.

## Ranked experiments

### 1. Fuse KDA prework and output gating while retaining the recurrence

Current `glm5_model.KdaLayer.apply` already reuses prepared convolution weights and `exp(A_log)`, and
uses the copy-free one-token QKV dispatch. It still builds separate convolution, SiLU, casts, Q/K square
reductions, reciprocal square roots, scales, decay/beta operations, and output RMS/gate operations around
`glm5_next.kda`. This repeats in 34 layers.

Evidence from the other engines:

- oMLX `Glm5NextLinearAttention.__call__` first tries `_decode_step`; its low-level `kda_decode_step`
  fuses convolution through gated RMS output for B=1 and one through eight tokens. Its wrapper requires
  a cache, no mask, kernel width four, safe lower bound, and the projection eligibility described above.
- oMLX `glm53_kda_prework.glm53_kda_prefill_eligible` enables its separate prework/epilogue route by
  default for BF16, B=1, at least 64 rows, ordinary non-speculating two-array caches, no padding/history,
  and 128-wide heads. It supports separately computed projections when a weight concat is unavailable.
- ds4 `glm53_graph_kda_attention` calls `ds4_gpu_glm53_kda_decode` after its projections. The Metal
  dispatch runs `kernel_glm53_kda_decode`, fusing convolution, gates, recurrence and output normalization.
  Its prefill path instead launches prepare, recurrence and output kernels in one command encoder.
- TensorFold `forward.kda_block` calls `kda.chain`, a fused CUDA body using preallocated output/state
  buffers. The actual call is in the forward path, not an unused helper.

Start with the output RMS/sigmoid-gate epilogue, then convolution/SiLU/Q/K normalization. Keep existing
projection outputs and FP32 recurrence order. This reduces dependent dispatches and intermediate writes
without changing stored weights. Only after these pass should the larger decode body absorb recurrence.

Parity gate: nonzero BF16 inputs at real 64-head/128-channel geometry; empty and nonempty states;
negative/large gate inputs; serial versus irregular chunks; output, convolution tail and FP32 state.
Match every existing materialization/rounding boundary and require byte equality for a bit-preserving
claim. Retain the current path as the oracle and compare full-model output IDs and peak memory afterward.
The external kernels' own “exact” labels are not a substitute for this comparison.

Do not begin by copying oMLX's per-core prefill recurrence. Its source explicitly says its dot reduction
order differs from the stock gated-delta kernel; its blocked and per-core variants match each other.
This is a separate numerical experiment requiring layer/state analysis and the quality gate. ds4's
FP32 activation/conv buffers and FMA-heavy Metal body likewise differ from our BF16 boundaries.

### 2. Fuse HC preprocessing before considering deferred cross-layer expansion

Current `Hc.collapse` performs FP32 conversion/flattening, RMS normalization, `hcMixExact`, and Sinkhorn
collapse; the model then applies a separate branch RMSNorm. This occurs twice per layer. HC expansion
is also materialized between each attention/feed-forward branch.

oMLX `_decode_hc_pre` replaces the preprocessing chain with `hc_mix` plus `exact_hc_norm` when its guards
hold. For one-token input, `_decode_deferred` can fold the previous expansion into `hc_pre_fused` and
produce the comb product separately with `hc_post_mm`. TensorFold `glue.hc_pre` uses partial/finish
kernels that include branch normalization and reusable activation group sums. In contrast, ds4's actual
`glm53_graph_hc_pre` still calls plain RMS, mix matmul, HC split/collapse and branch RMS separately;
its other DeepSeek HC fusion symbols should not be cited as an engaged GLM optimization.

First experiment: fuse collapse plus branch RMS, or RMS plus the mix projection, with the current BF16
HC weight reads and exact FP32 accumulation. Avoid a new persistent FP32 weight copy. Defer cross-layer
HC objects until a local fusion has a measured benefit; deferral changes ownership, trace points and
failure cleanup across layers.

Parity gate: all four residual streams, post weights, Sinkhorn comb matrices, normalized branch input
and final expansion; multiple nonuniform matrices and near-zero RMS inputs. Preserve the current
reference's sum order and intermediate rounding, especially the residual product versus post-branch add.

### 3. Optimize index scoring and sparse latent attention after the selection boundary

Sushi's index scorer uses scalar per-query/per-pool work over index heads. Its latent attention is bounded
FP32 online softmax over selected rows, with split partials. These are useful correctness baselines but
do not exploit the tensor units for the score/attention matrix products.

oMLX has two distinct mechanisms:

- `indexer_nax.indexer_scores_nax` produces a causally masked, head-summed score plane directly, with
  BF16 queries/keys/weights and 128-wide index heads. Its call site chunks queries under a score cap.
  Short calls can instead use `decode_kernels.dsa_decode_scores` and fixed top-512 selection.
- `sparse_mla_nax.sparse_mla_attention_nax` accepts 512-wide NoPE latents and selected token IDs and
  groups heads on NAX. The inspected model calls it for sparse prefill rows greater than eight, not
  indiscriminately for serial decode. One-token selection instead gathers selected latent rows before
  its short attention path. Do not claim the prefill NAX kernel is the active serial path.

ds4's indexed decode has an absorbed-query, latent-cache, split/group-eight attention path that also
receives value-projection weights. This demonstrates another possible fusion boundary, but its supported
GGUF types, FP16/FP32 cache choices and arithmetic differ. TensorFold uses tensor-core 512-key attention
chunks and a deterministic merge; its inspected GLM state stores expanded per-head K/V, a materially
larger memory choice than Sushi's one latent cache.

The first selective query is position 2051: through position 2050 there are at most 512 completed pools
plus the incomplete tail, so selection equals the full causal prefix. A 2048-token prompt followed by
64 generated tokens exercises sparse selection in roughly its last 60 decode forwards; a 512/64 run does
not. Use that distinction when reading performance results.

Focused experiments: compare current versus tiled index-score kernels first, with selected token IDs
as a hard gate; then compare short gathered-latent or grouped-head attention. Preserve bounded scratch,
scale 1/16, causal completed-pool eligibility and the zero-to-three-token tail. Test 2047–2055 boundaries,
nonzero offsets, ties/negative index weights, irregular chunks, and 4K/16K/64K history. A score rounding
change can change discrete selection even when its numeric error is small.

ds4's `glm_graph_dense_compact_attention_limit` uses its normal GLM 4K work window in the compact path.
That is not Sushi's 2051 exact dense-prefix condition. Extending dense attention to 4K must not be adopted
as a bit-preserving optimization. Likewise, expanded per-head caches should not replace the latent cache
just to reproduce a CUDA path: at equal precision the K/V storage ratio is
`heads * (key_width + value_width) / latent_width`, which is 64 for the official 64-head geometry.

### 4. Fuse routing arithmetic and the shared-expert finish without changing EXL3 math

The reviewed `glm5_forward.route` has separate FP32 logits, sigmoid, correction bias, partition, gather,
normalization and scale operations; the router weight itself is stored BF16. oMLX has a fused
logits/sigmoid/bias primitive whose FP32-weight guard is reached after its sanitizer widens that tensor,
then optional selection inside its affine gate/up kernel. The latter is not applicable to EXL3 without a
new reader. Its selected route order must also not be assumed identical to Sushi's `argpartition` order.

Fuse only logits/sigmoid/bias initially, reading retained BF16 weights and widening products to FP32,
while retaining current selection and normalization. Record actual fused-call counts: a zero count can
hide a storage-dtype rejection behind a correct fallback. Validate
selected IDs, their order and normalized scores, including ties. Different slot order changes the final
floating-point expert reduction even when the expert set is identical.

For experts, Sushi already has natural-order paired preparation, retained F16 intermediate planes,
GPU prefill routing metadata and its packed K2–K4 readers. TensorFold groups rows by expert and fixes K
splits independent of row count; its EXL3 path uses FP32 split planes and K4-only storage. oMLX's affine
MoE combines routed/shared branches in fewer dispatches but has no matching trellis arithmetic.
Do not replace the current clamped path with a normal fused EXL3 path that changes F16 stores or split
accumulation. A narrower candidate is adapting Sushi's existing `downFinishReduceWithShared` to the
clamped wrapper, preserving routed-output rounding before the shared add. Test every supported rate,
clamp saturation, distinct scales, slot orders and shared/routed output bytes.

### 5. Reuse scheduling lessons without treating batching or graph replay as serial speedups

The main scheduling gaps are already addressed in Sushi: async submissions every four decode layers,
a final logits/cache evaluation, optional two-layer bounded prefill scheduling, and scoped layer lifetimes.
oMLX's model uses an eight-layer one-token async ladder and a `LayerPipeline` for wide prefill, with at
most two layers in flight. Its optional allocator clearing and lazy-last-layer policy have different
memory/performance tradeoffs; copying its stride or clearing policy is not automatically an improvement.

One small remaining experiment is a cache-only intermediate-prefill call: oMLX's lazy-last policy can
avoid the last layer's unneeded FFN/output work when only cache updates are consumed. Sushi's diagnostic
currently computes a last-token head result for each prompt chunk. Prove unchanged final logits and all
cache states before omitting that intermediate result; this is a secondary experiment, not the main
explanation for the serial gap.

ds4 reuses an active Metal command buffer/encoder; individual kernel wrappers do not each wait when
inside that batch. Its graph has explicit decode flush intervals and extra boundaries for streaming or
profiling. TensorFold replays captured CUDA graphs only for captured row counts while the context is
inside its dense limit; long-context forwards fall back to eager `compute`. Its serial loop still
synchronizes for sampling/commit. Static buffers and graph replay are useful design references, but
porting an entire execution backend is less contained than the fusions above.

Native multi-session batching in ds4 is guarded: multiple compatible resident GLM sessions, no SSD or
placement mode, and positions below its dense work limit. oMLX's specialized single-sequence fusions
mostly decline padded/batched caches; ordinary model paths handle those cases. TensorFold's inspected
application is two-rank tensor parallel rather than a direct equivalent of continuous request batching.
These capabilities improve other workloads and should remain outside serial-first optimization.

## Reuse and validation rules

| Source | Reuse/licensing concern | Appropriate use here |
|---|---|---|
| oMLX compatibility/decode code | Apache-2.0; some qmv helpers originate in MIT-licensed MLX; preserve both attributions and the exact source revision | Closest Metal reference; reuse small kernels only after adapting dtype/shape guards and matching current arithmetic |
| ds4 | Root MIT license names ds4.c and ggml authors; KDA Metal source says it was adapted from the kimi-k3 branch | Preserve provenance and inspect any copied file's inherited notices; use its fused-buffer organization, not its quantization/cache semantics |
| TensorFold | MIT; `THIRD_PARTY_NOTICES.md` distinguishes original CUDA kernels from referenced model math and third-party dependencies | Reimplement ideas for Metal; CUDA/Triton/NCCL code is not a directly usable kernel on this box |

For each experiment, record actual engagement, same stored checkpoint and prompt IDs, warmup, cache mode,
chunk size, schedule and peak active memory. Keep the previous recorded baseline; do not rerun it solely
to make a new table. Compare output/state bytes before interpreting throughput. Changes to reduction or
rounding require independent layer/logit analysis and KLD, not a wider tolerance chosen after failure.
Ablation stage timings identify candidates but do not establish the speed of the final uninstrumented run.

## Source locator

All paths below are repository-relative at the revisions above.

| Repository | Symbols and files inspected |
|---|---|
| Sushi | `KdaLayer.apply`, `Hc.collapse` in `src/glm5_model.zig`; `Mla.applyMode`, `route`, `Model.forwardLast` in `src/glm5_forward.zig`; `indexScores`, `attentionChunk`, `attend` in `src/glm5_attention.zig`; `qkv` in `src/glm5_decode.zig`; clamped decode/prefill in `src/exl3/expert_exl3_kernels.zig` |
| oMLX model | `Glm5NextLinearAttention._fused_in_proj`, `_build_decode_groups`, `_decode_step`, `__call__`; `Glm5NextSparseAttention._forward`; `Glm5NextIndexer.__call__`; `Glm5NextMoE._decode_select`, `_decode_experts`; `Glm5NextDecoderLayer._decode_deferred`; `Glm5NextModel.__call__`, `LanguageModel.sanitize`, `glm5_next_cast_predicate`, all in `omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py` |
| oMLX kernels | `kda_decode_step`, `hc_mix`, `hc_defer_supported`, `hc_pre_fused`, `hc_post_mm`, `dsa_decode_scores`, `multi_qmv` in `omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py`; `glm53_kda_prefill_eligible`/`glm53_kda_prefill` in `omlx/patches/glm53_kda_prework.py`; `_percore`/`kda_recurrence` in `omlx/patches/glm53_kda_recurrence.py`; `indexer_scores_nax`/`sparse_mla_attention_nax` in `omlx/patches/glm_moe_dsa/{indexer_nax,sparse_mla_nax}.py`; `omlx/utils/layer_pipeline.py` |
| ds4 | `glm53_graph_kda_attention[_rows]`, `glm53_graph_hc_pre`, `glm_graph_forward_token`, `glm_graph_dense_compact_attention_limit`, `glm53_graph_native_session_batch_supported` in `ds4.c`; `ds4_gpu_glm53_kda_decode`, `ds4_gpu_glm53_kda_prefill`, command-buffer helpers in `ds4_metal.m`; `metal/glm53_kda.metal`; GLM pack/mode section of `README.md` |
| TensorFold | `kda_block`, `dsa_block`, `layer_forward`, `Buffers`, `State` in `src/tensorfold/families/glm5_next/cuda/forward.py`; `chain` in `kda.py`; `Graphs` in `graphs.py`; `Engine.forward`, `serial_decode` in `decode.py`; application construction in `engine.py`; `qmm.py`, `exl3_mm.py`, `weights.py`, `attention.py`, `sparse.py`, `glue.py` in the same directory |
