# GLM-5.3-Flash: native engine foundation

The `glm5_next` source adapter preserves official BF16 expert tensors for streaming tests. The native
text forward now runs the completed EXL3 checkpoint through an opt-in diagnostic harness. It is not
registered in `model.served_model_types`: production serving, full reference parity and quality gates
remain open. See [diagnostic usage](glm5-diagnostic.md) and [attention/cache details](engine-glm5-attention.md).

## Checkpoint and implementation status (2026-10-02)

The `GLM-5.3-Flash-Sushi-2.25bpw-A8g128-W12` checkpoint is complete: 129 routed projection banks,
37,152 expert projections, 449 affine trunk matrices and 1,169 retained tensors. Its indexed tensor payload is
98,336,815,992 bytes (91.5833 GiB), including the optional MTP and vision weights. This is stored tensor size,
not a measured runtime memory requirement. Large trunk matrices use affine8 group128; preserved small
BF16/FP32 tensors keep their source precision. Creation details belong to the private Sashimi repository.
The completed checkpoint has not passed native full-model generation or KLD.

`glm5_model.zig` provides stored-affine linear operations, clamped dense MLP, precise FP32 mHC mixing
and KDA projection/convolution/decay/output assembly. KDA and mHC now have independent oMLX fixtures,
including irregular chunks and nonzero initial recurrent state. Preparation owns static convolution and
decay constants; it has explicit cleanup. The checkpoint's mHC matrices remain BF16, while scale/base
and mixing arithmetic are FP32.

`glm5_forward.zig` composes all layers, resident affine embeddings/head, sigmoid top-k routing, shared
experts and request-local state. It evaluates each layer and its state before releasing temporaries.
A tiny four-layer model exercises KDA, MLA, routed/shared experts, affine embedding/head, reset and
last-row logits. Nonzero stored-affine MLA tests cover both projection orientations. The diagnostic
loader preserves stored dtypes and index ownership, and excludes unused vision and MTP tensors.

The first real-checkpoint diagnostic loaded 2,302 text tensors (95,471,433,976 bytes, 88.9147 GiB),
processed an eight-token prefix and generated four tokens. This proves load/bind/forward execution;
the truncated prompt and output do not establish coherent English, quality or steady throughput.
The current local directory is named `GLM-5.3-Flash-Sushi-2.4bpw`; its stored expert metadata remains
K2.25/W12. No conversion was repeated and no rate is inferred from the directory name.

## Source layout

The text model has 45 layers, hidden width 4096, expert width 2048, 288 routed experts and top-k 8. Layers 0–2
are dense; layers 3–44 contain experts. The optional MTP layer is layer 45 and is outside the trunk store.
Each expert has separate `model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight` tensors:
BF16 `[2048,4096]` gate/up and `[4096,2048]` down. One expert occupies 48 MiB; one complete layer is 13.5 GiB.
The union workspace therefore needs 13.5 GiB before the LRU cache is budgeted.

`expert_quant.QuantStore` exposes these as three present weight components and six absent scale/bias components
under `bf16_individual`. Header-derived spans may cross shards and need not follow expert order. Each shard's
header is cached once; no expert payload is read while opening the store. The adapter rejects missing tensors,
non-BF16 payloads, incorrect shapes, out-of-bounds spans, routed tensors in the dense prefix, and mixed layouts.
FP8 expert scale keys are rejected rather than silently treating the FP8 release as the BF16 teacher.

`expert_stream.Engine` imports the three components, returns transposed BF16 views for `gather_mm`, and keeps
`Prepared.quantized` false. It uses the existing synchronous exact-route path; speculative quantized execution is
not used. Source BF16 bits are preserved throughout. The optional MTP shard, shared experts and dense trunk
are not opened by this store; a future model loader must load the trunk separately and resolve MTP explicitly.

## KDA recurrence

`glm5_next.kda` runs the shared vector-gate Metal recurrence with FP32 state input/output, BF16 or FP32
Q/K/V and matching output precision. Its inputs are prepared queries/keys (L2-normalized, query additionally
scaled by Dk^-0.5), value, per-key-channel decay factors, and beta. Shape and dtype checks reject a scalar
per-head decay or a rounded BF16 state. Projection, convolution, normalization and output-gate prework are now assembled in the partial
`glm5_model.KdaLayer`; integration and full-layer reference validation remain required.

The recurrence is checked against an independent scalar FP64 calculation with nonuniform channel decays,
multiple batches and heads. Serial and multi-token execution are bit-identical for BF16 and FP32 inputs,
including their final FP32 states. This is primitive validation, not full-model parity.

## mHC and clamped packed experts

`glm5_next.hcCollapse` applies four-stream Sinkhorn mixing to precomputed FP32 mix projections; its
mixed activation retains the input precision, and its post/combination coefficients remain FP32.
`hcExpand` accumulates the residual contraction before adding the separately rounded FP32 branch
product, then rounds once to the activation dtype. Adding the branch before the contraction can change
BF16 results; a cancellation-sensitive independent regression covers that arithmetic boundary.

`exl3.moeClamped` preserves GLM's gate upper clamp and symmetric up clamp before SwiGLU, and accepts
packed expert rates from 2 through 4 bpw in eighth-bit increments. Its boundary checks require matching
routed-input/score shapes, H128-aligned bank dimensions, compatible gate/up/down expert counts,
U16 trellises and correctly shaped F16 scale banks before dispatch. Router-produced expert IDs must
remain within the bank's expert range. Scalar host comparisons exercise every supported rate in decode
and prefill. These are primitive checks, not full-model parity or quality measurements.

## Configuration and cache geometry

The text-only parser requires the GLM core geometry and rejects contradictory mHC, routing, pooling,
activation and projection-bias semantics. The dense/sparse MLP table and any redundant KDA/full-attention
layer lists must agree with the supported layer layout. Zero or overflowing kernel dimensions are refused;
the official vision metadata does not enable the unrelated generic vision forward.

Compressed MLA caches two 512-wide latent buffers while attention scales by the original 256-wide query,
so its scale is 1/16. For the official 45-layer geometry, the dense KV bill is 22,528 bytes/token; the generic
KV quantization adjustment follows the actual cache format. The BF16 pooled indexer history is 704
bytes/token, and the raw key-plus-gate ring is 180,224 bytes per slot. The recurrent-state plus convolution
checkpoint is 147,619,840 bytes, including FP32 KDA states. No Qwen FP32 indexer score bank is billed.

## Forward cache ownership

The in-progress KDA forward stores an owned convolution tail after multi-token calls. A contiguous
batch-one slice still aliases the full prompt buffer, so it uses the shared materialized-copy helper.
The copy stays lazy; the enclosing forward must evaluate the cache with its normal layer boundary and
release its operation scope. A pointer-alias regression checks that the evaluated tail retains its values
without retaining the parent allocation. Single-token decode keeps the inexpensive contiguous view.

## Remaining validation and optimization

The ordered milestones remain in the [native execution plan](plan-glm5-native.md). The
[correctness report](glm5-correctness-audit.md) and [efficiency report](glm5-efficiency-audit.md)
record the audited foundation; new complete-forward behavior needs its own review.

- Run the full 512-token prompt and 64-token generation with warmup, explicit timing denominators,
  actual memory and per-layer attribution. Targets are 1,000 tok/s prefill and 60 tok/s decode, not results.
- Compare full MLA/reference layer outputs and logits: absorbed latent attention and expanded dense
  attention have different BF16 rounding boundaries even when their projection algebra agrees.
- Measure KLD against a lossless BF16 teacher and validate 4K/16K contexts before 64K. The identity-prior
  quantized checkpoint has no accepted full-model quality result yet.
- Replace diagnostic per-layer synchronous evaluation only after proving bounded memory and state
  equivalence. Profile KDA, attention, routing and projection dispatch before choosing optimizations.
- Integrate Transformer/server ownership, memory admission and tested capability checks. The diagnostic
  cache currently uses lossless BF16 latents, not generic KV8; generic cache bills do not describe this
  standalone request implementation's exact allocation. Add reset/rollback/prefix lifecycle integration
  before enabling dependent features. Vision, MTP, batching and speculation remain unsupported.

## References and checks

Architecture references: the official `zai-org/GLM-5.3-Flash-BF16` configuration; oMLX `ecebb8e2`'s vendored
`mlx_vlm/models/glm5_next` implementation; TensorFold `bb4b4a3`'s `families/glm5_next/cuda` implementation.
No foreign kernel code is included in this source-adapter change.

The `GLM BF16` unit tests cover layout recognition, shuffled multi-shard reads, dense/MTP exclusion, malformed
source rejection, zero-copy imports, cache hits/evictions, union overflow and selected-expert matrix products.

## BF16 source sanity run

2026-10-02: the complete official 120-shard checkpoint was downloaded and structurally verified against its
index: 38,770 tensors and 642,646,653,816 payload bytes. A separate diagnostic Python runner used oMLX
`6745c39c`'s GLM forward and lossless BF16 reads of the selected routed experts. It did not use Sushi's native
forward, quantization, MTP or a persistent expert cache. The text trunk remained resident (16.775 GiB at load).
The runner used a reference HC fallback for two unavailable optional imports and disabled compiled/fused
decode so the synchronous expert reader could run outside graph tracing. Its selected-expert reader matched
resident BF16 expert results bit-for-bit on synthetic single-token and multi-token cases.

On an M5 Max 128 GB, a single greedy request used exactly 512 input tokens and generated 64 tokens:

| metric | result |
|---|---:|
| Prompt processing, including SSD expert reads | 6.4228 tok/s |
| Decode, mlx-lm generation timer | 0.5578 tok/s |
| Total generation wall time | 194.553 s |
| Peak active Metal memory | 34.099 GB |

Foreground `taskpolicy -a`, an exclusive GPU lock, max fans and a 10-second cooldown were used. The official
chat template requested low reasoning effort. Output began `</think>Rain forms through the water cycle.`
and continued with coherent English about evaporation, condensation and droplets, stopping at the token
limit. This establishes a source-checkpoint sanity result for that diagnostic forward, not full reference
parity, broad model quality or native Sushi throughput. The native integration work above remains required.

## Native load validation

The GLM KDA binder validates preserved convolution and norm shapes before graph construction and
requires FP32 decay parameters and hyper-connection scale/base coefficients. The original HC mixing
matrices are BF16 and remain stored that way; their multiplication accumulates in FP32. A scalar output norm must not
silently broadcast across every channel. Packed expert preflight rejects tensors in the dense prefix or
outside the configured trunk/MTP range, and any present MTP expert component requires a complete bank.
These checks do not enable the architecture gate or establish full-forward parity.

## Assembled KDA and mHC reference fixtures

The small synthetic `glm5_layers` fixture records oMLX `6745c39c` source hashes, seed, dtypes and
precision settings. It contains nonzero initial states, a cold start, full/serial/2–1–2 KDA calls,
and mHC input/output boundaries. The fixture generator is a reference test utility, not a pack converter.
Reference capture disables TF32; native HC mixing uses an explicit FP32 dot product and passes with the
backend's default settings. BF16 HC matrices are converted to FP32 in the dot product without retaining
an expanded matrix.

These checks exposed two arithmetic differences: SiLU must round its sigmoid to BF16 before the
multiply, and generic FP32 matmul could choose TF32 for HC mixing. SiLU now matches the reference
convolution and compiled dense FFN exactly on the activation corpus. The assembled KDA and mHC
comparisons use fixed bounds for reduction differences: KDA outputs/tails `0.001 + 0.01*abs(reference)`,
FP32 state `1e-5 + 0.001*abs(reference)`, HC activations `0.002 + 0.008*abs(reference)`, and HC FP32
coefficients `2e-5`. These tolerances were set before fixing the observed failures.

KDA `prepare` owns a combined convolution weight and `exp(A_log)` once per layer; `deinit` is idempotent.
Prepared and unprepared execution must produce identical output and recurrent state. This is layer
validation, not full-model or quantized-checkpoint quality evidence.
