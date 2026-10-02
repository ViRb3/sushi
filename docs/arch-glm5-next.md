# GLM-5.3-Flash: BF16 streaming foundation

The `glm5_next` source adapter reads the official BF16 checkpoint's individual routed-expert tensors through
Sushi's existing SSD fill pool, per-layer LRU and zero-copy Metal slabs. This is source and cache support only:
`glm5_next` is not in `model.served_model_types`, and cannot yet serve or capture a teacher.

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
per-head decay or a rounded BF16 state. The projection, convolution, normalization and output-gate prework
are still required before this primitive forms a complete layer.

The recurrence is checked against an independent scalar FP64 calculation with nonuniform channel decays,
multiple batches and heads. Serial and multi-token execution are bit-identical for BF16 and FP32 inputs,
including their final FP32 states. This is primitive validation, not full-model parity.

## Forward implementation still required

- Parse the nested GLM configuration and load the resident BF16 trunk, excluding routed experts before any
  tensor materialization. Keep router, hyper-connection, decay and recurrent-state arithmetic at their required
  precision. No affine quantization belongs in the teacher path.
- Wire KDA's per-key-channel recurrence to its decay prework and L2-normalized queries/keys. Qwen's scalar-per-head GDN gate is not
  equivalent. GLM also uses a sigmoid output gate and separate depthwise q/k/v convolutions.
- Implement mHC's Sinkhorn residual mixing, sparse MLA attention and the IndexPool selector. The attention
  layout has 34 linear layers and 11 sparse-attention layers, no Qwen n-gram table, and no MLA RoPE channels.
- Use sigmoid router scores with correction bias for selection, unbiased normalized weights scaled by 2.5,
  one shared expert, and GLM's clamped SwiGLU (gate upper bound 10, up bounds -10 to 10).
- Prove short and long-context reference parity, streamed/resident parity on a tiny model, then full-model
  teacher capture before opening the served-architecture gate. Vision and MTP require their own integration.

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
