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

## Forward implementation still required

- Parse the nested GLM configuration and load the resident BF16 trunk, excluding routed experts before any
  tensor materialization. Keep router, hyper-connection, decay and recurrent-state arithmetic at their required
  precision. No affine quantization belongs in the teacher path.
- Implement KDA's per-key-channel decay and L2-normalized queries/keys. Qwen's scalar-per-head GDN gate is not
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
