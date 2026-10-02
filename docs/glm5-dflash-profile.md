# DFlash verifier component profiling

Set `SUSHI_GLM_DFLASH_PROFILE=1` only on the native full-checkpoint DFlash diagnostic
to collect verifier component timings. Profiling is off by default. It is bound
after warmup and captured prefill, so the collection covers measured tree verification,
including any terminal short tree, rather than draft preparation or serial reference.
The diagnostic continues to check output tokens and complete committed target state.

This is a **synchronization-perturbed diagnostic**. Each marker evaluates required
outputs and waits, recording host graph construction plus evaluation time. It changes
queue overlap, allocation lifetime, cache state and synchronization cost. Profile-on
tok/s is not a normal throughput result; its JSON sets `synchronization_perturbed=true`,
`throughput_comparable=false`, and suppresses `speedup_vs_matched_serial`. Measure
ordinary throughput in a separate profile-off run. Do not subtract a universal
evaluation overhead or infer an isolated kernel's time from a grouped stage.

`component_profile.layers` contains named samples per target layer. Each sample has
nanoseconds, call count and total verification rows. `component_profile.global`
contains embedding and final head work. Zero-call fields were not used by that layer.

| Field | Included work and evaluated outputs |
|---|---|
| `kda_qkv` | Q/K/V projections and joined raw QKV |
| `kda_lowrank_beta` | Forget-gate lowrank pair and raw beta projection |
| `kda_prework` | Ancestor convolution, Q/K normalization, vector decay and beta; settles saved raw history and five replay inputs |
| `kda_recurrence` | Parent-indexed FP32 recurrence and BF16 outputs |
| `kda_gate_post` | Output-gate lowrank pair, normalization and gating |
| `kda_out` | Attention output projection |
| `mla_common` | Absorbed Q, latent projection, index keys and compression gates needed by the saved tape |
| `mla_branches` | Branch construction/cache updates, attention and value unembedding; sparse index-query/weight dependencies when used |
| `mla_out` | Concatenated branch output projection |
| `hc_attn`, `hc_ffn` | HC collapse outputs needed by normalization and later expansion |
| `attn_norm`, `ffn_norm` | RMS normalization |
| `expand_attn`, `expand_ffn` | HC residual expansion |
| `ffn_overall` | Entire FFN, including any child router/routed/shared/combine markers |
| `layer_settle` | Existing layer evaluation boundary, including saved tape and capture outputs |
| `embedding`, `head` | Embedding/broadcast; final HC mean, normalization, vocabulary head and greedy decision respectively |

Dense-prefix MLA does not consume index-query or index-weight projections. The
common marker deliberately does not evaluate those arrays. Sparse branch attention
evaluates its normal dependencies, so this work is charged to `mla_branches`.
The profiler never evaluates every temporary in an Ops scope. Existing layer/head
evaluation boundaries record time without adding a second evaluation there.

FFN child fields are `ffn_router`, `ffn_routed`, `ffn_shared` and `ffn_combine`.
Their enclosing `ffn_overall` sample is inclusive: do not add the children to that
sample when computing a verifier total. Differences between inclusive totals and
children also contain unmarked host work and profiling overhead. The diagnostic
separately reports `router_batch_calls`, routed FFN batch count, affine dispatches,
and KDA prework/post dispatch counts to demonstrate which paths actually engaged.

The helper API is `glm5_dflash_profile.bind(?*Profile)` with scoped restoration,
`enterLayer(index)` with scoped restoration, and `Timer.start(rows)` followed by
`finish("field", outputs)`. Each finish settles the listed outputs, accumulates
that interval and resets the timer. `record("field")` records after an existing
evaluation boundary. Bindings are host-thread-local and the caller owns the
collector's lifetime. An unbound marker does not read a clock or evaluate arrays;
no global process environment flag enables instrumentation inside the engine.

Focused tests verify the disabled no-evaluation contract on a lazy nonzero tensor,
layer-scope attribution, nonzero captured features and all committed KDA/MLA state
with profiling enabled, and unchanged captures after the binding is restored.
The broader DFlash filter passed 22 tests with the full-checkpoint gate skipped.
These component checks do not replace profile-on real-checkpoint parity or a
separate profile-off throughput measurement.
