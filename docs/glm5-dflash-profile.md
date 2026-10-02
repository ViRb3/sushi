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

## First full-checkpoint profile

At `daa6d524`, the profile-on 512-prefix/64-committed-token N3 run passed all token
and complete final-state checks. A separate process using the same binary with
profiling off passed the same checks and produced identical tokens, including
the earlier tree-fusion N3 output. Both retained the original BF16 assistant,
chunk-128 captured prefix, affine row tiles, batched FFN, one warmup, 23 rounds,
41 accepted drafts and 92 verified rows. The new batched router engaged 966 times.

The profiled verifier took 4427.24 ms. Disjoint top-level component totals account
for 4416.23 ms; the remaining 11.01 ms includes unmarked host work. The following
numbers include the added synchronization and must not be treated as GPU-only
times or substituted into the normal throughput measurement.

| Component | Total ms |
|---|---:|
| FFN overall, inclusive | 1687.91 |
| FFN routed experts, child | 1079.12 |
| FFN shared expert, child | 273.30 |
| FFN router, child | 159.29 |
| FFN combine, child | 133.93 |
| KDA Q/K/V | 330.26 |
| KDA lowrank/beta | 184.21 |
| KDA prework | 134.98 |
| KDA recurrence | 127.15 |
| KDA gate/post | 162.79 |
| KDA output projection | 227.57 |
| MLA common projections | 110.01 |
| MLA branch attention/unembedding | 186.54 |
| MLA output projection | 116.45 |
| HC attention / FFN collapse | 241.49 / 239.72 |
| Attention / FFN norm | 149.71 / 154.18 |
| Attention / FFN expansion | 150.95 / 151.93 |
| Existing layer settle | 18.64 |
| Head / embedding | 37.76 / 3.98 |

Several tiny elementwise/norm/expand stages cost about 0.14–0.15 ms per marker,
demonstrating how strongly synchronization perturbs this run. These observations
support investigating routed experts and KDA projections with separate controlled
component experiments, not assigning all recorded time to their arithmetic. The
routed-expert stage is the largest individually timed arithmetic stage; actual
route-overlap data is still needed before predicting a gain from expert reuse.
The profile does not identify the recurrence as the main remaining KDA cost.

With profiling **off**, normal committed-token throughput was **27.6650 tok/s**
versus **26.2660 tok/s** for the matched serial reference, a ratio of **1.05326**.
Draft/verify/replay/commit totals were 156.76 / 2087.18 / 44.23 / 24.23 ms.
Captured prefill measured 369.64 tok/s; decode-phase peak was 98.534 GB. The peak
counter resets after prefill, and the prefix path differs from the separately
measured dense chunk-512 serial benchmark.

This normal run does not establish a router throughput gain over the previous
27.9314 tok/s tree-fusion sample: matched serial throughput also declined by
approximately 1%, while the relative advantage remained about 5.3%. Do not use
the profile-on 13.716 tok/s as a throughput comparison; the JSON explicitly marks
it noncomparable and reports no speedup ratio. The 60 tok/s decode goal remains open.

Private artifact `glm53-dflash-profile-20261003` contains separate `profile-on`
and `profile-off` result/provenance/summary directories. Both processes had their
own GPU lock, interactive QoS, max-fan request and ten-second idle, then restored
fan auto and released the lock. Initial temperatures were 44.53°C and 48.78°C.
The shared binary SHA-256 is
`f9ee41a3d8b01375cc07a1b12084683521354c707143fc4dbc31ac10b29a9d3b`.
