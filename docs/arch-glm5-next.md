# GLM-5.3-Flash: native engine foundation

The `glm5_next` source adapter preserves official BF16 expert tensors for streaming tests. The native
text forward now runs the completed EXL3 checkpoint through an opt-in diagnostic harness. It is not
registered in `model.served_model_types`: production serving, full reference parity and quality gates
remain open. See [diagnostic usage](glm5-diagnostic.md) and [attention/cache details](engine-glm5-attention.md).

Related documents: [execution plan](plan-glm5-native.md), [DFlash2 study](plan-glm5-dflash2.md),
[correctness audit](glm5-correctness-audit.md), [efficiency audit](glm5-efficiency-audit.md),
[serial decode kernels](engine-glm5-decode.md),
[external runtime comparison](glm5-external-efficiency-comparison.md), and
[internal reuse comparison](glm5-internal-efficiency-comparison.md). This document is the GLM documentation index.

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


## First native full-prompt measurement

2026-10-02, integration `33c85349`: the resident text-only diagnostic completed the same official-template
512-token prompt used by the BF16 sanity run and generated 64 tokens of coherent English about rain.
MTP/speculation were off, BF16 attention cache and FP32 recurrent state were used, prefill chunk128,
one same-shape warmup with reset, no prefix reuse, and synchronous per-layer profiling enabled.
Foreground QoS, an exclusive GPU lock, max fans and a cool start were used.

| Metric | Native diagnostic result |
| --- | ---: |
| Prefill | 342.6664 tok/s |
| Serial decode, 63 forward steps after first prefill token | 18.4368 tok/s |
| Peak active Metal bytes | 96,066,533,660 (89.47 GiB) |

Output began `</think>Rain forms through the water cycle. The sun heats water in oceans, rivers, and lakes`
and continued coherently to the fixed length. This is one workload, not a KLD result. These measurements
are below the requested 1,000 tok/s prefill and 60 tok/s decode targets. Serial scheduling and memory
residency are the next tuning steps. DFlash2 integration follows serial tuning; its design study is separate.


### Serial tuning measurements

The following follow-ups use the same 512/64 prompt, warmup/reset, EXL3 checkpoint, BF16 attention
cache and exclusive-run protocol. Scheduling code is `379ff876`; diagnostic controls are `bfb5931c`.
The final two rows use the same binary and change only the named setting relative to async/chunk128.

| Arm | Prefill chunk | Prefill tok/s | Serial decode tok/s | Peak active GB |
| --- | ---: | ---: | ---: | ---: |
| Synchronous layer profiling baseline | 128 | 342.67 | 18.44 | 96.067 |
| Async4 decode, profiling off | 128 | 345.45 | 24.02 | 96.067 |
| Async4 plus fit residency, zero slack | 128 | 344.39 | 24.12 | 96.067 |
| Async4, original residency, larger prefill | 512 | 527.21 | 23.40 | 96.788 |

Async4 and fit residency each retained all 64 output IDs from the baseline. Fit has no demonstrated
speed gain at this precision. Chunk512 produced coherent English but changed one word near the end;
changing GEMM row shape changes rounding, so this is not a byte-equivalent scheduling improvement.
It remains an explicit benchmark setting, not proof of full-model quality equivalence. The requested
1,000/60 tok/s targets are still unmet; all measurements here are serial, without DFlash2.

The opt-in source-style dense-prefill path (`324bd016`, diagnostic `6ceb3eea`) measured
745.60 tok/s prefill and24.53 tok/s serial decode at chunk512, peak 96.788 GB. It produced
coherent English with different token choices from absorbed prefill, as expected from the
different BF16 rounding boundaries. Tiny reference and causal-boundary fixtures pass;
whole-model KLD is still required before treating this experiment as a quality-equivalent default.

An optional two-layer prefill schedule overlaps host construction with GPU work while keeping
at most two layer graphs in flight. Nonzero fixtures at17/33/2-token chunks preserve every logit
and cache bit; profiling retains synchronous layers. The final cache-inclusive evaluation also
settles an odd final layer. Actual full-model peak memory and throughput are measured separately.


## Draft-model integration hooks

The diagnostic model exposes resident embedding lookup and an unnormalized output-head projection
for a separate assistant. Optional capture requests name sorted target layer IDs and own their output
handles. Captures are the mean of the four post-layer residual streams, before final normalization.
They are evaluated with layer/final cache outputs to avoid retaining an unevaluated full residual history.
A regression checks shape, dtype, value, head-without-extra-normalization semantics and fail-closed
layer selection. These hooks do not by themselves implement speculative verification or serving.


The combined scheduling/QKV/metadata version (`b133d4ce`) measured747.83 tok/s prefill and24.57 tok/s
serial decode on512/64, peak 96.965 GB, with all 64 output IDs equal to the prior dense-prefill arm.
All2,142 timed KDA QKV calls used the new kernel. These extra changes did not demonstrate a material
speed gain beyond the earlier async/dense-prefill improvements.

A separate2048/64 context run (`ddcf219f`, chunk2048, eight warmup decode steps) measured863.10 tok/s
prefill and20.23 tok/s decode, peak99.689 GB. It crossed the live sparse-selection boundary and produced
coherent English. This is a different workload, not a same-length speedup or achievement of1,000/60.


The BF16-storage router correction plus paired cooperative gate/up projections (`0188bb9d`)
measured **749.28 tok/s prefill and 26.44 tok/s serial decode** on the same warmed 512/64 workload,
peak 96.965 GB (90.306 GiB). All 64 output IDs match the fused-KDA arm (25.50 tok/s decode).
Both router and paired-expert counters recorded 2,646 calls; QKV and KDA body recorded 2,142 each.
This combined change measured 3.67% faster decode; it does not isolate either component's contribution.
The earlier FP32-storage-only router arm recorded zero router calls and establishes no router gain.
The 1,000/60 targets and whole-model quality gate remain open.


The subsequent activation/sorted-finish/selection-batching arm (`24b1437b`) measured
765.94 tok/s prefill and 26.54 tok/s decode on the same 512/64 configuration. All 64 output IDs
remain equal; 2,880 dense/shared activation calls engaged. Prefill was 2.22% faster in this pair;
the 0.39% decode difference is not a robust isolated gain. Peak active memory was97.020 GB,
55.15 MB above the preceding arm despite removal of the sorted-output scatter intermediate;
do not infer a whole-model memory saving from the local buffer removal. Combined ReleaseFast
validation passed 3,028 tests with 105 skipped. These results remain diagnostic, without public serving.


A follow-up synchronous layer profile used the same 512/64 inputs and retained all output IDs.
It measured 48.42 ms/token with profiling barriers: 30.57 ms across 31 KDA/MoE layers,
13.58 ms across 11 MLA/MoE layers, 2.76 ms across the three dense KDA layers, and 1.50 ms
outside those layer timers. MLA layers average 1.235 ms versus 0.986 ms for KDA/MoE layers.
These are complete-layer times, not component attribution, and must not be compared directly
with async4 throughput. Further serial profiling should separate HC, projections and expert work
inside representative layers before choosing another large fusion.


DFlash2 now passes warmed full-checkpoint 512/64 token and final-state parity at 2, 4 and 8
verification rows, with the original BF16 assistant. The four-row arm measured27.03 tok/s versus
26.54 for its matched serial reference; the two-row and eight-row arms were slower. This small
single-run gain does not establish a robust default or satisfy60 tok/s. The matched comparison
commits all64 output tokens, unlike the native serial harness's63 timed forwards. Detailed rates,
phase costs, precision and memory scope are recorded in [the DFlash2 document](plan-glm5-dflash2.md).


### Prefill output fusion and decode attribution

A subsequent 512/64 run preserved all output IDs and reduced peak active memory from
97.020 to96.878 GB. The retained prefill output fusion measured783.03 tok/s with the
original serial decode operations (26.54 tok/s). It engaged34 prefill epilogues. The full
suite passed3,034 tests with108 skipped before the final policy-only default changes;
focused HC and middle-mode tests passed afterward.

| New prefill output fusion | HC decode fusion | Middle/down decode fusion | Prefill tok/s | Decode tok/s |
|---|---|---|---:|---:|
| On | On | On |782.27|26.11|
| On | Off | On |782.16|26.18|
| On | On | Off |782.33|26.49|
| On | Off | Off |783.03|26.54|

All four arms generated identical64 IDs. HC had no demonstrated end-to-end benefit and is
now opt-in. The middle/down fusion regressed serial decode and auto mode now keeps the original
one-row path; its short-multirow candidate remains available for separate verification tests.
Warm component timing must not substitute for whole-model evidence. The1,000/60 target remains open.

An untimed512-token routing capture covered all42 MoE layers:512 histogram entries each,
4,096 assignments per layer, with224 trailing entries zero for the288-expert model. It shows
that the existing NAX kernel already skips the unused second16-row operation in short windows.
Changing all32-row windows to16 would preserve the MMA operation count while increasing weight
redecode work; the next experiment targets accumulator register pressure instead of assuming
that every short expert run computes32 rows.


The retained parallel KDA prework fusion (`0f634491`) measured **836.77 tok/s prefill and
26.50 tok/s decode** on the same warmed 512/64 workload, with every output ID unchanged.
Both prework and output epilogues engaged34 times; experimental serial HC/middle fusions
were off. Peak active memory was96.727 GB, another151.39 MB below the output-fusion-only
control. Prefill improved6.86% from783.03 tok/s; decode is unchanged within this measurement's
noise. Full ReleaseFast validation passed3,038 tests with109 skipped. The1,000/60 targets
remain unachieved; current follow-up is quantized NAX tile locality, not a claimed missing
NAX dispatch. Rejected recurrence/window variants are documented with their measured costs.
