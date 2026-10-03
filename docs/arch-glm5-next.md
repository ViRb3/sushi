# GLM-5.3-Flash: native engine foundation

The `glm5_next` source adapter preserves official BF16 expert tensors for native streaming. The native
text forward runs resident EXL3 and streamed BF16 checkpoints through an opt-in diagnostic harness. It is not
registered in `model.served_model_types`: production serving, full reference parity and quality gates
remain open. See [diagnostic usage](glm5-diagnostic.md) and [attention/cache details](engine-glm5-attention.md).

Related documents: [execution plan](plan-glm5-native.md), [DFlash2 study](plan-glm5-dflash2.md),
[correctness audit](glm5-correctness-audit.md), [efficiency audit](glm5-efficiency-audit.md),
[serial decode kernels](engine-glm5-decode.md),
[head-packed NAX attention](glm5-attention-head-packed.md),
[native batched decode attention](glm5-decode-attention-batch.md),
[bounded IndexPool scoring](glm5-indexpool-nax-score.md),
[stored affine assistant formats](glm5-dflash-affine-storage.md),
[singleton and KDA endpoint round](glm5-singleton-endpoint-round-plan.md),
[bounded draft readouts](glm5-dflash-readout-horizon.md),
[verification latent overlays](glm5-dflash-latent-overlay.md),
[verification projection batching](glm5-mla-verify-batch.md),
[KDA leaf retention](glm5-dflash-kda-leaf.md),
[external runtime comparison](glm5-external-efficiency-comparison.md), and
[internal reuse comparison](glm5-internal-efficiency-comparison.md). This document is the GLM documentation index.

## Checkpoint and implementation status (2026-10-03)

The `GLM-5.3-Flash-Sushi-2.25bpw-A8g128-W12` checkpoint is complete: 129 routed projection banks,
37,152 expert projections, 449 affine trunk matrices and 1,169 retained tensors. Its indexed tensor payload is
98,336,815,992 bytes (91.5833 GiB), including the optional MTP and vision weights. This is stored tensor size,
not a measured runtime memory requirement. Large trunk matrices use affine8 group128; preserved small
BF16/FP32 tensors keep their source precision. Creation details belong to the private Sashimi repository.
The completed checkpoint runs native full-model generation; lossless-teacher KLD remains open.

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

Subsequent diagnostics produce coherent English. The A6-trunk `Sushi-2.3bpw` target retains
K2.25/W12 experts and runs with the selected A6g128 DFlash2 assistant. Native prefill measured
888.97 tok/s at512 and1,159.73 at2K. The opt-in N2/async4/grouped-expert verifier reached45.45 tok/s
on512/64 with unchanged serial token IDs and complete cache state; this is one measured run.
Cache remains BF16 compressed MLA with FP32 KDA state. Lossless BF16 teacher capture
and both target comparisons remain pending after the slow capture was stopped. See the [current execution plan](plan-glm5-native.md)
for workload distinctions and remaining validation.

The later accepted prefill stack measured 931.71/802.82/727.81/655.06 tok/s
on predictable HTTP 2K/4K/8K/16K workloads. Optional native B1/B3 attention
subsequently measured 46.04/46.19/46.57/44.20 tok/s decode on those rungs;
its separate 32K predictable run measured 608.77 prefill and 43.00 decode.
These workload-specific measurements remain below the 1500/60 goals. A4/group64
assistant consumption is supported, but A6 stays default because matched total
decode gains were within control drift. See the linked benchmark and assistant
documents for exact input lengths, parity scope and comparison limits.

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
are not opened by this store; the native diagnostic loader loads the text trunk separately and excludes MTP.

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
  actual memory and per-layer attribution. Current targets are at least 1,500 tok/s prefill and
  60 tok/s speculative decode using DFlash2; these are goals, not measured results.
- Compare full MLA/reference layer outputs and logits: absorbed latent attention and expanded dense
  attention have different BF16 rounding boundaries even when their projection algebra agrees.
- Measure KLD against a lossless BF16 teacher and qualify selected changes through 32K. The identity-prior
  quantized checkpoint has no accepted full-model quality result yet.
- Replace diagnostic per-layer synchronous evaluation only after proving bounded memory and state
  equivalence. Profile KDA, attention, routing and projection dispatch before choosing optimizations.
- Integrate Transformer/server ownership, memory admission and tested capability checks. The diagnostic
  cache uses BF16 compressed MLA latents. This is the required GLM default, including future serving
  integration; do not inherit the generic KV8 or `--fast` KV8 preset. KDA recurrent state stays FP32.
  Generic cache bills do not describe this standalone request implementation's exact allocation. Add reset/rollback/prefix lifecycle integration
  before enabling dependent features. Vision, MTP, batching and public speculation remain unsupported;
  DFlash2 runs through its separate validated diagnostic path.

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


Current follow-up references: [KDA recurrence and quantized NAX research](glm5-kda-prefill-research.md)
records rejected schedules and verified backend dispatch; [DFlash expert reuse](plan-glm5-dflash-expert-reuse.md)
sets the route-overlap evidence and exact-arithmetic requirements for a later grouped kernel.
Coordinate swizzling alone has not demonstrated a useful gain; tile-aspect experiments remain isolated.

### Private HC activation capture

`glm5_hc_capture.zig` contains the gated test `GLM HC private real checkpoint fixtures`.
It reuses the indexed native loader and tokenizer. Set `SUSHI_GLM_HC_CAPTURE_MODEL`,
`SUSHI_GLM_HC_CAPTURE_OUT` (an existing empty absolute private directory),
`SUSHI_GLM_HC_CAPTURE_PROSE`, `SUSHI_GLM_HC_CAPTURE_CODE`,
`SUSHI_GLM_HC_CAPTURE_REVISION` and `SUSHI_GLM_HC_CAPTURE_BINARY_SHA256` explicitly.
Each prompt must encode to at least 512 tokens; the exact first 512 tokens are used.
The coordinator must grant the exclusive full-model slot before this test runs.

Four safetensors files contain prose/code 512-token prefill and the following one-token decode.
Each includes the input IDs, RMS epsilon, HC epsilon and Sinkhorn iterations. Tensor names are
`layerNN.attn|ffn.{x,w,scale,base,mix,mixed,post,comb}` for layers 0, 3, 23 and 44.
Inputs and stored weights retain their original dtype; `mix` is the staged FP32 RMS/projection
reference. Metadata records the supplied revision/binary hash, computed config/index/prompt hashes,
model path and capture settings. These are quality fixtures, not timing runs; no activations belong
in the repository. Binary provenance is supplied by the invoking runner and labeled accordingly.

The request-local hook is absent by default. It validates layer ordering, range and record capacity
before changing state, retains its own array handles, and settles captured tensors with cache outputs.
A nonzero synthetic model test compares logits and all cache arrays through prefill and decode and
checks invalid selections leave request state unchanged. Actual full-checkpoint fixture generation
requires a separate explicit run; implementing this hook does not establish factored-HC quality.

### Affine6 copy-free serial QKV

`glm5_decode.zig` accepts uniform affine6/group128 QKV banks as well as the existing affine8 banks.
It infers stored bits from the packed row width, requires materialized contiguous uint32 weights and
BF16 scale/bias grids, and declines mixed six/eight-bit banks. Unsupported geometry retains native
MLX fallback. No weight concatenation or repacking is required.

The six-bit kernel follows the pinned MLX `quantized.h` implementation: eight values and six packed
bytes per lane, four-value input sums, scaled local inputs, and six cross-byte partial products per
four coefficients. The order of those partial products, group accumulation and SIMD reduction is
preserved; unpacking complete six-bit codes first would change floating-point rounding. Existing
MLX MIT attribution applies. Dedicated kernel caches keep six/eight-bit source variants separate.

Raw BF16 parity passed against MLX affine6/group128 for all three 4096→8192 projections and smaller
unequal output banks. Existing affine8, dense/strided refusal tests and mixed-rate refusal also passed.
These component results establish exact arithmetic; they do not establish full-checkpoint speed.

### Rejected HC column group sizes 6 and 12

A standalone scheduling experiment retained the exact RMS and FP32 dot kernels but grouped six or
twelve HC outputs per threadgroup instead of the qualified 24. No RMS factoring or TF32 arithmetic
was used. Captured prose/code 512-token inputs at layers 0, 3, 23 and 44, for both attention and FFN,
matched the saved FP32 mixes and all mixed/post/comb outputs bit-for-bit.

Four warmup pairs preceded 24 alternating AB/BA pairs per candidate and capture point, comparing
C6/C12 directly with C24. Sum of the eight point medians was:

| Fixture | C24 paired with C6 | C6 | C24 paired with C12 | C12 |
|---|---:|---:|---:|---:|
| Prose | 2607.749 µs | 3298.645 µs | 2558.854 µs | 2795.978 µs |
| Code | 2548.479 µs | 3265.356 µs | 2549.249 µs | 2775.043 µs |

C6 was 26–28% slower and C12 about 9% slower. Reducing per-thread accumulator count did not offset
repeated RMS/input work and scheduling costs in this experiment. These are component measurements,
not full-model timings. The GPU lock and foreground QoS were used with other workers paused and ten
seconds idle. Fan maximum was requested but reported RPM did not confirm spin-up. The isolated test
and raw samples were archived and removed from the runtime tree; C24 remains unchanged.

### Native BF16 expert streaming

`glm5_stream.zig` connects the native diagnostic model to the existing individual-expert reader and fixed
cache/union slabs. The indexed loader retains only text trunk tensors before materialization. Admission
reserves the stored trunk, request caches/workspaces, full expert union, I/O bounce buffers and at least one
slot per MoE layer before any retained trunk is evaluated. The engine preserves exact BF16 source bytes;
router IDs are remapped to slab slots without changing scores, clamping or FP32 weighted reduction.

Execution currently supports one admitted request, BF16 experts, GPU GatherMM and synchronous layer
completion. The supplied maximum context and chunk bound are enforced; a chunk must be at most 512.
Reset/deinit releases request ownership. Resident EXL3 operation retains its existing loader and scheduling.
DFlash pairing, FFN reuse and tree verification refuse a streamed target with
`GlmStreamingSpecUnsupported`; CPU BF16 GatherMM refuses with `GlmStreamRequiresGpu`.

ReleaseFast reproduction: `zig build test -Doptimize=ReleaseFast -Dtest-filter='GLM stream'`.
The focused suite passed 17 tests, including raw BF16 parity with resident GatherMM through eviction,
oversized route unions and retained earlier outputs, native binding without resident expert banks,
trunk-only loading, budget refusal and request ownership. Separate malformed-source coverage rejects
missing projections, FP16, wrong shapes, truncated shards, dense-prefix experts and mixed formats.

The 2026-10-03 cold smoke used the original BF16 checkpoint, 8 prefix tokens, 2 generated tokens,
chunk8, no warmup/profiling or MLX free-buffer cache, a 100 GiB total/memory/wired budget and 8 GiB request
reserve. The rebuilt test binary SHA256 is `d9ff24214580a6de297f9f9e6bd5830180e48e4ffb350a606a8251c917166acc`
(base `fe5984de` plus this streaming change, built 07:59:58 local time), with foreground QoS and exclusive
GPU lock `glm53-native-stream-smoke-v61`.
It completed with finite logits and IDs 304/279, using 31 cache slots per layer and 98,411,115,340 peak
active MLX bytes, below 107,374,182,400 budget bytes. This is short-path execution and memory evidence;
full-model reference parity, teacher capture, quantized streaming and public serving remain open.

The gated diagnostic test accepts `SUSHI_GLM_DIAGNOSTIC_STREAM_GIB` and optional
`SUSHI_GLM_DIAGNOSTIC_STREAM_RESERVE_GIB`; the former is a total GiB budget and the latter defaults to8.
Use the existing explicit model, prompt/tokens, output, memory, wired, prefill, decode and chunk controls.
The computed conservative reserve can raise the requested reserve for longer contexts.

### Lossless BF16 streamed KLD teacher

`tests/capture_glm_bf16_teacher.py` reuses the diagnostic lossless BF16 expert reader with the pinned
oMLX GLM forward, independently of native EXL3 execution. Run it in the existing oMLX Python environment,
with explicit `--model`, `--omlx`, `--prompts`, and private `--out` paths. The deterministic corpus is
`tests/fixtures/glm5_kld_2code_2prose.json`. `--prepare-only` audits headers and tokenizes without loading
model tensors. Acquire the exclusive GPU lock before actual capture; this is a long correctness run.

The protocol matches standard KLD capture: two raw code seeds and two raw prose seeds, each followed
by exactly 512 greedy continuation rows, including rows after EOS. Seed lengths are 240, 261, 190 and
183 tokens with the official tokenizer; prefix chunks are at most 512. Store full-vocabulary little-endian
FP32 logits and exact IDs. First-EOS scoring is a separate comparison metric, not an early capture stop.
Source tensors must be BF16 or FP32; expert tensors must be BF16. oMLX's standard sanitize operation
losslessly widens HC/router BF16 parameters to FP32 and reshapes convolution weights. Before/after trunk
dtype counts are recorded. TF32, optional oMLX HC/KDA prefill fusion, decode fusion, MTP and speculation
are disabled. MLA caches retain BF16 values and KDA state remains FP32.

`--ssd-budget-gb 100` uses the established GiB convention: 107374182400 bytes. The MLX allocation limit
is 100 GiB and its free-buffer cache is disabled. After loading the trunk, the lossless BF16 expert LRU
is capped at the remaining budget minus a 24 GiB reserve for workspaces, current expert unions and host
read buffers. Prefill reads expert unions directly; serial decode reuses exact BF16 arrays. Every row
records active/cache memory, MLX peak and process peak RSS; exceeding the requested budget aborts the run.
No timing claim should be derived from this capture.

`baseline.json` is published atomically only after all four prompts finish. It uses the existing
`mlx-serve-kld-baseline-v1` schema and standard prompt directories with `prompt_tokens.txt`,
`generated_tokens.txt` and `logits.f32`. A durable per-row journal commits logits before token/NLL
metadata. Resume truncates any uncommitted tail, validates source/script/corpus identities, reuses
completed prompts, and reconstructs an interrupted prompt by serial replay with byte checks against
all committed logits. Progress and partial manifests are not complete teacher artifacts. CPU journal/LRU
and tiny cached/uncached resident-reference tests cover recovery, eviction/reload and exact BF16 bytes.
