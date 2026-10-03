# GLM-5.3-Flash native execution plan

Status: active implementation, 2026-10-03. The native diagnostic runs the full text checkpoint and
produces coherent English. The active folder is `GLM-5.3-Flash-Sushi-2.4bpw`; its expert metadata is
K2.25/W12, with affine8 group128 trunk. Quantization does not need restarting. The additional `GLM-5.3-Flash-Sushi-2.3bpw`
pack is under measurement: it keeps K2.25/W12 experts and uses an affine6 group128 trunk.
The selected DFlash2 assistant is A6g128; BF16 assistant runs are historical baselines.
The public served architecture remains unsupported until the integration gates below pass.
Completed components and measurements are in [the architecture document](arch-glm5-next.md).
The [correctness audit](glm5-correctness-audit.md), [efficiency audit](glm5-efficiency-audit.md),
[external comparison](glm5-external-efficiency-comparison.md) and
[internal comparison](glm5-internal-efficiency-comparison.md) record the reviewed implementation.

## Progress

- Completed: KDA preparation, independent layer fixtures, bounded IndexPool/latent attention,
  stored-grid MLA comparisons, complete diagnostic forward, request reset and coherent 512/64 generation.
- Current A6-trunk target, warmed native512/64: 888.97 tok/s prefill,31.23 tok/s serial
  decode,94.314 GB peak. At2K/64 it measured1,159.73/24.42 and95.144 GB peak.
  Lane pair/down are enabled; dense SDPA prefill and async2/async4 schedules are explicit.
  Earlier A8-trunk results are separate measurements, not the same quantized model.
- Completed serial optimizations: async4 scheduling, copy-free QKV, fused KDA body, BF16-storage
  FP32 router and paired cooperative expert gate/up. Dense-prefill SDPA remains opt-in pending KLD.
- DFlash2 with the selected A6g128 assistant: N2/children4, async4, lane pair/down and
  opt-in grouped expert reuse reached45.45 tok/s on512/64 versus matched serial31.42.
  All64 output IDs and complete committed state match the previous42.43 tok/s N2 arm;
 24rounds/40 accepted drafts/72 verification rows are unchanged. Decode-only peak94.979 GB.
  This clears45 in one run; repeats and broader prompts/contexts remain open. See
  [grouped expert results](plan-glm5-dflash-expert-reuse.md) and [scheduling](glm5-dflash-scheduling.md).
- Completed reduced quality screen: the user stopped the teacher at one code prompt,
  512 BF16 continuation rows. A6/A8 trunk KL was0.09292886/0.09544605 against the same
  teacher IDs/logits, BF16 cache and FP32 KDA state. This is code-only coverage; see
  [the KLD report](glm5-kld.md). Further capture waits for improved native streaming.
- Remaining: optimize toward at least 1,500 tok/s prefill and 60 tok/s speculative decode
  using DFlash2; broader model quality coverage;
  broader long-context coverage; production loader/lifecycle/server integration. The sections below
  retain the acceptance criteria, including completed foundations, rather than implying each is missing.

## Intended first runnable configuration

Text only, one request, resident affine8 group128 trunk and resident EXL3 experts,
normal token embeddings resident, MTP off and speculative decoding off. Keep source
BF16/FP32 small tensors unchanged. Use BF16 compressed MLA cache by default and FP32 KDA
state. GLM must not inherit generic KV8 defaults, including the `--fast` preset. Do not load unused vision/MTP payload just because
it exists in the index. Keep lossless BF16 expert streaming available for validation.

The first execution target is a native diagnostic generation harness with the full
45-layer forward. Production serving, batching and prefix reuse must not silently
enter partially implemented paths. Explicit refusals are acceptable until those
features have their own state/restore coverage.

## 1. Establish layer ownership and reference fixtures

Touch `glm5_model.zig` and focused tests/fixtures before wiring the server.

- Give prepared layer constants explicit ownership and destruction, including partial
  initialization failure cleanup. Prepare combined convolution weights, `exp(A_log)`
  and required dtype conversions once. Keep immutable weights separate from request state.
- Use one `Ops` scope per layer. Evaluate outputs and compact cache state at the normal
  boundary, then release temporaries. Preserve the owned prefill convolution tail.
- Capture deterministic tiny KDA, mHC and MLA reference inputs/outputs using the recorded
  oMLX reference revision. Preserve intermediate dtypes and rounding; include initial
  state, final state, single-token decode and irregular prefill chunks.
- Exercise complete KDA assembly, not only the recurrence. Record per-boundary absolute
  and relative errors and the first divergence. Require exact equality for discrete
  decisions and established bit-exact paths; define numeric tolerances before accepting
  floating-point reductions, rather than widening them to accommodate a failure.

Exit: layer preparation/cleanup tests and KDA/mHC reference comparisons pass; repeated
requests do not retain old prefill buffers. A foundation test alone does not establish
that all lazily compiled Zig forward branches compile: invoke each through these tests.

## 2. Implement IndexPool and absorbed MLA

The native diagnostic implementation and small-shape oracle are complete. Retain these
contracts while optimizing or integrating the indexed GPU path.

- Implement source query/key/gate projections and normalization. Pool four tokens with
  the source positional bias and softmax over the pool axis. Maintain pooled keys and
  a bounded raw key/gate tail in request-local storage.
- For each query, select only completed causal pools. Select at most 512 pools (2048
  tokens), append the incomplete tail of zero through three tokens, and mask invalid
  slots. Below the selection budget use dense causal attention while still updating
  pooling history. Check the exact reference switch condition at the boundary.
- Preserve packed affine grids when slicing per-head `kv_b` key/value rows. Test both
  matrix orientations: absorb 256-wide queries into 512-wide latent space, attend with
  scale 1/16, then project latent results back through the value rows and output layer.
  This architecture has no MLA RoPE channels.
- Implement indexed latent attention without constructing a full long-context mask or
  a sequence-by-head-by-history score tensor. Chunk index scoring under an explicit
  scratch budget; account for its peak before choosing the full-model prefill chunk.
- Preserve BF16 compressed MLA cache as the GLM default in diagnostics and future serving.
  KV8 is not part of the performance target configuration; any later quantized-cache experiment
  needs an explicit opt-in and separate parity/quality results. Keep KDA recurrent state FP32.
  Billing must follow actual arrays and scratch ownership.

Exit: compare selected token IDs, pooled state and attention outputs against the oracle
at pool boundaries, 2048-token selection boundaries, nonzero offsets and irregular chunk
splits. Verify no future-token access, duplicate tail insertion or cross-request state.
Test cache reset and continuation beyond the selection budget.

## 3. Assemble the full native forward

Depends on stages 1 and 2. Extend `glm5_model.zig`, Transformer initialization/dispatch/
cleanup and the existing loader; avoid a second competing loading policy.

- Load only required text trunk tensors and resident expert banks. Gather packed affine
  embedding rows and dequantize on GPU; do not introduce SSD token embedding lookup.
- Assemble 34 KDA and 11 MLA layers in the configured order. Connect mHC collapse,
  branch norms, attention, expansion, FFN collapse and expansion at source boundaries.
- Implement sigmoid routing in required FP32 precision: correction bias affects top8
  selection only; gather unbiased scores, normalize and scale by 2.5. Add the shared
  expert and invoke the clamped EXL3 expert path. Exercise rates 2 through 4 bpw.
- Finish stream reduction, final normalization and the affine output head. Update logical
  token/cache offsets exactly once per forward and implement hidden-output requests
  used by diagnostics; reject unsupported modes explicitly.
- Keep request state in the existing context/cache ownership model. Add load/unload,
  failed-load cleanup, reset, cancellation and separate-request coverage. Either prove
  cache restore/rollback or explicitly disable the dependent serving features.

Exit: a tiny complete native model produces reference-compatible logits and state for
prefill followed by decode; serial and chunked runs agree. Routed/shared experts and
both layer types are exercised. Lossless resident/streamed expert modes agree on the
same tiny weights. Full ReleaseFast tests pass without changing served architectures.

## 4. Run the real checkpoint locally

Depends on stage 3. Use a dedicated diagnostic entry point/test harness while the public
architecture gate remains closed; do not globally enable an incomplete architecture to
make a smoke test convenient.

- Validate pack headers/stamps and resolve the actual loaded tensor set. Preflight
  resident weights, KV, FP32 recurrent state, index history and peak scratch against
  available memory. The 91.5833 GiB stored payload is not a measured RAM requirement.
- Acquire the GPU lock for each full-model run; no concurrent large load or conversion.
  Build ReleaseFast immediately before measurement and record binary/commit, settings,
  checkpoint identity, cache mode and prompt/token counts. Follow normal thermal/QoS rules.
- Start with a minimal prompt and a few tokens to expose loading, shape and allocation
  failures. Then run the requested 512-token prefill and 64-token greedy decode using
  the official text template, MTP off. Record output, timings and peak active Metal memory.
- Inspect coherent English, finite logits and state continuity. Repeat a deterministic
  request after reset to detect stale state. Coherent output alone is not a quality gate.

Exit: actual local native EXL3 generation succeeds, with a reproducible report and measured
memory. This is the first runnable milestone, not yet a performance or quality claim.

## 5. Validate quality, then enable supported serving

- Compare full-layer boundaries against the reference on the same stored weights to
  isolate engine errors from quantization error. Capture/verify a lossless BF16 teacher;
  never use the quantized checkpoint as its own quality reference.
- Run the standard 16x512 first-EOS KLD comparison and document the actual result. There
  is no established GLM acceptance threshold yet; do not invent a passing number or
  treat successful execution as proof of acceptable quantization quality.
- Extend context testing to 4K and 16K first, and 64K after those pass. Report reference
  agreement, cache growth, prefill, decode and failures separately. Preserve known BF16
  diagnostic timings as historical context, not a controlled native speed baseline.
- Open `served_model_types` only with the completed forward and tested capability
  checks. Add a live serving smoke test and cache/state tests for each enabled feature;
  keep unsupported vision, MTP, batching and speculative modes explicitly refused.

Exit: native quality is characterized, correctness failures are resolved, supported
serving modes pass their tests, and documentation states the precise supported scope.

## 6. Optimize measured bottlenecks

After correctness and memory gates, profile the complete forward. Audit per-token weight
transforms, host synchronization, temporary lifetimes, index scoring and expert dispatch.
Use interleaved comparisons on the same boot for small speed differences, with identical
settings and output checks. Keep improvements that alter numerics behind a repeated KLD
comparison. Do not claim that primitive parity or lower allocation count proves a speedup.

## Checkpointing the work

Commit each completed stage with its tests and update the architecture document. Keep
remaining items here, remove stale completion claims, and retain raw paths in the local
measurement ledger. Ask the two auditors to review the integrated state after stages 3
and 5; their current reports cover the foundation only. No additional quantization,
MTP/vision implementation or unrelated Qwen work is required to reach first generation.

## DFlash2 follow-up after serial tuning

The owner supplied a GLM DFlash2 draft checkpoint and requested that serial decode be tuned first.
The [separate DFlash2 study](plan-glm5-dflash2.md) maps mlx-serve's Qwen27B verification tree to GLM.
It identifies mHC feature capture, branch-local IndexPool and KDA replay requirements; tree support
is not implied by the existence of a compatible draft checkpoint. No speculative implementation
is part of the current serial optimization measurement.
