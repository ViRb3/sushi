# GLM-5.3-Flash correctness audit

Date: 2026-10-02. This report covers the native GLM components and loader work on the
`codex/glm53-streaming` branch. It records confirmed findings and regression checks;
it does not establish full-model correctness. The served-architecture gate remains closed.

## Scope and references

Reviewed:

- `glm5_next.zig`: per-channel KDA recurrence, FP32 recurrent state, mHC collapse/expand,
  and the clamped EXL3 expert interface.
- `exl3/root.zig`: routed input and packed-bank contracts.
- `model.zig`: GLM configuration parsing, compressed MLA geometry and state-memory bills.
- `glm5_model.zig`: affine linear binding, preserved KDA/HC weights and the initial layer operations.
- `expert_quant.zig` and `mimo_source.zig`: GLM expert nesting, layout discovery and pack preflight.

References were the official GLM-5.3-Flash BF16 configuration and tensor headers, the
existing EXL3 consumer contract, and oMLX `6745c39c`'s vendored GLM implementation.
The audit used source inspection, small synthetic tests and targeted GPU checks. It did
not load the complete native model or run a native quality evaluation.

## Confirmed findings and fixes

| Finding | Evidence and consequence | Fixed status |
|---|---|---|
| mHC expansion added the branch product before accumulating the residual contraction. | A cancellation-sensitive example with BF16 inputs produced `-0.06494140625`; independently contracting the residual and then adding the rounded branch product produced `-0.0654296875`. The GPU regression reproduced the divergence. This was a rounding-order difference, not a measured KLD regression. | `e0d38137`: residual contraction precedes branch addition; independent BF16 regression passes. |
| The clamped expert API did not validate bank dimensions and scale grids before dispatch. | An invalid `[E,129]` scale grid for a 128-wide projection returned a result instead of rejecting the input. Raw kernel indexing assumes the correct grid. | `e0d38137`: validates H128 alignment, matching projection geometry and expert counts, scale shapes, trellis rates and dtypes, routed input/score shapes, and checked row products. Fourteen negative cases pass. |
| GLM parsing silently accepted contradictory computational settings and incomplete geometry. | Defining fields such as mHC, router scoring, activation, projection bias and pooling settings could disagree with the implemented path. Missing common dimensions could inherit generic defaults; zero dimensions and inconsistent layer tables were accepted. The official nested vision metadata also enabled the generic vision flag. | `99a7321c`: strict required-field and semantic checks, positive/checked dimensions, consistent layer tables, and text-only vision handling. Twenty-five mutation cases reject invalid configurations; shared-expert counts zero and two remain supported. |
| A test-local `projection` name shadowed the model helper. | The new module failed compilation. | `f352c5d6`: renamed the local binding; targeted module tests compile and pass. |
| KDA preserved tensors lacked load-time shape/dtype checks. | A scalar output norm was accepted and could broadcast over every channel. Malformed convolution windows, narrowed decay parameters and incorrect bias widths were not rejected by the binder. HC coefficients likewise lacked their FP32 storage check. | `f352c5d6`: exact preserved-tensor contracts and positive/negative load tests. The malformed norm case was observed failing before the fix. |
| GLM pack validation could ignore orphan MTP expert components and misplaced banks. | Preflight looked only for an MTP gate trellis, so an isolated up/down/scale component could escape validation. Packed tensors in the dense prefix or outside the declared trunk/MTP range were not comprehensively rejected. The orphan-MTP regression failed before the fix. | `554fcbfd`: any present MTP expert component requires a complete bank; invalid layer placement is rejected. Index discovery also rejects packed fragments in the dense prefix. |

## Verification and limits

Targeted ReleaseFast checks passed for:

- The independent mHC BF16 rounding regression and scalar Sinkhorn collapse/expand comparison.
- Clamped expert input validation and scalar host comparisons for all 17 supported eighth-bit
  rates from 2 through 4 bpw, with one-row decode and 17-row prefill cases.
- Official GLM configuration geometry, invalid configuration mutations, supported shared-expert
  counts, and the preserved KDA/HC tensor contracts.
- GLM BF16/EXL3 discovery and malformed packed-bank preflight.

The KDA primitive's existing independent scalar recurrence and serial/chunk state checks were
reviewed alongside the implementation. The equations, projection names and source convolution
layout matched the inspected reference. No additional recurrence defect was demonstrated.

The cache accounting matched the coordinator's declared storage contract: two quantizable
512-wide MLA latent buffers, BF16 pooled indexer history, a raw key-plus-gate ring, and FP32 KDA
state. For the official geometry, the dense KV bill is 22,528 bytes/token, pooled history is 704
bytes/token, the ring is 180,224 bytes/slot, and one recurrent/convolution checkpoint is
147,619,840 bytes. Attention scales by the original 256-wide query (`1/16`), not the cached
512-wide latent. These are contract checks; they are not measured whole-model peak memory.

The efficiency auditor separately owns the convolution-tail allocation regression and the
combined full-suite run. This report claims only the targeted checks executed or inspected
by the correctness auditor. Numeric expert-ID bounds remain a router precondition; the public
expert API does not synchronize GPU IDs to the CPU on every call to check their values.

## Remaining unvalidated areas

The native forward still requires complete integration and independent validation of the full
layer loop, sparse MLA and IndexPool selection, router/shared-expert composition, embedding and
output paths, cache advancement/restoration, and any MTP path. Component tests do not prove those
connections correct.

Required release evidence remains native short/long-context reference parity, chunked versus
serial behavior through the complete forward, full-checkpoint coherent generation, quality
measurement against a lossless teacher, and measured memory/admission behavior. The separate
BF16 diagnostic generation recorded in the architecture document used a reference Python
forward; it does not validate the native Sushi forward. No native throughput or KLD conclusion
follows from this audit.
