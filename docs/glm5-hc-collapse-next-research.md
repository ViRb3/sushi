# Exact short-row mHC coefficient preparation

Source-only at accepted runtime `af51e72f`. Recommend one fixed
**SIMD32-cooperative HC_COLLAPSE coefficient/Sinkhorn body**, with the existing
mixed-output loop unchanged. It targets a serial dependency repeated throughout
verification, rather than another expert microvariant. No prototype, build or
GPU work accompanies this report; speedup is unmeasured.

## Source and evidence

`glm5_dflash_model.verify` calls `Hc.collapse` before attention and FFN in every
one of 45 layers: 90 collapses per complete verifier pass. `Hc.collapse` computes
its original RMS/FP32 projection, then `glm5_next.hcCollapse`. In HC_COLLAPSE,
thread 0 computes all four pre/post coefficients, the 4×4 row softmax and all
20 Sinkhorn iterations. The other 255 threads wait at one barrier before the
current mixed-output loop. The config parser/default confirms 20 iterations.
This serial coefficient work is concrete; its latency share is not measured.

The earlier optional HC fusion changed normalization/mix and had no demonstrated
serial-model benefit. It did not change this Sinkhorn coefficient body. The
recent singleton/fused-middle expert candidates lost, and four-headgroup scoring
was noisy. The [resource diagnostic](glm5-routed-resource-result.md) supplies
expert pipeline facts and overlapping command-buffer intervals, not an HC cost
or bandwidth budget. None of those results supports another expert arm.

A4 remains optional: its draft phase is consistently faster, but ordinary
acceptance and total decode are mixed, including more rounds at ordinary 8K.
A6 stays fixed for this exact target-work experiment. Faster drafting alone does
not remove the dominant verifier phase or establish 60 tok/s.

## One fixed exact body

Strict B1/T3/four streams/H4096, BF16 X, FP32 mixes/scale/base, positive finite
HC epsilon and exactly 20 iterations. Keep the same 256-thread group per row.
Only subgroup 0 performs coefficients; its first 16 lanes each own one matrix
entry, and the first four also emit pre/post. Other subgroups retain the current
mixed loop after the same final threadgroup barrier.

Use explicit subgroup shuffles to fetch each row's or column's four entries.
Each lane computes the original sequential four-element maximum and FP32 sum,
starting at the same -infinity/zero, then the original division. Preserve
`maximum = max(maximum, entry[k])` for k0–3 with the same operand order,
including NaN behavior; do not replace it with a subgroup maximum. Do not use
`simd_sum`, parallel prefix totals, reciprocal substitution or reassociation.
Preserve contraction-off/reassociation-off pragmas, precise exp, initial
`matrix/sum + epsilon`, later `matrix/(sum + epsilon)`, and the skipped row
normalization at iteration 0. All 32 subgroup lanes participate in each shuffle;
unused lanes cannot supply an entry or diverge around it.

Write the same row-major FP32 comb/post and shared pre/matrix arrays. Keep the
original four-stream accumulation and BF16 mixed store. Post/comb remain FP32.
There is no new global buffer or state, weight conversion, precision restoration,
commit rule or proposal policy. Original shared arrays remain 80 bytes; compiler
resources and latency must be measured, not inferred from this allocation.
T1/T2/T4, prefill and other configs retain the original implementation.

## Bounded gates and ownership

One worker owns a new isolated helper/probe/doc. Root alone owns the
`glm5_next.hcCollapse` delegation/counter seam, avoiding the independent HC_EXPAND
worker's shared-file conflict. No runtime hook precedes a component win.

Use the existing real HC capture contract: layers 0/3/23/44, attention and FFN,
with X/mix/scale/base/epsilon/iterations. Three adjacent rows from each 512-row
fixture are a constructed T3 component, explicitly not captured verifier rows.
Confirm fixture availability first; do not silently replace it with synthetic
inputs or start a target load. Prove every FP32 pre/post/comb and BF16 mixed bit
against the current body, plus downstream normalized/expanded outputs. Include
row permutations, signed zero, extreme logits, NaN/Inf and barrier/partial-shape
fallback guards. Any mismatch rejects the candidate without a tolerance arm.

One complete eight-block collapse component uses three warmups and eleven fresh
alternating pairs, including metadata/configuration, all outputs, evaluation
and frees. Keep identical immutable inputs and references; include measured
peak/engagement. This measures the coefficient-plus-mixed operation, not an
isolated Sinkhorn loop or all 90 actual model calls. There is no iteration/subgroup
sweep. Stop noise/loss without another variant.

Only a clear component winner receives root's fixed 8192-input/192-output
N2/children4/A6/native matched ABBA, equal references, every round/cleanup, strict
serial IDs and all valid state bytes, unchanged acceptance and measured peak.
Record actual 90-per-pass engagement and complete decode time. An accepted winner
then qualifies HTTP 2K–16K and selected 32K. No predicted percentage or 60 tok/s claim
precedes those gates; the geometry is context-independent but total costs are not.
