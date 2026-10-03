# Fixed four-output HC prefill expansion

Recommend one exact, fixed prefill HC expansion experiment. One thread owns the
four output streams for a feature, reusing their common residual and branch
loads. This changes neither dispatch count nor expert/attention arithmetic.
Its expected whole-work ceiling is modest; it cannot establish 1500/60 alone.
Start from accepted packed32 runtime `af51e72f`, current outcome checkpoint
`650c00cf`. No implementation, build or GPU job accompanied this research.

## Source evidence and fixed scope

`glm5_forward.Model.forward` calls `primitive.hcExpand` after attention and after
FFN in every layer. `glm5_next.HC_EXPAND` dispatches one thread per output
`[row,h,d]`. Each thread loads all four BF16 residual streams and the same BF16
branch feature, then computes its own FP32 four-term residual contraction,
separate FP32 branch product, final addition and BF16 store. Current prefill
already fuses HC collapse and KDA postwork; this proposal changes only expansion.
The previous HC multi-output work concerns the 24-column collapse dot, not this
four-stream residual expansion.

Strict candidate geometry: GPU/BF16 residual `[1,2048,4,4096]`, branch
`[1,2048,4096]`, original FP32 post/comb coefficients and original hardware guard.
Everything else retains `hcExpand`. Keep the original 256-thread group. The fixed
new grid is `2048*4096` threads; each writes four coalesced stream positions.
No threadgroup allocation, reduction, layout conversion, extra dispatch, retained
weight bank, new wait or async-policy change. Original output is still 64 MiB;
no resident or additional tensor bill is introduced. Admission keeps every
existing reserve and must still prove measured peak and ownership.

For each feature, use four explicitly named FP32 accumulators. For j=0,1,2,3,
load the residual scalar once and update each accumulator with its original
coefficient, in exactly the same per-stream order. Load the branch scalar once,
form four separate FP32 post products, then perform each original
`OutT(product + value)` store. Preserve both `fp contract(off)` and
`fp reassociate(off)`. Do not start the sum with the branch, contract an FMA,
share a sum between streams or add a different BF16 boundary.

## Complete-work ceiling and risks

At T2048/D4096 the original source issues 16 residual and four branch loads per
feature across four output threads. The candidate issues four and one. That
eliminates 240 MiB of logical input-load instructions per expansion, or
21.09375 GiB across 90 expansions in a full 45-layer chunk. This is **not measured
DRAM traffic**: the old reads can hit caches, and the compiler can already reuse
loads. Coefficient loads, output writes and every GEMM/recurrence remain necessary.
Four independent accumulators increase per-thread live values; fewer threads or
strided stream stores may offset the source load reuse. There is no occupancy
or bandwidth claim from the routed resource diagnostic.

The recorded forced T2048 component profile assigns 24.939 ms to attention
expansion and 24.110 ms to FFN expansion, 49.049 ms total out of 1788.124 ms.
Even eliminating that entire old subtotal would remove only 2.74% of that
perturbed total. It includes boundary waits and is neither a current HTTP budget
nor a forecast. The candidate therefore needs a clear complete-layer win;
source counts alone do not justify runtime integration.

## One proof and decision gate

First compare all output BF16 bits to current `hcExpand` at the exact geometry,
using original L0 HC coefficients and fixed activations. Include four distinct
streams/head order, cancellation, signed zero, NaN/Inf and the existing
`GLM HC expansion rounds residual contraction before branch addition`
regression replicated at admitted geometry. Check default-off fallback for
T2047/T2049, wrong width/stream count and F32 inputs. Preserve output ownership
through evaluation and release; do not infer layout or backing lifetime from
lazy metadata alone. Stop any mismatch without arithmetic restoration.

Then one original L0 complete-layer component gate composes both HC collapses,
norms, unchanged actual KDA projections/prework/R4/output, unchanged actual dense
FFN and the two expansion endpoints. Use the original stored L0 banks and small
tensors with one frozen BF16 `[1,2048,4,4096]` input and original initialized
FP32 KDA/BF16 convolution state. Prove complete layer output and every final
state bit, not only expansion outputs. Three warmups per arm and eleven
alternating pairs include construction, all preparation, copies, state handling,
endpoint evaluation and every free. Hold identical fixture/reference memory in
both arms and preserve the two-layer model scheduling contract. Report scoped
peak and both expansion engagement counts. No isolated component win is a model
claim; stop noisy/slower complete-layer results with no thread/group variant.

A clear winner receives root's one-load matched long-prefill ABBA with equal
held reference Requests, exact final logits/valid state and continuation IDs,
then HTTP qualification. No rounding threshold, stored tensor, cache precision,
assistant default or target policy changes.

Worker ownership: new isolated helper/probe/result document only, with reuse of
the existing primitive body and licensing. Root owns the narrow `hcExpand`
delegation, conservative admission, complete-model evaluation and acceptance.
No new profiler, bank cache, retrieval policy or expert kernel is proposed.
Evidence: [current profile](glm5-next-wave-performance-plan.md),
[resource limits](glm5-routed-resource-result.md),
[retention rejection](glm5-prefill-weight-retention-result.md) and
[cold MLA rejection](glm5-dense-prefix-packed-result.md).
