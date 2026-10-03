# Shared-bank native attention safety

The isolated Q256/K2052 wrapper passed finite arithmetic and causality guards.
The combined shared-retrieval candidate then passed its inclusive component
gate. The source remains uncommitted and changes no production dispatch or
default; the fixed long-context quality/model gates still decide acceptance.

Built-in native D512 attention with a Boolean array mask did not protect
future-local nonfinite rows. Placing BF16 NaN or infinity in the fourth local
row made all 98,304 earlier-query output values nonfinite and different from
the finite control. Valid historical nonfinite values also propagated, as
required; they were not silently sanitized. This safety-only diagnostic is
recorded under `glm53-shared-native-safety-20261003`.

The wrapper accepts `q[B,4,64,512]`, a contiguous BF16 `bank[B,2052,512]`,
Boolean `valid[B,2052]`, and the original scale. B is 1–16; only complete
four-query groups enter. The producer guarantees that the first 2048 bank
slots contain causally valid historical selection or zero invalid rows, and
that the last four slots are the group's local rows in chronological order.
The wrapper generates the original per-query Boolean mask. It retains the
conservative 64 MiB graph cap; its B16 ledger is 54,644,800 bytes, while combined
selector/producer peak must still be measured.

The copied native body is pinned to MLX
`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`, with MIT attribution in `NOTICE`.
BQ32/BK32, WM2/WN4, score reduction, online softmax, MMA operations and FP32
accumulators are unchanged. All 64 full historical tiles still load contiguous
bank fragments. Only the final four-row tile's K/V `load_rows` limit changes
to `min(4,logical_query+1)`, so future local rows become zero before MMA. The
original explicit score mask remains. There is no original-cache indirection,
valid historical sanitization or arithmetic restoration.

The source and probe are uncommitted. Qualification under
`glm53-shared-native-port-20261003` passed three tests, skipped the previously
recorded built-in diagnostic, and had no failures. Unchanged packaging and
safe finite outputs matched current native SDPA bit-for-bit at B1/B16,
including independent Q64 query/head-order controls: 6,684,672 BF16 values.
Earlier-query NaN/Inf outputs matched the finite control in 196,608 values.
Valid current-local and historical nonfinite values remained observable.
Empty masks returned zero; partial groups declined. One host optional-return
coercion error was repaired before the successful build.

Artifacts record source/shader/runtime/binary hashes, build recipe, flags and
thermal/fan state. Both short GPU jobs released their locks and restored
automatic fans. No wrapper-only performance claim is made.

The producer's complete captured T2048 gate subsequently matched 67,108,864
same-policy output bits against independent gathered Q64 native controls.
Its inclusive median was 130.207→60.846 ms with 11/11 wins, including selection,
metadata, gathering, masks, native attention and settlement. This is the
combined candidate's component result, not a wrapper-only timing or model
quality result. Mean pool recall was 69.91% and positive index-score missed
mass 21.89%; the retrieval approximation still needs the declared quality
screen. See [producer component evidence](glm5-shared-bank-prefill-component.md).


## Final disposition

The [fixed actual-model screen](glm5-shared-bank-quality-result.md) rejected
the policy on both16K inputs. Component math/safety and speed did not establish
acceptable model quality. All candidate source/delegation was archived and
removed, with accepted runtime unchanged; no default or performance claim landed.
