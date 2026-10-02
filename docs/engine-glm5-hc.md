# Exact one-token GLM HC normalization and mix

`glm5_hc_fused.mix` accepts BF16 residuals `[1,1,4,4096]`, BF16 or FP32 HC weights
`[24,16384]`, and a finite positive RMS epsilon. It returns the 24 FP32 mix values,
or no candidate for unsupported shapes/dtypes. It does not change model routing
by itself. `dispatchCount`/`resetDispatchCount` support engagement checks on the
single inference thread.

The kernel replaces widening, RMS normalization and the mix projection with one
dispatch. It computes the native 1024-logical-thread RMS reduction using 128 physical
threads per output group, then retains the existing four-SIMD-group projection tree.
A volatile FP32 normalized value preserves the multiplication boundary that the
staged normalized array provided. BF16 weights are widened at use; no persistent
weight copy or wide/normalized activation plane is prepared. Sinkhorn, post weights,
stream collapse and expansion remain unchanged.

The normalization mapping follows MLX's four-values-per-thread `rms_looped` and the
oMLX HC helper; `NOTICE` records MIT/Apache provenance. Local source inspection was
supplemented by direct comparisons with the linked runtime, rather than assuming
that a source revision alone guarantees the installed backend's reduction shape.
The path is qualified on the tested M5 Max/runtime at this exact geometry; another
backend must pass the same parity tests before enabling it.

## Validation

Focused ReleaseFast tests compare all 24 FP32 mix bits for zero, small, ordinary
and large finite BF16 activations at three epsilons. They also compare actual staged
`Hc.collapse` against candidate mixes plus the existing collapse primitive: BF16
mixed output, FP32 post weights and FP32 Sinkhorn matrices are bit-identical.
BF16 and FP32 HC weights and unsupported shape/dtype/epsilon cases are covered.
The reference helper uses `Hc.collapseReference` when available, and the parity test
asserts the baseline did not dispatch the candidate, preventing fused-versus-fused
comparisons after integration.

## Warmed component measurements

2026-10-03, M5 Max, ReleaseFast, foreground `taskpolicy -a`, exclusive `glm-hc-fused`
GPU lock. Comparisons alternated AB/BA in one process. Timers include CPU graph
construction plus synchronous evaluation of all three HC outputs. This is a component
experiment, not a full-model decode result.

| Method | Staged median | Fused median | Observation |
|---|---:|---:|---|
| One HC graph per evaluation, 16 warm pairs and 200 timed pairs | 253.500 µs | 255.334 µs | No benefit; evaluation/host latency dominates |
| 16 queued full-HC graphs per evaluation, 8 warm pairs and 64 timed pairs | 23.685 µs/HC | 21.781 µs/HC | 8.04% lower amortized latency; fused won all 64 pairs |

The queued run used 16 separate input and weight allocations at the real geometry;
all three output arrays of every graph were evaluated. Its median paired saving was
1.843 µs/HC. A second single-graph check in that process remained flat/slower
(436.417 versus 441.021 µs), reinforcing why the synchronous round-trip timing should
not be used to predict production scheduling gains.

The benchmark used a 10-second idle cooldown. Fan-max was requested, but RPM readback
was zero with a minimum-speed target, so achieved max RPM was not verified; the first
run's observed hottest sensor was about 38°C. Fans returned to automatic afterward.
Raw timings, binary/source hashes and controller status are retained in the private
measurement ledger. The queued binary SHA-256 is
`6a0578e96322fcd7f24920ecbba2f903bb65dbdf8499791725ff93d72201ae91`, built over base
`12b24dc5` plus this candidate. Engagement counters were added after that measurement;
the kernel arithmetic is unchanged.

Even multiplying the paired component saving by 90 HC calls gives only about
0.16 ms/token before overlap effects. A full-model comparison is still required.
If that result is flat, the strongest contained alternative is to compute the RMS
inverse once, then let the existing projection read BF16 input and reproduce the
FP32 normalized products: two dispatches without repeating normalization 24 times.
