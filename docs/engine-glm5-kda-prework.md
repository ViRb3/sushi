# Parallel GLM KDA prework

`glm5_kda_prework.apply` computes depthwise four-tap convolution, BF16 SiLU, Q/K L2
normalization, FP32 vector decay and BF16 beta sigmoid in one dispatch. One 128-thread
group handles each token/head. Recurrence, output gating/normalization and weight
projections remain separate.

Inputs are joined QKV `[1,T,3*H*128]`, BF16 projected forget-gate values `[1,T,H*128]`,
BF16 raw beta `[1,T,H]`, prepared BF16 convolution weights `[3*H*128,4,1]`, prepared
FP32 `exp(A_log)` `[H]`, FP32 `dt_bias` `[H*128]`, and an optional BF16 three-row
convolution history. It returns Q/K/V in BF16 `[1,T,H,128]`, FP32 decay of the same
shape, BF16 beta `[1,T,H]`, and an owned BF16 history `[1,3,3*H*128]`.

The kernel reads old history or current QKV directly instead of concatenating the
whole history and prompt first. It preserves the BF16 convolution, sigmoid and
SiLU stores, then the separate FP32 square/sum/rsqrt/query-scale operations. Decay
uses the existing prepared exponential and the same add/multiply/sigmoid/exp tree.
The supported lower bound is -5, matching the qualified unary range. Unsupported
shapes/dtypes/hardware return no candidate before dispatch.

`glm5_kda_fused.unaryModes` exposes the existing hardware-guarded, runtime-qualified
unary modes; no duplicate probes are introduced. `dispatchCount` and
`resetDispatchCount` track successful prework dispatches. All returned arrays belong
to `Result` and are released by `deinit`; callers must keep them alive while building
the recurrence graph or explicitly transfer/reference their handles.

## Exactness checks

The focused ReleaseFast suite compares every output bit against the existing MLX
operation chain for heads 1/3/64, rows 1/2/3/4/17/128/512, and both cold and nonzero
history: 42 geometry/state combinations. Fixtures include signed zeros and BF16
subnormals. The single-token tail uses integer bit copies, because a float conversion
can flush a subnormal that the reference view preserves. Multi-token tails reproduce
the existing owned-copy addition of zero. Lengths one and two retain the appropriate
old rows; lengths three and above retain only current QKV rows.

The reference V output is a strided view, so tests materialize contiguous copies
before comparing logical tensor contents. Invalid pointers, shapes, dtypes, head
counts and lower bounds are tested to decline without incrementing dispatch counts.
These checks cover raw prework, not a changed recurrence or a full-model quality result.

## Component measurement

2026-10-03, M5 Max, ReleaseFast, foreground `taskpolicy -a`, exclusive
`glm-kda-prework` GPU lock, ten-second idle cooldown, three warm A/B pairs and 24
alternating AB/BA timed pairs. The benchmark uses 512 rows, 64 heads, a nonzero old
history, and evaluates all six outputs. Time includes graph construction and evaluation;
weight projections, recurrence and output normalization are excluded.

| Path | Median |
|---|---:|
| Staged reference | 1,413.230 µs |
| Fused prework | 364.792 µs |

Latency fell 74.19% in this component test; the candidate won all 24 pairs and the
median paired saving was 1,061.958 µs. Multiplying the component difference by 34 KDA
layers suggests roughly 35 ms per prompt chunk before overlap effects, not a measured
full-model speedup. Integration must still pass layer/state parity and an actual model
run. Raw samples, binary/source hashes and fan-controller status are retained privately.
Fan-max was requested and automatic mode restored after the run; controller telemetry
is retained rather than assuming that a successful request proves achieved RPM.

The arithmetic adapts the qualified one-token KDA body and oMLX's parallel prework
mapping. Apache-2.0 and inherited MLX MIT provenance are recorded in `NOTICE`.

## Layer integration

Eligible multi-token calls in `KdaLayer.applyReference` now use this prework before
the unchanged recurrence. The forget-gate and beta projection results are computed
once and reused by either the fused path or the original staging fallback. Prepared
convolution/decay constants are reused. The candidate result remains alive through
recurrence construction; cache handles receive references to its compact history and
the recurrence's state. The existing one-token fused body remains the first choice.

A test-only switch forces the staged fallback for comparison; it has no production
behavior. A nonzero whole-layer regression compares consecutive two- and three-token
chunks from cold and warm histories. Output, compact history and FP32 recurrent state
match bit-for-bit, with counters proving that only the candidate arm dispatched the
prework kernel. Raw prework tests and the broader GLM-filtered ReleaseFast suite passed.
The diagnostic resets and reports `kda_prework_dispatches` alongside the existing body
and post-work counts. Full-model performance and output validation remain separate.
