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

## Parent-indexed DFlash prework

`applyTree(stream, Inputs, parents)` reuses the qualified shader arithmetic with
checked ancestor-window indices instead of linear token positions. The root reads
from the pre-tree three-row history; descendants read their own ancestors' raw QKV.
Parent indices must precede each child, and only the first node may have parent -1.
The entrypoint accepts 1–16 nodes, retains the same checked offset bounds, and
returns only Q/K/V, FP32 vector decay and BF16 beta. It does not apply the multirow
prefill history-copy operation or allocate a tail/state plane per branch.

`glm5_dflash_kda.applyLayer` retains `[old history; raw QKV rows]` in its tape. Replay
selects the accepted path's last three raw BF16 rows, preserving subnormals and
signed zeros exactly as single-token history. Recurrence remains the existing
vector-gate FP32 parent-state kernel. Eligible multirow results use the already
qualified fused output normalization/gate kernel; one-row output processing retains
the staged fallback. Unsupported prework geometry also retains the staged adapter.

Focused tests compare all five prework outputs bitwise with independent serial
ancestor windows for chain, binary and star trees, rows 1/3/16, heads 1/3/64,
and cold/hot history with subnormal/signed-zero inputs. Layer tests compare staged
and fused results, independent serial branches, one-row fallback and accepted-path
replay. A detached-tape test destroys the producing Ops scope before replay;
a separate history test checks raw BF16 bytes for short and deep accepted paths.
The native full-checkpoint diagnostic reports `tree_kda_prework_dispatches` and
`tree_kda_post_dispatches`. These checks establish component correctness; full
checkpoint throughput/parity results are recorded separately after measurement.
