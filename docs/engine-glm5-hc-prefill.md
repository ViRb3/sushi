# Exact HC prefill projection experiment

`Hc.collapse` uses the exact C24 RMS-fused candidate for eligible prefill inputs.
`SUSHI_GLM_HC_PREFILL=0` restores the staged path for comparisons. Serial and
small tree calls remain unchanged. The caller checks the row count before the
stream/device and dtype eligibility probes, avoiding device-handle creation on
serial calls. The RMS-fused entry point declines fewer than
128 rows, non-BF16 activations, unsupported geometry, invalid epsilon, or CPU
streams. It accepts stored BF16 or FP32 HC weights without conversion copies.

## Arithmetic and work reduction

The original projection assigns one 128-thread group to each of 24 outputs.
The candidate shares each normalized FP32 activation across 2, 4, 8, or 24
independent output accumulators. Each output preserves its original lane-strided
FMA sequence, SIMD sum, and `(p0+p1)+(p2+p3)` final reduction. No matrix-multiply
algorithm or accumulator dtype changes.

The separate RMS-fused variant accepts raw `[1, rows, 4, 4096]` BF16 activations.
It reproduces the pinned MLX RMS reduction with 128 physical threads simulating
1024 logical threads. A volatile FP32 normalized product preserves the rounding
boundary before projection. C24 computes RMS once per row; C8 repeats it three
times. The fused version removes the widened and normalized FP32 planes. Sinkhorn,
collapse, checkpoint storage, and output types are unchanged.

Shape validation uses wide arithmetic for index-product bounds. Public counters
record candidate dispatches; diagnostics report `hc_prefill_dispatches`. Each call owns its output; kernel handles persist,
while per-call configuration and temporary scalar/vector handles are released.

## Validation

ReleaseFast synthetic dot tests cover widths 128, 512 and 16384; rows 17, 128 and
512; BF16/FP32 weights; all output tiles; cancellation, zero and varied magnitudes.
RMS tests compare staged native RMS plus original dot against C8/C24 at rows
128/512, BF16/FP32 weights, four magnitudes and two epsilons. All comparisons use
raw FP32 bytes, not tolerances. Integrated full HC tests additionally compare
BF16 mixed outputs, FP32 post weights and Sinkhorn matrices at 1/17/127/128/512
rows, with dispatch counts confirming fallback below 128 and engagement above.
A nonzero four-layer model test compares logits and every cache array with the
policy on/off across 128, 17 and one row; its hidden128 geometry intentionally
tests fallback rather than candidate engagement. Production-width actual captures
check engagement and all three HC outputs against captured native values.

Actual checkpoint qualification uses prose and code captures at layers 0, 3, 23
and 44, both attention and FFN HC. All 24 mixes match captured native outputs
exactly at 512 rows and the first 128 rows. Decode fixtures exercise the fallback.
The fixture test is opt-in via `SUSHI_GLM_HC_PREFILL_FIXTURE` and
`SUSHI_GLM_HC_PREFILL_REPORT`; `SUSHI_GLM_HC_PREFILL_RMS=1` measures RMS plus dot,
and `SUSHI_GLM_HC_PREFILL_ROWS128=1` slices prefill fixtures to 128 rows.

## Component measurements

M5 Max, 2026-10-03, ReleaseFast, foreground `taskpolicy -a`, exclusive GPU lock.
Each capture record used four warm pairs then 24 alternating AB/BA pairs, with
synchronous output evaluation. Values below average the eight per-record medians.

| Input | Staged RMS + dot | Fused C24 | Reduction |
|---|---:|---:|---:|
| Prose, 512 rows | 686.85 µs | 334.23 µs | 51.34% |
| Code, 512 rows | 703.83 µs | 343.28 µs | 51.23% |
| Prose, 128 rows | 337.72 µs | 231.66 µs | 31.41% |
| Code, 128 rows | 366.13 µs | 247.21 µs | 32.48% |

Every record improved. C8 saved 42–44% at 512 rows and 26–27% at 128 rows.
For normalized-input dot alone, C8 was best: 45.85% prose and 45.24% code at 512
rows; C24 saved about 44%. Those separate sweeps cannot isolate normalization
cost by subtraction. C24 is the proposed combined prefill candidate.

Fan-max was requested and a ten-second idle cooldown preceded each locked run;
controller readback did not verify achieved max RPM. Fans returned to automatic.
Raw paired arrays, controller status, hashes and fixture provenance are retained
in the private measurement ledger. These are component results; full-model speed remains a qualification gate.

## Rejected factored-RMS alternative

Moving inverse RMS after a BF16-by-BF16 NAX projection changes FP32 reduction and
rounding. Before testing, gates were fixed at absolute error
`1e-4 + 1e-5*abs(reference)` against both an FP64 oracle and the staged baseline,
plus backward error at most `2e-6`. Split-K 128 and 1024 both failed the absolute
or drift gates. Ordinary inputs passed, but cancellation produced staged drift
around 0.035–0.037. Split-K128 was much closer to FP64 than the staged baseline
in that example; improved oracle accuracy still fails the required native drift
gate. Split-K1024 also lost about 0.00195 of the oracle residual. A large-magnitude
case failed too. No tolerance was widened and no timing was performed. Its source
and report are archived privately, not exposed as a runtime path.
