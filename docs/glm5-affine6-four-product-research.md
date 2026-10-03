# M3 affine6 canonical unpack research

Recommend one fixed **four-product unpack** for raw KDA Q/K/V, as an optional
numerical target with matching M1 serial/replay math. It changes inner dot work;
it is not the rejected launch-only join. Source inspection supports a bounded
experiment, but there is no measured speed or quality result.

The [actual attribution](glm5-qkv-attribution-result.md) held exact outputs for
one ordinary 8756-ID prefix T3 tree. Complete latency fell 48.868083→42.276333 ms,
6.591750 ms/13.4889%, with 11/11 wins and unchanged non-QKV counters. That is the
perfect-reuse class ceiling, not QKV kernel time or a 32K/HTTP speed prediction.
The [joint M3 dispatch](glm5-joint-m3-qkv-result.md) changed launches/concat only
and lost its inclusive gate; it left the arithmetic below unchanged.

## Concrete work change

`glm5_dflash_kda.linearRows` delegates affine rows to `glm5_dflash_qmm.project`,
then accepted `glm5_dflash_a6_hoist.project`. Raw projections are BF16
`[1,3,4096]` × original U32 A6/group128 `[8192,768]`, with BF16 scale/bias grids
`[8192,32]`. Each projection uses 1024 groups of 64 threads: two SIMD32 groups,
four outputs per SIMD, 16 K256 steps and three row accumulators per output.
Coefficients are already decoded once across M3; more launch joining does not
remove this work.

The hoist and pinned MLX `quantized.h` express four A6 values in three bytes
with six masked products. For bytes a,b,c the exact integer values are:

- q0 = a & 63;
- q1 = (a >> 6) | ((b & 15) << 2);
- q2 = (b >> 4) | ((c & 3) << 4);
- q3 = c >> 2.

Decode those four integers once per output/pack, retain them across all three
rows, and multiply the original unscaled BF16 inputs in q0,q1,q2,q3 order.
Keep the original BF16 quartet sum used for bias, K256 order, FP32 scale/bias
update, lane mapping, SIMD32 reduction and final BF16 store. Eight input values
then require eight products instead of twelve. Stored codes/grids, packed
loads, three separate projections, concat, output projection and all other KDA
operations remain unchanged. No bank expansion, weight retention, padding,
NAX dispatch or precision restoration is involved.

This is algebraically equivalent over real numbers, **not current bit parity**:
combining the two masked contributions changes FP32 addition grouping. An M1
version must use the identical arithmetic and geometry so serial/replay and M3
remain one numerical target. The user permits this opt-in numerical trade, subject to the frozen quality
gates below. The existing hoist already removes the obvious cross-row
coefficient duplication; this candidate changes the inner products themselves.

Finite checks local to each already-loaded input chunk/grid must preserve the
original masked-product body for nonfinite operands. Do not add a synchronized
whole-array finite scan or sanitize NaN/Inf. The check, unpack shifts and branch
costs belong in the complete gate and may erase the benefit.

## Ceiling, proof and ownership

One projection's source-level inner product steps fall 150,994,944→100,663,296;
102 current raw calls give about 5.13 billion fewer such steps per full T3 pass.
These are source arithmetic counts, not issued instructions, ALU utilization or
DRAM measurements. Packed logical bytes remain 26,214,400 per bank. Even an
ideal one-third reduction of a wholly dot-limited 6.59 ms class would suggest
only about 2.20 ms/4.5% of this verifier; that illustration is not a bound or
prediction because the counterfactual includes command/overlap effects. Zero
or negative improvement is plausible. It cannot alone establish 60 tok/s.

One worker owns a new isolated helper/probe; root owns narrow raw-only M1/M3
seams, counters, memory accounting and eventual qualification. Admit only GPU,
BF16, contiguous/materialized original A6/group128 K4096/N8192 and M1/M3.
Unsupported shapes remain ordinary math; a selected numerical mode must not
silently mix old M1 with new M3. No new retained buffer is needed. Preserve all
old reserves, source-head storage, FP32 recurrent state/accumulators, RAM
embeddings and BF16 MLA cache; record measured peak without storage credits.

Tests first: exact packed integer decode; M3 raw bits versus three candidate
M1 calls; complete chain/fork KDA outputs and every committed BF16 convolution/
FP32 state versus candidate serial ancestry. Directed cases include signed
zero, cancellation, ties, large finite values and nonfinite input/grid behavior.
Record old-hoist raw drift, changed bits, relative L2/max error and nonfinite
counts rather than claiming old parity. Use the existing full layer fixture;
no new target capture or compact bank is needed for this component.

Then one complete raw-QKV plus KDA/normal-commit component, current versus
candidate, equal resident inputs/references, three warmups and 11 alternating
pairs including graphs/evaluation/frees. Keep original other projections and
leaf behavior; stop mismatch, new nonfinite behavior, noise or loss. No unpack,
row tile or head-count variants follow.

A winner receives strict candidate serial/spec IDs and valid-state bytes plus
old-target forced-logit quality on frozen nonrepeated 16K code and prose prompts,
192 baseline-forced continuation positions each. Freeze before measurement:
per-input mean KL≤0.01 nats, max KL≤0.15, top1 agreement≥95%, mean forced NLL
increase≤0.02 nats and zero new nonfinite values. These are an experimental
same-pack numerical screen, not lossless-teacher checkpoint quality. Do not
relax them after results. Finally run current HC_ON/N2/A6/native matched ABBA 192
with equal references, complete clone/decode/cleanup clocks, acceptance and all
old bills. Require improvement beyond control drift on both frozen ordinary
and predictable inputs before the existing 2K–32K HTTP qualification. Prefix
length does not alter projection geometry; its full-model share and acceptance
can change, so no long-context gain is assumed.

Source-only at round closure `5929c9d6` / accepted runtime `4fcb541e`. No prototype,
build, GPU run, caller change or checkpoint rewrite accompanied this report.
