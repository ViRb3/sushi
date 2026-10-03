# Current verifier raw-QKV counterfactual

Source-only after `bd627696`, accepted runtime `4fcb541e`. Recommend **one fixed
actual-verifier attribution of all KDA raw-QKV construction**, not another
kernel. No prototype, build, capture or GPU job accompanies this report.

## Decision and exclusions

The qualified predictable decode range is about 51→48 tok/s and ordinary 46→42
across 2K–32K. Mini2 removed 26% of complete proposal work but failed both actual
model criteria. Replay is also a small ceiling; a new small delta/endpoint tape
would not address the dominant verifier and is not recommended.

Current `kda.applyLayer` constructs qraw/kraw/vraw through `linearRows`, then
concatenates BF16 `[1,3,24576]` before convolution/prework. There are 34 KDA layers.
The accepted R3 A6 hoist already shares masked coefficients while preserving
serial row arithmetic. The rejected QKV join changed command/concat construction,
not dot work, and its complete-layer gain was noisy 0.57%, 6/11. It does not measure
how much current verifier work is removable by a substantially different
projection implementation.

Other repeated paths remain excluded: general group3 sharing was mixed;
singleton gate/up lost 1.61%, grouped middle/down fusion lost 11.52%; all-endpoint
retention was noisy and failed a release guard. Canonical recurrence, short-row
NAX/shared scoring and cache/layout variants did not qualify. The expert replay's
17.8 ms and overlapping resource CB intervals cannot provide an additive verifier
or DRAM budget. No output-hoist, expert microvariant or retained-bank retry follows.

The question is: **Does removing this entire large projection class materially
reduce current complete verifier work, or would even a perfect replacement have
little useful ceiling?** A small delta closes this research direction. A large,
repeatable delta justifies a later separately researched projection strategy;
it does not select a kernel, imply a production output cache or promise 60 tok/s.

## One held-output experiment

Use one current actual N2/T3 proposal/tree and immutable request prefix from the
frozen nominal 8K ordinary workload, with HC_ON/A6/native/async4 and all accepted
flags. Record its exact tokens/parents, prefix offset and source/weight/flag
fingerprints. Capture only the 34 raw-QKV concatenated outputs once, outside clocks:
34×147456 = **5013504 BF16 bytes**, under an 8 MiB fixed payload bound including
metadata. No weights, recurrent states, prefix caches or general capture API.
Default-disabled diagnostic handling returns before validation/allocation/eval.

Then hold those same outputs and exact reference results equally before all
warm/timed arms. Control computes original Q/K/V and concat normally. The
counterfactual supplies the held per-layer raw output instead. Convolution,
prework, recurrence/retained leaf, GA/GB, post, output projection, HC/FFN/MLA,
head and commit remain current. No per-layer wait or profiling mode is added.

This boundary avoids a pruning trap: x still feeds FA/beta/GA and normal branches,
and raw still feeds convolution and accepted convolution-tail publication. Do
not replace final KDA output, which could prune gate/post work. Validate that
current non-QKV engagement and every replay/capture dependency remain evaluated;
retain the original final tape/capture settlement and async4 cadence. A command
count/queue effect is part of this whole counterfactual, not a kernel duration.

## Fixed proof and paired scope

One isolated diagnostic worker owns the bounded holder/probe; root alone owns
the tiny KDA interception and private current-model harness. Capture and replay
use the same epoch/tree/prefix only; mismatched order, metadata, shape/dtype or
incomplete 34-layer coverage rejects reuse. Other rows/configs never engage.
Do not reuse records across modified parents or another request.

Prove all held raw bits against control, full three-row logits/decisions and
complete valid committed MLA/KDA/convolution state. Keep accepted-path and budget
behavior unchanged. The holder remains resident in both arms through cleanup;
charge its 5.013 MB, reference states and all existing admission bills. Record
capture/preparation/retention cost separately, with no reserve credit.

Three warmups and eleven fresh alternating pairs measure **complete verification
plus normal prepareCommit and all endpoint/clone/tape frees**, with equal held
references. Clone timing is retained explicitly; CPU bit oracles stay outside the
interval. All clocks, engagement, peak and individual pairs persist before a
proof/adequacy error. No class sweep, shape variant or second prefix follows.

This is deliberately perfect reuse of captured outputs for exactly one tree/prefix,
with no claim that real requests can cache them. It produces a scoped removable-
work ceiling. It is
not production throughput, a cache benefit, isolated QKV time or GPU-family share;
class ceilings must not be added. If noisy or small, stop without a QKV kernel.
If substantial, any eventual fixed implementation still needs inclusive proof/
pairs and current ordinary+predictable 192-output ABBA with exact serial IDs/
valid state, acceptance, peaks and gain beyond control drift before HTTP 2K–16K
and selected 32K. The fixed projection geometry itself has no context switch,
but total workload/acceptance does; do not extrapolate this 8K attribution to 32K.
