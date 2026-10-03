# Singleton gate/up resource component

Rejected and archived candidate for the [singleton round](glm5-singleton-endpoint-round-plan.md).
It changes only qualified T3 gate/up resource separation. There is no runtime
delegation, down change, weight copy, routing prepass or speed claim.

`glm5_singleton_experts.moe` preserves the current lane input preparation,
serial-member F16 reductions, lane middle, grouped lane down and weighted BF16
finish. Two gate/up bodies are extracted from the original generated source:
one contains only singleton accumulators, the other only paired accumulators.
Both use original ballots/slots. The singleton dispatch writes a 24-slot bool
class mask before followers return; output tile 0/projection 0/thread 0 is its
single writer per slot. Families own separate output arrays, with no aliasing.

The middle clone selects gate/up pointer families uniformly per original slot,
then retains original arithmetic. Unwritten rows of the unused family are
never read. This avoids extra merge operations or changed signed-zero stores.
The additional two F16[24,2048] planes plus bool mask are 196632 bytes per call.
Fixed three configurations are cached; no unbounded geometry/weight cache.
Eligibility is B1/T3/H4096/I2048/E288/top8/n36/MCG/W12 with original BF16/F16
storage. Other shapes decline to the current caller-controlled path.

`glm5_singleton_experts_probe.proveStages` compares class masks, every written
gate/up F16 value and middle F16 value against current kernels. A separate
unused-family NaN poison check verifies pointer choice precedes loads.
`proveControlRoutes` covers all-single, all-pair and odd-triple membership on
one original full bank and saved input. The private harness owns actual 42-layer
BF16 output proof and the one inclusive paired timing, using original E288
addresses/order/settlement and all frees. No compact-bank or isolated-dot timing.

At `82aa9ea8` plus WIP, the private absent-API compile-red failed on missing
`moe` (exit1); the ReleaseFast guard source compiled green (exit0). Source/binary/log stamps
are under private evidence key `glm53-singleton-gateup-20261003`.

The first full-bank job stopped at singleton Metal JIT before numerical proof
or timing. Text extraction matched the cooperative body's inner N==64 rate
branch `else` instead of the outer singleton/pair boundary, leaving an open
brace. A packaging-only repair matches the outer brace depth and keeps both
original complete bodies. No arithmetic, mask, layout or configuration changed.
The failed binary/source/logs are preserved by the private harness; corrected
source was then used for one full-bank completion.

The [current full-bank comparison](glm5-singleton-current-replay.md) passed
every actual 42-layer mask/gate/up/middle/unused-family NaN poison check and
all 516096 saved routed BF16 outputs. Three warmups and eleven inclusive pairs
measured 17.816458 ms current versus 18.102708 ms split: 1.61% slower, zero paired
wins. Additional dispatch/resource separation did not establish a benefit;
this result neither proves spilling nor isolates an occupancy cause.

No model ABBA or further variant followed. Corrected helper/probe, red/stage
roots and source manifest were archived and removed after the private replay
owner confirmed all dependencies preserved. Root restored the readonly source
exports separately. Runtime remains `e1597cc2`; no runtime commit landed.
