# Bounded current routed capture

Uncommitted diagnostic for the [current-route round](glm5-m3-join-current-routes-round-plan.md).
No arithmetic, selector, kernel, default or throughput claim changes.
Helper qualification passed; the one loaded-model capture/replay is pending.

`glm5_dflash_routed_capture` is scoped by `bind(?*Capture)`. Default null returns
from the FFN tap before validation, allocation or array evaluation. The single
tap is after routed output construction and before shared addition, in the
multirow MoE branch; T1 returns before it. Prefill, serial oracle and baseline
replay remain unbound in the private harness.

Storage is fixed: three T3 rounds of42 ordered layer3–44 records, with actual
model-round IDs, original24 expert IDs and actual group2 engagement. The first
round additionally saves post-norm BF16 X, int32-compatible original IDs,
FP32 scores and routed BF16 Y for those42 layers. Payloads are CPU-owned raw
bits; no weights, prefix clone or GPU graph/cache references persist.

`Capture.init/deinit/reset/beginRound(u32)/completedRounds/complete` manage
the bounded lifecycle. `records()` returns a const flat slice of initialized
records; `saved()` returns a const42-entry payload slice only after the first
round completes. A round must consume all42 layers in order before the next
strictly increasing model-round ID. Missing context, duplicate/out-of-range
IDs, wrong shapes/dtypes, layer disorder and overflow fail without a partial
record append. The helper performs no serialization or replay orchestration.

Enabled capture explicitly settles ordered route IDs, and the first round
also settles/copies X/scores/Y. Therefore the entire capture is marked
synchronization-perturbed and cannot supply normal throughput or a per-child
latency sum. Private replay uses full original E288 bank addresses and keeps
the original serial/grouped/lane arithmetic and settlement cadence.

Focused tests check disabled lazy/malformed inputs, fixed capacity/layer order,
actual true/false engagement fields, exact BF16/signed-zero FP32 payloads,
invalid/duplicate ID refusal, complete first-round visibility and reset.
Source/proof preparation is separate from kernel timing or model acceptance.

## Helper qualification

At `fe37527c` plus the isolated helper/tap, the private absent-API compile-red
failed on missing scoped `bind`; the green focused root passed all three tests.
Fixed storage was2085720 CPU bytes,126 records and42 saved layers. All3024
ordered IDs,1032192 BF16 X/Y values and1008 FP32 score values matched raw
input bits, including negative zero. Actual true/false group2 engagement was
preserved. Layer/round order, incomplete rounds, three-round capacity,
invalid/duplicate IDs, visibility and reset passed. The disabled malformed
tap left a lazy array unavailable, confirming no evaluation.

Measurement key `glm53-current-routed-capture-helper-20261003`,2026-10-03:
ReleaseFast, foreground `taskpolicy -a`, exclusive per-job lock, confirmed
5359/5763 RPM at40.46°C and ten seconds idle. GPU released and fans returned
automatic; temporary roots were archived and removed. Private source/binary
stamps, compile-red/green logs and complete result are retained. This used
synthetic payloads with the real fixed schema, without a model or timing claim.


The [actual model measurement](glm5-current-routed-replay.md) passed token/state
and full-bank replay parity. After saving current fixtures, the diagnostic
helper/probe/tap were archived and removed. No runtime feature or default landed.
