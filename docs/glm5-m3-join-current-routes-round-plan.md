# M3 join and current routed measurement round

Start from accepted runtime `e1597cc2`, documentation checkpoint `de9c698e`.
[Joint QKV research](glm5-joint-m3-qkv-research.md) recommends one bounded
exact dispatch change. [Routed research](glm5-routed-decode-next-research.md)
requires current capture/replay before another expert kernel; old forced-wait
markers and compact-bank fixtures are not current latency budgets.

| Worker | Scope | Owned source |
|---|---|---|
| 1 | Exact joint M3 QKV | New joint helper/probe using original current hoist body |
| 2 | Current routed capture | Tiny default-off bounded capture helper/probe; one FFN diagnostic tap |
| 3 | Whole routed replay | Private replay/model harness using full original E288 banks |

Worker1 uses a bank grid and separate original buffers, preserving existing
R3 accumulators, coefficient math, K256 order and BF16 stores. Emit contiguous
[1,3,24576] directly. No bank repack, output-projection hook, M1/M4 extension
or extra retained state. Original L0 fixture supports full raw-QKV and
complete KDA chain/fork leaf hit/miss/replay-state proofs. One3-warm/11-pair
inclusive full-layer/commit gate, including frees, advances only a clear win.

Workers2/3 agree a small capture/replay interface. Capture three complete T3
rounds across42 routed layers with ordered IDs and engagement; retain post-norm
BF16 X, scores and routed BF16 output for one round, about2MiB without weights.
Disabled tap returns before validation/allocation/evaluation. No persistent
bank/state or general capture framework. Root delegates only the FFN diagnostic
tap to worker2; worker3/root own private orchestration, not public runtime.

Replay those42 whole chains with full resident original E288 banks, exact saved
inputs/IDs/scores, original bank order and current four-layer settlement cadence.
Include preparation, gate/up, middle, down, weighted finish, evaluation and
cleanup. Prove all replay BF16 outputs. A few warm baseline samples establish
cost/variance; there is no A/B kernel variant or per-child forced timing.
Analyze singleton/pair multiplicities and slot distances from current IDs.
Do not implement a split dispatch or another group3/word-sharing/window variant
without this evidence.

Root schedules one fixed8192-ID/192-output current-stack model job (A6 assistant,
N2/children4, nativeB1/B3, accepted flags, async4; prefill2048/async2). Capture
runs outside throughput evidence. If joint component wins, piggyback capture
after its matched same-prefix N2 ABBA; otherwise run only the bounded current
capture/replay job. No simultaneous full-model loads. Strict serial output and
complete valid-state proof remains required. Equal references and all rounds/
cleanup are included in any speed comparison. Capture synchronization is
explicit and cannot be reported as normal verifier latency.

Source stays WIP until model acceptance. Commit/push only accepted runtime;
failed helpers/taps are archived and removed. CPU/GPU jobs receive sequential
grants, foreground QoS, thermal protocol and per-job GPU locks. No variant sweep.
RoutineHTTP2K–16K; selected32K once for a final winner. Prefill1500/decode60
remain open, and no component/capture outcome is a throughput promise.
