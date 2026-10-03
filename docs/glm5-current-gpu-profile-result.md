# Current decode system trace: partial evidence

One target-PID Metal System Trace recorded successfully, exit 0, 50.45 MB and no
permission failure. **Formal kernel/family attribution failed**: shader-profiler
intervals were empty, encoder labels were generic, and ten named complete
verifier cycles could not be established. No second trace/template or library
instrumentation followed. Global correlated timing remains usable within its
measured late-decode coverage; it is not a kernel cost budget.

Accepted CLI hash begins `a3fe8286fbdb`, built at `0e82efad` with runtime source
exactly `e1597cc2`. One target+A6 assistant server used N2/children4, native B1/B3,
accepted flags, async4, prefill 2048/async2 and greedy 192/ignoreEOS. A frozen
HTTP body tokenized to **8156**. Warm/profiled requests produced the
same 192 output IDs; CLI hash was unchanged. No tensor/state dump or per-layer
forced evaluation was enabled. The collector shared the server owner's one
GPU job; it did not load a model, acquire another lock or stop the server.

Attach began at the first nonempty SSE token. Requested time limit/window were
3 s. Actual trace summary span was 3.815536 s; recorder exit included additional
postprocessing, so its wall lifetime was not used as the recording window.
The exported UTC/mach/timebase anchor aligned client monotonic/UTC pairs within
2.5µs. All 4788 positive target Compute intervals lay inside the actual SSE
decode window. The observed first/last target intervals cover 2.119166 s of late
decode, not the whole request or each verifier phase.

| Supported fact in that observed span | Value |
|---|---:|
| Target GPU active interval union | 2.021420 s / 95.39% |
| Gap union between target intervals | 97.746 ms |
| Median / maximum gap | 0.708 µs / 0.939 ms |
| Target GPU intervals matched to application command-buffer IDs | 4787/4788 |
| Command-buffer submissions | 7537 |
| Submissions with one / zero encoders | 4787 /2750 |
| Trace-reported CPU→GPU latency median / maximum | 4.360 ms / 6.089 ms |

GPU busy is an interval union; raw duration sums are nonadditive. Gaps are
**not proven CPU feed bubbles**: dependencies/queue/driver scheduling may
contribute. CPU→GPU latency is not idle time, and zero-encoder command buffers
may carry events/fences rather than be redundant. The trace does not justify
deleting those buffers or changing submission thresholds. It argues against
assuming large empty-GPU bubbles dominate this observed steady window.

Only actual TOC-selected GPU/submission/encoder bridge, CPU sample and clock
tables were exported, with the empty shader table retained as adequacy evidence.
CPU samples exist, but this report assigns no function-level CPU cause or
individual layer. No expert/KDA/MLA share, occupancy/spill attribution or60tok/s
forecast is supported. Current17.8ms independent routed replay remains separate.
Next work needs named kernel timing or a focused source audit of correlated
command/event lifetime, not another fusion based on these anonymous intervals.

Evidence key `glm53-current-systemtrace-20261003` retains the single record
command/options, frozen request/CLI hashes, SSE/clock/output reference,
TOC/table commands, raw trace, bounded global summary and failed formal adequacy.
Raw artifacts remain private because they include paths/environment metadata.
The copied-CLI dry startup initially failed relative dylib resolution before
any model/GPU load; the unchanged original CLI path was used afterward.
Foreground QoS and the server owner's confirmed maximum fans/thermal protocol
were used. Owned server ended exit0, fans returned automatic and lock released
before exports. All recorder/export/parser processes are terminal. Trace
elapsed throughput is perturbed diagnostic evidence, not a control or speedup.
