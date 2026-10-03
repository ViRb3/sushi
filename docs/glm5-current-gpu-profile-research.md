# One current decode system trace

Research only, accepted runtime `e1597cc2`, 2026-10-03. Recommend **one
three-second target-PID Metal System Trace**, triggered by the first real SSE
token of a warmed native8K/N2/A6 request. No model, profiler/capture run,
runtime source change or instrumentation framework was created for this study.

## Verified local capabilities

Installed `xctrace` lists Metal System Trace, Time Profiler, Metal Application,
GPU and Thread Activity. Read-only `--show-recording-options` explicitly starts
no recording. The shipped Metal System Trace template contains application,
driver/GPU/resource metadata and Time Profiler. Actual kernel labels/timestamp
availability still require an adequacy check on the recorded trace.

MLXC exposes `mlx_metal_start_capture/stop_capture`; MLX implements this with
MTLCaptureManager on the entire Metal device, using GPUTraceDocument for a
nonempty output path. There is no bounded metadata-only/tensor-size option in
that API. Do not invoke it, enable GPU frame capture or risk dumping94GB model
resources. System trace is the selected alternative, not that capture path.

`MLX_PROFILER_RANGE` is CUDA NVTX-only and expands to nothing on Metal.
Metal CommandEncoder already commits asynchronously with completion handlers,
but this checkout/MLXC/Zig interface does not expose GPU command-buffer timing
records or a kernel profiler CLI. The existing GLM profile forces evaluation;
it is unsuitable here. No library rebuild or new timestamp hooks are proposed.

## One actual-model measurement and exact recipe

Root owns one ReleaseFast `sushi glm-bench` process with current stored target,
A6g128 assistant, N2/children4, nativeB1/B3 ON, accepted flags, async4 verify,
prefill2048/async2, greedy192 outputs/ignoreEOS. Keep profile/route/union/capture
diagnostics OFF. One owner holds the GPU lock, foreground QoS, confirmed max
fans/thermal protocol. Load and warm before tracing; stamp binary/source,
MLX/MLXC, flags, actual memory/wiring limits and PID. Do not trace loading/JIT.
Root's clean ReleaseFast CLI at `0e82efad` has runtime source exactly
`e1597cc2`; preserve that binary identity for this job.

Reuse one frozen previously qualified8K HTTP body and its hash. HTTP accepts
chat messages, not raw token IDs: record `/tokenize` result and returned actual
prompt count; do not label it8192 unless verified. Exact8192-ID work instead
requires the private accepted-stack harness, not invented HTTP fields. This
proposal selects the existing HTTP path without modifying runtime.

The native bridge emits its first pending token after prefill/reservation and
before the decode timer resets. An external SSE observer ignores keepalives
and role-only/empty deltas, timestamps the first real token, then immediately
starts the following attach command. Attachment latency may miss early rounds;
this is a bounded steady-decode sample, not a complete-request attribution.

Recording options file (the two changed settings expose waits/context switches;
retain normal sampling and no kernel-stack sampling):

```json
{
  "Time Profiler": {
    "contextSwitchSampling": true,
    "recordWaitingThreads": true,
    "highFrequencySampling": false,
    "recordKernelStacks": false
  }
}
```

Set the variables to private artifact paths/that exact server PID; run once:

```sh
taskpolicy -a xcrun xctrace record \
  --template 'Metal System Trace' --attach "$PROFILE_SERVER_PID" \
  --time-limit 3s --window 3s --recording-options "$PROFILE_OPTIONS" \
  --output "$PROFILE_TRACE"
```

After the request completes, stop only the job's own server PID, restore fans
automatic and release the GPU lock before export/analysis:

```sh
xcrun xctrace export --input "$PROFILE_TRACE" --toc --output "$PROFILE_TOC"
xcrun xctrace export --input "$PROFILE_TRACE" \
  --xpath "$PROFILE_TABLE_XPATH" --output "$PROFILE_EVENTS"
```

Resolve each needed XPath from that trace's actual TOC, using the exact form
`/trace-toc/run[@number="1"]/data/table[@schema="ACTUAL_SCHEMA"]`; repeat
**export only** for target GPU intervals, command/encoder/submission events
and CPU thread samples/waits. Schema names vary; do not invent them or export
every resource/stack table. Bound exported rows to the recorded window and
summary output to kernel-family counts/duration distributions, GPU busy union,
idle gaps and CPU submission/wait patterns. Retain raw trace privately. Time
bound is not a byte guarantee: stop/report if recording storage exceeds1GiB;
do not escalate into full-resource capture. No second trace variant is implied.

## Adequacy, perturbation and next decisions

Record client UTC/monotonic pairs at request start, first real token and final
response, plus recorder start/end/PID. Use exported trace clock anchors to
convert to its timebase; never subtract unrelated clocks. Keep only target-PID
events inside confirmed decode, excluding attachment/tail teardown. Reject
ambiguous alignment, absent submission↔GPU correlation, missing/zero intervals
or fewer than ten recognizable complete verifier cycles. Main kernel families
must have usable function names/identifiers; anonymous intervals cannot support
expert/KDA attribution. Warm cached pipelines may lack creation/name events
during attach; do not guess their identities. Matching binary symbols matter
for CPU stacks.

GPU durations can overlap: report interval **union** for busy time and retain
per-family sums as nonadditive statistics. Submit-to-start delay can be queue
backlog, dependency or residency work, not a CPU bubble. A feed gap requires
idle target GPU with no ready queued work, correlated with CPU activity/waits.
There are no layer signposts here; do not pretend identical kernel names identify
individual layers. Existing17.8ms independent routed replay remains separate.

If GPU is continuously busy and one family dominates measured intervals,
next work targets that family's actual cost with an inclusive component gate.
If gaps correlate with graph construction/allocation while no work is queued,
next work targets host submission/lifetime, rather than another fused dot.
If committed work waits before GPU execution, inspect current fences/residency/
command-buffer lifecycle first; do not assume an old wiring result applies.
If intervals/labels are insufficient, report the missing evidence and stop.
Do not add per-layer waits, sweep templates or tune command-buffer thresholds.

Tracing/sampling can perturb scheduling, cache and elapsed time. Preserve all
request outputs/counters and compare IDs to the frozen body's accepted reference,
but trace throughput is not a speedup/control sample. Read-only capability
listing does not prove recording permission. On attach/privacy/Developer Tools
errors, preserve exact error/exit status and stop; request only the actual
missing permission through root. Do not hide prompts with `--no-prompt`, use
sudo, disable SIP or grant blanket access. No permission failure was observed
because no recording was attempted.
