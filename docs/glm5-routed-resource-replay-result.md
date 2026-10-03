# Unchanged routed resource replay

The corrected bounded diagnostic completed the original full-E288 42-chain
replay with exact BF16 outputs. This measures diagnostic overhead and resources;
it is not a kernel candidate or model performance acceptance.

One target load reused saved first-round T3 X/IDs/scores/Y from layers 3–44,
without an assistant, recapture or compact banks. Original bank order,
serial/grouped/lane preparation, gate/up, middle, down and finish stayed fixed.
Absolute async4 submitted eleven groups followed by the original final endpoint
settlement. No per-layer waits or command-buffer policy changes were added.
The same resident fixture/banks were held throughout; outputs were freed after
each pass. Settled CPU comparisons introduced no additional GPU evaluations.

| Pass | Chain including frees (ms) | Wall including CPU oracle (ms) | Exact BF16 values |
| --- | ---: | ---: | ---: |
| Settled warmup | 46.381083 | 46.381250 | Not compared |
| Unchanged reference | 17.893167 | 18.066500 | 516096 |
| Instrumented | 17.900875 | 18.030417 | 516096 |

The single diagnostic chain was 0.0431% longer. Begin/end, bounded completed-
callback handling and metadata output increased its complete wall to 18.349958
ms, 1.5690% above the reference's oracle-inclusive wall. These are single-pass
observations with different CPU oracle times, not a paired latency estimate.
Both proved passes had active 93,538,400,888 bytes and peak 93,546,101,368 bytes;
the memory/wired limit remained 115,448,725,504 bytes.

The collector retained 210 dispatch records and 96 completed command buffers
without overflow. Resource interpretation and command-buffer timestamp limits
are recorded separately in the [collector result](glm5-routed-resource-result.md).
The first-round distribution remains 1008 assignments, 748 singleton leaders
and 130 pair leaders; it is distinct from the saved three-round aggregate.

The first loaded attempt stopped at collector finalization because its 127-byte
name capacity truncated an actual 139-byte gate/up name. It retained 210 records,
96 buffers and 30,931 JSON bytes, below the global caps. Complete sample clocks
were not persisted in that attempt. The authorized packaging repair increased
only the name field to 160 bytes; recorder state became 43,744 bytes and output
30,944 bytes, within unchanged 512-record/128KiB limits. The harness now persists
completed clocks/proof status before handling collector errors. The earlier
zero-test build-filter error is also retained; neither failure is numeric evidence.

Artifacts `glm53-routed-resource-replay-20261003` (failed attempt and corrected
`attempt02-name160`) and `glm53-routed-resource-20261003` retain sources,
compiler commands, ABI/export checks, full metadata, hashes and telemetry.
The corrected binary SHA256 is
`0e2a4dffe5ea8b05e2e54127e804cb4553792f6cd08ca89f6e7fc09f95bc8051`;
private libmlx SHA256 is
`93dd43f31b697984a7bab7c62a7dfe5e960d610974e268c57d934b67604cf32d`.
ABI1 added only the three agreed diagnostic exports. Installed libraries,
original shader bodies and NAX metallib stayed unchanged. The accepted CLI was
not rebuilt or modified. Explicit private-library launch paths, foreground QoS,
maximum fans, required idle and a per-job GPU lock were used. The corrected
job passed one test and exited 0; process stopped, lock released and fans auto.
No model request, throughput comparison, profiler recursion or further variant
was run.
