# Unchanged routed replay: resource diagnostic

The corrected single target-only replay passed, exit 0: both reference and
instrumented passes matched all 516096 saved BF16 values. The collector returned
0 with full names, no overflow, 210 dispatch records and 96 completed command
buffers. This establishes bounded resource/correlation evidence for the unchanged
42-chain component, not a verifier share, bandwidth result or runtime speedup.

The original round0 operands and full E288 banks retained layer3–44 order,
absolute async4 boundaries, final settlement and frees. Five named consumer
families each have 42 records; no unknown built-in dispatch was recorded. These
resources come from the actual pipelines and dispatch entry points:

| Named family | Grid | Thread group | Static shared bytes |
| --- | --- | --- | ---: |
| `sushi_glm_lane_unsorted` | 1024×24×1 | 32×1×1 | 0 |
| `sushi_glm_exl3_pair_group2_lane_mcg_w12` | 16384×24×2 | 128×1×1 | 4096 |
| `sushi_glm_down_lane_prepare` | 512×24×1 | 32×1×1 | 0 |
| `sushi_glm_exl3_group2_lane_mcg_w12` | 32768×24×1 | 128×1×1 | 4096 |
| `sushi_exl3_down_reduce` | 8192×3×1 | 256×1×1 | 4096 |

Full generated identities, including templates and input/output types, are
preserved privately. All five report execution width 32 and maximum threads per
group 1024; the custom path configures zero dynamic shared bytes. These limits
and allocations do not establish registers, spilling, occupancy or resident
thread-group counts.

Every recorded command buffer has positive, individually ordered GPU start/end
properties. Actual record-to-buffer mapping gives the following phase sets:

| Dispatch phase set within a buffer | Buffers | Sum of GPU buffer intervals |
| --- | ---: | ---: |
| Prepare + gate/up | 12 | Not split |
| Middle + down | 12 | Not split |
| Gate/up only | 30 | 14.153415 ms |
| Down only | 30 | 7.403376 ms |
| Finish only | 12 | 2.727167 ms |

The three pure-phase sums already exceed the 17.900875 ms diagnostic chain clock.
Buffer intervals overlap and are **nonadditive**. A pure phase set identifies
which recorded dispatch family occupies that buffer; its interval may also
contain unrecorded blit/event/dependency work. These are scoped command-buffer
observations, not kernel times. Mixed intervals cannot be divided between
families; no actual-verifier or latency-share attribution follows.

Reference chain plus frees took 17.893167 ms and diagnostic 17.900875 ms, a
single-pass difference of +0.0431%. Walls including the CPU oracle were
18.066500/18.030417 ms; diagnostic begin/end-inclusive wall was 18.349958 ms.
These are diagnostic clocks, not paired acceptance evidence or a repeatable
overhead estimate. Both passes started at 93538400888 active bytes and peaked
at 93546101368 bytes, an equal 7700480-byte increment.

The public counter inventory contains only `timestamp/GPUTimestamp`.
Dispatch-boundary sampling is unsupported; no dispatch timestamp samples or
traffic/ALU counters were collected. Command-buffer timestamps come from
completion properties. There is no measured DRAM/cache/decode-ALU cost or
explanation for earlier singleton/fusion losses. The result therefore supports
resource facts and named buffer correlation, without recommending another shader.

The first attempt failed because its fixed 128-byte name field truncated the
139-byte gate/up identity. That attempt, including the lost final-clock
persistence, remains archived. One authorized packaging repair changed only
that field to 160 bytes; a harness-only change persisted samples before checking
collector status. The corrected recorder stores 43744 bytes of metadata and
30944 bytes of JSON within the unchanged 512-record and 128 KiB caps. No shader,
sampling, command policy or per-layer wait changed.

The disposable Release C++ build used pinned MLX64ea011cb65f, deployment 26.2 and
the byte-identical accepted NAX metallib. All original exports remain; exactly
three diagnostic C APIs were added. Installed libmlx/libmlxc/libjaccl/metallib
hashes stayed unchanged before build, after the first job and after completion.
The copied libmlxc/libjaccl/metallib also remained exact. Runtime source is
unchanged; no installed library or source submodule was modified.

Evidence keys `glm53-routed-resource-20261003` and
`glm53-routed-resource-replay-20261003/attempt02-name160` retain private source,
build/API/library/hash checks, full identities, dispatch/buffer data and clocks.
The original failed attempt is retained separately. All measurement processes
ended, GPU lock was released and fans restored to automatic. No further
collector variant, sampling run or model benchmark followed.
