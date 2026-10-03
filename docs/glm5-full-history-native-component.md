# Exact-membership full-history native attention: rejected

The strict T2048/N16384 native GQA64 candidate preserved original selected-key
membership and passed safety/layout guards, but made inclusive whole attention
**37.36% slower**, with no wins in 11 paired rounds. Its helper and probe are
archived and removed. No production seam, default, precision restoration,
layout variant or actual-model follow-up was retained.

| Arm | Median whole-attention time |
| --- | ---: |
| Current two-tile gathered native attention | 131.378 ms |
| Exact membership, full-history native attention | 180.460 ms |

The paired median slowdown was 37.25%. Three warmups preceded 11 interleaved
pairs. Candidate timing includes all original T16 selection, dense membership
construction, a fresh valid-history finite scan and CPU predicate wait, native
SDPA, empty-row handling, endpoint evaluation and cleanup. Both references and
the captured fixture remained held equally throughout timed arms. These are
component timings, not prompt-wide throughput.

The producer proof retained all 4,200,448 ordered original IDs and all
33,554,432 membership bits, including 4,197,376 live unique causal keys and
128 original NAX selector calls. Invalid IDs used a separate dummy column;
they could not clear key zero. This candidate changed no retrieval policy.

Native input Q was a head/query-transposed view, KV an immutable valid-history
view shared by all 64 heads, and the Boolean mask one physical 32 MiB plane.
Post-evaluation pointer/stride checks confirmed view sharing, Q head stride 512,
Q sequence stride 32768, final stride 1, and mask head stride 0. No Metal body port,
full-history/head copy, F32 cache or head-expanded mask was used. The current
native D512 loop still processed 512 key tiles rather than 65 gathered tiles.

Chronological key order and per-head query tiling changed numerical rounding:
551,348 of 67,108,864 BF16 output values differed from the original ordered 2051
path. Relative L2 was 0.000252650 and maximum absolute error 0.0078125. Both outputs
were finite. This is exact retrieval, not a lossless arithmetic claim.

All three joint tests passed. Distinct heads/query rows, empty-mask positive
zero, strict geometry and valid-history-only scanning passed. A nonfinite
capacity row beyond 16384 did not trigger fallback. Fresh NaN/Inf scans of any
valid history triggered the original gathered path: unselected/future
nonfinite values matched its safe output exactly, while selected valid
nonfinite values remained observable. The timed fixture used native 14 times
and fallback zero times after counter reset.

Candidate peak growth was 79,831,061 bytes above the resident fixture and held
current output, under the conservative 512 MiB per-pending-layer allowance.
This scoped measurement includes the candidate result and does not establish
a lower runtime bill. The full-history mask backing had 33,556,480 bytes with
a dummy-column stride 16385; the head broadcast retained stride 0.

Evidence key `glm53-full-history-native-20261003`, source HEAD `07f020bf` with
isolated WIP, preserves source/runtime/binary hashes, build recipe, flags, raw
pairs, drift/layout/safety checks, memory and thermal/fan state. A reserved Zig
identifier was corrected before the green build. The GPU lock was released
and automatic fans restored. The dependent producer remains under its own worker's cleanup ownership;
runtime source remained unchanged.
