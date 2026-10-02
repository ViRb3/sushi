# GLM DFlash scheduling and acceptance experiments

The target is at least 45 committed speculative tokens/s while keeping the default
BF16 compressed MLA cache. Configuration changes and scheduling need separate
measurements from kernel changes.

## Current evidence

The unprofiled `daa6d524` N3 sample emitted 64 tokens in 23 rounds: 2.783 tokens
per round, 41 accepted drafts, and 92 verified rows. It measured 27.665 tokens/s
against a matched 26.266 serial reference. Draft, verify, replay and commit totals
were 156.76, 2087.18, 44.23 and 24.23 ms. At unchanged acceptance, 45 tokens/s
requires about 61.84 ms per round rather than 100.54 ms, a 38.5% reduction.
Removing all assistant time alone would not reach that target.

The older same-revision width comparison at `f0093a9e` gives:

| Draft nodes | Verify rows | Rounds | Tokens/round | Tokens/s |
|---|---:|---:|---:|---:|
| 1 | 2 | 37 | 1.730 | 23.279 |
| 3 | 4 | 23 | 2.783 | 27.025 |
| 7 | 8 | 20 | 3.200 | 19.667 |

N7 increased average verifier latency from approximately 93.3 to 152.4 ms for
only 15% more tokens per round. This is evidence against simply increasing width,
not a universal rejection of N7 after verifier kernels improve.

## Asynchronous verifier candidate

`glm5_dflash_model.bindSchedule` scopes a host-thread-local cadence: 0 retains
synchronous per-layer evaluation; 2 and 4 enqueue groups of that many layers.
`SUSHI_GLM_DFLASH_ASYNC_LAYERS` exposes it only in the gated diagnostic, with
baseline 0 as the default until model qualification. Component profiling forces
synchronous evaluation. JSON records effective cadence and async/sync dispatches.

Each dispatch explicitly includes the group's final hidden array, every KDA
replay input and convolution history, each MLA tape's latent/key/gate arrays, and
captured assistant features in that group. MLX arrays retain their lazy dependencies
after each local Ops scope is freed. Tape/capture handles remain owned by Verified;
no request state is published during verification. Group enqueue bounds the lazy
construction interval to two/four layers. The final synchronous boundary evaluates
decisions plus every tape/capture array, including an incomplete final group.
Head dependencies alone are insufficient because they need not consume every
replay input. Existing MLA branch scratch flushing remains unchanged.

This can remove host waits and overlap construction with GPU work, but does not
reduce arithmetic or expert bytes. The existing tape allocations remain resident;
peak memory and complete state parity must be checked in real-model runs. No
speed forecast follows from serial async4's earlier improvement.

## Tight configuration sweep after scheduling qualification

Use the same binary, BF16 assistant, stored target quantization, BF16 MLA cache,
fixed prompt IDs, greedy sampling, warmup, 64 committed outputs and profiling off.
First compare N3 synchronous, async2 and async4; require token and full-state
parity plus reported dispatch engagement. Choose the best measured schedule.
Then run N2 and N4 against N3 with that schedule. The expected mechanism is a
better accepted-token/verification-cost tradeoff; gains are not assumed.

One additional N3 chain arm (`TreeParams.children=1`) can isolate branching
versus depth. The gated `SUSHI_GLM_DFLASH_CHILDREN` parameter now passes an explicit adapter
argument (range1–16; default4). Existing API wrappers retain children4. Other
tree defaults remain tau1.5, edge weight0.6, temperature1.0. A chain may improve depth at
fixed rows but loses sibling coverage. Record accepted drafts, rounds, verified
rows, phase times and matched serial throughput for every arm. Repeat the best
arm on a second prompt and a 2048-token prefix before selecting a policy.

Keep the assistant's trained block size8 and selector top-k16 fixed in the first
sweep. Shrinking the noncausal block changes assistant inputs/outputs rather than
just target scheduling. Increasing block width without retraining is not a free
acceptance improvement. Terminal budget clipping can save a final verification
round's excess rows but cannot explain a large sustained gain.

## MTP feasibility

The target does contain trained MTP weights: 54 tensors under
`model.language_model.layers.45.*`, including `eh_proj`, `enorm`, `hnorm`,
`shared_head.norm`, MLA and packed MoE. Searching only for an `mtp` namespace
misses them. The current diagnostic excludes layers outside the 45-layer target,
and the GLM forward has no implemented MTP module or retained-state contract.

A single MTP layer could cost less than the five-layer BF16 DFlash2 assistant,
but acceptance and target verification cost determine useful throughput. MTP
needs a loader/forward mapping and independent state/oracle tests before a fair
comparison. Existing MTP implementations for other families are not a validated
GLM substitute. DFlash2 already has measured acceptance and a callable verifier,
so its scheduling experiment remains the immediate path; MTP is a plausible
subsequent implementation experiment, not excluded by missing trained weights.
The local oMLX GLM runtime maps it as concat(enorm(next embedding), hnorm(raw
hidden)) through eh_proj, then a plain residual MLA/MoE block without HC,
shared_head.norm, and the shared vocabulary head. Raw hidden is the four-HC-stream
mean before final norm; a future implementation must retain that exact tap.

## First async full-checkpoint qualification

A fixed ReleaseFast binary at `50795e0c` ran separate locked async4 and async2
processes with the settings above: prefix512, committed64, N3, chunk128, BF16
assistant, default BF16 target cache, one warmup, component/route profiling off.
Both passed independent serial-token and complete committed-state parity. Their
64 output IDs were identical, and each used 23 rounds with 41 accepted drafts.

| Schedule | Committed tokens/s | Matched serial tokens/s | Ratio | Async / final sync calls | Decode peak bytes |
|---|---:|---:|---:|---:|---:|
| async4 | 32.001 | 25.556 | 1.2522 | 253 / 23 | 98,534,128,928 |
| async2 | 31.684 | 25.724 | 1.2317 | 506 / 23 | 98,534,129,440 |

Total draft/verify/replay/commit times were 159.73/1766.71/44.33/28.05 ms for
async4 and 161.07/1786.85/43.33/27.64 ms for async2. The one-percent difference
between schedules is provisional. Both improve on the older synchronous sample,
but a same-binary synchronous control is needed to isolate the scheduling effect
from unrelated intervening changes or run variation. The 45 tokens/s goal is open;
async4 remains opt-in pending policy selection.

Focused tests compare nonzero four-layer verification decisions, every KDA/MLA
tape array, captured assistant features, and all committed caches for budgets
1/3/5. A three-layer view covers incomplete async2 groups and an async4 run with
no intermediate dispatch; final settlement still produces identical state.
Component profiling overrides both schedules to the synchronous path. The
scoped binding rejects unsupported cadences and restores its previous setting.

Each full run used interactive QoS, its own GPU lock, fan-max request and ten-second
idle, restoring automatic fans afterward. Private artifact
`glm53-dflash-async-20261003` retains binary hash, source/build state, prompt/model
config hashes, controller telemetry, raw phase metrics and output/state results.
The built-in full-checkpoint test was run directly to avoid build-cache reuse
when changing only environment settings.

The subsequent opt-in lane-pair plus async4 composition measured **34.938 tokens/s**
against matched serial **27.691** (1.2617×). It retained the same 64 IDs, complete
state parity, 23 rounds/41 accepted drafts, and 98,534,128,928-byte decode peak.
The pair and full-chain counters both recorded 966 calls; async/final-sync counts
were 253/23. Verify time was 1596.68 ms, draft160.90, replay44.91 and commit28.26.
Both speculative and serial rates increased relative to async-only; this single
run should not attribute every difference exclusively to the kernel. Private
artifact `glm53-dflash-lane-async-20261003` preserves the fixed binary and raw data.
It is the BF16-assistant baseline for the following stored A8/A6 comparison.


## Stored affine assistants

The GLM diagnostic accepts the original BF16 assistant or stored affine6/group128
and affine8/group128 assistants. `loadAssistantStored` calls the existing loader
with load-time quantization disabled: packed tensors stay packed, and retained
small BF16 matrices stay BF16. Shape/dtype validation covers encoder, every
attention/MLP projection, dynamic-convolution projections and selector projection.
The packed weight/scales/biases must have compatible affine geometry; mixed
packed widths are rejected. The strict `loadAssistantBf16` entry point remains
available for callers that require the original precision.

The report derives `assistant_precision` and `assistant_storage` from loaded
matrices, rather than the directory name. The target keeps BF16 compressed MLA
cache and FP32 KDA state. Assistant quantization can change proposals and
acceptance; each arm must still match the target's greedy serial tokens and
complete committed state. Compare draft time, acceptance, total decode and peak
memory under the same verifier settings before choosing a default.

## Stored assistant precision comparison

The A8g128 and A6g128 assistants were then measured with the same lane+async4,
N3/children4, prefix512/committed64/chunk128 settings and fixed prompt. The stored
loader inferred 46 affine matrices, one retained dense matrix, and group size128
for each; their inferred bits were8 and6 respectively. Both used the same binary
at `606f5a57`; the BF16 comparison preceded only the loader/report extension.

| Assistant | Decode tokens/s | Matched serial | Draft total | Accepted / rounds | Decode peak bytes |
|---|---:|---:|---:|---:|---:|
| BF16 | 34.938 | 27.691 | 160.90 ms | 41 / 23 | 98,534,128,928 |
| A8g128 | 35.541 | 27.765 | 139.88 ms | 41 / 23 | 97,477,704,992 |
| A6g128 | 35.716 | 27.857 | 141.88 ms | 41 / 23 | 97,205,026,080 |

Every arm produced the same 64 target IDs and passed complete committed-state
parity with its own serial reference. Lane pair/chain engagement remained966
calls. A8 saved1.056GB of peak memory and A6 saved1.329GB relative to BF16.
A8 reduced draft time13.1%; A6 reduced it11.8%. Total gains were1.73% and2.23%,
respectively. The0.49% A6-versus-A8 throughput difference is too small to call a
reliable speed advantage from single runs; its additional272.7MB memory saving
is clear. This prompt showed no acceptance loss, not a guarantee across prompts.
The45tokens/s target remains open.

Private artifact `glm53-dflash-assistant-quant-20261003` contains separate locked
runs, inferred storage, counters, raw phase times, output/state checks, provenance
and binary hash. Both runs used the same foreground QoS and thermal protocol as
the BF16 arm, with profiling and route capture off.
