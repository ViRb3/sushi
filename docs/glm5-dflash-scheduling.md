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

### Same-binary BF16/A6 ABBA repeat

Because the pilot gain was small, the same `606f5a57` binary repeated
BF16–A6–A6–BF16 with a separate lock and cooldown for each arm. Settings and
prompt were unchanged. All four runs again passed the identical64-ID and complete
state checks, with23 rounds and41 accepted drafts.

| Ordered arm | Decode tokens/s | Matched serial tokens/s | Draft time |
|---|---:|---:|---:|
| BF16-1 | 35.0867 | 27.3321 | 159.91 ms |
| A6-1 | 35.5783 | 26.5219 | 143.63 ms |
| A6-2 | 36.0893 | 27.6050 | 140.98 ms |
| BF16-2 | 35.8335 | 27.9618 | 157.36 ms |

Mean decode was35.4601 for BF16 and35.8338 for A6: a modest1.05% difference,
with overlapping run ranges. Mean draft time fell10.3% (158.64→142.30ms), and
commit time fell23.59→15.54ms; verifier means were similar1579.49 versus1583.97ms.
A6 consistently saved approximately1.329GB of peak memory. This supports the
memory saving and reduced assistant work more strongly than a substantial overall
speed claim. A8/A6 speed remains effectively tied in the available measurements.
Acceptance on other prompts still needs qualification.

Private artifact `glm53-dflash-assistant-quant-abba-20261003` retains all four
results and an aggregate comparison. Binary SHA-256:
`62401b3a9d09b54d18be993f65df29ed4b6fb57f0e85734e00d24916efc427f0`.
Starting temperatures ranged45.4–56.0°C; the same fan/cooldown protocol was used.
No public cache or assistant default changed.


## Diagnostic prefill controls

The gated diagnostic now exposes `SUSHI_GLM_DFLASH_DENSE_PREFILL` and
`SUSHI_GLM_DFLASH_PREFILL_ASYNC`, both off by default, plus
`SUSHI_GLM_DFLASH_PREFILL_SYNC_LAYERS` (1–8, default2). They configure the request
before warmup and captured prefill; reset and cloned serial reference preserve
these settings. JSON records all three. These are independent of verifier
`SUSHI_GLM_DFLASH_ASYNC_LAYERS`; no public cache or scheduler default changes.
A larger sync interval is experimental, not presumed faster. Compare complete
captured-prefix state and subsequent serial/spec decisions for each configuration.

Private sweep scripts prepare A6 N2, N4 and N3/children1 arms against the existing
N3/children4 lane+async4 baseline. They retain the original chunk128, staged
captured-prefix settings for that acceptance sweep. Test dense/async prefill in
a separate arm so prefix rounding changes cannot masquerade as an acceptance
policy improvement. The scripts do not run automatically.


## Selected assistant policy

A6g128 is the selected assistant for subsequent GLM work. The controlled comparison
showed a clear 1.329 GB decode-peak saving and a small 1.05% mean throughput increase
with overlapping run ranges. BF16 remains a recorded historical baseline, not a
planned deployment or further benchmark arm. Target MLA cache stays BF16 compressed,
and KDA recurrent state stays FP32.

The additional `Sushi-2.3bpw` target changes the trunk to A6g128 while retaining
K2.25/W12 experts. Its indexed payload is 95,985,384,312 bytes; the text-only
2,302-tensor payload is 93,295,638,776 bytes. The earlier `Sushi-2.4bpw` results used
an A8g128 trunk and 95,471,433,976 text bytes. These target packs require separate
serial references; assistant acceptance and generated text can change with trunk
quantization. Runtime memory and throughput must be measured separately.


## Stored A6 target trunk support

The directory named2.3bpw contains K2.25/W12 experts with an A6g128 trunk; the
previous2.4bpw directory used the same expert rate with A8g128 trunk. Directory
names alone must not determine kernel bit width. Native projection/embedding
support is a separate prerequisite from the assistant's stored quantization.

`glm5_dflash_qmm` now infers6 or8 bits from the actual packed row width and input
width, checks the original128-group grids, and includes bit width in its config
cache key. The6-bit kernel retains MLX's grouped activation-sum order, local
power-of-two scaling and six split-byte accumulation terms per four values.
Simply unpacking codes and dotting them would change rounding. The8-bit body is
unchanged; incompatible geometry still takes the serial projection fallback.

Fresh ReleaseFast tests with `SUSHI_GLM_DFLASH_HEAD_FIXTURE=1` passed raw BF16
parity for both bits at rows1/2/3/4/5/8/16 and output/input geometries32/256,
1536/4096,8192/4096,4096/8192 and154880/4096. They compare each candidate row to
native single-row qmv and assert integrated row-tile engagement. Malformed grids,
dtypes and layouts still decline. These unit tests qualify projection arithmetic;
full-checkpoint target token and state parity remains required before reporting
new target throughput.

### New A6-trunk target and exact down-projection composition

After native6-bit trunk, copy-free QKV and verification-row projection tests passed,
the2.3bpw directory was measured with the selected A6 assistant. Its experts remain
K2.25/W12; its trunk is A6g128. This target can legitimately change decisions versus
the old A8-trunk target, so correctness compares to its own serial reference.

Both runs used one fixed binary, lane-pair enabled, async4, N3/children4,
prefix512/committed64/chunk128, captured staged prefill, profiling off and one warmup.

| Down kernel | Decode tokens/s | Matched serial | Verify total | Decode peak bytes |
|---|---:|---:|---:|---:|
| Original | 37.3434 | 29.3201 | 1522.62 ms | 94,983,104,800 |
| Lane candidate | 39.1121 | 31.3934 | 1446.18 ms | 94,982,990,368 |

Each passed64 target IDs and complete committed state against its own serial run.
The down-on/off IDs were identical, with22 rounds and42 accepted drafts in both.
Lane pair/chain counts were924 (22 rounds×42 routed layers); down count was0/924,
and affine row dispatches7194. The lower count than the earlier966 reflects fewer
rounds, not missing kernel engagement. Output was coherent English.

Down-on increased measured speculative throughput4.74% and reduced total verify
time5.02%; draft cost was unchanged at approximately134ms. Serial throughput also
increased7.07%, so the speculative/serial ratio fell slightly1.274→1.246. These
are single same-binary arms, not a confidence interval. Peak memory was effectively
unchanged. The45tokens/s goal remains open.

Captured prefill was363.3/366.6tokens/s with the intentionally unchanged chunk128
staged prefix settings. It must not be compared as if it used the separately
optimized native dense-prefix prefill configuration. Private artifact
`glm53-target23-a6-20261003` preserves binary/source hashes, pack config provenance,
raw counters, phase times, outputs, full-state checks and thermal records. Both
runs had independent GPU locks, foreground QoS and cooldown, restoring auto fans.

## MLA branch scheduling finding

The measured512-prefix N3 target runs both reported zero MLA branch flushes.
Their maximum declared branch scratch was16,075,936 bytes against the268,435,456
byte cap. `mlaTree` only performs an intermediate synchronous flush when its
planned branch batch is full before the final branch; all four rows fit here.
`attention.attend` likewise skips its bounded-chunk evaluation for each singleton
branch query. There is therefore no branch wait to remove on this workload.

No asynchronous branch-flush policy was added. At longer prefixes a future
pipelined policy must reserve space for both in-flight batches; replacing a wait
with async evaluation at the existing full-cap batch size would invalidate the
scratch bound. At64K cache-growth boundaries, the current conservative plan allows
only one branch, so even two-way overlap may not fit. Acceptance-policy experiments
are better grounded for the current512-token workload.

## New-target node and branching sweep

Using the same fixed binary and new A6-trunk target/A6 assistant with lane/down
kernels enabled and async4, a zero-code sweep retained prefix512/committed64,
chunk128, greedy sampling and one warmup. Every arm matched the same64 target
IDs and complete committed state. Matched serial rates stayed31.24–31.27tokens/s.

| Policy | Verify rows/round | Rounds | Accepted drafts | Total verified rows | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| N3, children4 baseline | 4 | 22 | 42 | 88 | 39.1121 |
| N3, children1 chain | 4 | 22 | 42 | 88 | 38.2765 |
| N2, children4 | 3 | 24 | 40 | 72 | **42.4344** |
| N4, children4 | 5 | 20 | 44 | 100 | 31.4162 |

The chain produced no acceptance gain. N2 traded slightly lower tokens/round for
cheaper verification: total verifier time1295.72ms versus baseline1446.18ms, despite
two extra rounds; draft150.44ms versus134.21ms. Its total throughput improved8.49%,
with1.3571× matched serial and a94,979,400,224-byte decode peak. N4's additional
acceptance did not justify1856.78ms of verification. The five-row verifier geometry
also crosses the row-tile size4 boundary; this is a plausible contributor, not an
isolated kernel attribution from these end-to-end measurements.

N2/children4 is the recommended next composed benchmark baseline. The45tokens/s
target remains approximately6.05% above this result. This fixed-prompt sweep does
not establish an adaptive policy across workloads. Private artifact
`glm53-target23-policy-20261003` contains each independently locked/cooled run,
raw phase times, counters, provenance and an aggregate comparison. No model,
assistant or public scheduler default was changed by the sweep scripts.
