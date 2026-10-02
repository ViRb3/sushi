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
versus depth. The adapter currently hardcodes the tree defaults: children4,
tau1.5, edge weight0.6, temperature1.0. Expose an explicit diagnostic parameter
before this arm; do not silently alter defaults. A chain may improve depth at
fixed rows but loses sibling coverage. Record accepted drafts, rounds, verified
rows, phase times and matched serial throughput for every arm. Repeat the best
arm on a second prompt and a 2048-token prefix before selecting a policy.

Keep the assistant's trained block size8 and selector top-k16 fixed in the first
sweep. Shrinking the noncausal block changes assistant inputs/outputs rather than
just target scheduling. Increasing block width without retraining is not a free
acceptance improvement. Terminal budget clipping can save a final verification
round's excess rows but cannot explain a large sustained gain.

## MTP feasibility

The current GLM target index contains no MTP tensors. Its diagnostic loader also
explicitly excludes MTP names; the GLM forward has no MTP module or retained MTP
state contract. Existing MTP implementations for other model families are not a
valid GLM assistant substitute. Without compatible trained weights, acceptance
and latency measurements, there is no basis to predict MTP will beat DFlash2.
The current five-layer BF16 DFlash2 assistant is callable and has measured
acceptance; verifier scheduling is the lower-risk next experiment.
