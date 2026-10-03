# DFlash optional cached KDA leaf

`SUSHI_GLM_KDA_KEEP_LEAF=1` emits one computed FP32 recurrence state per KDA layer during
three-node-or-smaller verification. It selects the terminal node of the first ancestry path:
row 2 for a three-node chain, row 1 for two sibling drafts. Other accepted endpoints use the
original replay. The feature remains off by default while full-model hit rate and throughput
are unmeasured.

The original tree kernel's state updates and y arithmetic are unchanged. It already holds every
required ancestor state in registers; an added final store emits only the selected leaf. That
state is FP32 `[1,H,Dv,Dk]`, with no compression or conversion. `Tape.replay` validates its path
as before, aliases the retained state when the accepted endpoint matches, and constructs the
same convolution tail. A miss calls the original primitive recurrence. Small hit/miss counters
record the selected commit route. See [`src/glm5_dflash_kda.zig`](../src/glm5_dflash_kda.zig).

## Component qualification

Tests cover chain/sibling leaf selection, raw verification-y equality, raw FP32 committed-state
equality against original replay, pointer sharing on a cache hit, and replay fallback on a miss.
The focused component artifact passed all four tests. The production-size fixture uses three
nodes, 64 heads, 128-wide keys/values, BF16 operands and FP32 decay/initial state.

| Commit endpoint | Baseline verify+replay | Retained verify+commit | Difference |
|---|---:|---:|---:|
| Cached chain leaf | 0.356 ms | 0.285 ms | 70.75 us saved, 24.8% |
| Root fallback | 0.306 ms | 0.325 ms | 18.38 us overhead |

Each endpoint has two warmup pairs and eight interleaved pairs with alternating arm order.
Timing includes tree verification, commit-state/convolution-tail construction and evaluation;
inputs are materialized beforehand. Post-clock tape/result teardown is excluded. This is one
layer without a model load, not a full-model decode measurement.

The medians imply a component break-even cached-endpoint hit rate of about 20.6%:
`18.38 / (70.75 + 18.38)`. A hit saves about 2.4 ms extrapolated across 34 KDA layers, while a miss
adds about 0.625 ms. Those are estimates; actual full-model counters and timing decide runtime use.

One retained state is 4194304 bytes at production geometry. Thirty-four layers add 142606336 bytes
(136 MiB), rather than storing all three nodes. Existing request reservation charges four recurrent
buffers: old persistent state + one retained leaf + a fallback replay state use at most three
FP32 matrix buffers, with existing convolution allowances. The budget framework is unchanged.

The run used source matching this commit, ReleaseFast, foreground `taskpolicy -a`, exclusive GPU
lock `glm-kda-leaf-v61`, max fans, and ten seconds idle below 90 C on 2026-10-03. Measurement key:
`glm53-kda-leaf-20261003`; raw paired arrays, fan status and binary/source stamps are in the private
measurement ledger. Normal convolution/state format and teacher precision remain unchanged.

## Rejected canonical three-row recurrence

One isolated comparison at `127bb4e7` plus the candidate replaced dynamic
`saved[3][4]` parent indexing for the only two valid W3 trees: chain `[-1,0,1]`
and fork `[-1,0,0]`. The chain carried four FP32 state values per lane across
rows. The fork kept four additional root values, restored them before row2, and
wrote the retained row1 state before computing the sibling. The original decay,
scalar accumulation/FMA order, `simd_sum` and BF16 output boundaries were reused
from the current shader. Grid and threadgroup geometry remained unchanged.

Qualification reused the existing leaf probe's deterministic synthetic
B1/W3/H64/Dk128/Dv128 fixture: BF16 Q/K/V and beta, FP32 decay and nonzero
FP32 initial state. This was not a captured full-model activation. For each
topology, all 24,576 BF16 Y values matched with retention on and off, and all
1,048,576 retained FP32 state values matched the general tree. Replay hit and
miss states matched independent sequential paths, including fork miss `[0,2]`.
A retained-state pointer check confirmed the hit reused that same allocation.
Unsupported parents declined. The test first failed on the missing candidate,
then both focused tests passed after implementation.

| Topology | General / canonical median µs | Median arm difference | Paired median | Wins |
|---|---:|---:|---:|---:|
| Chain | 198.437 / 193.546 | 2.46% faster | 0.96% faster | 6/11 |
| Fork | 163.880 / 168.895 | 3.06% slower | 1.94% slower | 4/11 |

Both arms drifted substantially during sampling. Three warmup pairs preceded
eleven alternating AB/BA pairs per topology. Each sample averaged eight fresh
recurrence/retained-leaf/tape/replay-hit/conv-tail graphs with endpoint evaluation
and all frees included. Replay misses were correctness checks, not additional
timed cells. ReleaseFast, MLX 0.32.3, foreground QoS, exclusive GPU lock, maximum
fans requested and ten-second idle were recorded; the lock was released and
fans restored automatic. No full model was loaded.

Neither topology establishes a repeatable inclusive gain. No production hook,
chain-only variant or full-model repetition was warranted. The isolated helper,
probe/root, source exposure, binary, samples and provenance were archived under
measurement key `glm53-canonical-w3-kda-20261003`, then the prototype and source
exposure were removed. Dynamic private indexing was a plausible source lead,
but this comparison does not establish costly spilling in the existing kernel.
