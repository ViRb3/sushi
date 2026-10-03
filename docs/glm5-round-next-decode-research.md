# Next GLM decode research: canonical KDA trees and joint QKV dispatch

Source-only audit at `f9f4c8d2`, 2026-10-03. No implementation, build, GPU job
or additional model load was performed. The selected precision remains BF16
compressed MLA cache, FP32 KDA state/accumulators and original retained small
BF16/FP32 weights. Both recommendations below seek exact outputs. Neither
requires precision restoration or an additional persistent weight bank.

## Cost and engagement evidence

The last completed HTTP ledger is measurement key
`glm53-commit-window-llmprobe-20261003`. The coordinator's current 2K–16K ladder
is still running; these inherited cells must not be presented as its result.
At predictable 32K, 191 timed forwards in 64 rounds measured 39.880 tok/s:
draft/verify/replay/commit were 6.680/64.680/2.277/0.971 ms per round. Verification
consumed about 86.4% of decode time. At unchanged tokens per round, 60 tok/s
requires approximately 49.74 ms total per round, about 25 ms less than this run.
The two candidates below are incremental, not a forecast that closes that gap.

The same 32K request recorded 6528 QKV-hoist calls, exactly `64 * 34 * 3`,
and 2176 KDA leaf hits, exactly `64 * 34`, with zero leaf misses. Its 704 query
and 704 value MLA broadcast calls equal `64 * 11` each. The 2K and 16K requests
had the same counts; 4K/8K used 65 rounds, 6630 QKV-hoist calls and 68 leaf
misses. Thus three-row KDA and the selected leaf path actually engage across
the measured context rungs. Leaf hits do not identify chain versus fork: both
topologies have a cached endpoint.

The six-round synchronization-perturbed verifier profile at `c8b03ba2`
recorded KDA recurrence 59.851 ms over 204 calls, QKV 79.968 ms over 204 calls,
and output projection 48.343 ms over 204 calls. Every recorded round verified
three rows, while emitted counts ranged 1–3. This establishes the executed
geometry and motivates inspection. It does not give GPU-only cost: the profile
forces waits, and the QKV sample predates the selected exact hoist. Do not
subtract those markers from unprofiled HTTP time or treat them as a removable
budget. The existing leaf component's roughly 0.285 ms inclusive verify/commit
measurement is another component reference, not an isolated recurrence time.

## Candidate 1: specialize only the two canonical three-row KDA topologies

The current [tree recurrence](../src/glm5_dflash_kda.zig) declares
`float saved[W][N]`, reads `saved[parent][i]` through a runtime parent buffer,
and separately keeps `state[N]`. At production W3/Dk128, N is four. Runtime
indexing of a thread-private array can require register selection or scratch
storage; a spill has not been established by generated-code inspection.

All valid three-row trees have either chain parents `[-1,0,1]` or fork parents
`[-1,0,0]`. Introduce only these two fixed sources behind a narrow opt-in gate:

- Chain: load four FP32 state elements per lane once and update that vector
  through rows 0, 1 and 2. No saved-row array or runtime parent read is needed.
- Fork: retain the four root-state elements, compute each child in a separate
  four-element working vector, and reset from the root for the second child.
  If keeping the selected leaf, write row 1's FP32 state when row 1 completes;
  this avoids retaining its vector while computing row 2.

Preserve the existing element loop order, separate decay multiply, K/state
dot, `simd_sum`, delta, state update, Q/state dot, output `simd_sum` and BF16
store. Keep the existing `(32,4,1)` threadgroup and grid, FP32 leaf format,
convolution tail and replay fallback. Do not port the changed-order oMLX dot
or repeat the rejected R8 prefill schedule. Generic W1/W2/other geometry and
unsupported modes retain the current kernel.

The initial gate can cover only the current leaf-enabled BF16 W3/H64/Dk128/
Dv128 path, with BF16 beta and FP32 decay/state. That is already the selected
HTTP path. One input bundle is about 4.235 MiB, dominated by the existing
4 MiB FP32 initial state; output Y is 48 KiB and the retained leaf is 4 MiB.
No new global buffer or state reservation is needed. Possible lower private
register/scratch demand is the performance hypothesis, not a memory bill.

**Minimal decision test:** one production-geometry component, both fixed
topologies, nonzero FP32 state and nearly unit decay. Compare every BF16 Y bit
and every FP32 selected-leaf bit against current `recurrentLeaf`; cover root
commit fallback without adding a topology sweep. Use three warmup pairs and
eleven inclusive alternating pairs per topology, including host construction,
evaluation and frees. Stop if either relevant topology consistently regresses
or both are flat. No new forced full-model profile is essential before this.
A winner then uses the existing three-node target token/full-state oracle,
including rejected siblings and a short final budget. Count actual specialized
chain/fork calls and generic fallbacks in the model gate.

This may remove a per-layer execution penalty across 34 KDA layers and is the
higher-priority decode experiment. It has no established latency saving yet;
even a recurrence win cannot by itself be credited with the full 25 ms gap.

## Candidate 2: one exact R3 QKV dispatch with the joined destination

Current [KDA verification](../src/glm5_dflash_kda.zig) constructs three
`linearRows` projections and then concatenates their BF16 results. The selected
[A6 hoist](../src/glm5_dflash_a6_hoist.zig) applies independently to Q/K/V.
Use one projection-selection grid dimension to run that unchanged hoisted body
against the three original banks, writing directly to `[1,3,24576]` in the
existing Q-then-K-then-V order.

Strict geometry is BF16 `[1,3,4096]`, three U32 `[8192,768]` A6/group128 banks
and their original BF16 `[8192,32]` scale/bias grids. Each bank is about 25 MiB;
the original three-bank fixture totals about 75 MiB. This remains 3072
64-thread threadgroups per KDA layer, with identical input/weight reads and
per-output arithmetic. It does not share weights across distinct projections,
change the output GEMV, or retry the inconclusive output-hoist experiment.

Three projection commands plus concatenation become one command. Across
34 layers this removes 68 projection launches and 34 concatenations per full
W3 verifier round. The existing three 48 KiB intermediate planes disappear;
the final 144 KiB raw plane remains. This saves about 4.78 MiB of intermediate
outputs across the round, with no expanded/repacked weights and no increased
live bound. The expected gain is command/allocation/copy overhead; there is no
claim that the old 79.968 ms QKV marker becomes available to remove.

[One-row GLM QKV](../src/glm5_decode.zig) already selects three original banks
inside one dispatch. Reuse that pointer-selection pattern, while retaining the
R3 hoisted arithmetic, rather than broadening its one-row dot implementation.
The old serial fused-QKV model result did not show material speed gain, so a
new R3 result must decide adoption independently.

**Minimal decision test:** actual L0 Q/K/V stored banks, one fixed nonzero
normalized R3 BF16 input, current three hoisted calls plus concat as control.
Compare all 73728 BF16 joined-output values. Time fresh full QKV-stage graphs,
including concat/allocation/eval/free, with three warmups and eleven alternating
pairs. A single joint-stage dispatch counter must prove engagement. Unsupported
bank, row count or storage takes the existing path before graph construction.
Only a consistent inclusive win proceeds to the same three-node model oracle;
there is no standalone output-projection variant or broad geometry sweep.

## Existing infrastructure and scope limits

Qwen's [fused GDN](../src/gdn_decode.zig) keeps a fixed causal token block's
state in registers, which supports the canonical-state scheduling idea. Its
head/scalar decay, convolution and arithmetic differ from GLM's per-key vector
decay, so it is not a drop-in recurrence. MiMo's grouped experts already show
that logical reuse can lose through occupancy; no further group-three, partner
prepass or padded NAX expert work is justified by the current evidence.

The [round-cost table](../src/round_cost.zig) records complete round time and
tokens by context bucket and rejects contended observations. It is a useful
future integration contract, but current GLM HTTP fixes N2/children4/async4.
An adaptive-width controller is not a kernel fix or a measured route to 60:
the earlier N3/N4 sweeps lost useful throughput, and another width sweep needs
new acceptance/cost evidence after verifier changes.

Long-context verification already uses immutable latent overlays and exact
native MLA projection broadcasts. Rejected per-node head-packed NAX attention
lost all nine tested node/context cases, and exact shared-factor merge also
lost. The packed-prefill cadence winner is not a decode candidate. Removing
another branch flush is unsupported where the measured flush count is zero.
The inherited 2K-to-32K verification increase is approximately 9.14 ms/round;
eliminating that growth alone would still leave the inherited 60 tok/s gap.

The diagnostic HTTP loop owns one model and handles each accepted connection
to completion, reporting `max_parallel=1`. Qwen/MiMo scheduler concurrency and
multi-request batching cannot be switched on through a GLM flag: independent
KDA initial states, tree roots, MLA prefixes, tapes and capture ownership would
need an explicit multi-request contract. Keep current admission and the sole
MLX inference owner during this wave. Report per-request latency separately
from any future aggregate concurrency throughput and never train round costs
from contended runs.

## Next three implementers after research

1. Canonical R3 KDA recurrence: isolated helper/probe, then coordinator-owned
   delegation in `glm5_dflash_kda.zig` after an exact paired win.
2. Joint R3 A6 QKV: isolated helper/probe and a tiny hoist-source seam; the
   coordinator owns its separate delegation in the same KDA caller.
3. Researcher 1's bounded NAX pool-dot tile cadence: independent index-scoring
   helper/control and existing actual 16K attention fixture, on top of the
   current packed cadence. It requires its own declared scratch delta.

Workers 1/2 share no implementation file until coordinator integration.
Schedule their small component timings, retain the latest HTTP baseline, then
run one combined correctness/real-performance gate and push only accepted
changes. Continue research if 1500/60 and stable 32K remain unmet.
