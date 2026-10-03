# Next GLM decode research: canonical KDA trees and shared-prefix scoring

Source-only audit at `f9f4c8d2`, 2026-10-03. No implementation, build, GPU job
or additional model load was performed. The selected precision remains BF16
compressed MLA cache, FP32 KDA state/accumulators and original retained small
BF16/FP32 weights. Both recommendations below seek exact outputs. Neither
requires precision restoration or an additional persistent weight bank.

## Cost and engagement evidence

The inherited HTTP ledger is measurement key
`glm53-commit-window-llmprobe-20261003`. The accepted `f9f4c8d2` wave subsequently
measured 2K/4K/8K/16K prefill 931.71/802.82/727.81/655.06 tok/s and decode
47.08/45.48/44.44/43.04 tok/s. At 16K it recorded 64 rounds and
6.334/60.336/1.503/0.986 ms draft/verify/replay/commit per round. This wave
changed prefill, not decode; do not attribute across-boot decode differences
to those kernels. The final 32K HTTP cell is still running.
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

## Candidate 2: exact three-query scoring over the shared committed pool prefix

Current [IndexPool selection](../src/glm5_attention.zig) computes scores separately
for each MLA tree branch. Its scalar SCORE kernel loops 32 heads and four
128-wide key elements per lane, rounds each dot to BF16, clamps it, rounds the
weighted term to BF16, accumulates heads in order, then rounds the total.
The [prefill NAX scorer](../src/glm5_indexpool_nax.zig) explicitly declines
queries of eight rows or fewer, so current W3 verification stays on this scalar
path. The short 512-prefix MLA profile does not measure this sparse indexer.

All three branches share the committed `P0 = source.processed / 4` completed
pools. With at most three ancestry tokens, each branch adds zero or one pool.
Use one fixed three-query SCORE shader over that prefix, retaining three
independent FP32 dot/total chains while loading each shared key coefficient
once. Preserve `#pragma clang fp contract(off)`, lane traversal, `simd_sum`,
all BF16 rounding boundaries, head order and negative weights. This is an exact
scalar scheduling candidate, not another EXL3 three-member decoder.

Strict input geometry is BF16 queries `[3,32,128]`, weights `[3,32]` and
contiguous BF16 prefix keys with width128. Completed pool count P is a runtime
scalar; never JIT a shader per prefix length. One prefix SIMD group replaces
three, while all per-query arithmetic remains. Near 32K the scalar baseline
uses approximately `3 * 8192` groups; the candidate uses approximately8192,
plus a suffix group. Actual 32K plus192 outputs can exceed8192 pools, so that
prefill-only cap must not disable the late32K decode path.

The suffix group reads each branch's own completed pool key, if present, and
uses its actual ancestry offset. Do not replace fork offsets with `offset+row`.
One output plane may be `[3,P0+1]`, but each branch must slice to its original
`Pbranch` before its original negative/argpartition/top512/expand sequence.
Keep all three original argpartition calls and their exact dimensions: padding
the partition itself can change tie ordering even when padded scores are
negative infinity. The final2051 selected IDs and their order are part of the
bit-equivalence contract, not just the unordered pool set.

The shared FP32 score plane is approximately96KiB at8192 pools, matching the
aggregate of the original three score planes, plus at most three unused suffix
slots. There is no pooled-history copy. Small suffix keys/flags and packed
queries require an explicit bounded bill if the implementation materializes
extra arrays; reuse original query/weight planes and borrowed prefix views.
Keep the existing branch scratch limit and fallback when three branch states
cannot be live under it. This can address a context-growing verification stage;
the inherited approximately9.14ms 2K-to32K verification increase is only a
loose source-prioritization lead, not an indexer-only measured budget.

**Minimal decision test:** reuse the actual16K capture's index queries, weights
and pooled keys, label constructed fork/suffix fixtures accurately, and compare
all score bits and ordered selected IDs against three unchanged selectors.
Cover negative weights, ties near the512 cutoff, odd history, chain/fork actual
offsets, zero/one suffix and a pool frontier above8192. Then time the whole
three-selector graph, including score construction, three original partitions,
expansion, endpoint evaluation and free: three warmup pairs and eleven fresh
inclusive alternating pairs. Stop a flat/losing arm; no NAX variation follows.
An exact winner proceeds to coordinator-owned `mlaTree` integration and the
existing target token/full-state oracle.

**Rejected-probe scope:** the archived decode-NAX probe's `timed()` loop created
and evaluated a fresh Ops scope for EACH node. Its helper admitted only
`[1,64,512]`, with SDPA Q `[1,1,64,512]` and KV `[1,1,2051,512]`. Therefore its
T1/T2/T3 measurements were one/two/three serial node calls, not one B3 gather or
SDPA. True B3 is distinct but remains parked: native attention would still have
only about six tensor groups, and any changed-order mode needs serial-target
consistency plus drift/selection validation. The exact scorer avoids that gate.

**Parked smaller option:** a joint R3 A6 QKV dispatch could emit the existing
`[1,3,24576]` plane directly, replacing three hoisted calls plus concat with one.
This would remove68 launches and34 concats per round and approximately4.78MiB
of intermediate outputs, without reducing issued dot work. Existing
[one-row GLM QKV](../src/glm5_decode.zig) demonstrates pointer selection, but
its older model result did not show a material gain. Prefer the shared-prefix
scorer's context-growing opportunity this round; do not implement both.

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
2. Exact shared-prefix SCORE: isolated `glm5_indexpool_shared_prefix` helper/probe
   and a tiny SCORE-source export. The coordinator owns the later `mlaTree`
   delegation, only after an inclusive three-selector win.
3. Researcher 1's bounded NAX pool-dot tile cadence: independent index-scoring
   helper/control and existing actual 16K attention fixture, on top of the
   current packed cadence. It requires its own declared scratch delta.

N3 policy is also parked: HTTP fixes two draft nodes, bounded readout handles
only N2, and the exact QKV hoist and MLA broadcast guards require three rows.
Group2 already admits four rows, but a wider policy falls back in those other
paths. The old N3/children1 chain did not improve acceptance and N4 lost; changing
N3 now requires a complete round-cost/acceptance comparison, not row-count
arithmetic. It is not a third implementation recommendation.

Workers 1/2 share no implementation file until coordinator integration.
Schedule their small component timings, retain the latest HTTP baseline, then
run one combined correctness/real-performance gate and push only accepted
changes. Continue research if 1500/60 and stable 32K remain unmet.
