# GLM-5.3-Flash next performance wave

Audit of `dfe17328` on 2026-10-03. Target: at least 1500 prefill tokens/s and
60 DFlash2 decode tokens/s, with stable approximately 32K operation. The selected
stack retains the original small BF16/FP32 tensors and resident embeddings,
BF16 compressed MLA caches, and FP32 KDA persistent state and accumulators.
Ordinary BF16 intermediates are permitted; precision restoration is excluded.
This is an implementation plan, not a throughput forecast.

## Evidence and current baseline

The authoritative HTTP baseline is `glm53-commit-window-llmprobe-20261003`:
29 completed requests, llmprobe 0.6.13 bench-only, runs 1, reasoning default,
2K–32K, 192 output IDs per measured cell. Source compiled as `2f446c5a` with
source-equivalent committed HEAD `dfe17328`; MLX 0.32.3/`64ea011c`, mlx-c
`56b2d39` with the global-scale compatibility patch. The ReleaseFast binary
hash begins `7688a81f7e56`. Foreground QoS, exclusive GPU lock, maximum fans and
cooldown were recorded. See [the HTTP protocol](glm5-benchmark-http.md).

The second predictable-context requests are the matching inherited cells below.
Decode uses 191 forwards after the first token from prefill, matching the HTTP
rate calculation; do not substitute the client aggregate or a different prompt.

| Rung | Input IDs | Prefill tok/s | Decode tok/s | Draft / verify / replay / commit ms per round |
|---|---:|---:|---:|---|
| 2K | 2036 | 947.64 | 46.85 | 6.16 / 55.54 / 0.87 / 0.95 |
| 4K | 4059 | 758.10 | 45.76 | 6.18 / 55.91 / 1.03 / 0.93 |
| 8K | 8225 | 665.03 | 44.20 | 6.27 / 57.98 / 1.15 / 0.93 |
| 16K | 16278 | 605.59 | 41.41 | 6.64 / 62.90 / 1.43 / 0.92 |
| 32K | 32747 | 541.33 | 39.88 | 6.68 / 64.68 / 2.28 / 0.97 |

At 32K, verification consumes 4.140 of 4.789 decode seconds, about 86.4%.
Assistant commit is now below 1 ms per round and drafting is almost independent
of context. The previous bounded-tail run measured 60.83 ms verification and
6.23 ms commit per round at 32K. The new ladder is slower globally despite the
bounded commit improvement; do not attribute all differences between separate
boots to code. At unchanged acceptance, 60 tok/s requires about 49.7 ms per
32K round, versus 74.8 ms now: roughly 25 ms must disappear, primarily from
verification. Reaching 1500 prefill tok/s at 32K requires approximately 64% less
prompt time. These incremental candidates alone do not establish those gains.

The current six-round verifier profile (`glm53-current-verify-profile-20261003`,
source `c8b03ba2`) points to routed FFN 145.49 ms, KDA QKV 79.97 ms, shared FFN
62.70 ms, KDA output 48.34 ms, and lowrank/beta 35.61 ms. It forces evaluations:
tiny norm/expand fields cost approximately 39–40 ms each. These are leads, not
GPU-only costs or removable time budgets. It predates the exact QKV hoist and
bounded assistant changes; those do not alter the routed or retained-BF16
projection arithmetic. Do not add inclusive `ffn_overall` to its children.

## Fresh cold-prefill attribution

One normal target-only component diagnostic ran from `dfe17328` plus an isolated
import root on the same runtime and selected flags: exactly 2048 captured input
IDs, chunk2048, two output IDs (one decode forward), one warmup, no assistant,
no route histogram. Foreground QoS, exclusive lock, maximum fans requested and
ten-second idle were used. It completed with exit zero, released the lock and
restored automatic fans. Measurement key:
`glm53-next-wave-prefill-components-20261003`; binary SHA-256 is `e9305946de4086d67f8c9a77aafe7b2653f6623d11f66ee1179b86e1b263a31d`.

| Disjoint prefill component | ms |
|---|---:|
| Routed FFN | 810.624 |
| Attention, all layers | 648.131 |
| Shared FFN | 91.919 |
| HC attention / FFN collapse | 46.868 / 46.729 |
| Dense FFN | 36.411 |
| Attention / FFN expansion | 24.939 / 24.110 |
| Router | 23.183 |
| FFN combine | 12.365 |
| Attention / FFN norm | 10.211 / 10.078 |

Total prompt time was 1788.124 ms with forced component evaluations. Routed FFN
is about 45.3%, attention 36.2%; the attention total includes 500.232 ms across
34 KDA layers and 147.899 ms across eleven cold dense-MLA layers. Components do
not split KDA projections from its recurrence; do not ascribe the KDA subtotal
to one kernel. Unmarked embedding/head/layer work accounts for the remainder.
The existing verifier profile is separately stamped at `c8b03ba2`, 2026-10-03
12:53 +0700. Its recorded flags omit the A6 hoist, so the old QKV marker must not
be used as the current hoisted projection's exact cost.

The fresh run confirmed 136 A6 dense-prefill calls, 34 retained clusters and
34 R4 recurrence calls. The cold prefix uses dense SDPA and records zero absorbed
head-batched MLA calls; it does not exercise sparse packed attention. Peak was
95,170,884,644 bytes, within the configured limits. The synchronous diagnostic's
1145.3 prefill tok/s and single-forward 8.74 decode tok/s are explicitly excluded
from HTTP throughput comparisons. This profile supports large routed/KDA costs
at 2K; workstream 2 is justified by source and long-prefix dispatch evidence,
not by pretending this cold-prefix sample measured sparse attention.

## Three independent workstreams

Run one isolated candidate per workstream. Stop a losing arm after its focused
whole-component comparison. Build and GPU timing require coordinator scheduling;
all full-model loads require exclusive GPU use. Implementations must be small,
with original fallbacks and no new generalized dispatch framework.

### 1. Routed verifier: reuse one 96-byte partner map

**Priority:** first decode candidate; largest arithmetic lead and a prepared
isolated prototype. The current group-two lane kernels rediscover equal-expert
partners from the same 24 route IDs in every gate/up/down tile. Replace that
repeated membership work with one GPU prepass emitting `partner[24]` U32,
96 bytes, reused by all three projections. Keep fixed slot grids, original slot
outputs, serial reduction order, clamp and F16 stores. No CPU route readback.

**Ownership:** worker owns `src/exl3/glm_group2_partners.zig` and its focused
probe/root. Coordinator alone touches the narrow production delegation in
`src/glm5_dflash_ffn.zig`/`src/exl3/glm_group2.zig` after a win. Do not overlap
those shared integration files with the other workers.

**Geometry and memory:** B1/T3, H4096/I2048, top-k 8, K2.25/MCG/W12, packed
rate36; T4 may be a correctness guard, not a second tuning study. Actual bank
layout and original serial 4 KiB lane partials remain unchanged. Added scratch
is 96 bytes per T3 call, plus the ordinary MLX output allocation; no expanded
weight bank or persistent cache is introduced.

**Expected ceiling:** membership scans disappear from many output groups, but
packed decoding, arithmetic and bank reads remain. A 10% reduction in a routed
component would be useful; no percentage is established. The 24.25 ms/round
profile marker is an intentionally loose upper bound including forced waits.
Do not promise a corresponding whole-model gain.

**One focused proof and performance test:** on captured real routes and resident
production expert banks, compare current complete `moeLayout` chain against the
prepass chain, including the prepass, preparation, middle/down, finish, host
construction, endpoint evaluation and frees. Check every output bit first;
cover repeated IDs, singleton/odd tails and partner bit boundaries in the
behavioral fixture. Use a small representative route set, three warmups and
11 interleaved AB/BA pairs with fresh graphs. Proceed to one short model gate
only if the inclusive paired result is consistent. Rejected group-three,
word-sharing, per-node NAX and altered reduction experiments are excluded.

### 2. Sparse prefill: overlap bounded 16-row attention tiles

**Priority:** first prefill candidate; directly attacks the context-growing
path after the cold 2048-row dense-SDPA chunk. `attendImpl` couples selector
work and packed SDPA to 16-row tiles and synchronously settles each tile.
`attentionChunk` settles the same output before returning, and the outer loop
asks for another evaluation. The second already-evaluated call is not itself
proof of another GPU wait; removing just that call is not the experiment.

**Ownership:** worker owns `src/glm5_attention.zig` and a focused cadence probe.
Keep `src/glm5_attention_nax_packed.zig` arithmetic and geometry unchanged.
Coordinator integrates any diagnostic counter or admission update. No edits to
retained projection or routed expert files.

**Concrete first change:** maintain at most two independent packed tile graphs
in flight, submit their output handles asynchronously, then settle each pair
before releasing the gathered banks. Keep per-tile selection, native SDPA query
batch geometry and causal masks unchanged. First isolate this schedule only;
Above approximately 14K, `glm5_indexpool_nax.scores` synchronously settles
each 2048-pool dot tile, so constructing the next selector can drain the queue.
Measure first at 8K with scalar selection, then 16K with NAX selection. If that
internal barrier erases the gain, stop this arm. A later focused alternative is
bounded two-dot-tile async submission in the scorer (two 2 MiB raw planes),
settled before argpartition; it needs its own paired proof. Do not combine
selector batching or wider SDPA tiles with this first arm.

**Geometry and memory:** BF16 `[16,64,512]` queries, gathered
`[16,2051,512]` BF16 KV, original 2051 selected indices and FP32 native attention
accumulation. Current limit is 64 MiB per pending MLA layer. Two pending tiles
need a conservative 128 MiB per layer, 256 MiB at async2: add 128 MiB to the
existing whole-request transient admission bill. Count actual retained arrays
and preserve the bound across partial final tiles and failures. Never retain
all 128 gathered banks of a 2048-row chunk (about 4 GiB per layer).

**Expected ceiling:** a 2048-row chunk has 128 tile boundaries per MLA layer.
The candidate removes up to half of the blocking tile boundaries, while retaining
all gathers, dot products and SDPA commands. There are eleven MLA layers;
32K records approximately 21K packed dispatches. That demonstrates opportunity,
not the proportion of prompt time spent waiting. A 5–15% gain in the whole sparse
attention component would justify integration; it is a hypothesis, not a claim.

**One focused proof and performance test:** with one fixed real T2048 query
block and selected indices at an 8K prefix, compare serial tile cadence against
two-tile cadence using the same native packed helper. Compare every BF16 output
bit and bounded peak live memory; test empty/masked/future/last ragged rows.
Include selection, gather, SDPA, output collection, evaluation and free in fresh
inclusive AB/BA samples. Three warmups and 11 paired samples suffice. If it wins,
one native 8K short model gate checks tokens and all cache state, then the
coordinator measures the selected HTTP arm. A new algorithm or wider NAX tile
requires a separate qualification and is outside this first candidate.

### 3. Retained KDA verifier projections: combine FA/GA column GEMV

**Priority:** smaller independent decode candidate. The current verifier uses
five retained BF16 column GEMV projections, separately projecting the same input
through FA128, GA128 and beta64. Prefill already owns an evaluated
`[320,4096]` joined BF16 bank per KDA layer. Slice its first 256 channels for
T3 verification using `weight[256,4096] @ input[3,4096,1]`, compact the two
128-channel results, and retain beta, FB and GB on their current paths. This
changes three first-stage commands to two. It does not retry the rejected
ordinary T3 NAX chain.

**Dispatch constraint:** native non-transposed GEMV selects BM1/BN8/SM1/SN32/
TM4/TN4 when `K >= 16 * output_width`. K4096 satisfies this at N128, N64,
and N256 exactly. Joining all 320 channels fails the inequality and switches
to BM4/BN1, changing key traversal and reduction. Do not implement that arm.
The N256 slice preserves the original template and has the same 64 output
groups as two 128-channel products, with one command instead of two. Verify
this source-derived arithmetic claim against all output bits before timing.

**Ownership:** worker owns a new `src/glm5_dflash_retained_cluster.zig` and focused
probe. Coordinator owns the narrow `src/glm5_dflash_kda.zig` delegation. Reuse
`KdaLayer.prepared_cluster`; no accessor or new prepared-weight owner is needed.
Do not modify the existing T2048 helper or general `linearRows` behavior.

**Geometry and memory:** B1/T3/K4096, BF16 input and retained joined weights,
FP32 reduction; compact outputs `[1,3,128]` each. Existing 85 MiB model bank
suffices. Added joined output is 1536 bytes per pending layer, 6144 bytes at
async4, plus existing output planes. Require bank availability, shape and compact
strides; fall back when the prefill bank was not prepared. No per-round bank
preparation, weight conversion, padding or custom shader is needed.

**Expected ceiling:** prior five-chain column batching saved approximately
52 microseconds versus serial at T3; further clustering is unmeasured. One saved
command per KDA layer is a modest opportunity, perhaps at most 1–2 ms per
verifier round. Stop if output compaction erases the inclusive gain. Lowrank/beta
and gate/post profile fields contain unrelated work and forced waits; their
approximately 12 ms per round sum is not this candidate's available saving.

**One focused proof and performance test:** use actual layer-zero retained
FA/GA/beta/FB/GB banks and a fixed nonzero normalized T3 BF16 input. Control is
the current five-column-GEMV chain. Compare every first-stage and downstream
output bit. Include both compact copies, beta, FB/GB, fresh graph construction,
endpoint evaluation and frees in 11 AB/BA pairs after three warmups. Check
one unsupported-input fallback and bank borrowing after the producing Ops scope
ends. Integrate only on a consistent inclusive win, then use the existing short
DFlash output/full-state gate. No follow-up variant sweep for a losing arm.

## Measurement and integration gates

1. Preserve the latest recorded HTTP cells; no old-binary baseline rerun.
   Component candidates use in-process interleaved controls, so they have a
   local comparison without repeating a loaded-model ladder.
2. Stamp exact HEAD, dirty-source hashes when applicable, binary hash/mtime,
   runtime and flags. Confirm actual engagement counters. A profile forces
   synchronization and cannot be called a throughput result.
3. Calibrate exact inputs through the tokenizer/template. HTTP rungs cross the
   2048-row dispatch threshold: predictable 2K has 2036 IDs and zero cluster
   calls; ordinary 2K has 2072 IDs and 34 cluster calls. Do not infer a cluster
   benefit by comparing those different workloads. Keep ordinary and predictable
   context rows separate, and report accepted drafts/tokens per step alongside
   rates. The ordinary 32K case has KDA misses; the predictable case has none.
4. After local wins, run one combined short model correctness gate and one
   profile-off 2K–16K HTTP ladder. Long-context qualification is one selected
   32K/64-output serial-token plus complete-state gate and one final 32K HTTP
   cell. Do not run the long ladder for each intermediate candidate.
5. Attribute current phases correctly: replay includes target cache preparation;
   commit is assistant context publication. The 32K publication bound is now
   five K/V pairs at 2560 rows, about 50 MiB. Original source context remains
   alive until successful publication. No destructive donation or cache alias
   mutation is authorized by a performance result.
6. A real exact-output paired win can land regardless of size. A small or noisy
   result does not justify another variant sweep. Preserve a losing experiment's
   evidence privately and remove its production hook. These three candidates
   are incremental; a remaining gap to 1500/60 is not grounds to overstate them.

## Excluded work

Do not reopen group-three mixed kernels, forced EXL3 NAX padding/cap25,
word-sharing, per-node packed decode NAX, shared FFN merging, the native T3
five-projection NAX chain, or the inconclusive output-A6 hoist without new
strong evidence. Keep the QKV-only exact A6 hoist. Do not tune recurrence merely
because its profile marker is large: its forced-evaluation tax and existing R4
qualification make that an unsupported priority. No precision restoration,
embedding offload, speculative precision reduction, new cache transaction API,
or broad parameter search belongs in this wave.
