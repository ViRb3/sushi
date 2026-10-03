# Current singleton-resource experiment

Research only at accepted runtime `e1597cc2`, 2026-10-03. Recommend one bounded
**gate/up-only compiled singleton split**, with selection folded into the
existing middle kernel. This is an occupancy/resource hypothesis, not evidence
of spills, saved bank traffic or an expected speedup. No prototype/build/GPU.

## Current evidence and scope

The current8192/192 N2/A6/native-target job matched every token and247959552
valid state bytes. Its full original E288 routed replay matched516096 BF16
output values. All42 independent saved chains, including prepare/gate-up/
middle/down/finish/evaluation/frees with current settlement, took17.823292ms
median over three samples, range0.383%. This excludes router/shared/MLA/KDA/
assistant work and is not whole-verifier attribution.

Three complete T3 rounds contained3024 assignments,461 pair leaders and2102
singletons:69.51% of assignments and82.01% of leaders were singleton work.
The saved first round alone had748 singleton/130 pair leaders, so85.19% of
its leaders were singleton. Pair distances averaged9.52 slots, median9.
These are actual operands/routes/full banks, unlike older compact fixtures.
The complete17.8ms component is the only relevant cost ceiling; singleton
counts do not establish its time share or predict a60tok/s model result.

`glm5_dflash_ffn.apply` calls `glm_group2.moeLayout(serial,grouped,lane)`:
lanePairPrepare → pairLayout → lane middle → grouped lane down → weighted
finish. Generated `LANE_PAIR_SOURCE` combines an original eight-accumulator
singleton body and a sixteen-accumulator pair body after inline MEMBERS.
Serial reduction uses4KiB threadgroup partials. Resource allocation is per
compiled pipeline; a dynamic singleton branch may inherit the larger body
footprint, but source does not prove this occurs or limits occupancy.

## One candidate, no routing prepass or output aliasing

Strict T3/S24, H4096/I2048, top8/E288, n36/MCG/W12/clamp10 only. Keep original
slot IDs and inline ballot pairing; no host classification, compact routing,
weight copies or alternate group size. Compile two gate/up kernels from the
same current lane body: singleton-only and paired-only. Each keeps original
per-member dot/FMA/reduction/F16 store order and writes original slots.
The singleton pipeline must contain no paired accumulator body, so it can
actually have a smaller compiled resource footprint.

The singleton gate/up dispatch also emits a24-entry class mask **before**
followers return: true only for even-ranked unmatched leaders. Exactly one
thread at output-tile0/projection0 writes each original slot's mask, including
false for paired followers. This handles triples without rebuilding MEMBERS
in a separate metadata command. The paired dispatch writes both pair slots;
singleton dispatch writes singleton slots. The two output families remain
separate MLX-owned arrays: do not mutate/alias another kernel's output buffer.

A narrow middle clone chooses the gate/up pointer family uniformly per slot
using that mask, then runs the unchanged lane-middle body. This replaces only
input pointer choice; no global merge, F16 addition or changed signed-zero
rounding. It must never read unwritten rows in the unused family. Down and
weighted finish remain the exact current grouped-lane kernels. Do not also
split down in this experiment.

Cost: one extra gate/up dispatch per layer, doubled gate/up candidate grids
and MEMBERS discovery, the same active dot work/decoded weight visits, and
an extra two F16[S24,2048] output planes (196608 bytes) plus24-byte bool mask.
Both new output families have4KiB on-chip partials. No zero-fill of unused F16
rows or extra projection merge should be necessary because middle chooses only
written slots; the class mask must be fully written. Actual register pressure,
cache behavior and added launch cost can erase the benefit. Bound all pending
graphs conservatively;42 calls add at most about7.88MiB of these new planes
before considering allocator lifetimes, not a persistent weight/cache bill.

## Ownership and gates

One worker owns an isolated helper/probe and only tiny generated-source
exports needed to reuse original lane/middle bodies. Coordinator owns later
FFN delegation, admission and model runs. Keep current production fallback
and all T1/T2/T4 paths unchanged. Do not add a routing metadata framework,
group3/word/window/grid/radix variant or precision restoration.

First prove class masks against independent original-slot occurrence ranks,
including odd triples, all-single and all-pair cases. Compare every written
gate/up F16 bit with the current kernel, every middle F16 bit and all516096
saved routed BF16 output values on full original banks. Include distinct
inputs/scales, signed zeros and unwritten-family safety; no uninitialized
buffer hashes. If available, report pipeline resource metadata, but never
infer spilling from accumulator counts alone.

One current42-layer whole-chain comparison uses the saved actual X/IDs/scores,
original bank order, same four-layer submissions/final settlement and all
allocations/classification/middle/finish/frees: three warmups, eleven pairs.
Stop noisy/losing or mismatching results without another split/layout variant.
A winner gets one matched8192/192 N2/A6/native-mode model ABBA with equal
reference memory, all rounds/cleanup and strict serial token/valid-state
parity. Routing geometry is prefix-independent, but routes and acceptance
change across2K–32K; selected32K is a final winner gate, not an overlap
extrapolation. Artifact key `glm53-current-routed-8k-20261003` anchors this
proposal; the old145ms forced-wait marker remains outside its cost argument.
