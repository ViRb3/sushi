# Fixed T3 KDA core fusion: research proposal

Recommend one exact, tree-aware KDA core helper for B1/T3/H64/D128, restricted
to parents `[-1,0,1]` and `[-1,0,0]`. Fuse convolution/prework, recurrence and
RMS/gating while preserving all projections, output projection and normal
commit/replay. This is a source-supported experiment, not a demonstrated gain.
Accepted runtime remains `4fcb541e`, qualified CLI `102cb8d5`; this report adds
no runtime implementation.

## Current work and distinct mechanism

`glm5_dflash_kda.applyLayer` currently constructs raw QKV, FA/FB and beta,
calls `glm5_kda_prework.applyTree`, then `recurrentLeaf`, constructs GA/GB,
and calls `glm5_kda_fused.post` before the output projection. The ordinary
one-token `KdaLayer.applyFused` already uses `glm5_kda_fused.step` to keep this
core head-local. There is no missing serial fusion to fix.

The proposed tree helper reuses that head-local organization. It retains the
original five replay arrays and selected leaf; it does not change endpoint
retention or replay policy. Unlike the rejected canonical recurrence, it keeps
the original `saved[3][4]` parent lookup and recurrence body. Unlike the rejected
all-endpoint candidate, it writes only the current selected FP32 leaf. The
four-product A6 and joint-QKV candidates changed projection work; this proposal
leaves their accepted baseline arithmetic and geometry untouched.

The current leaf kernel uses groups `(32,4,1)`, one SIMD per value channel.
Use the existing serial organization of 1024 threads per head: 32 SIMDs, each
processing `dv = sg + 32*j` for `j=0..3` sequentially. Each SIMD retains the
same four contiguous keys per lane, scalar `i=0..3` accumulation order and
`simd_sum` reduction. Reset and reuse its three saved states between value
channels; do not keep four value channels live at once. Group size changes
scheduling, not the proposed lane/key reduction order.

Use the existing parent-derived four-tap window map, including the preceding
three raw convolution rows. Preserve convolution and SiLU BF16 boundaries,
BF16 normalized Q/K/V and beta, FP32 decay/state/accumulators, raw recurrence
BF16 rounding in shared storage, and the exact current post body and BF16
gate/output stores. GA/GB must be available before the fused call; charging
that dependency change is essential. T1/T2 and unsupported trees retain the
existing path and numerical math.

## Cost, memory and external evidence

For T3, the five replay outputs total 246144 bytes: Q/K/V 147456, decay 98304
and beta 384. They remain globally materialized for misses. The selected leaf
remains 4194304 bytes. The helper can remove the separate 49152-byte raw-Y
plane, while keeping its BF16 rounding before RMS; this is not an admission
credit until ownership and peak are proved. Estimated shared arrays occupy
about 6 KiB per head. Keep original small stored tensors and compressed BF16
MLA cache unchanged; add no persistent cache or all-endpoint buffers.

Head-local operands replace recurrence's repeated logical Q/K/decay reads
(24 MiB per layer at this geometry) with shared reads. These are logical
accesses, not measured DRAM traffic: current caches may already serve them.
Higher register pressure, spills, fewer resident groups and earlier gate
dependencies may erase the benefit. Three core dispatches becoming one does
not establish a speedup.

oMLX's low-level `kda_decode_step` supports a 1024-thread head-local core with
precomputed low-rank/gate operands, but its temporal rows are a chain and it
does not supply Sushi's replay arrays. The current serial Sushi helper already
adapts this organization. ds4's `kernel_glm53_kda_decode` also keeps operands
head-local, but uses FP32 operands and different dot/FMA order. TensorFold's
CUDA KDA `chain` uses different scratch/backend contracts. These are evidence
for the organization, not interchangeable arithmetic or performance evidence;
see [the external comparison](glm5-external-efficiency-comparison.md).

No current unperturbed measurement isolates this core's removable cost. The
[13.49% raw-QKV counterfactual](glm5-qkv-attribution-result.md) belongs to a
different work class and one tree; it is not this candidate's budget. The
remaining verifier includes projections, experts, MLA and head work. The
benefit ceiling is the complete core's unknown share, not the full verifier,
and no 60-token/s forecast is justified.

## One decisive gate

An implementation worker owns one new fixed helper and private original-L0
probe; root owns the default-off verifier seam, ownership/admission and model
acceptance. No kernel/group/tree-width variants are proposed.

First require exact current-path and matching serial-ancestry comparisons for
both trees: full BF16 layer outputs, all five replay arrays, selected FP32
leaf, every accepted convolution tail/state and normal hit/miss commits.
Include initialized/empty state, signed zero and nonfinite cases without
sanitization, invalid guards, T1/T2 fallback and multioutput alias/release
proofs. Arithmetic mismatch stops this exact proposal; do not restore
precision or silently introduce tolerances.

Then use three warmups and 11 interleaved complete-layer/normal-commit pairs,
with equal held references. Include every projection, preparation, new gate
dependency, endpoint settlement, replay, construction and free. Record every
pair, engagement, peak and release outcomes. A noisy or losing result stops.

A clear component win permits one current-stack matched actual-model ABBA
gate with frozen 8K inputs, A6/HC/native/packed32 unchanged, 192 outputs and
strict serial/spec IDs plus all valid states. Charge clone/decode/commit and
cleanup, retain equal references and all existing reserves, and compare gain
with control drift. Root then qualifies 2–32K before acceptance. Geometry is
context-independent, but MLA cost and acceptance are not; the component alone
cannot establish long-context throughput or stability.
