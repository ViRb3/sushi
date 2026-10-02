# GLM-5.3-Flash efficiency audit

Audit date: 2026-10-02. This report covers the implemented EXL3 clamped-expert path, mHC and KDA primitives,
and the initial layer foundation in `src/glm5_model.zig`. It does not certify a complete GLM forward or open
the served-architecture gate. The absence of unfinished attention or model-loop integration was not treated
as an efficiency defect.

## Findings and fixes

| Finding | Evidence and impact | Resolution |
|---|---|---|
| Repeated prefill routing metadata | `moeSwigluClamped` called `innerGemmSorted` independently for gate, up and down. Each call rebuilt the same window table from identical sorted slots. | Commit `9087677b` shares one table across all three projections. Whether repeated evaluations drain outstanding GPU work depends on MLX scheduling; no stall count was measured. |
| Unnecessary activation copies and permutation work | The staged clamped path repeated token rows, gathered them into sorted order, prepared gate/up separately, then inverse-sorted the permutation. At 512 tokens, top-k eight and width 4096, each BF16 repeated/sorted token plane is 32 MiB. Decode also sorted slots although indexed GEMV accepts their original order. | Commit `9087677b` keeps decode slots in their original order. Prefill prepares gate/up directly from token rows and uses a scatter before the existing top-k reduction. Clamp arithmetic and final reduction order remain unchanged. |
| Prefill convolution tail retained its full parent | `KdaLayer.apply` used `contiguous(slice(conv_input, ...))`. A batch-one tail is already contiguous, so this operation can retain the entire prefill allocation. | The layer foundation in commit `f352c5d6` uses `compactConvTail` and the existing materialized-copy helper for multi-token calls. Single-token decode keeps the inexpensive view. No additional evaluation or synchronization was added. |

The 32 MiB figure describes tensor geometry, not a measured reduction in peak process memory. The optimized
path still needs its prepared projection planes and other intermediates.

## Verification

The clamped-routing characterization passed before the refactor. Afterward, the optimized and staged paths
produced identical output bits across every even packed width n32–64, covering all 17 rates from 2 to 4 bpw,
with MCG/W12, top-k eight, width 256, rows one and 17, and BF16 and FP32 inputs/outputs. A separate BF16 test
checks GLM's hidden/intermediate widths 4096/2048 at K2.25/W12 with the same decode/prefill row counts.
The GLM host-reference clamp test also covers every supported rate in that interval.

The convolution-tail regression failed before the fix because the returned pointer remained inside its
parent allocation, while its contents were correct. It passed after the fix, checking both unchanged values
and independent compact storage after evaluation.

Relevant tests:

- `exl3 clamped routing preserves staged bytes at every 2 to 4 bpw rate`
- `exl3 clamped routing preserves GLM production width bytes`
- `GLM EXL3 clamp supports 2 to 4 bpw in decode and prefill`
- `GLM prefill convolution tail owns its compact storage`

The GLM-filtered suite and full `zig build test -Doptimize=ReleaseFast` completed with exit zero after the
combined efficiency and correctness fixes. An earlier full-suite attempt encountered deliberately failing
concurrent config and malformed-bank tests; the subsequent combined-tree run passed. Checks ran alongside
other permitted workloads, without full-model timing. This audit establishes arithmetic parity and reduced
work in the graph, not a measured decode or prefill speedup.

## Deferred work and integration requirements

**Prepare immutable weights once.** `KdaLayer.apply` currently concatenates, transposes and materializes the
three convolution weights on every call. It also repeats the FP32 conversion/reshape of `dt_bias`,
`exp(A_log)`, and output-normalization weights. These values belong in an owned prepared-constants object
created after load validation and released by layer/model deinitialization. Existing layer structs borrow
weight handles; adding cached allocations without their owner and cleanup path would introduce leaks.
The mHC path also deserves a review of repeated static weight transforms once this ownership pattern exists.
No prepared-weight cache was added in this audit.

**Bound operation lifetimes.** `Ops` retains its intermediate handles until deinitialization and permits at
most 768 handles. The eventual model loop must use a bounded scope, such as one per layer, rather than keep
one scope across all 45 layers. It must evaluate the compact cache arrays with the normal boundary evaluation
and release the scope. A lazy copy alone retains its source graph until it is evaluated. This is an integration
requirement; the audited file did not yet contain the complete model loop.

**Profile before changing mHC arithmetic.** The collapse kernel assigns sigmoid/Sinkhorn work to one thread
while the other threads wait at a barrier. This is a serialization point, but its model-level cost was not
measured. Parallelizing it requires preserving or separately validating the reduction order.

**Keep serial KDA as the correctness reference.** KDA currently uses a recurrence that loops over token
positions. A block/chunk prefill implementation may improve long-prompt throughput, but an existing GDN path
is not a proven substitute for GLM's vector decay and FP32 state. Full native validation and profiling should
precede that optimization.

## Serial decode scheduling follow-up

The first native model loop evaluated every layer synchronously, including one-token decode. The decode
schedule now submits the residual and changed cache outputs asynchronously every four layers, then
performs one checked evaluation of final logits and all live request cache arrays before advancing the
offset. It preserves the arithmetic graph. Per-layer operation scopes can release their handles because
the live residual, cache handles and submitted graph retain the required dependencies.

`Request.decode_async` defaults to true. Multi-token calls retain synchronous layer boundaries, and
`Request.profile` also forces that schedule so per-layer timers measure completed work. A caller can
select synchronous decode explicitly for comparison. Async scheduling may temporarily retain an old
and new cache generation; it must not be treated as a reduction in the cache memory bill.

The scheduling regression uses a nonzero four-layer model with three KDA layers, one MLA layer, dense
and routed/shared feed-forward paths. Prefill followed by seven one-token calls produces bit-identical
logits and every recurrent, convolution, latent, pooled and raw-tail cache array under both schedules.
The test verifies nonzero recurrent state, request offsets and reset, and counts the actual synchronous
and asynchronous submissions. A second test completes 80 decode calls; active Metal memory after the
80th remains within 64 KiB of the 16-token warm snapshot, with unchanged reserved cache capacities and
both snapshots at complete pool boundaries. It also checks that profiling restores synchronous layers.
The focused ReleaseFast tests passed. These checks establish scheduling parity and bounded retention
in that fixture; throughput and full-model peak memory require the separate serial benchmark.
