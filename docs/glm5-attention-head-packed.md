# Sparse head-packed native attention

The isolated component `glm5_attention_nax_packed.zig` gathers each real
query's selected latent rows once and shares them across its 64 heads. It
reshapes Q from `[T,64,512]` to `[T,1,64,512]`, using the real query as batch
and the original heads as a sequence of 64 independent queries. K and V alias
one `[T,1,2051,512]` BF16 bank. An array mask `[T,1,1,2051]` broadcasts the real
query's valid, causal selection across all heads. Scale remains explicitly
1/16. There is no causal string, sink, or head-specific bias, so the reshape
introduces no position semantics or arithmetic.

Native MLX v0.32.3 full attention uses NAX for D512/Q64/GQA1. `force_fused=true`
refuses an unsupported fused shape. Q/K/V/cache/output stay BF16 and native
score/output accumulators stay FP32. Ordinary NAX compound rounding is accepted
under the user's precision policy; this is not a lossless kernel rewrite.

## Bounded gather and semantics

A copy kernel emits the single contiguous BF16 bank and its bool mask. It
checks selected IDs against zero, history, and `offset + real_row`, and writes
zero for invalid slots without reading cache data. This prevents masked NaN/Inf
key zero from poisoning attention while preserving valid key zero. All-invalid
query outputs are explicitly zeroed after SDPA. Valid IDs must be unique per
real query, as guaranteed by IndexPool selection.

The maximum chunk is 16 real queries. The gather is about 32.05 MiB at that
size; the conservative source ledger allows query/output copies, expanded mask
and index bookkeeping within 64 MiB. Cache strides must be `[512,1]`; automatic
flag-based kernel input copying is disabled to prevent a hidden full-history
copy. Strided selected IDs can receive a small contiguous copy. K/V need no
native copy because the gathered bank's last dimension is contiguous.

Each component chunk uses a fresh Ops scope and settles its output before
releasing the gathered bank. Only small output planes survive concatenation.
There is no full-history cast or floating `[Q,H,K]` score allocation. The
control API provides `enabled()`, `bind(on)` with `Binding.restore()`, dispatch
counters, and `transientBudget(chunk,pending_layers)`. The latter reserves
64 MiB for every pending layer when enabled and the original chunk exceeds
eight rows. The opt-in environment name is `SUSHI_GLM_ATTENTION_PACKED`.
The primitive remains independently callable for component qualification;
runtime callers select it through `enabled()`.

## Component qualification, 2026-10-03

Six focused tests passed on MLX `64ea011c` / v0.32.3. They covered geometry and
scratch, unsorted unique pool/tail selection, future and out-of-range IDs,
all-invalid rows, sole valid key zero, masked NaN/Inf key zero, distinct coded
heads, different real-row key sets, and ragged last slot 2050. Sole-key and empty
outputs were exact. Normal outputs had relative L2 difference 0.00136–0.00138
versus the original scalar online attention, with cosine above 0.99999909.
Difference versus full-history NAX was about 0.00078–0.00079 relative L2.
There were no nonfinite outputs in the six component cases.

The fixed-seed BF16 fixtures used 64 heads, latent width 512, 2051 selected
slots, query standard deviation 1 and cache standard deviation 0.5. Timings
include gather, masks, reshapes, SDPA, per-chunk evaluation, concatenation and
free. The scalar control uses the existing exact online indexed traversal;
the full-history NAX control builds a membership mask over the same history.

| Real queries | History | Scalar, ms | Full-history NAX, ms | Head-packed NAX, ms |
| --- | ---: | ---: | ---: | ---: |
| 16 | 4096 | 4.278 | 0.919 | 0.674 |
| 16 | 16384 | 5.222 | 2.785 | 0.687 |
| 16 | 32768 | 5.883 | 5.267 | 0.715 |
| 31 | 4096 | 7.868 | 1.069 | 1.785 |
| 31 | 16384 | 9.927 | 3.518 | 1.852 |
| 31 | 32768 | 11.775 | 6.793 | 1.893 |

The 31-query packed arm uses two settled chunks. Packing beats full-history
NAX at 16K/32K and remains nearly flat with history, but loses at 4K with
31 queries. This supports a measured selection policy, not a universal gain.
It establishes no full-model throughput or decode result.

The exclusive component run used ReleaseFast, interactive QoS, maximum fans
requested, 47.06°C initial temperature, and ten seconds idle. Two warmup rounds
preceded six alternating ABC/CBA rounds. Raw measurements, source and binary
hashes, command, runtime provenance, and fan telemetry are archived privately.
Control/admission APIs and an audit fix preventing flag-based full-cache copies
were added after the measured run; they do not change its attention math or
its already-contiguous fixture path. Their validation belongs to the subsequent
integration build. No model dispatch or default was changed by this component.

## Rejected decode/verification extension

A separate 2026-10-03 probe tested one to three nodes against the existing
eight-split scalar attention and original merge. The NAX gather understood
immutable prefix plus ancestry tails, using sibling paths `[0]`, `[0,1]`,
`[0,2]`; valid tail tokens were read from the branch overlay. Thus this was an
attention comparison for verification views, with shared IndexPool selection
excluded from both arms. Each node used a fresh scope, including tail take,
gather or scalar partials, native SDPA or merge, evaluation and free.

| Prefix | Nodes | Split8 + merge, µs | Overlay NAX, µs | Latency change |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 1 | 603.105 | 743.354 | +23.25% |
| 4096 | 2 | 777.250 | 1191.187 | +53.26% |
| 4096 | 3 | 1051.896 | 1606.230 | +52.70% |
| 16384 | 1 | 351.021 | 464.709 | +32.39% |
| 16384 | 2 | 587.646 | 845.042 | +43.80% |
| 16384 | 3 | 846.000 | 1347.104 | +59.23% |
| 32768 | 1 | 283.396 | 423.480 | +49.43% |
| 32768 | 2 | 560.917 | 898.542 | +60.19% |
| 32768 | 3 | 844.167 | 1249.646 | +48.03% |

All nine cases had zero paired wins over six alternating rounds, following
two warmups. Outputs were finite, with expected NAX relative L2 difference
0.00131–0.00139 and maximum absolute difference 0.00048828125. The native
Q64/D512 path has only two tensor threadgroups per node; its low parallelism
and additional gather/launch costs are possible explanations, not a measured
cost attribution. There is no cross-history scaling claim for these small
measurements.

Two focused tests passed on MLX v0.32.3. The quiet ReleaseFast run used an
exclusive GPU lock, interactive QoS, maximum fans requested, 47.06°C initial
temperature and ten seconds idle. Source, raw samples, errors and provenance
were archived; the candidate/probe/wrapper were removed. No model hook,
additional variant or full-model run was warranted.

## Full-model combination check

Source `a88d8921` combines packed attention with ordinary BF16 head-batched MLA
projections on the 2.3bpw target. A 4096-token prefix (the existing 2048-token
fixture repeated twice), followed by 64 forced reference continuation tokens,
completed with finite scores and matching top tokens at all 64 positions. Mean
KL relative to the same quantized model with both paths disabled was 0.00394027;
first-position KL was 0.00032350; maximum row KL was 0.060244. This measures kernel drift; it is not KLD against
the original BF16 checkpoint.

The candidate recorded 11 query and 11 value projection dispatches and 1408 packed
attention dispatches. Peak active MLX memory was 95,500,858,424 bytes. Both arms kept
BF16 compressed MLA cache and FP32 KDA recurrent state. ReleaseFast full suite,
HTTP admission tests and 12 staged-runtime checks passed. Foreground QoS, exclusive
GPU lock and max fans were used. Measurement key: `glm53-packed-headbatch-drift-20261003`.
Cold reference/candidate prefill times were 10.121/4.636 seconds; they are diagnostic
order-dependent timings. The separate llmprobe ladder determines throughput.
