# GLM indexed attention: BF16 NAX component qualification

The isolated masked D512 NAX path is faster than the existing scalar indexed attention on the
qualified synthetic geometry. At 32K history it reduces one 31-query attention call from
15.348 ms to 6.868 ms, with about 0.137% relative output error. It still computes the full key
history, so its cost grows with context. These component results do not establish full-model
prefill/decode rates or the 1500 tok/s prefill and 60 tok/s decode goals.

The probe is [`src/glm5_attention_nax_probe.zig`](../src/glm5_attention_nax_probe.zig). It exercises
[`src/glm5_attention_nax_mask.zig`](../src/glm5_attention_nax_mask.zig) without a model or runtime
hook. The user allows ordinary NAX compound rounding: the selected arm uses BF16 Q, K, V,
cache, and output with native float accumulators. No precision-restoration arm was measured.

## Corpus and correctness

Each history has 31 queries, 64 heads, 512-wide latent values, scale 1/16, and 2051 selected slots.
Q and cache are independent fixed-seed BF16 normal arrays with standard deviations 1 and 0.5.
Normal rows select 512 shuffled unique completed pools, expand each to four tokens, and append
only the causal incomplete tail. The first row is entirely invalid; the second has only key zero;
the third also includes negative, future, and out-of-range IDs. The selection test independently
checks uniqueness and causal live counts.

The reference is `glm5_attention_prefill.attend`: the existing exact online scalar traversal with
its qualified direct finalization. The candidate must return BF16 `[31,64,512]`. All-invalid rows
must be positive zero, and the sole-key row must equal the source cache's key-zero bits at every
head. Numerical metrics below exclude those two special rows.

| History | Relative L2 error | Cosine | RMS ratio | Maximum absolute error | BF16 mismatches / all outputs |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.00137715 | 0.999999106 | 0.99966920 | 0.00048828125 | 104099 / 1015808 |
| 16384 | 0.00137459 | 0.999999109 | 0.99967013 | 0.00048828125 | 104536 / 1015808 |
| 32768 | 0.00137247 | 0.999999112 | 0.99967193 | 0.00048828125 | 104318 / 1015808 |

Both implementations produced finite outputs everywhere; empty-row and key-zero checks were exact.
Approximately ten percent of raw BF16 outputs differ, so this is explicitly a rounding tradeoff.
The component gate requires relative L2 below 0.01, cosine above 0.9999, and RMS ratio between 0.99
and 1.01; passing that synthetic gate does not certify real-model quality.

A successful call uses the prototype's explicit C API `force_fused=true`, array bool mask, and
D512 full-attention shape. Unsupported fused shapes raise an error. Native MLX otherwise routes
this small-query masked shape to an unfused fallback; the probe relies on the actual forced call
and returned shape, without an invented engagement counter. It also checks observed peak memory
against the conservative temporary bill plus allowance, which would reject a full 32K Q*H*K
score allocation. Peak-minus-before uses saturating subtraction because queued frees can make
peak lower than an earlier active-memory sample.

## Paired timings

The measured operation includes fresh Ops, scalar arguments or mask/transpose construction,
forced evaluation, and array frees. Inputs are materialized before measurement. Each arm has two
warmup calls, followed by six interleaved pairs with alternating arm order. Each call has a fresh
scope; no full-history cast can accumulate across chunks. The BF16 arm uses the existing cache
storage directly and has no per-query KV gather.

| History | Scalar median | NAX median | Component speedup | Conservative temporary bill |
|---|---:|---:|---:|---:|
| 4096 | 7.850 ms | 1.085 ms | 7.24x | 8952862 bytes |
| 16384 | 10.356 ms | 3.531 ms | 2.93x | 9714718 bytes |
| 32768 | 15.348 ms | 6.868 ms | 2.23x | 10730526 bytes |

This run used source matching the accompanying component commit, ReleaseFast, MLX 0.32.3 source
`64ea011cb`, foreground `taskpolicy -a`, exclusive GPU lock `glm-nax-probe-v61`, max fans, and ten
seconds idle with maximum sensor temperature 50.43 C before the run. No concurrent build or GPU
job ran. Measurement key: `glm53-nax-probe-20261003`. Raw results, fan status, and binary/source
stamps are recorded in the private measurement ledger.

The NAX median grows 3.25x from 4K to 16K and 1.94x from 16K to 32K. The native kernel visits all
history tiles even when most membership bits are false. A real-model replay and quality/layout
checks are required before choosing a runtime policy; sparse tensor traversal remains a distinct
way to address the remaining context scaling.

## Running the diagnostic

Compile an isolated test root that imports the prototype and probe with the project's normal
MLX/EXL3 linking. `SUSHI_GLM_NAX_PROBE_OUT` enables the component test and names its result JSON;
without that environment it skips the GPU diagnostic. The pure selection test and prototype
geometry/membership tests remain ordinary tests. Run the generated test executable under the
[GPU measurement policy](process-measurement.md). This component test registers no serving path.
