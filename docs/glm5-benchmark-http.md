# Native GLM diagnostic HTTP benchmarking

`glm-bench` loads the native GLM target once and exposes a loopback-only HTTP
bridge for `llmprobe --bench-only`. It is a diagnostic benchmark command. It
does not register GLM in the public server, and it is not GLM agent/tool support.
Its model row explicitly reports `diagnostic_only`, `benchmark_only` and
`has_tools: false`; nonempty tools and tool messages receive named errors.

The routes are `GET /health`, `GET /v1/models`, `GET /props`,
`POST /v1/chat/completions` and a diagnostic `POST /tokenize` that counts the
same templated chat input for exact workload calibration. Each completion uses the existing tokenizer and
chat template, a fresh native Request and, when selected, a fresh DFlash context.
The target and optional assistant stay loaded; prefix reuse is disabled. One
thread owns all MLX calls, including frees, and services requests sequentially.
The public serving gates and cache precision remain unchanged: BF16 compressed
MLA caches and FP32 KDA state.

```sh
SUSHI_GLM_KDA_VALUE_ROWS=4 \
SUSHI_GLM_A6_DENSE_PREFILL=1 \
SUSHI_GLM_DFLASH_GROUP2=1 \
SUSHI_GLM_LANE_PAIR=1 \
SUSHI_GLM_DOWN_LANE=1 \
SUSHI_GLM_DFLASH_DENSE_ROWS=1 \
SUSHI_GLM_DFLASH_MINI_HEAD=0 \
taskpolicy -a zig-out/bin/sushi glm-bench "$GLM_MODEL_DIR" \
  --assistant "$GLM_ASSISTANT_DIR" --port 8094 \
  --ctx-size 132096 --prefill-chunk 2048 --memory-gib 110 \
  --benchmark-ignore-eos
```

An explicit `--assistant` selects the existing layerwise DFlash2 verifier,
N2/C4, affine-row/FFN mode and async4. Omitting it selects native greedy serial
output with async4 layer scheduling. Props and final usage metadata identify the
actual backend, assistant, MLX runtime version, chunk width and experimental
flags. The environment selects the existing qualified row/layout arms; the
bridge does not silently override them.

Sampling is greedy. Completion budgets honor `max_completion_tokens` or the
legacy `max_tokens`, and usage counts actual generated IDs, including the first
ID obtained from prefill. SSE sends the first available native token, subsequent
verified IDs, one finish chunk, a requested usage chunk and `[DONE]`. Incomplete
UTF8 byte fragments wait for subsequent tokens; final invalid sequences use the
same replacement policy as non-stream text. DFlash's already emitted pending ID
is not emitted twice when its verification round commits. No output is fabricated
and no delays are inserted to simulate token delivery.

`ignore_eos: true` is accepted only when the explicit
`--benchmark-ignore-eos` switch is present. This supports llmprobe's equal-length
`ignore_eos`/`min_tokens` probe; it does not become the default chat stop policy.
Every request logs its exact prompt/output counts and effective EOS mode.

Context admission includes the complete tokenized template plus requested
completion budget; input is never truncated. It checks model/configured context,
conservative native/assistant cache and activation memory, full cache-reserve
replacement bills and the unchanged DFlash branch scratch cap. With an assistant,
the qualified reserve helper expands latent/pooled capacity after prefill and
before the first speculative clone. It preserves processed positions and cache
bits while avoiding growth inside branch verification. If memory or scratch
cannot admit a rung, the HTTP error includes a named reason and maximum allowed
context. This remains an admission result, not a throughput value.

```sh
npx llmprobe@0.6.13 localhost:8094 --bench-only \
  --rungs 2k,4k,8k,16k,32k --runs 1
```

The owner-requested baseline completed through 32K. The later 64K/128K rungs
were cancelled and are not recorded as completed results. Run the selected
ladder once for a runtime/backend baseline. Subsequent
optimization iterations use only `--rungs 2k,4k,8k,16k` with the same settings.
The caller must hold the GPU lock for the loaded model and benchmark, restore
foreground QoS, pause competing compute and follow the fan/cooldown protocol in
[process-measurement](process-measurement.md). Record binary/runtime stamps and
raw llmprobe artifacts. Old 1f8 runtime measurements are not a same-runtime
control for the v0.32.3 baseline.

## Completed 2K–32K baseline

The user stopped the wider ladder after32K, dropping64K and128K. All five
measured calls below completed after their warmups in `llmprobe 0.6.13 --bench-only
--runs 1 --reasoning default`. Rates use the native server's timers and usage
counts; the interrupted client did not emit its final aggregate report.

| Rung | Actual input tokens | Prefill seconds | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| 2K | 2,036 | 2.103 | 968.2 | 41.33 |
| 4K | 4,059 | 9.341 | 434.5 | 41.21 |
| 8K | 8,225 | 25.496 | 322.6 | 36.72 |
| 16K | 16,278 | 57.118 | 285.0 | 34.48 |
| 32K | 32,747 | 129.610 | 252.7 | 30.89 |

Every measured call emitted192 tokens. Configuration:2.3bpw A6-trunk target,
A6g128 DFlash2, N2/children4, async4 verification, group2/lane/down/dense rows,
R4 recurrence and A6 dense prefill, chunk2048 with dense SDPA/async2 prefill,
mini head off, BF16 compressed MLA and FP32 KDA. The ReleaseFast binary's
source equals `d054bac0`; MLX was0.32.3/`64ea011c`, mlx-c `56b2d39` with its
global-scale compatibility patch. Exclusive GPU lock, foreground QoS, max fans
and ten-second cooldown were recorded. No competing compile/GPU work occurred.
Private artifact `glm53-llmprobe-baseline-20261003` retains exact command,
completed request IDs/times, props, binary/runtime provenance and truncation.

The source investigation identifies query absorption and value unembedding
as M1 quantized projections during indexed prefill. Head batching can admit
NAX on this runtime, but changes coefficient rounding and reduction order;
precision and meaningful long-prefix KLD checks are required before promotion.
The scalar attention loop's query batching/evaluation cadence is a separate
optimization candidate. Neither is a measured improvement in this baseline.

The initial v0.32.3 qualification passed 15 focused tests with one gated native
server skip. A live target+A6-assistant smoke then calibrated exactly 2048
prompt IDs through the tokenizer/template and generated eight native output IDs.
Stream and non-stream IDs and UTF8 text matched, usage/finish/DONE framing passed,
and a nonempty tool request returned the named 400. Both requests started with
fresh state. The reserve helper billed 31,719,424 bytes on this short request.
The private `glm53-bench-http-20261003` artifact retains requests, raw SSE,
response IDs, props/model metadata, red/green logs and binary/runtime/source
hashes. These are correctness/surface checks; the separately recorded full
llmprobe ladder is the throughput baseline.

### BF16 packed-attention candidate

`SUSHI_GLM_ATTENTION_PACKED=1` selects bounded head-packed native NAX attention
for sparse BF16 MLA prefill with more than eight input rows. Each call gathers
only the 2051 selected latent slots; it reuses one BF16 bank for K and V and
keeps native accumulation in FP32. Invalid and future slots contain zeros and
have false mask bits. All-empty selections return zero. Serial decode and the
DFlash verification attention path retain their existing dispatch.

The candidate uses at most 16 real query rows per settled chunk. Admission
adds a conservative 64 MiB per pending layer for the gathered bank and native
attention intermediates. It also now includes the existing head-batched MLA
permutation buffers: a 2048-row chunk with two pending layers adds 768 MiB.
The response diagnostic records both opt-in settings and the number of packed
attention dispatches for that request.

The owner permits ordinary BF16 NAX compound rounding with FP32 accumulators
and canceled precision-restoration work. The component qualification is in
[the packed-attention report](glm5-attention-head-packed.md). Full-model throughput
and long-prefix output drift must be measured before promoting this candidate.

### Packed attention and head-batched MLA: short ladder

Source `a88d8921`, MLX 0.32.3, 2.3bpw target and A6 DFlash2 assistant,
N2/children4/async4/group2, BF16 compressed MLA and FP32 KDA. Both
`SUSHI_GLM_ATTENTION_PACKED=1` and `SUSHI_GLM_MLA_PREFILL_BATCH=1` were
enabled; direct attention and the mini head were disabled. All other baseline
settings and the llmprobe 0.6.13 bench-only protocol were retained.

| Context | Prefill before / candidate tok/s | Decode before / candidate tok/s |
|---|---:|---:|
| 2K | 968 / 972 | 41.3 / 41.7 |
| 4K | 435 / 771 | 41.2 / 40.9 |
| 8K | 323 / 653 | 36.7 / 35.2 |
| 16K | 285 / 577 | 34.5 / 32.1 |

These are the second, predictable-context requests, matched to the completed
baseline table above. Nonce-dependent input lengths differ by at most four
tokens. The unchanged 2K cell is stable; prefill improves substantially from
4K through 16K. Decode did not improve and was about 7% slower at 16K in this
run. This prefill candidate remains opt-in. The 1500/60 targets are unmet.

The standalone process completed all 27 requests, saved client JSON/HTML and
server timers, and released its server and GPU lock. Foreground server QoS,
exclusive lock, confirmed maximum fan spin-up and ten seconds idle were used.
A during-run sample reached 95.1 C. Measurement key:
`glm53-packed-headbatch-llmprobe-20261003`. A fresh 32K qualification is deferred
until the selector and verification-cache fixes are integrated, to avoid
repeating expensive long-context runs for intermediate candidates.

The next stacked candidate also exposes per-request index-score, verification
projection, and bounded draft-readout dispatch counts. The index scorer's new
bounded dot plane and copies add a conservative 8 MiB per pending layer to
admission (16 MiB at the current async2 prefill schedule). These settings remain
explicit diagnostic switches while the combined model pass is pending.

### Stacked optimization qualification through 32K

The selected 2.3bpw + A6 assistant stack completed llmprobe 0.6.13
bench-only, runs 1, reasoning default through 32K. Source `c6b609f4`,
MLX 0.32.3, N2/children4/async4/group2, BF16 compressed MLA and FP32
KDA; all prior baseline lane/down/R4/dense settings were retained.
Packed prefill, MLA head batching, NAX index scores, exact verification
MLA broadcast, horizon2 readouts and KDA leaf retention were enabled.

| Context | Prefill baseline / stacked tok/s | Decode baseline / stacked tok/s |
|---|---:|---:|
| 2K | 968 / 944 | 41.3 / 46.4 |
| 4K | 435 / 748 | 41.2 / 44.6 |
| 8K | 323 / 664 | 36.7 / 42.5 |
| 16K | 285 / 606 | 34.5 / 39.0 |
| 32K | 253 / 545 | 30.9 / 34.4 |

These are the predictable-context requests with 192 actual output tokens,
matched to the original completed baseline. Inputs differ by at most 13 tokens
from that baseline. The unchanged 2K prefill cell is slightly slower; long
prefill improves by more than 2x. Decode improves at every measured context.
At 32K the selected stack reaches 545 prefill and 34.4 decode tok/s; the
1500/60 targets remain unmet. Ordinary-context results and all 29 server
records remain in the artifact rather than being mixed into this table.

Actual dispatch counts confirmed the selected paths. KDA leaf hits were
96.9–100% on these predictable requests, above the component 20.6% break-even
rate. The 32K cell recorded 21120 packed, 14069 index, 704 query/704 value
verification calls and 64 bounded readouts, with 2176 KDA hits and zero misses.
The optional retention remains explicit while broader prompt coverage is open.

From 16K to 32K, milliseconds per speculative round changed: drafting
9.88→14.82, verification 60.85→64.13, replay 1.58→2.23 and commit 3.05→5.59.
Verification dominates absolute time; assistant drafting and assistant
context publication explain much of the remaining context growth. The
commit timer covers assistant clone/append/evaluation; target accepted-cache
updates are inside replay. Branch latent replacement is eliminated. See
[the ownership audit](glm5-dflash-accepted-commit-copy.md).

ReleaseFast full suite, final HTTP tests and CLI build passed. The 16K, 32-row
same-model prefill drift check matched all 32 top tokens, mean KL 0.00398871.
The benchmark used foreground server QoS, exclusive GPU lock, confirmed
maximum fans and ten seconds idle at a cool start. It completed with client
exit 0, stopped its owned server, released the lock and restored automatic fans.
Measurement key: `glm53-stacked-llmprobe-20261003`.

### Bounded assistant-block and affine projection candidates

`SUSHI_GLM_DFLASH_BLOCK_TAIL=1` uses only the visible sliding-window prefix
for the temporary eight-row assistant block; persistent assistant state keeps
its existing format. Shape-dependent assistant rounding is reported in the
[block-tail qualification](glm5-dflash-block-tail.md). It remains opt-in until
full target-token/state and proposal-acceptance checks pass.

`SUSHI_GLM_DFLASH_A6_HOIST=1` hoists repeated coefficient decoding for the
three-row, affine6/group128, 4096-to-8192 target projections. Production-bank
outputs were bit-identical. `SUSHI_GLM_KDA_PREFILL_CLUSTER=1` groups retained
FA/GA/beta products on 2048-row normal prefill, including compact output copies.
Prepared banks are evaluated during load and included in measured resident
memory; their actual byte count is exposed. Admission adds only the new joined
activation plane: 1.25 MiB per pending layer, 2.5 MiB for async2. Per-request
cluster, block-tail, and affine-hoist dispatch counters confirm engagement.

### Bounded draft tail, exact A6 hoist and prefill cluster

Source `f45103ef` completed the same llmprobe 0.6.13 bench-only protocol,
runs 1, reasoning default, with the three new opt-ins enabled on the previous
measured stack. The 32K rung is one final selected-candidate qualification;
normal iterations remain 2K–16K. Actual target tokens and complete final state
passed the separate 32K, 64-token serial gate.

| Context | Prefill before / candidate tok/s | Decode before / candidate tok/s |
|---|---:|---:|
| 2K | 944 / 965 | 46.4 / 47.9 |
| 4K | 748 / 774 | 44.6 / 46.7 |
| 8K | 664 / 681 | 42.5 / 45.3 |
| 16K | 606 / 628 | 39.0 / 43.5 |
| 32K | 545 / 566 | 34.4 / 39.4 |

These are second predictable-context calls with 192 outputs. Inputs differ
from the previous table by at most 7 tokens. The unchanged 2K prefill cell also
improved, so the small 2–4% prefill increase cannot be attributed entirely to
clustering. Decode improved at every rung; the32K gain is 14.5%, with measured
39.44 tok/s. The 1500 prefill/60 decode targets remain open.

At 16K/32K, milliseconds per speculative round were drafting 6.12/6.22,
verification 57.99/60.83, target replay 1.36/2.28 and assistant commit 3.11/6.23.
The bounded block makes drafting almost independent of context over these
rungs. Assistant accepted-context publication still grows with history; the
transaction keeps its full cloned buffers until successful evaluation.

The 32K measured cell confirmed 64 block-tail calls, 6528 exact affine-hoist
calls, 510 prefill cluster calls, and 100% KDA leaf hits. Prepared banks occupy
85 MiB; BF16 compressed MLA and FP32 KDA storage remain unchanged.
All 29 server requests/client JSON/HTML are saved. The owned server stopped,
GPU lock released, fans returned automatic, and client exited 0. Foreground
server QoS, confirmed maximum fan spin-up, exclusive lock and cool ten-second
idle were used. Measurement key: `glm53-tail-hoist-llmprobe-20261003`.
