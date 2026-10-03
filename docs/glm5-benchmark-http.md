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

`SUSHI_GLM_DFLASH_COMMIT_WINDOW=1` requires the bounded block-tail arm. It crops
only the staged next assistant context before appending accepted features,
preserving absolute positions and the previous context until successful
publication. Ordinary cache growth gives five K/V pairs at 2560 rows, a 50 MiB
replacement bound. The per-request successful publication counter is recorded.
See [commit-window qualification](glm5-dflash-commit-window.md).

### Bounded assistant commit publication

Source `2f446c5a` completed the same 29-request llmprobe 0.6.13 bench-only
protocol with `SUSHI_GLM_DFLASH_COMMIT_WINDOW=1`. The separate selected 32K,
64-token gate retained exact target IDs and complete committed state.

| Context | Input IDs | Prefill tok/s | Decode tok/s | Assistant commit ms/round |
|---|---:|---:|---:|---:|
| 2K | 2036 | 947.64 | 46.85 | 0.95 |
| 4K | 4059 | 758.10 | 45.76 | 0.93 |
| 8K | 8225 | 665.03 | 44.20 | 0.93 |
| 16K | 16278 | 605.59 | 41.41 | 0.92 |
| 32K | 32747 | 541.33 | 39.88 | 0.97 |

These are the second predictable-context calls, 192 output IDs and 191 decode
forwards. At 32K, assistant commit fell from 6.23 to 0.97 ms per round; drafting,
verification and target replay measured 6.68, 64.68 and 2.28 ms. Verification
consumed 86.4% of decode time. The 64 successful bounded publications and all
other selected switches engaged. The replacement bound is now independent of
history length.

Overall rates did not improve consistently across rungs: verification and
prefill slowed between boots, masking the bounded commit saving. This run
qualifies the publication bound, not a general throughput improvement. The
1500 prefill/60 decode targets remain open. Predictable 2K has 2036 input IDs
and zero prefill cluster calls; ordinary 2K has 2072 IDs and 34 calls. Keep
these workloads separate when assessing the 2048-row cluster threshold.

The client exited 0, the owned server stopped, the exclusive GPU lock was
released and fans returned automatic. Binary/runtime/flags and all requests
are retained under measurement key `glm53-commit-window-llmprobe-20261003`.
The next candidates and their paired gates are described in
[the next-wave plan](glm5-next-wave-performance-plan.md).

### Bounded packed cadence and exact expert grid transpose

Source `f9f4c8d2` enables `SUSHI_GLM_PREFILL_CADENCE=1` and
`SUSHI_GLM_PREFILL_GRID_TRANSPOSE=1` on the preceding commit-window stack.
The first overlaps at most two unchanged 16-query packed attention graphs;
the second reorders physical expert-GEMM tiles for qualified full T2048 chunks.
Both remain opt-in. The full ReleaseFast suite/CLI build passed; the 8K,
32-output gate matched every target ID and complete final state, with 33
cadence calls and 168 expert-grid calls.

Same llmprobe 0.6.13 bench-only protocol, runs 1, reasoning default, second
predictable-context request and 191 decode forwards:

| Context | Before / new input IDs | Prefill before / new tok/s | Decode before / new tok/s | Prefill change |
|---|---:|---:|---:|---:|
| 2K | 2036 / 2037 | 947.64 / 931.71 | 46.85 / 47.08 | -1.68% |
| 4K | 4059 / 4061 | 758.10 / 802.82 | 45.76 / 45.48 | +5.90% |
| 8K | 8225 / 8225 | 665.03 / 727.81 | 44.20 / 44.44 | +9.44% |
| 16K | 16278 / 16274 | 605.59 / 655.06 | 41.41 / 43.04 | +8.17% |

The two new paths do not engage on the 2037-ID predictable 2K input, so its
small decline reflects the separate run rather than either implementation.
At 4K/8K/16K the cadence and grid counts were 11/42, 44/168 and 77/294.
The paired component tests independently support both changes; this HTTP run
measures their combined stack and does not divide its gain between them.
Decode code is unchanged by this wave, so its movement is not attributed to
the prefill candidates. The 1500 prefill/60 decode goals remain open.

Admission adds 64 MiB for the second packed tile per pending MLA layer, or
128 MiB at async2, above the original packed bill. Expert-grid ordering adds
no new arrays or weight copies. [Cadence qualification](glm5-prefill-cadence.md)
and [expert-grid qualification](engine-exl3-experts.md) give paired evidence.

All 27 server requests, flags, binary/runtime provenance and client JSON/HTML
are retained under key `glm53-prefill-wave-llmprobe-20261003`. The client exited
0; server stopped, GPU lock released and fans restored automatic. Foreground
QoS, maximum fans and a cool ten-second idle were used. The final selected 32K
qualification is recorded below; routine iterations stay at 2K–16K.

### Selected 32K qualification of the prefill wave

The single selected 32K llmprobe cell used the same flags and binary. Running
only this rung produced 33595 input IDs, versus 32747 in the inherited
full-ladder cell: 2.59% longer. Keep that count visible rather than treating
them as identical workloads.

| Measurement | Before | Prefill wave |
|---|---:|---:|
| Input IDs | 32747 | 33595 |
| Prefill tok/s | 541.33 | 604.18 |
| Decode tok/s | 39.88 | 42.39 |

The input-normalized prefill rate rose 11.61%, consistent with the independent
component evidence and the 4K–16K stack gains. Decode was unchanged by the
implementation; its separate-boot movement is not assigned to these changes.
The selected cell had 176 cadence calls and 672 expert-grid calls. Per-round
draft/verify/replay/commit times were 6.28/60.53/2.36/1.00 ms; verifier work still
dominates the remaining decode gap. The 2K-to-32K decode rates in this wave
were 47.08 and 42.39 tok/s; neither reaches 60.

A separate fixed-prompt 32K/64-output gate matched all target IDs and complete
final state against serial decoding from the same captured prefix. It engaged
165 cadence calls and 672 expert-grid calls, with decode peak 96.47 GB.
This correctness gate's committed-input-token rate is distinct from HTTP.
The ReleaseFast suite, CLI and standalone gate builds passed.

Artifact keys are `glm53-prefill-wave-32k-model-gate-20261003` and
`glm53-prefill-wave-32k-llmprobe-20261003`. All 21 HTTP requests/client output
are saved; exit 0, server stopped, GPU lock released and fans automatic.
No 64K/128K rung was run. Both exact candidates are accepted as opt-ins.


### Optional native B1/B3 decode attention

`SUSHI_GLM_DECODE_BATCH=1` selects mode-matched native B1 serial/replay and
B3 verification attention. It reserves 32 MiB at async4 and reports B1/B3
engagement. BF16 cache/output and FP32 accumulators remain; numerical drift
against the old scalar target is measured separately from speculative parity.
Default is off, and A6g128 remains the selected assistant.

Predictable 2K/4K/8K/16K decode measured 46.04/46.19/46.57/44.20 tok/s;
ordinary measured 42.16/41.33/42.09/41.71. Small-context performance is mixed.
Selected 32K used exactly 33595/33631 predictable/ordinary input IDs and
measured 608.77/614.65 prefill and 43.00/38.08 decode tok/s. The inherited
32K scalar-attention rates were 42.39/37.42 decode. These small separate-boot
changes are not assigned wholly to the attention kernel.

[Native attention qualification](glm5-decode-attention-batch.md) records the
14.68% paired complete-component win, exact B1/B3 outputs, strict 8K64 serial
token/state gate, old-target mean KL0.00323446 and full warmed tables.
Artifacts `glm53-native-decode-llmprobe-20261003` and
`glm53-native-decode-32k-20261003` ran at `39f8693d` plus hashed WIP, binary
`e825e47bb24c5697`, MLX0.32.3, foreground QoS and exclusive locks. Both exited
0, with server stopped, lock released and fans auto. Full ReleaseFast suite
and CLI passed. This opt-in does not attain the 1500/60 goals.

### Optional packed B32 prefill

`SUSHI_GLM_PREFILL_PACKED32=1`, together with packed attention and prefill cadence,
combines two unchanged T16 selectors into one native B32 gather/attention call.
Scoring modes and pool ordering stay unchanged. Exactly 32 rows use the new
batch; every smaller remainder uses fragments of 16 or fewer. Cold dense prefill
and native B1/B3 decode are unchanged. The default remains B16.

The conservative async2 reserve increases by 256 MiB: 128 MiB per B32 graph,
two graphs per layer, two pending layers. Metadata reports `prefill_packed32`
and `sushi_diagnostic.packed32_attention_calls`; zero calls at a small/dense rung
do not establish engagement. All original cache/activation/selector bills remain.

The [component and model gate](glm5-packed32-result.md) records exact ordered IDs,
BF16 outputs and full 16K prefix/64-token continuation state. Whole-attention time
fell 15.38%; matched model prefill latency fell 4.93%, versus 1.95% control drift.
This is an accepted memory/performance opt-in; the pinned HTTP 2K–32K
qualification below is complete. It does not claim 1500 prefill or 60 decode tok/s.

Pinned llmprobe 0.6.13 ran bench-only, one measured request per cell, with the
accepted target plus A6g128, native B1/B3, N2/children4/async4, chunk2048/async2,
greedy sampling and prefix cache off. All ten measured cells returned 192 IDs.
Rates use server timers: actual input count / prefill time, and 191 post-prefill
IDs / decode time. The arrows compare the inherited native-mode boot with the
packed32 boot; timings and exact output-ID arrays are retained privately.

| Rung | Workload | Actual input tokens | Prefill tok/s, inherited → B32 | Decode tok/s, inherited → B32 | B32 calls | Native B3 calls | Rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | Ordinary | 2073 | 861.36 → 893.75 | 42.16 → 46.11 | 0 | 770 | 70 |
| 2K | Predictable | 2037 | 930.49 → 986.45 | 46.04 → 49.60 | 0 | 704 | 64 |
| 4K | Ordinary | 4097 | 748.72 → 862.62 | 41.33 → 44.10 | 704 | 803 | 73 |
| 4K | Predictable | 4061 | 782.74 → 861.46 | 46.19 → 48.76 | 682 | 715 | 65 |
| 8K | Ordinary | 8261 | 725.45 → 796.55 | 42.09 → 45.20 | 2134 | 759 | 69 |
| 8K | Predictable | 8225 | 727.32 → 799.98 | 46.57 → 48.85 | 2123 | 704 | 64 |
| 16K | Ordinary | 16310 | 655.77 → 737.88 | 41.71 → 44.39 | 4895 | 748 | 68 |
| 16K | Predictable | 16274 | 654.76 → 735.43 | 44.20 → 47.19 | 4884 | 704 | 64 |
| 32K | Ordinary | 33631 | 614.65 → 664.19 | 38.08 → 40.10 | 10846 | 814 | 74 |
| 32K | Predictable | 33595 | 608.77 → 659.75 | 43.00 → 46.62 | 10835 | 704 | 64 |

These are separate-boot comparisons, with no rerun of the inherited baseline.
Current minus inherited input counts are +1/+2/0/−4 at 2K/4K/8K/16K for both workloads;
32K counts match. The 2K dense cells made zero B32 calls, so their movement is
not evidence for packed32. Decode arithmetic is unchanged; its movement is not
assigned to this prefill change. Mode-matched token/state parity is established by the separate model gate;
these HTTP comparisons do not establish cross-boot output parity.

Both jobs report MLX 0.32.3, BF16 MLA caches, FP32 KDA state, packed attention,
prefill cadence and packed32 enabled, together with the accepted projection,
NAX scoring, grid, KDA, assistant and native-decode flags. Post-job active memory
was 94,548,862,200 bytes in both boots, against a 115,448,725,504-byte admission
limit. HTTP does not expose an allocator peak; the matched gate's peaks and
conservative async2 increment remain documented separately. Final per-request
cache-growth reservations and complete settings are retained in diagnostics.

The server allowed benchmark ignore-EOS, but the measured pinned-client cells
reported `ignore_eos:false`; all nevertheless emitted the requested 192 IDs.
The private passive diagnostics subscriber retained final response diagnostics
without modifying requests, transport or the pinned client bundle. It copied
response chunks in memory and appended once per final response; that small
client overhead was not isolated. Request bodies were iterable and unavailable
to the subscriber. `/tokenize` and final usage supply actual counts, not input
ID arrays; no raw-body recovery or transport changes were made.

Artifacts `glm53-packed32-llmprobe-20261003` and
`glm53-packed32-32k-20261003` preserve 27 and 21 final responses respectively,
client JSON/HTML, output IDs, exact server times, counters, settings and telemetry.
The frozen original-path CLI was built from accepted runtime `af51e72f`, source
checkpoint `d1ba5b8c`, SHA256
`bbd0125a483c93f2f809a153c20be7ba6df8743ef9b9da1dc6225cc5081167d6`,
verified before and after both jobs. Later source WIP does not describe this
binary. Both clients exited 0; own servers stopped, GPU locks released and fans
restored to auto. Foreground QoS, maximum-fan confirmation and required idle
were used. No baseline rerun, 64K or 128K rung was run.
