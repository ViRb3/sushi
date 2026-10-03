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
  --rungs 2k,4k,8k,16k,32k,64k,128k --runs 1
```

Run that full ladder once for the selected runtime/backend baseline. Subsequent
optimization iterations use only `--rungs 2k,4k,8k,16k` with the same settings.
The caller must hold the GPU lock for the loaded model and benchmark, restore
foreground QoS, pause competing compute and follow the fan/cooldown protocol in
[process-measurement](process-measurement.md). Record binary/runtime stamps and
raw llmprobe artifacts. Old 1f8 runtime measurements are not a same-runtime
control for the v0.32.3 baseline.

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
