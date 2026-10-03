# ZCode coding-agent launcher

ZCode repository v3.14.3 (CLI package 0.16.9) uses Sushi through the standard OpenAI Chat Completions API. Install
ZCode and put its `zcode` executable on PATH, then select any chat model advertised
by the running server:

```sh
sushi launch zcode --model MODEL_ID
sushi launch zcode --url http://127.0.0.1:12345 --model MODEL_ID -- --prompt "Inspect this repository" --mode build --output-format json
sushi launch zcode --model MODEL_ID --print
```

`--model` must match an advertised `/v1/models` ID. Without it the launcher picks
the first loaded chat model, else the first chat row. Media and embedding rows
are excluded. There are no model-name or architecture checks in this launcher:
native and EXL3 chat engines use the same path. A diagnostic-only model, including
GLM before its public serving integration, must first become an advertised chat
model.

The launcher writes schema-version-1 `~/.sushi/zcode/provider_config.json` and
exports `ZCODE_PERSONAL_PROVIDER_CONFIG_FILE`, `ZCODE_DATA_BASE_DIR` and
`ZCODE_STORAGE_DIR` into the dedicated `~/.sushi/zcode` tree. Each launch refreshes
the model snapshot and selected default. ZCode keeps its normal project configuration
and skills. The integration requires no change to ZCode source.

The wire is `/v1/chat/completions`, with SSE reasoning, standard function calls,
and tool-result replay. Each model gets the server's advertised context and output
limit `clamp(context / 2, 1024, 65536)`. Older rows fall back to 32768 context and
8192 output tokens. ZCode maps its options to `max_tokens` and `reasoning_effort`.
Where advertised, its reasoning picker contains exactly the model's
`reasoning_efforts`; selection prefers medium, then another supported thinking
level. Older servers use none/low/medium/high. Vision follows the model row;
PDF, video, audio, native web search and structured JSON output are not advertised.

The CPU-only test `python3 tests/test_zcode_launch.py --bin zig-out/bin/sushi`
checks arbitrary model IDs, model filtering, budgets, selection and argument
quoting. Add `--zcode /absolute/path/to/zcode.cjs` for the real-client fixture:
streamed reasoning, fragmented calls to two Read tools, both file reads, replayed
call IDs/results and the final answer. Fixture model names verify transport, not
GPU inference quality or architecture availability.
