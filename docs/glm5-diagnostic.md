# Native GLM diagnostic runner

`glm5_diagnostic.zig` provides an opt-in test harness around the native GLM model.
It does not register GLM with the public server. The test is skipped unless
`SUSHI_GLM_DIAGNOSTIC_MODEL` is set. Run a full model only in an exclusive GPU slot,
after the tiny native layer and model lifecycle fixtures pass.

The dedicated loader requires the checkpoint's index and config. It loads only
index-owned text tensors, excludes vision and MTP (including layer indices beyond
`num_hidden_layers`), and preserves BF16, FP32, F16 and integer storage exactly.
Filtering occurs before tensor evaluation. The report counts actual retained
array bytes and records the configured expert rate/window; the directory name is
not used as a memory or quantization estimate.

Build the gated test with:

```sh
zig build test -Doptimize=ReleaseFast -Dtest-filter='GLM native diagnostic' --verbose
```

With the model environment unset this compiles and skips the real-model test.
The verbose output identifies the generated `test` executable. Invoke that binary
directly with the environment below to avoid build-run cache ambiguity. Follow
[measurement policy](process-measurement.md) for locking, foreground QoS, cooling,
and binary provenance.

| Environment variable prefix `SUSHI_GLM_DIAGNOSTIC_` | Value |
|---|---|
| `MODEL` | Required absolute checkpoint path |
| `OUT` | Required result JSON path; parent directory must exist |
| `TOKENS_FILE` | JSON ID array or object containing `ids`; takes priority over text |
| `PROMPT_FILE` | Raw text, used only when no tokens file is supplied |
| `PREFILL` | Default 512; first N supplied tokens, without repetition or templates |
| `DECODE` | Default 64 generated tokens; minimum 2 |
| `CHUNK` | Default 128 prefill tokens per chunk |
| `WARMUP` | Default 0; use 1 for a warmed measurement |
| `PROFILE` | Default 1; record time per model layer |
| `MEMORY_GIB` / `CACHE_GIB` | Default 110 / 2 GiB |
| `WIRED_GIB` | Optional; cannot exceed the reported recommended working set |

A smoke run can use prefill 8, decode 4, chunk 8, warmup 0. A 512/64 measurement
should use the same source-compatible prompt IDs as the reference run. The
runner rejects short input rather than silently repeating it. It records the
exact input/output IDs and decoded output text for coherence inspection.

Warmup executes all requested prefill chunk shapes and one decode tick, then
resets request state. There is no prefix reuse. Load/bind, warmup, prefill and
decode time are separate. Progress goes to `OUT.progress.json`; the harness emits
no success diagnostics to stdout/stderr. The normal test runner may print its
own test status when invoked directly.

The first generated token comes from prefill logits. Therefore 64 generated
tokens entail 63 timed decode forwards; both counts and the rate denominator are
explicit in JSON. Timing excludes progress-file writes and final text decoding.
Generation is greedy and fixed length; it records the first EOS and whether the
fixed-length diagnostic continued beyond it. This is a timing/coherence diagnostic,
not a public chat completion. MTP is off. Attention cache is BF16 and recurrent
KDA state is FP32. Active/peak MLX memory, stored tensor bytes, and applied limits
are recorded independently.

Tiny regression tests cover dtype/value preservation, indexed ownership,
vision/MTP exclusion, missing indexed tensors, and exact prompt-ID selection.
The full-model test is a separate gated run, not part of those fixture results.
