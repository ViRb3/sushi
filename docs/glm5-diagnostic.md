# Native GLM diagnostic runner

`glm5_diagnostic.zig` provides an opt-in test harness around the native GLM model.
It does not register GLM with the public server. The test is skipped unless
`SUSHI_GLM_DIAGNOSTIC_MODEL` is set. Run a full model only in an exclusive GPU slot,
after the tiny native layer and model lifecycle fixtures pass.

The dedicated loader requires the checkpoint's index and config. The gated runner validates EXL3 shard stamps and complete bank geometry before
loading arrays. The loader reads only index-owned text tensors, excludes vision and MTP (including layer indices beyond
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
| `WARMUP_DECODE` | Default 1; use 8 near the sparse-attention boundary to compile that path before timing |
| `PROFILE` | Default 1; record time per model layer |
| `MEMORY_GIB` / `CACHE_GIB` | Default 110 / 2 GiB |
| `WIRED_GIB` | Optional; cannot exceed the reported recommended working set |

A smoke run can use prefill 8, decode 4, chunk 8, warmup 0. A 512/64 measurement
should use the same source-compatible prompt IDs as the reference run. The
runner rejects short input rather than silently repeating it. It records the
exact input/output IDs and decoded output text for coherence inspection.

Warmup executes all requested prefill chunk shapes and the configured decode ticks, then
resets request state. There is no prefix reuse. Load/bind, warmup, prefill and
decode time are separate. Progress goes to `OUT.progress.json`; the harness emits
no success diagnostics to stdout/stderr. The normal test runner may print its
own test status when invoked directly.

The first generated token comes from prefill logits. Therefore 64 generated
tokens entail 63 timed decode forwards; both counts and the rate denominator are
explicit in JSON. Timing excludes progress-file writes and final text decoding.
Generation is greedy and fixed length; it records the first EOS and whether the
fixed-length diagnostic continued beyond it. This is a timing/coherence diagnostic,
not a public chat completion. MTP is off. Compressed MLA cache uses BF16 by default and recurrent
KDA state is FP32. This is the GLM cache policy for future serving too; generic KV8
and `--fast` KV8 defaults must not silently change it. Active/peak MLX memory, stored tensor bytes, and applied limits
are recorded independently.

Tiny regression tests cover dtype/value preservation, indexed ownership,
vision/MTP exclusion, missing indexed tensors, and exact prompt-ID selection.
The full-model test is a separate gated run, not part of those fixture results.

Decoded output is a JSON string when its bytes form valid UTF8. A fixed-length
run can end inside a multi-byte character; in that case `output_text` is null,
`output_text_utf8_valid` is false, and `output_bytes` preserves the exact bytes
as a numeric JSON array. Valid output also includes the raw byte array.

For serial tuning, the diagnostic also honors the existing `SUSHI_WIRED=fit|off|max`
policy after warmup/reset and before timing. It records the actual timed wired limit
and policy. An explicit diagnostic `WIRED_GIB` and `SUSHI_WIRED` cannot be combined.
The default remains the explicit recommended-cap limit used by the first measurement.

`SUSHI_GLM_DIAGNOSTIC_DECODE_ASYNC=1` selects async submissions every four layers for
one-token decode, with one final logits-and-cache evaluation. Set it to0 for the
synchronous comparison. `PROFILE=1` forces synchronous evaluation so per-layer
timings remain meaningful; the report names the effective schedule. Prefill keeps
its per-layer memory boundary in either mode.

`SUSHI_GLM_DIAGNOSTIC_DENSE_PREFILL=1` opts into the reference-style expanded-K/V
short-prefill attention experiment. It is off by default and retains absorbed
attention for decode and calls outside the dense-prefix eligibility boundary.
Changing BF16 rounding boundaries requires independent reference and quality
validation; it is not a byte-preserving scheduling change.

`SUSHI_GLM_DIAGNOSTIC_PREFILL_ASYNC=1` enables bounded asynchronous prefill.
`SUSHI_GLM_DIAGNOSTIC_PREFILL_SYNC_LAYERS` selects the host-wait interval, from 1
to 8 layers; the default remains 2. Larger intervals are experimental and must
be measured for throughput and peak memory. Asynchronous prefill is off by
default; profiling keeps synchronous layer boundaries. The final logits and all cache outputs settle before return. The
report records the effective prefill schedule, host-wait interval and successful copy-free QKV
dispatch count during the timed phase, excluding warmup.

`SUSHI_GLM_DIAGNOSTIC_COMPONENTS=1` adds synchronized per-component attribution and
forces the synchronous schedule, even with `PROFILE=0`. It reports separate prefill/decode
nanosecond totals for each layer's HC collapses, branch norms, attention, expansions,
dense FFN, routing, routed experts, shared expert and final FFN addition. Attention
measurements settle all cache side outputs. Input embedding work settles before the first
HC timer, and final head work remains outside the component totals. These timings include
host graph construction and synchronization; they are attribution evidence, not async
throughput or pure GPU kernel duration. With this option off, no component barriers run.
A nonzero fixture checks identical logits/cache state, synchronous dispatch and counter reset.


The first component run (512 prefill, 63 decode forwards, unchanged 64 output IDs) found
323.14 ms in routed experts, 203.46 ms in KDA attention, 37.33 ms in MLA attention,
77.00 ms in the two HC collapse groups, and31.33 ms in shared experts during prefill.
Decode contains a large per-evaluation floor: even the final FFN addition costs about
0.14 ms under this method. Use these totals to prioritize experiments; use queued paired
microbenchmarks and full-model runs to establish gains, rather than treating synchronized
component totals as pure GPU times or subtracting an assumed universal barrier constant.


For a separate untimed routing capture, `SUSHI_EXL3_UNION_HIST=1` enables per-expert
prefill counts on stderr and marks the JSON report. It forces route evaluation and logging;
never enable it in a throughput arm. Use `WARMUP=0`, one full prefill chunk and minimal decode
when capturing a single set of layer histograms for a replay microbenchmark.


For same-binary full-model attribution, `SUSHI_GLM_HC_FUSED=0` disables the one-token
HC candidate, and `SUSHI_EXL3_CLAMPED_MIDDLE=0` disables the fused clamped middle/down
candidate. HC is off by default and can be enabled with `SUSHI_GLM_HC_FUSED=1`. The middle/down
auto mode retains the original serial path and selects the candidate only for eligible short
multirow calls; explicit1 permits serial experiments. These controls distinguish
warm component results from full-model streaming behavior; check the respective engagement
counters. The initial combined arm preserved output IDs but did not show a decode gain.
