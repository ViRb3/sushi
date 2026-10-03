# Native GLM teacher-forced KLD comparison

`src/glm5_kld.zig` is a gated diagnostic, not a serving entry point. It compares
one native Sushi checkpoint with the completed four-prompt standard-v1 teacher
fixture. Run the two students in separate processes against the same fixture:
A6g128 trunk in the directory named2.3bpw and A8g128 trunk in2.4bpw, both with
K2.25/W12 experts. Neither comparison loads DFlash or MTP.

## Position and metric contract

Prompt IDs and generated IDs come directly from `prompt_tokens.txt` and
`generated_tokens.txt`; prompt text is never re-tokenized. Teacher row0 predicts
generated token0 after the complete prompt. Rowp>0 is scored after feeding teacher
tokenp−1. The final generated token is scored but not forwarded; the final request
offset is prompt length plus generated length minus1. Each prompt starts with an
empty request. Native prefill uses dense-prefix attention, async2 scheduling and
the teacher's512-token chunk size. Decode uses the ordinary native async4 path,
BF16 compressed MLA cache and FP32 KDA state.

Scoring reuses `kld.scoreRow`: full-vocabulary KL(teacher‖student), top1 agreement
with the teacher's greedy token, student NLL of that token, raw-logit cosine
similarity and cosine loss. Native logits are widened to F32 without another
quantization. All metrics are accumulated as row sums, then divided by scored
positions; category and overall results are position-weighted rather than averages
of prompt means. Per-position KLD is retained for every prompt.

Both all512 rows and the first-EOS-inclusive subset are reported. EOS IDs combine
the target configuration with `<|im_end|>` only when the tokenizer maps it to one
token, matching the existing KLD method. With no EOS, the subset contains all rows.
The reported first-EOS position is zero-based. Code and prose categories each contain
two prompts; this is not the standard sixteen-prompt release verdict. The teacher
uses the oMLX streamer, so the unmeasured teacher/student engine floor is explicitly
part of the quality scope.

## Preflight and provenance

Before loading model tensors, the runner requires completed `baseline.json`, four
unique code/prose prompt records,512 generated IDs per prompt and matching declared
lengths. It checks every token against the student vocabulary, exact F32 file sizes,
finite/nonzero teacher rows and greedy row/token alignment. Nonzero `top_k` metadata
is allowed: it describes an optional summary, while `logits.f32` must contain the
entire vocabulary. Missing/partial fixtures never start inference.

`identity.json` must declare512-token source chunks,512 continuations, BF16 cache
and FP32 KDA state. Student and teacher tokenizer SHA-256 must match. The report
includes teacher identity, source revision, caller-supplied binary SHA, student
config/index/tokenizer hashes, baseline/teacher-identity hashes, per-prompt ID-file
and logits hashes, variant flags and memory measurements. Model shard identity uses
inode/size/mtime metadata plus the index hash, not a full digest of all weight bytes;
it is rechecked before publishing a result.

Outputs are atomically renamed into place only when every prompt succeeds. Progress
uses a separate `.progress.json`. No partial KV or recurrent state is resumed. An
existing complete result can be reused only when its input/source fingerprint,
schema and complete row counts match; otherwise it is left intact and the runner
returns `GlmKldOutputExists`.

## Invocation

Build a ReleaseFast test binary filtered to `GLM KLD real teacher comparison`, then
run that fixed binary directly. Changing only environment variables must not rely
on Zig's cached test-run step. The invoking coordinator owns the exclusive GPU lock
and waits until teacher capture has released it.

Required environment variables:

- `SUSHI_GLM_KLD_MODEL`: one student checkpoint directory.
- `SUSHI_GLM_KLD_FIXTURE`: completed teacher directory.
- `SUSHI_GLM_KLD_OUT`: final JSON path.
- `SUSHI_GLM_KLD_SOURCE_REV`: source revision used for that binary.
- `SUSHI_GLM_KLD_BINARY_SHA256`: actual fixed executable hash.
- `MLX_ENABLE_TF32=0`: matches the teacher's arithmetic setting.

For the currently qualified path, set `SUSHI_GLM_LANE_PAIR=1` and
`SUSHI_GLM_DOWN_LANE=1` explicitly. The runner reports these flags and does not change
public cache defaults. Memory limit is110GiB, MLX cache limit2GiB, and wired limit is
the smaller of110GiB and the device's recommended working set.

## Tests

CPU tests cover weighted aggregation, inclusive EOS, exact fixture-length and
row/token validation, and completed-result reuse. A tiny nonzero native model test
constructs independent teacher rows, then checks self-KLD, row count, EOS count and
every final cache array after replay. ReleaseFast compile-only validation covers
that GPU test while the teacher owns the GPU; execution is deferred until release.
No full-model student result exists until the completed fixture is scored.
