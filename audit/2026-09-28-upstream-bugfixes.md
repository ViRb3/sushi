# Upstream correctness round

Base: sushi `ca3b84b1`. Upstream reference: ddalcu/mlx-serve `8222282a`.
Implementation branch: `codex/upstream-bugfixes`.

Implemented selective ports of #601 (owned SSD file accounting), #550 (failed-load
recovery on rescan), #552 (constrained-output logprobs), and #553 (affine KV dtype
through allocation, growth and dequantization). NOTICE and the corresponding
engine/API documents record the adaptations. No performance features are included.

## Validation

- All four new behavioral regressions failed before the production changes:
  SSD bytes 81326 instead of 165729; rescan remained in error; no constrained
  logprobs; f16 KV reconstructed as bf16.
- All four pass after the ports. Coverage additionally includes KV capacity
  growth, unchanged concurrently loading entries, and an error entry whose path
  differs from discovery.
- ReleaseFast executable builds successfully.
- Final full unit suite: **2724 passed, 93 skipped, 0 failures** (2817 total).
  The first full attempt had three error-latch failures and one scheduler crash;
  the DiskTier group and scheduler case passed in isolation, and the complete
  rerun passed. Unchanged main also passed its full suite: 2720 passed, 93 skipped.
- `tests/test_multi_model_dir.sh`: passed.
- `python3 tests/test_rescan_retry.py`: passed. The HTTP test uses a malformed
  tiny checkpoint and verifies error → unloaded plus refreshed disk bytes.
- `tests/test_no_local_paths.sh`, shell/embedded-Python syntax, and diff whitespace:
  passed.
- Live `tests/test_logprobs.sh` on Qwen3.8-Flash-Next-Sushi-3bpw: **71/71 passed**.
  All new JSON-object/JSON-schema checks passed for streaming and non-streaming
  content reconstruction and probability pairing. The first run reported 69/71:
  two legacy-completion assertions assumed the model would emit a token. A
  diagnostic rerun confirmed empty text, zero completion tokens, and a normal
  stop. The test now accepts that valid case only with those invariants, and
  still requires logprobs for emitted tokens. Legacy nonempty token-shape checks
  were not exercised by this checkpoint/prompt; the script reports that explicitly.

Each live run acquired the shared GPU lock under a `codex-upstream-*` owner and
released it when the script ended. No throughput claim or benchmark was made.
No main-branch merge or push was made.
