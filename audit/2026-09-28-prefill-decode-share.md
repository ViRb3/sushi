# Prefill/decode share (#568)

Base: main `78537962`. Branch: `codex/prefill-decode-share`.
Only mlx-serve #568 is included in this round.

- `--prefill-decode-share` overrides `SUSHI_PREFILL_DECODE_SHARE`; default 0.
- Share zero keeps sushi's existing one tick per prefill boundary. Nonzero share
  budgets hosted decode as `chunk_time * share / (1-share)`, stopping when decoders
  finish or the prefilling request is cancelled. Values above 0.9 clamp.
- Live decoders cap the initial base chunk at 1024 after explicit/env chunk picks.
  Admission still bills the original width; adaptive widening keeps its memory
  confirmation. Normal tail merging is unchanged.
- The interleave kill switch disables the effective share and its cap; `/props`
  reports that effective value. CLI startup resolves the flag before worker threads.

## Validation

The initial five focused regressions failed before implementation (parsing,
width cap, time budget, tick loop and `/props`) and passed after it. Additional
coverage checks flag precedence, finished/cancelled slots, overflow saturation,
the kill switch and prompt cancellation. ReleaseFast build succeeds.

The targeted live run used `INTERLEAVE_SHARE_ONLY=1`, an explicit 8192 chunk,
Sushi-2.6bpw, kv8, MTP off, context 32768, two slots and no prefix cache. Two probes
per arm, with a GPU lock for each server boot. It was a functional check with
other interactive work present, not a quiet throughput benchmark.

| Probe | Share 0: maximum gap | Share 0.5: maximum gap |
|---|---|---|
| 1 | 6023 ms | 966 ms |
| 2 | 5948 ms | 896 ms |

All four decoder output hashes match. Hosted decode was 0.5% of the competing
prefill wall time at share zero and 32.5% at share 0.5. The decoder finishes before
the prefill, so the target share does not apply to its remaining solo work.
Saved `/props` values were 0 and 0.5. Invalid CLI shares (`nan`, negative, text)
were refused by name. Shell syntax and repository path-hygiene checks passed.

Final full suite: **2743 passed, 93 skipped, zero failures** (2836 total), including the cancellation guard. No additional benchmark sweep was run.
