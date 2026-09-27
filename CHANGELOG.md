# Changelog

sushi began as a fork of [mlx-serve](https://github.com/ddalcu/mlx-serve) and was detached from it on 2026-09-17 at
mlx-serve commit `ef5e667` (two commits after mlx-serve v26.9.4). This file covers sushi's own changes since then;
earlier history is mlx-serve's, in that project's changelog.

## Unreleased

- **Sushi-2.6bpw for 64 GB Macs**: a new Qwen3.8-Flash-Next pack with 43.95 GiB of weights, 5.4 GiB less than
  Sushi-3bpw, so a 64 GB Mac serves 250k tokens of context at 8-bit KV.
- **Faster Sushi-2.6bpw**: its experts now read through the same fast kernels as Sushi-3bpw's, so it decodes 18% and
  reads prompts 12% faster on an M5 Max, level with Sushi-3bpw, with output unchanged.
- **Faster MiMo-V2.6-Flash**: the shipped Sushi-2.25bpw pack's experts now take MiMo's fast decode kernels, so it
  decodes about 70% faster on an M5 Max (35 to 61 tokens per second with MTP), with output unchanged.
- **Faster file rewrites and edits on Qwen3.8-Flash-Next**: when a reply copies text already in the conversation (a
  file returned with an edit, a tool call carrying a file), speculative decoding drafts from that text instead of the
  MTP head: 16-21% faster file edits and 9-11% faster file-writing tool calls on an M5 Max, 18-24% on edits deep in a
  long context, with greedy output unchanged; `SUSHI_MTP_LOOKUP=0` turns it off. Ported from mlx-serve, thanks @STRML.
- **Long conversations stay in the prompt cache**: without `--prefix-cache-mem`, the RAM prompt cache holds a whole
  conversation at the working context where memory allows (never less than 2 GB, and never at the expense of
  Flash-Next's n-gram table), so later turns of a long Flash-Next or MiMo chat reuse it instead of re-reading it.
  Ported from mlx-serve #575, thanks @STRML.
- **Very long sessions restore from the SSD cache**: a session of about 250k tokens or more no longer fails its SSD
  restore, and a restore that does fail falls back to reading the prompt instead of failing the request. Ported from
  mlx-serve #527, thanks @brandondyal.
- **A short chat no longer copies a long one's cache**: a chat that starts like a longer cached one copies only
  what it needs, so it no longer holds the long session's memory or pushes it out of the cache. Ported from
  mlx-serve #492, thanks @celestial-rose for the report.
- **Long Flash-Next sessions keep their cache when memory is tight**: a warm turn that cannot fit a second copy of
  its cached conversation takes the cache over instead of being refused. Ported from mlx-serve #518.
- **Repeated one-token prompts**: a one-token prompt that matches its own cached entry is read again instead of
  restored with nothing left to process. Ported from mlx-serve #518.
- **SSD restores need half the memory**: restoring a long session from the SSD cache no longer holds two copies of
  it for a moment.
- **Sessions past `--prefix-cache-mem` keep a cached prefix**: a Flash-Next conversation longer than the RAM prompt
  cache keeps the longest prefix that fits instead of being re-read in full every turn (a regression since v1.0.3).
- **Long prompts after a busy moment**: a long prompt that fits once the RAM prompt cache is emptied is no longer
  refused while the GPU is still finishing earlier work.
- **Faster Flash-Next MTP decode**: speculative decoding reads the hyper-connection weights once for a group of
  draft tokens instead of once per token, on the M5 as on earlier Macs, with identical output.
- **Faster Flash-Next decode**: each linear-attention layer now decodes and verifies MTP drafts in two GPU dispatches
  instead of three, about 2% faster decode on an M5 Max with identical output (ported from mlx-serve #517).
- **Leaner Flash-Next decoding**: the sparse-attention indexer updates its block keys in one GPU kernel instead of a
  chain of about ten, and the output is token-for-token the same.
- **`--preserve-thinking on|off`**: choose whether Qwen3.8 keeps every turn's thinking in the prompt (the default) or
  only the latest turn's, per model in model-settings.json or per request with `chat_template_kwargs.preserve_thinking`.
- **Continuing a reply with thinking on**: a continued assistant reply on Qwen3.8 and MiMo now resumes after the
  closed think block their templates write, instead of a malformed or missing one.
- **Files written through MiMo tool calls keep their last line break**: a tool argument's leading and trailing
  newlines now reach the client unchanged on MiMo.
- **Prompt reuse across Codex and Claude Code turns on MiMo**: a system or developer message sent mid-conversation now
  stays where it was sent, so the next turn still reuses the cached prompt instead of processing it all again.
- **Forced tool calls**: `tool_choice` `required` (Anthropic `any`) or a named function now makes Qwen3.8 and MiMo
  call one of the declared tools on every API, after their thinking when it is on, and naming a function missing from
  `tools` is a 400.
- **Streamed thinking matches the non-streamed reply**: a thought's trailing line break no longer rides out on the
  stream, so streamed and non-streamed reasoning are the same text on every API.
- **`sushi run` and `sushi pull` name the Sushi packs**: `qwen3.8-flash-next` fetches Sushi-3bpw, with tags `:2.6bpw`
  and `:4bpw`; the short names for models sushi does not serve are gone.

---

## v1.0.4 — Faster prompts and browser chat

- **Faster prompts on M1–M4**: Macs without the M5's neural accelerators read prompts about 4x faster on the EXL3
  packs (M2 Max, 3–4k-token prompts, default settings).
- **Faster Flash-Next prompts on 64 GB Macs**: when the n-gram table cannot stay in memory beside the model, prompt
  processing reads it in parallel by default instead of one row at a time.
- **Smaller contexts need less free memory to load**: resident Flash-Next EXL3 packs without separate sidecars or
  ANE now size load headroom from the chosen context instead of always asking for 7 GB above the weights.
- **Chat in your browser**: `sushi serve` and `sushi run` serve a chat page at `http://127.0.0.1:12345/` that streams
  replies, shows the model's thinking, takes images for vision models and keeps your conversations in the browser.
- **`/cd <folder>` in `sushi run`**: moves the folder the file tools and relative `/image` paths read from, the prompt
  always shows that folder and whether tools are on, and a model that asks for a file outside it now suggests `/cd`.

---

## v1.0.3 — Hotfixes

- **Prompt cache**: re-packing a model in place no longer restores stale SSD cache entries, and the RAM cache stays
  within its cap when every entry is in use.
- **Video and labels**: multi-part videos that fit are no longer refused, and `/v1/models` names an EXL3 pack's
  expert rate beside its dense width.

---

## v1.0.2 — Hotfixes

- **Stability**: two cached conversations can no longer share a prompt-cache key, a failed long-context cache copy no
  longer frees memory twice, and very low temperatures behave the same with and without MTP.
- **Memory and loading**: two resident models no longer over-commit memory at admission, and packs with invalid EXL3
  rate stamps are refused by name.

---

## v1.0.1 — Hotfixes

- **Tool calls and streaming**: streamed tool calls are always valid JSON, the reasoning budget applies to tool replies
  and Anthropic streams, and a disconnected Responses request is no longer stored as completed.
- **Prefix cache and loading**: fixes for SSD cache restores and failed cache writes, and malformed pack configs are
  refused by name.

---

## v1.0.0 — Qwen3.8-Flash-Next, sushi-packed

![Sushi-3bpw decode and prefill from 4k to 1M tokens on an M5 Max](https://raw.githubusercontent.com/beamivalice/sushi/main/docs/assets/perf-sushi3bpw-1m.png)

- **Two Qwen3.8-Flash-Next packs, EXL3 experts**: Sushi-3bpw (49.3 GiB, for 64 GB Macs) and Sushi-4bpw (63.7 GiB, for
  96 GB and up). At about 50 GiB, Sushi-3bpw has half the KLD of mlx-serve's iQ-MLX 3.3bpw; Sushi-4bpw matches oMLX
  oQ5e's quality in 20 GiB less memory.
- **Up to 1M tokens of context on one Mac**: 94 tok/s decode and about 1,900 tok/s prefill on an M5 Max, still 57 tok/s
  at 1M, with the 8-bit KV cache and the model's own MTP draft head on by default. `--mtp-typical 0.2` makes sampled
  decoding 15-20% faster.
- **Images in every API**: OpenAI chat and Responses, Anthropic messages and tool results all carry images to the
  vision tower, each where it was sent, and a large image needs about half the memory it did.
- **A drop-in local server**: OpenAI- and Anthropic-compatible HTTP on `127.0.0.1:12345`, clear of mlx-serve's 11234.
  `sushi run` chats in the terminal with read-only web and file tools, `sushi launch` sets up Claude Code, pi, omp,
  opencode and codex, and one thinking-effort vocabulary (off to max) works across every API.
- **Memory you can plan**: a load that would not fit is refused by name, concurrent long prompts wait instead of
  crashing, an unload answers once its memory is free, and `/v1/models` reports the real resident size.
- **Install with curl**: one ad-hoc signed binary for Apple Silicon on macOS 26.2 or later, with no Python at serve time.

---
