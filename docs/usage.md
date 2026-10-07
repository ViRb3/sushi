# Launch and agent usage

[Back to the README](../README.md)

## Install

```bash
brew install beamivalice/tap/sushi
```

Or the release binary (ad-hoc signed; curl does not quarantine it):
```bash
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz
```
A browser download is quarantined by macOS: clear it with `xattr -dr com.apple.quarantine sushi-macos-arm64`.
GPU memory and the largest context per Mac for each pack: [README, Memory](../README.md#memory).

## Recommended launch

**48 GB Mac, Sushi-2bpw**

Set the GPU memory limit first (it resets at reboot):
```bash
sudo sysctl iogpu.wired_limit_mb=43000
```

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-2bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2bpw

# images, 8-bit KV, 128k context
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 131072 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

**64 GB Mac, Sushi-2.6bpw**

Set the GPU memory limit first (it resets at reboot). 59,000 MB is the ceiling for this box: above it macOS runs out
of memory before the model does, and the kernel panics rather than the server refusing.
```bash
sudo sysctl iogpu.wired_limit_mb=59000
```

Then pick one of the two. They differ only in KV width; the context is set explicitly because auto-context reads free
memory, so its answer is not the same on two 64 GB machines.

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-2.6bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2.6bpw

# 1. images, 8-bit KV — the default quality
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2.6bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 250000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1

# 2. images, 4-bit KV — 1.8 times the context, at 8% KLD and 0.2 points of next-token agreement
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2.6bpw \
  --mtp --kv-quant 4 --mtp-head-kv-quant --ctx-size 450000 \
  --max-tokens 64000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

At 4-bit KV the quality cost: mean KLD 0.1355 at 8-bit KV to 0.1458, and next-token agreement 89.08% to 88.84%.

**64 GB Mac, Sushi-3bpw**

```bash
sudo sysctl iogpu.wired_limit_mb=59000
hf download beamster/Qwen3.8-Flash-Next-Sushi-3bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw

# 1. images, 8-bit KV — the default quality
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 128000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1

# 2. images, 4-bit KV — twice the context, at 9% KLD and 0.7 points of next-token agreement
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw \
  --mtp --kv-quant 4 --mtp-head-kv-quant --ctx-size 256000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

**96 GB+ Mac, Sushi-4bpw**

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-4bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 500000 \
  --prefill-chunk 2048 --max-tokens 64000 --prefix-cache-disk 20GB \
  --prefix-cache-entries 1 --temp 1
```

Set the GPU memory limit before serving (it resets at reboot):
```bash
sudo sysctl iogpu.wired_limit_mb=88000    # 96 GB Mac
sudo sysctl iogpu.wired_limit_mb=120000   # 128 GB Mac
```

**128 GB Mac, MiMo-V2.6-Flash-Sushi-2.3bpw**

```bash
sudo sysctl iogpu.wired_limit_mb=120000
hf download beamster/MiMo-V2.6-Flash-Sushi-2.3bpw --local-dir ~/.sushi/models/MiMo-V2.6-Flash-Sushi-2.3bpw

# images, 8-bit KV, MTP, the full 1M context
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/MiMo-V2.6-Flash-Sushi-2.3bpw --ctx-size 1048576
```

MTP and the 8-bit KV cache are on by default for MiMo, and thinking is on by default, as in Xiaomi's chat template.

- `--mtp-head-kv-quant` stores the MTP head's own KV at 8 bits too.
- `--preserve-thinking off` keeps only the latest turn's thinking in the prompt. Agents running long sessions may prefer
  it for the shorter context; each new instruction then re-processes the prompt from the first dropped thought.
- `--prefill-chunk 2048` is the widest prompt step per forward; a wider one costs memory without prefilling faster,
  and a request that does not fit steps down to a narrower chunk.
- `--prefix-cache-mem 1GB` also keeps seen prompt prefixes hot in RAM, faster than the SSD (RAM retention is off by default).
- Seen prompt prefixes live on the SSD by default, so a repeated prompt skips its prefill; the budget is sized per model and capped at 20GB, and `--prefix-cache-disk 20GB` sets it.
- `--prefix-cache-entries 1` keeps one conversation's prefix; raise it to 4-8 when several agents share the server.

## Coding agents

With the server running, `sushi launch <agent>` starts claude, pi, omp, opencode, codex, grok, hermes or aider against it:

```bash
sushi launch omp
```

pi, omp, codex, grok and hermes run from their own home under `~/.sushi/<agent>/`, so your usual config is untouched and
the session does not see your other providers, settings or history; claude, opencode and aider reach the server
through environment variables. `--print` writes the config and prints the launch script instead of running it.
