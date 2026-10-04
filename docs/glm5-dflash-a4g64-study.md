# GLM DFlash2 affine4 group64 study

Date: 3 October 2026, Asia/Bangkok.

A4/gs64 works with the native GLM DFlash2 engine after a small stored-format validation extension. It reduces assistant tensor payload by **23.5%** and makes the drafting phase about **11–14% faster**. Overall decode gains are modest and workload-dependent: predictable 32K is essentially unchanged, and ordinary 8K regresses. Keep A6/gs128 as the default until a matched-input interleaved comparison confirms an overall benefit.

## Scope and checkpoint

The target is `GLM-5.3-Flash-Sushi-2.3bpw`. Only its DFlash2 assistant changed. The candidate `GLM-5.3-Flash-DFlash2-A4g64` was converted directly from the original BF16 assistant, rather than requantizing A6. It uses ordinary MLX affine quantization without activation calibration. Forty-six large linear matrices use U32 packed 4-bit codes with BF16 scales and biases at group size 64. Thirty-five other tensors retain their original bytes, including selector codebooks, the hidden selector projection, norms and convolution bases.

| Assistant | Tensor payload bytes | Decimal MB | GiB |
|---|---:|---:|---:|
| Original BF16 | 2,342,160,896 | 2342.2 | 2.1813 |
| A6/gs128 | 1,013,090,816 | 1013.1 | 0.9435 |
| A4/gs64 | 774,539,776 | 774.5 | 0.7213 |

A4 saves **238.6 MB** (0.2222 GiB) versus A6. These are tensor payload sizes, excluding file headers, caches and allocator overhead; they are not a measured whole-engine RAM delta. The unchanged target dominates this 128 GB machine’s memory use.

## Conversion and runtime validation

Saved codes and scale/bias grids were checked against the quantizer outputs. Sampled rows from every converted matrix were independently unpacked and compared with CPU dequantization. All retained tensor bytes and the source checkpoint hash were verified unchanged. No source checkpoint was overwritten.

The existing GLM assistant storage check accepted only A6/A8 at gs128. An isolated engine snapshot based on accepted Sushi revision `18b27d76` adds A4/gs64 to that check, derives packed geometry from the stored group size and reports the correct label. Existing inference kernels already support this affine format; target computation was not changed. The primary checkout and the main thread’s source were not edited for this experiment.

The ReleaseFast CLI, focused stored-geometry test and standalone gate build passed. A 512-input, 64-output greedy gate reported **exact target-token and complete final-state parity** against serial decoding. Runtime inspection confirmed 46 affine linears at 4-bit/gs64 and one retained dense linear. This establishes functionality for that tested workload; the HTTP ladder is a performance measurement, not an exhaustive serial-state oracle at every rung.

## Benchmark protocol

- M5 Max 128 GB; MLX 0.32.3 with the same staged runtime as the A6 reference.
- `llmprobe@0.6.13 --bench-only --rungs 2k,4k,8k,16k,32k --runs 1 --reasoning default`.
- Native diagnostic HTTP, greedy sampling, N2 proposals, four children and verification async4.
- Prefill chunk 2048, async2; BF16 compressed MLA cache and FP32 KDA state.
- Same accepted lane/group2, packed prefill cadence, expert-grid transpose, bounded draft/commit window, exact A6 target hoist and KDA leaf settings as the A6 reference.
- Foreground `taskpolicy -a`, exclusive GPU lock, confirmed maximum fans and a cool ten-second idle.
- Separate side-run port 18854 and isolated engine binary; no concurrent model timing.

Each measured request produced 192 output IDs. Decode rate uses **191 forwards**, since the first token comes from prefill. A6 references are the existing accepted-stack measurements: `glm53-prefill-wave-llmprobe-20261003` for 2K–16K and `glm53-prefill-wave-32k-llmprobe-20261003` for 32K. They were inherited rather than rerun. Consequently these are separate-boot comparisons, with small prompt differences and possible thermal/clock variation.

## Predictable context results

Values in paired cells are **A6 / A4**. Rates are tokens per second.

| Rung | Actual input IDs | Prefill | Decode | Decode change |
|---|---:|---:|---:|---:|
| 2K | 2037 / 2037 | 931.7 / 957.8 | 47.08 / 47.83 | +1.60% |
| 4K | 4061 / 4061 | 802.8 / 825.4 | 45.48 / 46.86 | +3.05% |
| 8K | 8225 / 8225 | 727.8 / 753.4 | 44.44 / 47.37 | +6.59% |
| 16K | 16274 / 16274 | 655.1 / 687.2 | 43.04 / 45.37 | +5.41% |
| 32K | 33595 / 32753 | 604.2 / 598.6 | 42.39 / 42.43 | +0.10% |

## Ordinary context results

| Rung | Actual input IDs A6 / A4 | Decode A6 / A4 | Decode change | Rounds A6 / A4 |
|---|---:|---:|---:|---:|
| 2K | 2073 / 2073 | 42.65 / 44.86 | +5.18% | 71 / 70 |
| 4K | 4097 / 4097 | 42.17 / 42.44 | +0.65% | 71 / 74 |
| 8K | 8261 / 8261 | 41.15 / 39.74 | -3.44% | 70 / 76 |
| 16K | 16310 / 16310 | 36.45 / 40.15 | +10.15% | 76 / 73 |
| 32K | 33631 / 32789 | 37.42 / 38.82 | +3.74% | 75 / 71 |

The 32K A6 reference used a single-rung run, whereas A4 used the full ladder. The predictable input was 33,595 versus 32,753 IDs, and the ordinary input was 33,631 versus 32,789: A4’s inputs were about 2.5% shorter. Input-normalized rates help comparison but do not remove context-dependent cost or prompt effects. The 32K decode result should therefore be treated as approximately tied rather than a demonstrated gain.

## What changed inside decoding

| Predictable rung | Draft ms per round A6 / A4 | Verification ms per round A6 / A4 |
|---|---:|---:|
| 2K | 6.09 / 5.28 | 55.35 / 54.23 |
| 4K | 6.15 / 5.37 | 56.41 / 55.36 |
| 8K | 6.20 / 5.31 | 57.69 / 55.78 |
| 16K | 6.33 / 5.43 | 60.34 / 57.96 |
| 32K | 6.28 / 5.58 | 60.53 / 61.48 |

Drafting is faster at every rung, consistent with fewer stored weight bytes and a different native affine kernel. Target verification remains the largest decode cost, so the faster assistant cannot translate its full drafting gain into total throughput. Movement in the unchanged verifier also shows why separate-boot total-rate differences cannot all be assigned to A4.

Proposal acceptance matters. Ordinary 8K needed 76 speculative rounds with A4 versus 70 with A6 for the same 192 outputs. That additional verifier work outweighed the faster drafter and reduced decode by 3.44%. Conversely ordinary 16K used fewer rounds and was faster in this run. These measurements do not establish a general accuracy or acceptance advantage for either precision.

## Recommendation

A4/gs64 is a functional, smaller experimental assistant with a consistently faster draft phase. It is reasonable to retain the checkpoint for further testing. It is **not yet a clear overall replacement for A6/gs128**: ordinary performance is mixed, 32K predictable decode is tied, and one ladder per precision does not resolve modest differences. A matched-input interleaved comparison, including both ordinary and predictable prompts, is the appropriate adoption gate.

No default assistant setting was changed and no engine patch was merged or pushed. The experiment does not validate public GLM tool serving; it uses the existing benchmark-only bridge. No 64K/128K rung was run.

## Reproducibility and evidence

Artifact key: `glm53-dflash-a4g64-side-20261003`. Its contents include:

- Checkpoint conversion report and checksums, with the complete source/output identity.
- `source-snapshot.txt`, isolated engine source, build logs and focused storage test.
- `gate/result.json`, flags, binary hash and token/state oracle result.
- `bench/result.json`, HTML report, all 29 parsed server requests and `bench/comparison.json`.
- Benchmark command/flags, binary/runtime provenance, fan/temperature records and process exit status.

The client exited 0, the owned server stopped, fans returned to automatic and the GPU lock was released. The candidate remains a separate checkpoint.
