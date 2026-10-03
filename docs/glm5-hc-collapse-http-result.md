# HC collapse HTTP qualification

The accepted SIMD32 HC collapse opt-in completed pinned llmprobe 0.6.13
bench-only 2K/4K/8K/16K and selected 32K qualification. Both jobs are terminal
and cleaned up; no additional rung was run.
The matched model gate is the runtime acceptance evidence. These HTTP
comparisons inherit the recorded packed32 boot and do not isolate a kernel gain.

One target plus A6g128 used native B1/B3, N2/children4/async4, packed32,
prefill chunk2048/async2 and the accepted projection/KDA/grid/draft flags.
Only `SUSHI_GLM_HC_COLLAPSE_SIMD32=1` was added; rejected HC expansion stayed
explicitly 0. Settings report `hc_collapse_simd32:true` and each measured final
response reports positive `hc_collapse_simd32_calls`. BF16 caches and FP32
state/accumulators remain unchanged. Default settings were not changed.

All ten measured cells returned 192 output IDs. Rates use exact final server
timers: actual input count / prompt time and 191 post-prefill outputs / decode
time. The arrow compares inherited `af51e72f` with accepted `4fcb541e`.

| Rung | Workload | Actual inputs | Prefill tok/s, inherited → HC | Decode tok/s, inherited → HC | HC calls | B32 calls | Native B3 calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | Ordinary | 2078 | 893.75 → 889.73 | 46.11 → 45.66 | 6480 | 0 | 792 |
| 2K | Predictable | 2042 | 986.45 → 957.48 | 49.60 → 51.24 | 5760 | 0 | 704 |
| 4K | Ordinary | 4092 | 862.62 → 858.97 | 44.10 → 44.74 | 6480 | 693 | 792 |
| 4K | Predictable | 4056 | 861.46 → 857.48 | 48.76 → 49.72 | 5850 | 682 | 715 |
| 8K | Ordinary | 8263 | 796.55 → 791.32 | 45.20 → 43.95 | 6570 | 2134 | 803 |
| 8K | Predictable | 8227 | 799.98 → 791.93 | 48.85 → 50.50 | 5760 | 2123 | 704 |
| 16K | Ordinary | 16314 | 737.88 → 738.32 | 44.39 → 41.75 | 6750 | 4895 | 825 |
| 16K | Predictable | 16278 | 735.43 → 738.63 | 47.19 → 49.02 | 5760 | 4884 | 704 |
| 32K | Ordinary | 33579 | 664.19 → 668.35 | 40.10 → 41.66 | 6570 | 10835 | 803 |
| 32K | Predictable | 33543 | 659.75 → 667.37 | 46.62 → 47.80 | 5760 | 10824 | 704 |

Current minus inherited input counts were +5/−5/+2/+4/−52 at
2K/4K/8K/16K/32K for both workloads. The baseline was not rerun. Mixed rate and
round-count movement
across different boots and prompts cannot be assigned wholly to HC collapse.
The measured chunk 2048 prefixes retain the original long-row HC path, so
reported prefill movement is not attributed to this small-row helper. B32 made
zero calls in the dense 2K cells. Exact output-ID
arrays are retained, but these different-input HTTP boots do not establish
cross-boot token/state parity. The separate matched model gate establishes
same-input mode parity.

Measured requests reported `ignore_eos:false`, although the server allowed
benchmark ignore-EOS; every measured cell nevertheless completed 192 outputs.
The byte-identical private passive final-response subscriber changed no pinned
bundle, request or transport and wrote once per final response. Its small client
overhead was not isolated. Request bodies remained null/unavailable; no raw-body
recovery was attempted. Actual usage and tokenizer counts are recorded;
input ID arrays are not exposed by HTTP. The 2K–16K job retained 27 final
diagnostics among 29 raw subscriber entries;
selected 32K retained 21 among 23. Each job retains two non-diagnostic calibration
entries. All ten measured cells have complete final settings, counters and IDs.

Post-job active memory was 94,548,862,200 bytes under 115,448,725,504-byte
memory limit. HTTP exposes no allocator peak. The unchanged source admission
formula billed 6,600,589,312 bytes at 2K through 10,687,479,808 at 32K, before
actual planned cache growth. Largest measured growth was 203,292,672 bytes;
minimum source-calculated headroom over post-job active+reserve+growth was
10,212,383,496 bytes at 32K. This calculation records the unchanged ledger, not a
measurement of instantaneous request peak. All original reserves remain.

Artifacts `glm53-hc-qualified-llmprobe-20261004` and
`glm53-hc-qualified-32k-20261004` retain final diagnostics,
exact times/IDs/settings/flags, admission calculations, client JSON/HTML and
thermal records. Accepted runtime/source was `4fcb541e`. The frozen CLI was
built at checkpoint `99547a26` plus the byte-identical accepted runtime patch;
it was not rebuilt after commit. CLI SHA256:
`102cb8d5efcf279295ea059201eb0a88c3e7d85321a68dced625c0b4c6858ae3`.
Helper/probe/current-source and library-version hashes are recorded privately.
The original active CLI launched through foreground QoS and explicit accepted
library paths. Maximum fans and required idle preceded the exclusive job.
Both clients exited 0; own servers stopped, locks released and fans automatic.
CLI hash matched before/after each job. No old-baseline rerun or 64K/128K rung was used.


Predictable decode was 51.24/49.72/50.50/49.02/47.80 tok/s across 2K–32K;
ordinary was 45.66/44.74/43.95/41.75/41.66. These are qualified actual cells,
with mixed separate-boot movement and no 1500/60 claim. The helper remains an
accepted opt-in; its matched correctness/performance gate is independent of
these inherited comparisons.
