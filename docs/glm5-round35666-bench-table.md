# One round 2K–32K accepted-runtime benchmark

One user-requested pinned llmprobe0.6.13 bench-only ladder completed with runs1,
reasoning default, no repeated full ladder. Runtime was accepted `4fcb541e`,
qualified CLI SHA256 `102cb8d5efcf279295ea059201eb0a88c3e7d85321a68dced625c0b4c6858ae3`;
the expert-pair and long-tail experiments were unaccepted and disabled. Source
work in progress does not describe the executing immutable CLI. GLM2.3 target,
A6g128 DFlash2, N2/children4/async4, BF16 compressed MLA/FP32 KDA, native B1/B3,
packed32/current HC flags and prefill2048/async2 were fixed.

All rates are actual server token counts divided by server intervals, tokens/s.
Decode excludes the first token delivered from prefill, using outputs minus1.

| Context | Ordinary prefill | Ordinary decode | Predictable prefill | Predictable decode |
| --- | ---: | ---: | ---: | ---: |
| 2K | 869.79 | 46.18 | 918.33 | 50.61 |
| 4K | 844.65 | 39.39 | 844.02 | 49.33 |
| 8K | 769.25 | 46.05 | 779.09 | 49.64 |
| 16K | 730.25 | 42.47 | 721.95 | 48.08 |
| 32K | 660.77 | 42.99 | 663.04 | 47.56 |

The ordinary16K request stopped at177 outputs with EOS enabled; its denominator
is176. All other table cells produced192 outputs with denominator191. This is
a current baseline table, not a new-candidate192-output qualification. Actual
request diagnostics, output IDs/hashes, settings, counters and raw timings are
retained. Do not assign movement from earlier separate boots to a kernel change.

Actual ordinary/predictable input counts were2072/2036,4095/4059,8261/8225,
16314/16278 and32783/32747. The nominal32K counts differ from the previous
33543/33579 qualification and do not prove the new tail cap's target geometry.

Run-level `/props` active memory was94548731128 before and94548862200 after:
about94.55 decimal GB /88.055 GiB. These values describe settled residency,
not per-request peaks. The memory/working limit remained115448725504 bytes.
No per-cell peak, RSS estimate or new memory instrumentation is claimed.

The foreground server used confirmed maximum fans/idle and one exclusive job.
Client55835 exited0; owned server55723 stopped, fans returned automatic and lock
released. Frozen CLI hash remained unchanged. The single ladder produced31 raw
subscriber records; table extraction selected the ten actual ordinary/predictable
cells from saved server counts and timings, preserving body-unavailable nulls.
No request recovery, rerun or transport change was performed.

Artifact `glm53-round35666-bench-prep-20261004` preserves exact client/server
argv, flags, passive diagnostics, full client JSON/HTML, props, source/binary/
library provenance, table-data.json, extractor and telemetry. The extractor's
original192-output-only match was corrected to retain the valid177-output row;
no request or score changed. The 1500 prefill/60 decode targets remain unmet.
