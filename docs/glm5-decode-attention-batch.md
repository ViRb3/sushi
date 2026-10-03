# Isolated true B3 native decode attention

The coupled mode is an optional numerical target, gated by
`SUSHI_GLM_DECODE_BATCH=1` or a scoped `bind` override; the default is off.
It gathers the immutable BF16 latent prefix and each branch's ancestry
into one `[B,2051,512]` bank, then calls forced native rank-four SDPA with
`[B,1,64,512]` Q and shared K/V. B1 and B3 use the same gather source and native
math; only batch metadata, output dimensions and grid differ. Native accumulators
remain FP32 and output remains BF16. No precision restoration is present.

Offsets and lengths are explicit per branch. A fork's `[0,2]` tail maps its second
logical suffix position to tape row2, and uses the same position as `[0,1]`.
Invalid/future IDs are zeroed before source access; all-invalid outputs are
zeroed after SDPA. Prefix/tape strides are admitted explicitly and wrapper
copies are disabled for the gather. No full-history copy or cache donation is
used. The conservative bound is 8 MiB per pending layer, 32 MiB at async4.

The test plan precedes implementation: compare every B3 BF16 bit with three
native B1 calls, settling all control outputs once; record finite values,
relative L2/max error and changed bits versus current scalar split-eight plus
its original merge. Directed guards cover empty batch rejection, invalid and
future indices, sole key zero, masked nonfinite physical prefix rows, all-empty
selection, a last-slot key and distinct fork suffix rows.

The probe reuses actual 16K Q/index-Q/weights/cache tensors. Its logical prefix
has 16381 rows and its three-row suffix tape uses actual cache rows16381–16383.
Chain/fork arrangements are constructed fixtures, not captured speculative
trees. Both timed arms build three unchanged original scalar selectors and use
the same branch views and ordered selection. Selection, ancestry preparation,
gather/scalar partials/native SDPA, result construction, one endpoint vector
settlement and frees are included. Common captured input planes are resident
before timing. Three warmup pairs and eleven alternating pairs compare the
current complete scalar control with true B3 at one 16K fork geometry.

`SUSHI_GLM_DECODE_BATCH_FIXTURE` supplies the existing capture;
`SUSHI_GLM_DECODE_BATCH_OUT` supplies the isolated report. Two read-only attention
probe exports reuse selection and scalar split-eight arithmetic. The measured
component preceded coupled integration. Component results are qualified separately from whole-model throughput.

The component at `19ab4f9b` plus hashed WIP source passed both focused tests on
2026-10-03. B3 matched three native B1 calls at all 196,608 BF16 output values
across chain and fork arrangements. All directed guards passed. Against current
scalar output, 13,717 values changed, relative L2 was 0.001346278, maximum absolute
error was 0.00390625 and all outputs were finite. No restoration or parity
tolerance was applied to native B1/B3 equality.

The inclusive scalar median was 902.084 µs versus 769.667 µs for B3: 14.679% less
time, 11/11 paired wins and median paired reduction 16.401%. Individual reductions
were 8.03–45.68%, including slow control samples. Peak candidate allocation delta
was 6,961,275 bytes, below 8 MiB; its baseline retained fixture/scalar/native-B1
graphs and outputs, so this is not net overhead versus scalar.

Artifact key `glm53-decode-true-batch-20261003` holds inputs, WIP source hashes,
raw pairs and the ReleaseFast probe with SHA256 starting `302299d210308c4d`.
Runtime was MLX 0.32.3 / `64ea011c` and patched mlx-c `56b2d39`, foreground
`taskpolicy -a`, exclusive GPU owner `glm53-decode-true-batch`, maximum fans and
ten seconds idle after a 51.84 °C status reading. Actual fan RPM5347/5782 was
confirmed against maxima5349/5777. This is component evidence, not model speed.

The source integration routes ordinary short-row attention and serial/replay
through matching native B1 math. Dense/boundary queries use ordered all-key IDs;
sparse queries retain the original per-node scalar selector. The verifier uses
one B3 gather only for three admitted branches, otherwise native B1 per node.
An unsupported matching B1 raises a mode error rather than mixing scalar math
into the selected mode. Wider trees remain outside this arm.

The existing authoritative 256 MiB scratch limit is unchanged. Native mode
subtracts its 8 MiB allowance before deriving how many existing branch graphs
fit, and reports that allowance in live bytes. B3 requires all three branches;
native B1 groups retain the existing flush boundary. Config/device/dtype admission,
actual B1/B3 counters and a gated 8 MiB/layer budget are exposed for coordinator
integration; HTTP and diagnostic reservation/metadata remain coordinator-owned.

The coupled integration built and passed both focused checks. Ordinary serial
chain B1 and overlay chain/fork B1 matched native B3 bit for bit, with nine B1
and two B3 calls asserted. Default-off/scoped mode, conservative budget and
overflow checks passed. Artifact `glm53-decode-coupled-proof-20261003` records
the ReleaseFast binary with SHA256 starting `44e209bf31eed8c7`, source hashes,
foreground QoS and exclusive GPU owner `glm53-decode-coupled-proof`.

The private full-model evaluator is prepared under artifact
`glm53-decode-coupled-model-gate-20261003`: exact8192 IDs, N2/children4/async4,
64 outputs and accepted prefill flags. One target/assistant load supports a
strict native serial/speculative token and valid-state-byte oracle, then a
separate old-scalar greedy teacher/native-B1 forced-logit drift phase outside
gate timings.

The real-model run passed strict token and valid-state-byte parity for all
64 native serial/speculative outputs, using the A6g128 assistant. It recorded
715 native B1 calls (704 serial-oracle calls plus eleven shorter-tree calls) and
264 B3 calls across 25 speculative rounds, or 2.56 tokens per round. The scratch
high-water ledger was 69,088,292 bytes under the unchanged 256 MiB cap; measured
peak allocation was 95,560,321,984 bytes.

Separate old-scalar forced-logit drift over 64 positions measured mean KL
0.003234462, maximum KL0.048452418, top-1 agreement61/64 and zero nonfinite values.
There were 9,032,295 changed FP32 logit bits after ordinary widening for scoring.
This is a different numerical target, not old-scalar parity or a lossless teacher.

The cold provisional speculative rate was 23.993 tok/s and matched native-serial
rate29.753 tok/s. Mean draft/verify/replay/commit times were
8.031 / 90.964 / 5.876 / 1.387 ms per round. Verify median was 50.4035 ms;
the first and final shorter-tree rounds cost828.392 and251.167 ms respectively.
No rounds were dropped to manufacture throughput. These timings are not a warmed
old/new performance comparison and do not establish acceptance.

The model-gate ReleaseFast binary SHA256 starts `2f24df5eb9b55b7f`, with the same
runtime, foreground QoS, exclusive GPU owner `glm53-decode-coupled-model-gate`,
maximum fans and ten seconds idle after48.46°C. All seven tests passed, including
six import-only cases. No precision restoration or runtime default change has
been made. The warmed qualification follows below.

The warmed `glm53-native-decode-llmprobe-20261003` ladder completed with native
mode explicitly enabled and the A6g128 assistant. The inherited accepted control
is `glm53-prefill-wave-llmprobe-20261003`; it was not rerun. These server rates
use 191 decode forwards after the prefill token for 192 output IDs. Comparison
records19–26 keep ordinary and predictable requests separate:

| Kind / rung | Input IDs control → native | Control decode tok/s | Native decode tok/s | Change |
|---|---:|---:|---:|---:|
| Predictable 2K | 2037 → 2036 | 47.08 | 46.04 | −2.22% |
| Predictable 4K | 4061 → 4059 | 45.48 | 46.19 | +1.56% |
| Predictable 8K | 8225 → 8225 | 44.44 | 46.57 | +4.79% |
| Predictable 16K | 16274 → 16278 | 43.04 | 44.20 | +2.69% |
| Ordinary 2K | 2073 → 2072 | 42.65 | 42.16 | −1.15% |
| Ordinary 4K | 4097 → 4095 | 42.17 | 41.33 | −1.97% |
| Ordinary 8K | 8261 → 8261 | 41.15 | 42.09 | +2.29% |
| Ordinary 16K | 16310 → 16314 | 36.45 | 41.71 | +14.43% |

Small-context results are mixed and modest; both longer rungs improved. The
inputs differ slightly between boots, and target rounding changes can alter
acceptance. Ordinary 16K used 68 rather than 76 rounds and fewer KDA misses,
so its total improvement is not an isolated attention-kernel speedup. Ordinary4K
also crossed a full-chunk boundary (cluster 68→34 and routed-grid 84→42), preventing
prefill attribution to this decode mode. Native B3 calls in these eight cells
were 715/715/704/704 predictable and 792/770/770/748 ordinary; B1 calls were zero.
Mode-matched B1 correctness is established by the separate strict model gate.

The source was `39f8693d` plus recorded dirty hashes; ReleaseFast binary SHA256
starts `e825e47bb24c5697`. Runtime was MLX `64ea011cb65f`, mlx-c `56b2d39fc831`
with compatibility patch hash `3e1cdfdb9a38`, target26.2. Method was llmprobe0.6.13
bench-only, runs1, reasoning default, rungs2K–16K, profiling/capture off,
foreground `taskpolicy -a`, exclusive GPU owner `glm53-native-decode-llmprobe`,
maximum fans and ten seconds idle. `comparison.json`
and `server-requests.json` under the artifact key retain raw phase costs,
dispatch engagement and output counts.

This remains a default-off numerical target with the disclosed scalar drift.
The 8K token/state gate establishes mode-matched speculative correctness.
The selected 32K performance qualification follows below. The 60 tok/s
objective is not attained.


## Selected 32K and disposition

The same ReleaseFast CLI completed one selected 32K llmprobe cell, inherited
from `glm53-prefill-wave-32k-llmprobe-20261003`. Inputs matched exactly: 33631
ordinary and 33595 predictable IDs, each with 192 output IDs.

| Kind | Control decode tok/s | Native decode tok/s | Change | Native prefill tok/s | Rounds control / native |
|---|---:|---:|---:|---:|---:|
| Ordinary | 37.42 | 38.08 | +1.76% | 614.65 | 75 / 74 |
| Predictable | 42.39 | 43.00 | +1.45% | 608.77 | 64 / 65 |

Predictable draft/verify/replay/commit averaged 6.26/58.37/2.42/1.01 ms
per round; ordinary verification averaged 57.28 ms. Native B3 engaged
814/715 calls, B1 zero, with 176 packed cadence and 672 expert-grid calls
in each request. All 21 server requests, output counts and phases remain
in artifact `glm53-native-decode-32k-20261003`, exit 0. Foreground QoS,
exclusive owner `glm53-native-decode-32k`, confirmed maximum fans and a
cool ten-second idle were used; server stopped, lock released and fans auto.
Binary/runtime were unchanged from the warmed ladder above.

Retain this mode as an opt-in based on the paired inclusive component win,
strict native B1/B3 and 8K serial/speculative state proof, disclosed scalar
logit drift, and successful real-model 2K–32K qualification. The modest
separate-boot 32K changes alone are not a repeatable total-rate speedup.
Small-context results remain mixed; A6 is still the assistant default.
The final ReleaseFast suite, CLI, quiet-runner guard and local-path guard
passed. No 64K/128K run or precision restoration was added.
