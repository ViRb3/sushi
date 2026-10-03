# Bounded packed prefill cadence

The candidate keeps at most two independent 16-row tile graphs alive. Each
retains its selector arrays and packed helper ownership, submits its result
with `mlx_async_eval`, then settles both results before freeing either gathered
bank. A final unpaired tile follows the same cleanup. Error cleanup settles
submitted outputs before releasing their graph handles. The native packed
helper, causal selection, BF16 cache/output and FP32 accumulators are unchanged.

`glm5_attention.bindPackedCadence` selects the candidate for a scoped diagnostic;
`SUSHI_GLM_PREFILL_CADENCE=1` selects it explicitly; the default remains serial.
The transient budget adds a conservative 64 MiB per
pending MLA layer, or 128 MiB at async2, beyond the existing packed admission.
It is zero for disabled cadence, disabled packed attention or chunks at most
16 rows; overflow is an error. HTTP/native reservation and engagement counters
are coordinator-owned.
The small accumulated result planes are retained as in the serial caller;
gathered banks never accumulate across a whole T2048 chunk.

`SUSHI_GLM_PREFILL_CADENCE_CAPTURE` is an explicit safetensors output path. It
captures only the first eligible T2048 call at processed8192, including actual
Q, index query, weights, latent cache, pooled cache and exact offset/history/scale.
For standard geometry the file is about 153 MiB. The only optional target is
`SUSHI_GLM_PREFILL_CADENCE_CAPTURE_HISTORY=16384` for the winner's guard. This
forces evaluation and its model run is capture-perturbed, not a throughput measurement.

The focused `glm5_attention_prefill_cadence_probe.zig` probe uses
`SUSHI_GLM_PREFILL_CADENCE_FIXTURE` and
`SUSHI_GLM_PREFILL_CADENCE_OUT`. The probe compares every BF16 output bit for
T2048 and T33 (an unpaired final one-row tile), checks peak extra live allocation,
then records three warmups and eleven fresh interleaved AB/BA pairs. Samples
include selection, gathered banks, unchanged packed SDPA, output collection,
endpoint evaluation and frees. Native masking and future/invalid key guards
remain in `glm5_attention_nax_packed.zig`.

At source `21578aba` plus the candidate, the real T2048 8K fixture matched all
68,190,208 BF16 values across T2048 and T33. The median inclusive serial time was
130.201458 ms versus 105.953500 ms for the candidate: 18.62% less time, with
11/11 paired wins and median paired reduction 18.19%. The measured peak delta
was 128 MiB relative to resident fixture arrays plus the held serial reference
output. It includes the candidate's output and is not net memory overhead over
serial. Admission separately bills the conservative second bank; output planes
remain in the existing activation bill. Both focused tests passed. The run used the selected stack's packed
attention and NAX selector flags; at 8K the selector still uses scalar scoring.
This is a whole-attention component result, not model throughput.

These measurements used ReleaseFast, MLX 0.32.3 / `64ea011c`, the patched mlx-c
`56b2d39`, foreground `taskpolicy -a`, exclusive GPU owner
`glm53-prefill-cadence-probe`, maximum fans and a ten-second cooldown. Probe
binary SHA256 starts `5dc3d904cc26742e`; capture used a repeated fixed T2048
token fixture and warmup0. Raw fixture, source hashes and eleven paired samples
are recorded privately by the coordinator. The attention source SHA256 starts
`03a4780debd06c2b` for the 8K proof.

At 16K, with NAX selection engaged, all 68,190,208 BF16 values remained exact,
the same measured peak delta was again 128 MiB, and all three focused tests passed.
The serial median was 139.506750 ms versus 129.887917 ms: 6.89% less time,
11/11 paired wins and median paired reduction 7.31%. Sixteen candidate calls
were recorded across correctness, warmups and samples. The probe binary SHA256
starts `c3bad2af785165b1`, with the same runtime/QoS/fan protocol and exclusive
GPU owner `glm53-prefill-cadence-16k-probe`. The attention source SHA256 starts
`5b26960d40d5d18c` for this guard, including the scoped opt-in and admission API.

Artifact keys are `glm53-prefill-cadence-20261003` and
`glm53-prefill-cadence-16k-20261003`. The 16K capture additionally wrote the
routed layer20 T2048 fixture for its independent workstream; all captures are
excluded from throughput evidence. Full-model token/state and HTTP gates belong
to the coordinator.

The existing NAX scorer's internal waits reduce but did not erase the measured
gain while constructing the second selector. No wider geometry or selector
batching is part of this candidate.
