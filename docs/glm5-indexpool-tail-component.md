# Bounded IndexPool tail scorer component

The original captured P8394/T811 final-chunk selector plus unchanged B32
attention passed its component gate: medians **78.628875 → 71.037583 ms**,
9.6546% lower, paired median 10.0703% lower and **11/11 wins**. This is one
captured attention component, not model prefill/decode throughput or accepted
numerical quality. The [round plan](glm5-expert-pair-tail-round-plan.md) still
requires its frozen above 32768-ID quality and complete model gates.

The separate default-off `SUSHI_GLM_INDEX_SCORE_NAX_LONG` control extends the
pool limit to 8448 only with base NAX enabled. Original ≤8192 geometry/operator
behavior, minimum 3584 pools, T9–16, serial 2048-pool tiles, all dot/output guards,
BF16 dots/product boundaries and ordered FP32 head sum remain. No shader or
retrieval policy changed. Base-off/short query paths stay ordinary.
`transientBudget` retains 8 MiB and adds 512 KiB per pending layer in enabled mode;
all original score/attention/packed/cache/margin bills remain without credits.

## Proof and actual drift

The cap/bill red failed genuinely: expected 8448/found 8192 and expected 17825792/
found 16777216 at pending 2. Narrow green policy checks passed. GPU synthetics
proved 131072 original-domain score bits and 32816 expanded IDs exact under long
off/on. Boundaries 8193/8448/8449, T9/T16 versus short queries, signed weights,
cutoff ties, uniqueness, complete-pool/partial-tail causality, F32 rejection and
observable/masked nonfinite behavior passed. No stable tied-ID ordering was
invented. Synthetic runs made no timing/actual-retention claim.

Root's first-original-MLA capture used the frozen 33579-ID code prefix, original
2048 schedule, offset 32768, processed 33579 and scale 0.0625, with both new modes
off. The exact eight-key bundle contains original BF16 Q/index-Q/weights/valid
latent/pool arrays plus U32 offset/processed and F32 scale. All 811 actual rows
were audited before timing:

| Quantity | Observed result |
| --- | ---: |
| FP32 score values | 6,807,534 |
| Changed score bits versus scalar | 623 |
| Minimum selected-pool overlap | 511/512 |
| Maximum score difference | 0.5 |
| BF16 attention output values | 26,574,848 |
| Changed output bits | 19,232 |
| Output relative L2 | 0.0001439019 |
| Output maximum absolute difference | 0.0048828125 |
| New nonfinite scores/outputs | 0/0 |

These are measured same-pack operator differences, not old-target parity or
lossless-teacher quality. Both synthetic passes were mandatory before actual
execution. Current/candidate full output references and the original input
bundle stayed resident equally before peak/warm/timed arms. Every sample
included complete scoring/tile waits, partition/expansion, B32/fragment attention,
endpoint settlement and result frees. Three warmups preceded eleven alternating
fresh operations. Timed long engagement was 561 calls (51 per candidate ×11).

Candidate active baseline was 202752012 bytes and measured peak 404996984,
a **202244972-byte whole-attention growth**. This includes ordinary gathered
banks, score/select intermediates and output lifetimes. It is not net scorer
extra memory or a justification to replace the existing full admission ledger
with 8.5 MiB; root must retain every old bill and the additive reserve.

| Pair | Original ms | Long cap ms | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 78.403709 | 71.182916 | 9.2098% |
| 2 | 78.107375 | 70.178417 | 10.1514% |
| 3 | 78.670000 | 71.226666 | 9.4615% |
| 4 | 80.042750 | 70.483875 | 11.9422% |
| 5 | 79.617750 | 71.199708 | 10.5731% |
| 6 | 78.628875 | 70.655834 | 10.1401% |
| 7 | 78.306958 | 70.875833 | 9.4897% |
| 8 | 78.395041 | 71.757167 | 8.4672% |
| 9 | 78.738500 | 70.695834 | 10.2144% |
| 10 | 78.297000 | 71.222166 | 9.0359% |
| 11 | 78.992375 | 71.037583 | 10.0703% |

## Provenance and limits

Accepted runtime is `4fcb541e`, plan `35666a7a`, with hashed WIP. Artifact
`glm53-indexpool-tail-20261004` preserves every pair, per-selector drift,
synthetics, sources, commands, exits and thermal records. Capture artifact
`glm53-tail-actual-capture-20261004` records the original tuple/provenance.
Fixture SHA256 is `ada743717793309c92150c96718fbb46f320a3994c1ed916664c9211b0d4b56c`
(96379818 bytes); root verified its source/flags and capture limits separately.

Packaging history remains archived: the private Zig identifier `packed` was
renamed; an initially broad CPU filter also ran a tiny imported nine×four GPU
append test and was truthfully replaced by the narrow filter. The first actual
attempt stopped before audit/attention/timing because its adapter requested GPU
safetensor Load, unsupported by the pinned Metal backend. Only the loader moved
to the default CPU stream; numerical Ops stayed GPU. Private error/phase
persistence was added, with no fixture, cap, bill, operator or math variant.

Final repaired build PID 53649 exited 0; actual proof/timing PID 54014 exited 0,
all 6 tests passed. ReleaseFast, foreground `taskpolicy -a`, accepted libraries,
one exclusive per-job lock, confirmed maximum fans/idle and automatic cleanup
were used. No process/lock remains. Qualified CLI/source defaults were not
replaced. Component binary SHA256:
`726465ff52e0835469eaa5ee93c0957d57ec24d2449f5b5c906cb76b1486a8c4`;
helper SHA256 `034bb6a23bf487c52ce37c136608ccd110e95bf0abe3c011ac8c6939278d773f`.
Source/library/binary/fixture hashes were checked before numeric execution.
No quality, model-throughput or decode acceptance follows this component alone.
