# Four-subgroup scalar scorer component

The fixed candidate passed exactness but failed the decisive timing gate: only
five of eleven complete-call pairs won, with paired median 0.83% slower. No model
run or variant is warranted. It follows the
[round plan](glm5-dense-headgroups-round-plan.md) and
[research](glm5-index-headgroups-research.md). Runtime selection is unchanged.

`glm5_index_headgroups.scores(ops, q, keys, weights, offset, pools)` returns an
optional caller-owned array in `Ops`. It admits GPU BF16 T1–3/J32/I128 and the
original 2 MiB score bound, including pools beyond 8192. Unsupported input returns
null before graph construction. There are no endpoint waits, mode flags or
runtime counters in this isolated helper.

Each 128-thread group contains four 32-lane subgroups. Each subgroup visits its
fixed eight heads with the original per-lane dot order, `simd_sum`, BF16 dot
rounding and BF16 weighted-product rounding. Lane 0 stores each converted FP32
term in 128 bytes of shared memory. After one uniform barrier, thread 0 sums
heads 0–31 sequentially and applies the original final BF16/FP32 score store.
Future pools return uniformly before the barrier. No subgroup sum, query reuse,
padding, new global intermediate plane or precision restoration is introduced.

The probe uses two read-only original-source exports supplied by root:
`attention.scalarIndexScoreSource` and `attention.indexPoolExpandSource`.
The control compiles the unchanged scalar source; both arms use the original
negative/argpartition/top 512 and 2051-slot expansion, then the same native attention.
The public helper contains neither exported reference nor selector code.

Tests were written first. Proof covers all four pool phases, chain/fork paths,
a distinct constructed fork suffix, negative weights, signed zero, cutoff ties,
NaN/Inf, uniform future masking and 8500 pools. Existing real 16K fixture planes
were used; ancestry arrangements and repeated late 32K keys are explicitly
synthetic. Input graphs settled equally before clocks; proof copies are outside
timing. All score metadata, selection, ID concatenation, gather, native B3,
endpoint evaluation and owner frees are included in the complete-call clock.

## Recorded gate

ReleaseFast behavioral red compiled and failed at `ExpectedHeadGroups` with a
null helper. An earlier launch failed before MLX startup because system
`taskpolicy` removed an outer DYLD path; its error is preserved separately.
The corrected launch sets that path through `/usr/bin/env` inside `taskpolicy`.
Initial green reached the score proof but stopped at a probe dtype omission:
MLX partition indices are uint32. The sole correction added exact raw-byte
uint32 comparison, without a cast, shader or layout change.

The final focused run passed all three tests, exit 0:

| Exact evidence | Values |
| --- | ---: |
| FP32 score bits | 151443 |
| Ordered uint32 pool indices | 19968 |
| Ordered int32 token IDs | 79989 |
| Native B1/B3 BF16 outputs | 1769472 |

Three warmups per arm preceded eleven alternating complete-call pairs. Control
median was 0.720833 ms and candidate 0.699167 ms, a 3.01% standalone median reduction.
Only 5/11 paired samples won; paired median was 0.83% slower, with paired reductions
from −4.96% to 13.66%. The conflicting results reject a performance claim. There
is no model timing, projected decode gain or subgroup variant.

Measured candidate peak above the resident fixture was 7,075,919 bytes. Native
bank/score allowances remain unchanged; no reservation saving is claimed. Fixed
pool sizes 4095/4095/4096 and offsets 16381/16382/16383 were used throughout timing.
Foreground QoS, per-job lock, maximum fans and quiet idle were recorded. The
process ended, lock was released and fans restored to automatic.

Artifact `glm53-index-headgroups-20261003` retains exact build/launch recipes,
binary/source hashes, red and probe failures, final proofs, all pairs, peak and
thermal records. Helper SHA256
`26fb362752bd32c256ffdd08b67bc2d4603559a539012ada63ccf32493dc2275`.
No runtime hook or public test import was added.
Rejected helper/probe/private root were archived with their hashes and removed.
Root separately restores the two read-only exports. Runtime arithmetic and
selection remain unchanged; only this documentation lesson is retained.
