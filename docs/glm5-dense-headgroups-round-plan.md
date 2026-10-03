# Cold MLA and scalar index head groups

Start from accepted packed32 runtime `af51e72f`, documentation `d1ba5b8c`.
[Dense-prefix research](glm5-dense-prefix-research.md) confirms current cold
prefill already uses native D256 SDPA. Its one D512 absorbed proposal is a
memory/arithmetic trade, not a missing-native fix.
[Head-group research](glm5-index-headgroups-research.md) proposes one exact
four-subgroup scalar scorer. Neither promises a throughput gain.

| Worker | Scope | Ownership |
| --- | --- | --- |
| 1 | Cold T2048/B16 absorbed MLA | Isolated helper/probe; original projection and packed helper reuse |
| 2 | B1/B3 scalar score head groups | Isolated helper/probe; minimal original score reuse |
| 3 | Current HTTP qualification, then component/model orchestration | Private harnesses and sequential job lifecycle |

The accepted packed32 HTTP2K–16K and selected32K qualification runs first.
Other workers may prepare source only. No builds, fixtures or other GPU work
during that job. Root owns runtime seams, admission, heavy-job grants and
acceptance. Preserve the accepted CLI identity for qualification.

The cold candidate applies only to B1/T2048/H64, original A6g128 projections
and BF16 input/cache. Keep every causal key via explicit ascending2051-slot
IDs, original masked gather and fixedB16/two pending native D512 graphs.
Charge approximately4GiB bank writes and all projection copies. Prove membership,
future/invalid nonfinite guards, valid nonfinite propagation, fallback geometry
and ownership. Report BF16 drift against the existing expanded D256 MLA.
One complete-layer3-warm/11-pair gate includes projections, all copies/IDs,
gather, attention, settlement, output and frees. No batch/geometry variant.

Only a clear component winner receives a matched model gate. Before acceptance,
measure numerical-mode drift against the current target on fixed code/prose
inputs and64 teacher-forced continuation positions: per-prompt mean KL<=0.01,
max KL<=0.15, top1>=95%, NLL increase<=0.02 and no new nonfinite values.
These are target-mode drift checks, not lossless-teacher pack KLD. Validate
candidate serial/spec token/state consistency and unchanged stored dtypes.
No precision restoration or cross-mode state-bit claim if logits differ.

The score candidate uses exactly four32-lane subgroups, eight heads each,
preserving original dot order and BF16 dot/product rounding. Store32 converted
rounded terms in128B shared memory; one thread sums h0..31 in original FP32
order after a uniform barrier. Keep three original branch histories/offsets,
not one shared query calculation. Prove every score, ordered pools/IDs and
native B1/B3 output, including ties, signed zero, nonfinite data, suffixes,
odd history and pools beyond8192. One complete three-selector/native-attention
3-warm/11-pair gate includes all work/settlement/frees, not SCORE alone.

Stop mismatches, noise or losses without retuning. Exact score winner gets
matched8192/192 N2/A6/native ABBA, equal references, unchanged acceptance,
serial output/valid-state proof and measured peak. Runtime wins then qualify
HTTP2K–16K and selected32K, commit/push; rejected sources are archived/removed.
Keep BF16 compressed MLA, FP32 KDA state/accumulators, original small tensors,
resident embeddings and A6 default. No64/128K. Goals1500/60 remain unproven.
