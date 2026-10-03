# Next exact decode component: grouped middle/down fusion

Research only at accepted runtime `e1597cc2`, outcome checkpoint `f06a035c`,
2026-10-03. Recommend one bounded **group2 lane middle/down fusion** component.
This removes an existing command/global plane while retaining weight sharing.
Speedup is unmeasured; adverse prior fusion evidence makes a loss plausible.
No source, prototype, build or GPU job accompanies this recommendation.

## Current evidence and the distinct call path

The current first-round42-chain replay uses saved actual BF16 X, original
IDs/FP32 scores and full E288 banks, preserving layer3–44 order/async4 and all
frees. Original replay median was17.823292ms; the later matched control was
17.816458ms. These exclude router/shared/MLA/KDA/assistant/commit and do not
attribute whole-verifier latency. All516096 saved BF16 outputs were exact.

[Singleton gate/up splitting](glm5-singleton-current-replay.md) preserved every stage but lost1.61%,0/11 pairs;
[all-endpoint KDA retention](glm5-kda-all-endpoint-result.md) was noisy1.50%,6/11, with an unresolved release
assertion. Fewer register declarations and avoided replay work did not prove
useful complete-chain gains. Do not retry either layout or infer spilling.
QKV joins, mini-head, N3, group3/word/window/radix variants also remain excluded.

`glm5_dflash_ffn.apply` → `glm_group2.moeLayout(serial,grouped,lane)` currently
runs lanePairPrepare, grouped gate/up, **separate lane middle**, grouped lane
down and weighted finish. The lane branch always materializes F16[24,2048].
Existing `CLAMPED_MIDDLE_DOWN_SOURCE`/lane-fused code demonstrates the original
middle F16 boundary can be kept on-chip, but that is ungrouped and is bypassed
by this selected grouped path. This proposal keeps pair weight reuse, unlike
simply selecting the existing ungrouped fused arm.

Old synthetic eight-expert lane-fused timing was adverse: at R4,408.208µs
versus lane-separate359.167µs; R8/R16 also lost. It provides a reusable exact
body, not a speed claim. No current full-E288 grouped fusion result exists.

## One fixed candidate

Strict B1/T3/S24,H4096/I2048,E288,top8,n36/MCG/W12/clamp10. Gate/up, original
ordered IDs/scores and final weighted finish stay unchanged. One fused kernel
uses the current inline group2 ballots and leader/partner rules, including
odd singleton tails. Each leader computes its member-specific original lane
middle into threadgroup F16 storage, then performs current grouped lane down
with original K-tile/FMA/r-before-simdgroup order and F16 inner stores.

Use the existing fixed8 sequential output tiles per group from the old fusion,
not a tile/window sweep. Compute prepared middle once per member/group, issue
a uniform barrier, then process eight down tiles sequentially with the original
4KiB serial-member reduction plane. Every intermediate must round to the same
F16 bits before the down dot. Merely retaining an FP32 middle would change
the target and is forbidden. This is one kernel, not singleton/pair dispatch
splitting; no routing metadata prepass, compact bank, weight copy or new state.

It removes one middle dispatch and98304-byte global middle plane per layer.
Down input loads move from the device plane to on-chip F16. Output-tile groups
fall256→32 per candidate slot, with the same total down dot/weight decode work.
The price is repeated middle construction across those32 groups and static
8KiB prepared member storage plus4KiB partials:12KiB/TG rather than current
4KiB down partials. Singleton leaders still inherit that static shared maximum.
Only12.9% of saved first-round logical assignment visits were removed by group2; there is no
additional bank-traffic reduction here. Logical load counts are not DRAM,
cache, occupancy or spill measurements. The entire17.8ms replay is the only
component ceiling, not a forecast of60tok/s or an additive verifier share.

No persistent allocation is introduced. The global middle plane disappears,
but larger on-chip resource/lifetime effects can increase peak or reduce
occupancy; measure rather than credit an admission reduction. Existing
prefill and T1/T2/T4 behavior remain current fallback. Precision policy remains
stored weights, BF16 compressed MLA, FP32 KDA state/accumulators, resident
embeddings and inherited expert F16 boundaries, without restoration.

## One decisive gate

One isolated expert worker owns helper/probe and minimal source reuse; root
owns FFN delegation/bills/model evaluation. Reuse the current42 saved inputs
and full banks; no recapture/model ladder. Prove each prepared member's F16
middle, every F16 down output and all516096 final BF16 values against current
separate group2. Test all-single/all-pair/odd-triple membership, distinct scores,
lane order, signed zeros, uniform barriers and unused-member safety. Parse/
compile packaging errors must be repaired without an algorithm variant.

Then one three-warmup/eleven-pair complete42-chain comparison includes prepare,
gate/up, fused middle/down, finish, allocations/evaluation/frees, same layer
order and eleven absolute async4 submissions/final settlement. Preserve every
pair and measured peak. Stop mismatch/noise/loss; no down split, tile retune or
extra fusion variant follows.

Only a clear component winner receives root's one matched8192/192 N2/A6/native
model ABBA, equal held references, all rounds/cleanup, unchanged acceptance,
and strict serial token/full valid-state parity. Actual total decode and peak
decide acceptance. Geometry has no2K–32K switch, but routes and other context
costs do; selected32K follows only an accepted winner. This does not address
cold prefill or establish either1500/60 goal in advance.
