# Exact SIMD32 HC collapse: matched model result

The fixed N2/A6/native model gate passed exactness and reduced complete
clone+decode+cleanup time by 2.6475%, exceeding 1.9891% control drift. Root accepted
it for HTTP qualification as an opt-in. This modest one-prompt result does not
establish 60 decode tok/s; A6 remains default and 2K–32K qualification follows.

One target and A6 assistant load used the accepted packed32 stack. The same
8192 IDs came from the original official-template smoke with neutral WATER
filler repeated four times, not a pure-code prompt. One shared target prefill
and assistant preparation preceded a native-B1 serial oracle. Exact references
at 191 and 192 committed inputs remained resident before both warm policies and
all four arms. N2/children4, native mode and async4 stayed fixed; only scoped HC
collapse binding changed. HC expansion was explicitly off.

| Arm | Clone+decode+cleanup seconds | Verifier total ms | HC calls |
| --- | ---: | ---: | ---: |
| Original A | 4.701883209 | 4080.795535 | 0 |
| SIMD32 B | 4.622759626 | 3997.233997 | 7200 |
| SIMD32 B | 4.624009582 | 3999.223254 | 7200 |
| Original A | 4.796348416 | 4170.759335 | 0 |

Mean intervals were 4.7491158125 versus 4.623384604 seconds. Clone and endpoint
request/context frees are included; CPU token/state comparisons are outside
those clocks. The denominator is 191 delivered tokens after the first prefill
output. Every arm produced the same 192 serial IDs and exact complete valid
MLA/KDA/convolution state, with the same 81 rounds, 111 accepted drafts and 192
committed inputs. Each had 80 T3 passes and one T2 fallback: 90 eligible collapses
per T3 explains exactly 7200 candidate calls. Native counts were 880 B3/22 B1;
leaf hits/misses were 1632/1122 in every arm. No acceptance change caused the gain.

| Arm | Draft ms | Replay ms | Commit ms | Cleanup ms |
| --- | ---: | ---: | ---: | ---: |
| Original A | 452.079627 | 99.906458 | 64.589002 | 0.502250 |
| SIMD32 B | 457.668173 | 98.542207 | 64.898334 | 0.396834 |
| SIMD32 B | 458.160752 | 97.456205 | 65.163128 | 0.171541 |
| Original A | 458.645709 | 97.845829 | 64.950379 | 0.113875 |

All four started at 95548605432 active bytes. Peaks were
96315009184/96315173024/96315090848/96315188896 bytes. Held references and
assistant residency were equal; no reference was introduced after a measured
arm. The [complete component gate](glm5-hc-collapse-simd32-component.md) separately
proved stored/current outputs, FP32 post/comb, BF16 mixed/downstream values and
special cases, with a 27.64% eight-operation reduction. That percentage is not
model decode improvement. Pre arithmetic was source-inspected, not separately
materialized or directly compared.

## Memory scope and offline existing ledger

The private harness inherited a **118111600640-byte memory limit (110 GiB)** and
115448725504-byte wired limit. It reserved MLA growth against wired-minus-active
headroom; it did **not** execute the full HTTP admission ledger. The result must
not be described as two fixed 115 GB limits or as proof that all HTTP bill guards
ran. The runtime helper adds no global storage/admission bill or reserve credit.

An offline reproduction of the unchanged HTTP formula for 8192 input/192 output,
chunk2048/async2, current flags and A6 gives the following conservative amounts:

| Existing bill | Bytes |
| --- | ---: |
| Recurrent, latent and assistant cache | 1686962176 |
| Activations | 3221225472 |
| Existing attention/tree scratch and extra allowance | 557842432 |
| A6 expansion | 536870912 |
| MLA permutations | 805306368 |
| Packed32 plus cadence | 536870912 |
| Index, cluster and native decode | 52953088 |
| `requestReserve` total | 7398031360 |
| Additional existing `plannedGrowth` | 105250816 |
| Full existing HTTP reserve | 7503282176 |

All bills remain, including assistant cache, native scratch, both packed banks,
expansion/permutations and the fixed allowance. Rounded latent/pool capacities
are 8448/2304. Adding the entire reserve to the observed active baseline, already
including assistant and held references, gives 103051887608 bytes; adding it to
the largest observed peak gives 103818471072. These deliberately conservative
sums leave 12396837896/11630254432 bytes below the wired/working limit. They do
not subtract existing live caches/references or establish private runtime
admission coverage. HTTP qualification uses the existing actual admission path.

Evidence key `glm53-hc-collapse-model-8k-20261003` retains every arm/phase/ID,
parity/engagement status, source/library/flag hashes, peak, offline ledger and
lifecycle records. Numeric model binary SHA256 is
`22cdd5ed3c3349edf6470e006a78c464136b5206cd2aff51e8a0fbd82c890296`;
input SHA256 is
`09d9fa6146cb2c6b9fc0c33a7f20deca26b75a6f6b9b68b877e3fe1811edcb57`.
Later public formatting/imports have separate validation hashes. Foreground
QoS, maximum fans/quiet idle and one per-job lock were used; process ended
exit0, lock released and fans automatic. Private evaluator/root/fixture probe
were hash-archived and removed; the helper and small public directed primitive
probe remain for full-suite/CLI validation. No source variant or second model
comparison followed this gate.

## Public regression validation

A small self-contained directed primitive test covers exact mixed/post/comb
outputs, NaN/Inf/signed-zero/extreme coefficients, scoped policy/counter behavior
and T2/F32 original fallback. It loads no model or external fixture. Both test
artifacts compiled ReleaseFast, then the full ReleaseFast suite passed with
zero test output under the quiet guard. Evidence key
`glm53-hc-collapse-validation-20261004` retains commands, source hashes, logs
and terminal status. HTTP qualification remains pending.

The ReleaseFast CLI build also passed. Frozen CLI SHA256 is
`102cb8d5efcf279295ea059201eb0a88c3e7d85321a68dced625c0b4c6858ae3`,
built from checkpoint `99547a26` plus the recorded final runtime/test patch.
The forthcoming HTTP run will identify its runtime commit and verify this hash
before and after each job. No installed library or cache precision changed.
