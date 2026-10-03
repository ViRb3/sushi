# Grouped lane middle/down component

Rejected candidate for the [temporal/grouped round](glm5-temporal-grouped-fusion-round-plan.md).
Full-bank exactness passed, but the complete-chain comparison lost all pairs.

`glm5_grouped_middle_down.moe` accepts only the current B1/T3/H4096/I2048,
E288/top8/n36/MCG/W12 geometry. Lane gate/up preparation and grouped gate/up
are unchanged. The fused body retains original inline ballots and the complete
outer singleton/pair down branches, including every nested rate branch.
There is no inner-else text slicing, routing prepass, weight copy or split.

Each leader computes one or two original lane-middle results into separate
threadgroup F16 planes, preserving the store boundary before any down dot.
After a uniform barrier, eight fixed output tiles execute the original dot,
FMA/reduction order and F16 down stores sequentially. Singleton reads only its
first prepared plane; pairs read both. Weighted BF16 finish stays current.
The separate98304-byte global middle plane/command disappears, while repeated
middle construction and static12KiB shared storage are charged by timing.

The stage probe emits the exact staged F16 plane only from output group0 and
poisons an unused second member with F16 NaN before the down dot. This diagnostic
copy/poison is excluded from the regular source and timed arm. Both diagnostic
and **regular** down outputs must match current separate middle/grouped-down
bits. Structured singleton/pair/triple and explicit positive/negative-zero
input controls use one original full bank; all actual42 layers use current
saved inputs/full E288 banks. The private harness owns the516096-value final
BF16 proof and one inclusive3-warmup/11-pair full-chain comparison.

No tile tuning, down split, extra endpoint layout or precision restoration is
part of this candidate. Root owns runtime seams/admission/model gates.

The [current full-bank comparison](glm5-grouped-fusion-current-replay.md) passed
2064384 actual staged-middle F16 values,4128768 down values and the same count
for regular-versus-diagnostic down behavior. Four structured full-L20-bank
singleton/pair/triple/signed-zero cases passed196608 middle and393216 down
values, including unused-member NaN poison. Both arms matched all516096 saved
routed BF16 outputs. Two filtered tests passed; no restoration was added.

Three warmups and eleven complete42-chain pairs measured17.953958ms current
versus20.023042ms fused:11.52% slower, zero wins. Original layer/bank order,
eleven absolute async4 submissions/final settlement, all construction/evaluation
and frees were included. Removing a command and global plane did not compensate
for this combined schedule's costs. Repeated middle construction,12KiB shared
storage and reduced down-grid concurrency are known changes, not separately
measured causes or evidence of spilling.

No model ABBA or tile/split variant followed. Helper/probe/private roots and
source manifests are archived; root restores original exports separately.
No runtime commit or default change follows the rejection.
