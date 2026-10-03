# Rejected singleton gate/up resource split

The corrected singleton gate/up split preserved all stage and output bits but
lost all eleven complete-chain pairs. It is rejected; no model ABBA, extra
layout/algorithm variant or default change follows. Accepted runtime remains
`e1597cc2`.

| Current first-round 42-chain replay | Median |
|---|---:|
| Original serial/grouped/lane group2 | 17.816458 ms |
| Singleton/pair gate/up resource split | 18.102708 ms |

The candidate was 1.61% slower; median paired loss was 1.48%, with every pair
losing 1.19–1.89%. Three warmups preceded eleven fresh AB/BA pairs. Both arms used
the same immutable recorded operands and full original E288 banks, layer 3–44
order, eleven absolute async4 submissions, final endpoint settlement and all
output/vector frees. Preparation, classification, allocations and every
kernel were included. CPU comparisons ran only in the separate proof path.
This warm saved-workload component is not whole-verifier attribution.

The fixture is the first captured current 8K T3 round only: 1008 assignments,
748 singleton and 130 pair leaders (878 total). It is distinct from the prior
three-round aggregate. There was one target load, no assistant load, recapture
or compact-bank substitute. A6 remains default; A4/group64 is supported optional.

All 42 actual-layer proofs passed: 1008 masks, 4,128,768 chosen F16 gate/up values,
2,064,384 middle values and the same count with unused families NaN-poisoned.
Structured all-singleton/all-pair/odd-triple controls used one original full
L20 bank with saved X and passed 72 masks, 294,912 gate/up values and 147,456
ordinary/poisoned middle values. Both original and candidate whole arms matched
all 516,096 captured BF16 routed-output values. Guard checks passed; two filtered
tests completed, exit 0. No arithmetic, F16/BF16 boundary or precision restoration
was changed.

Candidate extra buffers are 196,632 bytes per call. Raw high-water increments
above resident fixture/banks were 7,700,480 bytes for control and 7,110,800 for
candidate. These allocator/lifetime scope measurements do not establish a
negative memory overhead or justify removing the explicit extra-buffer bill.
No runtime admission or dispatch was integrated.

Artifact `glm53-singleton-current-replay-20261003` retains two subattempts.
`attempt01-metal-source-packaging` stopped before numeric proof/timing on a
Metal closing-brace error: body extraction matched an inner N==64 else instead
of the outer singleton/pair split. The authorized brace-depth packaging repair
preserved original body text, arithmetic, geometry and workload.
`attempt02-brace-repair` contains the complete successful losing proof, all raw
pairs/stage reports, source/hash manifest, build/run commands and telemetry.
Final binary SHA256 starts `bce52c5436701eca`, helper hash `3c9804828521f7ad`,
source checkpoint `e88b2aa4`. Fixture SHA256 remains `88cd99a91dcfe49d`.
Foreground `taskpolicy -a`, exclusive lock, confirmed 5343/5776 RPM and 41.84°C
after ten idle seconds were recorded. The process ended, lock was released
and fans returned auto (manual false, TTL 0). The private replay/root and helper
sources were archived and removed; result docs remain for root's outcome
commit. No further runs occurred.
