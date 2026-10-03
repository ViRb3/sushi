# Rejected grouped middle/down fusion

Grouped middle/down fusion preserved every stage/output bit but lost all eleven
complete-chain pairs. It is rejected; no model ABBA, extra tile/layout variant
or default change follows. Accepted runtime remains `e1597cc2`.

| Complete current first-round 42-chain replay | Median |
|---|---:|
| Current serial/grouped/lane group2 | 17.953958 ms |
| Fixed grouped middle/down fusion | 20.023042 ms |

The candidate was 11.52% slower, with every pair losing 10.90–12.88% after three
warmups. Eleven fresh AB/BA pairs included preparation, allocations, current
gate/up, fused middle/down, weighted finish, eleven absolute async4 submissions,
final endpoint settlement and all output/vector frees. CPU comparisons were
outside timing. Extra middle reconstruction and 12 KiB shared storage were
included; this test does not isolate their individual costs or establish a
whole-verifier attribution.

Both arms retained identical immutable current 8K round 0 X/IDs/scores/Y and
original full E288 banks, layer 3–44 order. No recapture, assistant load or
compact-bank substitute. The replay population was 1008 assignments, 748
singleton/130 pair leaders (878 total), distinct from the three-round aggregate.
A6 remains default; A4/group64 remains supported optional.

All 42 actual-layer proofs passed: 2,064,384 staged-middle F16 values and
4,128,768 down values, including exact regular-versus-diagnostic down output.
Structured singleton/pair/triple/signed-zero controls on original L20 passed
196,608 middle and 393,216 down/regular-vs-probe values. Both complete arms
matched all 516,096 captured BF16 routed values. Guard checks passed; both
filtered tests completed, exit 0. Original F16 boundaries, FP32 accumulation,
ballots/reduction order and weighted finish remained unchanged; no restoration.

Raw high-water increments above the resident fixture/banks were 7,700,480 bytes
for control and 7,307,264 for candidate. They are allocator/lifetime measurements,
not evidence to reduce production reservations. No runtime seam or admission
was integrated.

Artifact `glm53-grouped-fusion-current-replay-20261003` retains every raw pair,
stage/control/whole-output proof, source/hash manifest, build/run command and
thermal/progress log. One target load only. Binary SHA256 starts
`04171df998878e77`, source checkpoint `da1d32f0`; fixed fixture SHA256 starts
`88cd99a91dcfe49d`. Foreground `taskpolicy -a`, exclusive per-job lock, confirmed
5351/5774 RPM and 38.34°C after ten idle seconds were recorded. PID 28839 ended,
lock is free and fans returned auto (manual false, TTL 0). Owned private harness/
root and producer helpers were archived and removed; only outcome docs remain
for root's commit. No further runs occurred.
