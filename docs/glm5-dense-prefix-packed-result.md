# Cold packed MLA rejection

Reject the fixed cold T2048/B16 absorbed candidate. It passed all four directed
tests but made the complete first MLA layer 6.23 times as slow. No model gate or
geometry/precision variant is justified.

The control already uses expanded BF16 K/V and native causal D256 SDPA. The
candidate reused original A6/group128 head-batched Q absorption and value
projection, explicit complete ascending causal IDs, the existing BF16 masked
gather and native D512 attention, with exactly two pending B16 graphs. Original
output projection, index/cache append, weights and retained small tensors stayed
unchanged. This was not a missing-native-dispatch fix.

| Directed evidence | Result |
| --- | --- |
| Ordered ID slots checked | 4,200,448 |
| Unique live causal key memberships | 2,098,176 |
| Earlier-query BF16 bits with future NaN/Inf versus finite control | 67,076,096 exact |
| Valid historical nonfinite values | Remain observable |
| Complete-layer BF16 cache/state values | 1,114,112 exact |
| Materialized cache strides | 512 / 1 |
| Cold geometry, fallback and checked reserve | Passed |
| Green test suite | 4 passed, exit 0 |

Complete output drift versus current expanded native MLA: 4,588,672 of
8,388,608 BF16 values differed; relative L2 0.00358060 and maximum absolute
error 0.0078125. Neither arm produced nonfinite values. This records the expected
reassociation/native rounding change; it is not a quality acceptance result.

After three warmups per arm, eleven alternating pairs included the entire
`Mla.applyMode`: projections, cache/index append, copies, ID construction,
gather, two-graph waits, native attention, value and output projections,
evaluation and frees. Control median was 13.179958 ms; candidate median was
82.054291 ms, a 522.57% increase. Candidate won 0/11 pairs. Paired median
increase was 521.78%, and every pair increased 508.24–531.02%.
The finite proof recorded a 973,234,196-byte candidate peak above the loaded
fixture plus held control result, including candidate output, retained owner
arrays and drift/state proof operations. It passed the separate 1 GiB directed
peak guard; this is not an isolated native-kernel footprint.

Provenance: `glm53-dense-prefix-packed-20261003`, original layer-three 26-tensor
fixture (99,030,528 bytes), synthetic frozen BF16 input seed 43201. The artifact
retains source-header/payload hashes for every original tensor, exact compiler
commands, source/binary/runtime stamps, raw timings/drift/safety records and
fan/thermal logs. Red compiled and failed at absent attention/candidate behavior.
The first red build lacked its private output directory; the first green build
needed `try` for Zig's error-union-to-optional conversion. Both compile failures
are preserved and neither changed arithmetic or scheduling. Qualified green
PID 57369 exited 0, released the GPU lock and restored fans to auto.

Helper SHA256 `c89855bd4bef945962d00be299ee2b489824b5d047ef1a1b01caaa09ba8b0816`; probe SHA256 `e3d0f29381f654a142789dfc5de9d90863f76b1defd7a527366d6572cd19e0fc`.
The private helper/probe/root are archived and removed. Root restored runtime
delegation and reservation; no runtime change is accepted.
