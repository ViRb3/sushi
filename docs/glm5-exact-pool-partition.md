# Rejected exact pool selector

The isolated BF16-key radix cutoff/stable-compaction/native sort of 512 survivors
candidate is rejected. It preserved the pinned ordered selector and attention
bits, but complete attention was 1.62% slower, with only 2/11 paired wins.
No model evaluation, radix/layout variant, precision restoration or runtime
integration follows. Accepted runtime/defaults stay unchanged.

| Inclusive T2048 attention, actual 16K planes | Median |
|---|---:|
| Original full-axis GPU partition | 131.010292 ms |
| Exact 512 survivor selector | 133.128416 ms |

Three warmups preceded eleven fresh AB/BA pairs. Median paired slowdown was
1.63%. Timing included unchanged NAX scoring, complete cutoff/compaction/sort,
expansion, gather, native SDPA, two-graph settlement, output collection,
endpoint evaluation and free. The private original-selector cadence first
matched production attention bits; both timing arms retained the same references.
Partition-only cost remains unisolated. This is no speed/quality verdict on a
new routed kernel or a different retrieval policy.

All 24,576 directed ordered IDs and 98,448 expanded IDs matched pinned GPU
partition, including negative values, both zeros, infinities, NaNs, BF16
neighbors/subnormals, cutoff/block-boundary ties and partial pool counts.
Actual fixture proofs matched all 1,048,576 ordered IDs, 4,200,448 expanded IDs
and 67,108,864 BF16 attention values. Three focused tests passed. Dtype/source
precision, query/history bounds, settled-stride, CPU and default-off guards
passed. Timing candidate engagement was exactly 1792 selector calls.

The conservative bound remains 2 MiB per selector graph, at most two graphs.
Whole-call high-water increment was 268,435,456 bytes above resident fixture
plus held production/private-control outputs; it includes candidate collection
and output, not net candidate-versus-control overhead. Directed high-water
above its held-control baseline was a raw 4 bytes, influenced by allocator/
previous-command lifetime; it does not establish a four-byte selector footprint
or justify reducing reservation. No admission was integrated.

Two proof-only repairs are preserved with the evidence. First, the original
`[16,512]` partition slice had actual strides `[3584,1]`; flat CPU readback
mistook row 0 ranks 512 onward for row 1. Materializing both arrays yielded
`[512,1]` and closed that false mismatch. Second, unscheduled MLX transpose
reports placeholder row-major strides. The directed negative-view proof now
settles only that view before its admission assertion. This does not imply
arbitrary lazy views reliably decline a host stride guard. The selector's
algorithm/math never changed; no host wait was added to its timed path.
Named bound/guard failures and a directed-pass prerequisite prevent timing
after a real failed proof. Neither technical attempt produced timings.

Artifact `glm53-exact-pool-partition-20261003` retains all raw pairs, source/
probe/root snapshots, earlier technical attempts, actual readback strides,
ReleaseFast build/run logs and hashes. Final binary SHA256 starts
`51e8bf1324d066fd`; source was based on `07f020bf`. Foreground `taskpolicy -a`,
exclusive per-job lock, confirmed 5356/5767 RPM fans and 38.88°C after ten idle
seconds were recorded. Exit 0, GPU released promptly, fans automatic. The owned
helper, probe and private root were archived and removed after rejection; this
worker edited no existing scorer/attention seam and made no default change.
Only this result doc remains for the coordinator’s documentation commit.
