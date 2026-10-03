# Exact joint M3 QKV dispatch: rejected

The bank-grid dispatch preserved all tested raw QKV, complete KDA and replay
state bits, but did not establish a repeatable inclusive speed gain. Median
complete layer plus commit was 547.500→544.375 µs (0.571% by arm medians); paired
median gain was 0.267%, with only 6/11 wins and host outliers. No variant or
actual-model arm follows. This does not establish model throughput improvement.

| Complete M3 KDA layer and chain commit | Median |
| --- | ---: |
| Current three R3 A6 hoists plus concat | 547.500 µs |
| One bank-grid R3 join | 544.375 µs |

Both arms retained the accepted exact M3 hoist baseline, original small
BF16/FP32 tensors, prepared convolution/decay, retained dense-row projections
and first-path FP32 leaf. Candidate changed only raw QKV construction. Three
warmups preceded eleven interleaved fresh layer/tape/commit/evaluation graphs,
including all frees. No samples or losing pairs were removed. Component input
was the original 23-tensor layer-zero fixture (111903232 bytes), fixed BF16
activations, nonzero BF16 convolution history and FP32 recurrent state; no
whole-model load or per-child forced profile was used.

The candidate used grid `(65536,1,3)`, group `(64,1,1)` and separate original
Q/K/V bank pointers selected by group z. Each thread retained only existing
R3 accumulators and coefficients. Contiguous `[1,3,24576]` was written directly,
without repacking weights, extra retained state or live three-bank accumulators.
It preserved K256 order, BF16 quartet sums, masked products, FP32 scale/bias
updates, SIMD reduction and BF16 stores. Issued dot work stayed unchanged.

All four green tests passed. The raw comparison covered 73728 BF16 values
against three current hoists plus concat. Full chain `[-1,0,1]` and fork
`[-1,0,0]` proofs matched outputs, prework, convolution history, retained FP32
endpoint and every accepted replay convolution/FP32 state against current and
independent serial ancestors. Cached hit reused its retained allocation;
misses followed exact replay. M1/M2/M4, output projection, dense/mixed A8,
F32 grids, lazy and materialized strided weights declined. The red helper
returned null and failed with missing raw output/zero joint engagement before
being restored for the green build.

Timed engagement was 42 baseline hoists and 14 joints, 140 unchanged retained
small-projection calls and 28 leaf hits, with no misses. Candidate peak growth
was 5261668 bytes above resident weights/input/state; the existing 4194304-byte
FP32 retained state was unchanged. Removing two launches and one concat per
layer did not translate into a robust component gain on the current stack.

Evidence key `glm53-joint-m3-qkv-20261003`, HEAD `fe37527c` with hashed WIP,
records original tensor hashes, red/green source and binaries, current runtime,
commands/flags, raw pairs, guards and thermal/fan telemetry. ReleaseFast,
foreground `taskpolicy -a`, exclusive per-job locks and confirmed maximum fans
with the required idle preceded both runs. Locks were released between red
and green builds and after the final run; automatic fans were restored.
The rejected helper/probe and root-owned delegation are archived for evidence;
cleanup removes the candidate without changing accepted runtime behavior.
