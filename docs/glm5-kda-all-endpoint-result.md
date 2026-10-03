# All T3 KDA endpoint retention: rejected

The complete-layer/commit surrogate did not establish a repeatable gain:
47.152875→46.443917 ms per 81 commits, 1.504% by arm medians and 1.262% by paired
median, with only 6/11 wins. Pair changes ranged from 3.383% slower to 4.321%
faster. This independent noisy timing gate rejects the candidate; no variant
or model run follows. The full green test process also exited 1 because the
separate sibling-release proof remained unresolved.

The candidate appended three endpoint stores after the entire original tree
recurrence loop. It preserved decay, dots, state updates, SIMD reductions and
BF16 Y stores, with three separate `[1,64,128,128]` FP32 buffers. It replaced
one retained plane with three, adding 8 MiB per T3 layer and 272 MiB across 34
verification tapes. Original small BF16/F32 weights, persistent state and
T1/T2 fallback remained unchanged. Root admission separately passed red/green
for the additional 285212672-byte reservation.

All normal timing work used the original 23-tensor layer-zero fixture,
synthetic BF16 activations and nonzero BF16 convolution/FP32 recurrent state.
Both arms enabled current M3 QKV hoist, retained dense-row projections and
current first-leaf baseline. Each sample included 81 fresh complete KDA layer,
tape, normal commit, convolution tail, endpoint evaluation and all frees.
Three warmups preceded 11 interleaved pairs; no outliers or losing pairs were
removed. There was no raw replay-only or per-child timing claim.

The surrogate had 80 T3 cases and one final T2. Its constructed interleaved
parent/commit mix was 43 chain length-three hits, 20 chain length-two misses,
13 root misses, 4 fork first-child hits and one T2 hit. This matches the saved
48/33 baseline hit/miss and 43/25/13 accepted-length counts, but is not captured
parent topology. Every measured baseline sample reported 48 hits/33 misses;
every candidate sample reported 81 hits/zero misses. Avoiding misses did not
produce a robust inclusive gain after charging endpoint stores and commit work.

The complete-layer test passed: chain/fork outputs, prework, all endpoint FP32
states, every accepted convolution tail/state, pointer aliases, invalid paths
and exact partial T1/T2 fallback matched current and independent serial
ancestry. Raw recurrence numerical comparisons occur before the failed guard
and showed no mismatch: 49152 BF16 Y values and 6291456 FP32 endpoint values.
The separate raw test then returned `TestUnexpectedResult` in its alias-release
guard group. Its optimized log did not record the assertion site or memory
values, so the exact availability/pointer/release predicate cannot be named
from the evidence. No cause is inferred and no rerun was performed.

Unresolved assertions cover Y-only evaluation making sibling states available,
independent endpoint pointers, selected pointer alias, releasing at least 8 MiB
of siblings, retaining at most 4 MiB plus 64 KiB allowance, and final release.
Therefore mathematical parity and normal selected-state alias checks passed,
but backing-release ownership was not fully qualified. The green process
reported three passing tests and one failing test, not a complete suite pass.
Candidate peak growth in the complete 81-commit directed scope was 13797836 bytes
above resident weights/input/state; this is not a qualified model memory bill.

Artifact `glm53-kda-all-endpoints-20261003` preserves the null-helper red failure,
qualified source, exact raw/full-layer proofs, assertion sites, all paired data,
original fixture hashes, binary/runtime/flags and thermal telemetry. Source
checkpoint was `e88b2aa4` plus hashed WIP. ReleaseFast and foreground
`taskpolicy -a`, per-job GPU locks, confirmed maximum fans and required idle
were used. The green process PID 20974 ended; locks were released between builds
and after execution, and fans returned automatic. Root restored its runtime
seams/admission; the isolated helper/probe/private root were archived and removed.
No runtime feature, default or source commit was retained.
