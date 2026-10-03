# All-endpoint FP32 KDA retention research

Source-only audit at `597fc5b9`, accepted runtime `e1597cc2`. Recommend one
bounded verify-plus-commit experiment retaining all three N2/T3 recurrent
endpoints. Exactness is plausible; a speed gain is unmeasured and added stores
may erase it. No prototype, build or GPU run was made. Do not change recurrence
arithmetic, tree policy, row width, small weights or precision.

The current `glm5_dflash_kda` tree kernel already computes each endpoint in
thread-private `saved[W][Dk/32]`. At T3/Dk128 each lane has three four-element
FP32 saved vectors. Each row performs the original decay multiply, K dot,
SIMD sum, delta, state update, Q dot, SIMD sum and BF16 Y store. The current
leaf variant writes only `saved[KEEP_ROW]` afterward. Keep that entire parent-
indexed arithmetic body, grid `(32,128,64)` and group `(32,4,1)` unchanged;
add only endpoint stores from the values already computed. No canonical
chain/fork recurrence rewrite or extra multistate accumulator is proposed.

`cachedLeafRow` follows the first-child path: chain `[-1,0,1]` retains row 2;
fork `[-1,0,0]` retains row 1. `Tape.replay` validates ancestry, aliases the
retained FP32 state only for that endpoint, and otherwise gathers saved
Q/K/V/decay/beta rows and invokes sequential `primitive.kda`. It always builds
the accepted three-row BF16 convolution tail. All-endpoint retention removes
only the miss recurrence and its input gathers; request cloning, MLA append,
convolution-tail gathering, capture slices, evaluation and publication remain.

The current 8K/192 artifact `glm53-current-routed-8k-20261003` recorded 81 rounds:
43 commits of 3 inputs, 25 of 2 and 13 of 1. Raw phases contain 80 T3 verifications
and one final T2 committing 2, not 81 T3 rounds. Leaf hits/misses 1632/1122 equal
48/33 complete 34-layer rounds. The final T2 already hits its retained endpoint;
for full T3, 47/80 rounds hit and 33/80 miss (41.25%). The observed length mix
therefore contains 13 root misses and 20 two-input misses; the exact parent
choices within those 20 misses were not captured. Do not invent that topology
trace from the aggregate counters.

Use three separate FP32 output buffers, each `[1,64,128,128]`, instead of one
`[3,64,128,128]` backing. A selected endpoint must alias just its own 4 MiB buffer
while `Verified.deinit` releases the other two. A slice into one 12 MiB backing
would keep unselected siblings alive in the committed request and complicate
next-round peak. Replace the current single leaf output, not supplement it.
Keep partial T1/T2 on the existing first-leaf path, unsupported geometry on the
original fallback, and all original BF16/F32 retained tensors unchanged.

| T3 retained state | Current → proposed |
| --- | ---: |
| Per KDA layer | 4 MiB → 12 MiB |
| Across 34 retained tapes | 136 MiB → 408 MiB |
| Additional live state and logical writes per T3 round | 272 MiB (285212672 bytes) |
| Selected endpoint payload after tape release | 4 MiB per layer; verify actual backing release |

Admission must add the full 272 MiB across 34 tapes, not only four pending async
layers. Stores happen every T3 verification, including already-hit rounds.
Avoided miss replay reads/writes roughly 8 MiB of state per layer; at 33/80 misses,
that is 112.2 MiB per full T3 round on average, below the additional 272 MiB writes.
These are logical spans, not measured DRAM traffic or a bandwidth forecast.
Avoided scalar recurrence, input-gather graphs and launches could still matter.
Physical register/spill demand and three-output allocation/free costs must be
measured rather than inferred from the private-array declaration.

The uncaptured-round replay clocks in this capture job averaged 1.278 ms, but the
request is synchronization-perturbed and not throughput evidence. This mixed
phase includes the unchanged work above; neither it nor the roughly 1.25 ms
headline is an available KDA savings budget. An optimistic upper bound is
less than the entire mixed replay phase before subtracting extra verification
stores. Only an inclusive normal commit decision can establish net benefit.

One component must prove BF16 Y plus all three FP32 endpoints against existing
sequential ancestry, complete KDA outputs/prework, every chain/fork accepted
convolution tail and state, retained aliases and actual unselected-buffer release,
root/shortened paths, partial
T1/T2 and invalid ancestry. Use original layer-zero stored tensors and nonzero
BF16 convolution/FP32 state. Keep current hoist, retained dense rows and leaf
baseline enabled. The new hook/bill/counters are coordinator-owned.

Then one three-warmup/eleven-pair complete layer/tape/commit test includes all
stores, selected-state alias, unselected frees and final evaluation. Weight
hits/misses and accepted lengths by the recorded 48/33 and 43/25/13 distribution,
including its T2 tail; explicitly label any constructed topology mixture as a
surrogate. The component must cover both chain and fork endpoint proofs and
cannot time miss replay alone. Stop a noisy/slower result without another
retention/layout/width variant. A winner requires one matched actual-model
N2/A6/native-attention ABBA with the frozen 8192 inputs, all 192 output IDs,
complete valid-state equality, unchanged acceptance, equal retained references,
measured peak/admission and all verification/commit/cleanup costs. No default
or source commit follows merely from eliminating replay misses.
