# Exact joint M3 QKV dispatch research

Source-only audit at checkpoint `de9c698e`, accepted runtime `e1597cc2`.
Recommend one bounded complete-layer experiment: replace three **current
hoisted M3** A6 projections and their concat with one bank-grid dispatch
writing the same joined BF16 result. No prototype, build or GPU run was made.
This is a scheduling/copy opportunity with an unmeasured, probably modest
ceiling; it is not a forecast of reaching 60 decode tok/s.

The current path is `glm5_dflash_model.verify` → `glm5_dflash_kda.applyLayer`
→ `linearRows(.affine_rows_ffn)` → `glm5_dflash_qmm.project` →
`glm5_dflash_a6_hoist.project`. `applyLayer` invokes Q, K and V independently,
then concatenates `[1,3,8192]` outputs into `[1,3,24576]`. The current hoist
admits only BF16 `[1,3,4096]`, U32 A6/group128 `[8192,768]` weights and
materialized row-major BF16 `[8192,32]` scales/biases. Other shapes retain
current fallback. Output projection, FA/FB/GA/GB and beta are outside this hook.

Use thread grid `(65536,1,3)` and group `(64,1,1)`: 1024 output groups for each
of three banks, 3072 total, matching the current three 1024-group launches.
Select Q/K/V pointers once from `threadgroup_position_in_grid.z`; retain
`output0=tg.x*8+simd_group*4` and `token0=tg.y*3`. Each thread owns only its
current `result[3][4]`, `local[3][8]`, `sum[3]` and per-output 12 coefficients.
There is no bank loop or `result[3banks][3rows][4outputs]` live array. Store to
`y[(token0+m)*24576+bank*8192+output0+r]`, producing contiguous token-major
Q/K/V order directly. Bank pointers are separate original buffers, not a
concatenated/repacked weight plane.

Keep the current K256 traversal, BF16 quartet-sum boundary, scaled locals,
12 masked-coefficient products, FP32 scale/bias updates, `simd_sum` and BF16
stores verbatim. Bank selection is uniform within a group, but pointer selection
and compiler scheduling still require bit proof. Input reads remain separate
across bank groups; issued dot work and weight traffic are unchanged.

| Storage or metadata | Current → proposed |
| --- | --- |
| Resident original Q/K/V banks | 75 MiB → 75 MiB; each 24 MiB U32 weight plus 1 MiB BF16 grids |
| Shared M3 input | 24 KiB, unchanged |
| Joined raw result | 144 KiB, unchanged |
| Separate Q/K/V output planes | three 48 KiB planes → none; 144 KiB transient allocation removed per layer |
| Kernel input handles | three vectors of 4 → one vector of 10; original nine bank arrays plus input |
| New metadata/state/weights | none; bank index comes from grid z |

Across 34 KDA layers per M3 round this removes 68 launches and 34 QKV concats.
The removed temporary planes sum to 4.78125 MiB of logical allocations per round,
not a measured peak or DRAM saving. No permanent decoded bank, additional leaf
state or new small-tensor copy is needed. Preserve all retained BF16 matrices,
BF16 convolution/out-norm weights, FP32 `A_log`/`dt_bias`, FP32 persistent KDA
state/accumulators and the existing prepared banks unchanged. Input/weight
contiguity and availability guards must match the accepted helper, including
any generic wrapper copy costs in the timed call.

Engagement is established: the recorded 64-round request had 6528 hoist calls,
exactly 64×34×3. The [accepted hoist component](glm5-dflash-a6-hoist.md) measured
491.459 µs for one resident production Q bank, versus 560.833 µs before hoisting;
that isolated apply/evaluate/free result is not current full-model QKV cost.
The old synchronization-perturbed 79.968 ms/six-round QKV marker omitted the
accepted hoist and forced child waits. It is neither a 13 ms removable round
budget nor a valid baseline here. Current native 32K predictable verification
averaged 58.37 ms/round, with 43.00 decode tok/s in a separate-boot qualification.
The [earlier one-row join](arch-glm5-next.md) lacked material model benefit and
still returned separate planes for concat. Neither result assigns a speed to
this M3 direct-output candidate. Only launch/host construction, concat traffic
and their dependencies can improve; all projection arithmetic remains.

One worker can own new joint helper/probe files. Root owns the narrow raw-QKV
caller hook and any production integration. Reuse the archived original
layer-zero 23-tensor fixture from `glm53-t4-kda-20261003` (111903232 bytes), with
M3 synthetic BF16 activations and nonzero BF16 convolution/FP32 recurrent state;
no new full-model capture is necessary. Baseline must explicitly engage the
current hoist and retained leaf, with all remaining layer math unchanged.

First compare all 73728 joined raw BF16 values against three current hoisted
banks plus concat. Decline M1/M2/M4, output-bank, mixed/dense/unsupported grids
and unmaterialized/strided weights. Then prove complete KDA output, prework,
retained FP32 endpoint and accepted replay convolution/FP32 state for chain
`[-1,0,1]` and fork `[-1,0,0]`, including cached hit and replay miss. Use one
inclusive complete M3 KDA-layer/commit component, three warmups and 11 fresh
alternating pairs, including graph construction, endpoint evaluation and frees;
no QKV-only timing can decide acceptance. Record baseline three-hoist and
candidate one-joint engagement and actual transient peak. Stop a losing/noisy
component without another bank layout, row-count or precision variant.

Only a repeatable inclusive winner receives one current-stack matched model
ABBA with identical frozen input IDs, one loaded target/assistant, equal retained
reference memory, all 192 delivered IDs and full valid target-state equality.
Keep draft policy and attention mode unchanged; include all rounds, acceptance,
phase costs and cleanup. A model gain smaller than control drift is rejected.
Do not adopt from the component alone or repeat the unhelpful one-row ladder.
