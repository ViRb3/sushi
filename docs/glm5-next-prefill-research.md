# Next prefill research: bounded KDA temporal working set

Source-only recommendation after checkpoint `f06a035c`, accepted runtime
`e1597cc2`: one strict T2048 KDA prework-plus-recurrence lifetime experiment,
using fixed 128-token temporal tiles. Leave all full-T2048 projections,
retained FA/GA/beta cluster, R4 value-row geometry, post normalization and
output projection unchanged. No prototype, build or GPU run was made.
This is distinct from rejected R8 recurrence tuning, QKV joins, near-full
eligibility, indexed/full-history attention and shared retrieval policies.

The accepted path in `KdaLayer.applyReference` constructs complete QKV/raw A/
beta planes, calls `glm5_kda_prework.apply` once, then runs the current
`glm5_kda_value_rows.run(...,4)` over 2048 steps. Its prepared q/k/v are BF16
`[1,2048,64,128]`; decay is the same shape in FP32 and beta is BF16
`[1,2048,64]`. Those five outputs occupy 160.25 MiB before convolution tail,
while recurrent state is 4 MiB. They remain reachable until recurrence and
caller graph reclamation. The current R4 shader carries FP32 state and reads
the same Q/K/decay for 32 independent value groups per head. A smaller live
working set could improve reuse and allocation pressure; source alone does
not establish cache misses or measured memory traffic.

The cold synchronized T2048 profile charged 500.232 ms to all 34 complete KDA
layers out of 1788.124 ms. It groups projections, prework, recurrence and post,
predates later scheduling, and is not normal HTTP attribution. The separately
recorded R4 recurrence cost around 3.62 ms is another scoped component, not a
removable 500 ms budget. Existing isolated schedules and the later R8 attempt
already show that small recurrence-only improvements need not matter.
This hypothesis targets prepared-plane lifetime across the complete layer,
not dot speed or an unmeasured removable percentage.

Preserve full-T2048 projection outputs and their native math. For each of
sixteen contiguous 128-token pieces, slice original joined QKV/raw A/beta,
run the existing prework body, then the unchanged R4 recurrence. Carry its
FP32 final state into the next piece. The proposed lifetime is explicit:
construct tile 0 prework/R4 from the caller state and submit its Y/state;
construct tile 1 from tile 0's state handle and submit its Y/state. Settle both
Y results and the second state once, retain only the two small Y outputs and
the latest FP32 state, then release both tile scopes, prepared arrays and the
older boundary state. Repeat eight pairs. Error cleanup settles every pending
output before releasing its graph. Holding all sixteen unevaluated scopes is
excluded: slicing alone would not reduce their eventual live storage.

This adds eight internal waits per KDA layer; all belong to the complete-layer
and model cost, including any lost outer async2 overlap. Collected Y pieces
still sum to 32 MiB until concat. Run the original full gate/post/output path
afterward, keeping its full-T2048 geometry and existing math. Unsupported
lengths, formats and configurations keep the current path; no tile-size sweep.

Interior convolution history must come from the original preceding three
raw QKV rows, not the previous tile's exported convolution tail. The current
prefill `conv_out` uses a BF16 add-zero boundary; reusing it can change signed
zero before later convolution. The first piece uses exactly the caller's
original conv state (or null), and the final piece supplies the original
last-three-row cache result. Existing R4 continuation tests at irregular
boundaries already support state round-trip parity, but do not prove this
complete prework/recurrence orchestration.

| Logical storage/work | Current → proposed |
| --- | --- |
| Prepared q/k/v/decay/beta | 160.25 MiB → 10.015625 MiB per 128-token piece; at most 20.03125 MiB for two |
| Persistent FP32 state | 4 MiB, unchanged; additional transient boundary states |
| Required whole Y | 32 MiB, unchanged; concat may add another 32 MiB |
| Prework + recurrence launches | 2 → 32 per layer |
| Additional state read/write spans | 120 MiB per layer at 15 interior boundaries |

The extra state spans total about 3.984 GiB across 34 layers per full chunk,
before Y concat and small convolution views. Allocation, metadata, eight
pair settlements and lost inter-layer overlap may erase any working-set gain.
All arithmetic/global Q/K/decay read requests still occur. These numbers are
logical spans, not measured DRAM. Keep current conservative admission until
actual peak and lifetime are proved; do not reduce a bill from this table.
No new persistent weight bank, F32 cache, small-weight conversion or larger
recurrence/private accumulator is needed.

One decisive component uses the original stored layer-zero KDA tensors with
full T2048 BF16 activations and nonzero BF16 conv/FP32 state. First compare each
prepared tile against slices of the current monolithic prework, all Y,
complete layer output, final FP32 state and final BF16 convolution tail
bit-for-bit, including signed-zero/boundary and cold-state cases. The original
large projection geometry must engage identically in both arms. Then one
three-warmup/eleven-pair whole-layer test includes projection/dequantization,
all prework, recurrence, state carries, tile settlement, concat, gate/post,
output projection, endpoint evaluation and every free. Record peak/ownership
and actual kernel counts. No prepared-input-only timing can decide acceptance.
Stop a noisy/slower or mismatched candidate without another schedule variant.

A clear component winner still needs one same-input, equal-reference-memory
long-prefill model ABBA and exact logits plus complete BF16 MLA/FP32 KDA state
through continuation. Keep accepted flags and chunk 2048/async2 unchanged outside
the candidate. The goal gap is not closed by a lifetime argument or a cold
profile subtotal; model performance decides whether this bounded experiment
merits an optional integration.
