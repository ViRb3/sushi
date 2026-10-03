# Steady 4096 quality rejection

The fixed 4096 scheduling candidate is rejected. Its constructed expert/trunk
component proofs were exact, but the actual 16K prefix gate changed final logits
and valid cache/state bytes before continuation or any timed ABBA arm. The
unchanged two-prompt numerical screen then failed its original per-prompt bounds.
Accepted runtime `4fcb541e` remains the baseline; this is no throughput result.

Both original nonrepeated 16,384-ID code/prose inputs used eight 2048 chunks for
the control and 2048/4096/4096/4096/2048 for the candidate. Native B1, packed32,
HC and all accepted helpers remained on; mini and tree core remained off. Each
prompt retained all 24 declared last-chunk positions and 192 baseline-forced
predictions (191 unchanged native scalar forwards). Final chunks remained 2048,
so labels and sampled absolute positions were unchanged. The last-layer residual
capture and individual head rows are diagnostic work, outside performance.

The gate requires all 216 rows per prompt: mean KL<=0.01, max KL<=0.15,
top1>=95%, mean forced NLL increase<=0.02 and new nonfinite=0. Bounds were not
changed after results. Both prompts completed even after the first failed.

| Prompt/group | Rows | Mean KL | Max KL | Top1 agreement | Mean NLL delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code, all | 216 | 0.03371849 | 1.27872750 | 209/216 (96.759%) | -0.03829337 |
| Code, late prefix | 24 | 0.23830220 | 1.27872750 | 20/24 (83.333%) | -0.44534747 |
| Code, forced tail | 192 | 0.00814553 | 0.10063949 | 189/192 (98.438%) | +0.01258839 |
| Prose, all | 216 | 0.00480489 | 0.18176161 | 211/216 (97.685%) | +0.00955034 |
| Prose, late prefix | 24 | 0.02878240 | 0.18176161 | 20/24 (83.333%) | +0.09064120 |
| Prose, forced tail | 192 | 0.00180771 | 0.03559333 | 191/192 (99.479%) | -0.00058601 |

All 432 rows were finite and scored. Code failed mean and maximum KL; prose
failed maximum KL. Tail-only summaries do not replace the declared full gate.
No numerical retry, scalar-score restoration, changed pool threshold or further
schedule variant followed.

Both prompts recorded identical engagement counts for their corresponding arms:

| Counter | Control | Candidate |
| --- | ---: | ---: |
| Prefix chunks | 8 | 5 |
| Expert grid | 336 | 210 |
| A6 dense expansion | 1,088 | 680 |
| KDA cluster | 272 | 170 |
| MLA query/value batch, each | 77 | 44 |
| Packed32 attention | 4,928 | 4,928 |
| Packed cadence | 77 | 44 |
| IndexPool NAX | 2,816 | 4,224 |
| Native B1 forced forwards | 2,101 | 2,101 |

Both forced states ended at 16,575. The first dense 2048 boundary remains fixed.
An identified source boundary is whole-chunk pool-count admission: appending the
candidate chunk through 14,336 enables the unchanged 3,584-pool NAX scorer for
query offsets 10,240–12,287 earlier than control. The original 8,192-pool ceiling
also remains. This explains changed engagement, not the observed mismatch's
measured root cause; there was no extra attribution run.

One target was loaded, with no assistant. Actual loaded active bytes were
93,535,640,312; peak active bytes were 97,903,035,404. The complete conservative
ledger was 16,210,998,272 with all old allowances retained and no credits;
peak plus ledger was 114,114,033,676 under both fixed memory/wired limits of
115,448,725,504. The ledger includes two independently evolving state owners,
4096 activations/full permutation and clustered outputs, all original scratch,
4 MiB wider metadata, 149,304,320 CPU-logit bytes and the original margin.

ReleaseFast build and frozen-source postcompile/prelaunch hash checks passed.
Accepted installed library hashes and binary remained unchanged. Foreground QoS,
confirmed maximum fans, idle and a per-job lock were used. The same live job
completed with expected exit 1 (`Steady4096QualityRejected`), then returned fans
to auto and released its lock. No model-performance arms or teacher/quant-pack
KLD claim were made.

Artifact key `glm53-steady4096-quality-20261004` preserves all 432 rows, inputs'
SHA256 records, commands/flags, compiled source copies/hashes, protocol, limits,
complete result, logs and cleanup. Corpus SHA256 is
`4a26fe8d323bfea64bd35368fc863c83d4c43e6a706b3bf190614e4efea16f78`.
Compiled source HEAD is `daf9eb23` with the documented 4096 WIP; binary SHA256 is `8da3f08baacd9f802193ea397095a7328bb98b3bf8175b77b8a9c35e847c98c6`.
