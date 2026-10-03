# Rejected fixed short-row IndexPool NAX scorer

The isolated M128/K128/C2048 B1/B3 scorer is rejected. Its complete native
three-branch attention median was 1153.166 µs for original scalar scoring versus
1314.917 µs for the candidate: 14.03% slower, 0/11 paired wins after three warmups.
No tile/layout variant, restoration, production integration or model run follows.

All 84,474 FP32 score bits, 36,918 ordered selected IDs and 98,304 BF16 attention
bits matched within the coupled mode, including natural serial prefix-versus-
suffix placement, distinct suffixes, mixed eligibility, above 8192 pools, ties
and negative weights. The sampled scalar score/selection drift was zero; this
is not a general scalar-equivalence claim. Measured complete-attention peak
increment was 7,549,406 bytes, within combined scorer/attention reservation 16 MiB.

Artifact `glm53-indexpool-decode-nax-20261003` retains raw eleven pairs,
source/probe/root snapshots, input references and ReleaseFast binary SHA256
starting `0195959c23dc1216`. Both focused tests passed under foreground
`taskpolicy -a`, exclusive GPU owner `glm53-indexpool-decode-nax`, maximum fans
and ten seconds idle after 42.90°C. Runtime was pinned MLX 0.32.3 and patched
mlx-c 56b2d39. Only the worker's untracked helper/probe/root were removed; existing
attention, verifier and prefill-scorer source were never edited by this worker.
Accepted runtime and defaults are unchanged.
