# Original nominal32K tail input capture

One original target forward on 33579 frozen code IDs produced the first real MLA
final-chunk input bundle: T811, P8394, offset32768, scale0.0625. Long scoring and
expert pairing were off; original2048 scheduling, BF16 compressed MLA and FP32
persistent state were unchanged. This is diagnostic input evidence, not timing,
quality or a lossless teacher capture.

The new code/prose corpus was frozen before any numeric execution. Both native
Sushi tokenizer/chat renders contain33579 IDs, retaining the early record and
final instruction. Code comes from the original nonrepeated consumer archive;
prose extends that archive with distinct consumer documents before one byte
truncation. No16K input repetition is used. Corpus SHA256:
`6ac32526f40449b84059f7116346728b26fa5547e1db1ee4f50d5346f8568b77`.
The24 affected last-chunk offsets are0/35/70/105/140/176/211/246/281/316/352/387/
422/457/493/528/563/598/633/669/704/739/774/810, followed by192 baseline-forced
predictions per prompt. Original numerical bounds remain unchanged.

| Captured array | Dtype | Shape |
| --- | --- | --- |
| q | BF16 |811×64×512 |
| index_q | BF16 |811×32×128 |
| weights | BF16 |811×32 |
| latent | BF16 |33579×512 |
| pooled | BF16 |8394×128 |
| offset, processed | U32 |scalar |
| scale | F32 |scalar |

Valid cache rows were retained; unused capacity tails were omitted. All eight
keys, dtypes/shapes and stored scalar values were independently verified from
the file header/data. The file is96,379,818 bytes, SHA256
`ada743717793309c92150c96718fbb46f320a3994c1ed916664c9211b0d4b56c`.

Loaded active memory was93535640312; peak95927694344. Full old request/growth
reserves were retained, with another256MiB diagnostic/copy allowance: bill
10955915264. Peak plus bill remained below both115448725504 memory/wired limits.
No assistant was loaded; its old cache allowance was nevertheless retained.
One model load, no warmup, original prefix and no generated continuation were used.

The capture selector first failed the missing33579/T811 tuple, then passed the
minimal tuple extension while original8192/16384 captures retained their shape.
Private packaging failures (empty test filter and a writer-name shadow) were
preserved; neither was numeric evidence. Final ReleaseFast build and process
47389 exited0 with foreground QoS, exclusive per-job lock, confirmed max fans,
required idle and automatic fan cleanup. All nine compiled source snapshots and
four private sources were hash-verified/archived before restoring the temporary
capture hook and removing private files. Accepted library hashes remained fixed.

Artifact keys `glm53-tail32k-quality-corpus-20261004`,
`glm53-tail-capture-tdd-20261004` and `glm53-tail-actual-capture-20261004` preserve
input and native formatting provenance, declared positions, commands, source/
binary/library hashes, capture, memory checks, process exits and cleanup.
The [tail component](glm5-indexpool-tail-component.md) uses these original inputs;
quality and actual-model acceptance remain separate gates.
