# Fixed 4096 trunk proof

The source-only fixed-shape extension builds on accepted runtime `4fcb541e` and
round plan `5de9c08d`. It changes A6 expansion, MLA head batching and the retained
KDA small-bank cluster eligibility and bills; arithmetic bodies and 2048 paths
remain unchanged. This is component correctness evidence, not model acceptance
or a performance measurement.

The ReleaseFast behavioral red build passed and its CPU run failed the three
4096 budget assertions with actual zero. Each preceding 2048 assertion passed.
After the minimum extension, all three budgets passed; unsupported 3072 remains
declined. The fixed async2 bills are:

| Helper | 2048 bytes | 4096 bytes |
| --- | ---: | ---: |
| A6 four expanded banks | 536,870,912 | 536,870,912 |
| MLA permutation copies | 805,306,368 | 1,610,612,736 |
| KDA cluster outputs | 2,621,440 | 5,242,880 |

The exclusive green fixture run passed all four tests. Original L0 A6 q/o banks
passed independent unpack/code-scale-bias BF16 coefficient checks and all
50,331,648 output values against two tuned 2048 calls. Original retained FA/GA/beta
banks passed 1,310,720 cluster outputs. Explicitly synthetic stored A6 MLA banks
passed 201,326,592 query/value outputs. Complete original L0 KDA passed all
16,777,216 BF16 output values, 73,728 BF16 convolution-tail values and
1,048,576 FP32 state words versus two sequential 2048 calls.

Rows were frozen synthetic 2048 BF16 activations (seed 46201) repeated once, with
nonzero history seeds 44/45. The test used original retained banks rather than a
model load. MLA bank scope is synthetic; it does not claim an original MLA weight
fixture or whole-attention proof. No tolerance or precision restoration was used.

Peak active memory was 7,948,043,324 bytes including original fixture banks and
all resident proof inputs/graphs/references. It is not a candidate net-overhead
measurement. Maximum fan RPM was confirmed after ten idle seconds; foreground
QoS and a per-job GPU lock were used, then the job exited 0, fans returned to auto
and the lock was released. No model, recapture or timed variant ran.

Artifact key `glm53-trunk4096-20261004` retains red/green sources, exact commands,
flags, source/binary hashes, failures, results and telemetry. All eight frozen
dependency hashes remained identical after compilation/execution, including the
default-off tree-core seam/helper; that recheck occurred after execution rather
than before launch. Green binary SHA256 is `433c693ce8396944a9e8bdfd094f28b1123c9426482562ae08cc9ff4cc232175`.
