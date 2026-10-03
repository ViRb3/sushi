# One-tree verifier raw-QKV attribution

Perfect held-output reuse removed **6.591750 ms** from this complete verifier
case: medians 48.868083→42.276333 ms, 13.4889% lower, with 11/11 favorable pairs.
This is accepted diagnostic evidence for a scoped removable-work ceiling.
It is not a production cache, kernel win, isolated QKV duration, bandwidth/
GPU-family share or HTTP throughput result. A later implementation remains a
separate research and validation decision.

One target/A6 load used the unchanged frozen ordinary 8756-ID prompt under
current `4fcb541e` full2/N2/children4, HC/native B1/B3/packed32 and async4.
The first actual proposal was frozen once: tokens `[73022,49235,198]`,
parents `[-1,0,1]`, target decisions `[49235,198,322]`. Normal prepareCommit
accepted all 3 inputs to offset 8759. No node marker, second tree/prefix, shape
sweep or extra projection was used.

Capture outside clocks retained 34 raw BF16 `[1,3,24576]` outputs, exactly
5013504 bytes. Observe controls constructed original Q/K/V+concat; reuse
supplied those held outputs. Both modes retained the same holder, original
prefix/assistant context and exact committed reference. Observer scope restored
before unchanged prepareCommit, so replay never entered it. Full 3-row logits
were borrowed from the existing head calculation and naturally settled by
original argmax/final tape evaluation, with no additional head work. Input
references were unnecessary: the normal settled tape's conv_input suffix
provided raw bit proof outside clocks. No weights or prefix caches were dumped.

Held raw 2506752 BF16 values, all 464640 three-row logit values, three decisions
and every valid committed MLA/KDA/convolution state matched exactly. Normal
capacity tails were excluded. One untimed observe/reuse proof preceded three
warmups per arm and 11 alternating fresh complete pairs. Timers included
request clone, verify, normal prepareCommit, original endpoint/tape settlement
and all output/pass/tape/branch/request frees; CPU bit oracles were excluded.

| Pair | Observe ms | Reuse ms | Paired reduction |
| --- | ---: | ---: | ---: |
| 1 | 48.991417 | 42.475834 | 13.2994% |
| 2 | 48.984125 | 42.341166 | 13.5615% |
| 3 | 48.944209 | 42.474042 | 13.2195% |
| 4 | 48.900792 | 42.346791 | 13.4026% |
| 5 | 48.861584 | 42.200333 | 13.6329% |
| 6 | 48.852667 | 42.276333 | 13.4616% |
| 7 | 48.829334 | 42.301000 | 13.3697% |
| 8 | 48.887250 | 42.240792 | 13.5955% |
| 9 | 48.840208 | 42.233667 | 13.5268% |
| 10 | 48.855583 | 42.213833 | 13.5947% |
| 11 | 48.868083 | 42.239125 | 13.5650% |

Median paired reduction was 13.5268%. Median clone/verify/commit/cleanup clocks
were 0.005208/48.043292/0.739125/0.072750 ms for observe and
0.005667/41.435917/0.742125/0.073166 for reuse. Phase medians need not sum to
the complete median. Cold capture plus normal verification/commit took
839.203375 ms; raw-file storage took 0.573750 ms. Neither is included in warmed
pair medians, and capture perturbation is not a production preparation cost.

## Adequacy and memory

All 22 timed passes completed 34-layer coverage and head observation, with
0/34 raw reuse in observe/reuse respectively. Counters stayed identical for
non-QKV work: HC 90; routed/grouped FFN 42; MLA query/value 11; native B3=11,
B1=0; cached-leaf hits 34/misses 0; async layer groups 11/final sync 1.
A6 raw-projection hoist dispatches deliberately fell 102→0. Its coefficient/
command effects belong to this complete counterfactual delta; they are not
an isolated dot-product time. Convolution/prework, recurrence, GA/GB/post/out,
HC/FFN/MLA/head and normal commit/replay remained in the evaluated path.

Both modes started at 95,291,582,200 active bytes. Maximum measured peak was
95,559,747,592. Memory and wired limits remained 115,448,725,504 bytes. Full
additional bill was 11,943,927,208: original one-assistant 11,383,406,592;
observer raw/reference/current-logit plus metadata 6,872,488; reference allowance
536,870,912; input/CPU allowance 16,777,216. Actual retained committed-request
storage was 255,050,240, within its allowance. All original reserves remained,
without credits for resident holders, references or sequential work. Load was
bounded before materialization and each complete replay checked active headroom.

The raw-only file was 5016371 bytes and the complete result JSON 22326 bytes,
well under the fixed 8MiB capture/metadata bound. Head aliases and the committed
state were RAM-only. Their logical costs are billed separately from the dump.
This experiment assumes exact outputs for one immutable epoch/tree/prefix;
real requests cannot generally reuse them. Queue/cache/overlap changes are part
of the observed whole operation. Do not add this ceiling to other class deltas,
assign it to individual GPU intervals or extrapolate to 32K/the 60 goal.

## Provenance and cleanup

Artifact `glm53-qkv-attribution-20261004` retains every pair/phase/counter,
exact tree/corpus identity, raw BF16 file, proof, source/weight-metadata/flag
fingerprints and telemetry. The private binary SHA256 is
`95a98d6b09f48c6badc1776e4769af405fc02b3b1db02eb7b71ac62a13c39f48`;
raw file SHA256 is
`62239afc0dcb7ad5064547bc90d199c2d8bc4088eeacefac7268316ac47ab6cf`.
A private loop-syntax compile error was preserved, then repaired without changing
workload/math. ReleaseFast build and dry-skip passed; the actual job exited 0,
all 7 tests passed. Foreground QoS, explicit accepted libraries, maximum fans/idle
and a per-job lock were used. Process ended, lock released and fans auto; binary
hash stayed fixed. Root hash-archived all 8 compiled sources and restored both
production diagnostic seams. The two owned evaluator/root files were verified,
archived and removed. Accepted runtime/CLI remain unchanged; no additional run
or runtime performance implementation follows this measurement alone.
