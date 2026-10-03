# GLM DFlash2 coarse draft readout

The Qwen optimization can migrate to GLM DFlash2 as a proposal-only readout experiment. It is a
3-bit copy of the target vocabulary head followed by exact re-scoring of 32 shortlisted rows. It
is not a separate autoregressive model, and it does not quantize the DFlash2 assistant to 3 bits.
The target's stored weights and verification readout stay unchanged.

## Actual implementation being reused

Qwen's native MTP head returns its mixer output to the readout. `buildRerankCoarse` makes a chunked
low-bit/gs64 copy of the target head; `rerankCoarseFlat` reads it at its stored bit width;
`draftTop32` selects 32 vocabulary IDs; `rerankShortlistRow` gathers the original packed rows and
re-scores them. `rerankSelect` chooses the highest exact score. A coarse shortlist miss can lose
acceptance, while target verification still determines the emitted token. See
[`src/mtp.zig`](../src/mtp.zig), especially `buildRerankCoarse`, `requantizeRows`,
`rerankShortlistRow`, `rerankShortlist`, `rerankSelect`, and `draftTop32`;
[`docs/engine-mtp.md`](engine-mtp.md#greedy-shortlist) describes its sampling and verification contract.

The inherited plain DFlash path already supports a draft-only low-bit head, but its `draftLogits`
uses that coarse head directly, without the Qwen top-32 exact re-score. Its default is off because
that lossy readout reduced acceptance enough to lose throughput at its measured block size. See
[`src/dflash.zig`](../src/dflash.zig), `DEFAULT_DRAFT_HEAD_BITS`, `buildDraftHeadBits`, and
`draftLogits`. Therefore enabling that existing direct coarse readout alone would not migrate the
Qwen trick.

## GLM compatibility and correctness

The inspected target and selected A6g128 assistant both have hidden size 4096 and vocabulary size
154880. The assistant has no embedding or vocabulary head tensors and no tokenizer files: it uses
the target's token IDs, raw embedding table, and head. Its selector codebooks each have shape
`[154880,256]`; its mask token is 154856 and its target taps are `[5,14,24,33,42]`. The five-layer
assistant's stored A6g128 projections and retained BF16 small tensors remain as supplied.
`loadAssistantStored` checks the assistant's declared vocabulary and target layer count, while
`validatePair` checks hidden width, selector shapes, token bounds, and layer taps. The tokenizer
semantics consequently come from the selected target; the assistant has no second tokenizer to
reconcile. See [`src/glm5_dflash.zig`](../src/glm5_dflash.zig), `loadAssistantStored`,
`validatePair`, and `proposeTreeWithChildren`.

The original GLM proposal reads the full target head for all eight assistant block rows, drops the
anchor, and selects top16 vocabulary candidates per remaining row. `lattice` combines these unary
scores with the predecessor/successor codebook edges. `bestFirstTree` applies log-softmax over
these **16 candidates for each parent**, then accumulates path scores. It does not use a full-vocab
probability normalizer. See [`src/glm5_dflash_tree.zig`](../src/glm5_dflash_tree.zig), `lattice` and
`bestFirstTree`.

The experiment should keep that selector and its top16 unchanged: obtain top32 through the coarse
head, re-score those rows with the original target storage, set all other vocabulary logits to
negative infinity, then call the original lattice builder. If the coarse shortlist retains the
original top16, their unary logits, edge scores, and conditional normalization are preserved;
tied scores may still expose ordering choices. If candidates are missed, the proposal changes and
acceptance may decline. Enlarging the selector from top16 to top32 would separately change edges,
normalizers, and tree ranking, so it is not part of this experiment.

Current GLM acceptance is greedy: `accept` follows only a child whose token equals its parent's
exact target argmax, retains the accepted ancestry, and returns the exact next target token as the
pending correction. It never consumes a draft probability or an acceptance ratio. Adding an
estimated full-vocab logZ from coarse logits would change proposal ranking without establishing
sampled correctness. A future sampled path would need the actual supported proposal distribution
and residual correction; Qwen's sampled shortlist normalizes over its exact shortlist support.
See [`src/glm5_dflash_tree.zig`](../src/glm5_dflash_tree.zig), `accept`, and
[`src/glm5_dflash_model.zig`](../src/glm5_dflash_model.zig), `prepareCommit`.

The target head is affine6/gs128: packed weights `[154880,768]` and BF16 scale/bias grids
`[154880,32]`. Its existing storage is 495616000 bytes. A 3-bit/gs64 copy adds 277544960 bytes
(264.69 MiB) resident. Re-scoring 32 source rows reads 102400 bytes per row of assistant output;
seven output positions read 716800 bytes. The theoretical head bandwidth saving is therefore about
217 MB per round, before selection/gather/dispatch costs, on a source head already stored at six
bits. At 546 GB/s that is only about 0.40 ms per round. This readout optimization alone cannot
explain the whole gap from 45.448 to 60 tok/s; actual round latency and acceptance determine whether
it is worth retaining.

## Experimental path and component qualification

`SUSHI_GLM_DFLASH_MINI_HEAD=1` opts into the readout at assistant load. Unset or `0` preserves the
existing full-head draft path. The selected A6g128 assistant projections and BF16 small tensors
are not re-encoded. The coarse head owns a separate runtime copy and uses the existing DFlash
head lifetime; the target head stays borrowed and unchanged. This narrow path supports blocks
up to eight rows, selector top-k at most 32, a positive finite output multiplier, and no positive
logit softcap. Unsupported opt-in geometry is refused explicitly. See
[`src/glm5_dflash_mini.zig`](../src/glm5_dflash_mini.zig), `build`, `project`, and `residentBytes`.

Re-scoring keeps the original assistant block width. MLX uses `affine_qmv_wide` for the original
eight-row readout; re-scoring one input row at a time would select `affine_qmv_fast`, whose reduction
can round differently. Each gathered 32-row head therefore reads all eight hidden rows, and only
its matching position survives. This preserves the original readout arithmetic while reading
very few original head rows.

At the implementation commit accompanying this document, the component tests check a synthetic A6g128 source with a ragged vocabulary and three draft rows,
source storage preservation, 32 finite re-scored logits per position, and raw BF16 equality with
its full source readout. An optional production-head test uses only the head shard and fixed-seed
random BF16 hidden rows, without loading the full target or assistant. It passed all 224 re-scored
logits against the original eight-row source readout, bit for bit. The coarse shortlist retained
all 7 original top-1 IDs and 101 of 112 top-16 IDs (90.2%) in that synthetic production-shaped block.
These are candidate coverage checks, not real-text acceptance measurements.

The same component run interleaved eight full/mini timing pairs after two warmup pairs:
full eight-row readout median 1.891 ms, mini seven-position readout median 1.845 ms. The 0.046 ms
difference is within noise and does not establish a decode speedup. It used ReleaseFast,
`taskpolicy -a`, exclusive GPU lock `glm-mini-production-head`, max fans, and ten seconds idle
with sensors below 90 C on 2026-10-03. The test binary was built at 08:25:16+07, and the run ended
at 08:25:52+07. Measurement key: `glm53-mini-head-component-20261003`.

## Full checkpoint qualification

One ReleaseFast binary built from `e4be4673` compared the selected A6g128 assistant
and 2.3bpw A6-trunk target with dense rows enabled, N2/children4, async4, group2 and
lane/down enabled. Both arms used prefix512/chunk128,64 committed inputs, one warmup,
BF16 compressed MLA cache and FP32 KDA state; profiling and route capture were off.
Each matched all64 output IDs and complete target state against its serial reference.
The mini head engaged24 times, used277,544,960 additional resident bytes and retained
the same24 rounds/40 accepted drafts/72 verification rows on this prompt.

| Readout | Decode tok/s | Draft ms | Verify ms | Replay ms | Commit ms | Decode peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| Original A6 | 45.8676 | 146.348 | 1,189.197 | 43.211 | 15.760 | 94,978,758,176 |
| 3-bit shortlist + A6 re-score | 46.0117 | 144.003 | 1,186.961 | 42.663 | 16.391 | 95,256,386,336 |

The0.31% overall difference is provisional and within run-to-run noise; this
establishes real-prompt acceptance and correctness, not a robust speed advantage.
The head remains default off. Broader prompts and contexts remain unqualified,
and the60 tok/s goal is unmet. Each arm acquired its own exclusive GPU lock,
used foreground `taskpolicy -a`, maximum fans and ten seconds idle below90°C,
with no overlapping compiler/GPU work. Private artifact `glm53-dense-mini-20261003`
contains the fixed binary hash, exact settings, tokens, state checks and telemetry.
