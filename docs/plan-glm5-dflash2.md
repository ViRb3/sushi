# GLM-5.3-Flash DFlash2 integration plan

Status: source study and proposed implementation, 2026-10-02. Serial GLM correctness and tuning come
first. This document does not enable speculation, report a GLM DFlash benchmark, or establish that
the Qwen verification kernels are compatible with GLM. The target is the local Sushi 2.4bpw checkpoint;
the independently supplied GLM DFlash2 assistant is a draft model, not the target's MTP layer.

## Sources and evidence

The following revisions were inspected locally; no model was loaded and no GPU test was run for this
study. References below use repository-relative paths and symbols so they survive worktree cleanup.

| Repository and revision | Source | Evidence |
|---|---|---|
| mlx-serve `599dd05bc74fb47624b4b95dd126813547e5ccdf` | `src/dflash.zig`: `loadDflashQuant`, `forwardBlock`, `lattice`, `bestFirstTree`, `appendContext` | Assistant loading, block forward, candidate tree and committed context |
| Same | `src/generate.zig`: `nextDflash`, `specTreeFor`, `dflashTreeRound` | Selection of tree mode, target verification, acceptance and commit |
| Same | `src/transformer.zig`: `SpecTree`, `specTreeSupported`, `ssmCommitTreePath`; `src/gdn_decode.zig`: `recurTree`, `replay` | Target state semantics |
| Same | `src/model.zig`: `rowExactArch`, `rowExactDecode`; `src/row_attn.zig`: `sdpa`; `src/lane_qmm.zig` | Numeric conditions behind serial-equivalent rows |
| Sushi `33c85349b4bd0b1e2414ff5d2b1a8edcbcf36225` | `src/glm5_forward.zig`: `Request`, `Model.forwardLast`; `src/glm5_model.zig`: `KdaLayer`; `src/glm5_attention.zig`: `State`, `attend` | Current GLM request ownership and missing speculative APIs |
| oMLX `6745c39cb66ba5ec130a761149fc1b8b832fcb07` | `omlx/patches/dflash_glm5.py`: `Glm5NextTargetOps`, `_contract_mhc_hidden`, `_rollback_glm_recurrent`; `tests/test_dflash_glm5.py` | Existing GLM chain adapter, capture contract and rollback tests; tree explicitly refused |
| TensorFold `bb4b4a35863af562fc4ccb2586300d8f94b5d6de` | `src/tensorfold/families/glm5_next/cuda/dflash2.py`: module contract, `Drafter` | Independent GLM capture and assistant convention; CUDA code is not a Metal performance result |

The inspected GLM assistant's `config.json` and safetensors header establish the following storage
contract. These are header facts, not a successful native bind:

- Five 4096-wide Qwen3-style draft layers, 32 query heads, 8 KV heads, head width 128, MLP width 12288.
- Block size 8, mask ID 154856, vocabulary 154880, target layer count 45, taps `[5,14,24,33,42]`.
- All five draft attention layers use sliding window 2048; block attention is noncausal. Draft RoPE
  theta is 10000. The draft's RoPE is independent of the target's NoPE MLA.
- Dynamic convolution kernel size 2, group size 16; selector rank 256, top-k 16.
- `fc.weight` is `[4096,20480]`; `hidden_norm.weight` is `[4096]`. Both selector codebooks are
  `[154880,256]`. The checkpoint has 81 BF16 tensors with 2,342,160,896 payload bytes (about 2.18 GiB).
  There is no embedding table or language head; both come from the target.

## What mlx-serve actually does

### Draft construction

`isDflashConfigJson` recognizes the block/mask/tap contract, including nested `dflash_config`, rather
than trusting `model_type`. This matters because this assistant declares `qwen3` but cannot run as a
standalone Qwen model. `loadDflashQuant` accepts the z-lab `fc`/`hidden_norm` spelling and unsuffixed
selector codebooks. `bind` checks hidden width, mask range, tap range, codebook coverage and target
capture capability. GLM should additionally check exact vocabulary/tokenizer compatibility and the
declared target layer count.

For each committed target token, `appendContext` concatenates the selected layer outputs in config
order, applies `fc` and RMS normalization, then creates every draft layer's context K/V. Keys receive
the draft layer's per-head RMS normalization and RoPE at the token's absolute position. Context append
must be contiguous: `first_pos == ctx.absLen()`.

At pending position P, `forwardBlock` consumes target embeddings of `[pending, mask, ..., mask]`.
Its two-tap grouped dynamic convolutions wrap both attention and MLP. The two output-side taps use
coefficients projected from the same normalized sublayer input as the input-side convolution. Draft
attention sees committed context plus the whole draft block. Sliding masking uses absolute distances
less than 2048, not a causal triangle inside the block. Temporary block K/V is removed before return;
only subsequent verified captures become persistent assistant context.

The target head projects the normalized draft outputs; row zero is dropped for DFlash2. The remaining
seven rows propose positions P+1 through P+7. `lattice` keeps the top 16 candidates per position and
computes unary logits plus predecessor/successor codebook edge scores conditioned on the draft hidden
state. This differs from taking seven independent argmax tokens.

`bestFirstTree` prioritizes cumulative sibling-normalized log probabilities, expanding up to four
children per node. Its current fitted defaults are temperature softening `tau=1.5`, edge weight 0.6
and, for keyed sampling, noise weight 0.7. These are Qwen/TensorFold policy choices to measure again
for GLM. Nodes are renumbered depth-first so the most likely accepted path often needs no row move.
On NAX, `dflashTreeRound` permits 15 candidate nodes plus the pending root in a 16-row forward; the
seven draft positions limit depth, not total node count. Non-NAX uses at most one node per draft
position under the current policy. A separately gated context-copy path can supply a full chain without
an assistant forward when at least 24 prior tokens support the continuation.

### Target tree mask and positions

`specTreeFor` constructs parent indices, row depths, ancestor paths and the GDN convolution source
table. Root row zero contains the already selected pending token at P. A tree row r has logical token
position `P + depth[r]`; its physical row number is not its token position. Siblings share positions.

`row_attn.sdpa` reads all committed keys before P and only that row's ancestors, including itself,
after P. It maps each logical position through the ancestor table. Applying an ordinary causal mask
to a flattened tree is incorrect: that would let a node attend to earlier siblings. Qwen GDN similarly
uses parent ancestry for both recurrence and the convolution history.

### Verification and acceptance

`nextDflash` chooses tree mode only with a selector, no Markov head, a greedy or supported keyed
sampling request, and `specTreeSupported()`. The latter depends on complete row-exact coverage.
`rowExactArch` currently covers non-MoE Qwen3.5-family targets without Hadamard transforms and
Nemotron-H; GLM is not in this gate. The Qwen27B result is not evidence that GLM EXL3 qualifies.

One target forward produces a next-token decision at every node. Starting at the root, the generator
follows a child only when its proposed token equals the parent's target decision. The last matched
row supplies the next pending token. For A accepted draft tokens, commit A+1 input rows: pending root
plus those A drafts. The next pending token has been selected but has not entered the target caches.
The token budget caps A before any cache commit.

Greedy mode uses target argmax. Keyed sampling uses the serial sampler's keys indexed by generated
position plus depth, so sibling rows at the same logical position share the serial-position noise.
This tree algorithm is not general stochastic rejection sampling over a flattened tree. Other sampled
requests remain on the chain path, whose selector proposal probabilities and residual distribution are
handled separately. Preserve that distinction when porting; do not attach a generic p/q test to trees.

### Commit and numeric prerequisites

`dflashTreeRound` compacts accepted KV rows, truncates rejected rows, restores the saved KV step
convention, and advances the model's absolute offset by A+1. `ssmCommitTreePath` replays retained GDN
prework from the round-input state, then reconstructs the final three convolution rows from the kept
path and pre-round window. Captured hidden rows are gathered along that same path and appended to
the assistant; rejected captures must never reach it.

The byte-equivalence claim depends on `lane_qmm`, fixed-order row attention and tree GDN preserving
the serial row's arithmetic. Multirow verification with ordinary GEMM can choose another reduction
order and change target decisions even with a correct mask. mlx-serve enables its row-exact path only
when the drafter is bound; its serial-with-drafter reference can therefore differ from an unbound stock
serial path. GLM measurements must distinguish those two baselines explicitly.

## Mapping to GLM

| Component | Reusable contract | Required GLM work |
|---|---|---|
| Assistant | Generic DFlash2 loader, convolution, lattice and selector | Bind through GLM embedding/head accessors; validate full header geometry and tokenizer |
| Captures | One `[1,T,4096]` tensor per requested layer output | Mean four mHC residual streams after the complete layer, before final norm |
| KDA | Parent-path recurrence and accepted-path replay | Per-key-channel decay and FP32 state; scalar-gate Qwen GDN replay is not interchangeable |
| MLA | Committed prefix plus ancestor visibility | Single 512-wide latent cache, NoPE, attention scale 1/16; no separate K/V copy |
| IndexPool | Query-specific sparse selection | Branch-local four-token pools, partial tails, physical row mapping and accepted-path commit |
| mHC | Independent operations for each token row | Keep all four streams through the target; only contract them for assistant captures |
| EXL3 experts | Per-token router and expert execution | Verify row-exact behavior across serial/chain/tree widths and gather/scatter order |
| Request ownership | Transactional verify and explicit commit | Extend GLM `Request`/`LayerState`; generic KV/GDN helpers do not own these caches |

oMLX `_contract_mhc_hidden` and TensorFold's GLM DFlash2 contract agree on the capture: mean of the
four streams **after** layers 5, 14, 24, 33 and 42. oMLX stores layer k's output at hidden-state index
k+1. Sushi's proposed capture seam belongs immediately after the second `hcExpand` in
`Model.forwardLast`, before reuse/free of that layer's result. Match dtype and reduction semantics
against a fixture; do not use the attention input, the branch collapse, or final-normalized hidden.
Source agreement is not yet a native capture-parity result.

The current GLM `Request` owns an offset, failure state and one `LayerState` per layer. Attention
`State` owns `latent`, `pooled`, `tail_keys`, `tail_gates` and `processed`. KDA owns FP32 recurrent
state and a three-row convolution history. These are the entire rollback contract, not just latent
cache length. A failed lazy evaluation currently poisons the request until reset; speculation must
preserve that fail-closed behavior on any incomplete transaction.

**IndexPool is the main extra tree problem.** Existing `State.append` assumes a single contiguous
token sequence. Appending tree rows into it would pool siblings together every four physical rows.
A node must instead complete pools using its ancestors and the committed prefix's zero-to-three-row
tail. Pool positional bias follows logical position modulo four. Existing committed complete pools
are shared, but new completed pools and tails are branch-local. Sparse selection compares shared
pools plus that node's branch pools, expands chosen pools to the corresponding physical latent rows,
and adds only the incomplete ancestor tail. After acceptance, rebuild/publish pools and tail for the
kept path; truncating latent storage alone cannot undo a rejected pool.

For GLM KDA, replay must preserve vector decay and FP32 state across every accepted token. One
state across all 34 KDA layers is already about 136 MiB before convolution history. Retaining a full
state for every node of a 16-row tree would add roughly 2.1 GiB for recurrent states alone. Prefer
bounded prework plus traversal/replay when parity is established; bill every snapshot and scratch
allocation before raising admission limits.

## Implementation order after serial tuning

1. **Capture and assistant parity.** Add optional requested-layer captures and GLM embedding/head
   interfaces without changing serial execution when unused. Port or reuse the generic assistant
   module with attribution in `NOTICE`. Verify stored BF16 first against a fixed reference block,
   including encoder, convolution boundaries, logits, candidates and selector edges. Stream captured
   prefill chunks into assistant K/V instead of retaining five full-context hidden tensors. Quantized
   assistant loading is a later measured option; do not silently inherit mlx-serve's NAX Q4 default.
2. **Exact chain rollback.** Implement an explicit begin/verify/commit transaction for a linear block
   first. Save input KDA state/conv history and enough layer prework to replay only the retained prefix.
   Snapshot IndexPool counts and its raw remainder; retain the block's raw index keys/gates so accepted
   pools can be reconstructed. Share immutable prefixes rather than copying a whole context. oMLX's
   `_rollback_glm_recurrent` and composite-pool undo are a useful correctness reference, not a proven
   fast implementation. Start with greedy blocks of 2–8 and keep unsupported serving features gated.
3. **Row-exact verification.** Measure every relevant affine projection, both MLA matrix orientations,
   EXL3 projection/gather/scatter, routing, mHC and KDA at rows 1–16. Use the same approved numeric
   path for serial and verification. If a faster path changes serial logits, pass Sushi's target-quality
   gate separately before using it as the speculation reference. Reject unsupported quant geometry.
4. **Tree state and kernels.** Add a validated parent/depth/path descriptor to the GLM forward. Add
   KDA ancestor-aware prework, vector-gated recurrence and commit replay. Add MLA ancestor addressing
   and IndexPool branch overlays as described above. Share pure per-row projection work, not mutable
   history. Keep a slow serial-per-branch oracle for tiny tests. Publish all kept caches, offset and
   captured assistant rows together; never expose a half-committed request.
5. **Generator and serving.** Wire pending-token ownership, EOS/stop/budget handling, sampling,
   metrics and a fallback to serial. Add explicit capability checks for tree, prefix-cache restore,
   KV modes and batching; a diagnostic forward does not imply scheduler support. Prefix restore must
   restore matching assistant context or rebuild the needed captures before speculation resumes.
6. **Tune policy.** Benchmark chain widths before 8/16-row trees, then vary node budget and branching
   independently of depth. Measure assistant precision, head precision, capture overhead, pipeline
   cadence and optional context-copy policy. Keep a measured break-even fallback; Qwen's block/node
   defaults and performance numbers do not transfer to a routed GLM model.

## Required tests and measurement

Tests must establish behavior, not source-text presence:

- Loader rejects wrong vocabulary, tap order/range, malformed selector/conv geometry and missing
  tensors. Assistant cannot be discovered as a standalone target. Dense and supported stored-affine
  arms match their own reference; no verification projection uses a draft-only head.
- Captures match oMLX's stream mean after the intended layers, including irregular prefill chunks.
  Draft absolute positions, RoPE and sliding masks match at the 2048 boundary. Block forward leaves
  committed assistant context unchanged until accepted captures are appended.
- Chain commit covers every acceptance count, including zero drafts, all drafts, token-budget cuts,
  and EOS/stop within the kept output. Compare next logits **and all cache state** to fresh serial
  processing of exactly the accepted prefix, followed by several more tokens.
- Tree fixtures use forks with repeated token IDs, siblings at the same depth and accepted paths
  that require physical row compaction. Each row equals independent serial processing of its own
  ancestors. Permuting siblings changes no logical-path result.
- Test every prefix-length residue modulo four, pool completion/rejection, capacity growth and the
  first selective position 2051. Include 4K/16K/64K states, negative index weights, tied scores and
  selected-pool order. Rejected descendants must not affect later pools, tails or recurrent state.
- Fault injection during snapshot, verify, lazy evaluation and commit must either restore a complete
  known state or mark the request unusable until reset. Cancellation and fallback free transient arrays.
- Test greedy serial/spec output equality on fixed prompts; separately test seeded keyed equivalence
  if implemented. For non-keyed stochastic chain sampling, test acceptance/residual distributions,
  not identical random bytes. Do not advertise sampled trees before their exactness contract passes.

Freeze the tuned serial baseline before speculation work. Record target and assistant hashes, binary
revision, ReleaseFast build, context, cache dtype, prompt IDs, output length, sampling, GPU lock, QoS
and thermal protocol from [process-measurement.md](process-measurement.md). Profile with barriers only
in diagnostic runs; report production end-to-end throughput without those barriers. Use warmed
fixed-context forward ladders for rows 1,2,4,8,16 and separate cold/warm prefill measurements.

Record draft forward, draft head, tree construction, target verify, state commit/replay and assistant
append time; accepted drafts per round; total committed tokens per round; rejected rows; copy-round
share; and peak/active memory. Useful throughput is committed output tokens divided by total elapsed
time, not verified rows per second. A round committing A+1 tokens must cost less than A+1 comparable
serial forwards to win. Compare predictable copy tasks with novel code/prose and multiple prompts;
high acceptance alone does not establish a speedup. Report any exactness-induced serial overhead as
well as the incremental speculation gain.

The requested 1,000 tok/s prefill and 60 tok/s decode remain performance targets. DFlash can reduce
decode cost only after target verification and state management are efficient; it does not solve the
serial prefill target and adds capture/context-encoding work. No throughput forecast follows from
the Qwen27B results or the existence of this assistant checkpoint.
