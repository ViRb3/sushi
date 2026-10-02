# GLM internal runtime reuse audit

Read-only review, 2026-10-02, against Sushi revision `ddcf219f`. This compares the GLM diagnostic
path with the Qwen and MiMo implementations in the same revision. No kernels were changed or GPU
measurements run for this audit. The coordinator's current reference is approximately 748 prompt
tokens/s and 24.6 serial decode tokens/s on the 512/64 workload; the ranking below is an engineering
assessment, not measured speedup attribution.

The worktree has no CodeGraph index, so this review used its source directly. It did not inspect or
modify the primary checkout. Links name the relevant source files; symbol names below are the stable
lookup points at the stated revision.

## Already reused or completed

Do not schedule these again: copy-free three-bank affine8 QKV decode; paired EXL3 token preparation;
GPU expert-window metadata; immutable KDA convolution/decay preparation; compact owned convolution
tails; asynchronous decode evaluation; bounded two-layer prefill scheduling; opt-in dense MLA SDPA;
and precise FP32 HC mixing. The metadata work deliberately retained the old scatter/reduce order,
leaving the small opportunity in item 1 below.

## Prioritized candidates

Effort estimates assume one engineer with the existing fixtures: **small** is roughly a focused day,
**medium** several focused days, and **large** a separate kernel/validation project. They are not
completion promises. Start with exact-byte candidates before arithmetic-changing candidates.

| Priority | Borrowable implementation | Mechanism and applicability | Required adaptation and gate | Effort |
|---|---|---|---|---|
| 1 — prefill, low risk | `finishMimoSorted`, `MIMO_REDUCE_SOURCE`, `buildMimoWindowTable` in [EXL3 kernels](../src/exl3/expert_exl3_kernels.zig) | Finish directly from sorted down-projection output using inverse routing. GLM currently computes this inverse in the GPU metadata builder, frees it, then materializes `scatterSorted` before reducing. | Retain inverse metadata and use the sorted reducer only for the aligned GPU-table arm. Keep the stride fallback. Prove output bytes and routing-order reduction unchanged at 288 experts, all rates, all/skewed/sparse routes and padded windows. | Small |
| 2 — serial decode | `moeSwigluFusedWithShared`, `pairGemv`, `downGemvFusedMid`, `downGemvPreparedMid` in [EXL3 kernels](../src/exl3/expert_exl3_kernels.zig) | GLM still invokes two indexed GEMVs, a separate clamped middle transform, and a down GEMV. The normal EXL3 chain fuses gate/up work and can fold middle work into down projection. | Add a clamp specialization; do not call the unclamped chain. Preserve FP32 EXL3 middle arithmetic and its F16 store after the down-input Hadamard. Preserve split-K summation. Gate equal gate/up packing rates; retain mixed-rate fallback. `preparedMidOn` currently caps experts at 256, so GLM's 288 needs explicit admission and testing. | Medium |
| 3 — decode and prefill | `gdnNormGateFused` in [Transformer](../src/transformer.zig) | One head-sized kernel can replace GLM's cast, square/reduce, rsqrt, norm multiplication, sigmoid and output cast chain after recurrence. The existing launch geometry fits 128-wide heads. | Borrow the geometry, not the BF16 arithmetic. Existing GDN rounds normalized values and sigmoid to BF16 early. GLM keeps normalization, weight multiplication and sigmoid gating in FP32 and casts at the end. Add a separate GLM specialization and use the assembled KDA fixture. | Small–medium |
| 4 — serial decode | `gdn_decode.step`/`stepFold`, `gdnPreworkFused` in [GDN decode](../src/gdn_decode.zig) and [Transformer](../src/transformer.zig) | Reuse the per-head threadgroup, convolution-window update and register-resident recurrence design to avoid many tiny GLM prework launches and temporary arrays. Start with precomputed GLM gate-projection outputs; folding the projections themselves is a second step. | GLM needs vector decay, FP32 state, FP32 decay parameters, sum-based L2 normalization and sigmoid output gating. The existing scalar-gate BF16 implementation is not directly compatible. See the explicit contract below. | Large |
| 5 — decode and prefill | `hc_prefill.norm`/`Pending`, `hcReadPreparedWidth`, `HcPrepared.callback` in [HC prefill](../src/hc_prefill.zig) and [Transformer](../src/transformer.zig) | Borrow the deferred residual/next-normalization pattern and shape-specialized launch caching. GLM currently writes expanded four-stream residuals and rereads them for the next HC normalization and mix projection. | Qwen HC's learned down/up/inject formula and 4×2560 prepared geometry differ from GLM's 4×4096 Sinkhorn HC. Build a GLM-specific combined expansion/normalization path, preserving residual-contraction-before-branch-add rounding and BF16 HC matrix storage with FP32 math. | Medium–large |
| 6 — dense/shared FFN | `fusedSwiGLU`, `hc_prefill.sigmoidTable` in [Transformer](../src/transformer.zig) and [HC prefill](../src/hc_prefill.zig) | A clamped BF16 activation kernel can replace separate gate clamp, up clamp, sigmoid and products after the dense/shared projections. It can cover the three dense layers and each shared expert. | Add GLM clamps before activation. Preserve BF16 sigmoid rounding and multiplication boundaries, unlike the EXL3 FP32 middle contract. Existing exhaustive BF16 sigmoid/SwiGLU tests are useful. Gate/up projection fusion is a separate change with its own GEMV rounding test. | Small–medium |
| 7 — long context only | `qsaSelectTopBlocks`/`qsaSelectTopBlocksSplit` in [Transformer](../src/transformer.zig) | Replace generic argpartition over the GLM pooled-score sheet with a specialized GPU top-k selector, retaining bounded score chunks. The selector consumes FP32 scores and per-row causal bounds independently of Qwen's score calculation. | Adapt shapes to `[1,rows,pools]`, select at most 512 pools, preserve GLM pool completion/tail rules, and compare selected sets including ties and invalid candidates. Selection order affects reduction rounding; validate that separately. This does not help the 512-token benchmark. | Medium |

### 1. Sorted finish is the most contained next experiment

`MIMO_REDUCE_SOURCE` is derived from the same `REDUCE_SOURCE` used by GLM's current
`downFinishReduce`; it changes the input-row lookup to use inverse routing. There is already a
`finishMimoSorted` versus scatter-plus-finish test. Extend that evidence rather than assuming the
MiMo test admits GLM automatically.

At 512 rows, top-8 and hidden width 4096, the half-precision scattered intermediate is 32 MiB.
Removing it avoids one 32 MiB write and a corresponding read per routed layer, plus a dispatch.
The finish kernel must still iterate routing slots in the current order and use the original expert
IDs and scores. This is an estimate of avoided intermediate traffic, not a throughput prediction.

### 2. Fused EXL3 decode has two independent opportunities

First port paired gate/up multiplication plus the clamp-aware middle/down chain. Then consider
`downFinishReduceWithShared`: GLM currently adds the shared expert in a separate operation. The
existing shared-output reducer preserves the routed output rounding before the add. Verify that
same boundary against GLM before using it; do not fold the shared branch into the routed FP32 sum.

The public pack may use different rates for gate, up and down. Current K2.25/W12 has matching gate/up,
but a reusable GLM path must not silently assume that every future 2–4 bpw bank does. Exercise each
supported rate, mixed-rate fallback, clamps at and around ±10, cancellation, overflow-sensitive
scales, duplicate routes where admitted, and all 288 expert IDs.

## KDA: what can and cannot transfer

`gdn_decode.zig` last changed at `92fd9b71`; the reviewed `transformer.zig` snapshot includes the
shared recurrence introduced for GLM at `1fa1b16e`. GLM already uses the shared **vector-gate**
recurrence through `glm5_next.kda`; reusing that recurrence again is not a new optimization.

| Contract | Existing fused Qwen GDN | GLM KDA requirement |
|---|---|---|
| Gate input | Scalar per value head | Per-head, per-key-channel vector |
| Decay | Scalar softplus form, including BF16 rounding | `exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias)))` in FP32 |
| `dt_bias` | One value per head in the fused path | FP32 `[heads,128]` |
| State | BF16 input/output and verify captures | FP32 input/output; do not insert a BF16 round trip |
| Q/K normalization | RMS-style mean and BF16 scale boundaries | Sum-based L2 with epsilon, Q multiplied by `128^-0.5`, then activation-dtype cast |
| Output norm/gate | Early BF16 normalized value and sigmoid in current GDN kernel | FP32 normalization/weight/gate arithmetic, then BF16 output |

The fused-step structure is valuable because a GLM layer has 64 heads of width 128 and a four-tap
convolution, but toggling an eligibility guard is unsafe. Test cold and nonzero initial state,
serial/irregular chunks, multiple heads, nonuniform channel decays, convolution rounding, and state
reset. If verify/rollback is added later, every captured state must also remain FP32.

The blocked/pipelined GDN prefill routes are a lower-priority research port. `gdnPrefillRoute`
explicitly rejects `vector_gate` before choosing `gdnKernelBlocked` or `gdnKernelPipelined`.
Per-channel decay changes the transition algebra; removing that guard is not an optimization.
Develop a vector-decay derivation and reference proof first, then measure register pressure,
state traffic and precision. Do not assume the scalar kernel's numerical contract survives.

## HC, affine and indexer caveats

- Qwen's HC prefill code supplies useful reduction, pending-residual and sigmoid-table techniques,
  but its HC computation is not GLM Sinkhorn mixing. GLM must retain 24 mixing coefficients,
  FP32 scale/base, source BF16 mixing matrices and precise accumulation. The internal DeepSeek-v4
  `hcPreGpu`/`hcPostGpu` and Sinkhorn builders are an additional mathematical reference, not a proven
  fast drop-in replacement. The current GLM collapse already combines Sinkhorn and stream collapse.
- The installed three-bank A8g128 QKV kernel is complete. The next projection opportunities are
  smaller remaining parallel projections, not repeating QKV. Source BF16 `f_a/g_a/b` and `f_b/g_b`
  must remain BF16; do not quantize them or concatenate/expand resident weights just to satisfy an
  affine fast path. Measure whether dispatch savings justify a separate mixed-bank implementation.
- `verifyQmm` is primarily a 4-bit, short-multirow path with different reduction order. It is not a
  serial affine8g128 solution. Its tiling ideas become relevant only after an 8-bit specialization
  and explicit precision tests, or when verified speculative execution is separately in scope.
- Qwen `qsaPoolNormRopeFused` is not GLM pooling: it requires rotary channels and uses a different
  pooling/normalization sequence. GLM uses gate-softmax weighted four-token pools and NoPE.
  Likewise the Qwen fused score sheet has different head/rounding assumptions: GLM's weighted
  ReLU scores and BF16 rounding boundaries must be preserved. Borrow selection independently first.
- Qwen `gatherQsa256`/packed attention are not a direct GLM replacement: the stored object here is a
  512-wide compressed latent, while attention scale still comes from 256-wide queries. Keep the
  established short-context expanded-versus-absorbed oracle rather than claiming reassociation is
  bit-exact. Dense prefill SDPA is already implemented and excluded from new proposals.

## Scheduling, caches and acceptance gates

Async evaluation and two-layer prefill scheduling have already been implemented. The new work should
remove remaining synchronization/materialization inside that schedule, rather than add another
scheduler. Keep persistent state evaluation and release ownership explicit; Qwen's checkpoint,
reservation and donation machinery is a useful later integration reference, not permission to enable
GLM prefix restore or speculative rollback without independent state tests. GLM's capacity-backed
latent buffers also require valid-length slicing; that issue was already fixed for dense SDPA.

For a byte-preserving candidate, compare old/new operations in one process, exercise real widths and
288 experts, then measure the final model binary with identical IDs/settings. For a change to
reduction order or activation rounding, require source-layer error characterization and model quality
validation. No percentage speedup or attainment of 1000/60 follows from this audit. If the current
11.15 GB/token weight-traffic estimate is accurate, 60 serial tokens/s requires about 669 GB/s of
effective weight reads before other traffic; verify that model before assuming dispatch fusion alone
can meet the target.

Suggested sequence: sorted finish → clamped fused EXL3 decode → GLM norm/gate epilogue → profile again.
Promote a larger KDA/HC fusion only when the remaining measurements justify it. Keep long-context
index selection and speculative paths separate from the current serial 512/64 comparison.
