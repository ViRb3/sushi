# Current wide-prefill attribution before another kernel

Recommend one fixed actual-L20 attribution job, not a new runtime kernel.
Current source and loser evidence do not support another consequential exact
expert implementation yet. The measurement decides whether to invest in exact
surrounding Hadamard/preparation/finalization work or leave that work alone and
retain the current GEMMs. Runtime remains `4fcb541e`, closure `bd627696`.
No prototype, build, GPU job or runtime edit accompanied this report.

## Router finding: discard the cast/rounding idea

The actual stored L20 router weight is BF16 `[288,4096]`.
`glm5_forward.routeReference` casts BF16 X to FP32; `mlx::core::matmul` promotes
both operands and output to FP32, hence also casts the stored BF16 weight.
Pinned MLX `env::enable_tf32()` defaults to 1. Accepted recorded launchers have
no TF32-off setting. On NAX-capable hardware, M2048/N288/K4096 does not satisfy
NAX split-K (`4096 < 3*2048`) and selects regular NAX, BM64/BN128/BK512,
WM2/WN4 and swizzle 0 on the current `d` device branch. This is default-source
dispatch evidence, not a captured router pipeline trace.

BF16 operands therefore do not fix missing NAX. Ordinary BF16 matmul also stores
BF16 logits before sigmoid; casting them to FP32 afterward cannot restore the
old FP32 output boundary. A special FP32-output kernel would add copied native
code/qualification for a small scope. The old forced T2048 router subtotal is
23.183 of 1788.124 ms (1.2965%), including sigmoid, correction, partition, gather
and normalization. Even zero router cost has that limited diagnostic ceiling.
Do not implement the proposed tiny cast or rounding variant or disable TF32
globally. Original stored small tensors remain untouched.

## Main-work uncertainty and exclusions

The same old forced profile assigns 810.624 ms to complete routed FFNs (45.3%).
The actual L20 current transposed eight-route component recently measured
17.351583 ms in its paired control. That includes sort/metadata, paired input
preparation, three original NAX GEMMs, middle Hadamard/SwiGLU and weighted
finish. Neither record splits the current wide-prefill phase. The recent resource
replay measures T3 decode chains; its overlapping command-buffer intervals and
unsupported dispatch/traffic counters do not answer this question.

WIN64, alternate groups, shared decode, prefetch, word sharing and register-heavy
fusion already have negative evidence. Fixed route6 removed 20.23% component
latency but failed both declared quality inputs; another route count is excluded.
BF16 retention and cold absorbed MLA also failed their gates. L20 gate/up
input-scale tensors are not aliases: only 5582 of 1179648 F16 coefficients match
bitwise, so sharing their prepared plane is not a source-supported exact shortcut.
No new shader is justified by logical reads, tile counts or presumed occupancy.

## One bounded experiment and decision

Reuse the existing actual T2048 L20 X/IDs/scores and original full E288 banks,
with top-eight, MCG/W12, WIN32 and accepted physical grid transposition. Explicitly
initialize MCG/W12 before every reference projection; the prior wrong-reference
attempt is not evidence for this pack. No target load or new capture is needed.

Compare the unmodified end-settled complete chain with an otherwise identical
private chain settled at five fixed endpoints: (1) sort/metadata/paired prepare,
(2) gate plus up, (3) middle, (4) down, (5) finish. Both construct fresh graphs;
include final frees. Hold the same fixture and materialized stage references
in both protocols before timing. First prove every F16/BF16 stage and final
8,388,608 output values. Use three warmups and eleven alternating complete-chain
pairs, recording group clocks, construction/cleanup, complete clocks and peak.
Inserted waits are diagnostic perturbations, not a scheduling candidate.
Report their complete-chain overhead; phase clocks must not be silently summed
into normal HTTP or model latency shares.

The concrete decision is whether the non-GEMM portion is large enough to warrant
an exact consumer fusion next. Require a conservative surrounding-work opportunity
of at least 5% of the original complete L20 clock after accounting for observed
inserted-wait overhead; otherwise stop preparation/metadata/finalizer optimization
work for the next wave. If the group clocks are dominated by perturbation, report
unknown rather than force a share. A large gate/up/down group rules out another
small pointwise or router tweak as a route to 1500. A clearly substantial non-GEMM
group identifies one next implementation target; it does not itself approve a
kernel, claim removable traffic or establish a model gain.

All surrounding traffic is currently necessary: paired prepared inputs 256 MiB,
gate/up 64 MiB each, middle 64 MiB, down 128 MiB and output 16 MiB. Stage-settled ownership
and equally held proof references can retain more memory than the normal graph;
record that peak explicitly. Use existing actual full-bank fixture only, below
a small single-layer job bound, and keep every runtime reserve. No persistent
weight cache, installed-library replacement, counter framework or profiler sweep.

Worker owns only a private fixed L20 probe/result; root may expose one read-only
current transposed projection API alias and owns resource grants. No production
hook is needed for attribution. An eventual exact runtime candidate still needs
its own complete-chain pair and matched-model logits/state/continuation gate.
Any later rounding change must first pass the frozen two nonrepeated 16K code/prose
screen, all 24 late samples plus 192 baseline-forced predictions per input:
mean KL≤.01, max KL≤.15, top1≥95%, mean NLL increase≤.02 and new NF=0. No threshold
or corpus adjustment follows failure. Full HTTP qualification remains root-owned;
1500 prefill/60 decode are unmet.

Evidence: [current component scope](glm5-next-wave-performance-plan.md),
[wide routed chain and quality rejection](glm5-prefill-route6-component.md),
[decode resource limitations](glm5-routed-resource-result.md) and
[expert loser history](engine-exl3-experts.md).
