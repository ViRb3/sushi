# Raw A6 four-product component: rejected

Complete original L0 KDA plus normal chain commit lost its fixed timing gate:
median **551.041 → 558.083 µs**, 1.2779% slower. Paired median was 0.1482%
slower, with only **5/11 wins**. No quality screen, actual-model arm, numerical
variant or performance repeat follows. The [round plan](glm5-affine6-four-product-round-plan.md)
and [raw-class attribution](glm5-qkv-attribution-result.md) remain separate
from this losing implementation.

The candidate decoded four exact A6 integers per three-byte pack and used eight
products per eight inputs instead of twelve masked terms. It preserved stored
tensors, bias quartet boundaries, K256/lane/update/reduction/BF16 store order
and FP32 persistent state. M1 retained the existing fused three-bank dispatch;
M2/M3 retained three projections plus original concat. Per-row/output safety
flags used already-loaded operands; unsafe provisional results were overwritten
by a fresh entire original masked dot before reduction/store. Guard and fallback
costs were included. This is a different FP32 grouping, not general old-target
bit parity or precision restoration.

## Proof and fixed scope

ReleaseFast green exited 0, all four tests passed. On the original 23 stored L0
tensors, explicitly synthetic BF16 seed 5304 activations and frozen nonzero
BF16 convolution/FP32 recurrent history:

- Candidate M1/M2/M3 raw equality covered 147,456 BF16 values, including fused
  M1 bank selection. Unsupported M4/F32 input and output-bank geometry declined.
- Chain and fork complete outputs, every replay convolution/FP32 state and
  retained leaf matched candidate serial ancestry exactly. Leaf hit/miss proof
  passed; no state tolerance was used.
- Old-hoist raw drift was zero at all 73,728 BF16 values; old complete outputs
  matched at 12,288 values for each topology. Both sets had zero nonfinite values.
  These normal synthetic samples do not establish old-grouping equivalence.
- Scoped default-off policy and exact packed integer checks passed.

The independent numerical auditor signed off source semantics and conservative
range bounds. Its directed cancellation, mixed nonfinite and late unsafe-grid
GPU audit **did not run**, because the complete operation failed performance.
No directed special-value hardware proof or long-prefix quality claim is made.

Both arms held the same original fixture/input/state. Three warmups preceded
11 alternating fresh layer/tape/normal-commit graphs, one endpoint evaluation
and all frees. No per-child forced profile was active. Timing engagement was
42 old hoists versus 42 candidate projection calls, 140 unchanged retained
small-projection calls and 28 leaf hits with zero misses. Candidate peak growth
above resident fixture/input/state was **5,409,124 bytes**; unchanged retained
FP32 state was 4,194,304 bytes. This is a component peak, not a full-model bill.

| Pair | Current µs | Candidate µs | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 536.167 | 524.500 | +2.1760% |
| 2 | 576.542 | 573.292 | +0.5637% |
| 3 | 530.916 | 535.583 | -0.8790% |
| 4 | 562.916 | 563.750 | -0.1482% |
| 5 | 530.292 | 542.958 | -2.3885% |
| 6 | 534.375 | 591.541 | -10.6977% |
| 7 | 609.083 | 563.542 | +7.4770% |
| 8 | 551.041 | 545.375 | +1.0282% |
| 9 | 567.125 | 558.083 | +1.5944% |
| 10 | 536.208 | 537.167 | -0.1788% |
| 11 | 552.500 | 563.208 | -1.9381% |

## Provenance and closure

Accepted runtime was `4fcb541e`, closure `5929c9d6`, plan `3596e053`, with hashed
WIP source. Artifact `glm53-raw-a6-four-product-20261004` preserves every pair,
raw/state results, source snapshots, commands, binary identities and telemetry.
The genuine null-stub red failed `MissingRawFourProduct`. Two earlier packaging
compile failures were preserved: a Zig reserved local identifier, then an
optional error-union return requiring `return try`; neither changed arithmetic.

Final green build PID 21433 and proof PID 21571 ended exit 0. Accepted staged
libraries, foreground `taskpolicy -a`, one exclusive per-job lock, confirmed
maximum fans/idle and automatic fan cleanup were used. Lock is free; no job
remains. Helper, private component probe/root and both root seams were archived
against the compiled hashes. Root verified cleanup and restored ordinary runtime.

| Compiled artifact | SHA256 |
| --- | --- |
| Helper | `8aef2c4d9fa67c56437c9cf61d32a82848797cea114327065c4e004fabd853f6` |
| Component probe | `112d7fd9c84de5deff19f4bf6a5c6ed0e5ec5fe5fe869ef0eace63a1ad5d8cb3` |
| Component root | `440c4cac22ebdcc24c1f4a0568cb2ecea604469ed51f4c636edfeac9ae214498` |
| Private green binary | `89bdd29c93a280ca68e6102b1b372831ee170481ebb6a7726785e220f8f202e8` |

The unchanged qualified CLI SHA256 is
`102cb8d5efcf279295ea059201eb0a88c3e7d85321a68dced625c0b4c6858ae3`.
Installed MLX/MLXC/metallib hashes still match the pre-build manifest. There is
no runtime/default/library/admission-bill change from this rejected component.
No speed forecast, DRAM attribution or 2K–32K improvement follows the arithmetic
counts. Goals 1500 prefill and 60 speculative decode remain unmet.
