# Four-row KDA qualification for optimized N3

Component qualified, model policy acceptance pending. This extends the optional A6 QKV
coefficient hoist from three to three-or-four rows and the optional retained
first-path FP32 KDA endpoint from at most three to at most four rows.
The output projection remains outside the hoist guard. The original recurrence,
parent traversal, arithmetic, reduction and BF16 stores remain unchanged.

The first ancestry endpoint follows the existing first-child rule: row3 for
`[-1,0,1,2]`, row3 for `[-1,0,0,1]`, row1 for `[-1,0,0,0]` and row2 for
`[-1,0,1,1]`. A hit aliases one retained `[1,64,128,128]` FP32 state
(4194304 bytes); a different accepted endpoint uses the original replay.
Convolution tails still use the accepted ancestry. No extra endpoint planes or
weight bank are introduced. Existing request state reservation remains valid.

`glm5_dflash_t4_kda_probe.zig` uses the original layer0 stored A6 Q/K/V/output
banks and original small BF16/FP32 tensors, with fixed synthetic BF16 input and
nonzero BF16 convolution/FP32 recurrent state. It compares each hoisted T4 QKV
output to native one-row projections and the complete KDA layer outputs and all
accepted replay endpoints to independent serial ancestor forwards. Four T4
tree shapes and shortened T1/T2/T3, including the current N2 fork, are covered.
T1/T2/T5 and the output projection must continue to decline the hoist.

The focused performance decision is one inclusive T4 layer plus chain-endpoint
commit comparison, with resident original tensors, three warmups and eleven
alternating pairs. Each sample averages four fresh layer/tape/commit/evaluation
graphs and their frees. This is component evidence only; the coordinated N3
model policy must pass serial state parity and actual throughput acceptance
before implementation commit or push.

## Focused component result

At `88f19e22` plus the three guard changes, the original guard first failed
with `MissingT4Hoist`; the full T4 layer also observed zero instead of three
hoist calls. After extending eligibility, all three focused tests passed.
All 98304 T4 QKV BF16 outputs matched native row projections. The T3 check
covered another 73728 values. Across the four T4 and four shortened/N2 trees,
all 102400 layer-output BF16 values, 1843200 replay convolution-tail BF16
values and 26214400 replay FP32 state values matched independent serial
ancestor forwards. Each retained endpoint matched and reused its allocation;
every other endpoint followed exact replay. Engagement was three hoist calls
at T3/T4 and zero at T1/T2. The original output bank and T5 still declined.

| Complete T4 layer and chain commit | Median |
|---|---:|
| Original affine rows and replay | 0.732531 ms |
| QKV hoist and retained first-path endpoint | 0.681875 ms |

The candidate was 6.92% lower by arm medians, 6.54% lower by the median paired
ratio, and won all eleven pairs. Both arms used the same 23 original layer0
tensors (111903232 bytes) and synthetic activations. Timing includes all
per-call frees; it is not an isolated dispatch or full-model decode claim.
There is still only one 4 MiB retained state per layer. N3 policy cost,
acceptance and total throughput remain for the coordinated model gate.

Measurement key: `glm53-t4-kda-20261003`, 2026-10-03. ReleaseFast, pinned MLX,
foreground `taskpolicy -a`, exclusive per-job GPU locks, confirmed max fans
5347/5786 RPM at 47.92°C and ten seconds quiet idle preceded the green run.
Both locks were released before editing/building and after the GPU job; fans
returned automatic. Private artifacts hold tensor hashes, source/binary stamps,
red/green logs, complete paired arrays and thermal telemetry.


## Final bundle disposition

The [matched N2/N3 model decision](glm5-optimized-n3-result.md) passed exact
192-token and valid-state checks, but gained only0.56% amid1.46% control drift.
The bundle was rejected; this component's source/probe was archived and removed,
and accepted runtime behavior restored. No standalone runtime change was landed.
