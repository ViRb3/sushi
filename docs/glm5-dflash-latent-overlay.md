# DFlash verification latent overlays

Trees with at most three nodes reuse the committed MLA latent prefix and read each branch's
ancestry from a short overlay. Verification no longer replaces the full latent buffer per branch.
The normal commit, IndexPool update arithmetic, KDA processing, selector policy, and cache
reservation are unchanged. Larger trees keep the existing full-cache path.

`MlaTape.appendIndex` gathers each ordered ancestor path, advances only its pooled/tail index
state, and returns the latent tail. `attendOverlay` maps absolute token IDs below the original
prefix length to the committed buffer and IDs above it to the ordered tail. Siblings at the same
logical position therefore use their own latent row. Capacity padding is never mistaken for a
committed prefix row. See [`src/glm5_dflash_model.zig`](../src/glm5_dflash_model.zig),
[`src/glm5_attention.zig`](../src/glm5_attention.zig), and
[`src/glm5_attention_overlay.zig`](../src/glm5_attention_overlay.zig).

The overlay shader keeps the existing eight splits, selected-token traversal, FP32 dot/online
softmax arithmetic, precise exponentials, and merge. Only the cache address changes. It supports
BF16 and FP32 inputs without re-encoding. Inputs have explicit contiguous stride checks and the
custom kernel does not request automatic full-prefix copies. Existing scratch admission remains
conservative and already covers the maximum 3072-byte BF16 tail per branch.

## Qualification

Focused tests compare raw output bits and complete pooled/tail state against ordinary full append
for BF16 and FP32, chain and sibling ancestry, all pooling residues, capacity growth, and the first
sparse-selection boundary. The committed source remains unchanged, and normal append produces the
same committed arrays. A nonzero tiny-model three-node test also matches target decisions, every
capture, and complete MLA/KDA committed state against independent serial ancestry for token
budgets 1,2,3. The focused caller artifact passed 10 tests with one optional diagnostic skipped.

One production-geometry component run used a 32768-token BF16 prefix, reserved latent capacity 33024,
64 heads, latent width 512, and three chain branches. All three ordinary forks had distinct replacement
buffers; all three overlays shared the prefix buffer. Outputs matched raw BF16 bits.

| Quantity | Full branch cache | Latent overlay |
|---|---:|---:|
| Latent bytes written per MLA-layer round | 101449728 | 6144 ancestry bytes |
| Three-branch layer median | 1.911 ms | 0.972 ms |
| Component speedup | | 1.97x |

The replacement-byte count follows the actual reserved shape and three observed replacement
buffers. A full eleven-MLA-layer round previously writes 1115947008 latent replacement bytes,
plus corresponding prefix reads. Normal accepted-path commit still performs its usual append.
The measured layer saving is 0.940 ms; extrapolating eleven layers suggests 10.3 ms per round,
but this is a component estimate rather than a full-model result.

Timing includes branch forks, existing IndexPool/tail updates, attention construction/evaluation,
and branch/Ops cleanup; the common final output-vector release occurs after the clock. Inputs and
cache reservation are outside timing. Two warmup pairs precede six interleaved measured pairs with
alternating arm order. The run used source matching this commit, ReleaseFast, `taskpolicy -a`,
exclusive GPU lock `glm-latent-overlay-v61`, max fans, and ten seconds idle below 90 C on 2026-10-03.
Measurement key: `glm53-latent-overlay-20261003`; raw arrays, fan status, and binary/source stamps
are recorded in the private measurement ledger. No model was loaded for the component timing.
