# Bounded assistant commit context

`SUSHI_GLM_DFLASH_COMMIT_WINDOW=1` is an experimental opt-in requiring
`SUSHI_GLM_DFLASH_BLOCK_TAIL=1`. Eligibility is five sliding layers, window 2048,
block 8, and initialized dense BF16 KV without rings. Other contexts retain the
existing commit path.

Only the cloned next context is cropped, keeping the last 2047 valid rows before
accepted captures are appended. Its local cache step/offsets shrink and
`base_pos` advances by the dropped count, preserving absolute length and RoPE
positions. The source context remains untouched until evaluation succeeds and
the existing atomic publication replaces it. Feature encoding, full eight-row assistant
block/convolutions, accepted target state, and all target arithmetic are unchanged.

Ordinary KV allocation rounds the cropped commit to 2560 rows: 50 MiB for five A6
K/V pairs, independently of prompt length. Subsequent block-tail attention sees
the same 2055-row input as before cropping; this step adds no rounding change.

On M5 Max with MLX v0.32.3 (2026-10-03), a fixed 32K actual A6 assistant/head
component compared three appended captures. All 20,992,000 retained/new BF16 KV
values matched; absolute positions and the full source KV SHA256 were unchanged.
Subsequent full eight-row hidden/readout, lattice scores/candidates, and N2 trees matched
exactly. Eight focused tests passed. Foreground QoS, maximum fans, and an exclusive
GPU lock measured median clone+crop+feature projection+append+eval time 4.273 ms for
full context versus 0.729 ms for the window, four alternating pairs after two
warmups. This is a component result; normal HTTP throughput and complete target
state qualification remain required before default enablement.
