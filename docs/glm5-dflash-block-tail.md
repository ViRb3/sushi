# Read-only sliding assistant block KV

`SUSHI_GLM_DFLASH_BLOCK_TAIL=1` is an experimental opt-in. It applies only to
five-layer, eight-row assistants whose layers all use the 2048-token sliding
window, with initialized dense BF16 context of at least 2048 rows. Other
assistants use the existing append/truncate path.

The first block query is at the context's absolute end. A context key is visible
only when its distance from that query is less than 2048. The last 2047 context
rows therefore contain every key any of the eight block queries can see. The
candidate concatenates those rows with the eight temporary block K/V rows and
builds the original mask with its adjusted absolute base. The complete trained
block forward, dynamic convolutions, RoPE positions, A6 weights, and persistent
context remain unchanged.

The existing context clone shares buffers. Updating its spare rows through
MLX slice-update copies shared buffers, even though only eight rows are written.
For the A6 assistant's five layers, eight KV heads, head dimension 128, and BF16
K/V, a reserved 33024-row context writes 645 MiB of replacement buffers per draft.
The bounded block inputs total 40.14 MiB at 2055 rows. They are temporary graph
values; no persistent tail cache or new cache format is introduced. MLA cache
and FP32 KDA state are unaffected.

Removing fully masked keys preserves the trained visibility rule. SDPA kernel
selection and reduction geometry can change, so output rounding must be measured
and the full-model quality gate must pass before default enablement.

The fixed 32K component on M5 Max, MLX v0.32.3 (2026-10-03), passed exact
visible-mask equality, unchanged source handles/counters, first-two readout top1,
and N2 tree equality for children 1/2/4. It used the actual A6 assistant and target
head with fixed-seed BF16 KV/embeddings, without the target trunk. Hidden output
relative L2 differed by 3.308% (max absolute 0.390625); readout relative L2 differed
by 3.485%. This is not a bit-exact optimization. Under foreground QoS, maximum
fans, and an exclusive GPU lock, four alternating warmed pairs measured median
assistant-forward time 11.361 ms for full-context append versus 4.569 ms for bounded
block input (59.8% lower). These are component timings; full-model distribution,
acceptance, and throughput qualification remain required. The switch stays off.

## Full target-state gate

At source `f45103ef`, the actual 2.3bpw target and A6 assistant completed a
32768-token prefix and 64 generated tokens with the candidate enabled. The
2048-token code fixture was repeated 16 times. Every generated token and the
complete committed target state matched independent serial decoding exactly.
The block-tail path engaged 24 times; the exact affine hoist engaged 2448 times,
and normal prefill clustering 544 times with 85 MiB prepared banks. Decode peak
active memory was 97,690,811,914 bytes, below the 110 GiB budget.

The synthetic assistant hidden drift is therefore not a demonstrated target
correctness loss. Proposal acceptance and normal HTTP throughput remain distinct
performance checks. The native code-prompt gate measured 21.623 tok/s with 39
accepted drafts across 25 rounds; its prompt, committed-token denominator, and
held serial snapshot differ from llmprobe's predictable-context table. Do not
compare those rates directly. Measurement key: `glm53-block-tail-model-gate-20261003`.
ReleaseFast full suite and rebuilt CLI also passed. Default remains opt-in.
