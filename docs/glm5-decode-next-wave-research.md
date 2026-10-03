# Next bounded decode/speculative wave

Research only at `e1597cc2`, 2026-10-03. Recommend completing one optimized
N3/children4 policy, alongside the independently proposed indexed-prefill
workstream. Native attention and A4 storage are supported opt-ins; neither is
an assumed default. No build, GPU job or runtime source change accompanied this
recommendation.

## Decision and measured limits

The warmed native/A6 predictable8K request measured46.57 decode tok/s, with
192 IDs in64 rounds; predictable16K measured44.20 with the same64 rounds.
Verify cost was55.795/58.648 ms per round respectively. These are the existing
`glm53-native-decode-llmprobe-20261003` server results, foreground/exclusive GPU,
profile off and the inherited accepted control. Ordinary16K improved partly
because rounds fell76→68 and KDA misses782→340; policy/acceptance matters.

Even perfect four-token acceptance with zero extra round cost would scale those
8K/16K rates only to62.09/58.93 tok/s. That is an ideal ceiling, not a forecast.
Historical N3/children4 cost65.736 versus53.988 ms verification (+21.76%) while
tokens/round improved only9.09%; it lost. That experiment lacked the current
three-row optimizations at T4. Qualifying those missing guards is a distinct
bounded policy experiment; merely setting nodes3 would repeat an unfair arm.

The larger remaining verifier phase considered here is KDA QKV. The recorded
six-round forced-wait profile (`c8b03ba2`) charged79.968 ms to204 KDA QKV calls,
about13.328 ms per round, versus145.49 ms routed FFN. The QKV profile predates
the accepted exact hoist and is not a removable latency budget. Current source
still constructs Q/K/V separately and concatenates `[1,3,24576]`.

One alternative is a same-body joined A6 QKV kernel on the three original
8192×4096 banks, reusing one input-local plane and emitting that joined result.
It preserves each bank's coefficient expression, K-update order, FP32 reduction
and BF16 stores. It could remove68 launches,34 concats and4.78 MiB intermediate
outputs per round, but reads all original weights and issues the same dot work.
More live accumulators can cause register/spill costs. The earlier one-row join
did not establish a material model gain. Defer this helper: the historical phase
marker does not justify predicting13 ms savings or consuming one of the two
decode slots needed to complete T4. No other verifier variant is proposed.

## Three-worker ownership

| Worker | One bounded scope | Owned files |
|---|---|---|
| 1: prefill | Direct indexed native K/V loads, preserving T16 selector/cadence and same native arithmetic; unchanged-loader clone first | New indexed-prefill helper/probe and reproducible native D512 header clone; no decode files |
| 2: N3 assistant/MLA | Horizon3 from the original full eight-row assistant block; four-row ancestry overlay; exact M1 broadcast4; native B4 equivalent to B1 | `glm5_dflash.zig`, `glm5_attention_overlay.zig`, `glm5_attention_decode_batch.zig`, `glm5_mla_verify_batch.zig`, MLA-only `glm5_dflash_model.zig` seam, dedicated probe/doc |
| 3: T4 KDA | Extend existing QKV hoist and retained endpoint to four rows, preserving original arithmetic and one retained FP32 endpoint | `glm5_dflash_a6_hoist.zig`, `glm5_dflash_kda.zig`, dedicated probe/doc |

Root owns the shared `dflash.zig` readout seam if needed, `attention.zig`,
HTTP/diagnostic policy, counters and all admission bills. Worker2 owns only the
MLA verifier seam in `glm5_dflash_model.zig`; worker3 does not edit that file.
Workers prepare isolated proofs and request narrow delegation after passing.
Do not overlap root callsites or the prefill worker's indexed-loader files.
No joint-QKV prototype, group-three retry, recurrence variant or width sweep is
part of this split.

## Required parity and memory gates

- Keep the same complete assistant eight-row forward. Check horizon3's first
  three logits, selector/lattice and resulting N3 tree against full readout;
  no truncation of transformer computation or destructive context publication.
- Check chain/fork four-row ancestry, odd/boundary offsets, invalid/future IDs,
  empty masks and distinct suffixes. Every native B4 output must match four
  native B1 outputs bitwise with one enclosing settlement. Serial/replay and
  fallback must remain on the same optional native target, never scalar mixing.
- Extend MLA query/value using M1 broadcast geometry, not head-M4 arithmetic.
  Compare T4 QKV-hoist output bits to original row projections and retain the
  same FP32 KDA recurrence/conv state. Keep one existing4 MiB retained endpoint
  per layer; do not add four endpoint planes. Test cache hits and replay misses
  against full serial state for every accepted ancestry.
- B4 gathered KV alone is8,400,896 bytes; the current packed temporary formula
  gives10,048,428 bytes. Reserve16 MiB/pending MLA layer (64 MiB async4), replacing
  the native B3 mode's8/32 MiB bill. Keep the authoritative256 MiB branch cap.
  The recorded32K-frontier W4 ledger149.735 MiB plus16 MiB fits, but recompute
  actual growth/capacity at admission and preserve mode-matched B1 fallback.
  Count valid cache rows only; no full-prefix copies or uninitialized-tail hashes.

Use small focused bit/state probes and one inclusive three-warmup/eleven-pair
component comparison where scheduling actually changes. Do not interpret
source-derived group counts as performance. Original BF16/FP32 weights,
BF16 compressed MLA caches, FP32 state/accumulators and resident embeddings remain;
there is no precision restoration. A4 may be selected explicitly after its
matched format proof, but keep the assistant identical in policy controls.

## One actual-model decision

After all T4 guards engage and strict serial token/complete valid-state gates
pass, run one matched192-output ABBA job at fixed8192 input IDs (the existing
official2048-ID prompt with WATER filler repeated4, not a pure-code fixture).
Load one target/assistant, warm both policies,
and clone the same immutable prefix into fresh requests. Compare current fully
optimized N2/children4 against complete N3/children4; hold native-mode and A6/A4
assistant settings identical: native target mode ON and A6 assistant fixed.
No other context or child-count sweep.

Use the server's192-output/191-forward rate convention; include all draft,
verify, replay, commit and endpoint cleanup costs. Record tokens and accepted
drafts per round, verifier rows, leaf hits/misses, counters, memory high-water and
all four elapsed times. Native serial/speculative equality is strict; any drift
against the scalar target is separately disclosed, not hidden by tolerances.
Root rejects a noisy or losing bundle without another variant and accepts/pushes
only a real speed/quality win. Final selected32K qualification remains a root gate;
neither N3 nor the unmeasured QKV alternative establishes60 tok/s in advance.
