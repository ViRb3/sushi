# Optimized N3 assistant and T4 MLA qualification

Uncommitted infrastructure for one bundled N3/children4 policy; N2 and the A6
assistant remain defaults. The assistant still computes its complete eight-row
block. Horizon3 projects only its first three draft rows and keeps anchor plus
three hidden rows for the selector. Overlay tails admit four ancestry rows.
Native B4 and M1 query/value broadcasts extend batch metadata, preserving
original BF16 operands/output and FP32 arithmetic, with no restoration.

`scratchLimitForBatch(rows)` and `transientBudgetForBatch(pending_layers,rows)`
reserve 16 MiB/layer for B4, 64 MiB async4. Existing B1/B3 `scratch_limit` and
`transientBudget` retain 8/32 MiB. The authoritative 256 MiB branch cap is
unchanged; native B1 fallback remains mode-matched when a full batch cannot fit.
Root owns policy, clipping and admission; the worker owns the MLA verifier seam.

The fixture-only `glm53-dflash-t4-mla-20261003` run passed four focused tests.
All 262,144 B4 output bits matched four native B1 calls across constructed
chain/fork ancestry using the captured 16K final four Q/latent rows. Candidate
peak delta was 9,281,710 bytes, under 16 MiB; its baseline retained fixture and
B1 graphs/outputs, so it is not net overhead versus control. Exact M1 query and
value broadcast4 checks passed against independent M1 calls.

The real A6 target head/assistant selector proof matched all 464,640 first-three
logit bits and selector/lattice/N3-tree values against full readout. The full
eight-row hidden fixture was retained. No target/assistant model was loaded.

Whole four-row selection, explicit ancestry, gather/native SDPA, one endpoint
settlement and frees measured 1599.875 µs for four B1 calls versus 1533.667 µs
for B4 after three warmups and eleven alternating pairs: 4.14% less median time,
9/11 wins, with two losses of 9.96% and 2.02%. This modest/noisy component result
does not establish an N3 throughput win. Actual overlay4 fallback, complete
token/state parity, shortened trees and real N2/N3 performance remain bundled
model-gate requirements.

The ReleaseFast binary SHA256 starts `642b0e32266ecebe`; exact source hashes and
raw outputs live under the artifact key. Runtime is MLX 0.32.3 / `64ea011c` and
patched mlx-c `56b2d39`, foreground `taskpolicy -a`, exclusive GPU owner
`glm53-dflash-t4-mla`, maximum fans and ten seconds idle after a 47.06 °C status
reading. No implementation commit, policy default change or speed acceptance
is implied by this qualification.


## Final bundle disposition

The [matched N2/N3 model decision](glm5-optimized-n3-result.md) passed exact
192-token and valid-state checks, but gained only0.56% amid1.46% control drift.
The bundle was rejected; this component's source/probe was archived and removed,
and accepted runtime behavior restored. No standalone runtime change was landed.
