# Bounded N2 draft readout horizon

`SUSHI_GLM_DFLASH_READOUT_HORIZON=1` is an experimental opt-in. Unset or `0` retains full readout (the usual diagnostic switch semantics).
Eligibility is the selected N2 policy with an eight-row assistant block and
mini-head disabled; other policies stay unchanged.

The assistant still computes all eight hidden rows, including its trained block
attention and convolution. Only the reused target vocabulary readout is narrowed
to draft positions one and two. The selector lattice receives anchor plus those
two hidden positions and two unary rows. Target verification and accepted-state
publication are unchanged; KDA state remains FP32 and MLA storage remains BF16.
Slices are views of the existing hidden tensor. At vocabulary 154880, the
projected BF16 output shrinks from 2478080 to 619520 bytes; candidate, unary, and
selector edge arrays also shrink from seven depths to two. No cache is added.

Best-first selection reads only the current depth unary and incoming edge. With
two selected nodes, only depths zero and one can be visited. Future-depth queue
entries created after the final selection are discarded. There is no backward
lookahead dependency on deeper lattice rows.

Focused qualification must compare the original first two readouts and candidate
IDs/unary/edge scores, then selected tokens and parents for sibling and chain
cases. Changes in matrix batch geometry require a numerical gate; no precision
restoration is introduced. Default stays off until parity and a phase timing pass.
The selected short run spent about 10.4% in all drafting, so this optimization
cannot by itself deliver the 60 tok/s target.

The fixed component check on M5 Max, MLX v0.32.3 (2026-10-03), used the actual
2.3bpw target's A6 group-128 head and the A6 assistant's retained BF16 selector
with the mini-head fixture's BF16 hidden seed. The first two vocabulary rows had
zero bit differences; candidates, unary scores, anchor/first-edge scores, and N2
trees agreed for children 1, 2, and 4. Nine focused tests passed. With foreground
QoS, maximum fans, and an exclusive GPU lock, four alternating warmed pairs
measured median readout+lattice+tree time 2.921 ms for full8 versus 1.239 ms for
horizon2 (57.6% lower). This excludes the unchanged assistant forward and target
verification; full-model acceptance and decode timing remain required.
