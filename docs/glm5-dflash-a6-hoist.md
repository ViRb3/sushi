# Exact three-row A6 coefficient reuse

The opt-in `glm5_dflash_a6_hoist` helper moves six-bit packed-byte reads, masks
and exact integer-to-FP32 conversions outside the verification-row loop. It
keeps twelve masked coefficients per output while processing three rows. The
original twelve multiply/add terms, input scaling, BF16 quartet sum, FP32
scale/bias update, SIMD reduction and BF16 output store remain unchanged.

The initial hook admits only the measured BF16 three-row geometry:
input 4096, output 8192, original U32 A6/group128 weights `[8192,768]`, and
BF16 scale/bias grids `[8192,32]`. It is selected through
`SUSHI_GLM_DFLASH_A6_HOIST`. Other formats, row counts and geometries retain the
existing row tile. No weights are decoded into a new bank, and there is no
padding, changed reduction or precision restoration. The existing affine-row
counter still increments; the helper also exposes `enabled()`,
`resetDispatchCount()` and `dispatchCount()` for engagement evidence.

## Production-bank component, 2026-10-03

One actual layer-0 `q_proj` bank was extracted from the active
GLM-5.3-Flash-Sushi-2.3bpw checkpoint, including its original weights, scales
and biases. The input was a fixed-seed BF16 three-row normal/RMS fixture.
All 24,576 BF16 output values matched the current row tile bit for bit.

| Arm | Median, µs | Change | Paired wins |
| --- | ---: | ---: | ---: |
| Current A6 row tile | 560.833 | reference | — |
| Hoisted masked coefficients | 491.459 | −12.37% | 11/11 |

Both arms included host apply, evaluation and free, with materialized resident
production bank inputs. Three warmups preceded eleven alternating AB/BA pairs.
The exclusive ReleaseFast run used MLX v0.32.3 / `64ea011c`, interactive QoS,
maximum fans requested, 46.19°C initial temperature and ten seconds idle.
Two focused tests passed. Raw source, command, production tensor hashes,
samples, binary/runtime provenance and fan telemetry are archived privately.

The source mechanism does not prove an instruction-count reduction by itself:
the compiler may already remove some repeated work. The result qualifies this
one geometry, without extending the timing claim to output/shared projections
or other row counts. Control/counter and the narrow caller delegation were
added after timing and require the combined integration build/model gate.
There is no full-model throughput claim from this component.

## Output-projection extension declined

A subsequent component at `2f446c5a` broadened only the guard to admit the
original layer-0 output bank, A6/group128 `[4096,1536]` U32 with `[4096,64]`
BF16 scale/bias grids. The unchanged shader produced all 12,288 three-row
BF16 outputs bit for bit. The generalized probe first failed on the original
QKV-only guard, then passed with this temporary extension.

Eleven alternating AB/BA single-call pairs measured 576.584 versus 535.834 µs,
but only seven pairs favored hoisting and several samples had large host
outliers. A bounded follow-up averaged eight fresh apply/evaluate/free calls
per sample, with three warmups and eleven alternating pairs. It measured
325.140 versus 316.026 µs, a 2.80% median reduction, again with seven paired
wins. Both arms drifted substantially across the run. These results do not
establish a repeatable output-projection gain, so the production guard remains
QKV-only. No full-model output-hoist arm was warranted.

Both runs used exclusive lock `glm-a6-output-hoist-v61`, interactive QoS,
MLX v0.32.3, maximum fans and ten seconds idle; the second run started at
54.20°C. Both focused tests passed.
The generalized probe retains fixture-derived dimensions and eight fresh calls
per sample. Exact source, temporary guard, tensor hashes, results and provenance
are archived under measurement key `glm53-a6-output-hoist-20261003`.
