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
