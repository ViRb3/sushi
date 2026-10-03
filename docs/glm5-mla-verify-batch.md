# Three-row MLA native projection batching

The opt-in helper `glm5_mla_verify_batch.zig` handles the active pack's BF16
three-row MLA projections with original A6/group128 banks. It selects one
native broadcast query QMM at `[3,64,1,256]`, then one value QMM at `[3,64,1,512]` with the same broadcast geometry. Banks remain U32 `[64,256,96]`, with original BF16
scale/bias grids `[64,256,4]`. There is no decoded or expanded bank.

Native MLX v0.32.3 uses qvm for query and classic qmv for value. This
experiment reduces native call count while preserving M1 arithmetic for every
row. It does not force a NAX tile onto short rows. A research head-M3 arm
switched value to qmv_wide and changed the BF16 quartet bias-sum boundary;
that arm is archived and is not integrated. Its error is not NAX compound
rounding, so no batch-dependent target policy or precision restoration is used.

The environment name is `SUSHI_GLM_VERIFY_MLA_BATCH`; default is off. Scoped
`bind(on)`/restore and per-bank dispatch counters permit diagnostic controls.
Only BF16/GPU/NAX-capable hardware and the actual three-row bank geometry are
eligible. Other row counts and storage keep the caller's existing path.
The selected helper needs no permutation copies. Integration
must retain only the small branch outputs and respect the existing scratch
bound while batching the value call after branch attention.

## Component result, 2026-10-03

One three-arm component compared three serial calls, one original-geometry
broadcast call, and one head-rebatched M3 call, using fixed-seed BF16 inputs and
the actual 64-head/query256→512/value512→256 dimensions. Native dot/output
accumulators remain FP32 and operands retain their stored BF16 precision.

| Projection | Three calls, µs | Native M1 broadcast, µs | Head M3, µs | Selected arm |
| --- | ---: | ---: | ---: | --- |
| Query | 279.500 | 260.000 | 271.250 | M1 broadcast, −6.98%, 10/11 paired wins |
| Value | 286.333 | 285.250 | 261.375 | M1 broadcast, cost neutral; exact arithmetic |

Query broadcast matched all 98,304 BF16 outputs exactly. Value broadcast also
matched all 49,152 outputs, but its 0.38% timing difference was near noise.
Head M3 query was exact; value differed in 20,984/49,152 bits, with relative L2
0.00251950 and maximum absolute difference 0.03125. The selected exact combination
reduced component medians from 565.833 to 545.250 µs (3.64%). Its measured
query saving was 19.5 µs per layer (about 215 µs across eleven MLA layers before
composition), with value cost neutral; this is not a full-model decode throughput claim.

The ReleaseFast run used MLX v0.32.3 / `64ea011c`, interactive QoS, an exclusive
GPU lock, maximum fans requested, 48.70°C initial temperature and ten seconds
idle. Three warmups preceded eleven alternating ABC/CBA rounds. Slice/concat
or permutation copies, native QMM, evaluation and free were included. Two
focused tests passed. An earlier runner had a cooldown-expression typo; its
no-idle timings were excluded and the same fixed binary was rerun correctly.
Raw source, commands, samples, drift, hashes and telemetry are archived.

The helper/control API was added after the component run. Branch-loop
integration and model validation remain separate. The selected broadcast arms
matched reference bits, but whole-verifier token/state equivalence must not be
inferred from the component alone.
