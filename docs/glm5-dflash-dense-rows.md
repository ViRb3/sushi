# GLM DFlash retained BF16 row scheduling experiment

The retained KDA FA/GA (`128×4096`), beta (`64×4096`) and FB/GB (`8192×128`) weights
currently fall back to separate serial `Linear.apply` calls in tree verification.
Stock `Linear.apply` on multiple rows failed exact BF16 parity: four of 451,008
tested output values changed, including FA at three rows. That arm is rejected.

`glm5_dflash_dense_rows.project` instead reshapes the inputs to `[rows,K,1]`, computes
`weight[N,K] @ columns`, and reshapes the result to `[1,rows,N]`. In pinned MLX
`d73eb752`, this keeps N=1 and uses ordinary GEMV with the rows in the batch grid.
The matrix orientation, K, output width and GEMV template match the serial calls.
The shared matrix's zero batch stride prevents batch folding into a wider M.
No new Metal kernel or weight copy is introduced. The candidate accepts only GPU
BF16 inputs, materialized contiguous BF16 weights, the retained shapes above, and two to four
verification rows; unsupported inputs return null for the caller's fallback.

The isolated probe compares actual `Linear.apply` and the current serial
`linearRows` fallback, using BF16 normal weights/inputs at all five projection
shapes, three deterministic seeds and row counts two, three and four. Every
output uint16 was compared. The column schedule had zero mismatches over 451,008
values. A separate five-projection chain (FA→FB, GA→GB, beta) also matched exactly
at all three row counts before timing.

The 2026-10-03 component run used the isolated expression on `a40d6cd7`, subsequently
extracted unchanged into the guarded helper. The ReleaseFast test binary SHA256
is recorded with the raw samples. The run used foreground `taskpolicy -a`,
GPU lock owner `glm-dense-batch-v61`, maximum fans and ten seconds idle. All other
workers held CPU/GPU work. Ten warmup pairs preceded 31 alternating AB/BA pairs,
with four fresh chains per sample and one evaluation vector per chain. Inputs
and weights were materialized; this is a small resident component benchmark.
No prior measurement existed for this exact five-projection chain.
These timings cover the arithmetic schedule before the admission guards were
extracted; final helper parity and refusal checks passed after extraction.

| Verification rows | Serial chain median | Column batch median | Median paired time reduction |
|---:|---:|---:|---:|
| 2 | 250.729 µs | 210.698 µs | 16.34% |
| 3 | 223.854 µs | 172.145 µs | 22.97% |
| 4 | 244.322 µs | 168.114 µs | 31.91% |

At the current three-row verifier width the median difference was 51.709 µs per
five-projection chain. The verifier now tries this helper after affine QMM declines
when `SUSHI_GLM_DFLASH_DENSE_ROWS=1`; the default remains off. The independent
serial-row mode remains unchanged. An integrated three-row test compares all 384
BF16 output bits and checks dispatch engagement. Diagnostic JSON reports
`dense_row_dispatches` and mini-head storage/readout counters.
Full-model decode speed, actual-checkpoint state parity, KLD and wider rows remain
unqualified; the component result alone does not establish those properties.

## Full checkpoint qualification

The ReleaseFast diagnostic built from `e4be4673` ran the A6-trunk 2.3bpw target
and A6g128 assistant with N2/children4, async4, grouped experts and lane/down
enabled, prefix512/chunk128,64 committed inputs and one warmup. Dense rows were
enabled; mini-head, profiling and route capture were disabled. All64 output IDs
and complete committed state matched the independent serial reference. The
helper recorded4,608 calls;24 rounds accepted40 drafts, as in the prior arm.

Decode measured45.8676 tok/s against the inherited45.4485 tok/s group2 result,
while matched serial measured30.9358. Verifier time was1,189.197 ms versus the
older1,201.903 ms; draft146.348, replay43.211 and commit15.760 ms. Decode-only peak
was94,978,758,176 bytes. The roughly0.92% throughput difference is a single-run
qualification, not a robust speed estimate or evidence that the60 tok/s goal
is reached. Default remains off pending repeat and broader prompts.

The run used foreground `taskpolicy -a`, exclusive GPU lock, maximum fans and
ten seconds idle below90°C; no compiler or other GPU job overlapped. The fixed
binary and exact settings are preserved in private artifact
`glm53-dense-mini-20261003`. The complete checkpoint ReleaseFast suite also passed.
