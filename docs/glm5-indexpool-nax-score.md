# IndexPool NAX score proposal

The bounded scorer is opt-in through `SUSHI_GLM_INDEX_SCORE_NAX`. Its initial
selector hook admits BF16 prefill chunks of 9–16 queries at 3584–8192 pools
(approximately 14K–32K history), with 32 index heads of width 128. Short decode and verify
rows, other geometries, and lower histories keep the original scalar scorer.
The top-512 retrieval policy and pool expansion remain unchanged.

The active GLM pack has 32 index heads of width 128, with four tokens per pool
and a 2048-token budget (512 pools). The existing `SCORE` kernel launches one
32-thread group per real query/pool pair, loops all 32 heads sequentially, and
performs a SIMD reduction for each head. At 16 queries and 32K history this is
131072 groups and about 537 million MACs. No existing profile found separates
this scorer from partition/expansion or latent attention; cold 2K dense-prefill
profiles do not consume sparse score selection.

The prototype flattens BF16 queries `[T,32,128]` to `[T*32,128]` and multiplies
them by a transposed BF16 pooled-key tile `[128,C]`. Native dense NAX computes
FP32 dot accumulators and stores BF16 dots. C is at most 2048 and T at most 16,
so the `[T*32,C]` dot plane is at most 2 MiB. The head-by-pool temporary remains bounded as history grows. Each tile is
settled and its dot plane released before the next tile.
Small FP32 score planes are retained and concatenated across pools.

| History | Pools | Query/head matrix | Pool tiles | Largest BF16 dot plane | Full FP32 score plane |
| --- | ---: | --- | ---: | ---: | ---: |
| 4096 | 1024 | 512×128 | 1 | 1 MiB | 64 KiB |
| 16384 | 4096 | 512×128 | 2 | 2 MiB | 256 KiB |
| 32768 | 8192 | 512×128 | 4 | 2 MiB | 512 KiB |

The epilogue keeps the original boundaries and order:

1. A pool is eligible only when `(pool+1)*4 <= offset+row+1`; otherwise its score
   is negative infinity, before reading its dot results.
2. Each dot has its BF16 boundary before ReLU.
3. Each `ReLU(dot)*weight` product rounds to BF16.
4. Heads 0 through 31 accumulate sequentially into FP32.
5. The final sum rounds to BF16 and is stored as FP32.

The negative/argpartition top-512 and pool-to-token expansion remain the current
operations. NAX changes dot accumulation order and may change BF16 score bits,
cutoff ties, and selected pools. The probe therefore records score drift,
per-row pool-set overlap, both cutoff tie counts, exact future exclusion,
BF16 boundary checks, uniqueness, and complete-pool/tail expansion. It compares
both score-only time and inclusive scoring/partition/expansion time, including
native matmul, tiled epilogues, evaluation and free. Inputs are independent,
fixed-seed BF16 fixtures at the actual dimensions, including signed weights.

The existing scalar kernel skips future pools before doing dots, while NAX
computes each rectangular tile. Near the start of a large prefilling chunk,
future-pool work can offset the tensor throughput gain. Native dense tile
padding and four tile evaluation boundaries can also lose. Neither fewer
dispatch groups nor arithmetic volume alone establishes a speedup.


## Focused component result, 2026-10-03

Three focused tests passed on MLX v0.32.3 / `64ea011c`. The fixed fixtures used
16 queries, 32 heads of width 128, and independent BF16 normal query/key/weight
inputs, including signed weights. Every active score retained the BF16 boundary;
future scores were exactly negative infinity; both selections remained unique
and causal with the correct partial tail. All 48 tested rows retained exactly
512/512 of the original pool set. Cutoff tie counts were identical in both arms
and ranged from 1 through 20 pools. This is fixture evidence, not a guarantee of
identical pool choices for arbitrary inputs.

| History | Scalar score, µs | NAX score, µs | Scalar full selector, µs | NAX full selector, µs |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 372.271 | 173.896 | 417.167 | 413.750 |
| 16384 | 711.709 | 502.542 | 775.709 | 525.396 |
| 32768 | 1235.791 | 825.542 | 1312.583 | 905.271 |

The full selector includes scoring, negative, partition, top-512 slice,
expansion, evaluation and free. Six alternating-order measured rounds followed
two warmup rounds, using fresh scopes. Full selection improved 32.27% at 16K
and 31.03% at 32K; the 0.82% difference at 4K is near noise, so the initial hook
starts near 14K to include the benchmark's actual 16K-rung prompt (16274
tokens, 4068 pools), while remaining in the same two-pool-tile geometry. No
separate 14K timing is claimed. Score bit mismatches were 0/7/6 over 16384/65536/131072 values;
relative L2 differences were 0 / 0.00002838 / 0.00003764 and maximum absolute
differences 0 / 0.015625 / 0.03125. No precision restoration was added.

The exclusive ReleaseFast run used interactive QoS, maximum fans requested,
51.26°C initial temperature and ten seconds idle. Raw source, command, scores,
pool overlap/ties, samples, binary/runtime provenance and telemetry are archived
privately. The opt-in selector hook and diagnostic control/counter API were
added after the component run; their integration build remains separate.
No model-level throughput claim is made by this result.

The admission ledger reserves an additional 8 MiB per pending layer for the
2 MiB raw dot plane, bounded native copies, query preparation and intermediate
score planes. Disabled mode and chunks of at most eight rows reserve zero.
This conservative control API is validated with the integration build.
