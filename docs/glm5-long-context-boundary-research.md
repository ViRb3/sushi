# Long-context scorer boundary: bounded prefill recommendation

Recommend one separately opt-in extension of the existing IndexPool NAX
**prefill** ceiling from 8192 to 8448 pools, retaining T9–16 and serial 2048-pool
tiles. It targets the final chunk of nominal 32K requests; it does not accelerate
B1/B3 decoding. No fresh expert-NAX decode candidate is justified by the current
evidence. Runtime stays `4fcb541e`, round closure `21be0cee`; source-only research.

## Boundary and lifetimes

`glm5_attention.indexScores` passes `state.processed/4` to
`glm5_indexpool_nax.tryScores`. The latter declines above 8192 pools or at eight
or fewer query rows. Packed B32 selection invokes two unchanged T16 selectors.
[Current qualification](glm5-hc-collapse-http-result.md) had 33543/33579 input IDs,
so final 775/811-row chunks expose 8385/8394 completed pools and fall back to scalar
scoring throughout. With B32 cadence, a cap 8448 admits an additional 528/561
selectors across 11 MLA layers, leaving the final 7-row fragment scalar. These
are derived call counts, not traffic or measured latency.

Each scorer tile constructs flattened BF16 `[T*32,128]` queries, a pooled-key
slice/transpose, native matmul and the unchanged BF16-boundary epilogue. It
synchronously evaluates the score tile, appends a retained alias, then destroys
local Ops before the next tile. MLX's non-traced evaluator detaches inputs;
completed dot/query/copy temporaries are not retained by the score tile. Retained
FP32 tile scores feed lazy concatenate, negative, partition and expansion in
the caller Scope. That Scope persists until packed attention settlement/free.
B32 has two selector scopes; paired cadence may retain two B32 tiles. Do not
bill only one final score array or remove synchronous tile boundaries.

| T16 geometry | P8192 | P8394 | Proposed P8448 bound |
| --- | ---: | ---: | ---: |
| Pool tiles | 4 | 5 | 5 |
| Last tile columns | 2048 | 202 | 256 |
| Largest BF16 dot bytes | 2097152 | 2097152 | 2097152 |
| Full FP32 score bytes | 524288 | 537216 | 540672 |

Existing output/dot 2 MiB guards still admit the extension. The final small native
matmul has tail specialization; P8193's one-column tail may take GEMV, not NAX.
That backend distinction needs proof, not a performance assumption. Original
BF16 pooled/latent storage, capacity/growth bills, score 2 MiB and packed scratch
limits remain. Keep the old 8 MiB NAX transient bill, plus a conservative 512 KiB
**per pending layer** for the new mode: four T16 selectors times parts/concat/
negative/partition increases at most 256 KiB from the extra 256 pools, with margin.
No existing bill is credited. Component peak and the complete caller ledger
must verify this lifetime bound before model work.

## One gate and honest ceiling

Do not change the tile cadence, query width, retrieval budget, head/product
rounding, epilogue order or original ≤8192 behavior. The new domain changes scalar
dot order to already-used native FP32 accumulation/BF16 dots, so exact old scores/
IDs are not guaranteed. Preserve BF16 MLA and FP32 KDA/accumulators. A 16K quality
run would not engage this boundary and cannot validate it.

One worker owns a narrow existing-helper cap/bill change and private probe;
root owns integration/admission and actual-model gates. Tests first: boundary
8192/8193/8448/8449, odd histories, future/partial-tail exclusion, signed weights,
ties and nonfinite behavior; preserve old-domain bits. One complete final-chunk
selector plus unchanged B32 attention gate at P8394/T811 uses original-ID policy,
three warmups and 11 alternating pairs, with all preparation/evaluation/frees and
measured peak. No actual P8394/T811 plane fixture is currently established;
a root-owned capture of one current MLA final-chunk input bundle is needed
before that component. The existing 16K capture must not be duplicated and called
actual 32K. A single Q/index-Q/weight/latent/pool bundle is about 96.4 MB and is
capture-only outside clocks, not an additional persistent runtime cache. Record score drift/selected sets versus scalar; do not invent
stable tie ordering. Stop noisy/loss or scratch failure; no tile variant.

A winner must pass the frozen numerical bounds on nonrepeated code and prose
**above 32768 actual IDs**, using 24 declared affected late-prefix samples and 192
baseline-forced continuation positions each: meanKL≤.01,maxKL≤.15,top1≥95%, mean
NLL increase≤.02,newNF=0. Freeze the inputs/positions before execution; no threshold
adjustment. Then prove within-mode serial/spec IDs and valid states, and one
matched complete nominal 32K prefill ABBA with equal held references and every
old reserve. Improvement must exceed control drift before HTTP qualification.

The old synthetic P8192 selector improved 1312.583→905.271µs; it is not this
actual-tail baseline or a valid saving to multiply into verifier time. Only
~2.3–2.4% of these requests' token rows newly engage; late-row work can cost more,
but its actual share is unmeasured. This bounded fix may be small/noisy and does
not supply a 1500/60 forecast. The cap has no causal decode geometry benefit.

## Expert NAX exclusion

[The forced T3 expert NAX experiment](engine-exl3-experts.md#rejected-glm-three-row-forced-nax-chain)
already used original n36/MCG/W12 F16 storage/preparation, Hadamard boundaries,
FP32 accumulators and implicit M16/N32/K16 on one-to-three live rows. Production
289-window and bounded 25-window chains both lost; the latter remained 6.78–18.83%
slower, with only 1–2/11 wins. An unsorted/grouped wrapper could remove sorting,
but the padded consumer arithmetic is the same rejected work, not a new matrix
mechanism. Current singleton-heavy routes and nonadditive resource command-buffer
times establish no missing removable cost that overturns those results. No
reader/WIN/group/padded-MMA retry is recommended; the 60 goal remains open.
