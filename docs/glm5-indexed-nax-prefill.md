# Indexed native D512 prefill: rejected

Direct selected-cache fragment loads were exact on the captured 16K MLA
fixture, but made inclusive T2048 attention **85.15% slower**. All 11 paired
rounds lost. The prototype is archived; no helper, default, production hook,
shared MLX change or reduced memory admission was retained.

| Arm | Median whole-attention time |
| --- | ---: |
| Current gathered KV, native D512, two-tile cadence | 130.376 ms |
| Indexed KV loads, same native body and cadence | 241.391 ms |

The paired median slowdown was 84.87%. These are component timings, not model
throughput. Both arms include fresh graphs, the current ordered selector,
gather or mask creation, attention, empty-row zeroing, settling, concatenation,
endpoint evaluation and cleanup. Three warmups preceded 11 interleaved pairs.
No isolated load-only speed claim or full-model follow-up was made.

The pinned body came from MLX `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`:
BQ32/BK32, D512, WM2/WN4, Q64 and 2051 selected slots per real query.
An unchanged contiguous packaging first matched current linked SDPA exactly.
The candidate then replaced only K/V fragment addressing, retaining native
score reduction, online softmax, MMA operations and FP32 accumulators. Q,
cache and output remained BF16; no arithmetic restoration was used.

Packaging plus indexed T33/T2048 proofs covered 68,714,496 BF16 output values.
All 4,268,131 ordered selected IDs matched. Masked nonfinite key zero, valid
key zero, head order, final ragged slot 2050 and final one-row cadence passed.
The conservative 64 MiB per-tile bound passed. Candidate peak growth in the
proof was 186,712,064 bytes above a resident fixture and held control output;
this includes candidate outputs and proof bookkeeping, not a net overhead
comparison.

A separate added cache-layout rejection assertion failed: the helper returned
a result where the probe expected null for a lazily transposed zero cache.
The raw assertion and source are preserved without repair or another retry.
Thus the full test exit was 1 (five tests passed, one failed), even though
numerical and timing gates completed. Two earlier compile-only packaging
failures are also archived. The decisive performance loss independently
rejects this candidate; exactness does not establish runtime acceptance.

Evidence key: `glm53-indexed-prefill-nax-20261003`, source HEAD `88f19e22` with isolated uncommitted source. It records source, shader, runtime, binary and
fixture hashes, build command, flags, compiler diagnostics, raw pairs, memory,
thermal and fan state, and cleanup. The copied Metal subset and attribution
were removed from the checkout with the rejected helper and probe. All three GPU attempts released their locks and restored automatic fans.
