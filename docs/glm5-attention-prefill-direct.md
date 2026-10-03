# GLM single-split prefill finalization

Experimental `SUSHI_GLM_PREFILL_DIRECT=1` selects a direct single-split output
kernel. Unset or any other value retains the existing path. Decode and verify
blocks using eight splits are unchanged. The test override is inference-owner
only; it is not a per-request concurrent setting.

The common attention body preserves selected-token order, causal validity checks,
SIMD dot reduction, precise exponential and FP32 online update order. The direct
end reproduces the old one-part merge expression and final dtype cast. It removes
the FP32 partial/stat arrays and the separate merge kernel. Cache storage remains
BF16, and no KDA or IndexPool operation changes.

For H64/D512 BF16, cap each retained chunk output at the existing 8 MiB bound:
128 rows rather than the partial-storage bound of 63. Score planes retain the
existing 2 MiB limit. Score, negated score and partition output together account
for up to 6 MiB of explicit arrays; expanded IDs add 128*2051*4=1050112 bytes.
Partition-internal allocation remains additional, as in the existing path.
No chunks overlap. Complete 2048-row output remains 128 MiB, plus the existing
128 MiB concatenation destination; the candidate does not duplicate the full
cache or construct full-history attention scores.

A 2048-row indexed MLA call constructs 16 settled chunks instead of 33, reducing
eager waits from 363 to 176 across 11 MLA layers. These are source-derived counts,
not a throughput result. Default remains off until component and full-model
qualification. Tests must compare raw BF16/FP32 bits with the unchanged old
one-part merge, including masked rows, signed zero and NaNs, and exercise causal
pool/tail boundaries. No numerical threshold is relaxed.

The full4K-prefix/64-position diagnostic on the2.3bpw target subsequently
matched all64 reference top1 decisions and distributions: mean/first KL0,
cosine1, BF16 MLA and FP32 KDA. Head batching was forced off in both arms.
The reference and candidate used independent Requests and the same loaded
weights/teacher-forced continuation. Peak was95,336,773,556 bytes. Cold-reference
prefill9.166 seconds versus candidate8.498 is diagnostic timing, not a controlled
speed claim. The next gate is the shorter2K–16K llmprobe arm.

Private artifact `glm53-long-prefill-drift-20261003` retains `direct.json`,
source/binary/runtime hashes and exact flags. The gated
`glm5_prefill_drift.zig` diagnostic also reports raw logit mismatches on future
runs; this recorded run predates that additional counter. The option remains
off by default pending throughput qualification.

## Component qualification

ReleaseFast component tests passed 4/4 on MLX 0.32.3. Direct output matches the
original partial-plus-merge raw BF16/FP32 bits for dense and selected attention,
D512/H64/128-row geometry, unsorted selected IDs, future/invalid slots, all-masked
rows, signed zero, NaN and infinity inputs. A wrong all-masked result was
mutation-checked and failed the raw-bit test.

An integrated 129-row query starting at offset 2047 compares the original 63-row
chunk policy with the direct policy, through the real pool scorer, sorting,
ID expansion and causal-tail handling. Both BF16 and FP32 outputs match bit for
bit. The processed offset is unchanged. Decode's eight-split path is untouched.
These are component checks; full-model state/token parity and end-to-end timing
remain coordinator-owned gates before enabling the experimental flag.
