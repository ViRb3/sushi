# Current routed traffic: measure before another kernel

Source-only research after the head-group rejection. Recommend **one bounded
named-kernel/resource diagnostic of the existing 42-chain replay**, rather than
another expert shader. The accepted packed32 stack leaves this routed consumer
unchanged. No prototype, build or GPU job accompanies this report.

## What the consumer already does

`glm5_dflash_ffn.apply` selects
`glm_group2.moeLayout(serial, grouped, lane)` at the qualified T3/H4096/I2048,
top8/E288/n36/MCG/W12/clamp10 geometry. Its chain is lane prepare, grouped gate/up,
lane middle, grouped lane down and weighted finish. `PAIR_SOURCE` dispatches
gate/up as the two z planes of one command. `MEMBERS` chooses original ordered
leaders/partners inline; paired members reuse each decoded weight before their
independent FMAs. The generated lane body already loads prepared inputs as
half4 and keeps the original serial reduction/F16 stores.

`INDEXED_COOP_SOURCE` reads one packed lane window per input tile, extracts four
codeword pairs, and calls `exl3_decode2`. MCG/W12 masks the codewords, applies the
fixed integer multiply/mask/xor and rounded half additions, then converts to
FP32 for the original FMAs. There is no expanded-weight plane or codebook table
load to eliminate. Removing that arithmetic through a new lookup would add
dependent memory reads; source does not establish a useful cost ceiling for it.
No unpack cache, precision restoration or lookup candidate is recommended.

## Actual ceiling and missing evidence

Artifact `glm53-current-routed-8k-20261003` proved all 516096 routed BF16 values
and recorded a 17.823292 ms median for 42 complete original-bank chains, with
0.383% range across three samples. It excludes routing/shared/KDA/MLA/drafting/
commit and is not whole-verifier attribution. Singleton splitting subsequently
lost 1.61%, 0/11 pairs; grouped middle/down fusion lost 11.52%, 0/11. Their exact
proofs did not establish removable register or traffic costs. The system trace's
95.39% compute interval union is global late-decode evidence, not EXL3 bandwidth.

Three captured rounds have 3024 assignments, 2563 leaders and 2102 singleton
leaders. Round0 has 878 leaders and 863 distinct layer/expert identities. One
three-projection expert contains 7077888 packed bytes at the actual dimensions:
round0 therefore has 6214385664 bytes of logical leader matrix visits and
6108217344 distinct packed bytes referenced. These are source/address accounting,
**not measured DRAM reads**, compulsory cache misses or a bandwidth estimate.
Repeated lane window loads, cache/coalescing and compiler lowering are unknown.

The actual ordered route records also show 310 same-layer expert identities
shared from round0 to1, out of 802 in round1 (38.65%); round1 to2 shares 338 of
795 (42.52%). Three rounds do not establish a reusable working set or an
amortization horizon. Banks are already immutable and resident; copying their
selected portions would add traffic. No temporal weight cache follows these IDs.

## One bounded diagnostic

Reuse the saved round0 X/IDs/scores/Y and all original E288 banks. One private
executor owns the existing target-only replay, original layer3–44 order,
absolute async4 boundaries and final settle/frees. Root owns scheduling and any
future runtime seam. Do not recapture, compact banks, split stages with waits,
change command-buffer limits or repeat a trace-template sweep.

Use a private, disposable MLX diagnostic build to attach metadata to the actual
pipeline/dispatch path, not an alternate shader. The narrow existing points are
`Device::get_kernel_`, `CommandEncoder::set_compute_pipeline_state`, both dispatch
methods and command-buffer completion. Retain the generated function identity,
grid/group dimensions, pipeline thread execution width/max threads, static and
explicit threadgroup-memory bytes, and command-buffer GPU start/end timestamps.
Correlate mixed command buffers honestly; their duration is not a per-kernel
duration. If public dispatch-boundary timestamp sampling is supported, place
fixed samples around the original dispatches and resolve them only after the
already required endpoint settle. No per-layer forced waits are introduced.

At startup inspect supported counter sets. Record actual memory-byte/cache/ALU
counters only when their documented semantics and sampling boundary are available.
Timestamps and pipeline metadata alone **cannot** diagnose bandwidth, occupancy
or spilling; unavailable counters must remain explicit missing evidence. Stop on
permission/capability failure rather than use a tensor-bearing Metal capture.
Keep a fixed 512-record/sample bound, under 128 KiB of diagnostic storage/output;
no tensor payload, general profiler framework or installed-library replacement.

In one loaded job, settle warmup, execute one unchanged reference pass and one
instrumented pass. Require all 516096 saved output bits again, retain both full
wall clocks and quantify instrumentation overhead. These are diagnostic passes,
not performance-acceptance samples. Accept attribution only with complete named
gate/up/down coverage, valid timestamp ordering and explicitly supported counter
semantics; otherwise report resources/timing only and stop.

This measurement can establish whether an eventual exact load/decode proposal
targets a substantial observed phase or merely moves cost into another command.
Any later candidate still needs stage/output bits and one complete 42-chain
three-warmup/eleven-pair gate, followed only on a clear win by root's matched
8192/192 N2/A6/native model ABBA, equal references, strict serial IDs/valid state,
acceptance and peak. T3 expert geometry has no 2K–32K switch; routes and the other
context-dependent costs do. No 60 tok/s forecast or new policy is supported yet.
