# GLM DFlash cache reservation

The diagnostic HTTP owner may reserve lossless MLA storage after prefill and before
cloning the request or constructing its first speculative round. The requested
frontier is checked `input_tokens + max_output_tokens + verify_rows`.

`glm5_dflash_reserve.plan` computes 256-row-rounded latent and pooled capacities.
Its additional-peak bill counts each full replacement allocation plus its zero
padding, summed conservatively across affected buffers and layers. The caller
must include this bill in admission; `reserve` validates all active states and
rejects insufficient available peak bytes before allocating or modifying them.

Reservation evaluates and replaces one buffer at a time on the inference owner.
It preserves the existing dtype and all stored bits, including the valid latent
and pooled prefixes. It leaves processed offsets and compact key/gate tails
untouched. A later execution failure requires the caller to dispose the request;
validation and memory rejection leave it unchanged. No old replacement is retained
after the operation's temporary scope ends.

The existing 256 MiB branch scratch policy stays unchanged. Pre-reserving enough
capacity removes its doubling-growth charge at 128K; it does not change attention,
pooling, recurrence, verification, sampling or token commitment arithmetic.

## Qualified checks

The CPU ledger verifies the real 512/128-wide BF16 geometry at 131072 tokens:
growth rejects before reservation; reserving through 131331 yields 131584 latent
rows and 33024 pooled rows. The unchanged scratch planner then admits one branch
at the last output frontier, with an additional-peak reservation bill of
143785984 bytes per MLA state.

The focused ReleaseFast suite passed on MLX 0.32.3. GPU checks compare raw cache
bits, including signed zero, for BF16 and FP32 at 1023 and 2047-token boundaries.
They cover independent prefill, latent and pooled growth, subsequent appends,
dense and sparse attention, unchanged tail handles and offsets, idempotent
reservation, and rejection of insufficient headroom or an invalid second layer
before either layer changes. Overflow and invalid geometry/item widths are
CPU-checked. These checks establish cache-helper equivalence; the HTTP owner
still supplies request admission and full-model execution qualification.
