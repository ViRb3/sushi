# A4 current-native consumer result

A4 passed the fixed current-native correctness gate, but failed the declared
replacement-performance criterion. Ordinary decode was slightly slower;
predictable gain was smaller than control drift. A6g128 remains the default,
and A4g64 remains an optional supported consumer format. No ladder followed.

The frozen ordinary/predictable inputs contained **8756/8720 IDs**, produced
from pinned llmprobe 0.6.13's fixed 32768-byte code archive and fixed nonce.
Canonical request bodies, complete official rendered templates and exact
little-endian U32 arrays were hashed before the target job. No input was retuned.
The tokenizer-only run loaded no target weights and used no GPU stream.

One target load and both assistants stayed resident throughout. Each prompt
had one shared target prefill and independent settled assistant preparation.
Immutable prefix, both prepared contexts and serial 191/192 references were held
equally. One warm 192 request per assistant preceded A6/A4/A4/A6. Native B1/B3
and packed32 were on, N2/children4/async4 and chunk2048/async2 were fixed with
all accepted flags; the new HC prefill hook was explicitly off. Greedy requests
ignored EOS. Both prompts used exactly 192 outputs per arm.

All eight arms matched every output ID and all valid MLA latent/pooled/tail and
initialized FP32 KDA/convolution state. Ordinary consumed 191 inputs to offset 8947;
predictable consumed 192 to offset 8912. Capacity tails were excluded. Serial
native B1 engagement was asserted; measured B1 calls were zero because all
verification used B3. Shared-prefill B32 calls were 2299/2288.

## Complete measured arms

| Workload | Assistant | Decode seconds | Clone µs | Cleanup ms | Rounds | Accepted drafts | Committed inputs | Native B3 calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary | A6g128 | 4.001411791 | 11.209 | 0.419917 | 68 | 123 | 191 | 748 |
| Ordinary | A4g64 | 4.032204750 | 11.042 | 0.108333 | 69 | 122 | 191 | 759 |
| Ordinary | A4g64 | 4.051216791 | 10.667 | 0.085375 | 69 | 122 | 191 | 759 |
| Ordinary | A6g128 | 4.061230875 | 10.833 | 0.036750 | 68 | 123 | 191 | 748 |
| Predictable | A6g128 | 3.813484084 | 12.625 | 0.689709 | 65 | 127 | 192 | 715 |
| Predictable | A4g64 | 3.782208125 | 11.209 | 0.513500 | 65 | 127 | 192 | 715 |
| Predictable | A4g64 | 3.885336000 | 11.667 | 0.166709 | 65 | 127 | 192 | 715 |
| Predictable | A6g128 | 3.977112583 | 11.833 | 0.147167 | 65 | 127 | 192 | 715 |

Delivery rates use 191 post-prefill outputs / server-side decode interval;
actual committed counts are separate. Oracle work is outside timing. Clone and
cleanup are separately recorded and included in composed request cost alongside
shared target prefill and the corresponding assistant preparation. These are
private model intervals, not HTTP end-to-end timings.

| Workload | A6 / A4 delivery tok/s | A4 latency reduction | A6 control drift | Frozen gate |
| --- | ---: | ---: | ---: | --- |
| Ordinary | 47.3790 / 47.2572 | −0.2577% | 1.4839% | Failed |
| Predictable | 49.0335 / 49.8204 | 1.5795% | 4.2007% | Failed |

The predeclared rule was `g = 1 − mean(A4)/mean(A6) > 0` and
`g > abs(A6_last − A6_first)/mean(A6)` separately on both prompts, with zero
ID/state differences and admission mandatory. One ABBA does not establish
statistical confidence; the modest predictable movement is inconclusive.

## Phases and memory

Phase means divide total phase time by actual round counts.

| Workload | Assistant | Draft ms/round | Verify ms/round | Replay ms/round | Commit ms/round |
| --- | --- | ---: | ---: | ---: | ---: |
| Ordinary | A6 | 5.5949 | 51.7386 | 1.0849 | 0.8208 |
| Ordinary | A4 | 5.0127 | 51.6596 | 1.0911 | 0.7661 |
| Predictable | A6 | 5.6707 | 52.3832 | 0.9959 | 0.8345 |
| Predictable | A4 | 5.0334 | 52.0971 | 1.0183 | 0.7857 |

Faster drafting did not meet the whole-decode gate. Ordinary A4 required 69
rounds versus 68 with A6; predictable used 65 with both. Movement in unchanged
verification cannot be assigned wholly to assistant precision.

Shared target prefill was 11.146320/10.742765 seconds ordinary/predictable.
A6/A4 preparation was 60.386/50.871 ms ordinary and 67.504/55.610 ms predictable.
Composed request-generation rates were approximately 12.60/12.60 ordinary and
13.06/13.12 predictable output tok/s; networking and serialization are excluded.

Loaded active memory with both assistants was 95,323,270,904 bytes. Every arm
started at 96,531,122,296 bytes ordinary or 96,529,646,328 bytes predictable.
Maximum measured prefill/prepare/decode peaks were 97,761,855,556 /
96,277,901,088 /97,309,602,976 bytes. Both memory and wired limits remained
115,448,725,504 bytes. These equal-resident controls do not measure an A4
whole-engine RAM saving.

The full conservative bill was 14,264,893,440 bytes above actual active memory:
held target states 2,790,850,560; both assistant KV 1,468,006,400; both FP32
preparation allowances 4,294,967,296; target activations 3,221,225,472;
original scratch 289,406,976; all prefill transients 1,898,446,848;
native async4 33,554,432; fixed margin 268,435,456. No existing reserve or
already-held state was credited. Maximum admitted active memory was
101,183,832,064 bytes; all staged checks passed and source growth billed zero.

## Provenance and cleanup

Evidence key `glm53-a4-current-native-8k-20261003` retains exact bodies,
templates/IDs/hashes, both preparations, every arm/round/phase, ledger, complete
valid-state oracle results and telemetry. Accepted runtime was `af51e72f` plus
recorded default-off HC source hashes, with that hook explicitly disabled.
Binary SHA256:
`fe394116f77657331adbe2de34546ec3608a43984a3717248dd6ab63b4ce275b`.
Frozen corpus SHA256:
`12ff7d3fbaeca8e53b7da001bd1155c2c3777c4d2be04952bd563d21f3c62b75`.
The source-only reserved-name compilation error is retained; no numeric retry
occurred. Build/freeze and loaded job exited 0 (8 passed, 1 skipped each), with
foreground QoS, maximum fans and required idle. The process stopped, GPU lock
released and fans restored to auto; binary hash stayed fixed. No runtime/default,
installed-library, precision or CLI change was made for this consumer test.
