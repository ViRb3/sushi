# Horizon2 mini-head matched model rejection

Mini2 is rejected as noise/inconclusive by both frozen model criteria. Exact
serial target IDs/valid state and faster drafting passed, but neither inclusive
latency gain exceeded its full2 control drift. Accepted runtime remains
`4fcb541e`, with no default change, HTTP ladder, rerun or variant.

The complete proposal component had won 26.13% in all 11 pairs and preserved
selected original A6 logits/masks/proposals. All 12 captured real draft rows
retained top1 and 192/192 top16. That evidence justified this one model gate;
it did not establish a whole-engine win.

One target, one A6g128 assistant and one separate coarse head used the existing
frozen 8756/8720-ID ordinary/predictable corpus. Current HC/native B1/B3/packed32,
N2/children4/async4, chunk2048/async2 and greedy 192 outputs ignoring EOS stayed
fixed. Coarse, immutable target prefix, prepared assistant context and serial
191/192 references were held equally before every arm. One complete warm
request per mode preceded full2/mini2/mini2/full2. Full2 explicitly bypassed
legacy mini despite the resident coarse object. Partial N1 retained the ordinary
full8 source-head/seven-draft fallback in both modes, without horizon1 changes.

## Complete arms and fixed decision

The decision interval includes clone+decode+cleanup; oracle work is excluded.
Both pure and inclusive delivery rates use 191 post-prefill outputs. Actual
committed inputs are recorded separately. Every arm produced 192 exact serial
IDs and matched all valid MLA latent/pooled/tail and initialized FP32 KDA/
convolution state. Ordinary consumed 191 inputs to offset 8947; predictable
consumed 192 to offset 8912. No uninitialized capacity tail was compared.

| Workload | Mode | Inclusive seconds | Pure decode seconds | Clone µs | Cleanup ms | Rounds | Accepted drafts | Mini2 calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary | full2 | 3.921356416 | 3.920897458 | 11.417 | 0.447541 | 68 | 123 | 0 |
| Ordinary | mini2 | 3.916706584 | 3.916531209 | 10.084 | 0.165291 | 68 | 123 | 68 |
| Ordinary | mini2 | 3.936930207 | 3.936879333 | 9.958 | 0.040916 | 68 | 123 | 68 |
| Ordinary | full2 | 3.992594166 | 3.992538458 | 9.833 | 0.045875 | 68 | 123 | 0 |
| Predictable | full2 | 3.804451875 | 3.803987666 | 14.667 | 0.449542 | 65 | 127 | 0 |
| Predictable | mini2 | 3.815121125 | 3.814963750 | 10.666 | 0.146709 | 65 | 127 | 65 |
| Predictable | mini2 | 3.814127583 | 3.814068500 | 11.500 | 0.047583 | 65 | 127 | 65 |
| Predictable | full2 | 3.832359750 | 3.832302333 | 10.834 | 0.046583 | 65 | 127 | 0 |

| Workload | Pure full2 / mini2 tok/s | Inclusive full2 / mini2 tok/s | Inclusive latency gain | Full2 drift | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Ordinary | 48.2723 / 48.6413 | 48.2692 / 48.6399 | 0.7621% | 1.8003% | Failed |
| Predictable | 50.0243 / 50.0719 | 50.0209 / 50.0705 | 0.0990% | 0.7309% | Failed |

The frozen rule was `g = 1 − mean(mini2 inclusive)/mean(full2 inclusive) > 0`
and `g > abs(full2_last − full2_first)/mean(full2 inclusive)` separately on
both inputs. Neither passed. One ABBA is not a statistical confidence estimate;
no cold arm was dropped and no second run was requested.

Mini2 projection counts were 0/68/68/0 ordinary and 0/65/65/0 predictable.
Horizon2 counts equaled 68/65 rounds in every arm, and legacy-mini calls were
zero throughout. Measured native B1 calls were zero, with native B1 serial
oracle engagement separately asserted. Native B3 calls were 748/715 per arm; shared-prefill B32 calls
were 2299/2288. The model trace contains no N1 fallback engagement, so its
preserved source contract is not presented as an exercised model case.
Acceptance and actual committed counts were identical across modes.

## Phases, cold costs and memory

Phase means divide each pair's total phase time by actual rounds.

| Workload | Mode | Draft ms/round | Verify ms/round | Replay ms/round | Commit ms/round |
| --- | --- | ---: | ---: | ---: | ---: |
| Ordinary | full2 | 5.6837 | 50.6369 | 1.0189 | 0.8039 |
| Ordinary | mini2 | 5.3760 | 50.4952 | 1.0229 | 0.8075 |
| Predictable | full2 | 5.7294 | 51.2351 | 0.9190 | 0.8179 |
| Predictable | mini2 | 5.4749 | 51.4286 | 0.9205 | 0.8213 |

Shared target prefill was 10.187503/10.718970 seconds ordinary/predictable;
assistant context preparation was 59.580/65.400 ms. These are common setup
costs, not an inherited HTTP throughput comparison. Composed request cost in
the raw result includes shared prefill, that preparation, clone, decode and
cleanup; cold coarse costs are recorded separately.

Cold coarse preparation took 53.495666 ms and release 0.010875 ms. Before-build
active memory was 94,548,731,128 bytes; preparation peak/immediate-after value
was 95,103,821,048. Later settled resident memory was 94,826,276,088, exactly
277,544,960 above before-build, matching the declared coarse payload. After
release, active was 94,548,862,200. The immediate value includes transient
residency and is not substituted for settled coarse payload accounting.

Arm starting active memory was 95,824,412,280 bytes ordinary and 95,822,937,720
predictable, identical across modes. Maximum decode peak was 96,602,829,344;
maximum shared prefill/assistant-preparation peaks were 97,139,031,608 /
95,613,134,112. Both memory and wired limits remained 115,448,725,504 bytes.
The full 13,808,435,200-byte conservative bill stayed above actual active memory:
original one-assistant 11,383,406,592, plus 277,544,960 coarse residency and
2,147,483,648 cold-preparation envelope. Original target/assistant/head/cache/
reference/scratch reserves were retained, with no credit for already-held
arrays or sequential preparation. All staged admission checks passed.

## Provenance and cleanup

Evidence key `glm53-mini2-matched-8k-20261004` retains every arm/phase/output
ID, valid-state result, acceptance/count, ledger, cold/release value and telemetry.
The corpus was inherited unchanged from `glm53-a4-current-native-8k-20261003`;
real activation fixture `glm53-mini2-current-activation-20261004` was preserved.
Source baseline was accepted `4fcb541e` plus the recorded narrow helper/caller
seam. Binary SHA256:
`9ad2f965d2371ee669453177ddd327116f4bfae7feaab81170a29aa2d658bd2b`.
Focused ReleaseFast build and scoped-mode/frozen-input CPU protocol passed.
Loaded job exited 0 with all 9 tests passed: functional tests passed while the
recorded performance gate returned false. Foreground QoS, explicit accepted
libraries, maximum fans/idle and an exclusive per-job lock were used. Process
ended, lock released and fans auto; binary hash was unchanged. No precision
restoration, checkpoint change or speed claim beyond this fixed evidence follows.

Owned private model sources, helper dependencies and binary were hash-verified
and archived, then only the two owned model files were removed. The restored
caller's exact compiled patch was reconstructed and verified against its recorded
SHA256 in the artifact, without changing active source. Every result and the
original activation fixture remain preserved.
