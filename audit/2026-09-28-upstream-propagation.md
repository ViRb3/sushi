# mlx-serve → sushi propagation preparation

Compared on 2026-09-28 after fetching both remotes:

- sushi `main` and `origin/main`: `ca3b84b1b04c52e93dee1923ed8a292b42182c07`.
- ddalcu/mlx-serve `main`: `8222282a66258de0e5eafd84b5d3025b6eb19fcc`.
- Shared ancestor: `ef5e667db52976377e907ccf92acd051b721ea6a`.

This is a source-based triage of the recent Qwen4 and shared-engine changes, not a
runtime validation or an exhaustive audit of every post-fork commit. No runtime
code has been changed. Upstream commits are references for selective ports;
merging upstream wholesale would restore removed architectures and overwrite
sushi's EXL3, MiMo, cache, and attention work.

## First ports: observable correctness

| Upstream | Finding in sushi | Preparation / regression gate |
|---|---|---|
| [#601 · 68fba98b](https://github.com/ddalcu/mlx-serve/pull/601) SSD tier accounting | **Missing.** `persistQsaHistory` returns zero bytes when no new QSA checkpoint arrives, although the owned file survives. The SSM-only append still uses the older separate deltas. | Port `held_bytes` preservation and the before/after `nonChunkBytes` delta in `src/kv_disk_cache.zig`. Reproduce three turns: initial QSA checkpoint, spec/SSM-only commit, then extension without checkpoints. Assert entry and tier bills equal owned files; retain sushi's chunked restore and ring ownership rules. |
| [#550 · c38fd6e1](https://github.com/ddalcu/mlx-serve/pull/550) rescan recovers failed loads | **Missing.** `ModelRegistry.rescan` skips every existing ID, including `.error_state`. | Port the same-ID/same-path reset under the registry lock; preserve `registerStubWithMeta` and streaming metadata. Test repaired directory, refreshed size, cleared error, and unchanged live entries. Use a supported model type in fixtures. |
| [#552 · 6c814888](https://github.com/ddalcu/mlx-serve/pull/552) structured-output logprobs | **Missing.** `Generator.nextConstrained` has no wrapper retaining this position's raw logits and publishing its logprob. | Port the wrapper and token routine split in `src/generate.zig`. Test one correctly paired logprob per emitted token for JSON/schema constraints, including forced tokens and streaming/non-streaming parity. Adapt the model-backed fixture to sushi's loader. |
| [#553 · 7e859b04](https://github.com/ddalcu/mlx-serve/pull/553) affine KV scale dtype | **Missing; lower urgency for the usual bf16 activations.** `updateAffine` still grows scales/biases as bf16 unconditionally. | Preserve quantizer-emitted K/V scale dtypes. Test f16 and bf16 at initial allocation and growth, including `denseView`. This is shared KV correctness, not a claim that published bf16 packs currently fail. |

## Qwen4 performance ports worth preparing

| Upstream | Status / adaptation | Validation |
|---|---|---|
| [#558 · 3dc1dab5](https://github.com/ddalcu/mlx-serve/pull/558) GDN verify norm-gate and rollback concat in recurrence | **Missing follow-up to the already ported #517.** Sushi's `gdn_decode.zig` still emits recurrence then runs the norm-gate epilogue. | Port only the non-Hadamard path into `gdn_decode.zig`, `transformer.zig`, and needed MLX bindings. Compare output, every captured rollback state, and partial acceptance with the existing chain; then measure EXL3 MTP. |
| [#554 · 080a62e2](https://github.com/ddalcu/mlx-serve/pull/554) GQA-aware causal SDPA split | **Correction: Qwen4's served widths are already covered.** Before the older fixed-width fallback, sushi routes GQA=12 at widths 2–9 through `sdpaTickIdenticalGroups`, preserving the decode tick's window and split boundaries. | Preserve this path. Upstream's broader 10–15-row/other-GQA policy is a separate unmeasured extension, not a missing optimization for current Qwen4 MTP. |
| [#584 · 42b34e2c](https://github.com/ddalcu/mlx-serve/pull/584) batched Qwen4 decode ladder | **Missing.** Sushi has the generic opt-in ladder, not the Qwen4 batched default or its deferred-PLE guard. | Carry both `h` and deferred `mlp_out` into early eval. Flush the host-filled PLE leaf only when token IDs are available; otherwise skip the ladder. Adapt without requiring the optional GPU PLE port. Test lazy/eager IDs, batch and solo, explicit off, and output equality before measuring stride 4. |
| [#545 · 22749988](https://github.com/ddalcu/mlx-serve/pull/545) lazy next-round greedy MTP draft | **Missing new mechanism, despite older pre-draft code being present.** No padded `HeadPlace`/`forwardPlaced` or lazy built/kept counters. | Separate larger port in `generate.zig` and `transformer.zig`, explicitly Qwen4-only. Adapt all tagged-union switches for sushi's MiMo arm. Test full/partial/zero acceptance, cancellation, EOS/token budget, prefix reuse, lookup, and coexistence with the already landed greedy tail. |
| [#539 · e541ec86](https://github.com/ddalcu/mlx-serve/pull/539), [d500d429](https://github.com/ddalcu/mlx-serve/commit/d500d429) GPU PLE gather and opt-in default | **Absent; optional experiment.** Sushi uses host PLE and its own mmap/page-cache policy. | Treat both upstream commits as one proposal: `--ple-gpu` stays off by default. Port buffer lifetime, deferred gather, memory accounting and CLI precedence together. Verify host/GPU n-gram parity and memory pressure with EXL3 resident and bf16 streamed weights. Do not assume upstream throughput gains transfer. |

## Shared scheduler/API follow-ups

| Upstream | Disposition |
|---|---|
| [#568 · 3f7f0d9e](https://github.com/ddalcu/mlx-serve/pull/568) prefill/decode time sharing | Missing `--prefill-decode-share`. Prepare a separate scheduling feature after correctness fixes. Preserve sushi's admission, request-specific chunks, MiMo slot bills and inference-thread ownership. Test decoder progress during a long prefill, cancellation and accounting; measure TTFT and inter-token latency together. |
| [4e00f2af](https://github.com/ddalcu/mlx-serve/commit/4e00f2af) live KV residency | Missing server-side counter: sushi's `/props` reads `resident_hot_cache_bytes` only. Port scheduler publication and `/props` reporting, not Swift tray code. Account for sushi's ring buffers/shared storage, and avoid counting a restored entry twice. This changes telemetry, not admission policy by itself. |
| [#590 · eca42620](https://github.com/ddalcu/mlx-serve/pull/590) `ignore_eos` | Small applicable API feature buried in a large DFlash/app change. No `ignore_eos` handling in sushi's server. Extract only request EOS selection and tests for chat/completions, stream/non-stream, token limit and separately active stop strings. Defer the unrelated DFlash/drafter catalog and app update work. |
| [#574 · 0ae3f66b](https://github.com/ddalcu/mlx-serve/pull/574) failed MTP load cleanup | The generic loader cleanup is absent, but this is not the native Qwen4 in-checkpoint head implementation. The scheduler still attempts generic `loadMtp` for non-MiMo sidecars, so audit malformed/partial sidecar reachability and apply ownership fixes if reachable. Do not bring in Nemotron or HY3 architecture work. Use missing-tensor fixtures and active-memory checks. |
| [#578 · a358fc71](https://github.com/ddalcu/mlx-serve/pull/578) pulled model `org/name` registration | Missing API change in `registerByPath`, but upstream's triggering caller is Ollama pull, which sushi removed. Defer unless a retained sushi caller needs explicit IDs. Do not restore the Ollama surface to obtain this fix. |

## Already covered or superseded: do not duplicate

| Upstream | Evidence in sushi |
|---|---|
| [#580 · 185bfe2c](https://github.com/ddalcu/mlx-serve/pull/580) batched S=1 fused QSA | Sushi already sets `QSA_ATTN_MIN_S_DEFAULT = 1`; batched dispatch consults that floor and calls `qsaAlignedSparseAttn`. Upstream separates solo floor 2 from batch floor 1; copying that policy would undo sushi's unified per-position arithmetic. |
| [#555 · 9a546737](https://github.com/ddalcu/mlx-serve/pull/555) dense bf16 fused QSA verify | Sushi already serves dense and packed rows with its own split-K body and extends packed widths to 40. Compare alignment/strided-view regressions as test ideas; do not replace the kernel or regress the packed-cache memory bounds. This is coverage of the capability, not proof that every upstream edge-case test passes. |
| [79365009](https://github.com/ddalcu/mlx-serve/commit/79365009) “fast qwen4”, plus [#519](https://github.com/ddalcu/mlx-serve/pull/519) / [#534](https://github.com/ddalcu/mlx-serve/pull/534) rows dispatch | Sushi's affine `moeDecodeDispatchArm` already allows `B*S` from 2 through 16 without an Ultra gate. EXL3 returns through `moeExl3` before that affine path. There is no new all-chip EXL3 speedup to cherry-pick here. |
| [#556 · 364b9fd0](https://github.com/ddalcu/mlx-serve/pull/556) pooled-key upkeep | Ported as `fc1c0ff5`. |
| [#517 · c24c70a0](https://github.com/ddalcu/mlx-serve/pull/517) GDN prework/recurrence fusion | Ported as `ad5e6be8`; #558 remains separate above. |
| [#575 · b25765be](https://github.com/ddalcu/mlx-serve/pull/575), [#527 · a7a0dcc3](https://github.com/ddalcu/mlx-serve/pull/527) hot-cache default and restore fixes | Adapted in `7eb1a87a`, with further restore peak work at `21a50179`. Preserve those changes while porting #601. |
| [#523](https://github.com/ddalcu/mlx-serve/pull/523) / [#533](https://github.com/ddalcu/mlx-serve/pull/533) prompt lookup | Ported in `caa02a4f`, recorded in NOTICE. |
| [#602 · 8222282a](https://github.com/ddalcu/mlx-serve/pull/602) sampled MTP greedy tail | Already on sushi main as `a66a57af`. |
| [#569 · 468a11a3](https://github.com/ddalcu/mlx-serve/pull/569) exclusive listener | Sushi's `startListener` already creates the socket with `SO_REUSEADDR` only and binds before loading; `df0bd4a9` and `tests/test_port_conflict.sh`. |

The small-model fusion bundle `90991390` reports Flash Next flat and targets
other architectures; it is not a priority EXL3 port. Ollama capability reporting,
GGUF fallback, Nemotron, image generation, and Swift UI changes are outside the
current two-model sushi scope.

## Landing sequence and acceptance

1. Separate correctness commits: #601, #550, #552, then #553. Each starts with a
   failing behavioral regression and updates the matching engine/server doc.
2. Measured #558 kernel port, then the guarded #584 ladder; preserve sushi's existing Qwen4-specific #554 equivalent.
3. Qwen4-only #545 MTP orchestration after the smaller changes settle. Keep MiMo
   behavior unchanged and test both served families.
4. Separate feature work for prefill sharing, live KV telemetry and `ignore_eos`.
   GPU PLE remains an opt-in experiment until its memory/performance gates pass.

Use isolated branches from the pinned sushi head when implementation starts;
do not repurpose any existing worker checkout. No worktree or implementation
branch was needed for this preparation. Preserve the pre-existing CLAUDE.md edit.

For each port: behavioral test red → minimal port → ReleaseFast build and full
unit suite → relevant integration tests. Kernel tests must engage the changed
arm and cover output/state parity, not merely successful execution. Full-model
tests and benchmarks acquire the GPU lock per run. Use existing recorded
baselines when applicable; follow `docs/process-measurement.md` for new timing.
Changes that alter output require the repository's lossless-teacher KLD gate.
Update NOTICE with the upstream author and adaptation in the same landing.

No builds, model loads, GPU tests, benchmarks, commits or pushes were performed
for this inventory. Implementation readiness here means identified source
changes, dependencies, exclusions and regression gates, not validated patches.


## Follow-up: first performance port

Correctness ports #601/#550/#552/#553 were rebased over `--fast` and merged into
main as `6c9ae932`. The completed worktree was reused on
`codex/qwen4-upstream-perf`, based on `ad47f8bb`. Upstream was refreshed through
`a4285ef7` (the extra commit changes release validation, not engine kernels).

The #558 adaptation is implemented on that performance branch. It preserves
sushi's stored bf16 carry state and probes pipeline capability on independent
inputs so deferred PLE remains lazy. The width sweep found a two-row win and a
five-row regression, so the default serves two rows only; see
[the measurements](../docs/perf-baselines.md#gdn-verify-fold). The next missing
port is #584; #545 and optional GPU PLE remain separate work.


Validation of the two-row default: ReleaseFast build succeeds; the final full
suite reports **2732 passed, 93 skipped, zero failures**. Six saved live MTP/serial
output pairs (chat, streaming chat, Messages and copy tasks with lookup on/off)
match byte for byte. The last live boot was initially refused by memory preflight;
after memory settled, its three remaining checks passed (fixed depth, seeded
stream/non-stream equality and lookup engagement). GPU locks were released and
fans restored to auto. The performance commits have not been merged into main.
