# Engine: prefix cache and SSD-first

How prompt-prefix KV reuse works: the hot RAM cache, hybrid (GDN/QSA) restore points, the SSD tier and SSD-first
mode, checkouts and donations, spec state riding the cache, and GLM's native state. Read this before touching `src/prefix_cache.zig`,
`src/kv_disk_cache.zig` or `src/kv_disk_writer.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kv-cache](engine-kv-cache.md),
[engine-memory-admission](engine-memory-admission.md), [engine-mtp](engine-mtp.md),
[arch-mimo-v2](arch-mimo-v2.md#sliding-layers-the-ring).

## Code map

| File | Role |
|---|---|
| `src/prefix_cache.zig` | Hot prefix cache (`--prefix-cache-entries`, `--prefix-cache-mem`) |
| `src/kv_disk_cache.zig` | SSD tier (`--prefix-cache-disk`) |
| `src/kv_disk_writer.zig` | SSD-first mode's background writer thread |
| `src/restore_dump.zig` | Prefix-cache restore diagnostics (`tests/diff_restore_dump.py`) |
| `src/glm5_prefix.zig` | GLM restore points: KDA checkpoints, MLA rows, their SSD layout ([GLM](#glm)) |

## Basics

- KV reuse via prompt-prefix matching; invalidated after tool calls + pad-only gens (`commitDeclinesPadOnly`: only an
  ALL-pad generation declines); hot cache spills to SSD; RAM invalidation propagates to disk.
- Restore ALWAYS clamps (`truncate(final_len)`); a failed restore hands back an EMPTY cache; every eviction loop has
  a no-progress exit (checked-out entries are unevictable). A restore leaves a token to forward: a one-token prompt
  prefills cold (its full hit restored everything and the empty prefill crashed upstream).
- **Media keys are a CHAIN** (`MediaSpan`: per block, a hash of its pixels, its position and every block before it;
  the entry key is the last). An entry keyed by a request's block k restores up to block k+1, so a turn that appends
  a screenshot reuses everything before it; any other key mismatch shares only the text before the first media row
  (`crossKeyBoundary`). The splice resumes at the placeholder count inside the restored prefix.
- **A restore is not bit-identical on a HYBRID** (≤ 0.047 nats; the chunking class ~0.3 nats top-5 for QSA state) ⇒
  byte-stable greedy needs `--prefix-cache-entries 0`. A hybrid cache hit moves the top logprob ~0.2 nats, so any
  scorer boots with the cache off.
- The always-on SSM snapshot sits 30 tokens BEFORE prompt end; a restored tail inside that window forwards as ONE
  span (`ssmSnapshotBackoff`). Guard: `tests/test_hybrid_reuse_equivalence.sh`.
- **A ringed (sliding-window) entry restores at its end or at one of its ring checkpoints**
  (`KVCache.ringCheckpoint`, `Entry.ring_cps`, up to `RING_CHECKPOINT_MAX` = 8): each ringed layer's
  window + 30 rows at a position (down to the window when the ring holds no more, as one restored off a checkpoint
  does); the slot's own are its restore point, its prompt end and its message marks (`SlotRingCps`). A reply longer than the
  ring's slack compacts it past where the next turn diverges (the previous reply re-renders); the checkpoint's rows
  go under the ringed layers (`restoreRing`) and the usual clamp follows.
  A checkpoint restore of fewer than `RING_RESTORE_MIN_TOKENS` (64) cold-prefills: below it a restore cost more than
  the cold prefill it replaced.
  Below both, `SlidingRingRewindPastWindow` → cold prefill, and the declined entry keeps its recency: promoted, a
  header-only match made the entry a later turn needed the next count-cap victim.
- **A ringed prefill marks the message starts it forwards** (`ringMarkPositions`: `<|im_start|>` past the restored
  prefix and short of the prompt end's reach, at most `RING_MARKS_MAX` = 4, thinned keeping the first and the last):
  each ringed KV write fills a mark it reaches before its compaction drops the rows (`KVCache.ring_marks`), so no
  chunk is split and the forward is unchanged. A new session sharing only another's system prompt and tools
  diverges inside its first user message, below every fork and prompt end, and restores at that mark.
  Measured (MiMo 2.3bpw, kv8, MTP and prefix cache at their defaults, a 12,042-token tools + system prefix, a
  ~300-token first task, 12,346-token prompts, `taskpolicy -a`, lock per boot, busy box, 2026-10-01). One boot of
  63476cd1: the first session cold-prefills in 9,876 ms; the second and third restore 12,037 / 12,042 tokens and
  prefill in 434 / 420 ms. One boot of main 819b4751: the first session cold in 15,682 ms, and the second and third
  cold again (`cached_n` 0) in 14,970 / 14,582 ms. Splitting a chunk at the boundary instead would cost ~0.4 s per
  split on MiMo (the fixed per-chunk cost the 2025- and 4096-row prefill meter rows imply).
- **Ring checkpoints thin span-preserving** (`thinRingCps`, on merge, inheritance and shed): the lowest (a shared
  preamble's mark) and the newest stay longest; kept highest-first, a conversation's later turns pushed the preamble's
  mark out after two turns.
- **The SSD tier restores a ringed entry only at a ring file** (`bestRingMatch`, `restoreIntoRinged`): chunks hold the
  global layers, `r{pos}.safetensors` each restore point's ringed rows (the RAM entry's checkpoints plus its end,
  `RING_DISK_MAX_PER_ENTRY` = 8 kept, thinned as in RAM, salvaged per file at scan; manifest v9, which an older reader
  drops) ([arch-mimo-v2](arch-mimo-v2.md#sliding-layers-the-ring)).
- **A disk restore fills its buffers chunk by chunk** (`restoreKvInto`): each chunk is evaluated into buffers
  allocated at the restored length before the next file opens. A lazy `mlx_load_safetensors` holds its file open until
  eval (one eval at the end failed past the soft limit of 256 files), and a concatenation at the end held every chunk
  beside the result, twice the restored KV before any bill saw it. Each chunk's eval is drained before the next chunk
  writes: undrained, a write that beat the command buffer's release copied the whole buffers instead of donating, up
  to three copies of the restored KV at once on CI's M1 VM. The restore entry points drop the MLX latch they
  raised, or the cold fallback's prefill fails on it. Measured on a 150k-token Sushi-3bpw entry (147 chunks; b9dbbf53
  plus this change, `--ctx-size 262144 --prefix-cache-disk 20GB --prefix-cache-entries 1`, arms O P M M P O, 4
  restores per boot, `taskpolicy -a`, fans max, a lock per boot, 2026-09-27): a warm restore takes 170-177 ms with the
  fill, 185-188 ms with a per-chunk eval and the final concatenation, 183-187 ms on b9dbbf53; the first restore of a
  process takes 611-618 ms with the fill, 642-917 ms without it, 307-326 ms on b9dbbf53 (one eval per chunk).
- **A commit that forked off another entry inherits that entry's ring checkpoints below the fork** (`bestRingDonor`,
  refcount-shared and billed per entry like SSM checkpoints): a request appending to the conversation (a client's
  side request: the chat + a reminder) otherwise holds only its own prompt end, and once the count cap evicts the
  main entry the next main turn, diverging where the reminder was appended, cold-prefilled every turn. The slot's
  own checkpoint at its restore covers a donor that another slot's commit evicts before this one commits.

## Candidate ranking and trimming

- **Hybrid candidates rank by RESTORABLE checkpoint position, not raw match** (`findBestRestorableMatch` RAM,
  `bestHybridMatch` disk). Ringed candidates rank by `ringRestore`; an un-restorable one stays eligible at 0, so a
  lookup with nothing better still declines by name.
- **A lookup that restores 0 rows is not a use**, SSD-first or not (a hybrid with no usable checkpoint, the QSA
  history decline): the entry keeps its recency and its admission protection drops, else the count cap's next
  victim is an entry that can serve.
- Checkpoint retention thins the INTERIOR with a dense newest quarter (`spanPreservingDropIndex`, `ThinPolicy`).
- An oversized candidate is TRIMMED to the longest restorable prefix that fits (`trimLenForBudget`,
  `KVCacheSnapshot.trimmedCopy` is a REAL copy); a QSA trim bills the bank on the final retained checkpoint.
- **A ringed candidate trims only where it restores** (`ringTrimLen`): its end, dropping the reservation's spare
  capacity, or a ring checkpoint whose rows become its ring (`trimmedCopy`'s `ring_cp`). Its ringed layers are a
  constant, not a per-token price: priced per token, every MiMo target fell below the ring and every entry declined.
- A decline carries its `TrimDecline` reason; a RAM-budget decline spills to SSD (`spillDeclinedToDisk`).

## Budget

- An in-place SSD commit keeps the bill for an owned QSA history file when no new QSA checkpoint arrives; sidecar-only commits bill the change in all retained non-chunk files, including rings.

- A commit declines and frees its incoming snapshot when checked-out residents prevent satisfying either the entry-count or byte cap; the request continues and one `[hot-cache]` line names the limiting cap.
  The byte cap is judged AFTER the new entry sheds checkpoints (`retainNewEntry`): a qwen4_exp trim is priced against
  its shed survivors, and judging it unshed declined every session past the budget, so each turn cold-prefilled.

- **The hot-cache budget is CLAMPED at load** to what the weights leave under the GPU ceiling and is a HARD cap; it
  FOLLOWS residency (`reviseHotCacheBudgets` after every load/unload, repeated for 10 s because the OS returns pages
  lazily).
- **An unnamed `--prefix-cache-mem` holds one session at the working context** (`oneSessionFor`, >= 2 GB, both arms)
  where the ceiling holds it beside the weights, the n-gram page cache (`page_cache_claim`) and a cold full-context
  prompt's bill (MiMo refuses rather than evicts); else that room, at most half the bill, so an outgrown session's
  trim copy fits beside it. A flag stands, `2GB` too; context sizing and the chunk pin still read the raw ask.
- A replacement over the budget sheds ring checkpoints (thinned as above) before the entry goes (`shedRingCheckpoints`).
- Measured (b9dbbf53 plus this change, Sushi-3bpw, auto context 1M, kv8, MTP on; a ~200k-token three-turn session; `taskpolicy -a`,
  fans max, GPU lock per boot; 2026-09-27): unset, the budget is 11516 MB and turns 2-3 prefill in 0.35 s (199.7k
  reused); `--prefix-cache-mem 2GB` keeps a 139k-147k prefix and prefills in 36.0 / 31.3 s (turn 1: 114-115 s cold). The
  n-gram table stayed 100% resident (mincore) in both arms.
- Eviction is WORKLOAD-fair (`cache_key`: `prompt_cache_key` > `metadata.user_id` > system-prompt hash;
  `lruIndexExcluding`).

## SSD-only storage

`--no-prefix-cache-ram --prefix-cache-disk 10GB` keeps reusable text prefixes on SSD without retaining idle
KV snapshots in the RAM cache. The live request still needs KV memory, and queued disk writes can hold
buffers temporarily. The entry count must remain positive: `--prefix-cache-entries 0` disables both tiers.
With RAM and disk disabled, SSM checkpoint capture is disabled too. `/props` reports
`settings.prefix_cache.ram_enabled=false` and `mem_bytes=0` when RAM retention is off.
Context and prefill-chunk sizing also reserve zero idle-cache bytes in this mode, regardless
of `--prefix-cache-mem`; live KV and temporary SSD write buffers still consume memory.

Qwen prefill chunks write through continuously in SSD-only mode. Hybrid SSM checkpoints, MiMo ring
restore points and GLM state ([GLM](#glm)) survive restart. Image-bearing entries remain ineligible for disk persistence. RAM+SSD defaults
are unchanged. `SUSHI_PREFIX_CACHE_DIR` can select an absolute cache directory; unset, the root stays
`~/.sushi/kv-cache`. Live tests use a separate root without changing home settings.

Ported from [mlx-serve #680](https://github.com/ddalcu/mlx-serve/pull/680), with Sushi's ring checkpoint handling.

## SSD-first

- Disk fingerprints include the model path, config size/mtime and overrides, plus sorted indexed weight-shard (or unindexed safetensors) names and size/mtime and `ngram_table.bin` size/mtime; payloads are statted through symlinks, never content-hashed.

- `prefix_cache.ssdFirstActive` = a disk tier AND (capable arch OR RAM retention disabled), mirrored onto `HotPrefixCache.ssd_first` +
  `DiskTier.ssd_first`: with RAM enabled it floors at ONE session, `--prefix-cache-mem` = the IDLE allowance.
  SSD-only storage retains no idle RAM entry.
- Spill and EVICT are two decisions (`PersistOutcome`: only `.persisted` + an agreeing index + landed files license
  discarding RAM); writes ride `kv_disk_writer.zig` (FIFO, `meta.json` last, epoch fence at the ONE removal site);
  per-chunk write-through; a diverging turn hard-links the donor's LANDED chunks; a full-prefix hit CHECKS the entry
  OUT so the first append donates.
- **The free-space probe runs only before an actual store** (after the superseded check): the idle spill commits every
  idle entry at each request finish, so a copy already on disk must cost no probe; below the store floor it still
  counts as persisted.
- **A checkout is a PROMISE until the append DONATES** (`donateCheckout` right before `Generator.initWithOptions`,
  below every refusal; `releaseCheckout` hands an undonated entry back intact).
- **Off SSD-first, a warm share that does not fit is taken over, not refused** (`checkoutRestored`, qwen4_exp's
  admission pass): a full-entry hit is checked out on demand and billed as donated. The tradeoff: a request that
  fails after donating loses the entry. Disk checkpoints come off the TOP of
  the flush budget; the disk tier serves the pre-media text prefix only.
- **A media commit persists its text to the SSD tier, never a media row** (`diskTextLen`): the record stops at the
  first item (none when the boundary is unknown), a hybrid at its last checkpoint at or below it (`hybrid`, set at
  load; the QSA bank is sliced onto that checkpoint), a ringed cache only where a ring checkpoint sits at or below
  it. Spec snapshots (DFlash window, MTP history) are not persisted with a cut record. Idle spill still skips media
  entries: the commit already wrote their text.
- **"Free disk" is what the OS will GRANT** (`sushi_volume_free_for_use`, statfs fallback): purgeable space is released
  on demand. The `volumeSpace` test must not race the OS's purgeable answer.

<a id="spec-state"></a>
## Spec state rides the cache

`Entry.mtp` + `restoreSpecSnap`, adopt only on `base + step == matched`; MTP trims to `mtpCommittedLen`; survives the
SSD tier (`spec.safetensors`). An adopted spec cache has ONE owner at a time (`runPrefill` clears its locals BEFORE
`initWithOptions`).

<a id="glm"></a>
## GLM-5.3 (`glm5_next`)

GLM's state lives in its slot's `glm5_forward.Request`, not a `KVCache`: 34 FP32 KDA states, 11 MLA latents and a
pooled index (`src/glm5_prefix.zig`; [arch-glm5-next](arch-glm5-next.md)).

- **A restore point is a pool boundary** (a multiple of 4), where the IndexPool tail is empty. The state there is:
  - the KDA conv and recurrence of every linear layer, an `SSMCheckpoint` of 147,619,840 bytes;
  - latent rows [0,P) and pooled rows [0,P/4), a prefix of the entry's `MlaRows`.
  One `MlaRows` (every row through the entry's newest checkpoint) serves all of the entry's checkpoints.
- **Checkpoints sit on a 2048-token grid and at the prompt end.**
  - The grid is absolute multiples of GLM's widest chunk (`glm_checkpoint_stride`), never a request's width.
  - A narrower chunk divides 2048, and `nextChunkEnd` ends a chunk on every grid point, so every grid point is a
    real chunk boundary whatever widths a request steps through.
  - The prompt-end backoff grows to 30-33 so its position is a pool boundary (`glmSnapshotBackoff`).
  - At most 8 per entry (`glm5_prefix.checkpoint_cap`), thinned span-preserving with a dense newest quarter.
- **A restore on the grid is bit-identical to cold when both run the same chunk widths from that point on.** That
  holds at the 2048 default. The suffix runs the same absolute chunks, tail merge, backoff and final span.
  - A request stepped down to narrower tail chunks near 1M matches only a cold run that steps down the same way.
  - Guards: the generator fixture tests (including a mid-request step-down), the hot-cache fixture tests, and
    `tests/test_glm_prefix_reuse.sh`.
- **A restore takes the nearest checkpoint at or below the match** (owner decision), usually the previous prompt's
  end. Off the grid, only the chunk around the checkpoint runs in a different shape, and the suffix rejoins the cold
  grid at the next boundary.
- **Measured bound** (Sushi-2.3bpw, kv8, `/v1/completions`, greedy, 256 tokens, top-5 logprobs, against the same
  prompt cold; this change on b8267038, `taskpolicy -a`, lock per boot, 2026-10-05):
  - Appended prompt restored at the previous prompt-end checkpoint:
    - First-token |Δ logprob| was 0.06, 0.07 and 0.02 nats at 8.6K, 32K and 60K tokens; a second 8.6K run gave 0.25.
    - While greedy agrees, the chosen token moves at most 0.56 nats.
    - Greedy flips at near-ties: at token 0 at 32K and token 6 at 60K. At 8.6K it held for all 256 tokens.
  - Restore on the grid, at 4,096, 28,672 and 55,296 tokens: every token and every top-5 logprob equal to cold.
- **TTFT, same runs:**
  - Appended prompt: 0.57, 0.41 and 0.43 s warm against 11.1, 42.4 and 79.6 s cold.
  - Grid restore with up to 2K tokens to prefill: 2.1, 1.6 and 3.3 s against 6.2, 37.5 and 75.8 s.
  - Decode is unchanged, 29.9 to 30.2 tok/s.
  - SSD-only (`--no-prefix-cache-ram --prefix-cache-disk 12GB`), turns growing from 32K to 42K tokens: 6.5 to 7.6 s
    warm against 33.9 s cold at 28K. A restart restored 41,720 tokens from disk in 115 ms.
- **Sushi-2.5bpw + vision + A4, cache on at its defaults** (this change on fbdedc01, kv8, `--prefix-cache-entries 1`
  so the cold arms follow an eviction, `taskpolicy -a`, lock held, 2026-10-05):
  - It advertises 1,048,576 and admits every request at 2048-row chunks.
  - An 8.6K appended prompt reuses 8,552 tokens and prefills in 0.60 s against 13.0 s cold. First-token |Δ logprob|
    is 0.10 nats; greedy flips at token 1.
  - A grid restore at 4,096 of a 5.5K prompt prefills in 2.4 s against 7.5 s, every token and top-5 logprob equal
    to cold.
- **Checkpoint state is copied bit for bit** (`bitsOwnedCopy`: an integer view plus an integer zero).
  `materializedOwnedCopy` adds a float zero, which turns -0.0 into +0.0, and the restored KDA state carried that
  into the next chunk.
- **The unnamed RAM tier is 1 GiB** (`GLM_PREFIX_CACHE_MEM_DEFAULT`; `--prefix-cache-mem` overrides). The
  advertised context reserves nothing for it: admission evicts it to admit a long prefill.
  - At kv8 it keeps a 30K session at its prompt end with 5 of 8 checkpoints, a 60K one with 4, and trims a 140K one
    to its checkpoint near 121K with 1. Each case keeps the assistant window.
  - For long reuse add `--prefix-cache-disk`, or run SSD-only (`--no-prefix-cache-ram --prefix-cache-disk 12GB`).
- **Rows are a real copy at commit**, through the newest checkpoint the destination keeps
  (`HotPrefixCache.glmCommitLen`, chosen before the copy; a restore resumes from a checkpoint, so later rows are never
  read). The RAM tier keeps rows, checkpoints and window within its budget, SSD-only one flush (2 GiB), and the
  checkpoints above the chosen row are freed first, so there is no full copy followed by a trim. A share would keep
  the request's reservation (up to the whole context) alive while billing only the rows.
  - Billed in `kv_bytes` beside the checkpoints: 6,688 bytes per row at kv8, 11,968 at BF16.
  - A budget trim lands on a checkpoint (`MlaRows.trimmedCopy`), sheds interior checkpoints and keeps the window.
- **A restore shares the rows; the first append copies them.** A checkout releases them, so that append donates.
- **The assistant window rides the entry as prefill left it** (`Generator.glm_prefill_window`). Every GLM restore
  point is at or below the prompt end, and a window cropped at the reply's end misses it once the reply passes
  2,016 tokens.
- **SSD tier**:
  - Rows go in the usual chunk files as a dense pseudo-cache of two entries per layer (`glm5_prefix.diskEntries`).
    kv8 is keyed `{off, 8, 64}`, which the manifest keeps.
  - KDA checkpoints go in `s{pos}` files, at most 8 per entry, the window in the spec sidecar.
  - `spec.safetensors` is replaced in place before the manifest commits and equal windows share a size, so it carries a
    `d.pos`/`m.pos` = `base:step` stamp; a load that finds it absent or different declines the spec (trunk restores).
  - A restore reads only its own checkpoint file (`DiskTier.restoreIntoKda`); the QSA check that rereads the
    newest one is Qwen's.
  - GLM is never SSD-first while RAM retention is on. Under SSD-only storage it is: the commit captures the rows,
    checkpoints and window into the pending flush, keeps no idle RAM entry, and the background writer persists them
    after the response. There is no prefill write-through; a flush is bounded by the 2 GB readback, and a later
    turn's commit extends a partial entry.
- **A decode-phase cancel commits in `cullDecoding`**, before `releaseNativeState` resets the request that the
  cleanup drain's commit would otherwise read. The drop decision is taken once per slot under `queue_mu`; the commit
  and the release run outside it on the dropped slots, so a late cancel waits for the next tick.
- **`commitImpl` owns the transferred checkpoints on every outcome**, including a failure of the retention snapshot.
- **One schedule drives the capture and its bill** (`generate.glmCaptureSchedule`: the grid points in the tail, the
  prompt-end checkpoint, pool alignment, cold/warm backoff, the cap). The configured stride never enters, and
  `glmChunkEnd` keeps the tail merge from absorbing a grid point, so the billed count is the captured count.
- **Bills.** A GLM request holds up to 9 checkpoints during prefill (the cap plus the copy taken before each thin)
  and one assistant window. The commit moment is billed beside the live cache (`glmCommitStateBytes`): the row copy
  (at most the RAM budget, or one SSD flush), the checkpoints and the window, whichever of that and the prefill's
  transient is larger. At 1M tokens it is the smaller, so it costs no checkpoints. SSD-only adds the writer's 1 GiB permit (the previous request's staged flush) and
  reserves no idle cache.
  - Only the inference thread's admission pass bills the checkpoints (`WarmPrefix.checkpoints`). The connection
    thread, the context sizer and the cache clamp bill none, so the advertised context is the cache-off one.
  - The pass evicts RAM entries LRU first, sparing the one it restored from. If the request still does not fit, it
    keeps fewer checkpoints, down to the prompt end and then none, rather than be refused
    (`scheduler.fewerCheckpointsToAdmit`). With none it commits nothing.
  - The checkpoints yield to the width: the prefill width is chosen as if the request kept none.

## Guards

`tests/test_prefix_cache_*.sh` (budget revisit, disk, hot, mem, workloads), `tests/test_hybrid_reuse_equivalence.sh`,
`tests/test_mimo_ring_reuse.sh`, `tests/test_mimo_ring_fork_ssd.sh`, `tests/test_qwen4_mtp_head_persist.sh`,
`tests/test_glm_prefix_reuse.sh`. Grep the log for `[cache]`, `[hot-cache]`, `[disk-cache]`.
