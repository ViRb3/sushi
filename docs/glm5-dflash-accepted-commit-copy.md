# DFlash accepted target append ownership

The remaining accepted-target MLA append uses a functional copy of the reserved
latent buffer. No in-place mutation was added: the current transaction requires
the original target request to remain usable until the assistant update succeeds.

## Phase attribution

`roundTreeLayerwiseConfigured` charges `Verified.prepareCommit` to `replay_ns` and
`commitVerified` to `commit_ns` ([source](../src/glm5_dflash.zig)). The former clones
the target request, replays accepted KDA state when needed, appends accepted MLA
rows, and evaluates the resulting arrays. The latter clones and updates the
assistant context, evaluates it, then publishes both prepared states.

The qualified 16K/32K round means were therefore:

| phase | 16K | 32K | includes |
|---|---:|---:|---|
| replay | 1.58 ms | 2.23 ms | accepted target cache writes and KDA replay |
| commit | 3.05 ms | 5.59 ms | assistant context clone/append/evaluation |

The 5.59 ms value is not a measurement of target latent append cost. Even removing
all target preparation would save at most its measured 2.23 ms in this run.

## Why ordinary donation does not apply

[`prepareCommit`](../src/glm5_dflash_model.zig) clones the committed request before
normal `MlaTape.append`. [`cloneRequest`](../src/glm5_dflash.zig) shares each latent
array's storage. MLX `SliceUpdate::eval_gpu` first calls `copy_gpu` on the whole old
buffer, then copies the small update into the output
([backend](../lib/mlx-src/mlx/backend/metal/indexing.cpp)). A vector copy can reuse
storage only if the input is donatable
([copy contract](../lib/mlx-src/mlx/backend/common/copy.h)); the still-live committed
request prevents exclusive ownership. Reserved capacity avoids growth allocation,
but does not by itself make the existing buffer donatable.

`commitVerified` performs fallible assistant construction and evaluation before
destroying the original request or context. Mutating spare rows in the shared
latent allocation would violate MLX's immutable input/alias contract and would
change the retained request's allocated bytes on failure. An exact consuming
ownership implementation would need an explicit transaction/API change, with
rollback storage or a different publication order. That change was not justified
for this bounded phase cost. Copying only a compact prefix would save reserved
padding (33024 versus 32768 rows in the measured case), less than 1%, while still
copying the valid prefix.

The verification-only latent overlays already avoid branch prefix replacements;
normal accepted append and its rollback behavior remain unchanged. This is a
source audit, with no new performance measurement or precision change.
