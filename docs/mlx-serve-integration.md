# mlx-serve integration

mlx-serve serves Sushi packs' EXL3 routed experts by importing Sushi's `sushi_exl3` module
(`src/exl3`). This file is the contract between the two repos: what mlx-serve calls, how it pins
Sushi, and how a new engine reaches it. The module itself is described in
[engine-exl3-experts](engine-exl3-experts.md).

## How mlx-serve consumes the module

- **Submodule.** mlx-serve carries Sushi as `lib/sushi` (`.gitmodules`: url
  `https://github.com/ddalcu/sushi`, a fork of this repo, branch `exl3-module`). The committed
  gitlink is the pin: a full commit, nothing floats.
- **Build.** mlx-serve's `build.zig` (`addExl3Module`) builds `lib/sushi/src/exl3/root.zig` as the
  module `sushi_exl3`, with one import, `mlx_host`, pointing at mlx-serve's own host root.
  `-Dsushi-dir=/abs/path` builds against a Sushi checkout instead of the submodule.
- **Use.** `ModelConfig.exl3: ?sushi_exl3.Spec` marks a pack whose routed experts are EXL3
  (`expert_quant` in `config.json`); every other module stays the host's own affine path.

## The contract

The module reaches `mlx`, `log` and `io_util` only through `mlx_host`, whose root must expose them
as `pub const`. It never imports a Sushi file by path, so it builds inside any host that provides
those three.

mlx-serve calls exactly these names. Renaming one, or changing its signature or meaning, breaks
mlx-serve's build on its next bump; add new names instead, and say so in the CHANGELOG.

| name | used for |
|---|---|
| `Spec`, `parseExpertQuant` | reading `expert_quant` from `config.json` |
| `admitTopK`, `trellisAdmitted` | refusing a pack the kernels cannot serve, by name, at load |
| `Bank` (`Proj`) | one layer's `trellis`/`suh`/`svh` banks |
| `moe` | the routed SwiGLU dispatch (decode chain or prefill GEMM) |
| `format.Decode`, `format.Codebook.mcg` | the decode spec passed to `moe` |
| `kernels.DECODE_ROWS_MAX`, `kernels.usesPrefillArm`, `kernels.rowsOfShape` | the host's own row planning around `moe` |

The module's tests run as their own artifact (`exl3-test`) on `zig build test`; a change that
passes them and keeps the names above is safe to hand over.

## MLX pins

Each repo builds its own MLX and mlx-c; the module adds no pin of its own. Both repos pin the
same commits: MLX d73eb752 (v0.32.2 plus the sorted `gather_qmm` NAX 32K-row fix,
ml-explore/mlx#3922) and mlx-c 56b2d39. Move both pins together. MLX v0.32.3 needs a newer
mlx-c: its `gather_qmm` gained a `global_scale` argument that mlx-c 56b2d39 does not pass.

## Handing a new engine to mlx-serve

1. Land the change on Sushi `main` with `zig build test` green, `exl3-test` included.
2. Make the commit reachable from the submodule's url (`ddalcu/sushi`): the fork must fetch it
   from this repo, or the change must go through it.
3. Give mlx-serve the full commit hash. mlx-serve bumps the gitlink
   (`git -C lib/sushi fetch && git -C lib/sushi checkout <sha>`, then commits `lib/sushi`) and runs
   its own tests.

The gitlink is the whole handoff: no release asset or separate manifest is needed.
