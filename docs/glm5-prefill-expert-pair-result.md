# Expert-only pair: functional proof passed

The fixed expert-pair helper passed its directed real-bank proof. Six retained
F16 stages (603979776 values) and both BF16 outputs (8388608 values each) matched
two separate original 2048 calls exactly. Route metadata and owned-view survival
also passed. No timing or model run occurred; the independent complete-layer
gate remains required by the [round plan](glm5-expert-pair-tail-round-plan.md).

`glm5_prefill_expert_pair.run` takes two original BF16 `[1,2048,4096]` inputs,
U32 `[1,2048,8]` IDs and FP32 scores plus the original bank/decoder. It joins
only those three tensors, calls the fixed current grid and returns two owned
BF16 views. Caller policy is default off; directed `run` is independent of
that policy. The caller retains each output before `Result.deinit`. There are
no new waits, compact copies, nonexpert operations or routing arithmetic.

The grid admits only 2048/4096 and keys its four bounded configurations by
K/N/route rows. The archived physical-grid shader, WIN32, MCG/W12/n36,
K16/native MMA and F16/BF16 boundaries remain unchanged. Temporary
`projectForProbe` exposes the same projection body for private stage checks.
The helper's checked join bill is 67371008 bytes per pending layer,
134742016 at pending2; root owns the separate metadata and full lifetime bills.

## Workload and scope

Half A is the original actual L20 T2048 routed fixture, with all three original
full E288 banks. Half B is distinct constructed data: row `(i*127+13)%2048`,
BF16 activation multiplied by -0.75, and original IDs/scores permuted in the
same order. All 16384 second-half IDs and FP32 score bit patterns matched that
declared permutation; IDs were valid and unique within each token's eight slots.
These are not two actual adjacent captured chunks or newly computed router
decisions. No duplicate-input benchmark was relabeled as fresh evidence.

The proof inverse-routed prepared gate/up, gate/up, middle and down values into
original slot order before comparing. It independently checked sorted IDs,
order/inverse mappings, expert segments, starts, lives and unused zero windows.
Production helper outputs matched the private reconstruction and separate
controls. Both views remained exact after retaining them with `Ops.result`
and freeing the original `Result` before evaluation. Unsupported partial
halves, decoder and clamp declined without incrementing the helper counter.

| Scope | Assignments | Segments | Live windows | Padded rows | Capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joined pair | 32768 | 265 | 1174 | 4800 | 1312 |
| Half A | 16384 | 265 | 671 | 5088 | 800 |
| Constructed half B | 16384 | 265 | 671 | 5088 | 800 |

The route multiset is preserved by the construction, explaining equal half
counts. Window counts are logical metadata, not measured traffic or latency.
The private `provePair` entry can check newly generated complete-layer routes
without duplicating this stage implementation.

Peak active memory was 8342113030 bytes; the resident fixture/input baseline
was 2099650566, yielding an increment of 6242462464. This directed proof holds
multiple full reference/stage graphs; it is not a runtime or model memory bill.

## Provenance

Artifact `glm53-prefill-expert-pair-20261004` retains original fixture/bank
identity, exact source snapshots, commands, runtime/binary hashes, all proof
counts and telemetry. Source checkpoint was `35666a7a` plus hashed WIP.
Helper SHA256 starts `d5e27ce2af27b647`, unchanged grid `60a88f99e609eb4b`,
private proof `3404376b9ca754e0`, green binary `96b98bc2fede5c9c`.

Red compiler 41870 exited 0; proof 42021 returned the genuine `MissingExpertPair`
failure before green. Green compiler 42898 and functional proof 43059 exited 0,
both filtered tests passed and all eight recorded compiled sources remained
hash-identical. ReleaseFast, foreground `taskpolicy -a`, accepted libraries,
confirmed maximum fans/idle and the exclusive per-job lock were used. The
process ended, lock released and fans returned automatic. No speed, numerical
quality, complete-layer state or actual-model acceptance follows this proof.
