# Fixed temporal KDA schedule: rejected

The strict T2048/H64/R4 candidate was exact, but complete-layer latency rose
from 14.527666 to 16.246584 ms: 11.83% slower, 0/11 wins. The paired median loss was
11.75%. No schedule/tile variant or model arm follows. Original runtime
behavior is retained; no precision or default change was accepted.

The helper reused the original prework and R4 bodies for sixteen 128-token
pieces. Each async pair had an explicit FP32 state dependency, then one
settlement; it retained only small Y outputs/latest state and released both
tile scopes. Original raw preceding-three-row QKV views supplied interior
convolution history. Eight waits, additional state stores/reads and final Y
concat were included. All full-T2048 QKV/raw-A/beta projections, retained cluster,
gate/post and output GEMMs remained unchanged. No new shader or weight bank
was introduced.

All four final tests passed. Prepared piece comparisons covered 201,719,808
BF16/FP32 values across nonzero, raw signed-zero and cold cases. R4 Y, final
FP32 state and convolution tail matched monolithic processing; every raw
boundary prefix matched its original view. Complete original layer-zero
output/state/tail equality passed for nonzero and cold state. Unsupported
length/head/state precision declined. The red null helper failed for missing
behavior before implementation. A first green attempt passed the raw proof
but stopped before timing because the probe tried to clone empty cold-state
handles; an authorized harness-only repair created fresh empty destinations
with `initialized=false`. No recurrence or schedule changed in that retry.

Three warmups preceded eleven interleaved fresh complete-layer samples. Both
arms used the original 23-tensor layer-zero fixture, fixed BF16 activations,
nonzero BF16 convolution/FP32 initial state and accepted dense/cluster/R4 flags.
Timing included projection/dequantization, all prework/recurrence, every tile
wait and state carry, concat, gate/post/output GEMM, endpoint evaluation and all
frees. No outliers or losing pairs were excluded, and no child-stage timer was
substituted for the complete-chain result.

Engagement was 14 temporal calls/112 pair waits, 238 prework and 238 R4 calls
(14 monolithic plus 14×16 temporal), 112 unchanged full-shape dense projections
and 28 retained-cluster calls. Candidate complete-call peak growth was
661,274,628 bytes above resident tensors/input/state. This includes the original
full projection/activation storage and does not establish net savings versus
control. The conservatively added 128 MiB two-pending-layer bill was retained
during qualification; no reservation reduction is inferred from source scopes.

Evidence key `glm53-kda-temporal-20261003`, HEAD `da1d32f0` plus hashed WIP,
retains fixture/source/binary/runtime hashes, red/green logs, the failed
cold-clone attempt, authorized repair, exact proof and all raw pairs. The final
GPU process PID 30441 exited 0. ReleaseFast, foreground `taskpolicy -a`, per-job
locks, confirmed maximum fans and required idle were used. Locks were released
between builds and after runs; fans returned automatic. Root restored its
model/admission seams, and the isolated helper/probe/private root were archived
and removed. No runtime commit or actual-model run was made.
