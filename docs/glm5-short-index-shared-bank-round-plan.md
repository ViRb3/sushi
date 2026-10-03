# Short-row index scoring and shared-bank prefill round

Start from accepted runtime `e1597cc2`, outcome checkpoint `cfb554f9`.
[Short-row scorer research](glm5-index-decode-nax-research.md) and
[shared-bank prefill research](glm5-shared-retrieval-prefill-research.md)
define two distinct candidates. Goals1500/60 remain open. No measured speed
is assigned to either candidate before its inclusive test.

| Worker | Scope | Owned source |
|---|---|---|
| 1 | Mode-matched B1/B3 NAX IndexPool scoring | New short-row scorer/helper/probe; no production attention seam |
| 2 | Four-query anchor retrieval and contiguous bank | New shared-bank helper/probe, explicit anchor-position extension to existing prefill scorer |
| 3 | Safe shared-bank native attention | New contiguous D512/Q256/K2052 wrapper and minimal pinned Metal body; future-local K/V masking and NOTICE |

Root owns ordinary/verification attention integration, admission/counters,
quality/model evaluation and final source acceptance. Worker2 and3 share one
prefill candidate and agree a narrow helper interface before editing. Keep
existing N2 and all accepted helpers unchanged except explicitly proven
backward-compatible interfaces. No N3 policy, indexed-cache loader or precision
restoration retry belongs here.

Worker1 fixes M128/K128/C2048 in B1 and B3, including suffix columns and partial
tiles. Preserve original BF16 score boundaries, sequential FP32 head sum,
logical-query eligibility, completed-pool ancestry and ordered partition input.
Prove every score/selected-ID/attention bit within this mode, then one whole
three-branch attention comparison on actual16K planes, three warmups/eleven
pairs. A losing/noisy scorer is removed without another layout/tile variant.
Winner gets strict serial/spec state, meaningful16K/64 old-scorer forced-logit
drift and one matched192-output actual-model comparison. Ordinary NAX reduction
drift is recorded under existing owner policy; no rounding restoration.

Worker2 keeps fixed absolute group4: first-query anchor512 completed pools plus
four unique local rows. Sixteen anchors span64 actual queries, with explicit
positions preserving native prefill scorer semantics. Only complete aligned
groups enter; fragments use original attention. Report recall and index-score
missed mass against independent retrieval, then inclusive wholeT2048 attention.
These are diagnostics, not an accuracy verdict.

Worker3 first proves unchanged contiguous native-body packaging at the new
geometry. Bool masks alone must not allow future local NaNs to poison earlier
queries. Prove safe zero-before-load behavior for invalid future local K/V
without sanitizing valid historical data or changing full historical tile loads.
Do not reintroduce arbitrary original-cache fragment gathers. Stop and report
if the safe path requires a broader algorithm or loses its component gate.
Keep64MiB/graph and128MiB pair admission conservative until measured peak fits.

Shared-bank quality screen is fixed before running: on nonrepeated16K code and
prose/retrieval inputs, compare the declared late-prefill samples and192
baseline-forced continuation positions against the current same-pack runtime.
For each input require meanKL≤0.01 nats, maximumKL≤0.15, top1≥95%, mean forced
NLL increase≤0.02 nats and zero new nonfinite values. These are conservative
experimental selection bounds, not a claim that the approximation meets the
checkpoint's standard lossless-teacher KLD or downstream task quality. A failed
screen rejects the candidate; do not adjust bounds after seeing results.
A passing component and screen receives one matched long-prefix prefill ABBA
with equal retained-reference memory and strict serial/spec state from the
same approximate prefix. Baseline/approximate cache equality is not expected.

Workers prepare isolated source/tests first and request CPU/GPU slots.
Schedule heavy builds and fixture/model jobs sequentially, foreground QoS,
thermal protocol and per-job GPU lock. Commit/push only accepted runtime.
RoutineHTTP2K–16K; selected32K once for a final winner. No width/group sweep.


## Outcome

Both candidates were rejected and owned runtime changes restored.
[Short-row scoring](glm5-indexpool-decode-nax.md) matched all mode bits but
slowed complete three-branch attention14.03%, losing11/11pairs.
Shared-bank prefill won its53.27% inclusive component and exact seam/safety
proofs, but [failed the fixed long-prefix quality screen](glm5-shared-bank-quality-result.md)
on both inputs. No approximation/default, native port or new bill was retained.
The runtime remains `e1597cc2`;1500/60 and stable2K–32K remain open.
