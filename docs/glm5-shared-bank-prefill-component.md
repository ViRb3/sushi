# Shared-bank prefill component

Uncommitted candidate; component passed, fixed quality/model gates pending. The policy follows
[the round plan](glm5-short-index-shared-bank-round-plan.md) and its unchanged
quality bounds. The candidate delegation is default off while qualification
continues; no default or accepted runtime change has landed.

`glm5_attention_shared_bank` processes exactly64 real queries: sixteen
absolute-aligned groups of four. Each first query selects512 completed pools
with original prefill NAX score arithmetic and partition order. One contiguous
BF16 bank per group contains2048 historical rows plus four unique local rows.
Invalid historical rows are zeroed before access. The native safety helper
owns per-query future-local zero-before-load; ordinary masks alone are unsafe.

`glm5_indexpool_nax.scoresAtPositions` changes only completed-pool eligibility
to explicit logical positions. The original `scores`/`tryScores` entry points
keep their original consecutive-position shader, geometry and mode checks.
The candidate admits the current NAX history band and full64-row aligned
chunks only. Unsupported histories, mixed precision, noncontiguous latent
storage and fragments decline to the existing attention path.

The focused probe uses the existing actual16K T2048 planes. It checks original
consecutive API parity, every anchor score/ordered pool ID, gathered BF16 rows,
validity and uniqueness, and all approximate-policy output bits against
independent per-query gathered native attention. Independent per-query
retrieval recall and missed positive index-score mass are diagnostics; they
do not establish forced-logit quality. One inclusive three-warmup/eleven-pair
whole-attention comparison retains at most two graphs, with conservative
64MiB per graph and128MiB pair bounds. It includes selection metadata,
gather/masks, native attention, collection, endpoint evaluation and all frees.

## Actual16K component result

At `2e7786a2` plus the isolated candidate, the original scorer failed the
explicit-position probe's compile-red recipe because `scoresAtPositions`
was absent. The extended source passed all three focused tests. Original
consecutive scores, all512 anchor selections and ordered pool IDs per block,
all gathered BF16 values/validity/unique IDs, and all67108864 BF16 attention
outputs matched their references. The attention reference gathers independent
per-query banks with the **same approximate IDs**, zeroing future rows before
native Q64 attention; it does not claim original independent retrieval parity.
Misaligned offsets and incomplete64-row batches declined. The safety helper's
separate nonfinite proof is recorded by its owner.

| Inclusive T2048 attention, actual16K planes | Median |
|---|---:|
| Current independent retrieval, packed two-graph cadence | 130.206875 ms |
| Four-query anchor retrieval, shared-bank two-graph cadence | 60.845875 ms |

Eleven alternating pairs all favored the candidate; arm medians and median
paired ratio both imply53.27% less time. Three warmups preceded the samples.
This includes every selector/position array, partition, gather/mask, native
attention, collection, endpoint evaluation and free. No isolated load or
tensor-group speed claim is made. Whole-call measured extra peak was268435456
bytes above the resident fixture, including retained tile outputs/final
collection, within the asserted conservative pair-plus-output bound. This is
not a measured64MiB per-graph peak or permission to reduce admission.

Retrieval is materially different. All1536 nonanchor rows recovered549785 of
786432 independently selected pools:69.9088% average recall, with16.40625%
worst-row recall. The missed positive index-score fraction was21.8931%,
defined as the sum of `max(score,0)` for missing independent selections divided
by that sum over all independent selections. No row had zero denominator.
Completed group-local pools count as recovered because their four raw rows
are present. These diagnostics are neither forced-logit KL nor task quality;
the round plan's fixed long-prefix quality bounds remain unchanged and decisive.

Measurement key `glm53-shared-bank-prefill-20261003`, 2026-10-03: ReleaseFast,
foreground `taskpolicy -a`, exclusive per-job GPU lock, confirmed5348/5752 RPM
fans at43.65°C and ten seconds idle. Private artifacts retain original actual
fixture key, source/binary hashes, compile-red/green logs, complete paired
arrays, diagnostics and thermal telemetry. GPU was released promptly and fans
returned automatic. No full model or quality evaluation has run. Coordinator
owns quality/model gates, admission and eventual opt-in delegation.

## Default-off production seam proof

The coordinator's packed-cadence delegation tries64 shared rows and uses the
original16-row path on decline/fragments. A separate filtered fixture test
passed without repeating the paired benchmark: all67108864 T2048 BF16 values
matched the direct whole helper, all3178496 T97 values matched an explicit
shared64 plus original33 oracle, misalignedT97 matched shared-off attention,
and default-off T2048 remained bit-identical. Counters were exactly32/1/0/0
respectively. Packed attention, index scoring and cadence were enabled, with
shared mode scoped on/off. Source remains uncommitted pending fixed quality
and model acceptance.

Evidence key `glm53-shared-bank-prefill-seam-20261003`: one focused test plus
its root passed, ReleaseFast/foreground QoS, exclusive GPU lock, confirmed
5350/5746 RPM at41.96°C and ten seconds idle. The lock was released and fans
restored automatic. Private source/binary stamps and complete assertions are
retained. This closes component-to-caller parity only; no quality result or
new performance number is attributed to this run.


## Final disposition

The [fixed actual-model screen](glm5-shared-bank-quality-result.md) rejected
the policy on both16K inputs. Component math/safety and speed did not establish
acceptable model quality. All candidate source/delegation was archived and
removed, with accepted runtime unchanged; no default or performance claim landed.
