# Near-full prefill and batched decode round

Start from the last accepted runtime `f9f4c8d2`, with measurement/report
checkpoint `71ed6e16`. Its predictable 2K/4K/8K/16K prefill rates are
931.71/802.82/727.81/655.06 tok/s; selected 32K has 33595 actual IDs and
604.18 prefill/42.39 decode. The 1500/60 goals remain open. The previous three
prototypes were removed after failing their component or real-model gates.

Two completed researcher reports define this round:
[near-full prefill](glm5-round-nearfull-prefill-research.md), `f505564f`, and
[batched decode attention](glm5-batched-decode-attention-research.md), `576acabf`.
They cover existing measured costs, context growth, speculative consistency and
bounded scheduling/memory. No implementation was measured during research.

| Worker | Candidate | Owned implementation |
|---|---|---|
|1|A6 dequant/native NAX for 1536–2048 actual rows|Existing A6 helper and isolated partial-row probe|
|2|Exact expert-grid transpose for 1536–2048 rows|Existing grid helper/probe and bounded config replacement|
|3|True B3 native decode attention|New overlay-aware gather/SDPA helper/probe; component first|

Worker1 retains the existing two bank geometries, original quantization and
opt-in policy. No padding or shared/dense MLP extension. The existing 512 MiB
async2 premium must cover any configured chunk that can produce an eligible
remainder. Prove 2037 outputs and row boundaries, including permitted native NAX
rounding evidence. Actual model evaluation is the same 2037-ID prompt in ABBA
with profiling off, all other accepted flags identical and 136 observed A6 calls
in candidate arms. Do not compare unlike synchronized profile and HTTP rates.

Worker2 derives T and S=8*T in every original-chain stage. Keep n36/MCG/W12,
WIN32, clamp 10 and original slot/reduction order. Maintain only the two existing
projection config geometries and rebuild when S changes. Prove an old lazy
graph survives config replacement before evaluation. One real L20 partial-row
complete-chain proof/timing advances to the 2037-ID real-model ABBA. No weight
copy, output padding or new allocation bill is introduced.

Worker3 uses the existing real 16K fixture and explicit chain/fork suffixes,
original ordered selections and offsets. Native B1/B3 retain rank 4 and identical
BQ32/BK32/D512 kernels; B3 has only six tensor groups, so gain is unmeasured.
Prove raw B3 results equal three native B1 results and record drift versus the
current scalar path. Time three complete branches with one endpoint settle in
both arms, including selection/gather/SDPA/output/free. The old losing experiment
evaluated three B1 calls separately. Bound 8 MiB per pending layer, 32 MiB at async4;
never materialize a full prefix. If the component wins, an opt-in target mode
must use matching native B1 for serial/replay and fallback before speculative
integration. Preserve BF16 compressed cache and FP32 accumulators/state.
No precision restoration or N3/T4 expansion belongs in this candidate.

Builds and GPU runs are scheduled one at a time for quiet measurements. Each
worker stops a losing arm after one focused proof and paired timing; no variant
sweep. Implementation source stays WIP with hashes until its real-model result
is accepted. The coordinator owns production integration, admission, counters,
quality/state gates and pushes. Commit only accepted runtime changes; preserve
rejection evidence privately and document the lesson. Then two researchers
start the next round. Routine HTTP iterations remain 2K–16K, with one selected
32K qualification. Original small tensors and resident embeddings remain intact.

## Outcome

Both near-full prefill candidates were rejected after their real-model gates.
The exact expert-grid extension was 1.24% slower on the fixed 2037-ID ABBA;
the A6 extension was 3.115% slower in its equal-memory ABBA. Their component
wins did not carry through to the model. Production helpers remain restricted
to the accepted 2048-row geometry; documentation commits `d24b440c` and
`39f8693d` retain the rejection evidence.

True B3 native decode passed its component and coupled token/state proof.
Its warmed 2K–16K HTTP run is mixed at short contexts and faster at 8K/16K.
It remains default-off after the selected 32K qualification passed: 43.00
predictable and 38.08 ordinary tok/s, both using matched input counts.
[Native decode qualification](glm5-decode-attention-batch.md) records numerical
drift against the old scalar target separately from exact mode-matched
serial/speculative equality. No precision restoration is used.

A user-supplied A4/gs64 assistant study prompted a separate stored-format
consumer extension and matched A6/A4 test. Both precisions matched the target's
192 serial tokens and valid final state on fixed 1176-ID ordinary and 1140-ID
predictable inputs. A4 reduced drafting time about 11%, while total decode
gains of 1.53%/0.96% were smaller than control timing drift. Retain optional
A4 support, keep A6 as default, and do not expand this into another ladder.
The next two researchers address larger prefill and verifier costs before
allocating the next three implementation workers. Goals remain unmet.
