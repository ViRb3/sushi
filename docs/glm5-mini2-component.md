# Fixed mini2 proposal component

The fixed complete-proposal component passed exactness and won all eleven pairs;
actual model acceptance is pending. The null red stub failed at the intended check; the single frozen green body
is now applied, passed its focused build/proof gate. No shared source/caller or
public test-runner edit was made. The [round plan](glm5-route6-mini2-round-plan.md)
fixes N2, horizon2, A3/gs64, shortlist32 and selector top16.

`projectTwo(source, coarse, hidden, stream)` returns optional caller-owned
BF16 `[1,2,154880]` logits. The narrow `stages(ops, ...)` probe API returns
borrowed caller-Ops handles for coarse logits, two ordered ID sets, both M2
re-score planes and the masked logits. No waits, source writes or new kernel
are introduced. Coarse preparation/lifetime reuse the existing mini consumer.

Source geometry is only full normalized BF16 `[1,8,4096]` assistant blocks,
original A6/gs128 head and separate A3/gs64 copy. Both draft positions use the
same M2 re-score geometry as current full2. The unchanged full eight-row
assistant and target verification are outside the readout component. Current
full2 is control, with coarse held equally; old mini7 is never its baseline.

The verified capture dependency supplies six real blocks from frozen ordinary/
predictable prompts, three rounds each. Hidden is already final-normalized;
anchor and actual committed prefix are scalar U32. No synthetic input is called
real retention. Head/selector source formats were inspected from actual headers;
selector codebooks and hidden-projection tensors retain original BF16 storage.

The probe independently creates the expected masked BF16 vocabulary plane from
current full2 logits and selected IDs on CPU. It checks every selected source
logit across both M2 query rows, the complete mask, independent top32 ranks/uniqueness/cutoff ties,
positive transforms and exact lattice/proposal result on the same masked oracle.
Actual full-head top1 and original selector top16 retention are diagnostics;
proposal differences from full2 are reported, not treated as numerical equality.
The CPU check accepts any valid cutoff-tied set; actual GPU ID order is retained
unchanged for masks/re-scoring/proposals. Zero/tie and genuine selected-NaN guards are synthetic and labeled separately.
F32/short-block/incorrect-head geometry declines before candidate graphs.

One ordinary-round0 readout plus complete transforms/selector/lattice/N2 tree
gets three warmups and eleven fresh alternating pairs, including endpoint reads
and all frees. All six real blocks are proven first. Coarse cold preparation
and release are measured separately; its 277544960 resident bytes and temporary
peak are explicit. Existing head/cache/scratch bills receive no credit. A clear
winner alone gets root's current HC_ON/A6/native ordinary+predictable matched
model gates; no head rewrite, default change, variant or acceptance claim yet.

Private evidence key `glm53-mini2-component-20261004` holds the stub, fixed green,
probe/source hashes and exact focused ReleaseFast recipes. Activation provenance
belongs to `glm53-mini2-current-activation-20261004`; the capture is not run by
this helper worker. No model load is needed for the component.

Activation SHA256 is
`4c6991cb9ab4926dbf78ee021ca9fa221eab76f44377bb5d7833eb5ee4332ad3`;
393216 hidden bytes, with six shape/dtype/anchor/prefix contracts checked from
the actual file. Capture was explicitly perturbed; it is not a timing run.

Behavioral red compiled ReleaseFast and failed at `ExpectedMiniHorizon2` as
intended; geometry passed. Final red build PID 95277 and run PID 95665 ended;
lock released/fans automatic. Pre-proof JSON preserves cold coarse preparation
65.117042 ms and 277544960 resident bytes, with all six real blocks loaded.
That red run performed no target-model load, numerical green proof or timing.

## Complete green result

The fixed green passed both focused tests, exit 0. One compile packaging correction
added `try` when lifting the owned Arr return into an optional error union; graph,
geometry, arithmetic and scheduling were unchanged. The original compile failure
and source are retained separately.

| Exact/retention evidence | Result |
| --- | ---: |
| Selected original A6 BF16 logits across both M2 query rows | 1024 exact |
| Complete masked BF16 vocabulary values | 2478080 exact |
| Real draft rows retaining original top1 | 12/12 |
| Original selector top16 candidates retained | 192/192 |
| Changed proposals on six real blocks | 0 |

Counts include separately labeled synthetic zero/tie and selected-NaN guards;
retention refers only to the twelve real ordinary/predictable draft rows.
GPU-selected IDs retain their actual order. The independent CPU rank/cutoff check
accepts any valid tied set without demanding a stronger stable-ID contract.
Masked-oracle proposals and positive transforms also matched exactly. This small
corpus does not establish general acceptance or target quality.

Three warmups preceded eleven alternating complete-proposal pairs on actual
ordinary round0. Current full2 median was 1.156041 ms versus mini2 0.853917 ms,
26.1344% lower. Every pair won; paired median reduction was 26.6979%, range
23.7943–27.1921%. All raw pairs are preserved in the artifact. The clock includes
coarse projection, top32, original-row gathers, both M2 re-scores, masks,
transforms, selector/lattice/N2 tree, endpoint reads, settlement and all frees.
It excludes unchanged assistant forward and target verification; it is not a
26% decode-throughput gain or a comparison against old mini7.

Cold coarse preparation was 61.975084 ms, separate release 0.000750 ms. The coarse
head remains 277544960 resident bytes, held equally in both controls. Candidate
peak above those equal held source/selector/activation/coarse inputs was 7133720
bytes. Original head/cache/scratch bills remain; startup/retained-head cost and
actual model acceptance still decide adoption. No default or checkpoint changed.

Final build PID 97554/run PID 97860 ended, lock released and fans automatic.
Foreground QoS and confirmed maximum fans/quiet idle were recorded. Exact
source/binary/library/activation hashes, cold JSON, complete proofs, all pairs,
peak and telemetry remain private under `glm53-mini2-component-20261004`.
No caller hook, target-model job, bit/group/shortlist variant or retest followed
this component. Only a clear current-full2 model gate can support acceptance.

After the numerical gate, a small scoped policy test and API were added without
head-math changes: `Mode {full2, mini2}`, `bind/restore`, `currentMode` and
`dispatchCount/resetDispatchCount`. Null mode preserves the legacy caller;
`SUSHI_GLM_DFLASH_MINI2` may select the new mode when unbound. Root owns caller
integration and the coarse object's lifetime. Both explicit modes bypass the
legacy mini7 path: mini2 applies only to bounded N2/block8; partial N1 retains
current full8 source readout/seven draft rows in both modes. There is no horizon1
optimization. The same coarse object remains resident in full2 controls.
These policy additions have separate hashes and await model validation; no
caller or default change was made by this helper worker.
