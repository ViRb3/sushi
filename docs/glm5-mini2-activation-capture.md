# Current full2 assistant activation fixture

One bounded capture produced six real complete eight-row assistant blocks for
the horizon2 mini-head consumer proof. This is a **perturbed shadow-forward
fixture**, without duplicate-hidden parity, retention, target-quality or speed
claims. No coarse head was loaded and no stored weights or caches were dumped.

The existing frozen ordinary/predictable inputs contained 8756/8720 IDs. One
target and one A6g128 assistant load used current full2/N2/children4, native
B1/B3, HC collapse on and packed32 on; old mini and route6 stayed off. Before
actual rounds 0/1/2, an exact committed assistant-context snapshot supplied the
pending anchor plus seven mask tokens to the unchanged trained eight-row
forward. Normal current full2 verification/commit then advanced the request;
anchors and branches were not forced.

Returned hidden was BF16 `[1,8,4096]`, already final-RMS normalized. Consumers
must not normalize it again. Fresh host-data leaves severed all forward/cache
graph ownership before retention in the fixture. Original target head format
was A6/group128; selector top16, output multiplier 1 and softcap 0 stayed fixed.

| Prompt | Actual round | Anchor ID | Committed target offset | Accepted drafts |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 0 | 73022 | 8756 | 2 |
| Ordinary | 1 | 322 | 8759 | 2 |
| Ordinary | 2 | 14 | 8762 | 1 |
| Predictable | 0 | 16 | 8720 | 2 |
| Predictable | 1 | 198 | 8723 | 2 |
| Predictable | 2 | 19 | 8726 | 2 |

The fixture includes `ordinary.round0.hidden/.anchor/.prefix` through round 2
and matching `predictable` keys. Anchor/prefix are U32 scalars; prefix is the
actual global committed target offset. Metadata also preserves assistant
absolute length, base position and cached rows, including the real cropped
2050-row windows after the first commits. Complete bodies/input hashes were
inherited from the frozen corpus; no input was retuned.

Six hidden payloads total 393216 bytes, plus 48 scalar bytes. Safetensors file
size was 394851 bytes; capture manifest, flags and provenance bring the required
payload/metadata total to 401823 bytes, below 1048576. Shapes/dtypes, natural
anchor/context alignment and frozen-count/hash protocol passed. Both memory
and wired limits stayed 115448725504 bytes; the unchanged conservative ledger
with one assistant was 11383406592 bytes above active memory and all admission
checks passed without reserve credits.

Artifact `glm53-mini2-current-activation-20261004` retains safetensors,
manifest/flags/source/binary hashes, commands, payload bound and telemetry.
Tensor file SHA256:
`4c6991cb9ab4926dbf78ee021ca9fa221eab76f44377bb5d7833eb5ee4332ad3`.
Binary SHA256:
`8868055270890783a538f62070e9bd8cb87088cc1c6a453d3fc61719f43c926f`.
Runtime was accepted `4fcb541e` with recorded source hashes. Focused ReleaseFast
build and CPU frozen-input protocol passed; the capture passed all 8 tests,
exit 0. Foreground QoS, explicit accepted-library paths, maximum fans/idle and
an exclusive per-job lock were used. Process ended, lock released and fans auto.
The two private capture sources were hash-verified, archived and removed;
prepared quality sources and runtime/CLI were untouched. Component proof and
matched model gates remain separate decisions.
