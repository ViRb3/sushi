# Exact full-history membership producer

Archived producer for the [exact-mask round](glm5-exact-mask-partition-round-plan.md).
There is no production delegation, selector change, new retrieval policy or
performance claim. Native attention/finite safety and the one inclusive
T2048 comparison belong to the separate wrapper worker.

`glm5_attention_full_history_mask.select` delegates128 original T16 selections
at T2048/N16384, keeping the original math/order and two-result settlement
bound. All returned arrays belong to the caller's Ops scope. `membership`
checks nonnegative/bounded/causal IDs, sends invalid slots to a separate dummy
column, then returns a bool[T,N] view of that backing. Valid IDs are unique
per query under the original IndexPool contract. Invalid writes cannot clear
key0. There is no cache/head copy or floating score plane.

The actual16K fixture producer proof passed all three focused tests at
`07f020bf` plus isolated source:

- All4200448 selected int32 IDs/order matched independent original calls.
- All33554432 membership bits matched CPU truth;4197376 live IDs were unique
  and causal, including all four tail phases.
- Actual original NAX selector engagement was128 calls. Key0, empty rows,
  negative/extreme/out-of-range/future IDs and slot2050 passed.
- Evaluated bool head broadcast was a stride0 view with final stride1,
  over33556480 bytes of physical dummy-column backing. No2GiB head mask was
  materialized.

Measured producer proof peak increased140681257 bytes above the resident
fixture. It includes retained selector parts and an **extra32MiB compact
membership copy for the CPU oracle**; this is not a joint native-attention
peak or runtime bill. The wrapper's conservative512MiB pending-layer allowance
and the whole-call timing/memory gate remain independent requirements.

The initial attempt passed ID/membership/counter checks but failed a stride
assertion that inspected a lazy broadcast before evaluation. MLX installs the
actual zero stride during view evaluation. One authorized probe-only eval
before inspection completed the proof; initial failure/source are retained.
No producer algorithm, layout variant or benchmark was changed.

Evidence key `glm53-full-history-mask-producer-20261003`,2026-10-03. ReleaseFast,
foreground `taskpolicy -a`, per-job exclusive GPU lock, confirmed5357/5768 RPM
at41.65°C and ten seconds idle. Locks were released before rebuild/after GPU;
fans returned automatic. Private source/binary/tensor references, compile-red/
green logs, complete result and both attempts are archived. No whole-attention
timing, quality or model evaluation accompanied this producer proof.

## Joint candidate rejected

The [joint native component](glm5-full-history-native-component.md) preserved
retrieval and passed its layout, rounding and finite-fallback proofs, but took
180.460ms versus131.378ms for current packed attention:37.36% slower, with
zero wins in eleven pairs. No variant or actual-model run was justified.
Producer module/probe and private roots were archived and removed; this exact
membership proof remains evidence, not an accepted runtime change. No selector
or other worker source was changed during producer cleanup.
