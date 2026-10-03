# Accepted current decode trace workload

One accepted CLI server loaded the Sushi 2.3 target and A6g128 assistant. The
frozen HTTP body tokenized to **8156 actual prompt tokens**. One warm request
and the next identical profiled request each produced 192 greedy ignore-EOS IDs;
all 192 IDs matched. Both recorded 81 N2/children 4 rounds and 891 native B3 calls.
No model-oracle framework, benchmark ladder, recapture or runtime edit was used.

The CLI was built from clean checkpoint `0e82efad`, runtime source `e1597cc2`,
before packed32 edits. SHA256 `a3fe8286fbdb9df2…e08e3d95d86e` matched before and
after the job; it was never rebuilt. The frozen body SHA256 starts
`81becded75986fd87`. No saved 8K HTTP body was available, so the prior qualified
2048-token smoke user content was repeated four times and its exact HTTP bytes
frozen. The count above comes from `/tokenize`, not a raw-ID override.

The collector recorded one nominal three-second Metal System Trace against
server PID 39485, exit 0, 50,445,539 bytes, with no permission error. Client markers
preserve paired UTC/monotonic anchors at request start, first nonempty SSE delta,
last real delta and response completion. Role-only/empty frames and keepalives
were excluded. The first visible delta index is not an inferred BPE-token index.

Recorder process spawn was within the observed decode interval, but process end
was after its last token. **Actual trace anchors and tables must establish usable
decode overlap**; its entire process lifetime cannot be treated as the recorded
window. Labels, correlated positive GPU/submission intervals and enough verifier
cycles remain collector adequacy requirements. No expert/KDA cost attribution or
throughput claim follows merely from successful recording and output parity.
Profiled request rates are not benchmark controls.

Artifact `glm53-current-systemtrace-20261003` retains private server/client/
collector scripts, exact body, warm/profiled raw SSE and IDs, tokenize/props,
binary/provenance, clock markers and trace/terminal logs. The owned server exited 0
before export, GPU lock was released and fans returned auto (manual false, TTL 0).
Foreground QoS, confirmed 5352/5783 RPM and 37.45°C after ten idle seconds were
recorded. A dry copied-binary launch failed RPATH resolution before model load;
using the original active CLI path fixed packaging without patch/rebuild.
One server/model load only; no second trace or extra generation request.
