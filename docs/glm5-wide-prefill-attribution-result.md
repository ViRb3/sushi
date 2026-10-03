# Fixed L20 endpoint attribution

Do not pursue a tiny router or one-stage surrounding-work fusion from this
result. The actual five-endpoint diagnostic preserves all bits, but inserts a
4.09% complete-clock penalty. Its gate/up and down groups dominate the staged
case. Unperturbed removable non-GEMM latency remains unknown; this is not a
runtime scheduling candidate or model throughput measurement.

One actual T2048/top-eight L20 fixture and original full E288/n36/MCG/W12 banks
used the accepted transposed WIN32 projection body through a read-only API alias.
No shader was copied or changed. MCG/W12 was explicitly initialized. The normal
control is unmodified `glm_prefill_grid.tryMoe` with one final endpoint. The
private diagnostic settles five groups: sort/metadata/paired preparation;
gate/up; middle; down; finish. These waits, private wrapper and materialized
ownership can alter overlap, cache and clocks; the complete difference is not
pure GPU wait cost or a per-family latency share.

Numerical characterization passed: metadata arrays 50,752 values, prepared/gate/
up/middle/down F16 arrays 301,989,888 values and final BF16 array 8,388,608 values
matched between end-settled and five-settled compositions. The unmodified current
chain matched both final 8,388,608-value BF16 endpoints. Same original fixture,
full banks, current final output and both complete 12-stage materialized reference
sets stayed held equally before every timed arm. Stage proof and CPU scans stayed
outside timing.

| Staged group | Median ms |
| --- | ---: |
| Sort / metadata / paired preparation | 0.738333 |
| Gate + up | 10.797000 |
| Middle | 0.538791 |
| Down | 5.512833 |
| Finish | 0.475000 |

Three warmups per protocol preceded eleven alternating pairs. Complete medians
were 17.352958 ms normal and 18.061958 ms staged, 4.0858% longer; paired median was
4.0177% longer. All eleven staged arms were slower, range 2.4853–4.9198%.
All construction, endpoint evaluations and final frees are included. Cleanup
medians were 0.000042 ms normal and 0.001750 ms staged. Group construction/evaluation
are combined in the recorded group clocks, not falsely separated into GPU-only
durations. Every sample's five group clocks, total/cleanup and peak remain raw.

| Pair | Normal ms | Five endpoints ms | Change |
| --- | ---: | ---: | ---: |
| 1 | 17.369375 | 18.053916 | +3.9411% |
| 2 | 17.375334 | 18.061958 | +3.9517% |
| 3 | 17.364666 | 17.968375 | +3.4767% |
| 4 | 17.404000 | 18.260250 | +4.9198% |
| 5 | 17.332792 | 18.167833 | +4.8177% |
| 6 | 17.335708 | 18.032208 | +4.0177% |
| 7 | 17.410917 | 17.843625 | +2.4853% |
| 8 | 17.352958 | 18.008500 | +3.7777% |
| 9 | 17.325750 | 18.127708 | +4.6287% |
| 10 | 17.314375 | 18.081791 | +4.4322% |
| 11 | 17.327542 | 18.144000 | +4.7119% |

The sum of phase medians is 18.061957 ms; GEMM groups are 90.2994% of that staged
summary. Surrounding groups total 1.752124 ms. Arithmetic subtraction of the
0.709000 ms complete penalty leaves 1.043124 ms, about 6.01% of the normal clock,
but this is only the declared screening heuristic. It is not a rigorous latency
bound or removable share: perturbation need not belong to those groups and
normal execution overlaps differently. Every single surrounding group is less
than 5% of normal before charging perturbation. No one-phase implementation is
selected; combining unrelated phases would be a different broad fusion task.
The current three GEMMs are the main observed staged work, but no decode-ALU,
DRAM/cache, occupancy or spilling cause is inferred.

Maximum fresh extra peaks were 470,227,240 bytes normal and 469,965,056 bytes
staged above all equally-held reference/fixture memory. Both protocols construct
fresh graphs. These are scoped active-allocation peaks, not total model memory,
a new admission bill or permission to reduce any existing reserve. No target,
assistant or new capture was loaded.

The red stub compiled and failed at intended `MissingEndpointAttribution`.
Changing only that private stub switch enabled the fixed staged protocol; green
compiled and passed both tests, exit 0. Artifact
`glm53-prefill-endpoints-20261004` preserves commands, red/green sources,
actual fixture/source/binary/runtime hashes, every stage/pair/peak and thermal
records. Qualified probe SHA256
`c65381fedd1288262ddec79a310623680bac916828b9df264758487d62cfd180`;
binary SHA256 `f18f8209a31dd4a87bcbad2fecaf7d72c2e7209de9880d6b9b9f36cc4460d536`.
Red PID 8929 and green PID 9433 ended; foreground QoS, explicit accepted libraries,
confirmed maximum fans/idle and per-job GPU lock were used. Locks released and
fans automatic. Only documentation follows this diagnostic; no runtime/model
acceptance claim or further phase/tile experiment is supported.

Root restored the read-only alias; this worker hash-verified, archived and removed
only its two private probe/root files. Runtime remains unchanged.
