# Rejected IndexPool pool-tile cadence

The bounded two-pool-tile NAX cadence is rejected. Its real-model prefill result
was too small and affected by drift to establish an accepted gain. The original
serial scorer remains in production; the prototype, probe and temporary evaluator
are archived privately. No alternative pool variant was attempted.

At source `58402254` plus the hashed WIP prototype, the existing real 16K fixture
with accepted packed-attention cadence enabled in both arms measured 129.559250
versus 108.509459 ms in whole attention: 16.25% less median time and 11/11 paired
wins after three warmup pairs. All 68,190,208 BF16 attention values, 262,112 FP32
score bits and 32,768 selected pool IDs matched, including partial and unpaired
tiles. The original native geometry and rounding were retained.

At source `c7e648a2` plus the same prototype, the actual-model evaluator loaded
the target once, warmed one 2048-ID request,
then ran four fresh 16384-ID requests in ABBA order. It repeated the fixed
official-template 2048-ID prompt eight times, using chunk2048, dense prefill,
async2 and profiling off. Times were 21.941960 / 22.297037 / 22.562843 /
23.122556 seconds. Control averaged 22.532258 seconds and candidate 22.429940:
only 0.454% less time amid strong monotonic slowdown. This did not qualify the
component win as a clear model performance improvement.

Last logits and every valid MLA prefix/pool/tail and initialized KDA conv/SSM
state byte matched outside timing. Capacity tails were excluded. Pool-cadence
calls were 0 / 2816 / 2816 / 0, confirming actual engagement in both candidate
arms. The candidate preserved BF16 compressed MLA caches and FP32 KDA state.

Artifact keys `glm53-indexpool-cadence-20261003` and
`glm53-indexpool-real-model-20261003` retain source hashes, raw results, binaries,
probe and private evaluator. Binary SHA256 prefixes are `dff4113eb104bfa3` and
`fc76a67e30913ea2`. Both runs used ReleaseFast, MLX 0.32.3 / `64ea011c`, patched
mlx-c `56b2d39`, foreground `taskpolicy -a`, exclusive GPU locks, confirmed maximum
fans and ten seconds idle. No implementation commit was made.
