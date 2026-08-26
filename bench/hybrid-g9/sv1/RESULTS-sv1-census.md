# RESULTS — SV1: current-stack decode census + dot-pad×F2 composition

Adjudicated 2026-08-26. Box: instance 48709950 (post-F2 anchor
7.25 ms). Receipts in `receipts-sv1/`.

## Composition cert: PARTIAL (K6-B frame) — knob stays opt-in

```
off_a/off_b: 7.251 / 7.249 ms   (A/A spread 0.002 ms)
dotpad_on:   6.476 ms           (ratio 0.893; 127 tokens identical)
```

`GNF4_GEMV_DOTPAD=1` composes cleanly with the F2 defaults:
**154.4 tok/s single-stream** on this box (138 default). Ratio 0.893
lands in the K6-B PARTIAL band (≤0.85 PASS / ≤0.95 PARTIAL), same
tier as K6-B's own cert — the knob remains opt-in, now with a
composed receipt.

## Census: budget REFUSED coverage; busy floor NOT DELIVERED

The profiled census ran, but `step_budget.py` refused coverage
(recorded in `sv1_progress`), and the raw kernel table was then LOST
in teardown: the pull returned an empty file, the `[ -s ] || rm`
guard dropped it silently, and the box was destroyed on a file COUNT
rather than a manifest diff. Process failure recorded (memory +
teardown scripts now refuse destroy on a manifest mismatch).

**Consequence for the 275 target:** the pre-committed decision number
(busy floor vs 3.64 ms) is still unmeasured. It re-runs as a rider on
the BV3b box. The composition cert stands regardless: the certified
single-stream ladder is 138 default / 154.4 with the knob, and the
gap to 275 is 2.8 ms — undecidable until the floor lands.
