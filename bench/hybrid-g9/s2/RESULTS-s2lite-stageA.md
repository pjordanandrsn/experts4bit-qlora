# RESULTS — S2-lite Stage A: gate PASSED, economics REFUTED-FOR-CELL

Measured 2026-08-25 under PREREG-s2lite (+ the capture amendments in
this PR). Receipts in `receipts-s2/stageA/` (RTX 5090, e4b `1b483d1`
plus this PR's harness fixes — the timing instrument iterated live and
its final form is what this PR merges; instance destroyed, vast zero).

```
S2-LITE STAGE A VERDICT: REFUTED-FOR-CELL
  best K=16: T_pred 0.46x < 1.2x
```

## The mechanism is CORRECT — the gate passed first

`S2_GATE k=16 match=17/17 bitwise=True continuation=True PASS=True`

One verify-mode step, fed the sequential oracle's window as its
draft, reproduced all 17 argmax tokens exactly, and the 4-step
continuation after rewind matched the sequential world's. The
staggered-lens verification (K+1 rows, one slot, lengths base+1..K+1)
is right; what fails is the price.

## The price, measured four ways

1. **Graphed verify (K=16, singleton grouping): 47.71 ms** against a
   7.41 ms anchor (A/A 0.008). Three capture blockers were fixed to
   get this number (side-stream warm for the T=17 shapes;
   `rewind_nosync` because the checked rewind's `.item()` is illegal
   under capture; `FORCE_SINGLETON_GROUPS` because the MoE grouping's
   `unique_consecutive` host-size sync cannot be captured — the same
   constraint b1d hit at T=1).
2. **MoE cost vs routing overlap** (controlled sweep, 136 rows):
   distinct=8 → 1.47× a decode step's expert cost; 32 → 2.17×;
   64 → 3.43×; 128 → 6.10×. Grouped routing's cost is set by DISTINCT
   experts touched, not rows.
3. **Real routing overlap**: across 68 recorded decode steps,
   17-token windows touch **41.5 distinct experts** per layer
   (min 24.6, max 66.8) of 128; 33-token windows touch 53.8. Far more
   concentrated than random (~78) — and far above the ~8 that
   verification-for-free requires.
4. **Bounds**: correcting the measured 47.7 ms by the measured
   grouping delta gives ~30 ms (0.72× — speculation LOSES);
   composing from the census gives ~16 ms (1.37×, ≈185 tok/s) as the
   optimistic ceiling for a device-grouped verify that does not exist.

## Why this refutes the LANE, not just the cell

The refutation is architectural: **sparse MoE at B=1 makes
verification expensive in exactly the proportion that drafting is
useful.** Each accepted token still pays its own experts' weight
traffic — the thing speculation amortizes in dense models is the one
thing an MoE does not share across tokens. Acceptance would need to
grow faster than distinct-experts-per-window, and the measured curves
move together (2.948 accept / 41.5 distinct at K=16 → 3.447 / 53.8 at
K=32). The 425 goal does not survive this: even a floor-cost bespoke
GEMV composed with a device-grouped verify lands near ~340-350 tok/s
(bounds in the S1/S2 receipts), short of 425. **425 single-stream is
REFUTED for this model on this hardware** under everything measured
this campaign; the composed ceiling is ~350.

## What stands after S2

- The verify mechanics (lens_override, rewind, verify mode) are
  correct, tested, and merged — available to any future lane
  (batch>1 speculation, shared-expert models) without rework.
- The prompt-lookup acceptance table (2.948–3.926 tokens/step) is
  real and paid for; it transfers to any architecture whose verify
  is cheap.
- The remaining REGISTERED throughput program: the fusion tail
  (~0.5–1 ms of the 7.41 ms step) and the bespoke loads-floor GEMV
  (K5's fresh-prereg branch; K4 floor pair 12.4 µs/layer ⇒ GEMV
  ~0.6 ms/step ⇒ step ~3.5 ms ⇒ **~280 tok/s**), which is now the
  campaign's limit lane.
