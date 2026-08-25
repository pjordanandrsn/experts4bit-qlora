# RESULTS — S2-lite Stage A: gate PASSED; the SINGLETON verify bound REFUTES the cell as measured; the grouped question is registered, not answered

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

## What this verdict adjudicates — and what it deliberately cannot

The 47.71 ms number was measured with `FORCE_SINGLETON_GROUPS`,
because the normal T>1 grouping path's `unique_consecutive` host-size
sync cannot be captured. Singleton grouping executes one M=1 group
per route assignment: it DISABLES expert-weight reuse across rows by
construction. The verdict above therefore adjudicates the singleton
bound — a valid, conservative UPPER bound on verify cost, refused
under the registered map exactly as the map requires — and the bars
are not moved or reinterpreted here.

What it cannot do is price the best grouped verification path, and
this cycle's own controlled data proves substantial reuse is
available:

- K=16 verify = 17 rows × top-8 = **136 route assignments over 41.5
  distinct experts/layer = 3.28 assignments per loaded expert**;
  K=32 = 264 over 53.8 = **4.91 per loaded expert**.
- The controlled 136-row curve runs 1.47× a decode step's expert cost
  at 8 distinct experts to 6.10× at 128 — cost tracks the
  **distinct-expert union**, not token×expert invocations.

So the correct statement of the concern is:

> Verification can amortize expert weight traffic across rows routed
> to the same expert. The measured concern is that the distinct-expert
> union grows with the verification window, potentially faster than
> useful accepted-token yield.

The negative hypothesis remains live on the existing numbers: even
with perfect grouping, K=16 touches 41.5/8 = **5.19×** ordinary
decode's distinct experts for 2.948 accepted tokens/verify, and K=32
touches 6.73× for 3.447 — the union may outgrow the yield. But that
is a prediction, not a receipt. **PREREG-s3-grouped-verify** (fresh
prereg, committed before any implementation) registers the direct
test: a fully device-side, capture-safe grouped verification arm,
compared three ways — singleton vs eager-grouped vs captured
device-grouped — so the reuse-disabled artifact in 47.71 ms is
quantified rather than inferred. **No statement about 425 tok/s is
made until that grouped receipt exists**; the composed ceiling is
recomputed then.

## What stands after S2

- The verify mechanics (lens_override, rewind, verify mode) are
  correct, tested, and merged — available to any future lane
  (batch>1 speculation, shared-expert models) without rework.
- The prompt-lookup acceptance table (2.948–3.926 tokens/step) is
  real and paid for; it transfers to any architecture whose verify
  is cheap.
- The remaining REGISTERED throughput program: S3 (the grouped
  verify arm above), the fusion tail (~0.5–1 ms of the 7.41 ms
  step), and the bespoke loads-floor GEMV (K6, registered; K4 floor
  pair ⇒ step ~3.5 ms ⇒ ~280 tok/s trajectory).
