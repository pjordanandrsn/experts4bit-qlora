# PREREG — online rate estimation at the controller horizon

Registered before any measurement. The offline slot-pricing line closed
([RESULTS-regime.md](RESULTS-regime.md)) with a standing recommendation:

> slot-value decisions at the tail want ONLINE rate estimation and
> in-situ conversion measurement, not offline constants

This registration makes the estimation half falsifiable. Scope: the
ESTIMABILITY prerequisite only — whether the engine's own trailing
counters predict near-future bracket touch-mass well enough to steer a
controller. The controller itself (dynamic placement updates) is engine
work under its own future registration; nothing here changes serving
behavior.

## Claim

At the horizon a placement update would use (~32 decode steps), a
trailing mean of the engine's own per-step bracket touch counts
predicts the next 32 steps' mean with error (a) strictly below the
static offline profile's at every bracket and (b) at most **half the
certified offline envelope** at the tail — the two properties a
controller needs and a static profile provably lacks (tailvar: u_b4 =
56.3%; co-routing: tail rates are content-conditional).

## Design

* Same 10-window structure as tailvar (disjoint 28k-token slices,
  guard gaps), one run per window, uniform placement, `--gen-tokens
  128` (~140 decode steps/window), with the new per-step
  touched-expert series captured (`--series`; engine appends the
  already-computed unique tensor sync-free; series rolls back through
  interleaved prefills like every other counter).
* Brackets frozen from the committed co-routing manifests, as before.
* **Estimator (frozen, parameter-free beyond N):** trailing mean of
  the previous N=32 steps. Target: mean of the next 32. Evaluation
  points stride 8. **Offline baseline:** the static fit-set mean
  (windows w0–w4), evaluated on the same targets in held-out windows
  w5–w9.
* Cross-window boundaries (trailing-32 at the end of w vs the first 32
  of w+1) are REPORTED, never scored — a persistence estimator may
  legitimately lose across a hard content switch, and a controller
  would re-converge in ~one horizon.

## Bars ([online_verdict.py](online_verdict.py), self-tested on three
worlds: shifting-level world certifies; stationary world is
UNINFORMATIVE; heavy iid step-noise world is REFUTED)

* **H1**: online p90 relative error < offline p90 at ALL 4 brackets.
* **H2**: max online p90 at the tail brackets (b3, b4) ≤ **28%** (half
  the certified u_b4 = 56.3%).
* **H3 (control)**: max offline p90 at the tail ≥ 30%, else
  UNINFORMATIVE (a window set the static profile already handles
  cannot certify an improvement over it).
* NVMe tier empty in every run (assert).

**ONLINE-CERTIFIED** = H1 ∧ H2 with H3 informative. REFUTED means
short-horizon rates are not estimable from trailing counters at this
workload scale — closing the online path too, and with it any
data-driven slot control at B=16 (the honest end of the whole lever).

## Hard stop

One box, one scored run of the 10-window set; environment pin as
before. ≈ $0.80. On CERTIFIED, the deliverable is the measured online
error table beside the offline u_b — the quantitative case for the
controller registration that follows.
