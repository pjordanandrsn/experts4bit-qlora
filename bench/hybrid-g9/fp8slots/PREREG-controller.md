# PREREG — the slot controller, stage C1: decision value

Registered before the scored measurement. The controller line the
online cycle earned: trailing-counter estimation is real where the
static profile fails (+9–11.7 p90 points at b2–b4) but persistence is
catastrophic across content switches. C1 certifies the DECISION VALUE
of a concrete controller rule on the engine's true objective, from
strictly-causal counter arithmetic — no engine changes, no timing, no
weight movement. **C2** (in-engine swap machinery + end-to-end wall
time) is registered only if C1 certifies.

## The frozen controller rule ([controller_verdict.py](controller_verdict.py))

Budget S = 4045 slots (the 10 GB operating point). Deployed static
baseline = top-S by the pooled design-set rates (the committed
[receipts-online](receipts-online/) series — the profile a deployment
would ship). Every 8 steps the controller re-estimates each expert's
touch rate as `max(trailing-32 mean, 0.25 × prior)` and performs only
swaps that are BOTH economically material and statistically real:
in/out pairs with estimated gap `> max(4/32, 3 × combined binomial
sd)`. Objective: per-step DRAM touch-mass; swaps are charged 2
uniques each (an H2D expert copy ≈ 106 µs ≈ 2 uniques at the measured
dense marginal) in the adjusted metric.

Two rule revisions happened DURING design (disclosed, design-set only):
the naive full-replace controller lost its 27.9% raw gain to churn
(16,235 swaps → net 2.7%); the economic gate recovered it and the
noise-aware term then cut tie-margin churn 5× (final design-set:
raw 19.8%, swap-adjusted 17.3%, 1,637 swaps).

## Scored data — fresh windows, never measured

Ten windows at the SHIFTED grid `offset = 14500 + i × 28400`,
`span = 28000` (disjoint from each other; deliberately offset half a
stride from every previously measured window), gen 128, uniform
placement, per-step series captured. The committed design-set series
supply the prior; the fresh set is scored. Same box flow as the online
cycle; NVMe empty asserted.

## Bars (design-set sizing disclosed above; scored on fresh only)

* **B1**: swap-adjusted reduction ≥ **10%** vs deployed static.
* **B2**: raw reduction ≥ **15%**.
* **Control (could it have failed)**: the split-half oracle (top-S from
  each window's EVEN steps, evaluated on its ODD steps — selection on
  sampling noise cannot survive the split) must gain ≥ 8% over static,
  else UNINFORMATIVE. Self-tested three ways: block-shift world
  CERTIFIED (74% of oracle gap captured); stationary world
  UNINFORMATIVE (the naive within-window oracle was rejected during
  design for overfitting +9.5% out of pure sampling noise — the
  split-half form fixed it); sub-threshold-shift world REFUTED (real
  oracle gains of 23.8% that the swap economics rightly refuse to
  chase).

CONTROLLER-CERTIFIED = B1 ∧ B2 with the control informative. REFUTED
closes C1 as specified — a different rule needs its own registration —
and C2 is not built. UNINFORMATIVE means the fresh grid lacked
non-stationarity; one re-draw of the window grid is permitted (offsets
re-randomized), disclosed, at most once.

## Hard stop

One box, one scored fresh set (plus at most one disclosed grid
re-draw). ≈ $0.85. On CERTIFIED, C2's registration inherits: the rule
verbatim, the measured swap counts (its H2D budget), and the boundary
change-point data from the online cycle.
