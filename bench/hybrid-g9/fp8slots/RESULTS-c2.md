# RESULTS — slot controller C2: C2-CERTIFIED

Scored under [PREREG-c2.md](PREREG-c2.md) by the committed
[c2_verdict.py](c2_verdict.py); receipts in [receipts-c2/](receipts-c2/).
Box: EPYC 9655 + RTX 5090 (reference class; the first rental died with a
live-looking API record — the HANDOFF's dead-box law — and was destroyed
unpaid-for-nothing), G0 120.9% PROCEED. Cycle ≈ $1.00 across both
rentals, boxes destroyed, zero instances.

## The gates, in order

1. **B3 void-gate PASSED**: `SWAP-SELFTEST-OK` — a promoted expert's
   slot matched its DRAM source bit-exactly across all four segments,
   the swap-back restored the hot stacks bit-identically, and a fixed
   greedy continuation was unchanged. The in-engine swap mechanism is
   machine-checked correct.
2. **Determinism**: the two static arms agreed on uniques exactly
   (268,648 = 268,648).
3. **Wall scoreability**: delta 4,455 ms vs 3× static-pair spread
   1,122 ms — scoreable with a 4× margin (the regime cycle's lesson,
   satisfied rather than shrouded this time).

## The verdict

| metric | static (A1/A2) | controller (B) | bar |
|---|---|---|---|
| uniq_dram total | 268,648 | **207,556 (−22.7%)** | ≥ −12% |
| dram bucket total | 19,903 / 20,277 ms | **15,635 ms (−22.2%)** | ≥ −5% |
| whole-run wall (unscored) | 273.9 / 276.8 s | 267.0 s (−3%) | reported |
| swaps | — | 1,948 | reported |
| controller overhead | — | 1,180 ms (~0.8 ms/step) | reported |

**C2-CERTIFIED.** The in-engine controller beat even C1's constrained
sim (22.7% vs 17.5% uniques) — the live rule adapts to the routing it
actually serves rather than a replayed series — and the mass reduction
translated ~1:1 into dram-bucket wall, netting the pageable H2D and the
controller's own python time.

## What is now certified, line-wide

* **C1**: the rule's decision value on held-out content (16.0% net).
* **C2**: the engine mechanism (bit-exact swaps between steps) and the
  wall value at the operating point (22.2% of the CPU expert bill,
  ~4.5 s over a 10-window serve; ~3% of whole-run wall with attention
  dominating the remainder).
* Together they close the arc the refuted offline pricing opened: the
  slot lever is real, but it is a CONTROLLER property, not a static
  pricing property — rates are content-conditional (tailvar), offline
  tail pricing cannot see them (u_b), and the value ships as a runtime
  rule with noise-aware economic gating.

## Standing state after C2

The controller is a bench-driver capability (`c2_serve.py --controller`)
over a production engine mode (`enable_hybrid_tier(swappable=True)`).
Productionizing into the serving scheduler proper (G9-class engine
work), change-point handling at hard content switches (the online
cycle's boundary data: persistence errs 73–232% for ~one horizon), and
any cross-layer slot rebalancing (the ~1-point gap C1 measured) are
follow-on engineering under future registrations. The measurement
program this line needed is complete.
