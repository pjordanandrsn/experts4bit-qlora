# RESULTS — productionization: REFUTED on B2; the component itself proved out

Scored under [PREREG-productionization.md](PREREG-productionization.md)
by the committed [p_verdict.py](p_verdict.py); receipts in
[receipts-prod/](receipts-prod/). Box: EPYC 9655 + RTX 5090 (reference
class), swap self-test OK, static arms bit-identical, NVMe empty.
Cycle ≈ $1.02, box destroyed, zero instances.

## Verdict

**REFUTED** — B1 ∧ B2 was the claim, and B2 failed. Per the prereg's
own disposition: the change-point design is wrong and ships disabled
(`cp=False` stays the default); no re-run (the disclosed re-run
allowance was only for a B1 parity defect, and B1 passed).

| bar | result |
|---|---|
| B1 parity (decode-only ≥ 21.7%) | **PASS — 34.2%** (B), 36.5% (C) |
| B2 boundary (CP first-32 cut ≥ 10%) | **FAIL — 4.7%**, 22 resets vs ~9 real switches |
| B3 wall (≥ 5% when scoreable) | pass in substance: −20.8% / −20.4%, spread 45 ms |

## What B1's pass established (recorded fact inside a refuted registration)

The engine-owned `SlotController` — one runner hook, no driver logic —
delivers **34.2% decode-only uniques reduction** and **20.8%
dram-bucket wall reduction** on the C2 workload. The decode-only
number confirms the accounting prediction made when Bugbot exposed the
prefill-inclusive counters: C2's 22.7% was diluted by common-mode
prefill mass; the truer metric reads a third off the CPU expert bill.
The component, its runner hook, and its six CI unit tests are merged
engine code and stay.

## The change-point autopsy (three independent causes)

1. **False positives**: 22 resets against ~9 boundaries — the
   trailing-8 hot-set-mass ratio at 0.5× is too twitchy on step noise.
2. **Little to win**: the plain controller already re-converges within
   ~2–3 epochs by construction; most boundary regret sits in the first
   epoch, before ANY estimator can act. The theoretical headroom at a
   32-step horizon was smaller than the online cycle's raw boundary
   errors suggested (those measured the ESTIMATOR, not the placement it
   feeds).
3. **Production-disqualifying cost**: `_hot_mass` walks the whole
   trailing window with per-element tensor indexing in Python on every
   step — 644 s of controller overhead in the CP arm vs 1.7 s plain.
   Even a correct detector must be O(1)/step on counters.

Any revival is a NEW registration: an O(1) counter-based detector
(e.g., epoch-aligned hot-mass from the already-maintained trailing
counts), evaluated at the first-EPOCH horizon where the regret actually
lives. Nothing in the certified value depends on it.

## Line disposition

The productionization line closes with: the component in the engine
(hook + tests + `cp=False`), the CP feature refuted and disabled, and
the standing follow-ons unchanged (scheduler-native attachment beyond
the bench driver, cross-layer rebalance) — each its own registration if
wanted. The measurement program's slot-lever arc is COMPLETE: refuted
offline pricing → certified uncertainty → certified estimability →
certified decision value → certified engine value → productionized
component with its guard rails, and a refuted bell-and-whistle honestly
recorded beside it.
