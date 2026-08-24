# RESULTS — online estimation: REFUTED as registered, on a 0.3-point tie at the stable bracket

Scored under [PREREG-online.md](PREREG-online.md) by the committed
[online_verdict.py](online_verdict.py); receipts in
[receipts-online/](receipts-online/). Box: EPYC 9655 + RTX 5090
(reference class), G0 117.9% PROCEED. Ten gen-128 windows with the
per-step touched-expert series captured; NVMe empty throughout.
Cycle ≈ $0.83, box destroyed, zero instances.

## Verdict

**REFUTED** — H1 (online p90 < offline p90 at ALL four brackets)
failed at exactly one bracket, by 0.3 percentage points:

| bracket | static mean | online p90 | offline p90 | online wins? |
|---|---|---|---|---|
| b1 deep | 288.1 | 20.1% | 19.8% | **no (−0.3 pts)** |
| b2 | 85.5 | 19.7% | 28.7% | yes (+9.0) |
| b3 | 42.8 | 25.4% | 37.1% | yes (+11.7) |
| b4 tail | 28.7 | **27.0%** | 38.4% | yes (+11.4) |

* **H2 PASSED**: online tail p90 = 27.0% ≤ 28% — the estimability bar,
  half the certified offline envelope, held.
* **H3 held**: offline tail p90 38.4% ≥ 30% — the window set carries
  the non-stationarity the static profile fails on.
* Boundary report (unscored, as registered): across hard content
  switches the persistence estimator is catastrophically wrong —
  tail-bracket median 73–86%, worst 232%. A controller would need
  change-point handling; re-convergence takes ~one horizon.

## What actually died, stated precisely

The registered H1 conflated two claims: "online estimation is useful
where the static profile fails" (true on these receipts: +9 to +11.7
points at b2–b4) and "online strictly dominates everywhere, including
the bracket where the static profile is already adequate" (false by
0.3 points at b1, where BOTH estimators sit at the target's own
32-step noise floor — tailvar measured b1's cross-window cv at 9–11%,
so there was nothing for recency to recover there). The bars are bars:
the claim as written is REFUTED and per the hard stop this line closes
without a re-run.

**A prereg-drafting lesson is recorded alongside**: the prereg's
consequence clause ("REFUTED means short-horizon rates are not
estimable") over-reached its own bar structure — estimability had its
own bar (H2) and that bar PASSED. Consequence text must map
one-to-one onto the bars that can fire, or a refutation's meaning
gets misstated in advance. This RESULTS corrects the record: what is
closed is the strict-dominance claim; the estimability measurement
stands as reported.

## What the receipts leave behind (reported, none of it certified)

* Online trailing-32 error table beside the certified offline u_b:
  [20.1, 19.7, 25.4, 27.0]% vs offline [19.8, 28.7, 37.1, 38.4]% p90 —
  and vs the certified window-level envelopes [22.5, 45.7, 52.3,
  56.3]%.
* The boundary numbers above, for any future change-point design.
* Any future controller registration starts from a new prereg with a
  dominance-where-it-matters bar — that is a decision for the program
  owner, not a re-run of this one.
