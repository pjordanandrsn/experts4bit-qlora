# RESULTS — the co-routing model: M1-REFUTED, and the mechanism is not co-routing

Scored under [PREREG-coroute.md](PREREG-coroute.md) by the committed
[coroute_verdict.py](coroute_verdict.py); receipts in
[receipts-coroute/](receipts-coroute/). Box: the reference host class
(EPYC 9655 + RTX 5090, B_dram 433 GB/s, G0 117.4% PROCEED). W1 = 142
decode steps on corpus slice [0, 140000); W2 = the standard 5-point
sweep on [150000, 290000). Determinism held (pass-identical uniques at
every point); NVMe stayed empty; M0 failed 2/4 brackets so the window
pair carries signal. Cycle ≈ $0.77, box destroyed, zero instances.

## Verdict

**M1-REFUTED on all four brackets** — per-expert step-touch
frequencies measured on one content window do not predict held-out
bracket uniques:

| bracket | +slots | ΔU meas | M1 pred (±bar) | M2=M0 pred | M0 |
|---|---|---|---|---|---|
| 7→9 | 809 | 197.9 | 225.5 (±22.5) FAIL | 221.2 | not refuted |
| 9→10 | 405 | 56.8 | 51.4 (±5.1) FAIL | 50.9 | not refuted |
| 10→10.75 | 303 | 27.5 | 21.5 (±2.1) FAIL | 21.4 | refuted |
| 10.75→11.5 | 304 | 24.5 | 10.0 (±1.4) FAIL | 9.9 | refuted |

Deviations flip sign across brackets (−12%, +10%, +28%, **+145%**) —
this is not a scale factor, and no B_eff can absorb it.

## The surprise: B_eff = 16.00

The one-parameter fit on W1's own data landed at the nominal batch
size exactly: **on its own window, token-independence describes the
touch frequencies** (M2 ≡ M0 to three digits). The within-step
co-routing hypothesis — imported from the slot-value refutation's B1
failures — is NOT what breaks uniques predictions here. (Caveat
recorded: the least-squares fit is dominated by the many small-p tail
experts; a mass-weighted fit could land elsewhere. The bracket table,
not the fit, carries the verdict.)

## What the mechanism actually is: content non-stationarity of the tail

The tail bracket's experts (mass ranks ~4348–4652) were touched 2.4×
more often on W2 than W1's calibration measured — and the deep-DRAM
bracket slightly LESS. Tail-expert routing is workload-conditional:
which cold experts fire depends on what text is being served, and a
142-step window pins their rates no better than ±145% three brackets
down the mass ranking. The hot mass is stable (the two big brackets
sit within ~12% cross-window); the tail — exactly where slot-pricing
margins live — is not.

Reconciliation with the slot-value cycle: that experiment's profile
pass and sweep shared one (full-corpus) window, so its B1 failures were
a genuine same-window effect at its operating point; this experiment
shows the FIRST-ORDER cross-window error is non-stationarity, which
same-window designs cannot see. Both are real; non-stationarity is the
larger and the one any deployable model must survive.

## Disposition (per the registered hard stop)

M1-REFUTED closes the direct-transfer model: "recorded, and the
co-routing line needs a workload-conditional model, not a re-run."
Consequences for the slot-pricing re-derivation queue:

* Input (a) is **re-scoped**: not a co-routing correction but a
  **tail-rate uncertainty model** — any slot-value claim must carry
  window-to-window tail variance (measured here: ±12% hot, +145% at
  three-brackets-deep) or estimate rates online.
* Bracket predictions from ANY single offline profile inherit this
  floor; offline placement remains sound (hot mass stable), offline
  tail PRICING does not.
* Inputs (b) regime-split constants and (c) reference-host
  re-derivation are unchanged and still open.
