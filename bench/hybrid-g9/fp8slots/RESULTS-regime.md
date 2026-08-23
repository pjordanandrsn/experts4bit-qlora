# RESULTS — regime-split constants: UNINFORMATIVE, with the investigation the prereg ordered

Scored under [PREREG-regime.md](PREREG-regime.md) by the committed
[regime_verdict.py](regime_verdict.py); receipts in
[receipts-regime/](receipts-regime/). Box: EPYC **9755** + RTX 5090
(the only eligible host on the board; the fit is host-labeled by the
prereg's own terms), B_dram 493 GB/s, G0 103.4% PROCEED. Gate ACCEPT
(median pass noise 1.1%, worst 4.6%). Cycle ≈ $0.80, box destroyed,
zero instances.

## Verdict

**UNINFORMATIVE** — the flat-58 null missed only 1 of the 4 scored
brackets (bar ≥ 2). Per the registered hard stop this triggers an
investigation, not a re-run, and the line closes without certified
constants.

| scored bracket | ΔU | ΔC | ΔT | model | allow | flat-58 |
|---|---|---|---|---|---|---|
| 8→9 (dense) | 77.9 | 3.7 | 2.44 | 2.64 PASS | 1.08 | 4.52 **misses** |
| 9.5→10 | 20.8 | 3.2 | 1.38 | 1.49 PASS | 0.30 | 1.21 fits |
| 10.4→10.75 | 8.8 | 3.0 | 0.69 | 1.21 PASS | **1.87** | 0.51 fits |
| 11.1→11.5 | 3.9 | 1.0 | 0.52 | 0.43 PASS | **1.15** | 0.22 fits |

Fitted (reported, NOT certified): c_fix = 356.5 µs/layer-step,
c_u = 16.8 µs/unique, R² = 0.907 over all 8 brackets.

## The investigation

1. **Why the null survived: allowance inflation, not fit.** The two
   noisiest gate points (10.4 GB at 3.9%, 11.5 GB at 4.6% — both
   within the accepted gate) sit at the endpoints of the two sparse
   scored brackets. Their pass-spreads propagate into `3×spread`
   allowances of 1.87 and 1.15 ms on measured signals of 0.69 and
   0.52 ms. A bracket whose allowance exceeds its own signal cannot
   discriminate any hypothesis — the null "fits" there vacuously, and
   the model's PASSes there are equally vacuous.
2. **The design gap is mine and is now named**: the prereg gated the
   BOX on per-point noise but never required per-bracket
   discriminability (the slot-value design's spoiler, applied
   per-bracket). The 0.35–0.4 GB sparse steps produce ~0.5–1.2 ms
   bracket signals — the same order as 3×spread even on a good box.
   Any successor needs coarser sparse brackets (a single 10.0→11.5
   bracket carries ~2.6 ms) or a per-bracket spoiler that voids
   non-discriminating cells instead of letting them vote.
3. **What replicated anyway**: the dense-regime rejection of the flat
   58 — its third appearance (slot-value cycle, its A/A pair, and
   here: 4.52 predicted vs 2.44 ± 1.08 measured). The dense marginal
   is decisively below the flat average on every host measured:
   ~26 µs/unique (9655, 421–433 GB/s) and ~17 µs here (9755,
   493 GB/s) — direction consistent with bandwidth scaling, values
   host-labeled, neither certified.
4. c_fix = 356 µs/layer-step is poorly conditioned (three of four fit
   brackets have nearly identical ΔC ≈ 3.0–3.7) and should not be
   quoted even informally.

## Disposition — the offline slot-pricing line, closed end to end

Per the hard stop, no re-run. The fp8slots line's final state:

* **Refuted**: the original pricing `P(touched) × 58 µs`, both its
  factors, and the ~17–35 µs/slot band (slot-value cycle).
* **Certified**: the tail-rate uncertainty envelopes
  `u_b = [22.5%, 45.7%, 52.3%, 56.3%]` (tailvar cycle) — the one
  artifact any future offline claim must carry.
* **Replicated but uncertified**: dense-marginal ≪ flat-average
  (three independent appearances); conversion constants remain
  host-labeled reports.
* **Standing recommendation, now with receipts behind it**: slot-value
  decisions at the tail want ONLINE rate estimation and in-situ
  conversion measurement, not offline constants — the tail's ±56%
  rate envelope and the sparse regime's noise-shrouded time signals
  point the same way.
