# PREREG — the within-step co-routing model (re-derivation input (a))

Registered before any measurement. This is the first of the three
inputs [RESULTS-slotvalue.md](RESULTS-slotvalue.md) names for any
future slot-pricing claim:

> (a) a within-step co-routing model replacing token independence (the
> measured ΔU table here is its calibration data)

The refutation measured `P = 1 − (1 − p)^16` over-predicting bracket
uniques by 22–40%. This registration replaces the modeled quantity with
the measured one and tests whether it TRANSFERS.

## Models under test

* **M1 (registered model — direct estimand):** per-expert step-touch
  frequency `P̂_e = touched_steps_e / steps`, measured on a calibration
  window. Bracket prediction is `Σ P̂_e` by linearity of expectation —
  no independence assumption anywhere.
* **M2 (portable approximation, reported):** `P = 1 − (1 − p)^B_eff`
  with scalar `B_eff` fitted on the calibration window (grid 1→16 step
  0.05, least squares over experts). One number that says how far the
  effective batch is from the nominal 16.
* **M0 (the refuted null, falsifiability control):** independence at
  B=16. It must FAIL again on the fresh window — a window pair on which
  M0 passes carries no co-routing signal and the run is UNINFORMATIVE.

## Design — two disjoint windows, deterministic counters, no noise gate

Uniques are deterministic given a prompt window (greedy decode; the
slot-value sweep measured bit-identical uniques across its two passes),
so no A/A box-noise gate applies. The inferential weight is
**cross-window generalization**:

1. **W1 (calibration):** one run at `vram_gb = 10.0`, `--gen-tokens
   128` (more steps → tighter P̂), corpus slice `[0, 140000)`
   (`--prompt-offset 0 --prompt-span 140000`). Collects per-expert
   `touch` and `hist` counters (decode-only via the committed rollback
   mechanism).
2. **W2 (held-out):** the standard sweep — ladder {7.0, 9.0, 10.0,
   10.75, 11.5} GB, two passes, `--gen-tokens 48`, corpus slice
   `[150000, 290000)` (`--prompt-offset 150000 --prompt-span 140000`).
   The slices are DISJOINT with a 10k-token guard gap — a bare offset
   is not enough, since prompts spread over the whole remaining corpus
   (Bugbot, e4b#192). The pass pair is the determinism check: any
   uniques mismatch across passes at any point ⇒ VOID.
3. Everything else as the slot-value run: Qwen3-30B-A3B, B=16, chunk
   512, pool 32 / torch-intraop 8, solver constants 55/2, measured
   routing profile for placements, NVMe tier must stay empty.

## Bars (window-sampling error, computed by the committed
[coroute_verdict.py](coroute_verdict.py))

Per bracket, with `σ = sqrt(Σ_e P̂(1−P̂) · (1/S1 + 1/S2))`:

* **M1 CERTIFIED** iff `|ΔU_meas − Σ P̂_e| ≤ max(10%, 3σ)` on **all 4**
  brackets.
* **M2** reported with its bracket score at `max(15%, 3σ)`.
* **M0** must exceed `max(15%, 3σ)` on ≥1 bracket, else UNINFORMATIVE.

Self-tests committed: a synthetic co-routed world (planted B_eff = 6)
certifies M1 and refutes M0; a shifted-window world refutes M1.

## Hard stop

One box, one scored W2 sweep. M1-REFUTED means step-touch frequencies
are non-stationary across windows at this workload scale — recorded,
and the co-routing line needs a workload-conditional model, not a
re-run. VOID (determinism broken) pauses the line for a stack
investigation. Environment pin from the last cycle: `accelerate`
installed, `torchvision`/`torchaudio` removed before any serving run.
