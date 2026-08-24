# PREREG — closing G8 B=16: balance ≥ 0.80 at a legitimate serving config

Registered before any measurement. The campaign's headline gate has
been open since 2026-08-19 ([HANDOFF.md](../HANDOFF.md)):

> G8 | balance >= 0.80, B=16 | **OPEN — best clean 0.698 (Zen 5);
> needs ~-14% on the wall**

The metric, decoded from the committed receipts
([b16close/g8_CTRL.json](../b16close/g8_CTRL.json)) and verified on
both batch rows: `balance_ratio = min(gpu_ms, dram_ms) / max(gpu_ms,
dram_ms)` over per-decode-step medians (B=8: 13.14/15.05 = 0.873 ✓;
B=16: 15.73/22.88 = 0.688 ✓). The productionization run measured the
controller taking 20.8% off the dram bucket — more than the −14% the
gate needs — and, separately, that a mass-spread placement carries a
far smaller DRAM bill than the g8 run's layer-concentrated one
(sixteen whole layers in DRAM, from their calibration constants). This
registration measures both candidate closers on ONE box with the
gate's own metric.

## Arms (one box, the C2/prod workload at B=16, gen 128, ten windows)

* **A1, A2 (static, mass-spread)** — the deployed solver placement over
  the design-set profile at 10 GB; the A/A pair for balance noise.
* **C (controller)** — A's placement + the engine `SlotController`
  (plain, `cp=False`).
* **R (context, never scored)** — the solver run with the COMMITTED g8
  calibration ([b16close/calib.json](../b16close/calib.json)) in place
  of the box's own: does the layer-concentrated shape and its ~0.69
  balance reproduce from the cost constants alone?

The driver now reports per-step `gpu_ns` (the amortization dict's
CUDA-event-bracketed GPU expert bus — the same quantity class the g8
receipts used) beside `dram_ns`, and computes the g8 balance formula
over whole-serve medians.

## Bars ([g8close_verdict.py](g8close_verdict.py), self-tested on
static-closes / controller-closes / undetermined / not-closed receipts)

* **The gate**: the best serving arm among {mean(A), C} has
  `balance_ratio ≥ 0.80` AND `(best − 0.80) > 3 × |A1 − A2|` (the
  balance spread). CLOSED names the arm; a best within 3× spread of
  the bar is UNDETERMINED; below it, NOT-CLOSED.
* Determinism (static arms' uniques equal), swap self-test, NVMe empty
  — voids as before.
* **Honesty clause**: min/max balance is the GATE's definition, not the
  program's objective (the spec's revised objective is
  bytes-through-DRAM and wall). Balance penalizes a too-fast CPU tier
  symmetrically, so the controller can WIN on wall while LOSING on
  balance at this placement. Both walls and both medians are reported;
  the RESULTS must state which arm closes the gate and which minimizes
  the wall, separately, and may not blur them.

## Hard stop

One box, one scored four-arm run. NOT-CLOSED leaves the gate open with
the two walls on record — no re-run, no bar adjustment; the gate would
then need either their exact thin-routing config reproduced (a
different registration) or GPU-side work.  ≈ $1.20.
