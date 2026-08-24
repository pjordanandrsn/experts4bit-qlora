# PREREG — productionization: the controller as an engine component

Registered before any measurement. C2 certified the controller running
as BENCH-DRIVER code ([RESULTS-c2.md](RESULTS-c2.md)); this line moves
it into the engine proper and adds the change-point handling the
boundary data demands. Productionization is engineering, so the bars
are parity and regression bars, not discovery bars.

## The increment (this PR)

* **`engines/slot_controller.py`** — `SlotController`: the frozen
  C1/C2 rule (epoch 8, trailing-32, prior floor 0.25, swap gate
  `max(4/32, 3σ)`, per-layer) as an attachable component. It is the
  amortization series' sole production consumer and TRIMS it as it
  reads, so a long-running server holds O(32) history. Self-times
  (`ctrl.ns`), counts swaps and change-point resets.
* **Change-point reset** (`cp=True`): when the trailing-8 touch-mass of
  the CURRENT hot set falls below 0.5× its trailing-window mean, the
  history truncates to the last 8 steps — re-convergence in ~one epoch
  instead of one horizon (the online cycle measured 73–232% persistence
  error at hard switches).
* **The production hook**: `PagedModelRunner.run_decode` fires
  `slot_controller.on_decode_step()` between decode forwards when a
  controller is attached — one attribute, one call, no behavior change
  when absent.
* **The bench driver sheds its controller**: `c2_serve.py`'s inline
  rule is DELETED; `--controller` now attaches the engine component and
  `--controller-cp` its change-point variant. Single source of truth.
* **CI guard**: `tests/test_slot_controller.py` — six CPU-only unit
  tests over mocked tier states (gated swap fires, noise gate blocks
  tied margins, per-layer isolation, series trim on/off, change-point
  reset, refuses unarmed/unswappable states). These ran green against
  the real component before this registration.

## Protocol (one box, the C2 workload verbatim)

Swap self-test (void gate, unchanged), then **A1 → B → C → A2**:
static, engine controller, engine controller + change-point, static —
the ten C1 fresh windows served sequentially, same seed and config as
C2 (`receipts-c2/` is the driver-run reference).

## Bars ([p_verdict.py](p_verdict.py), self-tested: certify /
parity-fail / boundary-fail)

* **B1 (parity — amended pre-run, disclosed)**: the engine-owned arm's
  DECODE-ONLY uniques reduction ≥ **21.7%** (C2's 22.7 − 1 point), no
  upper cap. Bugbot's review exposed that the C2 driver's counters were
  prefill-inclusive (the rollback lived only in step_decomp); this line
  ports the rollback, and removing common-mode prefill mass can only
  RAISE the measured reduction — so parity is a no-regression floor,
  not a band. C2's receipts stand as measured (prefill mass was
  common-mode across its arms); the accounting change is recorded.
  (Static arms must agree exactly; mismatch voids.)
* **B2 (boundary value)**: over windows 1–9 (each begins at a hard
  content switch), the CP arm's first-32-step uniques ≤ **0.90×** the
  plain arm's, AND the CP arm's total reduction within 2 points of the
  plain arm's (the reset must not cost steady-state value).
* **B3 (wall, scoreable-only)**: when each controller arm's dram-bucket
  delta exceeds 3× the static-pair spread, both arms ≥ 5% reduction
  (C2's bar, held by the production component).

**PROD-CERTIFIED** = B1 ∧ B2 (∧ B3 when scoreable).
**PROD-DECISION-ONLY** = B1 ∧ B2 with the wall unscoreable. REFUTED
closes: B1 failing means the componentization changed behavior (a bug
by definition — fix under a new PR, re-run allowed ONCE since parity
bugs are engineering defects, disclosed); B2 failing means the
change-point design is wrong and ships disabled (`cp=False` default
stays), recorded.

## Hard stop

One box; one scored four-arm run (+ the single disclosed re-run only
for a B1 parity defect with an identified and fixed cause). ≈ $1.20.
