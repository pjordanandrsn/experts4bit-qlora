# PREREG — T5: the dispatch-algebra diet (SPEC-425, gate 5)

Registered 2026-08-24, before any measurement. Prior art this builds on:
`RESULTS-t1b-and-probe.md` (merged #216) measured, per decode step at the
certified operating point (Qwen3-30B-A3B, B=16, SlotController on,
reference EPYC + RTX 5090 host class): wall ≈ 141 ms, device work
≤ ~32 ms, host dispatch ≥ ~110 ms, and a host op spray of ~5,000
`copy_`, ~2,400 `.to`, ~880 `index_select`, and **174 `aten::nonzero`
calls — each a device→host synchronization**.

## Named mechanism (from code reading, not yet from attribution)

Per layer, `_HotResidency.forward` + `_HybridTier._cold_contrib` issue:

1. `hot_row.nonzero()` (hr) — sync;
2. `(~hot_row).nonzero()` (cr) — sync;
3. `is_dram[flat[cr]]`-derived `nonzero` splits in the cold path — sync,
   even though in swappable/controller mode `is_dram` is ALL-TRUE and
   the split is placement-known without looking at the data;
4. `row_token`/`row_slot` aranges rebuilt every layer;
5. `row_token.index_select(0, hr)` and friends computed 3–4× per branch.

At 48 layers that code shape predicts ~2–4 nonzero syncs/layer ≈ the
measured 174/step. Each sync serializes the host launch path with device
execution; the spray itself is host busy-work.

## Registered edits (arm B), behind one mechanism-named flag

Engine flag `dispatch_diet: bool = False` (constructor-plumbed through
`enable_hybrid_tier`; no model names — portability rule). Bench flag
`--dispatch-diet`. Edits:

- **E1 single-sync split**: replace the hr/cr double-`nonzero` with ONE
  small host read `n_cold = int((~hot_row).sum())`; when `n_cold == 0`
  take an all-rows hot fast path (no partition at all); otherwise ONE
  stable `argsort(hot_row)` yields cr = perm[:n_cold], hr = perm[n_cold:].
- **E2 placement-known cold split**: when the placement has every
  non-hot expert DRAM-resident (`is_dram.all()`, cached host-side at
  placement-change time, not read per step), `_cold_contrib` forwards
  cr straight to the DRAM path — no mask, no nonzero.
- **E3 hoisted index algebra**: `row_token`/`row_slot` cached per
  (T, k, device) on the tier; rebuilt only on shape change.
- **E4 de-duplicated gathers**: each `row_token.index_select(0, sel)` /
  `row_slot.index_select(0, sel)` computed once per branch and reused.

Out of scope (T5b candidates, NOT this cycle): attention-side `.to`
churn, batched cross-layer dispatch, pinned-staging descriptor reads,
CPU-side `_group` restructure.

## Instruments

`step_decomp.py` unchanged except the `--dispatch-diet` flag (identity
otherwise). Op counts and sync counts come from the EXISTING
`--torch-profile-out` path (12-step active window); the verdict
calculator parses `aten::nonzero` / `aten::to` / `aten::copy_` /
`aten::index_select` "# of Calls" per window from each arm's table.
Amortization instrument OFF in ALL arms (production shape); token
identity via `generated_tokens` equality across arms.

## Design

Same box, same bake, same calibration, same prompts (window base fixed,
`--prompt-offset 0 --prompt-span` as in prod cert). Sequence:
A/A gate (two baseline runs) → A1 profile run → B run + profile →
A3 drift confirm. Primary series: median per-step decode wall over the
measurement window (warm-drop as shipped).

## Bars and consequences (before any number)

- **Gate G0 (A/A)**: |A/A′ median spread| < 7.5% (half the primary
  bar). FAIL ⇒ box is not measurement-grade: destroy, re-rent, restart.
  No verdict may quote a number from a box that failed G0.
- **Gate G1 (identity)**: `generated_tokens` bit-identical across
  A1/B/A3. FAIL ⇒ the diet changed semantics: **REFUTED**, revert flag
  default, file the divergence; no performance claim may be made.
- **H1 (attribution, measured from A1's stacked profile)**: ≥ 50% of
  per-step `aten::nonzero` calls attribute to
  `hot_residency.py::forward` / `hybrid.py::_cold_contrib` frames.
  FAIL ⇒ STOP-AND-AMEND: the registered edits aim at the wrong sites;
  no B run until a revised prereg names the real ones.
- **H2 (mechanism)**: arm B's profiler window shows `aten::nonzero`
  ≤ 60/step (from 174) AND total (`copy_` + `.to` + `index_select`)
  calls reduced ≥ 15% vs A1's window.
- **H3 (primary)**: median step wall improves ≥ 15% in B vs
  min(A1, A3).
  - H2 pass ∧ H3 pass ⇒ **CERTIFIED**: flag flips default-on in a
    follow-up PR; T5b registration decided by remaining ladder math.
  - H2 pass ∧ H3 in [8%, 15%) ⇒ **PARTIAL**: ship default-on only if
    no op-count regression anywhere in the table; record honestly.
  - H2 pass ∧ H3 < 8% ⇒ **REFUTED** (syncs were not the cost): revert;
    the ladder re-points at host busy-work (T5b restructure) instead.
  - H2 fail ∧ H3 pass ⇒ **CERTIFIED-WITH-OPEN-MECHANISM**: the wall win
    is real but mis-explained; ship, and file the attribution gap.
  - H2 fail ∧ H3 fail ⇒ **REFUTED**: revert.
- **Reported, no bar**: A1 vs a side run with the amort instrument ON
  (the historical bench shape) — quantifies instrument tax once, so
  every prior campaign number can be read against production shape.

## Verdict calculator

`t5_verdict.py` beside this file; self-tested in both directions
(synthetic pass and synthetic fail fixtures) before the box is rented.
S_STEPS (measurement window) = as shipped in step_decomp defaults;
medians computed by the calculator from the rep JSON, never by eye.
