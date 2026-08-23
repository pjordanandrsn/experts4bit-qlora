# PREREG — regime-split conversion constants (input (b), and (c) by construction)

Registered before any measurement. The second re-derivation input from
[RESULTS-slotvalue.md](RESULTS-slotvalue.md):

> (b) regime-split conversion constants (dense-slope + fixed-cost terms
> — the cell model's own H_cell form, applied at serving shapes)
> replacing the flat 58

and (c) folds in: the constants are fitted on the measured host, so the
fit IS the host re-derivation; portability to another host class is a
re-run of this same instrument, not a new registration.

## Model under test

Per-step DRAM-tier time `T = c_fix · C + c_u · U`, where `U` =
DRAM uniques/step and `C` = layers-with-any-DRAM-work/step (a new
engine counter, `dram_steps`, billed once per layer per step with DRAM
work — the per-call fixed cost's true multiplicity). Bracket form:
`ΔT = c_fix·ΔC + c_u·ΔU`. Dense brackets have ΔC ≈ 0 (every layer
still busy) and isolate `c_u`; sparse brackets empty whole layers and
carry the fixed term — the structure that made the flat 58 fail in
opposite directions at opposite ends of the slot-value sweep.

## Design

* 9-point ladder `vram_gb ∈ {7.0, 8.0, 9.0, 9.5, 10.0, 10.4, 10.75,
  11.1, 11.5}` → 8 brackets spanning dense→sparse. Two passes,
  descending then ascending; fresh subprocess per point; measured-
  routing profile for placements (one profile pass first); default
  full-corpus window for profile and sweep alike — the scored quantity
  is the time conversion GIVEN measured ΔU/ΔC, so rate-transfer
  uncertainty (input (a)'s certified u_b) is deliberately out of frame.
* Operating config as before: Qwen3-30B-A3B, B = 16, gen 48, chunk 512,
  pool 32 / torch-intraop 8, NVMe must stay empty.
* **A/A gate** on the dram bucket (times jitter; uniques do not):
  median pass noise ≤ 5%, worst ≤ 10%, else destroy and re-hunt — up
  to 3 hosts, then UNRUNNABLE.

## Bars ([regime_verdict.py](regime_verdict.py), self-tested on three
worlds: planted (95, 26) certifies with constants recovered; a flat
c_u=58 world is UNINFORMATIVE; a quadratic world with regime signal is
REFUTED)

* Fit `(c_fix, c_u)` by least squares on the ODD brackets (1,3,5,7);
  score the EVEN brackets (2,4,6,8):
  `|ΔT − (c_fix·ΔC + c_u·ΔU)| ≤ max(20%, 3 × pass-spread)` on ≥ 3 of 4.
* **The flat-58 null must miss ≥ 2 scored brackets at the SAME
  allowance** — "fail where it failed before" — else UNINFORMATIVE
  (which would contradict the slot-value receipts and triggers an
  investigation, not a re-run).
* Physicality (`c_u > 0, c_fix ≥ 0`) is checked only after the null
  gate, so a noise-driven coefficient in a flat world cannot preempt
  the honest UNINFORMATIVE. R² over all 8 brackets is reported.

## Deliverable on CERTIFIED

`(c_fix, c_u)` for this host class, and the line's closing pricing
form with input (a) composed in:
`ΔT(bracket) = (c_fix·ΔC + c_u·ΔU) × (1 ± u_b)` — the offline
slot-value formula with both its conversion and its rate uncertainty
measured, replacing the refuted `P × 58`.

## Hard stop

One scored sweep; 3 gate hosts; environment pin as before. ≈ $0.70.
