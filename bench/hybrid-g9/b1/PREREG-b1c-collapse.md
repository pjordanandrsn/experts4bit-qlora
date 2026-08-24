# PREREG — B1c: the all-resident collapse fast path

Registered 2026-08-24, before any measurement. The optimization the B1
decomposition licensed (BRANCH-2, `RESULTS-b1-decomposition.md`): the
hybrid executor costs 25.4% of the all-resident B=1 step even when
nothing is offloaded, and the entire R0→R1 gap sits in the experts-
dispatch bracket (32.8 → 12.8 ms). The collapse removes that machinery
from the token-critical path when — and only when — the placement is
all-VRAM.

## The mechanism (registered edits)

Engine flag `collapse_resident: bool = False` (constructor-plumbed via
`enable_hybrid_tier`, mechanism-named, no model names). In
`_HotResidency.forward`, before any dispatch algebra:

- `_all_hot()` — a PLACEMENT-STATIC cached predicate: every expert hot
  AND `g2h` is the identity permutation (asserted once at cache time,
  device-side; any future hot-stack reordering falls back to the
  baseline path rather than mis-indexing). Invalidated by
  `swap_expert` alongside `_cold_static`. No per-step data inspection.
- When true: `_forward_collapsed` — cached `row_token`, one full-rows
  activation gather, `_fused_over_stack(xr, flat, h_*)` with `flat`
  passed DIRECTLY as local ids (identity map), router weights applied
  via `reshape(-1)` and a `view(T, k, H).sum(1)` reduction. No
  `is_hot` gather, no partition, no `nonzero`, no cold-branch checks,
  no `index_put_`, no zeros allocation.
- Arithmetic-identity claim: same kernel, same row order, same
  weight values per cell, same fixed-order reduction as the baseline
  all-hot path ⇒ bitwise-equal outputs. Gated on-box by G1 (now a
  REAL gate, post-#223).
- Subset placements are structurally untouched: the predicate is
  placement-static and false, so the baseline path runs unchanged.

## Arms (B=1, reference EPYC + RTX 5090 class, pinned B1 command lines)

- **C0**: `--placement-override all-vram` (the B1 R0 arm re-measured
  on this box) — baseline.
- **C1**: C0 + `--collapse`.
- **R1**: `--engine pipelined --chunk 1` — the reference bar from B1.
- Two timed runs per arm (A/A), one profiled run for C0 and C1
  (`--host-brackets --region-ops-out --torch-profile-out`).
- **B=16 sanity**: one run at the certified command line (solver
  placement, subset) WITH `--collapse` — the flag must be inert:
  GS shape-gate must pass and the rep must show the flag on. This is
  the non-regression proof for the certified point.

## Gates and bars (before any number)

- **G0**: per-arm A/A spread < 7.5%.
- **GS**: the B=16 sanity run passes the certified band (step ∈
  [115, 165], attn ≤ 55) with the flag ON.
- **G1**: C0 ≡ C1 tokens bitwise (the arithmetic-identity claim made
  falsifiable). C1 vs R1 recorded (expected identical, as R0≡R1 was).
  FAIL ⇒ REFUTED regardless of speed; revert.
- **H-C (primary)**: C1 median step improves **≥ 15%** vs C0
  (recovering ≥ ~60% of the measured 25.4% abstraction tax).
  PARTIAL band [8%, 15%); < 8% ⇒ REFUTED, revert.
- **H-M (mechanism)**: C1's profiled experts bracket (`moe_host`)
  ≤ 60% of C0's (from 32.8 ms toward pipelined's 12.8).
- **Reported, no bar**: gap-closed fraction (C0−C1)/(C0−R1); C1 vs R1
  residual gap; kernel occupancy per arm (the #216 convention with the
  region-row exclusion).
- Consequences: H-C pass ∧ H-M pass ∧ G1 pass ⇒ **CERTIFIED — the
  default flips ON in the same RESULTS PR** (the #220 lesson: a
  certified flag left opt-in poisons the next campaign's baselines).
  H-C pass ∧ H-M fail ⇒ CERTIFIED-WITH-OPEN-MECHANISM (wall win real,
  bracket attribution mis-modeled; ship + file). H-C partial ⇒ ship
  only with zero regressions (B=16 sanity + brackets). H-C fail ⇒
  REFUTED: revert, and the ladder falls through to the M=1 executor
  branch (BRANCH-3-HOST work) directly.

## Verdict calculator

`b1c_verdict.py`, self-tested both directions before the box; receipts
in `receipts-b1c/`. Unit tests pin the predicate algebra (all-hot ⇒
identity g2h on the real construction path; subset ⇒ predicate false)
in CI; the bitwise forward equality is the on-box G1.
