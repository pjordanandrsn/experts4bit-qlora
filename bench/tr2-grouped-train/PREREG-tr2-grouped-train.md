# PREREG — TR2: grouped expert kernels for the training step

Registered 2026-08-25, before any measurement, deriving every bar
from RESULTS-tr1-census (anchor 51.68 s/step; GPU ~8–11% busy;
≥2.92M launches/step; device-work floor 5.65 s/step). Honors the
RESULTS pre-commitment as corrected in review: the win is bounded
above by 9.1×; under the additive launch model (wall ≈ 5.65 s busy +
N × ~15.8 µs) a 2× wall requires a launch cut ≥ ~2.3×.

## Treatment (existing machinery, wired — not invented)

`enable_hybrid_train` already routes expert forward/dgrad through
gnf4's `gemm_4bit_grouped` / `dgrad_4bit_grouped` — one grouped
launch over all of a layer's routed rows instead of the per-expert
bitsandbytes dequant→GEMM→elementwise chain the default trainer runs
(the chain TR1 measured at ≥2.92M launches/step). The treatment is a
small trainer wiring: `TRAIN_ARENA=<arena path>` env → the trainer
calls `enable_hybrid_train(model, arena, manifest)` after load, with
an all-VRAM placement (the census box fits the 30B without offload).
The mechanism itself is the machinery this program certified for
serving; per the standing directive, this is a routing problem, not
permission to invent another GEMM.

## Arms (one box passing the serving-anchor health gate)

1. `base_a`, `base_b` — the default (bnb) trainer, the TR1 recipe
   verbatim (SEQ=192, STEPS=20, GRAD_ACCUM=4, TOKEN_BUDGET=1024
   pinned), census ON. Fresh baselines on the same box; TR1's
   receipts are the registration basis, not the comparison arm.
2. `hyb_a`, `hyb_b` — identical recipe + `TRAIN_ARENA`.
3. `lc_base`, `lc_hyb` — 10-step launch-count probes (the TR1
   profiler config: CUDA-only, active=2); launch count = sum of
   table calls / active steps. Excluded from A/A per TR1 amendment.

## Bars (gain frame off the fresh base median; TR1 anchor class)

- Baseline sanity: base median within ±10% of TR1's 51.68 s anchor,
  else REFUSE (different workload than registered).
- **PASS** — hybrid step cut ≥2× (hyb ≤ base/2) AND the quality
  gate. The launch count is a MODEL CHECK, not a pass gate — the
  claim is wall time, and gating PASS on a launch multiple would
  refuse treatments the model itself says clear 2×
  (bars-follow-the-claim; the 10× first draft was corrected in
  review). Ships `TRAIN_ARENA` as the documented default path for
  arena-holding models.
- **PARTIAL** — cut ≥1.25×; ships opt-in with disclosure.
- **REFUTED** — < 1.25×.
- **Model check (recorded regardless of tier):** the additive model
  predicts wall ≈ busy + launches × host-cost. REPORT each arm's
  implied per-launch host cost; FALSIFIED if wall cut ≥2× with
  launch cut < 2×, or launch cut ≥ 8× with wall cut < 2× — either
  direction breaks the launch-storm account and is recorded in
  RESULTS-tr2.

## Quality gate (learning, not tokens)

The two paths decode NF4 through different layouts (bnb blockwise vs
gnf4 arena), so weights differ in low bits and loss trajectories may
legitimately diverge — bitwise identity is NOT claimed (the F2
lesson, applied in advance). Gates:
- finite loss every timed step, both arms (census receipt);
- hybrid final held-out eval loss ≤ base final + 0.05 absolute;
- hybrid must LEARN: final eval < its own BEFORE eval by ≥80% of the
  base arm's improvement.
- A/A: per-arm pair must pass the amended TR1 composer gates
  (closure, trend, per-step, budgets, loss).

## Calculator

`tr2_verdict.py`, self-tested before any receipt exists; receipts in
`bench/tr2-grouped-train/receipts-tr2/`. REFUSE on: baseline sanity,
any census-composer refusal, missing launch-count probes, quality
gate breach, or an arena/manifest mismatch at enable time (the
wiring must refuse a vacuous treatment, never silently train bnb —
the F2 vacuous-arm rule).
