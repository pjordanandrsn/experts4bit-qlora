# HP2 (DRAFT): e4b vs Unsloth held-out parity, with the instrument gated at the mean level

**Status: DRAFT, prospective.** This is $0 design under the work-well task `heldout-parity-mean-bias` (mode `prospective`,
authority RED = maintainer). It is **not registered**: nothing here authorizes a box or spend. The maintainer's RED ruling decides
whether to register it, revise it or park it. Registration would then open a separate well task with its own envelope. HP2 is a new
registration, **not a rerun of HP1**, and none of HP1's draws are scored data here. The reason is HP1's closed finding: **a per-row
|gap| tolerance cannot certify a mean comparison** (`bench/hp1/RESULTS-HP1.md`). A new instrument rule therefore needs a new
registration, with its bar fixed before any data.

## The question

On Qwen3-30B-A3B @ `ad44e777` (one RTX 5090), is e4b's held-out loss, read by one evaluator, at **parity** with Unsloth
2026.9.14 (cu130, grouped_mm), or **better** or **worse**? This is the same field recipe as HP1: 200 steps, r = alpha = 16, the
registered Alpaca rows, 48 held-out rows.

- **e4b's arm is K00** (shipped settings: N(0,1/r) init, native adapter dtype). HP1's knob block never ran, so there is no selected
  cell. K00 is the one setting a user gets.
- **axolotl is out.** HP1 Amendment 3 dropped it after its in-loop and exported reads disagreed (`hp1k-5090-8`). Why is open on
  TC1 #1767 (AXO1767). axolotl re-enters only by an amendment, after #1767 shows the gap was our export's and the export is fixed.

## What HP1 taught (and what HP2 changes because of it)

HP1 closed under its Amendment 4 (`bench/hp1/RESULTS-HP1.md`):
- The 48-layer calibration (`hp1cal-5090-2`) measured 48 per-row zero-adapter gaps (evaluator minus e4b's in-loop step 0).
- Per row the gaps were large: mean |gap| 0.021957, sd 0.0304, max 0.08298.
- Their **mean** was small: −0.00132, with a standard error over the 48 rows of 0.00438, so a 95 % CI of [−0.0099, +0.0073].
- HP1's own early check on its 48 held-out rows read +0.006999 (`hp1k-5090-9`). That is consistent with the calibration's mean
  (z = 1.34).

The recorded lesson, verbatim: *the tolerance used per-row |gap| while the comparisons are of means; the signed mean gap was
−0.0013.*

HP1's verdict compares row MEANS of two frameworks' adapters, both read by the same evaluator on the same rows. A per-row gap common to
both arms cancels in that paired difference. What can move the verdict is a gap that **differs between the two frameworks**: the
evaluator reading e4b's base and forward one way and Unsloth's another. So HP2 gates that **differential mean bias**, with its interval.
It REPORTS per-row |gap| but never gates on it.

## The instrument statistics (registered before any data)

All reads are from the one evaluator (`bench/hp1/hp1_eval.py`, as merged), on each framework's own exported base, matched by digest.

1. **Base differential, row-paired (`δ0`).** On R calibration rows, for each row r:
   - g_e4b(r) = evaluator zero-adapter on e4b's base − e4b's in-loop step-0 loss;
   - g_uns(r) = the same on Unsloth's base and Unsloth's in-loop step 0;
   - d(r) = g_e4b(r) − g_uns(r).

   δ0 = mean_r d(r), with SE_δ0 = sd_r(d) / √R.
2. **Trained differential, seed-paired (`δN`).** For each confirmation seed s and each framework:
   - ρ_f(s) = evaluator row mean of f's exported adapter − f's in-loop step-N row mean (HP1's 6(b) quantity, signed);
   - e(s) = ρ_e4b(s) − ρ_uns(s).

   δN = mean_s e(s), with its 90 % t-interval (df = seeds − 1), as HP1's verdict uses.
3. **Reported, never gated:** per-row |gap| (mean, sd, max) for each base; each framework's signed mean bias b_f = mean_r g_f(r);
   σ_seed.

## The gates, each with its null

Each gated statistic has a null, its value under no difference between the two frameworks' paths through the evaluator:
- **δ0 = 0:** the evaluator reads e4b's base and Unsloth's base with the same mean offset from each framework's own forward.
- **δN = 0:** the same holds for the trained adapters' round trips.

A gate passes when the interval around the null-centred statistic is narrow enough, not when the point estimate happens to be small.

- **G1, pre-scoring (after the calibration phase, before any confirmation arm).** |δ0| + 2 × SE_δ0 ≤ 0.01, half the 0.02 margin
  cap, which no σ_seed could rescue. Above it, HP2 CLOSES before spending on the confirmation arms.
- **G2, at the verdict.** For the realized margin m = min(max(σ_seed, 0.005), 0.02), the δN interval must lie within ±m/2, and
  |δ0| + 2 × SE_δ0 ≤ m/2. Otherwise the headline CLOSES: no verdict is read.
- **The zero-adapter and 6(b) VOID rules apply at the mean level.** A run VOIDs if |row-mean gap| exceeds 3 × the calibration's
  SE for that base (and 6(b) at the same 3 × SE on the round trip). That replaces the 2-layer 0.005 / 0.02 constants. Each bound is
  recorded in the evaluator output (`zero_adapter_tol`, as Amendment 4 added).

## Rows and power

- **Calibration rows.** R = 148 rows of `tc1_arm`'s held-out pool, rows 97–244 (`hp1_caltokens.py`'s rule, extended). They are
  disjoint from HP1's 48 held-out rows, from `hp1cal-5090-2`'s rows 49–96 and from every train row. The pool's first 48 must still
  reproduce HP1's held-out rows byte-for-byte.
- **Why 148.** At HP1's per-row sd of 0.030, SE = 0.030 / √148 = 0.0025, so G1 can pass if |δ0| is near 0. The pairing in d(r)
  should shrink the sd when the two frameworks' per-row gaps share their row-to-row structure (same template, same rows). That is
  measured, not assumed: if sd_r(d) ≥ 0.030, G1 closes on the numbers. This is the calibration that can fail.
- **Scored rows.** HP1's registered 48 held-out rows. HP1 read them only in-loop, for K00, with no competitor and no comparison.
- **Seeds.** 3410–3412, paired by `order_sha`, as HP1 registered for confirmation. The verdict bands are HP1's: BEATS, BEHIND,
  PARITY (TOST within ±m) or INCONCLUSIVE.

## The verdict, and how a null read is worded

The headline statistic is Δ = mean over seeds of (e4b K00 − Unsloth) held-out row means, both read by the evaluator. Its null is
Δ = 0: no difference in held-out loss. The bands are fixed above. Their wording in every RESULTS line is fixed here:
- **BEATS / BEHIND:** the 90 % interval excludes 0 on that side and |Δ̄| > σ_seed. Stated with Δ̄, the interval and σ_seed.
- **PARITY (TOST within ±m):** worded as **"no measurable difference at margin m: Δ̄ with its 90 % interval [lo, hi] inside
  ±m"**. Never "equal", "matches" or "the same".
- **INCONCLUSIVE:** worded as **"no measurable effect, with its interval: Δ̄, [lo, hi]"**, neither inside ±m nor clear of 0.
- **A gate closure (G1 / G2):** worded as "the instrument could not read margin m", with δ0 / δN and their intervals. Never as a
  statement about the frameworks.

## The box (one; each phase gated)

One RTX 5090, under the registered host rule (driver ≥ 580, RAM ≥ 98 GB, disk ≥ 220 GB: tc1_run's 200 GB floor + 20). The guard is
6 h.

| Phase | Work | Estimate |
|---|---|---|
| A | installs (venv-e4b, Unsloth cu130; no axolotl venv), fetch, the box-side shakedown (e4b + Unsloth only) | ~0.75 h |
| B | calibration: one e4b K00 step and one Unsloth step on the 148 rows, both bases exported, the evaluator's zero-adapter pass on both → δ0, **G1** | ~0.75 h |
| C | confirmation: e4b K00 and Unsloth on seeds 3410–3412 (6 runs), exports | ~1.6 h |
| D | the evaluator's pass on the 6 runs → δN, the verdict with **G2** | ~0.6 h |

That totals about 3.7 h. If G1 closes in phase B, the box stops there, at about 1.5 h.

Worst at the guard: 6 h × $0.85 + 90 GB × $0.011 = **$6.09** on Vast (RunPod: 6 × $1.10 = $6.60). One box-started attempt and
one pre-create spare. HP1's billed total so far is $3.898.

## Predictions (scored HELD / FALSIFIED / UNTESTED)

- **H1:** G1 passes. The mean bias is near 0 for both frameworks (HP1's e4b calibration mean was −0.0013), and the row pairing
  shrinks sd_r(d) below 0.030.
- **H2:** K00 vs Unsloth reads PARITY or INCONCLUSIVE. BEATS is not predicted (TC1b's matched pair differed by 1e-4).

## For the maintainer's ruling (open questions)

1. Is 3 × SE the right VOID bound at the mean level, or 2 × SE? (G1 already uses 2 × SE.)
2. Is one box with an internal G1 stop acceptable, or should the calibration be its own box, as in HP1 Amendment 4?
3. Should HP2 wait for #1767's outcome, so that axolotl could be in scope from the start?
