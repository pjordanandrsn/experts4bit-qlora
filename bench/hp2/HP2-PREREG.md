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
  TC1 #1767 (AXO1767). AXO1767 has since read KERNELS (re-derived by the maintainer): axolotl's exported adapter does **not**
  score like its in-loop one. axolotl therefore stays out; it re-enters only by an amendment, after its export is fixed.

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
   σ_seed; **Δ_base** = the evaluator's zero-adapter row mean on e4b's base − the same on Unsloth's base (the untrained floor the
   headline Δ is read against); and each base's zero-adapter gap on the 48 scored rows (see the VOID rules below).

## The gates, each with its null

Each gated statistic has a null, its value under no difference between the two frameworks' paths through the evaluator:
- **δ0 = 0:** the evaluator reads e4b's base and Unsloth's base with the same mean offset from each framework's own forward.
- **δN = 0:** the same holds for the trained adapters' round trips.

A gate passes when the interval around the null-centred statistic is narrow enough, not when the point estimate happens to be small.

- **G1, pre-scoring (after the calibration phase, before any confirmation arm), two tiers, both recorded in `hp2/g1.json`:**
  - **CLOSE:** |δ0| + 2 × SE_δ0 > 0.01, half the 0.02 margin cap, which no σ_seed could rescue.
  - **STOP:** otherwise, if **m_need = 2 × (|δ0| + 2 × SE_δ0)** > **0.005**, the expected margin (the σ_seed floor; HP1/TC1 seed
    spreads sit below it). m_need is the smallest margin G2's δ0 condition could pass at. Above the expected margin, G2 would
    close on δ0 alone at any σ_seed we expect, so the box does not spend on the confirmation arms to learn that.
  - Either tier exits 4 and the driver ends the draw; the box tears down as on any end. It is a reading, **not a strike**.
  - **The joint condition, stated correctly.** G1 is one inequality in δ0 and SE together, not a bound on each. At sd_r(d) = 0.030
    and R = 300 (SE 0.00173) the CLOSE tier passes iff |δ0| ≤ 0.0065, and the STOP tier cannot pass at any δ0: it needs
    SE_δ0 ≤ 0.00125, so sd_r(d) ≤ 0.0217. With each base's per-row gaps at HP1's sd of 0.030, that is a row-pairing correlation
    **ρ ≥ 0.74** between g_e4b(r) and g_uns(r) (sd_r(d) = 0.030 × √(2(1 − ρ))). At R = 148 it was ρ ≥ 0.87; raising R to the pool's
    limit is what lowers it. ρ is measured on the box, not assumed.
- **G2, at the verdict.** For the realized margin m = min(max(σ_seed, 0.005), 0.02), the δN interval must lie within ±m/2, and
  |δ0| + 2 × SE_δ0 ≤ m/2. Otherwise the headline CLOSES: no verdict is read.
- **6(b) round-trip VOID, centred, at 3 × SE** (maintainer's corrected ruling on this PR). A sound instrument's 48-row mean gap
  sits at that framework's offset b_f, not at 0. So run r of framework f VOIDs iff
  |ρ_f(r) − b̂_f| > 3 × sd_f × √(1/48 + 1/300): three standard errors of a 48-row scored mean about the 300-row estimate of its
  centre. (At sd_f = 0.030 the half-width is 0.0140.) Centre and half-width per framework, rounded up to 1e-6, are written by G1
  to `hp2/roundtrip_bounds.json` and passed to the evaluator (`--roundtrip-bounds`). They replace the 2-layer 0.02 constant for HP2.
- **The zero-adapter check is reported, never a VOID.** The zero-adapter gap is deterministic per base: a rerun on the same base
  and rows returns the same number, so a VOID would only repeat. A miss is a reported fact that δ0 already carries
  (`zero_adapter_outside_tol` in the evaluator output, run with `--zero-report-only`). VOID is kept only for stochastic quantities.
- **Family-wise false-VOID rate, as registered.** The confirmation pass makes 6 VOID checks, one round trip per run. At 3 × SE each
  is about 0.27 % two-sided for a sound instrument, so the family is 1 − (1 − 0.0027)^6 ≈ **1.6 %** (Bonferroni also 1.6 %). At
  2 × SE it would be about 25 %, which is why VOID is not set there.

## Rows and power

- **Calibration rows.** R = 300 rows of `tc1_arm`'s held-out pool, rows 97–396 (`hp1_caltokens.py`'s rule, extended). They are
  disjoint from HP1's 48 held-out rows, from `hp1cal-5090-2`'s rows 49–96 and from every train row. The pool's first 48 must still
  reproduce HP1's held-out rows byte-for-byte.
- **Why 300.** The ruling: raise R as far as the fresh pool allows. 300 rows is what G1's STOP tier needs to pass at a pairing
  ρ ≥ 0.74 rather than 0.87 (see the joint condition). The pairing in d(r) shrinks the sd only when the two frameworks' per-row
  gaps share their row-to-row structure (same template, same rows). That is measured, not assumed: this is the calibration that
  can fail, and it fails on the numbers.
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

## G1 on the box (maintainer ruling on this PR: one box, an internal stop)

- G1 is computed on the box by the registered reducer (`bench/hp2/hp2_reduce.py g1`). It writes `hp2/g1.json` before any
  confirmation arm starts.
- The stop is enforced in code: `g1` exits **4** on a closure, and the driver (`tc1_hp2_family`) ends the draw on that code.
  Nobody decides it at runtime.
- It is tested on δ0 fixtures built from REAL calibration records: `hp1cal-5090-2`'s 48 per-row gaps
  (`tests/fixtures/hp1/eval_cal_hp1cal-5090-2.json`), paired with an Unsloth side that tracks them, is shifted from them, or is
  reordered.
- A G1 closure is a reading ("the instrument could not read margin m"), **not a strike**.

## The box (one; each phase gated)

One RTX 5090, under the registered host rule (driver ≥ 580, RAM ≥ 98 GB, disk ≥ 220 GB: tc1_run's 200 GB floor + 20). The guard is
6 h.

| Phase | Work | Estimate |
|---|---|---|
| A | installs (venv-e4b, Unsloth cu130; no axolotl venv), fetch, the box-side shakedown (e4b + Unsloth only) | ~0.75 h |
| B | calibration: one e4b K00 step and one Unsloth step on the 300 rows, both bases exported, the evaluator's zero-adapter pass on both → δ0, m_need, **G1** | ~0.9 h |
| C | confirmation: e4b K00 and Unsloth on seeds 3410–3412 (6 runs), exports | ~1.6 h |
| D | the evaluator's pass on the 6 runs → δN, the verdict with **G2** | ~0.6 h |

That totals about 3.85 h. If G1 closes or stops in phase B, the driver ends the draw and the **box tears down** there, at about
1.65 h.

Worst at the guard: 6 h × $0.85/h + 90 GB × $0.011 download = **$6.09** on Vast (RunPod: 6 × $1.10 = $6.60). The $0.85/h cap is
on the offer's total hourly price, which **includes storage at the 220 GB disk** (the launcher pins `vast_disk_gb` and prices the
offer at it), so the worst case carries storage. The manifest sets `lane_record.min_disk_gb` = 200 (the harness floor) with
`vast_disk_gb` = 220. One box-started attempt and one pre-create spare. HP1's billed total so far is $3.898.

## Predictions (scored HELD / FALSIFIED / UNTESTED)

- **H1:** G1 passes both tiers. The mean bias is near 0 for both frameworks (HP1's e4b calibration mean was −0.0013), and the row
  pairing reaches ρ ≥ 0.74, so m_need ≤ 0.005.
- **H2:** K00 vs Unsloth reads PARITY or INCONCLUSIVE. BEATS is not predicted (TC1b's matched pair differed by 1e-4).

## Rulings taken (maintainer, on this PR, before registration)

1. VOID at 3 × SE; G1 at 2 × SE (one-sided readability).
2. One box with an internal G1 stop, under the four conditions above.
3. Don't wait for #1767. axolotl joins by its own amendment, if #1767 shows its exported adapter scores like its in-loop one.
4. Corrected ruling after families' review: the 6(b) VOID is centred on b̂_f with half-width 3 × sd_f × √(1/48 + 1/R); the
   zero-adapter check is report-only (deterministic per base); state the joint G1 condition and the ρ requirement; raise R to
   the pool's limit and record m_need, stopping at G1 when it exceeds the expected margin; report Δ_base; carry storage in the
   worst case; the internal G1 stop tears the box down.

## Harness

- `bench/hp2/hp2_reduce.py`: the `g1` and `verdict` subcommands. It reuses `hp1_reduce`'s `paired_reading`, `pooled_sd`,
  `order_problems` and `manifest_problems`.
- `bench/hp1/hp1_caltokens.py`: `--start` / `--count` (HP2: rows 97–396).
- `bench/hp1/hp1_eval.py`: `--roundtrip-bounds` (per-framework centre and half-width, inline or `@file`), `--zero-report-only`.
- `bench/tc1/tc1_run.sh`: the `qwen3hp2` token, the `hp2cal` recipe and `tc1_hp2_family` (phases A–D). `tc1_drive.sh` stages
  `hp2_reduce.py`.
