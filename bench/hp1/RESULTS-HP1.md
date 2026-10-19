# HP1 results: CLOSED under Amendment 4 (no competitor reading)

HP1 asked whether e4b's held-out loss on Qwen3-30B-A3B reaches parity with, or beats, Unsloth and axolotl (`HP1-PREREG.md`). It
**closed before any scored comparison**. The calibration registered in Amendment 4 showed that the one evaluator cannot read HP1's
parity margin at 48 layers. No competitor arm was scored, so this record makes **no parity, BEATS or BEHIND claim** about either
competitor.

## The only datum: e4b K00 (shipped settings), one in-loop curve

| Run | Model | Steps | Held-out loss, in-loop (48 rows) | Trainable | Peak VRAM | Train wall-clock |
|---|---|---|---|---|---|---|
| `hp1k-5090-9` / `hp1_k00_s3407` | Qwen/Qwen3-30B-A3B @ `ad44e777`, 48 layers, NF4 | 200 | 1.91403 (step 0) → 0.86954 (step 200) | 642,514,944 | 23.3 GB | 326 s |

This is e4b's own in-loop read, on one seed (3407), on one RTX 5090. It is **not** an evaluator number: the run's base export
failed the early base check (below), so no evaluator score exists for it. It carries no σ_seed and no comparison.

## Why HP1 closed

1. **`hp1k-5090-9`: the early base check VOIDed the e4b base.** The digest matched its training run (12,723 slots). The STRICT
   zero-adapter check read evaluator 1.921029 against in-loop step 0 1.914030, +0.006999 against the 2-layer bound of 0.005.
2. **Amendment 4 replaced that bound with a calibration that could fail.** `hp1cal-5090-2` ran one e4b K00 step on 48 calibration
   rows (held-out pool rows 49–96, disjoint from HP1's), exported its base, and read per-row zero-adapter gaps. `bench/hp1/hp1_calib.py`
   returned **`CLOSE_SHAPE`**:
   - median |gap| 0.015915; mean |gap| 0.021957; max |gap| 0.08298 (row 39); max/median 5.214; signed mean gap −0.001322;
   - rule 3 (pre-box ceiling) closes it alone: tol = 2 × 0.021957 = **0.043914 > 0.01**;
   - rule 1 (shape) also closes it, marginally: 0.08298 > 5 × 0.015915 and > 0.005.
3. **Independently re-derived.** `calib.json` reproduced byte-identically (sha256 `2cf7ac6c6ec49d3b643464298e3008328d846dc1771b25b3cc3eb95e110638e1`):
   https://github.com/pjordanandrsn/experts4bit-qlora/pull/1772#issuecomment-6257138140

Box 1's selection block (`hp1k-5090-10`) and box 2's confirmation block never ran.

**Lesson (recorded verbatim, not a re-band):** the tolerance used per-row |gap| while the comparisons are of means; the signed mean
gap was −0.0013.

A follow-on held-out-parity read that registers that mean-level bias statistic, with its seed interval, is a prospective idea only.
It is not authorized, and it is not a rerun of HP1.

## Draws and spend

| Run | Outcome | Cost | Note |
|---|---|---|---|
| `hp1k-5090-1` | OK (SHAKEDOWN_FAILED) | $0.943 | shakedown tokens labelled for the wrong fam (fixed, #1735) |
| `hp1k-5090-2` | NOT_RUN | $0.001 | launcher pre-flight host failure (not a strike) |
| `hp1k-5090-3` | OK (SHAKEDOWN_FAILED) | $0.939 | axolotl shake arm offline at load (fixed, #1739; Amendment 3) |
| `hp1k-5090-4`…`-7` | REFUSED before create | $0 | GitHub fetch timeout; `no_such_ask` × 3 |
| `hp1k-5090-8` | OK (SHAKEDOWN_FAILED) | $0.821 | axolotl VOID at the shake evaluator: Amendment 3 fired, axolotl dropped |
| `hp1k-5090-9` | OK (EARLY_BASE_CHECK_FAILED) | $0.552 | the zero-adapter gap above; K00's curve is this draw's |
| `hp1cal-5090-1` | HARNESS_ERROR | $0.008 | the box's 200 GB free-disk floor refused a 150 GB box (not a strike) |
| `hp1cal-5090-2` | OK (`HP1_CALIB=CLOSE_SHAPE`) | $0.634 | the Amendment 4 calibration |
| **Total** | | **$3.898** | envelope $16.50 |

Every receipt is in the receipt store under `receipts/experts4bit-qlora/2026-10-18/` and `2026-10-19/`, by run id.
