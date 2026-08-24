# RESULTS — T5b Phase A: the certified-point host bill, decomposed

Run 2026-08-24 against `PREREG-t5b-host-decomposition.md` +
`AMENDMENT-t5b-h-a.md`, same box as the B1 cycle (destroyed + verified
zero). Receipts in `receipts-t5b/`; verdict by the amended
`t5b_verdict.py` (self-tested, 8 branches, including the guard that
the pre-amendment instrument still STOPs on the same numbers).

## Gates

- **G0**: A/A spread 0.24% (141.0 / 140.7 ms) — the tightest box of
  the campaign.
- **GS (shape-gate, first live firing)**: PASS — step 141.0 ∈
  [115, 165], attention_host 46.0 ≤ 55. The #220 default flip held;
  the stale per-seq shape (215.8 / 124.6) would have aborted here.

## The decomposition (B=16, certified point, bracketed attribution run)

| region | ms/step | share |
|---|---|---|
| MoE experts host | **79.0** | **44.8%** |
| attention host | 58.7 | 33.3% |
| router/top-k | 6.4 | 3.6% |
| lm_head | 0.06 | 0.0% |
| sched + drain | 1.2 | 0.7% |
| residual (dense/norms/glue) | 30.9 | 17.5% |

Coverage 82.5% ≥ 80 with a single region ≥ 25% ⇒ **PHASE-A-PASS**.
(First attribution run, without the block region, read coverage 79.37%
and the calculator returned STOP — receipts kept; the amendment
replaced the instrument, never the bar, and the amended calculator
still STOPs the old receipts.)

## Phase B registration (fixed by the prereg's bar formula)

**Target: the MoE experts host path — 44.8% of the certified step.
Phase B wall bar = 22.4% (share/2), partial floor 6%, REFUTED ⇒
revert.** Phase B edits are to be registered by amendment BEFORE any B
arm runs, per the prereg.

The B1 campaign's receipts (same day, same box) sharpen the target for
free: at B=1 the pipelined executor runs the SAME experts math in
12.8 ms of host bracket where the hybrid takes 32.8-34.0 — the
per-layer hybrid machinery, not the arithmetic, is the cost. The
BRANCH-2 collapse fast path and T5b Phase B are converging on the same
address from two independent preregistrations.
