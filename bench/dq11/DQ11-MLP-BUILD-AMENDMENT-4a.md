# DQ11 MLP-adapter BUILD — amendment 4a: clarification of (b), before data (#1083)

The maintainer ruled this on 2026-10-20 (RED), before any `dq11-mlp-phase1-a2000-2` data. It came from the harness
review (e4b#1873). Amendment 4's merged text is not edited; this file governs where the two differ. Everything else in
Amendment 4 stands.

## 1. Too few unfused paths: the stricter reading

Amendment 4's "fewer than two paths in U" disagreed with its own parenthetical ("only today's path plus at most one
draw", which is |U| ≤ 2). The stricter reading governs: **with |U| ≤ 2, (b) reads `INDETERMINATE`.** Today's path plus a
single draw gives each pseudo-candidate a band of one other path, and that cannot calibrate anything.

## 2. The false-reject bound is 1/(|U| + 1)

The false-reject bound for a neutral fused path is **P(FAIL | neutral) ≤ 1/(|U| + 1)**, where U is today's path plus the
post-(a) draws. "1/17 unconditionally" held only when all 16 paths survive (a). The record reports |U| and the bound it
implies.

## 3. A floor on that bound before (b) can FAIL

**A FAIL from (b) needs 1/(|U| + 1) ≤ 1/10, that is |U| ≥ 9.** With |U| < 9, a (b) failure reads `INDETERMINATE: bar
too weak at |U| = n`, never `PHASE1_FAIL`. This keeps Amendment 4's intent that `INDETERMINATE` means "the bar can't
judge". It only turns would-be FAILs into INDETERMINATE and never creates a PASS. At the predicted |U| = 16 (only R2
lands), it changes nothing.

## The harness

`outcome_a4` reads, in order:
1. |U| < 3: `INDETERMINATE`.
2. Any calibration failure: `INDETERMINATE`.
3. A (b) failure with |U| < 9: `INDETERMINATE` ("bar too weak").
4. A (b) failure with |U| ≥ 9: `PHASE1_FAIL`.
5. Otherwise: `PHASE1_PASS`, with its named tensors.

`read_layers_a4` records `u_paths` (|U|) and `false_reject_bound` (1/(|U| + 1)).

In (c) 3, the harness takes D over each ordered pair: D(P, Q) = ‖B_P − B_Q‖ / ‖B_Q‖, and P's score is its mean D(P, Q).
So the null's 2× consistency check runs over the six ordered today-vs-today distances rather than three. That is
stricter only: it can make KNOB_OFF `INDETERMINATE` more often, never a PASS.

The tests cover |U| = 2 →
`INDETERMINATE`, |U| = 8 with the fused path over the margin → `INDETERMINATE`, and |U| = 9 → `PHASE1_FAIL`. A mutant
moves each threshold, and each is killed.
