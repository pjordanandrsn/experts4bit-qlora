### P229 registered: K2b's microbench, the int8 lm_head at 2, 4 and 8 rows on K37's released `linear_i8` (serving roadmap lever K)

A registration, no box yet. On one RTX 5090, at Qwen3-30B-A3B's lm_head (151936 x 2048, one random head swapped by
e4b's own resolvers), the lane reads the route `E4B_INT8_HEAD_ROWS` (e4b#2024) installs: grouped-nf4-gemm K37's
`linear_i8` at its shipped plan, from the v0.50.0 release.
- **The arms:** the route at M rows (I) against the served path at those rows (`F.linear` on the grid's bf16 rounding,
  D) and against the head's one-row route (K36, K).
- **The rule, per M in 2, 4 and 8:** PASS where I/D's upper bound is at most 0.70 and I/K's at most 1.25, raw and
  clock-adjusted. The verdict is GO_ROWS_N for the largest N with every read M up to it passing (the knob's licensed
  bound), or NO_GO at M = 2, which makes K2a the explicit next decision.
- **The premise:** the route's real kernels compiled on the 5090 (11 passed) and its CPU suite (45 passed).
- **What follows:** the served read with P153's quality bars per bucket, before any default.
- **Files:** `bench/p229/` (PREREG, bench, reducer with its self-test, runner, drive, staged pins) and
  `tests/test_p229.py` (25 tests: 10 reducer mutants, and p229-shake-1's record as the real fixture with five
  departures and a mutant).
