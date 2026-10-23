### P228 registered: K1's selection and microbench, a pinned cuBLASLt GEMM for o_proj and the router at 2..8 rows (serving roadmap lever K)

A registration, no box yet. On one RTX 5090 at Qwen3-30B-A3B's `o_proj` (2048 x 4096, 48 copies) and router gate
(128 x 2048, 384 copies), the lane selects, per M in 2, 4 and 8, the cuBLASLt configuration `E4B_DENSE_LT_PIN`
(e4b#2019) should pin.
- **The selection:** from cuBLASLt's own heuristic list. Each candidate is checked, must be bitwise run to run and
  within 1.5e-2 of `F.linear`, and is timed. The fastest is chosen, with a scheme-2 preference within 1.02x.
- **The reading:** e4b's installed pinned path (P) against the served `F.linear` (D) at the same M.
- **The rule, per site:** PASS where P/D's upper bound, raw and clock-adjusted, is at most 0.85 at both M = 2 and M = 4.
- **The premise:** K1's GPU and CPU suites on the 5090 itself (37 passed).
- **What follows:** the verdict's `pin_table` feeds the code PR that fills `LT_PINS`.
- **Files:** `bench/p228/` (PREREG, bench, reducer with its self-test, runner, drive, staged pins) and
  `tests/test_p228.py` (16 tests, 9 reducer mutants).
