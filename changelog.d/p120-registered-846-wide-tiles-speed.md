### P120 registered (#846): does `E4B_INT4_WIDE_TILES=1` make SC2e's 64-row decode step faster with identical tokens? Two SC2e rows join the claims register (bench, tests and docs only)

- **Why.** P119 read the chained tile-table builder at 1.39 ms (8.9 %) of the 64-row step in its uniquely named kernels
  alone. grouped-nf4-gemm #515 builds the same integers in one launch up to 1,024 rows, and #1357 routes calls of
  257–1,024 rows to it under `E4B_INT4_WIDE_TILES=1` (opt-in).
- **The box** (`bench/p120/p120_box.py`): SC2e's int4 stack built eager with one slot. P119's 64-row eager bracket
  profiled with the knob 0 and 1 (the premise and the engagement). Four captured-graph arms in ABBA order, 256 timed
  bucket-64 steps each with every token recorded. A mutant that shifts the table's expert ids, so the token gate can
  fail.
- **The rule** (`bench/p120/p120_reduce.py`, 30 self-test cases), every bound a ratio within the box:
  - `DEFAULT_ON` iff both ABBA ratios are ≤ 0.98 and every token is identical;
  - `NOISY` above 1.5 % same-setting disagreement;
  - `PREMISE_ABSENT` below a 5 % builder share;
  - otherwise `NO_GAIN` or `SLOWER`.
- **The premise on the card** before any fetch: the decode-graph bucket tests, and grouped-nf4-gemm's cumsum-rank
  tests compiled for the card.
- **Budget:** proof on Granite (guard 0.75 h); reading guard 1.5 h; lane ceiling $3.00.
- **Claims register:** `e4b.serve.sc2e.64-slots-default-buckets.qwen3.5090.2026-10-07` (ceiling 8 req/s) and
  `e4b.serve.sc2e.64-slots-buckets-auto.qwen3.5090.2026-10-07` (ceiling 12 req/s), from SC2e's committed receipts.
