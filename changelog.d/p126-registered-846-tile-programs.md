### P126 registered (#846): the one-launch cumsum tile table split over P programs on SC2e's 64-row step, P ∈ {1, 4, 8}, tokens identical (bench and tests only)

- **Why.** P122 read the one-program chunked table at 2.51 ms of Qwen3-30B-A3B's 64-row eager step. grouped-nf4-gemm
  #524 splits it over P programs, giving the same integers, and #1433 passes `programs=P` under
  `E4B_INT4_TILE_PROGRAMS=P`.
- **The box** (`bench/p126/p126_box.py`):
  - profiles the eager 64-row step at P = 1, 4 and 8;
  - for each candidate, times two interleaved blocks against P = 1, as in P124's Amendment 1 (one runner per setting,
    both alive, strict alternation, both orders; 256 timed and 32 traced steps of each);
  - runs a mutant that must move the tokens.
- **The rule** (`p126_reduce.py`, 35 self-test cases): VOID on any engagement, determinism or mutant fault;
  TOKENS_DIFFER on any token; NOISY when a candidate's blocks differ by more than 1.5 %; DEFAULT_ON_4 or DEFAULT_ON_8
  for the better candidate whose block ratios are both ≤ 0.98.
- **Budget.** Proof on Granite (guard 0.75 h), reading guard 1.5 h, lane ceiling $3.00.
