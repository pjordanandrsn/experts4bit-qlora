### P124 registered (#846): the attention projections above 16 rows on the int4 small-M GEMM, speed at 64 and 32 rows and teacher-forced quality (bench and tests only)

- **Why.** In lane P119, `Int4Linear` above 16 rows served a cached bf16 copy with cuBLAS: 2.05 ms of SC2e's 15.64 ms
  64-row step, reading 1.81 GB a step where the int4 grid is 510 MB. grouped-nf4-gemm #522 lets the K16 small-M GEMM
  take a 32- or 64-row tile, and #1410 routes 17–64 rows to it under `E4B_ATTN_INT4_WIDE=1`.
- **The box** (`bench/p124/p124_box.py`):
  - profiles the eager 64- and 32-row steps with the route off and on (P119's bracket);
  - times four captured-graph ABBA arms of 256 steps at each depth, plus 32 traced steps for the GPU busy fraction and
    the peak memory;
  - runs P117's teacher-forced passes: the route off as reference and floor, on as the subjects, two mutants, and a
    captured function check.
- **The rule** (`p124_reduce.py`, 38 self-test cases): VOID on any engagement, determinism, function or mutant fault;
  QUALITY_FAIL on P110's bar; DEFAULT_ON (or DEFAULT_ON_64 / _32) when both ABBA ratios at a depth are ≤ 0.98.
- **Budget.** Proof on Granite with int4 attention (guard 0.75 h), reading guard 2.0 h, lane ceiling $3.00.
