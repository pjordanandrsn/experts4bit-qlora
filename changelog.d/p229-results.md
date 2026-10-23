### Added

- P229 results (`bench/p229/RESULTS-p229.md`). `p229-5090-3`, the one reading of PREREG-p229.md (Amendments 1-6) at e4b
  `32fda64d`, read `GO_ROWS_8`. At 2, 4 and 8 rows of Qwen3-30B-A3B's 151936 x 2048 lm_head on its per-row int8 grid, the
  `E4B_INT8_HEAD_ROWS` route (grouped-nf4-gemm K37's `linear_i8` at its shipped plan) runs the head call in a median 0.51
  of `F.linear`'s time on the grid's bf16 rounding (upper bounds 0.5109, 0.5148 and 0.5085; bar 0.70), and within a
  median 1.01-1.03 of the one-row int8 GEMV (bar 1.25). The knob is licensed to its maximum of 8 rows, so `MAX_ROWS` is
  unchanged. As registered, the served read with P153's quality bars comes next, and the default stays off. $0.061. The
  receipt copy is under `bench/p229/receipts/p229-5090-3/`.
