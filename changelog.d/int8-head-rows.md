### Added

- `E4B_INT8_HEAD_ROWS` (opt-in, `1` to `8`; unset is `1`, today's route): with the lm_head on its int8 grid, calls
  of 2 to the set number of rows run grouped-nf4-gemm K37's `dense_i8.linear_i8` over the head's int8 codes and scales,
  instead of `F.linear` on their bf16 rounding (half the bytes). One row keeps K36, and more rows keep `F.linear`.
  Outputs at those rows change within bf16's output rounding of the same grid. Refused at startup when the head is not
  on its int8 grid or the package lacks K37; unset imports nothing. The record is `levers.int8_head_bw.rows`, with the
  kernel per row count in its engagement record. No speed or quality claim: the design is
  `bench/dense-small-m/DESIGN-K2.md`, and its readings come first.
