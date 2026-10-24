### Added

- P232 results (`bench/p232/RESULTS-p232.md`). `p232-5090-1`, the one reading of PREREG-p232.md (Amendments 1-2) at e4b
  `f7589a02`, read `FLIP_ROWS_8`, TRIPWIRE none. On Qwen3-30B-A3B's default server with the lm_head on its per-row int8
  grid, `E4B_INT8_HEAD_ROWS=8` (grouped-nf4-gemm K37's `linear_i8` at 2 to 8 rows, in place of `F.linear` on the grid's
  bf16 rounding) decodes 2, 4 and 8 requests 1.0311x, 1.0243x and 1.0165x as fast (the smallest of four ABBA pairs),
  inside P153's quality bars at every group size (NLL gap at most 0.0001 nats; argmax 0.9974-0.9980). As registered, a
  separate PR may make 8 qwen3_moe's default, citing this read and P229; until then the knob stays opt-in. $1.00. The
  receipt copy is under `bench/p232/receipts/p232-5090-1/`.
