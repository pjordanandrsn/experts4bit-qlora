### Lane K33's runner (#1313): the GNF4_GEMV_BW decode-GEMV bench's box side (bench and tests only)

- **What.** `bench/k33/` drives grouped-nf4-gemm's lane K33 (`kernel/PREREG-k33-nf4-decode-gemv-bw.md`, gnf4 #501) on one
  RTX 5090: K28's runner with the tripwire, the premise and the bench replaced. No model is fetched.
- **The box.** Refusals (card class, disk) come before the install of gnf4 at `GNF4_SHA`. The tripwire proves the installed
  commit carries `_gemv_nf4_bw`, the `bw_*` tally keys and an empty `_BW_SHAPES`, and that `prmt32` is the decode on the
  card. The premise is `kernel/test_nf4_gemv_bw.py` compiled on the card, 27 passed and none skipped (rc 23). Then
  `k33_bench.py` runs from the clone at `GNF4_SHA`. Every GNF4 decode knob starts unset.
- **Tests.** `tests/test_k33_staged_pin.py`: the runner's pin, its shape and order, the exit codes, and the drive's dry run.
