### Changed

- `E4B_INT8_HEAD_ROWS` defaults to **8 on `qwen3_moe` at Qwen3-30B-A3B's head shape** (151936 x 2048; lane P232). The
  int8 lm_head's 2- to 8-row calls run grouped-nf4-gemm K37's `dense_i8.linear_i8` over the head's own codes and scales,
  instead of `F.linear` on the grid's bf16 rounding; one row keeps K36's GEMV, and more rows keep `F.linear`.
  - **Read:** P232 (`p232-5090-1`, `bench/p232/RESULTS-p232.md`) on an RTX 5090 at 575 W read FLIP_ROWS_8, TRIPWIRE
    none. 2, 4 and 8 requests decode **1.0311x, 1.0243x and 1.0165x** as fast as at `1` (the smallest of four ABBA pairs
    at each), after P229's captured microbench of the head call (GO_ROWS_8, `bench/p229/RESULTS-p229.md`).
  - **Outputs change by default at 2 to 8 concurrent requests:** K37 and `F.linear` round differently within bf16's
    output rounding of the same int8 grid, so greedy text can differ (76 of 112 greedy rows matched in P232). One
    request and 9 or more are unchanged. `E4B_INT8_HEAD_ROWS=1` restores the previous output.
  - **Quality:** inside P153's bars at every group size (NLL gap at most 0.0001 nats, argmax 3064-3066 of 3072), read on
    Qwen3-30B-A3B. Same-shape checkpoints inherit the default unread.
  - **Scope:** gated on the head's shape and on the int8 head being on (`E4B_INT8_HEAD_BW` resolved 1). Other heads
    (Qwen3-235B-A22B's 151936 x 4096) and every other family stay at 1 until a read of their own.
  - **Requirement:** K37 ships in grouped-nf4-gemm 0.49.0; below it the default resolves to 1 (recorded). The `[fast]`
    floor is not moved here.
  - **Never a crash:** with the int8 head off, a grouped-nf4-gemm without K37, or speculative decode (`E4B_PAGED_SPEC`,
    whose gate and draft were read with the head on `F.linear`) on, the default resolves to 1 with the reason in
    `/health`'s `levers.int8_head_bw.rows.source`. An explicit `2` to `8` still refuses each of them.
  - The claims are `e4b.serve.p232.int8-head-rows.qwen3.5090.2026-10-24` and
    `e4b.kernel.p229.int8-head-rows.qwen3.5090.2026-10-23`.
