### Changed

- `E4B_INT8_HEAD_BW` defaults to **1 on `qwen3_moe` at Qwen3-30B-A3B's head shape** (151936 x 2048; #1313, lane P153).
  The lm_head serves from its per-row int8 grid, and its single-row decode GEMV runs through grouped-nf4-gemm K36's
  `gemv_i8_bw`.
  - **Read:** P153 (`p153-5090-1`) on an RTX 5090 read FLIP. One request is **1.052x** as fast: the smallest of four
    ABBA pairs, 1.0520-1.0530, a 0.189 ms step saving.
  - **Outputs change:** the head serves from its int8 grid at every batch size, so logits differ from the bf16 head.
    P153's W1 tokens matched over 160, and 12 of 16 W16 rows diverged by 160, inside P150's quality gates.
    `E4B_INT8_HEAD_BW=0` restores bf16-head output.
  - **Sixteen requests:** 0.992, an observed cost not attributed. The two candidates are the resident int8 copy and
    the different tokens the rounded head decodes at 16 rows.
  - **Quality:** read on Qwen3-30B-A3B @ad44e777 (P150's GO at both of P125's gates; P153's SANE bias +0.0022, argmax
    0.989). Same-shape checkpoints inherit the default unread. `E4B_INT8_HEAD_BW=0` opts out.
  - **Scope:** the default is gated on the head's shape, the one K36 has a tuned plan for. Other heads (Qwen3-235B-A22B's
    151936 x 4096) and every other family stay off until a read of their own.
  - **Requirement:** the default needs grouped-nf4-gemm >= 0.48.0; below it the default resolves off (recorded in
    info.int8_head_bw); the next release raises the [fast] floor.
  - **Never a crash:** a head the swap refuses (tied, biased, an int4 head, not bf16) and a grouped-nf4-gemm without
    K36 resolve the default off, with the reason in `/health`'s `levers.int8_head_bw.source`. An explicit `1` still
    refuses them.
  - **Memory:** with `E4B_PAGED_MAX_SEQS=auto`, the copy's bytes (N x K + 4N, 0.29 GiB at this head) enter the
    footprint the width is sized against whenever the knob resolves on. Explicit widths are unchanged.
  - The claim is `e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19`.
