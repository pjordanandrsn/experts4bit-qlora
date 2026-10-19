### Changed

- `E4B_ATTN_SPLIT_CAP` defaults to **16 on `qwen3_moe`** (#1313, lane P143). The paged FP8 attention decode now
  runs at most 16 splits; the library planned 32 from the block table's capacity.
  - **Read:** P143 Amendment 3 (#1788, `p143a3-5090-1`), the binding served read on an RTX 5090: one request is
    **1.044x** as fast (the smallest of four ABBA pairs, 1.0442-1.0448), and 16 requests are unchanged (0.992).
    Quality is exact: at contexts <= 1024 the cap is bitwise-identical to the plan.
  - **Earlier read:** Amendment 2's NO_FLIP (#1778, g1 1.0297 under a noisy self-pair) is on the record.
  - **Scope:** only buckets 1, 2 and 4 move. Every other family keeps the library's plan, and Mixtral waits for a
    read of its own.
  - **Measured basis:** Qwen3-30B-A3B on an RTX 5090 (170 SMs). Other `qwen3_moe` checkpoints (e.g. Qwen3-235B-A22B)
    and other cards take the default by family without a read of their own. The plan is SM-scaled, so the cap moves
    B 1-6 on a PRO 6000 and B 1-4 on a 4090.
  - **Way back:** `E4B_ATTN_SPLIT_CAP=0`. The claim is `e4b.serve.p143.attn-split-cap.qwen3.5090.2026-10-19`.
