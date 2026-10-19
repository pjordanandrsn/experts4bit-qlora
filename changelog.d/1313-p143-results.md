### Added

- **P143's record** (#1313), the paged FP8 attention decode's split cap: `bench/p143/RESULTS-p143.md`, the public
  receipt copies of all four launches under `bench/p143/receipts/`, and the claim `e4b.serve.p143.attn-split-cap.qwen3.5090.2026-10-19`.
  - **Step 1** read GO_FIXED.
  - **Amendment 2** read NO_FLIP (g1 1.0297 under a noisy self-pair).
  - **Amendment 3**, the binding read, is FLIP: one request is 1.044x as fast on an RTX 5090, 16 requests are
    unchanged, and SANE is exact.
  - **Total:** $0.922.
  - **Default:** this record changes none. The default moves with the flip PR.
