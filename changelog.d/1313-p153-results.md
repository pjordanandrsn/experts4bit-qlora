### Added

- **P153's record** (#1313), D2's served read of the int8 lm_head: `bench/p153/RESULTS-p153.md`, the public receipt
  copy under `bench/p153/receipts/`, and the claim `e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19`.
  - **The read** is FLIP: one request is 1.052x as fast on an RTX 5090, 0.998 of P148's projection. Sixteen requests
    read 0.992, inside the 0.99 line; S1 is below S0 in every W16 pair, an observed cost not yet attributed. SANE is
    inside the bars.
  - **Total:** $0.445.
  - **Default:** this record changes none. `E4B_INT8_HEAD_BW` stays opt-in until a separate default PR.
