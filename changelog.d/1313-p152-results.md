### Added

- **P152's record** (#1313), stream (B)'s per-name census of Qwen3-30B-A3B's served step at W16 and W1:
  `bench/p152/RESULTS-p152.md` and the public receipt copy under `bench/p152/receipts/`.
  - **The read** is READ. At 16 requests the grouped NF4 small-M expert GEMM takes 71.2 % of the unprofiled 16.50 ms
    step (gate_up 50.9 %, down 20.3 %); at one request the expert GEMVs already run at their isolated speed.
  - **The W16 expert byte floor** reads UNREAD: the registration assumed int4 experts through K19, and the served store
    is NF4. An NF4 floor would be its own registration.
  - **Total:** $0.521 (one power-floor refusal at $0.015, not a strike, and the read at $0.506).
  - **Default:** this record changes none.
