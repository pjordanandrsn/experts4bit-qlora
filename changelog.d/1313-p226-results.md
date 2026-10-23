### Added

- **P226's record** (#1313), E32, K25's plan at Qwen3-30B-A3B's bucket 32: `bench/p226/RESULTS-p226.md` and the public
  receipt copy under `bench/p226/receipts/`.
  - **The read** is NO_GO. Every plan the selection carried is slower than `c1` at 256 routed rows: the first is 0.968x
    (lower bound 0.967, clock-adjusted 0.964, against the 1.05 bar), over 64 held-out steps of 32 requests' routing. The
    self-pair read 1.0001.
  - **The contract held:** all 144 plans per projection kept `c1`'s bits, and every arm matched on every rep.
  - **Reported only:** `c1` runs at 81 % of the copy rate at bucket 32 (at most 1.50 ms a step of headroom), and 1.77x
    faster than K25's built-in plan.
  - **Total:** $0.305. **Default:** this record changes none; `c1` stays qwen3_moe's plan at bucket 32.
