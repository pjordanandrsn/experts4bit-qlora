### P119 read (#846): where the 64-slot server's steps spend their device time (bench only; descriptive, no claim)

- **The reading** (`p119-5090-1`, one RTX 5090, $0.827; lane $1.029): on SC2e's int4 stack (Qwen3-30B-A3B), a 64-row
  decode step is 15.6 ms of device time, 44 % of it K19; above 256 routed rows the chained tile table adds 2.9 ms
  against 0.9 below, and `Int4Linear` above 16 rows runs bf16 cuBLAS. The 512-token prefill is 41.2 ms of device time,
  48 % int4 experts and 26 % small elementwise kernels; last-logits saves 0.9 ms. D2D copies are 4 per layer under bulk
  KV bookkeeping.
- **Predictions:** Q1–Q6 and Q8 held; Q7 (the LM head's share) missed. The standalone head bracket undercounted its
  calls; `RESULTS-p119.md` says by how much.
- **Files:** `bench/p119/RESULTS-p119.md`, `bench/p119/receipts/p119-5090-1/` (with `SHA256SUMS`).
