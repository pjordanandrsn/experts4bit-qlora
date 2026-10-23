### Added

- P228 results (`bench/p228/RESULTS-p228.md`). `p228-5090-r1`, the one reading of PREREG-p228.md (Amendments 1-7) at e4b
  `32fda64d`, read `o_proj=PASS router=PASS`. On one RTX 5090 (cuBLASLt 120804), the cuBLASLt configuration selected on
  the card for both of Qwen3-30B-A3B's dense sites (algo 21, tile 5, split_k 16, reduction scheme 4) runs `o_proj` in a
  median 0.60 of `F.linear`'s time at 2 and 4 rows (upper bounds 0.5981 and 0.6009; bar 0.85) and the router gate in a
  median 0.28 (upper bounds 0.2832 and 0.2843). At 8 rows, which never rules, the router reads 0.88 (above the bar): the
  pin is flat, and cuBLAS's own choice goes split-K there. All three predictions held. As registered, a code PR fills
  `LT_PINS` from the verdict's pin table, a served read with B3's quality bar comes before any default, and
  `E4B_DENSE_LT_PIN` stays opt-in. $0.04 (the lane $0.052 with three boxes that did not start). The receipt copy is
  under `bench/p228/receipts/p228-5090-r1/`.
