### Added

- P227 results (`bench/p227/RESULTS-p227.md`). `p227-5090-1`, the one reading of PREREG-p227.md (Amendments 1-4) at e4b
  `b7146363`, read `qwen3=MISS mixtral=PASS`, and all three predictions held. On Mixtral-8x7B the per-row GEMV route
  (`E4B_NF4_SMALLM_BW=1`) runs the expert call in 0.36 of K25's time at two decode tokens and 0.47 at four (upper bounds;
  bar 0.70). On Qwen3-30B-A3B it runs in 0.75 and 0.90, a MISS. B2's multi-row trigger fires on both families (B over
  the byte floor at eight tokens: 2.00 and 1.59; threshold 1.25). As registered: Mixtral opens B3, a served read under
  its own registration; B2 opens with a design note, Mixtral first; and the default stays off. $0.083. The receipt copy
  is under `bench/p227/receipts/p227-5090-1/`.
