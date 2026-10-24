### Added

- P233 results (`bench/p233/RESULTS-p233.md`). `p233-5090-1`, the one reading of PREREG-p233.md (Amendments 1-4) at e4b
  `2a0a9afc`, read `SOLVER_SANE` with the TRIPWIRE fired at both group sizes. On Qwen3-30B-A3B's server,
  `E4B_PAGED_PLACEMENT=solver` at 8.0 GiB VRAM / 3.0 GiB DRAM (23 of 48 MoE layers tiered) keeps teacher-forced quality
  against `all-vram` inside P153's bars at T = 1 and 8 (NLL gap +0.002077 and -0.000047 nats; argmax agreement
  0.974284 at both; KL 0.003925 and 0.003729), but the agreement is below the tripwire's 0.98 look-bar and the KL above
  0.002. The written look in the results finds the disagreement smaller in every window than the all-vram server's own
  chunked-prefill floor arm, unbiased in NLL, and matched in size at T = 8 by a placement with no DRAM tier. `solver`
  may be proposed as an opt-in placement citing this read, never a default; no speed is read. $1.45. The receipt copy
  is under `bench/p233/receipts/p233-5090-1/`.
