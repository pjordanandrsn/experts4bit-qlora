### P132 registered (#1313): where SD2's 1 -> n row verify increment goes, split by kernel class on one RTX 5090

- **The question.** SD2's verify of n rows cost 4.12 / 4.56 / 4.90 ms more than one row (n = 2 / 3 / 4). The A2000
  inventory showed the expert path switching kernels at T >= 2: the bandwidth GEMV at one row; tile table + K25 at 2-4
  rows. P132 times that split by kernel class on the read target (e4b `88ae1cfd`, gnf4 0.45.0) with P123's Nsight
  census, plus SD2's own V1 timing re-run unprofiled.
- **The prediction, registered before any data.** `bench/p132/desk_prediction.py` bounds every component from existing
  receipts and leaves a 2.38-3.17 ms hole at n = 2. H, the hypothesis that the expert path holds it, predicts BUILD.
- **The rule.**
  - VOID unless the re-run's verify times sit within 5 % of SD2's and the increment within 10 % of the target.
  - H reads SUPPORT at a share >= 0.70 at every n, REFUTE below 0.50 at every n, INCONCLUSIVE otherwise.
  - BUILD a 2-4-row expert kernel iff the measured savable part S(n) reaches 2.358 / 1.683 / 1.558 ms, SD2's own FASTER
    bar solved for full_ms. NO_BUILD closes multi-token B = 1 levers for this stack.
- **The canary runs first in the box** (buckets 1-4 captured, one replay per condition, a short census through the
  classifier), so a failure costs minutes. The class map is v1.2: P123's v1.1 plus `_rope_norm_qk`, shown not to move
  any P123 class.
- **Budget.** One RTX 5090 pinned by machine and offer, a 1.25 h guard, a ceiling of about $1.40. It waits for DQ11 to
  close and the maintainer's ACK.
