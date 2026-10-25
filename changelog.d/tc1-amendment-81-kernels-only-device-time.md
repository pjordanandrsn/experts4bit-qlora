### TC1 amendment 81: device time is kernels-only, at the source and over the 26 profiled runs

- **At the source.** `bench/tc1/tc1_arm.py`'s profile summary leaves torch's user-annotation rows out of `device_ms`, the memcpy figure
  and the family fractions, since the `Optimizer.step#…` span covers kernels that are rows of their own (e4b#2129). It reports their time
  as `user_annotation_ms`, and names its method: `device_ms_method` is `kernels_only_at_source`, `all_cuda` on a torch without the
  flag, or `no_device_rows` when the profile has no CUDA row. `f15v2.kernels_only_device_ms` refuses anything but a legacy profile, so no figure is corrected twice. Amendments 78-80's tests
  read the arm at their frozen commits.
- **The re-derivation.** `bench/tc1/device_rederive.py` (new) reads each committed TC1 run twice with the run's own staged reducer:
  - as recorded, which must reproduce the committed RESULTS (a proof step);
  - kernels-only, with the five profile fields each recomputed from its formula and the families re-attributed exactly.
  Committed receipts are never edited. The corrected cells live in `bench/tc1/am81-rederive/<run>/`, and `--check` re-derives them.
  - 26 runs: 26 reproduced, 0 refused, **0 verdict flips**.
  - Moved: TC1's field-recipe device ratio Unsloth / e4b 1.924 -> **1.953**, and packed 4,096-token rows 1.160 -> **1.161**.
  - Every near-bar prediction still holds.
- **Where the figures are quoted.** `device_rederive.py --quotes` lists every register row and document line quoting a moved figure,
  by two methods (`bench/tc1/am81-rederive/QUOTES.md`). 13 register rows are superseded by `<id>.am81` successors, and 3 rows gain a
  correction in their notes. README.md, docs/STATUS.md, docs/CHOOSING.md and docs/REPRODUCE.md quote the successors.
