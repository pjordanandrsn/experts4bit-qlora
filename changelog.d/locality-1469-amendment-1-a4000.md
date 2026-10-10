### #1469 census Amendment 1: one rented RTX A4000 instead of the RTX A2000 (bench and tests only)

- No RTX A2000 with at least 12 GB was offered in 3 h; every A2000 listed was the 6 GB card. The census needs
  about 1.8 GB more GPU memory than calibration used.
- The RTX A4000 (16 GB) is sm_86 like the A2000, so the census runs the same NF4 host-residency path and kernels.
  Expert ids are correctness-class data, and no timing is quoted.
- `locality_run.sh` now expects the A4000 class, and its host-RAM floor is 48 GB, down from 64 (about 15 GB of
  pinned NF4 experts, plus one checkpoint shard in flight). `staged.sha256` is regenerated, and the README gains the
  amendment.
