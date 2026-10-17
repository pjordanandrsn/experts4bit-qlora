### Added

- R2-Mixtral Amendment 3 (`bench/fam235/PREREG-r2-mixtral.md`, #1630), which registers the 8x22B box. It sets:
  - the RTX 5090 class (at least 32,000 MiB), 134 GB of host RAM and 350 GB of free disk;
  - R2's own fitted caps (box 3,600 s, repeat 2,040 s, fetch 7,080 s, install 360 s) and a 0.063 GB/s pull floor in `r2_run.sh`;
  - the verdict's class and estimates.
  The fit records are in `bench/fam235/r2-fit/`, and the worst case is 4.57 h. Gates 1, 2, 6's fetch-gate run and 7 remain before
  any box. No rental, budget or default change.
