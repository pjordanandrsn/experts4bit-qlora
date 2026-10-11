### Added

- R2-Mixtral Amendment 1 (`bench/fam235/PREREG-r2-mixtral.md`, #1630). The Mixtral-8x7B calibration's separation check failed for
  the routing plant (0.670 < 2U = 0.7052), so the global band's form is void on this family, as registered. It is replaced by one
  depth-local band, band_j = 2 × u_cal(round(4j/7)), used both ways on the 8x22B box. An out-of-sample 8x7B read on new tokens
  validated it. Numbers: `bench/fam235/r2-mixtral-a1-numbers.json`. Docs only: no rental, budget, default change or claim.
