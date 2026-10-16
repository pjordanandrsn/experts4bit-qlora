### Added

- R2-Mixtral Amendment 2 (`bench/fam235/PREREG-r2-mixtral.md`, #1630), which raises the host-RAM floor from 128 GB to 134 GB.
  The floor is measured by Q235 Amendment 5's depth-scaling read on 2–8-layer cuts of Mixtral-8x22B. The pinned homes match the
  estimate exactly from 2 to 6 layers. The 8-layer point is undercounted under swap pressure and is not used. The floor adds
  Amendment 4's four fp32 host check copies (7.6 GB) and the held layer outputs. Numbers: `bench/fam235/r2-memslope.json`.
  Docs only: no rental, budget, default change or claim.
