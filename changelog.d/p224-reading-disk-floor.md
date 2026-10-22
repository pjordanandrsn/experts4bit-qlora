### Fixed

- P224's reading now refuses a box with less than 100 GB free before any install (it was 60 GB). The fetch, the
  bake and the install need about 85 GB, so a 60 GB box would have failed mid-run. The reading's manifest rents a
  120 GB disk. The shakedown keeps its 40 GB floor.
