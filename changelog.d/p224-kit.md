### Added

- P224 kit: the box runner (`p224_run.sh`), its driver (`p224_drive.sh`), the shakedown's fixture-base script
  (`fixture_base.py`) and the staged hashes (`staged-p224.sha256`), for the registered read and its Amendment 1. The
  runner refuses before any install unless the guard fits its terms (6,990 s for the reading, 4,660 s for the shakedown,
  x 1.25), holds the image's torch, pins e4b CI's grouped-nf4-gemm and Loggetta main `90a7697d` (0.6.1 plus the torch
  preflight; the released 0.6.2 also carries the demo-data fix, which the fixtures never use), and recomputes the
  canary pin on the box. `tests/test_p224.py` asserts the staged hashes, the pins and both fit tables against the runner.
