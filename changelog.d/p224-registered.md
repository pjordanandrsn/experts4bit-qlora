### Added

- P224 registered: one trained attention adapter on the paged server, priced against its reference path.
  `bench/p224/PREREG-p224.md` registers, before any data, the end-to-end read of the shipped `E4B_PAGED_ADAPTER`: an
  attention-only canary adapter trained by Loggetta and served merged at load, against the same adapter under
  Loggetta's own reference path, on one RTX 5090 with Qwen3-30B-A3B. Five gates (effect within the paged base's own
  gap, the HTTP canary in four modes, decode cost inside a frozen ABBA band, the served identity, the fingerprint
  checks on the box); a second canary trained at the default double-quantized expert absmax prices ADAPTER_PORTABILITY
  F1 as reported. The rule (`p224_reduce.py`), the canary data (`p224_canary.py`, sha256-pinned) and the box
  (`p224_box.py`) are fixed here with their self-tests and mutants (`tests/test_p224.py`); the runner follows in the
  kit PR. The shakedown box also generates two expert-LoRA fixtures for the D0 premise read, declared and never
  graded. No default, gate or claim moves.
