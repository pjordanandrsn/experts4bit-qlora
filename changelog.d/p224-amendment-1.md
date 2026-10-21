### Changed

- P224 Amendment 1, before any data: the shakedown's fixture trains use Alpaca's full-text column at a pinned revision
  (`dce01c9b`, `--format text`), since the built-in demonstration stops at an empty output (row 284) before 200 x 512
  tokens; and the box saves the trained base (fixture-base), checked against both fixtures' receipts, so the D0 premise
  read does not depend on CPU-vs-CUDA quantizer parity. The shakedown fit becomes 4,660 s under a 1.7 h guard, worst
  case USD 0.55. The reading, its rule and its budget are unchanged.
