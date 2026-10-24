### Changed

- P232 Amendment 2 (after the shakedown, before any reading). `p232-shake-1` (an RTX 5090 on a 500 W board, explore tier)
  ran the kit end to end at three reps and reduced to `SHAKEDOWN`. Its twelve reducer inputs are now the reducer's real
  fixture (`bench/p232/fixtures/p232-shake-1/`, sha256-pinned): they reduce to `SHAKEDOWN` with every integrity check
  passed, six departures each VOID, and made reading-shaped they VOID on the 575 W floor alone (108 reasons, every one
  the 500 W cap), which a reducer without the power check does not catch. The guard read 609 s over its need after this
  host's boot. `p232_reduce.py`'s docstring names the fixture, so its staged pin moves; no rule or output changes.
