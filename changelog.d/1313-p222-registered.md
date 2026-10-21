### Added

- P222 registered (#1313): K25's decode x plan at Mixtral-8x7B's W16 shapes, a microbench premise whose control is
  Mixtral's served default (the select tree at today's plan; no read licenses `mixtral`). Block A sweeps P156's 144-plan
  grid under K39's decode on one seeded routing draw and selects M; block B grades M against the tree on a second draw,
  with P (the decode alone), C1 (e4b's `c1` pair) and TM (M's plan under the tree) reported. GO needs M's median minus
  2 SE, raw and clock-adjusted, at 1.10 or more. Kit: `bench/p222/` and `tests/test_p222.py`. No default, gate or speed
  claim changes.
