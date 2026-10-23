### Added

- P226 registered: the K25 plan at bucket 32. `bench/p226/PREREG-p226.md` registers, before any data, whether any tile
  plan in P156's 144-plan grid beats `c1` (qwen3_moe's served K25 plan, chosen at 16 rows) at bucket 32, where the
  served decode sends 32 x top-8 = 256 routed rows through K25. The reading box first captures its own routing: 32
  distinct requests at bucket 32 on the served default, the K25 route and the `c1` plan asserted over the window, the
  picks hashed before any timing (`p226_capture.py`, over P154's routing pass). The router-replay file cannot stand in:
  its 32 rows are 16 prompts recorded twice. Then P156's sweep (reported) carries three plan pairs, and P160's instrument
  rules on the first against `c1` on held-out steps (`p226_bench.py`, `p226_reduce.py`, `tests/test_p226.py`). Bucket 64
  is out of scope: it takes the M-tile route, not K25. No default, gate or claim moves.
