### P227 registered: the per-row bandwidth GEMV against K25 at 2 to 8 decode tokens, two families (serving roadmap lever B)

A microbench registration, no box yet. At 2, 4 and 8 decode tokens on the NF4 store, it times e4b's MoE expert call
(the collapse's own `_fused_over_stack`) through the served route and through the per-row route. The served route is the
device tile table and K25, at the family's served decode and plan. The per-row route is `E4B_NF4_SMALLM_BW=1`, which
sends one group per (token, slot) row to grouped-nf4-gemm's K33.

- **Two families, each ruled on its own:** Qwen3-30B-A3B (48 layers, routing from P153's router replay, stream S1) and
  Mixtral-8x7B (8 layers, P222's seeded draw). The routing tables are pinned by sha256.
- **The bar:** PASS where the per-row route's time over K25's has an upper bound (median + 2 SE), raw and
  clock-adjusted, of at most 0.70 at both T = 2 and T = 4. Otherwise MISS, a finding that sizes the lever.
- **B2's trigger:** the per-row route's median over the byte floor at T = 8 above 1.25 opens a multi-row GEMV for
  that family.
- **The rest:** K2 (the self-pair), a spacer before every timed replay, Williams order, NVML clocks with a 575 W floor.
  The premise is e4b#2009's GPU contract. The guard fit has 25 % headroom.
- **Amendment 1, before any reading:**
  - review 1's four items: K's decode and plan from e4b's own tallies; a shakedown forces K33's gate; a run with no
    family read is not a reading; an unreadable record is VOID;
  - TOLERANCE_FAIL is final for its family;
  - two shakedowns (`p227-shake-1`, `-2`, an RTX A4000, about $0.03 together). The second's record is the reducer's
    sha256-pinned real fixture.
- **Files:** `bench/p227/` (PREREG, bench, reducer with a 45-check self-test, runner, drive, staged pins, the fixture)
  and `tests/test_p227.py` (29 tests: the real record, its departures and an old-rule mutant, and 9 reducer mutants).
