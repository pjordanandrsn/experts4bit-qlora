### SC2f registered (#846): `serve_paged` at 64 slots with P130's two prefill defaults on against both off, box N (bench and tests only)

- **Why.** P130 took 5.18 ms off each 512-token first-chunk prefill replay (#1605; defaults #1608 and #1615). At 64 slots
  past 12 req/s every prefill step stalls every decoding row, so `serve_capacity` on SC2e's s64a costs puts the gain at
  the knee: +0.053 / +0.086 mean attainment over 12–14 req/s (`bench/sc2/sc2f_basis.py model`).
- **The box.** `SC1_BOX=N` (`bench/sc2/sc2f_box_n.sh`): eight SC2e s64a servers, on (both knobs unset: the release's
  defaults) and off (both 0), ordered on, off, off, on in draw 1 and the reverse in draw 2, so each arm has a same-seed
  replicate in every draw. Per server: warm, the 64-request burst, a FLIPS check of the start `/health` (rc 43), serial,
  a ladder at 4, 8, 12, 13, 14 and 16 req/s, and a knee block of four more seeds at 12, 13 and 14. A tripwire refuses an
  e4b without both defaults.
- **The rule** (`bench/sc2/sc2f_reduce.py`, 30 self-test cases): per-server gates ROUTES, SLOTS, FLIPS, ENGAGED, PROMPTS;
  FASTER iff the knee gain clears max(0.03, the same-seed replicate spread) in both draws, REGRESSED on a lower ceiling or
  a loss at 4–8 req/s, KNEE_SATURATED when the off arm's knee leaves less headroom than the
  bar, FLAT otherwise; predictions Q1–Q6 (prefill forward, serial and loaded TTFT/TPOT, the knee gain,
  the ceilings, the 64-row decode step) reported beside it.
- **Harness.** `sc1_run.sh` and `sc1_drive.sh` gain box N; `staged.sha256` regenerated; the box tests that enumerate the
  box letters now include N.
