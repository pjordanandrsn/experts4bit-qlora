### SC2f amendment 10: the power class is {575, 600} W; a recorded clock fail is never admitted

- `bench/sc2/sc2_qualify.py admit`: machines listing 575 or 600 W form one class. Outside host continuity and the named
  hosts, a machine whose latest receipt with a loaded clock record fails `card_clock.passes(..., 2700, mode="clock")` is
  never admitted (`recorded_clocks()`). Recorded-pass machines are walked before unrecorded ones.
- `power` passes an enforced limit of 575.0 or 600.0 W. `judge-proof` requires the proof's `forensics.txt` and
  `host_facts.json` to equal the qualify step's reading (`--qualify-power`), which `sc2_qualify_drive.sh` now passes.
- The floor (2,700 MHz) and the clock mode are unchanged. `bench/sc2/sc2f-am10/` holds the census behind it: over 228
  screen reads, 600 W listings reached an admit point 61 times against 575 W's 7. Pre-data. SC2h and SC2i take it by
  reference.
- The run sampler runs `--format=csv,noheader,nounits` and `judge-run` reads it by position, with per-column sanity checks
  (sc2f-prove-11 was void because driver 610.43.02 printed `clocks.sm` as `clocks.current.sm` in a header the judge looked
  up by name). The qualify fetch excludes the burn's venv and caches. A harness fault voids the run; the create still uses
  an attempt.
