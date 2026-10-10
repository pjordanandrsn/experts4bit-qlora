### SD2 Amendment 3, part 1 (#1313): the read's box stages and verdict

- **The box** (`bench/sd2/sd2_box.py`) gains the read's three stages: `--read-v` (V0 on rows outside the gate's
  calibration, judged before any timing, then V1's timed terms), `--read-e ARM` (one arm of the end-to-end read) and
  `--read-q` (P115 Phase B's quality instrument plus the verify-shaped ON1 / ON2 passes).
- **The verify pass is checked on CPU first.** On a causal toy target its per-position log-probs equal a T == 1
  teacher-forced pass, and its two construction mutants do not (`tests/test_sd2_read.py`).
- **The reducer** (`bench/sd2/sd2_reduce.py`) gains `--gate-v` (VOID / E_ONLY / ALL after stage V) and `--read`, the
  registered rule, self-tested on every verdict and every VOID item. The proof's rule and its 39 cases are unchanged.
- Part 2 (the runner, the driver, the package-diff audit, the amendment text and the target pin) follows once the
  build PRs have merged.
