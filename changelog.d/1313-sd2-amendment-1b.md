### SD2 Amendment 1b (#1313): the proof's time budget, after `sd2-prove-1`

- **What happened.** `sd2-prove-1` stopped before its fetch (rc 40, $0.046): SD1's fetch budget did not fit a 0.75 h
  guard after the installs (P109's trap).
- **Per-mode checks.** Every time-left check in `bench/sd2/sd2_run.sh` is now a per-mode variable.
  `tests/test_sd2.py` holds P109's rule and a slow-host walk through every check.
- **The guard and the ceiling.** The guard is 1.25 h and the ceiling $1.75; the expected cost is about $1.10.
- **A short deadline refuses at once** (rc 17).
- **The GPU tests come first.** They run before the 62 GB fetch and stop the lane on failure (rc 24).
