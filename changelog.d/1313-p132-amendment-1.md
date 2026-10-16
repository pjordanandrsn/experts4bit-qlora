### P132 Amendment 1 (#1313): the canary's census floor fits what SC1b keeps

`p132-5090-1` stopped at its in-box canary (rc 26, $0.731) before timing anything.

**Cause.** SC1b's census keeps at most N − 3 of N bracketed replays:
- it makes one row for each period between consecutive replays;
- it then trims both edges.

The registration's canary floor was `CANARY_STEPS - 2`, so 8 replays kept 5 against a floor of 6. The canary could
not pass on any host. The main census (64 replays, 61 kept against 56) is unaffected.

**Change.** `bench/p132/p132_run.sh` now captures 12 canary replays with a floor of `CANARY_STEPS - 4`, which tolerates
one long period. `staged-p132.sha256` re-pins `p132_run.sh` only.

**Tests.** `tests/test_p132.py` first reproduces the run's two void `why` strings byte for byte from its real canary
records (`tests/fixtures/p132_canary/`, digests checked), then checks the amended floor and the main census against the
reader's own count. The reducer, box, class map, rule and predictions are unchanged.
