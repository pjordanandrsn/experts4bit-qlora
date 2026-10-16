### P132: the canary's step floor is what SC1b can keep (strike 1)

Lane P132's first box run, `p132-5090-1`, stopped at its in-box canary. SC1b kept 5 steady steps of 8 against a floor of
6, which it can never reach: 8 bracketed replays delimit 7 periods, and SC1b trims the first and the last.

The runner now demands `CANARY_STEPS − 3`, every step SC1b can keep, so one more lost step still voids the canary.
`tests/test_p132.py` reproduces the run's verdicts from its records (`tests/fixtures/p132-5090-1/`). This change touches
bench and tests only: SC1b, the census and the rule are unchanged (`PREREG-p132.md`, Amendment 1).
