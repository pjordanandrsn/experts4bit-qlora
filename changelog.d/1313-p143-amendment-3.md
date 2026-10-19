### P143 Amendment 3 (#1313): the confirmatory served read of the attention split cap, registered (binding)

Amendment 2 (`p143a2-5090-1`) read NO_FLIP: g1 = 1.0297 against the 1.03 bar, under a W1 self-pair of 0.9845. The
verdict stands. Amendment 3 asks the same question on stricter terms, fixed before data:
- eight arms (ABBA twice) at five reps;
- the flip needs all four S1/S0 pairs at W1 >= 1.03 and W16 no slower than 0.99;
- a NOISY rung (any W1 self-pair outside [0.985, 1.015]) VOIDs and counts as a strike;
- SANE as in A2.

It is binding: a NO_FLIP closes the lever as opt-in for good. Its driver decides LANE DEAD by launch identity (`bench/common/lane_liveness.sh`). The kit is `bench/p143/p143a3_*`, and A2's box runs at its
registered bytes; the tests are in `tests/test_p143a3.py`. Nothing changes by default.
