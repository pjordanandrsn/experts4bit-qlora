### P223 kit: the schedule-invariance price read's box, reducer and runner

`bench/p223/` adds the kit that `PREREG-p223.md` registered. It is built from P149's box Q+S kit by substitution: the
four arms (D0, I0, D1, I1) in two ABBA blocks of fresh processes, with no Q, no token-pinned blocks and no SANE phases.

Each process records the engagement the registration binds:
- the split cap and the plan per bucket;
- the runner's buckets, with its own replay and pad-row counters;
- every decode call's rows and bucket;
- the int8 counters.

It also keeps every timed rep's tokens for the schedule-invariance read.

`p223_reduce.py` implements the registered rule: VOID, NOISY, the label per server, the price and the cross-W check. It
reports each process's rep repeatability apart from the schedule read, and grades the predictions. The runner writes its
phases with `bench/common/phases.sh`.

`tests/test_p223.py` pins the staged pieces, the arms, the runner's order and the fit table against the 3.5 h guard. It
kills a mutant per rule.

No default, gate or claim moves.
