### The guard-fit rule counts a venv's creation as the start of an install

`tests/test_lane_guard_fit.py` orders each gated runner's pre-install fit check against its first install. A runner
whose packages go in from a script (Q235's pinned bootstrap, after `python3.11 -m venv`) showed no recognisable install
and failed with "no install found". The install pattern now also matches a venv's creation (`python`, `python3` or
`python3.N -m venv`), with a mutant ("a venv created first") that must fail. `bench/common/README.md` ("Guard fit")
says so. Every existing runner's status is unchanged: the frozen legacy list still matches exactly, and the rule's
other tests pass.
