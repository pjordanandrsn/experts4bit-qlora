### Every new rental runner checks its guard fit before the install, with 25 % headroom

`tests/test_lane_guard_fit.py` requires every runner that gates on time left, other than a frozen list of today's runners,
to refuse before its first install unless the time left covers its install allowance, its largest need and its reserve,
times 1.25. The refusal prints the quantity it compares and is not a strike, but two consecutive headroom refusals on
one lane count as one strike. The check is described in `bench/common/README.md` ("Guard fit"). It follows three guards
sized to fit exactly on 2026-10-19 (P149 box K, smt-sib, P154). No existing runner changes.
