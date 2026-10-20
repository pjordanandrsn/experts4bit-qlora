### P160 Amendment 1 (#1313): hosts with no recorded power cap, as a fallback

P160's launch takes only RTX 5090 hosts with a power cap of at least 575 W recorded within 3 days, and on 2026-10-20 none
was offering. Amendment 1, registered before any data, admits hosts whose cap is unrecorded, but only after a launch
loop under the registered terms has run at least 100 rounds over 2.5 h with no pin.
- The box's 575 W floor (rc 17 before any install) and the reducer's every-sample VOID are unchanged.
- Each rc 17 probe is fetched back and its cap is appended to the host registry. There are at most 3 probes, none a strike.
- Run ids go to `p160-5090-6` so that probes cannot use up the reading's ids. The lane's ceiling is 6 x $1.16 = $6.96.
- The arms, the rule, the guard and the per-run worst are unchanged.

Files: `bench/p160/PREREG-p160.md`, `bench/p160/p160_run.sh`, `bench/p160/p160_drive.sh`, `bench/p160/staged-p160.sha256`,
`tests/test_p160.py`.
