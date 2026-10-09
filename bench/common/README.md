# Rental lane liveness

`lane_liveness.sh` is shared by the TC1, P127, FAM and locality rental drivers. The controller sends it over SSH stdin; it is separate from each lane's pinned, staged instrument.

At launch, the driver captures the detached child's PID and its Linux `/proc` start ticks, boot ID and uptime. Polls check that exact process identity, so an SSH probe cannot match itself, a reused PID cannot impersonate the lane, and an `exec` from a guard into a runner stays live. Zombie and dead process states count as missing.

Two consecutive definite missing observations end the wait. A changed boot ID, uptime shorter than the lane's age, or an uptime reset ends it as rebooted (uptime comparisons allow five seconds for sampling and rounding). Failed SSH calls, malformed snapshots and unreadable existing process entries are unknown; they reset the missing streak and never end the wait. A stall remains informational. Completion markers, the final receipt fetch and the existing dead-lane exit code still govern the result.

`tests/test_lane_liveness.py` executes the probe against synthetic `/proc` files and all four actual polling loops with fake SSH. It covers self-matching `pgrep`, PID reuse, zombies, rebooted hosts, partial output from failed transport and healthy long runs. Existing lanes keep the driver they started with; adoption takes effect at the next launch.
