# Rental lane liveness

`lane_liveness.sh` is shared by the TC1, P127, FAM and locality rental drivers. The controller sends it over SSH stdin; it is separate from each lane's pinned, staged instrument.

At launch, the driver captures the detached child's PID and its Linux `/proc` start ticks, boot ID and uptime. Polls check that exact process identity, so an SSH probe cannot match itself, a reused PID cannot impersonate the lane, and an `exec` from a guard into a runner stays live. Zombie and dead process states count as missing.

Two consecutive definite missing observations end the wait. A changed boot ID, uptime shorter than the lane's age, or an uptime reset ends it as rebooted (uptime comparisons allow five seconds for sampling and rounding). Failed SSH calls, malformed snapshots and unreadable existing process entries are unknown; they reset the missing streak and never end the wait. A stall remains informational. Completion markers, the final receipt fetch and the existing dead-lane exit code still govern the result.

`tests/test_lane_liveness.py` executes the probe against synthetic `/proc` files and all four actual polling loops with fake SSH. It covers self-matching `pgrep`, PID reuse, zombies, rebooted hosts, partial output from failed transport and healthy long runs. Existing lanes keep the driver they started with; adoption takes effect at the next launch.

On a reboot verdict the driver also writes `host-fault.json` into the launcher's run directory (`E4B_RENT_RUN_DIR`): `kind` reboot, the old and new boot IDs and the uptimes as `evidence`, and the time. The rental launcher copies it into the receipt as `environment.host_fault`, and a committed HARNESS_ERROR receipt that carries it can exclude the machine from later launches (`--exclude-vast-host-fault-receipt`, adertha-agents#204). Without a run directory nothing is written and the driver says so.

The heartbeat's `unknown_probes` counts consecutive unverified probes and resets on a definite observation. It is diagnostic only and never controls lane termination.

## Token-scope hygiene

TC1, P127 and FAM call `token_scope.py` on the controller before copying `HF_TOKEN_FILE`. The helper checks the supplied file through `huggingface_hub.HfApi.whoami` at the public Hub endpoint, with a 20-second verification deadline. It allows classic read tokens and fine-grained tokens whose permissions are all reads, excluding job, endpoint and webhook permissions. A verified unsafe token refuses staging with exit 78. Unrecognized metadata or a failed verification stages no token and continues unauthenticated; the launch environment disables implicit Hub authentication in that case and when no token file is present. Locality stages no token.

The helper logs only read-only yes/no or unverified, staging none; it suppresses library diagnostics and exception text. Tests use fake tokens, metadata and network failures. See Hugging Face's [token documentation](https://huggingface.co/docs/hub/security-tokens) for the role definitions.
