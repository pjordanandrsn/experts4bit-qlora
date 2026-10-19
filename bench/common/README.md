# Rental lane liveness

`lane_liveness.sh` is shared by the TC1, P127, FAM and locality rental drivers. The controller sends it over SSH stdin; it is separate from each lane's pinned, staged instrument.

At launch, the driver captures the detached child's PID and its Linux `/proc` start ticks, boot ID and uptime. Polls check that exact process identity, so an SSH probe cannot match itself, a reused PID cannot impersonate the lane, and an `exec` from a guard into a runner stays live. Zombie and dead process states count as missing.

Two consecutive definite missing observations end the wait. A changed boot ID, uptime shorter than the lane's age, or an uptime reset ends it as rebooted (uptime comparisons allow five seconds for sampling and rounding). Failed SSH calls, malformed snapshots and unreadable existing process entries are unknown; they reset the missing streak and never end the wait. A stall remains informational. Completion markers, the final receipt fetch and the existing dead-lane exit code still govern the result.

`tests/test_lane_liveness.py` executes the probe against synthetic `/proc` files and all four actual polling loops with fake SSH. It covers self-matching `pgrep`, PID reuse, zombies, rebooted hosts, partial output from failed transport and healthy long runs. Existing lanes keep the driver they started with; adoption takes effect at the next launch.

On a reboot verdict the driver also writes `host-fault.json` into the launcher's run directory (`E4B_RENT_RUN_DIR`): `kind` reboot, the old and new boot IDs and the uptimes as `evidence`, and the time. The rental launcher copies it into the receipt as `environment.host_fault`, and a committed HARNESS_ERROR receipt that carries it can exclude the machine from later launches (`--exclude-vast-host-fault-receipt`, adertha-agents#204). Without a run directory nothing is written and the driver says so.

The heartbeat's `unknown_probes` counts consecutive unverified probes and resets on a definite observation. It is diagnostic only and never controls lane termination.

## Token-scope hygiene

TC1, P127, FAM and SD1 call `token_scope.py` on the controller before copying `HF_TOKEN_FILE`. The helper checks the supplied file with one standard-library GET of the public Hub's `whoami-v2` endpoint (the token as a Bearer header; what `huggingface_hub.HfApi.whoami` does), with a 20-second verification deadline. It needs no third-party package: controllers' python may lack `huggingface_hub`. The Mac mini's did, and until this change that turned every check into "unverified" and every launch from it unauthenticated. It allows classic read tokens and fine-grained tokens whose permissions are all reads, excluding job, endpoint and webhook permissions. A verified unsafe token refuses staging with exit 78. Unrecognized metadata or a failed verification stages no token and continues unauthenticated; the launch environment disables implicit Hub authentication in that case and when no token file is present. Locality stages no token.

The helper logs only read-only yes/no or unverified, staging none; it suppresses library diagnostics and exception text. Tests use fake tokens, metadata and network failures. See Hugging Face's [token documentation](https://huggingface.co/docs/hub/security-tokens) for the role definitions.

## Guard fit

Every rental runner that gates on time left (`can_run`) refuses BEFORE its first install unless its guard fits its work with
headroom. `tests/test_lane_guard_fit.py` enforces the structure on every runner not on its frozen legacy list. The check:

```sh
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_X + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((DEADLINE - $(date +%s)))
# rc NN: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; finish NN; }
```

- `INSTALL_TO_GATE_S` is the runner's allowance for everything between this check and its largest time-left gate (the
  install, the self-tests, the fetch, the bake). `NEED_X` is that gate's need and `RESERVE` the fetch-and-teardown
  reserve its `can_run` keeps.
- The check sits at top level (unindented, outside any function body or branch), before the first install (pip, pip3,
  conda, mamba, uv, apt-get or pipx) and before any script sourced or run from another file. A runner with no
  recognisable install fails the test rather than skipping the order check.
- The headroom covers the allowances, which are estimates from the hosts measured so far. The check runs against the box's
  own remaining time, so a slow boot is caught as well.
- A refusal here costs cents and is not a strike. Two consecutive headroom refusals on one lane count as one strike: the
  guard, not the hosts, is then wrong.
- The static side stays per lane: each lane's own test asserts its declared fit x 1.25 <= its registered guard, as
  `tests/test_p154.py` does. No PREREG is parsed.
