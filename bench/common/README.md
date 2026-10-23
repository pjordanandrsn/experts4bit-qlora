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
  conda, mamba, uv, apt-get, pipx, or a venv's creation, which starts the install when the packages go in from a
  script such as Q235's bootstrap) and before any script sourced or run from another file. A runner with no
  recognisable install fails the test rather than skipping the order check.
- The first install the test finds may be inside a helper's DEFINITION (`pipx(){ ... pip install ...; }`), so put
  the check above any install helper, not only above its first call. `FIT` must name an install allowance
  (`INSTALL*`): a check made before the install has to count the install still ahead of it.
- The headroom covers the allowances, which are estimates from the hosts measured so far. The check runs against the box's
  own remaining time, so a slow boot is caught as well.
- A refusal here costs cents and is not a strike. Two consecutive headroom refusals on one lane count as one strike: the
  guard, not the hosts, is then wrong.
- The static side stays per lane: each lane's own test asserts (boot + its declared fit) x 1.25 <= its registered
  guard, as `tests/test_p155.py` does (300 s + 3950 s, 5313 s <= 5400 s). `boot` covers the time from the launch,
  where the runner's deadline starts, to this check (the instance's creation and boot, the ssh wait, the staging):
  `LEFT` has already lost it when the check runs. A guard sized to the fit alone refuses every box. The box itself needs
  only boot + fit x 1.25 <= guard (300 s + 4937 s = 5237 s for P155); the static form is the stricter of the two,
  keeping the headroom on the boot as well, since the boot varies by host. No PREREG is parsed.

## Registration rules (new lanes)

Each rule below was added to a lane by a pre-data amendment on 2026-10-23, after a box hit the gap it closes. Every
lane registered after this section merges states all seven in its PREREG. Existing lanes keep their registered text,
including their own classification rules (P225's hand reading of a refused classify, for one), and amend it only if they
hit one of these gaps.

1. **One launch commit, and one pinned stack.** The PREREG names both, and later merges to main can change neither:
   - the **launch commit**: the commit every box's kit and driver launch from (the registration's merge, or the latest
     amendment's). SC2 at 36cdd2d4, P225 at 8fe4cd2d, P227/P229 at b7146363, P228 at 67faee63;
   - the **stack under test**: the e4b, kernel and loggetta commits (and the versions) the box installs. Usually it is
     the launch commit itself. A replication pins the replicated lane's stack instead, so its boxes differ only in the
     machine.
   A lane that moves to a newer launch commit or stack first lists, in a pre-run note, every change to the default path
   it measures since the old one, with why the reading still stands.
2. **Run ids.**
   - Every box has its own id and receipt, and an id is never reused, because the ledger holds one row per run.
   - Register a reading id for EVERY box the cap allows to start (`<lane>-<card>-1` to `-N`, with N at least the cap),
     so a re-run after a started box that was not read has its own id. Also register retry ids (`-r1` to `-r8`) for any
     box that follows one that was created but did not start (NOT_RUN, or a refusal before the install). A refusal at
     create uses no id: the launcher receipts it under `<id>-refused-<ns>-<pid>` (with `environment.requested_run_id` =
     the id), and the next attempt takes the same id again. An ALARM (a possible orphan), or a create already sent to a
     provider the sweeper tracks, keeps the id: its receipt is that id's record.
   - Ids are labels, not budget. No code may parse an id's number; the caps below govern alone.
   - If boxes that did not start ever use up `r8`, that is a launcher or market fault to fix, not a reason to add ids.
3. **Outcome classes, and what each counts toward.** Each class is decided from the launcher's receipt and the
   runner's exit, never classified by hand:

   | outcome | when | started? | counts toward |
   |---|---|---|---|
   | refused at create | the provider refused; nothing was created ($0). Its receipt is `<id>-refused-*`, and it uses no id | no | nothing |
   | NOT_RUN | the launcher's pre-flight tore the instance down before the runner ran (no RUN_ID, exit file or marker) | no | nothing |
   | refused before the install | the runner's own refusal before its first install: the power floor (rc 17), the guard fit (rc 41), a host refusal | no | a host refusal counts toward the lane's refusal cap, if it registers one. A single rc 41 is no strike; two consecutive rc 41 count as one strike toward the stop (Guard fit) |
   | started | the runner passed those refusals and began its install | yes | the started-box cap, always; the stop only when NOT read (HARNESS_ERROR, VOID, a fault mid-install or mid-run). A read ends or advances the lane by its own rule |

4. **A started-box cap, a stop, and a bounded worst.**
   - The PREREG states the most boxes that may START (the cap) and each box's worst case. Retry ids never add a started
     box: they only replace ids used by boxes that did not start.
   - The stop ends a lane that keeps failing to produce a reading. It counts STRIKES: a started box that was not read is
     one strike, and two consecutive rc 41 are one strike (Guard fit). A read box is never a strike. The PREREG states
     how many strikes stop the lane. A capped host therefore cannot end a lane, and alternating faults cannot dodge
     the stop.
   - The lane's worst is bounded by its ids: at most N + 8 created boxes (N reading ids plus r1 to r8; a refusal at
     create uses no id and costs $0, so it is not among them), of which at most the cap start. So the lane worst =
     cap x the box worst + (N + 8 - cap) x the worst of a box that does not start. That is its time to the refusal at
     the box rate (at most rule 5's boot term, plus the fit check), plus any download before it. A NOT_RUN or rc 17/41
     box costs its time to the refusal.
5. **Boot slack.** A box's deadline runs from the create, and the launcher alone may wait `ssh_ready_s` (420 s on
   Vast) for ssh before it stages the kit. So the boot term in Guard fit's static check is at least `ssh_ready_s` plus
   a staging margin of 120 s or more: guard - FIT_NEED >= 540 s on Vast. The lane's test asserts it. A boot term below
   the launcher's own ssh wait halts the lane on any slow-ssh host. P225's 300 s term did, at 443 s from launch.
6. **Pins at the frozen commit, once closed.** When a lane records its verdict, its tests assert its own literal pins
   (staged hashes, the runner's kernel pin against CI's) against its FROZEN commit, never the live tree:
   - read with `git show <frozen>:<path>`, where `<frozen>` is the commit at which those pins were last asserted
     equal, usually the read's merge;
   - an unreadable frozen commit fails under `CI=true` and skips in a shallow clone;
   - the test names the frozen commit in a constant with a comment, as `tests/test_p224.py`'s `P224_FROZEN` does
     (`HO2_FROZEN` follows with e4b#2030).

   A later move of a shared pin (CI's kernel pin, a shared staged file) then neither re-pins nor fails a closed lane.
   A lane still OPEN that a shared move would break needs an amendment, not a frozen read.
7. **Every reducer gate fails closed.** A missing, `None` or empty input reads VOID or UNREAD, never a pass. Each gate
   has a self-test with its evidence removed. Where the registered text separates the two, say which input reads which.
   For example, SC2j reads a server with no end `/health` as UNREAD only when the runner recorded a deadline skip for it,
   and as VOID otherwise (a crash or a lost server). Added 2026-10-23, from P230's bucket-16 control and SC2j's plans and
   `kv_blocks_waits`.
8. **A fetch never imports links.** A rented box is a third-party host, and `rsync -a` (or any `-l`) copies a symlink
   as a symlink. A box could leave a relative link in its work dir that resolves, on the controller, to a file outside
   the run dir. Fetch with `--no-links` after `-a` (`rsync -az --no-links ...`), or with flags that include neither `-a`
   nor `-l` (`rsync -rtz ...`): rsync then skips each link ("skipping non-regular file"). The receipt commit refuses a
   link too, so this rule is the second layer, not the only one. It applies to a driver at its next edit or launch.
   Closed drivers are not swept: `tests/test_fetch_no_links.py` freezes every script that kept links on 2026-10-23 by
   its sha256, and a new script, or a frozen one whose bytes change, must pass. The test finds rsync in any command
   position (`if ! rsync ...`, `$(rsync ...)`, inside a function body) and reads the flags in order: `--no-links -a`
   still keeps links. Added 2026-10-23.

## Host facts

`host_facts.py OUT` writes the box's own measured host facts as `host_facts.json`, for the host registry. A runner calls it once on the box and leaves the file beside its receipt: `python3 host_facts.py "$OUT_DIR/host_facts.json"`. With `-` instead of a path it prints one `HOST_FACTS_JSON {...}` line.

**It is a byte-identical copy** of adertha-agents' `adertha/compute/host_probe.py`, the source of record beside the registry that reads its keys. Edit it there first, then copy it here byte for byte; adertha-agents' CI reports any drift. The launcher also runs the same probe on every rented box after its pre-flight and writes `host_facts.json` beside the receipt, so a lane need not call it for the registry to see its box.

It records only what the box measures itself, from `/proc`, `/sys` and `nvidia-smi`:
- the host's logical CPUs, physical cores, threads per core and NUMA nodes (`host_*`). Inside a container these
  describe the whole host, not what the run may use;
- the run's own share: `affinity_cpus` (`sched_getaffinity`) and the cgroup CPU quota, v2 or v1 (`cpu_quota_cpus`, or
  `cpu_quota_unlimited` when there is none, so "unlimited" is never mistaken for "unreadable");
- `numa_balancing` and the transparent-hugepage `enabled` and `defrag` modes;
- per-core speed:
  - `cpu_max_mhz` and `cpu_max_mhz_lo` are the highest and lowest rated maximum clock over the run's CPUs (its
    affinity). On a hybrid part they differ, P-cores against E-cores. They are labels: not what a busy, shared host
    delivers.
  - `cpu_loop_mops` is a fixed single-thread Python loop, best of three, well under a second. It measures what the run
    actually got from one core.
  - `cpu_loop_mops` compares only at the same `cpu_loop_python` (e.g. `312`).
  - To see a host change during the run, a runner may call the helper at the start and again at the end;
- `thp_mem_alloc_enable`, the `THP_MEM_ALLOC_ENABLE` setting from the helper's own environment, as an integer; left out
  when unset or not an integer;
- `omp_num_threads`, the `OpenMP` thread count from the helper's own environment, so call it with the run's env set.
  It is left out when unset or not a positive integer. Unset means the runtime's default (libgomp: the affinity count;
  torch's intra-op: the physical cores);
- GPU 0's power limit, maximum SM clock and memory.

Every value is a number or a small enum, and `sources` names where each one came from. It writes no identifiers: no hostname, container id, serial, UUID or CPU model string. A fact it cannot read is left out, and the helper always exits 0, so it never costs a run.

`measured_at` stamps the reading. The registry rejects a file whose stamp falls outside the run's own window, so a copy shipped with a kit is never taken for a reading. `tests/test_host_facts.py` runs it against a synthetic `/proc` and `/sys` and a stub `nvidia-smi`.

## THP window records

`thp_window.py` (standard library only) records transparent huge pages around a lane's timed window, inside the measured
process. The launcher's host-facts probe reads the host's THP mode once per run. A reading that can depend on huge pages
also needs to know whether huge pages were possible at the window and whether the process held any:

```python
import thp_window
before = thp_window.snapshot()       # just before the timed window
...                                  # the timed work
rec = thp_window.window(before)      # both snapshots, the seconds between, the host's thp_* counter deltas
```

A snapshot holds:
- the host's THP mode (`thp_enabled`, `thp_defrag`, or `absent` on a kernel without THP);
- this process's `THP_MEM_ALLOC_ENABLE` (left out when unset);
- its `AnonHugePages`, `Anonymous` and `Rss` from `/proc/<pid>/smaps_rollup`;
- the host's `thp_*` counters from `/proc/vmstat`. These count every process on the host, co-tenants included.

A fact that cannot be read is left out. A lane stages the file beside its box script and lists it in its staged hashes.
`tests/test_thp_window.py` reads synthetic `/proc` and `/sys` trees.

## Phase stamps

Every NEW rental kit writes `phases.jsonl` in its box work directory, and the driver fetches it with the receipts. A later read of where rented time goes (setup against measurement) then needs no regex over free-form logs. Existing kits and committed receipts are not changed.

Source `phases.sh` in the runner and bracket each phase:

```sh
. ./phases.sh                   # staged next to the runner, like any other helper
phase_start install setup; pip install ...;      phase_end install
phase_start fetch setup;   python fetch.py ...;   phase_end fetch
phase_start armA measure;  python box.py --arm A; phase_end armA
phase_start reduce reduce; python reduce.py ...;  phase_end reduce
```

- One JSON object per line, appended: `{"phase": NAME, "event": "start", "kind": KIND, "t": EPOCH}` and `{"phase": NAME, "event": "end", "t": EPOCH}`, where `t` is integer epoch seconds.
- `KIND` is `setup` (install, fetch, slice, bake, premise, self-tests), `measure` (anything a reading is taken from), `reduce` (the verdict on the box), or `other`. The writer refuses any other kind, and any name outside `[A-Za-z0-9_.:-]+`, with rc 2 and writes nothing. That includes a missing argument, under `set -u` too. A runner under `set -e` therefore STOPS on a refused stamp. Stamp names and kinds are constants in the runner, so a refusal is a kit bug, which the $0 shakedown must catch before a paid box. Don't wrap stamps in `|| true` to hide one.
- Phases run one at a time: a phase ends before the next starts, and `t` never goes backwards.
- P149's runner wrote the same lines without `kind` before this convention existed. The reader classes those `unknown`; it never guesses a kind from a name.

`phases.py PATH` prints seconds per kind, the first measurement's start, and any phase left open (a box that died mid-phase). `phases.py --check PATH` exits 1 with one line per fault: a format fault, or a phase left open (on a fetched receipt the box is gone, so an open phase means it died). Verifiers of fetched receipts run plain `--check`. `--check --allow-open` is for a live box, where the current phase is legitimately open. `tests/test_phases_common.py` reads P149's committed record first (sha-pinned), then checks the writer's refusals and each reader fault.

**Boot is not in `phases.jsonl`, and never `ssh_ready_s`.** The time from the create to a usable box is the launcher's, read from its receipt:
- `<provider>_ssh_ready_after_s`, from the create to the first authenticated ssh, where the receipt has it;
- else `<provider>_preflight_seconds`, a different quantity: the pre-flight's own duration, from its start rather than the
  create, probes included. A boot figure states which field it came from.

`ssh_ready_s` is the configured readiness ALLOWANCE (`--ssh-ready-s`; newer receipts also write it as `ssh_ready_allowance_s`), not a measurement. It takes only the configured values (180, 300, 420 s), so a ratio of it to the billed time measures nothing. A receipt with neither measured field has an UNKNOWN boot, never 0 and never the allowance.
