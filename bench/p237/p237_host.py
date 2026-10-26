#!/usr/bin/env python3
"""p237_host.py -- the runner's small decisions, as functions the tests drive (standard library only; the box's system python3 runs
it before any install):

- ``cpu <model name>``: the host CPU class floor (PREREG "The host floors"). Fast classes only: AMD EPYC 9xx5 (Zen 5), Ryzen 7000 or
  9000 desktop, Intel Core 13th/14th gen or Core Ultra desktop. EPYC Zen 2-4, Xeon, Threadripper and mobile parts are refused.
  Prints the class; exit 0 admitted, 18 refused (the launcher's host-floor code: the runner passes it through).
- ``ram_gib``: rc 16's host RAM (amendment 1): the whole GiB this process can use, min(MemTotal, every cgroup memory limit on its
  path), then a one-token record (``memtotal=..GiB,cgroup=..GiB|none``). Exit 1, printing nothing on stdout, when a value is unread.
- ``cap_gib <total bytes>``: the full card's cap as the exact float of total / 2**30, so tc1_arm's fraction cap * 2**30 / total is
  exactly 1.0 (total < 2**53, so the quotient is a dyadic rational a double holds exactly, and repr round-trips it).
- ``nauto <receipt> <log>``: UA's resolved swap from its tc1_arm receipt. Prints N (an engaged swap), ``0`` (a refused row whose
  reason names an empty or absent swap AND whose log carries one of Unsloth 2026.10.8's not-engaged lines), or ``unread``.
- ``probe <receipt> <cap_gib>``: one UT probe's outcome: ``ok <max_reserved>`` (all steps ran and the reserved peak is at most the cap
  minus 512 MiB), ``over <max_reserved>``, ``oom``, or ``fault <status>``.
- ``tune <n_auto> <outcome>...``: N_t from the probes' outcomes in probe order (N_auto - 1, - 2, ...), by the PREREG's rule.
"""
from __future__ import annotations

import json
import re
import sys

HOST_FLOOR = 18
# The guard fit (the maintainer's ruling before data; the runner's 41 check and the PREREG's guard and cost lines are this, and
# tests/test_p237.py ties the three): guard = (INSTALL + FETCH + N_PROC x LOAD_S + RESERVE) x 1.25, against a cap of
# min(the run-worst USD 15 at the box rate, the per-run wall-clock override's 18 h). The rental policy's GLOBAL cap keeps
# teardown.max_wallclock_h at 6 (the launcher refuses any longer launch); a rental-policy decision (2026-10-26) added a PER-RUN
# override up to 18 h, allowed only when the run's worst at its guard is <= USD 15. Both are registered here as constants, pinned by
# the test, and move only with that policy. A box longer than 6 h launches only once the override is in the policy.
INSTALL_S, FETCH_S, RESERVE_S, FIT_MARGIN_PCT = 2700, 3600, 900, 125
USD_PER_H, DOWNLOAD_USD, RUN_CAP_USD = 0.60, 1.10, 15.0
POLICY_MAX_WALLCLOCK_H = 6                             # the global cap: every launch without the per-run override
RUN_OVERRIDE_MAX_H = 18                                # the per-run override's ceiling (rental policy, 2026-10-26)
PROBES_MAX = {"read": 4, "shake": 1}                   # the shakedown runs ONE probe (N_auto - 1): it tests the probe path, never tunes


def n_proc(mode):
    """The processes a box runs: every primer and draw in the schedule, plus the probes it may run (PROBES_MAX per probe slot)."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p237_schedule
    rows = p237_schedule.schedule(shake=mode == "shake")
    return sum(1 for r in rows if r["kind"] != "probes") + PROBES_MAX[mode] * sum(1 for r in rows if r["kind"] == "probes")


def fit_s(nproc, load_s):
    """The guard fit in whole seconds, exactly as bash computes it: ((INSTALL + FETCH + N x LOAD_S + RESERVE) * 125) / 100."""
    return (INSTALL_S + FETCH_S + int(nproc) * int(load_s) + RESERVE_S) * FIT_MARGIN_PCT // 100


def cap_s():
    """The longest guard a box may hold: the per-run override's ceiling, or the run-worst USD bound at the box rate, whichever is
    less (the override is allowed only when the worst is <= USD 15, so the USD bound binds with it)."""
    return int(min(RUN_OVERRIDE_MAX_H * 3600, (RUN_CAP_USD - DOWNLOAD_USD) / USD_PER_H * 3600))


def worst_usd(guard_s):
    return round(guard_s / 3600 * USD_PER_H + DOWNLOAD_USD, 2)


# Amendment 1 (pre-data): the shake's launch wall. pod-launch boots, waits for SSH and stages before the runner's rc 41 reads LEFT, so
# the wall a box asks for is the guard fit plus a launch slack (the maintainer's ruling, 2026-10-26, P238's precedent); the shake's is
# 16,500 + 1,500 = 18,000 s (5.0 h), under the global 6 h cap, no override.
LAUNCH_SLACK_S = 1500


def wall_s(mode, load_s):
    """The launch wall in whole seconds: the guard fit for ``mode`` at ``load_s``, plus the launch slack."""
    return fit_s(n_proc(mode), load_s) + LAUNCH_SLACK_S
PROBE_STEPS, MARGIN_BYTES = 8, 512 * 2 ** 20
#: Unsloth 2026.10.8's own lines when ``offload_layers="auto"`` swaps nothing (models/_utils.py:9703-9705, models/loader_utils.py:1045,
#: models/_utils.py:9015), matched by prefix
NOT_ENGAGED = ("Unsloth: offload_layers = 'auto' swaps nothing", "Unsloth: offload_layers = 'auto' loads every layer onto the GPU")
CPU_CLASSES = (
    ("epyc-zen5", re.compile(r"\bAMD EPYC 9\d\d5[A-Z]?\b")),
    ("ryzen-7000-9000-desktop", re.compile(r"\bAMD Ryzen [3579] [79]\d\d\d(X3D|X|F)?(\s|$)")),
    ("core-13-14-desktop", re.compile(r"\bCore\(TM\) i[3579]-1[34]\d\d\d(K|KF|KS|F)?(\s|$)")),
    ("core-ultra-desktop", re.compile(r"\bCore\(TM\) Ultra [579] 2\d\d(K|KF|F)?(\s|$)")),
)


def cpu_class(name):
    """The fast class ``name`` belongs to, or None. A mobile suffix (H, HX, U, V, P) never matches: each pattern ends the number at
    a desktop suffix or a space."""
    for cls, rx in CPU_CLASSES:
        if rx.search(name or ""):
            return cls
    return None


# Amendment 1 (pre-data): the pin screen's view of a Vast offer, so it refuses what the runner would. The runner's lscpu reading above
# is unchanged.
SCREEN_MIN_LISTED_RAM_GIB = 128                        # Vast's listed cpu_ram (the offer's SHARE of the host) >= the runner's floor


def vast_cpu_class(name):
    """The registered class of a Vast offer's ``cpu_name``. Vast writes the trade mark sign (U+2122) after "Core" and the registered
    sign (U+00AE) after "Xeon" where lscpu writes "Intel(R) Core(TM) i9-14900K", so the two marks are spelled as lscpu spells them and
    CPU_CLASSES decide, unchanged."""
    return cpu_class(str(name or "").replace("\u2122", "(TM)").replace("\u00ae", "(R)"))


def vast_ram_gib(cpu_ram_mb):
    """Vast's listed ``cpu_ram`` (MB: the offer's share of the host) in whole GiB, as the screen compares it."""
    return int(float(cpu_ram_mb or 0) / 1024)


# Amendment 1 (pre-data): rc 16 reads the RAM this process can use. A container's /proc/meminfo MemTotal is the whole host's, so a Vast
# slice of a 1 TB host read 1 TB while its cgroup held it to its share; the floor reads min(MemTotal, every cgroup memory limit on
# this process's path).
CGROUP_V1_UNLIMITED = 1 << 60                          # v1 writes "no limit" as a page-rounded 2**63 - 1; anything this large is none


def _limit(text):
    """One cgroup limit file's value in bytes, or None for no limit (v2 "max", v1's huge sentinel). Unparseable text raises."""
    t = text.strip()
    if t == "max":
        return None
    v = int(t)
    return None if v >= CGROUP_V1_UNLIMITED else v


def cgroup_limits(root="/sys/fs/cgroup", self_cgroup="/proc/self/cgroup"):
    """Every memory limit on this process's cgroup path, leaf to the visible root: [(file, bytes or None)]. cgroup v2 (the "0::"
    line) reads ``memory.max``; v1 (the line naming the memory controller) reads ``memory.limit_in_bytes`` under ``memory/``. Only a
    FileNotFoundError skips a file (the root of a hierarchy has none); any other OSError (an unreadable directory or file) and an
    unparseable value raise. Under a cgroup namespace, a limit above the namespace's root is not visible here."""
    import os
    try:
        lines = open(self_cgroup, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        lines = []
    found = []
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            base, fname = root, "memory.max"
        elif "memory" in parts[1].split(","):
            base, fname = os.path.join(root, "memory"), "memory.limit_in_bytes"
        else:
            continue
        rel = [p for p in parts[2].split("/") if p]
        for k in range(len(rel), -1, -1):            # the leaf, then each ancestor, then the hierarchy's own root
            f = os.path.join(base, *rel[:k], fname)
            try:                                     # never exists(): it answers False for ANY stat error, and absence here permits
                with open(f, encoding="utf-8") as fh:
                    text = fh.read()
            except FileNotFoundError:                # absent (a hierarchy's root has none); any other OSError raises: never "no limit"
                continue
            found.append((f, _limit(text)))
    return found


def usable_ram(meminfo_text, limits):
    """rc 16's reading: (whole GiB the process can use, a one-token record). min(MemTotal, the lowest limit); None when MemTotal is
    unread."""
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", meminfo_text, re.M)
    if not m:
        return None, "memtotal=unread"
    total = int(m.group(1)) * 1024
    caps = [v for _, v in limits if v is not None]
    low = min(caps) if caps else None
    use = total if low is None else min(total, low)
    rec = f"memtotal={total // 2 ** 30}GiB,cgroup={'none' if low is None else f'{low // 2 ** 30}GiB'}"
    return use // 2 ** 30, rec


def cap_gib(total_bytes):
    return repr(int(total_bytes) / 2 ** 30)


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def n_auto(cell, log_text):
    """N (int) when UA's swap engaged; 0 only for the refused-and-not-engaged row with Unsloth's own line in the log; None otherwise."""
    if not cell:
        return None
    rec = cell.get("unsloth_offload_layers") or {}
    res = rec.get("resolved")
    if cell.get("status") == "ok":
        n = (res or {}).get("n_layers")
        return n if isinstance(n, int) and n > 0 else None
    reason = str(cell.get("reason") or cell.get("why") or "")
    empty = "carries no block swap" in reason or "holds no layer" in reason
    line = any(ln.startswith(p) for ln in (log_text or "").splitlines() for p in NOT_ENGAGED)
    return 0 if cell.get("status") == "refused" and empty and line else None


def max_reserved(cell):
    rows = cell.get("step_mem") or []
    vals = [r.get("max_reserved") for r in rows]
    if not vals or any(not isinstance(v, int) for v in vals):
        return None
    return max(vals)


def probe_outcome(cell, cap):
    if not cell:
        return "fault none"
    st = cell.get("status")
    if st == "oom":
        return "oom"
    if st != "ok" or cell.get("steps") != PROBE_STEPS or len(cell.get("step_ms") or []) != PROBE_STEPS:
        return f"fault {st}"
    mr = max_reserved(cell)
    if mr is None:
        return "fault no-step-mem"
    return f"{'ok' if mr <= float(cap) * 2 ** 30 - MARGIN_BYTES else 'over'} {mr}"


def tune(nauto, outcomes):
    """The PREREG's tuning rule. Probes run at N_auto - 1, N_auto - 2, ... (outcomes in that order) and stop at the first OOM; N_t is
    the smallest N whose probe is ``ok``. No ``ok`` probe (the first one OOMed, or every probe ran over the margin) gives N_t = N_auto,
    reported as "tuned equals auto". A fault in a probe is not an outcome the rule ranks: it returns None (UT does not run)."""
    best = None
    for k, o in enumerate(outcomes, 1):
        kind = o.split()[0]
        if kind == "oom":
            break
        if kind == "fault":
            return None
        if kind == "ok":
            best = nauto - k
    return nauto if best is None else best


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "cpu" and rest:
        cls = cpu_class(" ".join(rest))
        print(cls or "refused")
        return 0 if cls else HOST_FLOOR
    if cmd == "fit" and len(rest) == 2 and rest[0] in PROBES_MAX and rest[1].isdigit():
        n = n_proc(rest[0])
        print(n, fit_s(n, int(rest[1])), cap_s(), PROBES_MAX[rest[0]])
        return 0
    if cmd == "ram_gib" and len(rest) in (0, 3):
        meminfo, self_cgroup, root = rest or ("/proc/meminfo", "/proc/self/cgroup", "/sys/fs/cgroup")
        try:
            gib, rec = usable_ram(open(meminfo, encoding="utf-8").read(), cgroup_limits(root, self_cgroup))
        except (OSError, ValueError) as e:              # an unread limit is never "no limit": the runner stops (9)
            print(f"ram unread: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if gib is None:
            print(f"ram unread: {rec}", file=sys.stderr)
            return 1
        print(gib, rec)
        return 0
    if cmd == "cap_gib" and len(rest) == 1:
        print(cap_gib(rest[0]))
        return 0
    if cmd == "nauto" and len(rest) == 2:
        try:
            log = open(rest[1], encoding="utf-8", errors="replace").read()
        except OSError:
            log = ""
        n = n_auto(_json(rest[0]), log)
        print("unread" if n is None else n)
        return 0
    if cmd == "probe" and len(rest) == 2:
        print(probe_outcome(_json(rest[0]), rest[1]))
        return 0
    if cmd == "tune" and rest:
        n = tune(int(rest[0]), rest[1:])
        print("unread" if n is None else n)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
