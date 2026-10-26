#!/usr/bin/env python3
"""p237_host.py -- the runner's small decisions, as functions the tests drive (standard library only; the box's system python3 runs
it before any install):

- ``cpu <model name>``: the host CPU class floor (PREREG "The host floors"). Fast classes only: AMD EPYC 9xx5 (Zen 5), Ryzen 7000 or
  9000 desktop, Intel Core 13th/14th gen or Core Ultra desktop. EPYC Zen 2-4, Xeon, Threadripper and mobile parts are refused.
  Prints the class; exit 0 admitted, 18 refused (the launcher's host-floor code: the runner passes it through).
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
# min(the run-worst USD 15 at the box rate, the rental policy's teardown.max_wallclock_h). The policy value is adertha's GLOBAL
# compute policy (adertha/compute/policy.json, "teardown": {"max_wallclock_h": 6}; rent.py's check_wallclock refuses any launch above
# it); it is registered here as a constant, pinned by the test, and moves only with that policy (the owner's call).
INSTALL_S, FETCH_S, RESERVE_S, FIT_MARGIN_PCT = 2700, 3600, 900, 125
USD_PER_H, DOWNLOAD_USD, RUN_CAP_USD = 0.60, 1.10, 15.0
POLICY_MAX_WALLCLOCK_H = 6
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
    """The longest guard a box may hold: the policy's wall clock, or the run-worst USD bound at the box rate, whichever is less."""
    return int(min(POLICY_MAX_WALLCLOCK_H * 3600, (RUN_CAP_USD - DOWNLOAD_USD) / USD_PER_H * 3600))


def worst_usd(guard_s):
    return round(guard_s / 3600 * USD_PER_H + DOWNLOAD_USD, 2)
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
