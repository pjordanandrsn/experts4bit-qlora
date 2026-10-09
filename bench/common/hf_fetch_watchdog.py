#!/usr/bin/env python3
"""A byte-growth watchdog around ``huggingface_hub.snapshot_download`` for rented boxes (P127 Amendment 2; e4b#1313).

Why: the hub's plain-HTTP path resumes a dead socket, but nothing bounds a fetch that stops making progress without an
error. Examples are a cache ``.lock`` held by another process (``WeakFileLock`` waits forever, logging only at INFO), a
connection trickling below the read timeout, or a long 429 ``Retry-After`` sleep. A lane's ``alarm`` then spends the
whole step budget waiting. The pattern reporters use (hub#4196, #4223, #4520):

- run the fetch in its own process group;
- every ``--poll-s`` seconds, sum the bytes under the repo's ``blobs/`` (``*.incomplete`` included);
- if nothing grows for ``--stall-s`` seconds, kill the whole group, prune the orphaned ``*.incomplete`` files (the hub
  has no cross-process resume since v1.18, so a rerun restarts in-flight shards and skips finished ones) and rerun;
- give up after ``--max-restarts`` reruns;
- on an exit 0, refuse the result while any ``*.incomplete`` remains (hub#4223 exited 0 with partials).

``--budget-s`` bounds the whole call. The snapshot directory is printed as the LAST line of stdout, so a runner's
``tail -1`` finds it; every watchdog line goes to stderr with a ``[watchdog]`` prefix.

Exit codes: 0 ok, 75 gave up after the restarts, 124 budget spent, 2 usage. ``--self-test`` runs the cases below
against fake fetchers in a temporary directory, with no network.

Standard library only; the child imports ``huggingface_hub``. Shared under ``bench/common/`` so other lanes (FAM, P129)
can adopt it by amendment.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time

EXIT_OK, EXIT_GAVE_UP, EXIT_BUDGET, EXIT_USAGE = 0, 75, 124, 2
CHILD_SRC = ("import json, sys\nfrom huggingface_hub import snapshot_download as s\n"
             "print(s(**json.loads(sys.argv[1])), flush=True)\n")


def log(msg: str) -> None:
    print(f"[watchdog] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", file=sys.stderr, flush=True)


def hub_cache() -> pathlib.Path:
    if os.environ.get("HF_HUB_CACHE"):
        return pathlib.Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return pathlib.Path(home) / "hub"


def blobs_dir(repo: str, cache: pathlib.Path | None = None) -> pathlib.Path:
    return (cache or hub_cache()) / ("models--" + repo.replace("/", "--")) / "blobs"


def blob_bytes(d: pathlib.Path) -> int:
    """Bytes under ``d``, ``*.incomplete`` included; 0 before the hub creates it. A file vanishing mid-scan is skipped."""
    total = 0
    if d.is_dir():
        for p in d.iterdir():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def partials(d: pathlib.Path) -> list[pathlib.Path]:
    return sorted(d.glob("*.incomplete")) if d.is_dir() else []


def hub_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("huggingface_hub")
    except Exception:  # noqa: BLE001 -- the version is recorded, never required here
        return "absent"


def _spawn(cmd: list[str]) -> subprocess.Popen:
    kw: dict = {"stdout": subprocess.PIPE, "stderr": None, "text": True, "errors": "replace"}
    if os.name == "posix":
        kw["start_new_session"] = True            # its own process group: the kill reaches every worker it starts
    else:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kw)


def _kill(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue
    else:
        proc.kill()
        proc.wait(timeout=10.0)


def run(cmd: list[str], measure_dir: pathlib.Path, *, poll_s: float = 30.0, stall_s: float = 180.0,
        max_restarts: int = 3, budget_s: float = 2700.0, grace_s: float = 10.0) -> tuple[int, dict]:
    """Run ``cmd`` under the watchdog; returns (exit code, record). ``record["snapshot"]`` is the child's last stdout line
    on success."""
    t0 = time.monotonic()
    rec: dict = {"attempts": [], "restarts": 0, "snapshot": None}
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            orphans = partials(measure_dir)
            for p in orphans:
                p.unlink(missing_ok=True)
            log(f"attempt {attempt}: pruned {len(orphans)} orphan .incomplete file(s)")
        proc = _spawn(cmd)
        last: list[str] = []

        def pump(stream=proc.stdout, sink=last):
            for line in stream:
                sys.stdout.write(line)
                sys.stdout.flush()
                if line.strip():
                    sink[:] = [line.strip()]
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        seen, grew_at, outcome = blob_bytes(measure_dir), time.monotonic(), None
        while outcome is None:
            try:
                proc.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                pass
            now, b = time.monotonic(), blob_bytes(measure_dir)
            if b > seen:
                seen, grew_at = b, now
            if proc.poll() is not None:
                outcome = "exit"
            elif now - t0 > budget_s:
                outcome = "budget"
            elif now - grew_at > stall_s:
                outcome = "stall"
        if outcome != "exit":
            log(f"attempt {attempt}: {outcome} ({seen} bytes, {now - grew_at:.0f} s without growth) -- killing the group")
            _kill(proc, grace_s)
        reader.join(timeout=5.0)
        rc = proc.returncode
        left = partials(measure_dir)
        snap = last[0] if last else None
        rec["attempts"].append({"attempt": attempt, "outcome": outcome, "rc": rc, "bytes": seen,
                                "partials": len(left), "s": round(time.monotonic() - t0, 1)})
        if outcome == "budget":
            log(f"budget {budget_s:.0f} s spent")
            return EXIT_BUDGET, rec
        if outcome == "exit" and rc == 0 and not left and snap and os.path.isdir(snap):
            rec["snapshot"] = snap
            log(f"ok after {attempt} attempt(s), {seen} bytes")
            return EXIT_OK, rec
        why = (f"exit {rc}" if outcome == "exit" and rc != 0 else
               f"exit 0 with {len(left)} .incomplete file(s)" if outcome == "exit" and left else
               f"exit 0 without a snapshot directory ({snap!r})" if outcome == "exit" else outcome)
        log(f"attempt {attempt} failed: {why}")
        if rec["restarts"] >= max_restarts:
            log(f"gave up after {max_restarts} restart(s)")
            return EXIT_GAVE_UP, rec
        rec["restarts"] += 1


# ------------------------------------------------------------------------------------------------- self-test --
FAKE = r'''
import os, pathlib, subprocess, sys, time
mode, blobs, snap, state = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
n = int(state.read_text()) if state.exists() else 0
state.write_text(str(n + 1))
blobs.mkdir(parents=True, exist_ok=True)
def steady():
    part = blobs / f"shard.{n}.incomplete"
    for _ in range(5):
        with open(part, "ab") as f: f.write(b"x" * 1000)
        time.sleep(0.02)
    part.rename(blobs / "shard")
    snap.mkdir(parents=True, exist_ok=True)
    print(snap, flush=True); sys.exit(0)
def stall():
    with open(blobs / f"shard.{n}.incomplete", "ab") as f: f.write(b"x" * 500)
    time.sleep(3600)
if mode == "steady": steady()
if mode == "stall_once": stall() if n == 0 else steady()
if mode == "stall_always": stall()
if mode == "error_once":
    if n == 0: sys.exit(1)
    steady()
if mode == "partial_ok_once":
    if n == 0:
        (blobs / "left.incomplete").write_bytes(b"x" * 10); snap.mkdir(parents=True, exist_ok=True); print(snap); sys.exit(0)
    steady()
if mode == "trickle":
    while True:
        with open(blobs / "slow.incomplete", "ab") as f: f.write(b"x")
        time.sleep(0.05)
if mode == "grandchild":
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    (state.parent / "grandchild.pid").write_text(str(g.pid))
    stall()
'''


def _gone(pid: int, within_s: float) -> bool:
    """True once ``pid`` no longer runs. A zombie counts as gone: the kill reached it, and its reaping is up to PID 1. In
    a container PID 1 is often a shell that reaps late or never."""
    deadline = time.monotonic() + within_s
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            if pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except (OSError, IndexError):
            pass
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def self_test(stall_s: float = 8.0) -> int:
    """Deterministic under load. The stall timer starts at spawn, so an attempt meant to finish must finish well inside
    ``stall_s`` even when the fake's interpreter starts slowly on a busy host. A fake writes for about 0.1 s after
    startup, so 8 s leaves orders of magnitude for scheduling delay. Only the fakes built to stall (they write once,
    then sleep for an hour) can reach it. A case therefore never gains an unplanned restart, and the counts asserted
    are exact. The budget case is time-bound by design: a trickle never stalls and never ends."""
    fast = {"poll_s": 0.1, "stall_s": stall_s, "grace_s": 1.0}
    cases = [  # (mode, run kwargs, expected rc, expected restarts)
        ("steady", {"max_restarts": 3, "budget_s": 120}, EXIT_OK, 0),
        ("stall_once", {"max_restarts": 3, "budget_s": 120}, EXIT_OK, 1),
        ("stall_always", {"max_restarts": 1, "budget_s": 120}, EXIT_GAVE_UP, 1),
        ("error_once", {"max_restarts": 3, "budget_s": 120}, EXIT_OK, 1),
        ("partial_ok_once", {"max_restarts": 3, "budget_s": 120}, EXIT_OK, 1),
        ("trickle", {"max_restarts": 3, "budget_s": 1.5}, EXIT_BUDGET, 0),
    ]
    if os.name == "posix":
        cases.append(("grandchild", {"max_restarts": 0, "budget_s": 120}, EXIT_GAVE_UP, 0))
    bad = 0
    for mode, kw, want_rc, want_restarts in cases:
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            fake = td / "fake.py"
            fake.write_text(FAKE)
            blobs, snap, state = td / "blobs", td / "snap", td / "state"
            cmd = [sys.executable, str(fake), mode, str(blobs), str(snap), str(state)]
            rc, rec = run(cmd, blobs, **fast, **kw)
            ok = rc == want_rc and rec["restarts"] == want_restarts
            if rc == EXIT_OK:
                ok = ok and rec["snapshot"] == str(snap) and not partials(blobs)
            if mode == "stall_once":
                ok = ok and not (blobs / "shard.0.incomplete").exists()          # the orphan was pruned
            if mode == "grandchild":
                gpid = int((td / "grandchild.pid").read_text())
                ok = ok and _gone(gpid, within_s=10.0)                              # the group kill reached it
            print(f"{'ok  ' if ok else 'FAIL'} {mode}: rc={rc} restarts={rec['restarts']} attempts={len(rec['attempts'])}")
            bad += not ok
    n = len(cases)
    print(f"hf_fetch_watchdog self-test {'OK' if not bad else 'FAILED'} ({n - bad}/{n} cases)")
    return 0 if not bad else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--revision")
    ap.add_argument("--allow", action="append", default=None, help="an allow_patterns entry (repeatable)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--stall-s", type=float, default=180.0)
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=2700.0)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.repo and a.revision):
        ap.print_usage(sys.stderr)
        return EXIT_USAGE
    kwargs = {"repo_id": a.repo, "revision": a.revision, "max_workers": a.max_workers}
    if a.allow:
        kwargs["allow_patterns"] = a.allow
    d = blobs_dir(a.repo)
    log(f"huggingface_hub {hub_version()}; {a.repo}@{a.revision}; blobs {d}; poll {a.poll_s:.0f} s, stall {a.stall_s:.0f} s, "
        f"restarts <= {a.max_restarts}, budget {a.budget_s:.0f} s")
    rc, rec = run([sys.executable, "-c", CHILD_SRC, json.dumps(kwargs)], d, poll_s=a.poll_s, stall_s=a.stall_s,
                  max_restarts=a.max_restarts, budget_s=a.budget_s)
    log("record " + json.dumps(rec, sort_keys=True))
    if rc == EXIT_OK:
        print(rec["snapshot"], flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
