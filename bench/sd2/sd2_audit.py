#!/usr/bin/env python3
"""sd2_audit.py -- lane SD2 (e4b#1313): the read's package-diff audit (bench/sd2/PREREG-sd2.md, Amendment 3; P129
Amendment 4's pattern).

The read's target is merged main, not the proven integration commit `539a2d26`. Every `experts4bit_qlora` file that
differs between the two must be in `audit_read.tsv`, each with a one-line reason it is off the SD2 serve path (or on it).
Stage 0 (the target's GPU tests) and V0 re-prove the build on the real target, and V0 fails closed.

`--repo R --from A --to B --list audit_read.tsv [--out audit.json]` exits 0 when the changed files equal the list, and
31 otherwise. It names every changed file that is not listed, and every listed file that no longer differs (a stale
list: the pin moved). The driver runs it before a launch, and the box runs it again on its own clone, keeping
`git diff --stat` in its receipts.

`--self-test` checks the comparison and the list's format on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PACKAGE = "experts4bit_qlora/"
WHERE = ("off", "on")


def read_list(path) -> dict:
    """``audit_read.tsv``: one row per file, ``path<TAB>off|on<TAB>reason``; ``#`` lines are comments."""
    out = {}
    for n, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[1] not in WHERE or not parts[0].startswith(PACKAGE) or not parts[2].strip():
            raise ValueError(f"{path}:{n}: expected 'experts4bit_qlora/<file><TAB>off|on<TAB>reason', got {line!r}")
        if parts[0] in out:
            raise ValueError(f"{path}:{n}: {parts[0]} is listed twice")
        out[parts[0]] = (parts[1], parts[2].strip())
    return out


def audit(changed, listed) -> dict:
    changed, listed = sorted(set(changed)), set(listed)
    unlisted = [f for f in changed if f not in listed]
    stale = sorted(listed - set(changed))
    return {"ok": not unlisted and not stale, "changed": changed, "unlisted": unlisted, "stale": stale}


def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def changed_files(repo, a, b) -> list:
    return [f for f in _git(repo, "diff", "--name-only", a, b, "--", PACKAGE).splitlines() if f.strip()]


def self_test() -> int:
    bad = []
    if not audit(["experts4bit_qlora/a.py"], {"experts4bit_qlora/a.py": 1})["ok"]:
        bad.append("equal")
    got = audit(["experts4bit_qlora/a.py", "experts4bit_qlora/b.py"], {"experts4bit_qlora/a.py": 1})
    if got["ok"] or got["unlisted"] != ["experts4bit_qlora/b.py"]:
        bad.append("unlisted")
    got = audit([], {"experts4bit_qlora/a.py": 1})
    if got["ok"] or got["stale"] != ["experts4bit_qlora/a.py"]:
        bad.append("stale")
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_read.tsv")
    if os.path.exists(here):
        try:
            if not read_list(here):
                bad.append("the list is empty")
        except ValueError as e:
            bad.append(str(e))
    if bad:
        print("sd2_audit self-test FAILED:", bad)
        return 1
    print("sd2_audit self-test OK (equal, unlisted, stale, the list's format)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--from", dest="a")
    ap.add_argument("--to", dest="b")
    ap.add_argument("--list")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.repo and a.a and a.b and a.list):
        ap.print_usage()
        return 2
    listed = read_list(a.list)
    rec = audit(changed_files(a.repo, a.a, a.b), listed)
    rec.update({"from": a.a, "to": a.b, "list": {f: list(v) for f, v in listed.items()},
                "stat": _git(a.repo, "diff", "--stat", a.a, a.b, "--", PACKAGE)})
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    print("SD2_AUDIT " + ("ok " if rec["ok"] else "REFUSED ")
          + json.dumps({"changed": len(rec["changed"]), "unlisted": rec["unlisted"], "stale": rec["stale"]}), flush=True)
    return 0 if rec["ok"] else 31


if __name__ == "__main__":
    sys.exit(main())
