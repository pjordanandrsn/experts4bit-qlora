#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's pre-rental fetch gate (bench/q235/PREREG-q235.md, "Before any rental", gates 3 and 5). From a CPU host, before the rental
controller can quote, every locked thing the box will fetch is resolved, and the files the box runs are scanned for unfilled values.
Any refusal stops the launch. It never installs, never downloads a whole file, never quotes and never rents. The resolvers and the
listing checks are lane P130's (``bench/p130/p130_fetch_gate.py``); the locks are Q235's.

- **The checkpoint.** ``model_info(files_metadata=True)`` at the locked revision must list exactly ``model_files.json``'s rows
  (name, size, blob id, LFS sha256). Each file's resolve HEAD must answer with that commit, size and etag (P130's
  ``check_listing``), and every shard the index names must be locked (``check_shards``).
- **The wheels.** Every ``wheels.json`` URL, all from allowlisted origins, must answer a one-byte range GET.
- **The git pins.** grouped-nf4-gemm at ``GNF4_SHA`` and this repository at the launch commit must each fetch, by sha, exactly
  that commit.
- **No placeholders.** The sealed closure, the locks, the drive and the registration (``PREREG-q235.md``) must carry no unfilled
  value (TODO, TBD, FIXME, XXX, PLACEHOLDER, ``<NAME>``, ``{{ name }}``, ``???``).

    python q235_fetch_gate.py --e4b-sha SHA --out GATE.json            (exit 0 passed, 1 refused, 2 usage)
    python q235_fetch_gate.py --e4b-sha SHA --check-report GATE.json   (the drive's check: passed, this commit, under a day old)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GNF4_SHA = "724ccc454f006c1a46836e434e997f31f293747f"
GIT = {"gnf4": "https://github.com/pjordanandrsn/grouped-nf4-gemm.git",
       "e4b": "https://github.com/pjordanandrsn/experts4bit-qlora.git"}
MAX_AGE_S = 24 * 3600
SCHEMA = "q235-fetch-gate/1"
PLACEHOLDER = re.compile(r"\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b|\bPLACEHOLDER\b|\bCHANGEME\b|\bREPLACE_ME\b|<[A-Z][A-Z0-9_]{2,}>"
                         r"|\{\{\s*[\w.]+\s*\}\}|\?\?\?")


def _p130():
    spec = importlib.util.spec_from_file_location("p130_fetch_gate_for_q235", REPO / "bench" / "p130" / "p130_fetch_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage():
    spec = importlib.util.spec_from_file_location("q235_stage_for_gate", HERE / "q235_stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the pure checks (the tests drive these with fixtures) -----------------------------------------------------------

def check_against_lock(listing: list, lock: dict) -> list:
    """The hub's listing at the revision against the lock: the same names, and for each the same size, blob id and LFS sha256."""
    have = {r["file"]: r for r in listing}
    want = {r["file"]: r for r in lock["files"]}
    out = [f"{n}: locked, not listed at the revision" for n in sorted(set(want) - set(have))]
    out += [f"{n}: listed at the revision, not locked" for n in sorted(set(have) - set(want))]
    for n in sorted(set(want) & set(have)):
        for k in ("size", "blob_id", "lfs_sha256"):
            if have[n].get(k) != want[n].get(k):
                out.append(f"{n}: {k} {have[n].get(k)} against the lock's {want[n].get(k)}")
    return out


def check_probes(rows: list, probes: dict) -> list:
    """Every locked wheel URL answered its one-byte GET."""
    out = []
    for r in rows:
        p = probes.get(r["url"]) or {"error": "not probed"}
        if "error" in p or not (200 <= int(p.get("status") or 0) < 300) or p.get("bytes") != 1:
            out.append(f"{r['name']}: {p.get('status')} {p.get('error', 'answered ' + str(p.get('bytes')) + ' bytes')}")
    return out


def placeholder_files() -> list:
    stage = _stage()
    files = [REPO / p for p in stage.CLOSURE]
    files += [HERE / n for n in ("q235.sha256", "locked_inputs.json", "q235_drive.sh", "q235_stage.py", "PREREG-q235.md")]
    files += sorted(HERE.glob("AMENDMENT*.md"))
    # Not scanned: TC1's instrument (registered and reviewed in its own lane) and this scanner, whose text names the markers.
    return [f for f in files if f.name != "tc1_arm.py"]


def scan_placeholders(files: list) -> list:
    out = []
    for f in files:
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for m in PLACEHOLDER.finditer(line):
                out.append(f"{f.relative_to(REPO)}:{i}: unfilled value {m.group(0)!r}")
    return out


# ---- the gate ---------------------------------------------------------------------------------------------------------

def gate(e4b_sha: str) -> dict:
    from huggingface_hub import HfApi
    p130 = _p130()
    lock = json.loads((HERE / "model_files.json").read_text())
    wheels = json.loads((HERE / "wheels.json").read_text())
    rep = {"schema": SCHEMA, "generated_at": int(time.time()), "e4b_sha": e4b_sha, "gnf4_sha": GNF4_SHA,
           "model": {"repo": lock["repo"], "revision": lock["revision"]}, "refusals": []}
    names = [f["file"] for f in lock["files"]]
    commit, listing, selected, heads = p130._hub_repo(HfApi(), lock["repo"], lock["revision"], "model", allow=names)
    rep["model"].update(resolved=commit, listed=len(listing), selected=len(selected))
    refusals = [] if commit == lock["revision"] else [f"the revision resolved to {commit}, not {lock['revision']}"]
    refusals += check_against_lock(listing, lock)
    refusals += p130.check_listing(listing, selected, heads, lock["revision"])
    idx_meta = next((r for r in listing if r["file"] == "model.safetensors.index.json"), None)
    if idx_meta is not None:
        from huggingface_hub import hf_hub_download
        idx = json.loads(Path(hf_hub_download(lock["repo"], "model.safetensors.index.json", revision=lock["revision"])).read_text())
        refusals += p130.check_shards(idx, selected)
    probes = {r["url"]: p130._probe(r["url"]) for r in wheels["packages"]}
    refusals += check_probes(wheels["packages"], probes)
    rep["wheels"] = {"count": len(wheels["packages"]), "failed": sum(1 for r in wheels["packages"] if "error" in probes[r["url"]])}
    got = {"gnf4": p130._git(GIT["gnf4"], GNF4_SHA), "e4b": p130._git(GIT["e4b"], e4b_sha)}
    rep["git"] = got
    refusals += p130.check_git(got, {"gnf4": GNF4_SHA, "e4b": e4b_sha})
    files = placeholder_files()
    rep["placeholders"] = {"files": len(files)}
    refusals += scan_placeholders(files)
    rep["refusals"] = refusals
    rep["passed"] = not refusals
    return rep


def report_ok(rep: dict, e4b_sha: str, now: float | None = None):
    """None when the report lets ``e4b_sha`` launch, else why not."""
    now = time.time() if now is None else now
    if rep.get("schema") != SCHEMA:
        return f"schema {rep.get('schema')}"
    if rep.get("passed") is not True or rep.get("refusals"):
        return f"the gate refused: {(rep.get('refusals') or ['not passed'])[:3]}"
    if rep.get("e4b_sha") != e4b_sha:
        return f"the gate ran for {rep.get('e4b_sha')}, not {e4b_sha}"
    age = now - float(rep.get("generated_at") or 0)
    if not 0 <= age <= MAX_AGE_S:
        return f"the gate report is {int(age)} s old (at most {MAX_AGE_S})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e4b-sha", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--check-report", type=Path)
    a = ap.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", a.e4b_sha):
        print("refusing: --e4b-sha is not a 40-hex commit", file=sys.stderr)
        sys.exit(2)
    if a.check_report:
        why = report_ok(json.loads(a.check_report.read_text()), a.e4b_sha)
        print(why or "ok", file=sys.stderr)
        sys.exit(1 if why else 0)
    rep = gate(a.e4b_sha)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"passed": rep["passed"], "refusals": rep["refusals"][:10]}))
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
