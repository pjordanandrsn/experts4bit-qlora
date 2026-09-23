#!/usr/bin/env python3
"""The CI tooling both repositories carry is ONE file, byte-identical in both.

experts4bit-qlora and grouped-nf4-gemm are one system released as two packages
(``docs/system-manifest.json``). Their CI scripts were copied from one to the
other and then edited separately. By 2026-09-23 nine of the thirteen scripts
present in both under the same name had forked -- ``check_system_manifest.py``
by 734 diff lines -- so the same check name enforced two different rules, and a
fix made in one repository did not reach the other. ``SHARED`` lists the
scripts that are one file again; this check keeps them that way.

Direction: grouped-nf4-gemm is the upstream for shared tooling, as it is for the
manifest (kernel-first). A change to a shared script lands there first; the
runtime repository then copies it, and its CI runs this script with
``--sibling`` pointed at the kernel repository's ``main``. A shared script that
genuinely needs to behave differently per repository reads the difference from
data (its own ``pyproject.toml``, the manifest), never from two copies.

Without ``--sibling`` it checks only that every ``SHARED`` path exists here, so
a rename cannot silently drop a file from the set. With ``--sibling`` it also
fails when the sibling is not the other package of this system (a checkout of
this same repository would compare equal to itself and prove nothing), when a
shared file is missing there, or when any shared file differs by a byte. Scripts
present in both repositories under the same name but not yet in ``SHARED`` are
listed as NOTEs -- the forks still to reconcile -- and do not fail.

Usage::

    python scripts/check_shared_tooling.py                     # SHARED exists here
    python scripts/check_shared_tooling.py --sibling ../other  # and matches the sibling
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

#: Repository-relative paths that must be byte-identical in both repositories.
#: Add a script here only once the two copies ARE one file (``cmp`` says so);
#: this file itself is the first entry, so the list cannot fork either.
SHARED: tuple[str, ...] = (
    "scripts/check_shared_tooling.py",
    "scripts/build_llms_bundle.py",
    "scripts/check_discovery_contract.py",
    "scripts/check_docs_examples.py",
    "scripts/check_readme_links.py",
    "scripts/check_wheel_metadata.py",
    "scripts/discovery_common.py",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_name(root: Path) -> str | None:
    try:
        with open(root / "pyproject.toml", "rb") as f:
            return str(tomllib.load(f)["project"]["name"])
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
        return None


def _system_packages(root: Path) -> set[str]:
    """The package names ``docs/system-manifest.json`` says make up the system."""
    try:
        manifest = json.loads((root / "docs" / "system-manifest.json").read_text(encoding="utf-8"))
        return {str(p["package"]) for p in manifest["packages"].values()}
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return set()


def check(root: Path, sibling: Path | None) -> tuple[list[str], list[str], list[str]]:
    """Return ``(ok, notes, failures)`` lines."""
    ok: list[str] = []
    notes: list[str] = []
    fail: list[str] = []
    if not SHARED:
        return ok, notes, ["SHARED is empty: this check would pass on anything"]

    missing_here = [p for p in SHARED if not (root / p).is_file()]
    if missing_here:
        fail.append(f"shared file(s) missing here: {missing_here}")
    else:
        ok.append(f"{len(SHARED)} shared file(s) present here")
    if sibling is None:
        return ok, notes, fail

    ours, theirs = _package_name(root), _package_name(sibling)
    system = _system_packages(root)
    if theirs is None:
        fail.append(f"sibling {sibling} has no readable pyproject.toml project.name")
        return ok, notes, fail
    if theirs == ours:
        fail.append(f"sibling {sibling} is {theirs!r}, the same package as this repository; "
                    "comparing a checkout with itself proves nothing")
        return ok, notes, fail
    if system and theirs not in system:
        fail.append(f"sibling package {theirs!r} is not one of this system's packages {sorted(system)}")
        return ok, notes, fail
    ok.append(f"sibling {sibling} is {theirs!r}")

    differing = []
    for rel in SHARED:
        here, there = root / rel, sibling / rel
        if not there.is_file():
            fail.append(f"{rel}: missing in the sibling")
        elif here.is_file() and _digest(here) != _digest(there):
            differing.append(rel)
    if differing:
        fail.append(f"{len(differing)} shared file(s) differ from the sibling: {differing}. grouped-nf4-gemm is "
                    "upstream for shared tooling: land the change there, then copy the file byte-for-byte")
    elif not any(f.endswith("missing in the sibling") for f in fail):
        ok.append(f"all {len(SHARED)} shared file(s) are byte-identical in the sibling")

    shared_names = {Path(p).name for p in SHARED}
    here_scripts = {p.name for p in (root / "scripts").glob("*.py")}
    there_scripts = {p.name for p in (sibling / "scripts").glob("*.py")}
    for name in sorted((here_scripts & there_scripts) - shared_names):
        same = _digest(root / "scripts" / name) == _digest(sibling / "scripts" / name)
        notes.append(f"scripts/{name} is in both repositories but not in SHARED"
                     + (" (already identical: add it to SHARED)" if same else " (forked; not yet reconciled)"))
    return ok, notes, fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="this repository's root (default: .)")
    ap.add_argument("--sibling", default=None, help="a checkout of the other package's repository")
    a = ap.parse_args(argv)
    ok, notes, fail = check(Path(a.root).resolve(), Path(a.sibling).resolve() if a.sibling else None)
    for line in ok:
        print(f"OK: {line}")
    for line in notes:
        print(f"NOTE: {line}")
    for line in fail:
        print(f"FAIL: {line}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
