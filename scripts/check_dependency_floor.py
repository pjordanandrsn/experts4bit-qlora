#!/usr/bin/env python3
"""Dependency-floor drift: every CURRENT document that states the
grouped-nf4-gemm floor of the ``fast`` extra must state the floor that
pyproject.toml carries. Standard library only; no network.

The floor is the single ``>=`` clause of the grouped-nf4-gemm requirement in
``[project.optional-dependencies] fast``. Documents scanned: README.md,
AGENTS.md, llms.txt, docs/SOLUTIONS.md, docs/INDEX.md, docs/solutions/*.md,
docs/capabilities.json, .github/workflows/ci.yml (the ``--requires``
assertion), every document docs/INDEX.md lists under its "Current" heading,
and docs/system-manifest.json -- for the manifest only the ``compatibility``
record whose ``consumer_versions`` contains this version is read; older
records are history by construction.

A version statement is ``grouped-nf4-gemm>=X``, ``grouped-nf4-gemm >= X``,
``grouped-nf4-gemm ≥ X``, ``requires grouped-nf4-gemm X``, or a bare ``>= X``
(``pins >=X``, ``≥ X``) on a line that names grouped-nf4-gemm, gnf4 or the
``fast`` extra. A ``>=`` that belongs to another dependency of this project
(``torch>=2.2``, ``bitsandbytes >= 0.50.0``) is not one. Every statement must
equal the pyproject floor unless the LINE is explicitly historical: it carries
one of ``HISTORICAL_MARKERS`` as a whole word, or pins a release as a
``vX.Y.Z`` tag. Excused lines are printed, so a reviewer sees what was
excused. Never scanned: CHANGELOG.md, receipts under bench/, anchored
documents (a sibling ``.ots`` file or the ``<!-- ots-attestation-footer -->``
marker), the blockquoted examples of docs/RELEASE_NOTES_GUIDE.md, and
pyproject.toml's own comment ladder (the source is not a document).

Exit 1 on any drift; 2 when the check cannot run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_system_manifest import (  # noqa: E402
    EXTRA, KERNEL_PACKAGE, MANIFEST, current_record, fast_requirement, floor_of, parse_version,
)
from discovery_common import load_pyproject, pep503_name, read_text  # noqa: E402

#: Always scanned (when present), in addition to the "Current" list in docs/INDEX.md.
FIXED_DOCUMENTS = ("README.md", "AGENTS.md", "llms.txt", "docs/SOLUTIONS.md", "docs/INDEX.md",
                   "docs/capabilities.json", ".github/workflows/ci.yml")
SOLUTION_GLOB = "docs/solutions/*.md"
INDEX = "docs/INDEX.md"
RELEASE_NOTES_GUIDE = "docs/RELEASE_NOTES_GUIDE.md"
ANCHOR_MARKER = "<!-- ots-attestation-footer -->"

#: A line carrying one of these (whole words, case-insensitive) is describing
#: history, not stating the current floor. Extend deliberately: every marker
#: is a way for a stale line to be excused, and excused lines are printed.
HISTORICAL_MARKERS = ("raised from", "raised to", "landed in", "first shipped", "first released",
                      "below the current", "no longer", "used to", "at the time", "until", "older",
                      "was", "were", "history", "historical")

#: ``name >= 1.2.3`` / ``name>=1.2.3`` / ``>= 1.2.3`` -- the name is optional and may be prose.
_STATEMENT = re.compile(r"(?:(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(?P<gap>\s*))?(?:>=|≥)\s*v?(?P<ver>\d+\.\d+\.\d+)")
_REQUIRES = re.compile(r"\brequires\s+grouped-nf4-gemm\s+v?(?P<ver>\d+\.\d+\.\d+)", re.I)
_TAG_PIN = re.compile(r"(?<![A-Za-z0-9])v\d+\.\d+\.\d+(?![A-Za-z0-9])")
_MENTIONS = re.compile(r"grouped-nf4-gemm|(?<![a-z0-9])gnf4(?![a-z0-9])|\[fast\]|`fast`|\bfast\s+extra", re.I)
_KERNEL_NAMES = frozenset({pep503_name(KERNEL_PACKAGE), "gnf4"})


def historical_marker(line: str) -> str | None:
    """The first HISTORICAL_MARKER the line carries as a whole word, else None."""
    low = line.lower()
    for m in HISTORICAL_MARKERS:
        if re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", low):
            return m
    if _TAG_PIN.search(line):
        return "vX.Y.Z tag pin"
    return None


def statements(line: str, other_packages: frozenset[str]) -> list[str]:
    """Every grouped-nf4-gemm version the line states (see the module docstring)."""
    found = [m.group("ver") for m in _REQUIRES.finditer(line)]
    mentions = _MENTIONS.search(line) is not None
    for m in _STATEMENT.finditer(line):
        name = m.group("name")
        if name is not None:
            n = pep503_name(name)
            if n in _KERNEL_NAMES:
                found.append(m.group("ver"))
                continue
            if n in other_packages or not m.group("gap"):
                continue                        # torch>=2.2.0, bitsandbytes >= 0.50.0, or glued to a foreign name
        if mentions:
            found.append(m.group("ver"))
    return found


def is_anchored(path: Path) -> bool:
    if path.with_name(path.name + ".ots").is_file():
        return True
    try:
        return ANCHOR_MARKER in read_text(path)
    except (OSError, UnicodeDecodeError):
        return False


def index_current_documents(root: Path) -> list[Path]:
    """The documents docs/INDEX.md links under its ``## Current`` heading."""
    idx = root / INDEX
    if not idx.is_file():
        return []
    out: list[Path] = []
    in_current = False
    for line in read_text(idx).splitlines():
        if line.startswith("## "):
            in_current = line.lower().startswith("## current")
            continue
        if not in_current:
            continue
        for target in re.findall(r"\]\(([^)\s#]+)", line):
            if target.startswith(("http://", "https://")):
                continue
            p = (idx.parent / target).resolve()
            if p.suffix == ".md" and p.is_file() and p not in out:
                out.append(p)
    return out


def scan_documents(root: Path) -> list[Path]:
    docs: list[Path] = []
    for rel in FIXED_DOCUMENTS:
        p = root / rel
        if p.is_file():
            docs.append(p)
    docs += sorted(root.glob(SOLUTION_GLOB))
    for p in index_current_documents(root):
        if p not in docs:
            docs.append(p)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    py = load_pyproject(root)
    fr = fast_requirement(py)
    if fr is None:
        print(f"FAIL: pyproject.toml has no {KERNEL_PACKAGE} requirement in the {EXTRA} extra")
        return 2
    req, spec = fr
    floor = floor_of(spec)
    if floor is None:
        print(f"FAIL: the {EXTRA} requirement {req!r} has no single >= floor")
        return 2
    floor_t = parse_version(floor)
    print(f"pyproject {EXTRA} floor: {req}")
    other_packages = frozenset(
        pep503_name(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", r).group(1))
        for reqs in [py.get("dependencies") or []] + list((py.get("optional-dependencies") or {}).values())
        for r in reqs if re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", r)
    ) - _KERNEL_NAMES
    failures = 0
    n_ok = n_hist = 0
    for doc in scan_documents(root):
        rel = doc.relative_to(root).as_posix()
        if is_anchored(doc):
            print(f"skip (anchored): {rel}")
            continue
        for i, line in enumerate(read_text(doc).splitlines(), 1):
            if rel == RELEASE_NOTES_GUIDE and line.lstrip().startswith(">"):
                continue                        # an example release note, not a statement
            vers = statements(line, other_packages)
            if not vers:
                continue
            marker = historical_marker(line)
            excerpt = line.strip()[:110]
            for ver in vers:
                if parse_version(ver) == floor_t:
                    n_ok += 1
                    print(f"OK: {rel}:{i}: {KERNEL_PACKAGE} >= {ver}")
                elif marker:
                    n_hist += 1
                    print(f"historical ({marker!r}): {rel}:{i}: {KERNEL_PACKAGE} >= {ver}: {excerpt}")
                else:
                    failures += 1
                    print(f"FAIL: {rel}:{i}: states {KERNEL_PACKAGE} >= {ver}, pyproject's {EXTRA} floor is {floor}: {excerpt}")
    # docs/system-manifest.json: only the record for this version is current.
    mpath = root / MANIFEST
    if mpath.is_file() and "compatibility" not in json.loads(read_text(mpath)):
        print(f"note: {MANIFEST} has no compatibility table (scripts/check_system_manifest.py reports that)")
    elif mpath.is_file():
        manifest = json.loads(read_text(mpath))
        hits, err = current_record(manifest, str(py.get("name", "")), str(py.get("version", "")))
        if err or len(hits) != 1:
            failures += 1
            why = err or f"{len(hits)} compatibility records contain version {py.get('version')}"
            print(f"FAIL: {MANIFEST}: {why}")
        else:
            rec_floor = floor_of(str(hits[0].get("floor", "")))
            if rec_floor is not None and parse_version(rec_floor) == floor_t:
                n_ok += 1
                print(f"OK: {MANIFEST}: record {hits[0].get('consumer_versions')!r} floor {hits[0].get('floor')}")
            else:
                failures += 1
                print(f"FAIL: {MANIFEST}: record {hits[0].get('consumer_versions')!r} floor {hits[0].get('floor')!r} != {floor}")
    if failures:
        print(f"FAIL: {failures} statement(s) drift from pyproject's {EXTRA} floor ({req})")
        return 1
    print(f"OK: {n_ok} statement(s) equal the {EXTRA} floor {floor}; {n_hist} historical line(s) excused and listed above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
