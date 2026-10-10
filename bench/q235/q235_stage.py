# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Seal Q235's transport closure: the files the box runs, each checked against the committed ``q235.sha256``, plus the locked
input assets (``locked_inputs.json``) from an asset directory. Prints the flat path list ``tc1_drive.sh`` stages
(``TC1_EXTRA_STAGE``); the box re-checks every file with ``sha256sum -c q235.sha256`` before anything runs.

    python q235_stage.py --assets DIR        (the list, or a refusal)
    python q235_stage.py --write             (rewrite q235.sha256 from CLOSURE after a reviewed change)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Every file the box runs or reads, by its path in the repository; each travels flat, under its own name.
CLOSURE = ("bench/q235/q235_run.sh", "bench/q235/q235_bootstrap.py", "bench/q235/q235_fetch.py", "bench/q235/q235_tiny.py",
           "bench/q235/wheels.json", "bench/q235/model_files.json", "bench/q235/instrument/q235_box.py",
           "bench/q235/instrument/q235_tokens.py", "bench/q235/instrument/mk_slice.py", "bench/tc1/tc1_arm.py",
           "bench/dq11/dq11_require_git.sh")
MANIFEST = HERE / "q235.sha256"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write() -> str:
    names = [Path(p).name for p in CLOSURE]
    if len(set(names)) != len(names):
        raise ValueError("two closure files share a name")
    lines = [f"{sha(REPO / p)}  {Path(p).name}" for p in CLOSURE]
    lock = json.loads((HERE / "locked_inputs.json").read_text())["assets"]
    lines += [f"{row['sha256']}  {name}" for name, row in sorted(lock.items())]
    MANIFEST.write_text("\n".join(lines) + "\n")
    return MANIFEST.read_text()


def stage(assets: Path) -> tuple:
    """The paths to stage, refused unless every closure file and asset matches the manifest and the lock."""
    want = {}
    for line in MANIFEST.read_text().splitlines():
        digest, name = line.split()
        if not re.fullmatch("[0-9a-f]{64}", digest) or Path(name).name != name or name in want:
            raise ValueError(f"q235.sha256: bad or repeated line {line!r}")
        want[name] = digest
    paths = []
    for p in CLOSURE:
        path = REPO / p
        if want.get(path.name) != sha(path):
            raise ValueError(f"{p} differs from q235.sha256 (rewrite it with --write after review)")
        paths.append(path)
    for name, row in json.loads((HERE / "locked_inputs.json").read_text())["assets"].items():
        path = Path(assets) / name
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"] or want.get(name) != row["sha256"]:
            raise ValueError(f"asset {name}: missing, or not the locked bytes")
        paths.append(path)
    if set(want) != {p.name for p in paths}:
        raise ValueError(f"q235.sha256 and the closure disagree: {sorted(set(want) ^ {p.name for p in paths})}")
    paths.append(MANIFEST)
    if any(re.search(r"\s", str(p)) for p in paths):
        raise ValueError("tc1_drive.sh's flat transport needs paths without whitespace")
    return tuple(paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--assets", type=Path)
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        print(write(), end="")
    else:
        print(" ".join(str(p) for p in stage(a.assets)))


if __name__ == "__main__":
    main()
