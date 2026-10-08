#!/usr/bin/env python3
"""Stage registered instrument bytes and their local imports from one Git tree.

No imports from a measured wheel, no checkout edits, no network or GPU work.
The flat directory supports the original instruments' sibling-module imports.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


class Refused(ValueError):
    """The registered source cannot be staged faithfully."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, timeout=60).stdout


def source_bundle(repo, pins):
    commit = pins["source_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise Refused("source commit must be a full recorded Git object id")
    paths = git(repo, "ls-tree", "-r", "--name-only", commit, "bench").decode().splitlines()
    modules = {}
    for name in paths:
        p = PurePosixPath(name)
        if p.suffix == ".py":
            modules.setdefault(p.stem, []).append(name)
    pending = list(pins["instruments"])
    files, names = {}, {}
    while pending:
        name = pending.pop()
        if name in files:
            continue
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or not name.startswith("bench/"):
            raise Refused("instrument path escapes bench")
        if p.name in names and names[p.name] != name:
            raise Refused(f"flat instrument name collision: {p.name}")
        data = git(repo, "show", f"{commit}:{name}")
        if name in pins["instruments"] and sha(data) != pins["instruments"][name]:
            raise Refused(f"registered instrument bytes changed: {name}")
        files[name], names[p.name] = data, name
        if p.suffix != ".py":
            continue
        tree = ast.parse(data, filename=name)
        imports = {n.module.split(".")[0] for n in ast.walk(tree)
                   if isinstance(n, ast.ImportFrom) and n.module}
        imports |= {a.name.split(".")[0] for n in ast.walk(tree)
                    if isinstance(n, ast.Import) for a in n.names}
        for module in imports & modules.keys():
            if len(modules[module]) != 1:
                raise Refused(f"ambiguous local instrument import: {module}")
            pending.extend(modules[module])
    return files


def stage(repo, pins_path, destination):
    pins_bytes = pins_path.read_bytes()
    pins = json.loads(pins_bytes)
    files = source_bundle(repo, pins)  # Complete all verification before creating output.
    manifest = {"schema": 1, "source_commit": pins["source_commit"],
                "source_pins_sha256": sha(pins_bytes), "files": {}}
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in sorted(files.items()):
        flat = PurePosixPath(name).name
        (destination / flat).write_bytes(data)
        manifest["files"][flat] = {"source_path": name, "sha256": sha(data),
                                   "registered": name in pins["instruments"]}
    (destination / "source-pins.json").write_bytes(pins_bytes)
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    (destination / "stage-manifest.json").write_bytes(encoded)
    checks = {**{n: v["sha256"] for n, v in manifest["files"].items()},
              "source-pins.json": sha(pins_bytes), "stage-manifest.json": sha(encoded)}
    (destination / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(checks.items())))
    verify(destination)
    return manifest


def verify(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.iterdir()):
        raise Refused("symlink in staged instruments")
    records = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if PurePosixPath(name).name != name or name in records:
            raise Refused("invalid/duplicate staged checksum path")
        path = root / name
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != expected:
            raise Refused(f"staged bytes changed: {name}")
        records[name] = expected
    all_files = {p.name for p in root.iterdir() if p.is_file()}
    if all_files != set(records) | {"SHA256SUMS"} or any(p.is_dir() for p in root.iterdir()):
        raise Refused("unexpected staged file/directory")
    pins = json.loads((root / "source-pins.json").read_bytes())
    manifest = json.loads((root / "stage-manifest.json").read_bytes())
    if manifest["source_commit"] != pins["source_commit"] or manifest["source_pins_sha256"] != records["source-pins.json"]:
        raise Refused("stage manifest is not bound to pins")
    expected_files = {PurePosixPath(n).name: (n, h) for n, h in pins["instruments"].items()}
    for flat, (name, expected) in expected_files.items():
        info = manifest["files"].get(flat, {})
        if info != {"source_path": name, "sha256": expected, "registered": True} or records.get(flat) != expected:
            raise Refused(f"registered file not faithfully staged: {name}")
    for flat, info in manifest["files"].items():
        if records.get(flat) != info["sha256"]:
            raise Refused(f"closure file checksum mismatch: {flat}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--pins", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    args = ap.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        if not all((args.repo, args.pins, args.out)):
            ap.error("--repo, --pins and --out required")
        result = stage(args.repo, args.pins, args.out)
    print(f"RA staged bytes verified: {len(result['files'])} files")


if __name__ == "__main__":
    main()
