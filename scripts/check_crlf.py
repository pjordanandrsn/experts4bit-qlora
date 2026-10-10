#!/usr/bin/env python3
"""Reject CR line endings in added/changed text files; CSV files are exempt.

With --base, inspect the working tree against its merge base with HEAD, including
untracked, non-ignored additions. Without a base, inspect all tracked files and
those additions (main/manual CI). Deleted files, symlinks and Git-style binary
files (a NUL in the first 8,000 bytes) are skipped. No file is rewritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def paths_to_check(root: Path, base: str | None) -> list[Path]:
    if base and base != "0" * 40:
        ancestor = git(root, "merge-base", base, "HEAD").decode().strip()
        names = git(root, "diff", "--name-only", "-z", "--diff-filter=ACMRT", ancestor)
    else:
        names = git(root, "ls-files", "-z")
    additions = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return sorted({Path(name.decode(errors="surrogateescape")) for name in (names + additions).split(b"\0") if name})


def check(root: Path, base: str | None) -> tuple[int, list[str]]:
    checked = 0
    failures = []
    for relative in paths_to_check(root, base):
        path = root / relative
        if relative.suffix.lower() == ".csv" or path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8000]:
            continue
        checked += 1
        lines = [str(i) for i, line in enumerate(data.split(b"\n"), 1) if b"\r" in line]
        if lines:
            failures.append(f"{relative}: CR line endings on line(s) {', '.join(lines[:5])}")
    return checked, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="PR base or prior main SHA; empty checks all tracked files")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        checked, failures = check(args.root, args.base)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: cannot check line endings: {error}", file=sys.stderr)
        return 2
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print(f"OK: {checked} text file(s) checked for CR line endings; CSV files exempt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
