#!/usr/bin/env python3
"""Change-impact contract (docs/change-impact.json), checked mechanically from
a git diff. Standard library plus git; no network.

    python scripts/check_change_impact.py --base origin/main            # working tree vs a ref
    python scripts/check_change_impact.py --base <sha> --strict

The diff is BASE against the working tree (untracked files count as added),
so the same command works before a commit and in CI, where the tree is the
pull request's merge commit. Classes detected:

  dependency-floor   the grouped-nf4-gemm requirement of the ``fast`` extra in
                     pyproject.toml changed. Must change in the same diff:
                     docs/system-manifest.json, docs/capabilities.json,
                     .github/workflows/ci.yml (the ``--requires`` assertion)
                     and every current solution document (docs/SOLUTIONS.md,
                     docs/solutions/*.md) that stated the old floor. FAIL.
  measured-result    docs/claims.json gained a claim, or a claim's ``status``
                     or ``value`` changed. docs/STATUS.md must change in the
                     same diff. FAIL; WARN with --allow-claims-only (a
                     register-only correction whose position did not move).
  public-api-change  a symbol entered or left ``__all__`` in
                     experts4bit_qlora/__init__.py. docs/capabilities.json or
                     a docs/solutions page, and CHANGELOG.md, must change.
                     WARN; FAIL with --strict.
  new-kernel-capability (consumer side)  the package newly imports a kernel
                     module named in docs/system-manifest.json
                     ``packages.kernels.import_names``. Consider whether the
                     ``fast`` floor must rise (pyproject.toml,
                     docs/system-manifest.json). WARN only.

Prints each class with its trigger and the missing companions. Without
--base, or with an empty value (a non-pull-request CI event), it prints SKIP
and exits 0. Exit 1 on FAIL; 2 when git cannot resolve the base.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_dependency_floor import historical_marker, statements  # noqa: E402
from check_system_manifest import KERNEL_PACKAGE, MANIFEST, fast_requirement  # noqa: E402

PYPROJECT = "pyproject.toml"
CLAIMS = "docs/claims.json"
STATUS = "docs/STATUS.md"
CAPABILITIES = "docs/capabilities.json"
CI = ".github/workflows/ci.yml"
CHANGELOG = "CHANGELOG.md"
SOLUTIONS_INDEX = "docs/SOLUTIONS.md"
SOLUTIONS_DIR = "docs/solutions"
INIT = "experts4bit_qlora/__init__.py"
PACKAGE_DIR = "experts4bit_qlora"
_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)")


# ----------------------------------------------------------------------- git --

def _git(root: Path, *args: str, ok_codes=(0,)) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode not in ok_codes:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip() or p.returncode}")
    return p.stdout


def base_text(root: Path, base: str, rel: str) -> str | None:
    p = subprocess.run(["git", "-C", str(root), "show", f"{base}:{rel}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def head_text(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.is_file() else None


def changed_files(root: Path, base: str) -> set[str]:
    out = set(_git(root, "diff", "--name-only", base, "--").split())
    out |= set(_git(root, "ls-files", "--others", "--exclude-standard").split())
    return out


def base_files(root: Path, base: str, prefix: str) -> list[str]:
    return _git(root, "ls-tree", "-r", "--name-only", base, "--", prefix).split()


# ------------------------------------------------------------------ triggers --

def _project_table(text: str | None) -> dict:
    if text is None:
        return {}
    try:
        import tomllib
    except ImportError:
        print("FAIL: this check needs Python >= 3.11 (tomllib)", file=sys.stderr)
        sys.exit(2)
    return dict(tomllib.loads(text).get("project", {}))


def fast_floor_change(root: Path, base: str) -> tuple[str | None, str | None] | None:
    """``(old, new)`` requirement strings when the fast floor differs, else None."""
    old = fast_requirement(_project_table(base_text(root, base, PYPROJECT)))
    new = fast_requirement(_project_table(head_text(root, PYPROJECT)))
    o, n = (old[0] if old else None), (new[0] if new else None)
    return None if o == n else (o, n)


def claim_changes(root: Path, base: str) -> list[str]:
    def table(text: str | None) -> dict[str, tuple]:
        if text is None:
            return {}
        doc = json.loads(text)
        claims = doc.get("claims") if isinstance(doc, dict) else doc
        return {c["id"]: (c.get("status"), c.get("value")) for c in claims if isinstance(c, dict) and "id" in c}
    old, new = table(base_text(root, base, CLAIMS)), table(head_text(root, CLAIMS))
    out = []
    for cid, (st, val) in new.items():
        if cid not in old:
            out.append(f"added {cid} [{st}]")
            continue
        ost, oval = old[cid]
        if st != ost:
            out.append(f"{cid}: status {ost!r} -> {st!r}")
        if val != oval:
            out.append(f"{cid}: value {oval!r} -> {val!r}")
    return out


def all_symbols(text: str | None) -> set[str] | None:
    """The string constants of ``__all__`` (assignments and ``+=``), or None
    when the file is absent, does not parse, or binds no ``__all__``."""
    if text is None:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    found: set[str] | None = None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            names = {e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            found = names if (found is None or isinstance(node, ast.Assign)) else found | names
    return found


def kernel_import_changes(root: Path, base: str) -> tuple[list[str], list[str]]:
    """``(new_names, where)``: kernel modules the package imports now and did
    not at BASE, and the head lines that import them."""
    manifest = head_text(root, MANIFEST)
    if manifest is None:
        return [], []
    try:
        names = [str(n) for n in json.loads(manifest).get("packages", {}).get("kernels", {}).get("import_names", [])]
    except (ValueError, AttributeError):
        return [], [f"{MANIFEST} is not valid JSON; scripts/check_system_manifest.py will say so"]
    if not names:
        return [], []
    def pattern(mods: list[str]) -> str:                # POSIX ERE: git grep on macOS has no \s or \b
        return r"^[[:space:]]*(from|import)[[:space:]]+(" + "|".join(re.escape(n) for n in mods) + r")([^A-Za-z0-9_]|$)"

    pat = pattern(names)

    def imported(tree: str | None) -> set[str]:
        """Kernel modules imported under the package at ``tree`` (None: the working tree, untracked files included)."""
        args = ["grep", "-h", "-E"] + (["--untracked"] if tree is None else []) + [pat] + ([tree] if tree else [])
        out = _git(root, *args, "--", PACKAGE_DIR, ok_codes=(0, 1))
        return {m.group(1) for ln in out.splitlines() if (m := _IMPORT.match(ln))} & set(names)

    new = sorted(imported(None) - imported(base))
    if not new:
        return [], []
    where = _git(root, "grep", "-n", "--untracked", "-E", pattern(new), "--", PACKAGE_DIR, ok_codes=(0, 1)).splitlines()
    return new, where


def solution_docs_stating(root: Path, base: str, version_req: str) -> list[str]:
    """Current solution documents that stated the floor of ``version_req`` at
    BASE on a non-historical line (the documents that must move with it)."""
    m = re.search(r"(\d+\.\d+\.\d+)", version_req)
    if not m:
        return []
    ver = m.group(1)
    out = []
    for rel in [SOLUTIONS_INDEX] + base_files(root, base, SOLUTIONS_DIR):
        text = base_text(root, base, rel)
        if text is None:
            continue
        for line in text.splitlines():
            if ver in statements(line, frozenset()) and not historical_marker(line):
                out.append(rel)
                break
    return out


# ---------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--base", default="", metavar="REF", help="git ref to diff the working tree against")
    ap.add_argument("--allow-claims-only", action="store_true",
                    help="a claims.json change without docs/STATUS.md is a WARN, not a FAIL")
    ap.add_argument("--strict", action="store_true", help="public-api-change with missing companions FAILs")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if not a.base.strip():
        print("SKIP: no --base ref (not a pull request); nothing to diff")
        return 0
    base = a.base.strip()
    try:
        base = _git(root, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
        changed = changed_files(root, base)
    except (RuntimeError, OSError) as e:
        print(f"FAIL: cannot read the diff: {e}")
        return 2
    print(f"diff: {base[:12]} .. working tree, {len(changed)} file(s) changed")
    failed = warned = triggered = 0

    def missing(companions: list[str]) -> list[str]:
        return [c for c in companions if c not in changed]

    def report(cls: str, trigger: str, need: list[str], hard: bool, note: str = "") -> None:
        nonlocal failed, warned, triggered
        triggered += 1
        print(f"CLASS {cls}: trigger: {trigger}")
        for c in need:
            print(f"  {'changed' if c in changed else 'MISSING'}: {c}")
        gone = missing(need)
        if note:
            print(f"  note: {note}")
        if gone and hard:
            failed += 1
            print(f"FAIL: {cls}: {len(gone)} companion(s) missing from this diff: {', '.join(gone)}")
        elif gone:
            warned += 1
            print(f"WARN: {cls}: {len(gone)} companion(s) missing from this diff: {', '.join(gone)}")
        else:
            print(f"OK: {cls}: every companion changed")

    # dependency-floor
    fc = fast_floor_change(root, base)
    if fc:
        old, new = fc
        need = [MANIFEST, CAPABILITIES, CI] + [d for d in solution_docs_stating(root, base, old or "") if d not in ()]
        report("dependency-floor", f"{PYPROJECT} fast extra {old!r} -> {new!r}", need, hard=True,
               note="scripts/check_dependency_floor.py and scripts/check_system_manifest.py verify the new value")

    # measured-result
    cc = claim_changes(root, base)
    if cc:
        shown = "; ".join(cc[:6]) + (f"; +{len(cc) - 6} more" if len(cc) > 6 else "")
        report("measured-result", f"{CLAIMS}: {shown}", [STATUS], hard=not a.allow_claims_only)

    # public-api-change
    old_all, new_all = all_symbols(base_text(root, base, INIT)), all_symbols(head_text(root, INIT))
    if old_all is not None and new_all is not None and old_all != new_all:
        added, removed = sorted(new_all - old_all), sorted(old_all - new_all)
        api_docs = [CAPABILITIES] + [f for f in changed if f.startswith(SOLUTIONS_DIR + "/")]
        docs_changed = any(f in changed for f in api_docs)
        need = ([CAPABILITIES] if not docs_changed else [next(f for f in api_docs if f in changed)]) + [CHANGELOG]
        report("public-api-change", f"{INIT} __all__: +{added} -{removed}", need, hard=a.strict,
               note=f"{CAPABILITIES} or a {SOLUTIONS_DIR}/ page, and {CHANGELOG}; see docs/change-impact.json")
    elif INIT in changed and (old_all is None or new_all is None):
        warned += 1
        print(f"WARN: public-api-change: {INIT} changed but __all__ could not be read on one side")

    # new-kernel-capability (consumer side)
    try:
        new_imports, where = kernel_import_changes(root, base)
    except RuntimeError as e:
        new_imports, where = [], []
        warned += 1
        print(f"WARN: new-kernel-capability: could not compare kernel imports: {e}")
    if new_imports:
        triggered += 1
        warned += 1
        print(f"CLASS new-kernel-capability: trigger: the package newly imports kernel module(s) {new_imports}")
        for ln in where[:12]:
            print(f"  {ln}")
        print(f"WARN: new-kernel-capability: consider the fast floor ({PYPROJECT}) and the compatibility record "
              f"({MANIFEST}); see docs/change-impact.json")

    if failed:
        print(f"FAIL: {failed} class(es) with missing companions ({warned} warning(s))")
        return 1
    if triggered:
        print(f"OK: {triggered} class(es) triggered, {warned} warning(s), no hard failure")
    else:
        print(f"OK: no change-impact class triggered by this diff against {base[:12]} "
              f"({KERNEL_PACKAGE} floor, claims, __all__ and kernel imports unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
