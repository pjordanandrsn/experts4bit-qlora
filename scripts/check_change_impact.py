#!/usr/bin/env python3
"""Change-impact contract (docs/change-impact.json): the companion changes a
class of change requires, detected from a git diff. Standard library plus git;
no network.

One file, byte-identical in experts4bit-qlora (the runtime package) and
grouped-nf4-gemm (the kernel package). It reads which package this repository
is from docs/system-manifest.json -- ``check_system_manifest.system_role``: the
``packages`` entry whose ``package`` is pyproject's [project].name, or failing
that the one whose ``repository`` is its Source URL -- and runs that role's
``PROFILES`` entry. A repository the manifest does not name cannot be checked
(exit 2).

    python scripts/check_change_impact.py --base origin/main            # a branch, before or after committing
    python scripts/check_change_impact.py --base <sha> --allow-claims-only
    python scripts/check_change_impact.py --base <sha> --strict

The diff is ``git merge-base BASE HEAD`` against the working tree, untracked
(not ignored) files counted as added: a local run sees uncommitted work, and a
branch behind its base is not charged with -- or credited for -- what landed
on the base since it forked. In CI the tree is the pull request's merge commit,
whose merge base with the PR base is the PR base itself. The merge base needs
the history between them: a shallow checkout has none (actions/checkout's
default depth 1), and the check then exits 2 rather than diff against the wrong
commit -- check out with ``fetch-depth: 0``.

Both roles:

  measured-result    docs/claims.json gained a claim, or a claim's ``status``,
                     ``value`` or ``unit`` changed. docs/STATUS.md must change
                     in the same diff. FAIL; WARN with --allow-claims-only (a
                     register-only correction whose position did not move).
  dependency-floor   the pyproject version changed: CHANGELOG.md must change.
                     FAIL. The pyproject dependencies changed: README.md and
                     docs/capabilities.json should. WARN.
  public-api-change  the set of entrypoints in docs/capabilities.json changed:
                     CHANGELOG.md must change (FAIL) and README.md should (WARN).
  Every class a trigger reports must be one docs/change-impact.json names
  (FAIL otherwise); its ``classes`` are read as a list of ``{"id": ...}``
  entries or as an object keyed by id.

Kernel role only (``PROFILES`` says, for each key, why the runtime does not run it):

  new-kernel-capability  a module added to [tool.setuptools] py-modules or
                     packages, or a new @triton.jit kernel in a module pyproject
                     ships: docs/capabilities.json and CHANGELOG.md must change.
                     FAIL. A layout constant (BLOCK / GROUP / STRIDE / ...)
                     changed in a shipped module: docs/KERNEL_CONTRACT.md
                     should. WARN.

Runtime role only:

  dependency-floor   the grouped-nf4-gemm requirement of the ``fast`` extra
                     changed. Must change in the same diff:
                     docs/system-manifest.json, docs/capabilities.json,
                     .github/workflows/ci.yml (the ``--requires`` assertion)
                     and every current solution document (docs/SOLUTIONS.md,
                     docs/solutions/*.md) that stated the old floor on a line
                     that is not historical. FAIL.
  public-api-change  a symbol entered or left ``__all__`` in the package's
                     ``__init__.py`` (the package is packages.runtime
                     ``import_names``). docs/capabilities.json or a changed
                     docs/solutions page, and CHANGELOG.md, must change. WARN;
                     FAIL with --strict.
  new-kernel-capability (consumer side)  the package newly imports a kernel
                     module named in docs/system-manifest.json
                     ``packages.kernels.import_names``. Consider whether the
                     ``fast`` floor must rise. WARN only.

Prints each triggered class with its trigger and every companion (``changed``
/ ``MISSING``). Without --base, or with an empty one (a non-pull-request CI
event), it prints SKIP and exits 0. Exit 1 on any FAIL; 2 when the check
itself cannot run (the role, the contract, the base, the merge base or the
diff cannot be read).
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
from check_system_manifest import MANIFEST, fast_requirement, system_role  # noqa: E402
from discovery_common import ContractError, load_pyproject, read_text  # noqa: E402

CONTRACT = "docs/change-impact.json"
PYPROJECT = "pyproject.toml"
CLAIMS = "docs/claims.json"
STATUS = "docs/STATUS.md"
CAPABILITIES = "docs/capabilities.json"
CI = ".github/workflows/ci.yml"
CHANGELOG = "CHANGELOG.md"
README = "README.md"
KERNEL_CONTRACT = "docs/KERNEL_CONTRACT.md"
SOLUTIONS_INDEX = "docs/SOLUTIONS.md"
SOLUTIONS_DIR = "docs/solutions"
#: The runtime package's ``__init__`` and directory as the runtime copy named
#: them; ``package_dirs`` reads the same from the manifest.
INIT = "experts4bit_qlora/__init__.py"
PACKAGE_DIR = "experts4bit_qlora"

LAYOUT_WORDS = re.compile(r"(BLOCK|GROUP|STRIDE|ALIGN|LAYOUT|NIBBLE|PACK|ROW|TILE|WIDTH|BYTES)")
CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::\s*[^=]+)?=\s*(.+?)\s*(?:#.*)?$", re.M)
# Indented too: the kernel tree defines kernels inside functions/classes (kernel/fp8_kv.py,
# kernel/fp8_paged_attn.py), and a nested kernel is still a new kernel.
JIT_KERNEL = re.compile(r"^[ \t]*@triton\.jit(?:\([^)]*\))?[ \t]*\n(?:^[ \t]*@[^\n]*\n)*^[ \t]*def\s+(\w+)", re.M)
_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)")

#: Which triggers each role runs beyond the ones both share. Every key is a
#: place the two copies this file replaces differed: the rule reads a surface
#: only one package has, so in the other role it would either never fire or
#: fire on every change for the wrong reason.
PROFILES: dict[str, dict] = {
    "runtime": {
        "role": "runtime",
        # A module added to [tool.setuptools] py-modules / packages. The runtime
        # ships through [tool.setuptools.packages.find]: there is no list to
        # diff, so the rule could never fire here. Its public surface is
        # `__all__` (`api_all`).
        "module_list": False,
        # A new @triton.jit kernel, and a changed layout constant, in a module
        # the py-modules / packages list ships. The runtime has no such list,
        # and its contract's new-kernel-capability is the consumer side
        # (`kernel_imports`): its one @triton.jit is a row gather inside the
        # pipelined residency engine (engines/pipelined.py), not a capability,
        # and docs/KERNEL_CONTRACT.md does not exist there.
        "kernel_sources": False,
        # The grouped-nf4-gemm requirement of the `fast` extra. The kernel's
        # pyproject has no `fast` extra on itself.
        "fast_floor": True,
        # `__all__` of the package's __init__.py. The kernel ships flat modules
        # without `__all__` (its surface is the capabilities.json entrypoint
        # set): every edit to one would warn that `__all__` cannot be read.
        "api_all": True,
        # A kernel module newly imported by the package. Every kernel module
        # imports its siblings, so in the kernel this would fire on internal
        # refactors; it is the consumer's signal that the floor may have to rise.
        "kernel_imports": True,
    },
    "kernels": {
        "role": "kernels",
        "module_list": True,
        "kernel_sources": True,
        "fast_floor": False,
        "api_all": False,
        "kernel_imports": False,
    },
}


class GitError(RuntimeError):
    pass


# ----------------------------------------------------------------------- git --

def _git(root: Path, *args: str, ok_codes=(0,)) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode not in ok_codes:
        raise GitError(f"git {' '.join(args)}: {p.stderr.strip() or 'exit ' + str(p.returncode)}")
    return p.stdout


def base_text(root: Path, base: str, rel: str) -> str | None:
    p = subprocess.run(["git", "-C", str(root), "show", f"{base}:{rel}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def head_text(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.is_file() else None


#: The kernel copy's names for the same two readers.
_show = base_text
_head_text = head_text


def untracked_files(root: Path) -> list[str]:
    return [ln for ln in _git(root, "ls-files", "--others", "--exclude-standard").splitlines() if ln.strip()]


def changed_files(root: Path, base: str) -> set[str]:
    """Tracked files that differ between ``base`` and the working tree, and
    every untracked (not ignored) file."""
    out = {ln.strip() for ln in _git(root, "diff", "--name-only", base, "--").splitlines() if ln.strip()}
    return out | set(untracked_files(root))


def base_files(root: Path, base: str, prefix: str) -> list[str]:
    return _git(root, "ls-tree", "-r", "--name-only", base, "--", prefix).split()


def merge_base(root: Path, base: str) -> str:
    """``git merge-base BASE HEAD``; a ``GitError`` naming the shallow checkout
    when there is none (the history between them was not fetched)."""
    try:
        return _git(root, "merge-base", base, "HEAD").strip()
    except GitError as e:
        shallow = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, text=True).stdout.strip() == "true"
        hint = (" -- this checkout is shallow, so the history between them is missing "
                "(actions/checkout: fetch-depth: 0)" if shallow else "")
        raise GitError(f"no merge base of {base[:12]} and HEAD: {e}{hint}") from e


# --------------------------------------------------------------------- role --

def repo_role(root) -> str | None:
    """``"runtime"`` / ``"kernels"``: this repository's ``packages`` entry in
    docs/system-manifest.json, read by ``check_system_manifest.system_role``;
    ``None`` when the manifest or pyproject.toml is missing or names neither."""
    try:
        manifest = json.loads(read_text(Path(root) / MANIFEST))
        return system_role(manifest, load_pyproject(root))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def profile_for(root) -> dict:
    """``PROFILES`` for this repository's role; a ``ContractError`` (exit 2)
    when the role cannot be read -- a check that guessed would run the wrong rules."""
    role = repo_role(root)
    if role is None:
        raise ContractError(f"cannot tell which package of the system {Path(root)} is: {MANIFEST} names no "
                            "packages entry for pyproject's [project].name or Source URL")
    return PROFILES[role]


def contract_classes(path: Path) -> set[str]:
    """The class ids docs/change-impact.json names: its ``classes`` as a list of
    ``{"id": ...}`` entries (the kernel's shape) or an object keyed by id (the
    runtime's). A ``ContractError`` when it has neither, or an entry has no id."""
    try:
        doc = json.loads(read_text(path))
    except (OSError, ValueError) as e:
        raise ContractError(f"{path.name}: {e}") from e
    classes = doc.get("classes") if isinstance(doc, dict) else None
    if isinstance(classes, dict) and classes:
        return {str(k) for k in classes}
    if isinstance(classes, list) and classes:
        if not all(isinstance(c, dict) and isinstance(c.get("id"), str) and c["id"] for c in classes):
            raise ContractError(f"{path.name}: a classes entry has no string 'id'")
        return {c["id"] for c in classes}
    raise ContractError(f"{path.name}: no 'classes' (a list of {{'id': ...}} or an object keyed by id)")


def package_dirs(root: Path, role: str) -> list[str]:
    """The ``import_names`` of ``packages.<role>`` that are package directories
    (``experts4bit_qlora``), from docs/system-manifest.json."""
    try:
        names = json.loads(read_text(root / MANIFEST))["packages"][role].get("import_names") or []
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []
    return [str(n) for n in names if (root / str(n) / "__init__.py").is_file()]


# ------------------------------------------------------------------ triggers --

def _toml(text: str | None) -> dict:
    if not text:
        return {}
    try:
        import tomllib
    except ImportError:
        print("FAIL: this check needs Python >= 3.11 (tomllib)", file=sys.stderr)
        sys.exit(2)
    return tomllib.loads(text)


def _project_table(text: str | None) -> dict:
    return dict(_toml(text).get("project", {}))


#: The kernel copy's name.
_project = _project_table


def _py_modules(pyproject_text: str | None) -> set[str]:
    st = _toml(pyproject_text).get("tool", {}).get("setuptools", {})
    return set(st.get("py-modules") or []) | set(st.get("packages") or [])


def _module_path(pyproject_text: str, module: str) -> str:
    st = _toml(pyproject_text).get("tool", {}).get("setuptools", {})
    pkg_dir = st.get("package-dir") or {}
    if module in pkg_dir:
        return f"{pkg_dir[module]}/__init__.py"
    base = pkg_dir.get("", "")
    return f"{base}/{module}.py" if base else f"{module}.py"


def _constants(text: str | None) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in CONSTANT.finditer(text or "") if LAYOUT_WORDS.search(m.group(1))}


def _kernels(text: str | None) -> set[str]:
    return set(JIT_KERNEL.findall(text or ""))


def _entrypoints(text: str | None) -> set[tuple[str, str]]:
    if not text:
        return set()
    doc = json.loads(text)
    return {(c["id"], ep) for c in doc.get("capabilities", []) for ep in c.get("entrypoints", [])}


def _claim_list(text: str | None) -> list[dict]:
    """The claim entries of docs/claims.json (``{"claims": [...]}`` or a bare
    list); a ``ContractError`` when the file is neither -- a register this
    check cannot read must not pass as unchanged."""
    if text is None:
        return []
    doc = json.loads(text)
    claims = doc.get("claims") if isinstance(doc, dict) else doc
    if not isinstance(claims, list) or not all(isinstance(c, dict) for c in claims):
        raise ContractError(f"{CLAIMS}: 'claims' is not a list of objects")
    return claims


def _claims(text: str | None) -> dict[str, tuple]:
    """``{id: (status, value as JSON, unit)}`` -- the kernel copy's reading."""
    return {c["id"]: (c.get("status"), json.dumps(c.get("value")), c.get("unit"))
            for c in _claim_list(text) if "id" in c}


def claim_changes(root: Path, base: str) -> list[str]:
    """Claims added at the head, and claims whose ``status``, ``value`` or
    ``unit`` differs from BASE. A value differs when it is unequal OR serialises
    differently (``1`` vs ``1.0``, ``1`` vs ``true``): the two copies' readings."""
    def table(text: str | None) -> dict[str, dict]:
        return {c["id"]: c for c in _claim_list(text) if "id" in c}
    old, new = table(base_text(root, base, CLAIMS)), table(head_text(root, CLAIMS))
    out = []
    for cid, c in new.items():
        if cid not in old:
            out.append(f"added {cid} [{c.get('status')}]")
            continue
        o = old[cid]
        if c.get("status") != o.get("status"):
            out.append(f"{cid}: status {o.get('status')!r} -> {c.get('status')!r}")
        val, oval = c.get("value"), o.get("value")
        if val != oval or json.dumps(val) != json.dumps(oval):
            out.append(f"{cid}: value {oval!r} -> {val!r}")
        if c.get("unit") != o.get("unit"):
            out.append(f"{cid}: unit {o.get('unit')!r} -> {c.get('unit')!r}")
    return out


def fast_floor_change(root: Path, base: str) -> tuple[str | None, str | None] | None:
    """``(old, new)`` requirement strings when the fast floor differs, else None."""
    old = fast_requirement(_project_table(base_text(root, base, PYPROJECT)))
    new = fast_requirement(_project_table(head_text(root, PYPROJECT)))
    o, n = (old[0] if old else None), (new[0] if new else None)
    return None if o == n else (o, n)


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


def kernel_import_changes(root: Path, base: str, package_dir: str = PACKAGE_DIR) -> tuple[list[str], list[str]]:
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
        out = _git(root, *args, "--", package_dir, ok_codes=(0, 1))
        return {m.group(1) for ln in out.splitlines() if (m := _IMPORT.match(ln))} & set(names)

    new = sorted(imported(None) - imported(base))
    if not new:
        return [], []
    where = _git(root, "grep", "-n", "--untracked", "-E", pattern(new), "--", package_dir, ok_codes=(0, 1)).splitlines()
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

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="this repository's root")
    ap.add_argument("--base", default=None, metavar="REF",
                    help="git ref whose merge base with HEAD the working tree is diffed against "
                         "(absent or empty: SKIP, exit 0)")
    ap.add_argument("--contract", default=CONTRACT, help=f"the contract, relative to --root (default {CONTRACT})")
    ap.add_argument("--allow-claims-only", action="store_true",
                    help="a claims.json change without docs/STATUS.md is a WARN, not a FAIL")
    ap.add_argument("--strict", action="store_true",
                    help="runtime role: an __all__ change with missing companions FAILs")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if not a.base:
        print("SKIP: no --base ref (not a pull request); change-impact check not run")
        return 0
    try:
        prof = profile_for(root)
        known = contract_classes(root / a.contract)
    except ContractError as e:
        print(f"FAIL: {e}")
        return 2
    try:
        base = _git(root, "rev-parse", "--verify", f"{a.base}^{{commit}}").strip()
        mb = merge_base(root, base)
        untracked = untracked_files(root)
        changed = changed_files(root, mb)
    except (GitError, OSError) as e:
        print(f"FAIL: cannot compute the diff: {e}")
        return 2
    print(f"diff: {mb[:12]} (merge base of {a.base} and HEAD) .. working tree, {len(changed)} file(s) changed "
          f"(packages.{prof['role']})")
    if untracked:
        print(f"note: {len(untracked)} untracked file(s) counted as added: {untracked[:8]}")
    failed = warned = triggered = 0

    def report(cls: str, trigger: str, need: list[str], hard: bool, note: str = "") -> None:
        nonlocal failed, warned, triggered
        triggered += 1
        print(f"CLASS {cls}: trigger: {trigger}")
        if cls not in known:
            failed += 1
            print(f"FAIL: {cls}: class not in {a.contract}")
        for c in need:
            print(f"  {'changed' if c in changed else 'MISSING'}: {c}")
        gone = [c for c in need if c not in changed]
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

    try:
        base_py, head_py = base_text(root, mb, PYPROJECT), head_text(root, PYPROJECT)

        # -- dependency-floor ----------------------------------------------------
        if PYPROJECT in changed:
            bp, hp = _project_table(base_py), _project_table(head_py)
            if bp.get("version") != hp.get("version"):
                report("dependency-floor", f"{PYPROJECT} version {bp.get('version')} -> {hp.get('version')}",
                       [CHANGELOG], hard=True)
            if (bp.get("dependencies") or []) != (hp.get("dependencies") or []):
                report("dependency-floor", f"{PYPROJECT} dependencies changed", [README, CAPABILITIES], hard=False)
        if prof["fast_floor"]:
            fc = fast_floor_change(root, mb)
            if fc:
                old, new = fc
                need = [MANIFEST, CAPABILITIES, CI] + solution_docs_stating(root, mb, old or "")
                report("dependency-floor", f"{PYPROJECT} fast extra {old!r} -> {new!r}", need, hard=True,
                       note="scripts/check_dependency_floor.py and scripts/check_system_manifest.py "
                            "verify the new value")

        # -- new-kernel-capability (the kernel's sources) --------------------------
        if prof["module_list"] and PYPROJECT in changed:
            for mod in sorted(_py_modules(head_py) - _py_modules(base_py)):
                report("new-kernel-capability", f"module {mod!r} added to pyproject py-modules",
                       [CAPABILITIES, CHANGELOG], hard=True)
        if prof["kernel_sources"]:
            for mod in sorted(_py_modules(head_py)):
                path = _module_path(head_py or "", mod)
                if path not in changed:
                    continue
                base_t, head_t = base_text(root, mb, path), head_text(root, path)
                if base_t is None:
                    continue                                # a new module: the py-modules trigger above
                new_k = sorted(_kernels(head_t) - _kernels(base_t))
                if new_k:
                    report("new-kernel-capability", f"new @triton.jit kernel(s) {new_k} in {path}",
                           [CAPABILITIES, CHANGELOG], hard=True)
                bc, hc = _constants(base_t), _constants(head_t)
                moved = sorted(k for k in hc if bc.get(k) != hc[k])
                if moved:
                    report("new-kernel-capability", f"layout constant(s) {moved} changed in {path}",
                           [KERNEL_CONTRACT], hard=False)

        # -- public-api-change ---------------------------------------------------
        if CAPABILITIES in changed:
            b, h = _entrypoints(base_text(root, mb, CAPABILITIES)), _entrypoints(head_text(root, CAPABILITIES))
            added, removed = sorted(h - b), sorted(b - h)
            if added or removed:
                trig = f"entrypoints added {[e for _, e in added]} removed {[e for _, e in removed]}"
                report("public-api-change", trig, [CHANGELOG], hard=True)
                report("public-api-change", trig + " (entry-point table)", [README], hard=False)
        if prof["api_all"]:
            pkgs = package_dirs(root, prof["role"])
            if not pkgs:
                warned += 1
                print(f"WARN: public-api-change: packages.{prof['role']}.import_names names no package directory "
                      f"with an __init__.py; __all__ was not compared")
            for pkg in pkgs:
                init = f"{pkg}/__init__.py"
                old_all, new_all = all_symbols(base_text(root, mb, init)), all_symbols(head_text(root, init))
                if old_all is not None and new_all is not None and old_all != new_all:
                    added, removed = sorted(new_all - old_all), sorted(old_all - new_all)
                    api_docs = [CAPABILITIES] + [f for f in changed if f.startswith(SOLUTIONS_DIR + "/")]
                    docs_changed = any(f in changed for f in api_docs)
                    need = ([CAPABILITIES] if not docs_changed
                            else [next(f for f in api_docs if f in changed)]) + [CHANGELOG]
                    report("public-api-change", f"{init} __all__: +{added} -{removed}", need, hard=a.strict,
                           note=f"{CAPABILITIES} or a {SOLUTIONS_DIR}/ page, and {CHANGELOG}; see {a.contract}")
                elif init in changed and (old_all is None or new_all is None):
                    warned += 1
                    print(f"WARN: public-api-change: {init} changed but __all__ could not be read on one side")

        # -- measured-result -----------------------------------------------------
        cc = claim_changes(root, mb)
        if cc:
            shown = "; ".join(cc[:6]) + (f"; +{len(cc) - 6} more" if len(cc) > 6 else "")
            report("measured-result", f"{CLAIMS}: {shown}", [STATUS], hard=not a.allow_claims_only)

        # -- new-kernel-capability (consumer side) -------------------------------
        if prof["kernel_imports"]:
            for pkg in package_dirs(root, prof["role"]):
                try:
                    new_imports, where = kernel_import_changes(root, mb, pkg)
                except GitError as e:
                    new_imports, where = [], []
                    warned += 1
                    print(f"WARN: new-kernel-capability: could not compare kernel imports: {e}")
                if new_imports:
                    triggered += 1
                    warned += 1
                    print(f"CLASS new-kernel-capability: trigger: the package newly imports kernel module(s) "
                          f"{new_imports}")
                    if "new-kernel-capability" not in known:
                        failed += 1
                        print(f"FAIL: new-kernel-capability: class not in {a.contract}")
                    for ln in where[:12]:
                        print(f"  {ln}")
                    print(f"WARN: new-kernel-capability: consider the fast floor ({PYPROJECT}) and the compatibility "
                          f"record ({MANIFEST}); see {a.contract}")
    except (ContractError, GitError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"FAIL: the check could not run: {e!r}")
        return 2

    if failed:
        print(f"FAIL: {failed} class(es) with missing companions ({warned} warning(s)); contract: {a.contract}")
        return 1
    if triggered:
        print(f"OK: {triggered} class(es) triggered, {warned} warning(s), no hard failure")
    else:
        print(f"OK: no change-impact class triggered by this diff against {base[:12]} "
              f"({len(changed)} file(s) differ from merge base {mb[:12]}; {warned} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
