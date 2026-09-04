#!/usr/bin/env python3
"""Validate docs/capabilities.json against its schema and against the repository.

Standard library only. Exit 1 on the first class of failure found, after printing
every failure in that run; exit 2 when the check itself cannot run. Checks:

  * valid JSON, and the fields/types/enums the schema requires;
  * project.canonical_package equals [project].name in pyproject.toml;
  * every documentation path exists;
  * every entry point resolves against what the wheel ships:
    ``module:Symbol`` is a def/class/assignment/import bound at MODULE level
    (top-level statements, descending only into module-level if/try blocks --
    that is how experts4bit_qlora binds Experts4bit) of a module pyproject
    ships ([tool.setuptools] py-modules, or the packages.find include globs);
    ``module`` is such a module; ``cli:name`` is a console script in pyproject
    and ``cli:python -m module`` a shipped module with an
    ``if __name__ == "__main__"`` block; ``flag:NAME`` occurs in one of the
    capability's documentation paths;
  * every claim ID exists in docs/claims.json with an ACTIVE status
    (verified / confirmed / measured / measured-private): open, projected,
    retired and superseded claims cannot back a capability;
  * no string under a capability carries a number with a unit -- the
    schema's own rule: numbers live in docs/claims.json;
  * every install command's positional targets name the canonical package
    (or a related project's) and never an alias;
  * capability IDs are unique;
  * --related <path/to/other/capabilities.json>: the other project lists this
    one back (reciprocity), when given.

Run with --import to also import each ``module:Symbol`` (needs the runtime).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import (  # noqa: E402
    ACTIVE_STATUSES, ContractError, install_targets, load_claims, load_pyproject, module_file, module_shipped,
    pep503_name, read_text,
)

#: A number with a unit anywhere under a capability is an error: the schema
#: says numbers never appear here, and a number in prose drifts from the
#: claims register the moment the register is corrected.
NUMBER_WITH_UNIT = re.compile(r"(?<![\w.])\d+(\.\d+)?\s*(ppl|tok/s|GB|MB|ms|µs|us|%|×|x\b|nats)")

_TRY_NODES = tuple(t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None)) if t is not None)


def _schema_check(doc: dict, schema: dict, errors: list[str]) -> None:
    """A small, exact checker for the shape this schema uses (required keys,
    types, enums, patterns, additionalProperties) -- not a general validator."""
    def walk(node, sch, path):
        t = sch.get("type")
        if t == "object":
            if not isinstance(node, dict):
                errors.append(f"{path}: expected object")
                return
            for k in sch.get("required", []):
                if k not in node:
                    errors.append(f"{path}: missing required key {k!r}")
            props = sch.get("properties", {})
            if sch.get("additionalProperties") is False:
                for k in node:
                    if k not in props:
                        errors.append(f"{path}: unknown key {k!r}")
            for k, v in node.items():
                if k in props:
                    walk(v, props[k], f"{path}.{k}")
        elif t == "array":
            if not isinstance(node, list):
                errors.append(f"{path}: expected array")
                return
            if "minItems" in sch and len(node) < sch["minItems"]:
                errors.append(f"{path}: needs at least {sch['minItems']} item(s)")
            for i, v in enumerate(node):
                walk(v, sch.get("items", {}), f"{path}[{i}]")
        elif t == "string":
            if not isinstance(node, str):
                errors.append(f"{path}: expected string")
                return
            if "const" in sch and node != sch["const"]:
                errors.append(f"{path}: must be {sch['const']!r}")
            if "enum" in sch and node not in sch["enum"]:
                errors.append(f"{path}: {node!r} not in {sch['enum']}")
            if "pattern" in sch and not re.fullmatch(sch["pattern"], node):
                errors.append(f"{path}: {node!r} does not match {sch['pattern']}")
    walk(doc, schema, "$")


def _target_names(t: ast.AST) -> set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for e in t.elts:
            names |= _target_names(e)
        return names
    if isinstance(t, ast.Starred):
        return _target_names(t.value)
    return set()


def _module_bindings(tree: ast.Module) -> set[str]:
    """Names bound at module level: defs, classes, plain / annotated / tuple
    assignments and imports (``import a.b`` binds ``a``), descending only into
    module-level if/try blocks. Not ``ast.walk``: a name bound inside a
    function or class body is not an attribute of the module."""
    names: set[str] = set()

    def visit(body) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    names.update(_target_names(t))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                names.update(_target_names(node.target))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.asname or a.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name != "*":
                        names.add(a.asname or a.name)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, _TRY_NODES):
                visit(node.body)
                for h in node.handlers:
                    visit(h.body)
                visit(node.orelse)
                visit(node.finalbody)

    visit(tree.body)
    return names


def _has_main_guard(tree: ast.Module) -> bool:
    """A top-level ``if __name__ == "__main__":`` (either operand order)."""
    for node in tree.body:
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        t = node.test
        operands = [t.left, *t.comparators]
        if (len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                and any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
                and any(isinstance(o, ast.Constant) and o.value == "__main__" for o in operands)):
            return True
    return False


def _check_numbers(cap: dict, cid: str, errors: list[str]) -> None:
    def walk(node, path: str) -> None:
        if isinstance(node, str):
            m = NUMBER_WITH_UNIT.search(node)
            if m:
                errors.append(f"{cid}: {path} carries a number with a unit ({m.group(0)!r}); numbers live in "
                              f"docs/claims.json, reference the claim id instead: {node[:90]!r}")
        elif isinstance(node, dict):
            for k, v in node.items():
                if k == "symptoms":
                    # user phrasings ("train a 30B MoE on a 24 GB GPU") are search
                    # queries, not measured results; the register rule targets results
                    continue
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(cap, "capability")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--capabilities", default="docs/capabilities.json")
    ap.add_argument("--schema", default="docs/capabilities.schema.json")
    ap.add_argument("--related", action="append", default=[], help="another project's capabilities.json to check reciprocity against")
    ap.add_argument("--import", dest="do_import", action="store_true", help="also import module:Symbol entry points")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    errors: list[str] = []
    cap_path = root / a.capabilities
    try:
        doc = json.loads(read_text(cap_path))
    except (OSError, ValueError) as e:
        print(f"FAIL: {a.capabilities}: not valid JSON: {e}")
        return 1
    schema = json.loads(read_text(root / a.schema))
    _schema_check(doc, schema, errors)
    if errors:
        for e in errors:
            print("SCHEMA:", e)
        return 1
    proj = doc["project"]
    py = load_pyproject(root)
    pname = py.get("name")
    if proj["canonical_package"] != pname:
        errors.append(f"canonical_package {proj['canonical_package']!r} != pyproject name {pname!r}")
    aliases = {pep503_name(x) for x in proj.get("aliases", [])}
    related_pkgs = {r["canonical_package"] for r in proj.get("related_projects", [])}
    allowed_pkgs = {pep503_name(x) for x in {proj["canonical_package"]} | related_pkgs}
    cli = py.get("scripts", {}) or {}
    try:
        claims, _vocab = load_claims(root, proj["claims_file"])
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 1
    by_id = {c["id"]: c for c in claims}
    parsed: dict[Path, ast.Module] = {}

    def tree_of(path: Path) -> ast.Module:
        if path not in parsed:
            parsed[path] = ast.parse(read_text(path), filename=str(path))
        return parsed[path]

    ids_seen = set()
    for cap in doc["capabilities"]:
        cid = cap["id"]
        if cid in ids_seen:
            errors.append(f"{cid}: duplicate capability id")
        ids_seen.add(cid)
        _check_numbers(cap, cid, errors)
        for d in cap["documentation"]:
            if not (root / d).exists():
                errors.append(f"{cid}: documentation path missing: {d}")
        doc_text = "\n".join(read_text(root / d) for d in cap["documentation"] if (root / d).is_file())
        for ep in cap["entrypoints"]:
            if ep.startswith("cli:"):
                name = ep[4:]
                m = re.fullmatch(r"python -m ([\w.]+)(?:\s.*)?", name)
                if m:                       # a `python -m module` CLI
                    mod = m.group(1)
                    f = module_file(root, mod, py)
                    if not module_shipped(mod, py):
                        errors.append(f"{cid}: `python -m` module is not shipped by pyproject (py-modules / packages): {ep}")
                    elif f is None:
                        errors.append(f"{cid}: `python -m` module not found in the tree: {ep}")
                    elif not _has_main_guard(tree_of(f)):
                        errors.append(f"{cid}: `python -m` module {f.relative_to(root)} has no `if __name__ == \"__main__\"` block: {ep}")
                elif name not in cli:
                    errors.append(f"{cid}: console script not in pyproject: {ep}")
            elif ep.startswith("flag:"):
                if ep[5:] not in doc_text:
                    errors.append(f"{cid}: flag {ep[5:]} is not documented in this capability's documentation")
            elif ":" in ep:
                mod, sym = ep.split(":", 1)
                f = module_file(root, mod, py)
                if not module_shipped(mod, py):
                    errors.append(f"{cid}: entry point module is not shipped by pyproject (py-modules / packages): {ep}")
                elif f is None:
                    errors.append(f"{cid}: entry point module not found in the tree: {ep}")
                elif sym not in _module_bindings(tree_of(f)):
                    errors.append(f"{cid}: entry point not bound at module level in {f.relative_to(root)}: {ep}")
                elif a.do_import:
                    try:
                        m = __import__(mod, fromlist=[sym])
                        getattr(m, sym)
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{cid}: import failed for {ep}: {e!r}")
            else:
                if not module_shipped(ep, py):
                    errors.append(f"{cid}: module is not shipped by pyproject (py-modules / packages): {ep}")
                elif module_file(root, ep, py) is None:
                    errors.append(f"{cid}: module not found in the tree: {ep}")
        for claim_id in cap["claim_ids"]:
            c = by_id.get(claim_id)
            if c is None:
                errors.append(f"{cid}: claim id not in {proj['claims_file']}: {claim_id}")
            elif c["status"] not in ACTIVE_STATUSES:
                errors.append(f"{cid}: claim {claim_id} has status {c['status']!r} -- only "
                              f"{sorted(ACTIVE_STATUSES)} can back a capability")
        for inst in cap["install"]:
            cmd = inst["command"]
            targets = install_targets(cmd)
            if not targets:
                errors.append(f"{cid}: install command has no pip install target: {cmd!r}")
            for name, kind in targets:
                if kind == "source":
                    continue
                n = pep503_name(name)
                if n in aliases:
                    errors.append(f"{cid}: install command uses alias {name!r}, not the canonical package: {cmd!r}")
                elif n not in allowed_pkgs:
                    errors.append(f"{cid}: install command names {name!r}, which is neither canonical nor a related project: {cmd!r}")
    for rp in a.related:
        other = json.loads(read_text(rp))
        back = {r["canonical_package"] for r in other["project"].get("related_projects", [])}
        if proj["canonical_package"] not in back:
            errors.append(f"reciprocity: {other['project']['canonical_package']} does not list {proj['canonical_package']} as related")
        if other["project"]["canonical_package"] not in related_pkgs:
            errors.append(f"reciprocity: this project does not list {other['project']['canonical_package']} as related")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OK: {a.capabilities}: {len(doc['capabilities'])} capabilities, "
          f"{sum(len(c['claim_ids']) for c in doc['capabilities'])} claim references, canonical package {pname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
