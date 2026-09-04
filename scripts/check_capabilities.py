#!/usr/bin/env python3
"""Validate docs/capabilities.json against its schema and against the repository.

Standard library only. Exit 1 on the first class of failure found, after printing
every failure in that run. Checks:

  * valid JSON, and the fields/types/enums the schema requires;
  * project.canonical_package equals [project].name in pyproject.toml;
  * every documentation path exists;
  * every entry point resolves: ``module:Symbol`` is a def/class/assignment in
    the module's source (static AST, no import -- the kernels need a GPU);
    ``module`` is a source file; ``cli:name`` is a console script in pyproject
    and ``cli:python -m module`` a module main;
    ``flag:NAME`` is a documented flag (the NAME occurs in a documentation path);
  * every claim ID exists in docs/claims.json and is not retired/superseded;
  * install commands name the canonical package (or a related project's) and
    never an alias;
  * capability IDs are unique;
  * --related <path/to/other/capabilities.json>: the other project lists this
    one back (reciprocity), when given.

Run with --import to also import each ``module:Symbol`` (needs the runtime).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

RETIRED = {"retired", "superseded", "withdrawn", "falsified"}


def _load_pyproject(root: Path) -> dict:
    p = root / "pyproject.toml"
    try:
        import tomllib  # py3.11+
        return tomllib.loads(p.read_text())
    except ImportError:  # 3.9/3.10: the two fields this script reads
        txt = p.read_text()
        name = re.search(r'^\s*name\s*=\s*"([^"]+)"', txt, re.M)
        scripts = re.search(r"\[project\.scripts\](.*?)(?:\n\[|\Z)", txt, re.S)
        cli = {}
        if scripts:
            for m in re.finditer(r'^\s*([\w-]+)\s*=\s*"([^"]+)"', scripts.group(1), re.M):
                cli[m.group(1)] = m.group(2)
        return {"project": {"name": name.group(1) if name else None, "scripts": cli}}


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


def _symbol_in_source(root: Path, module: str, symbol: str) -> bool:
    """True when ``symbol`` is defined, assigned or imported ANYWHERE in the
    module's source (conditional imports inside try/if count -- that is how
    experts4bit_qlora binds Experts4bit)."""
    for cand in (root / (module.replace(".", "/") + ".py"),
                 root / module.replace(".", "/") / "__init__.py",
                 root / "kernel" / (module + ".py")):
        if cand.exists():
            tree = ast.parse(cand.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                    return True
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == symbol:
                            return True
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        if (a.asname or a.name.split(".")[-1]) == symbol:
                            return True
            return False
    return False


def _module_exists(root: Path, module: str) -> bool:
    return any(c.exists() for c in (root / (module.replace(".", "/") + ".py"),
                                    root / module.replace(".", "/") / "__init__.py",
                                    root / "kernel" / (module + ".py")))


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
        doc = json.loads(cap_path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {cap_path}: not valid JSON: {e}")
        return 1
    schema = json.loads((root / a.schema).read_text())
    _schema_check(doc, schema, errors)
    if errors:
        for e in errors:
            print("SCHEMA:", e)
        return 1
    proj = doc["project"]
    py = _load_pyproject(root)
    pname = py.get("project", {}).get("name")
    if proj["canonical_package"] != pname:
        errors.append(f"canonical_package {proj['canonical_package']!r} != pyproject name {pname!r}")
    aliases = set(proj.get("aliases", []))
    related_pkgs = {r["canonical_package"] for r in proj.get("related_projects", [])}
    allowed_pkgs = {proj["canonical_package"]} | related_pkgs
    cli = py.get("project", {}).get("scripts", {}) or {}
    claims_doc = json.loads((root / proj["claims_file"]).read_text())
    claims = claims_doc["claims"] if isinstance(claims_doc, dict) and "claims" in claims_doc else claims_doc
    by_id = {c["id"]: c for c in claims}
    ids_seen = set()
    for cap in doc["capabilities"]:
        cid = cap["id"]
        if cid in ids_seen:
            errors.append(f"{cid}: duplicate capability id")
        ids_seen.add(cid)
        for d in cap["documentation"]:
            if not (root / d).exists():
                errors.append(f"{cid}: documentation path missing: {d}")
        doc_text = "\n".join((root / d).read_text(errors="replace") for d in cap["documentation"] if (root / d).exists())
        for ep in cap["entrypoints"]:
            if ep.startswith("cli:"):
                name = ep[4:]
                m = re.fullmatch(r"python -m ([\w.]+)(?:\s.*)?", name)
                if m:                       # a `python -m module` CLI: the module must exist
                    if not _module_exists(root, m.group(1)):
                        errors.append(f"{cid}: `python -m` module not found: {ep}")
                elif name not in cli:
                    errors.append(f"{cid}: console script not in pyproject: {ep}")
            elif ep.startswith("flag:"):
                if ep[5:] not in doc_text:
                    errors.append(f"{cid}: flag {ep[5:]} is not documented in this capability's documentation")
            elif ":" in ep:
                mod, sym = ep.split(":", 1)
                if not _symbol_in_source(root, mod, sym):
                    errors.append(f"{cid}: entry point not found in source: {ep}")
                elif a.do_import:
                    try:
                        m = __import__(mod, fromlist=[sym])
                        getattr(m, sym)
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{cid}: import failed for {ep}: {e!r}")
            else:
                if not _module_exists(root, ep):
                    errors.append(f"{cid}: module not found: {ep}")
        for claim_id in cap["claim_ids"]:
            c = by_id.get(claim_id)
            if c is None:
                errors.append(f"{cid}: claim id not in {proj['claims_file']}: {claim_id}")
            elif str(c.get("status", "")).lower() in RETIRED or str(c.get("tier", "")).lower() in RETIRED:
                errors.append(f"{cid}: claim {claim_id} is {c.get('status')}/{c.get('tier')} -- a retired claim cannot back a capability")
        for inst in cap["install"]:
            cmd = inst["command"]
            m = re.search(r"pip install\s+(?:-[\w-]+\s+)*\"?([A-Za-z0-9_.-]+)", cmd)
            if not m:
                errors.append(f"{cid}: install command is not a pip install: {cmd!r}")
                continue
            pkg = m.group(1)
            if pkg in aliases:
                errors.append(f"{cid}: install command uses alias {pkg!r}, not the canonical package: {cmd!r}")
            elif pkg not in allowed_pkgs and not pkg.startswith("git+"):
                errors.append(f"{cid}: install command names {pkg!r}, which is neither canonical nor a related project: {cmd!r}")
    for rp in a.related:
        other = json.loads(Path(rp).read_text())
        back = {r["canonical_package"] for r in other["project"].get("related_projects", [])}
        if proj["canonical_package"] not in back:
            errors.append(f"reciprocity: {other['project']['canonical_package']} does not list {proj['canonical_package']} as related")
        if other["project"]["canonical_package"] not in related_pkgs:
            errors.append(f"reciprocity: this project does not list {other['project']['canonical_package']} as related")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OK: {cap_path.relative_to(root)}: {len(doc['capabilities'])} capabilities, "
          f"{sum(len(c['claim_ids']) for c in doc['capabilities'])} claim references, canonical package {pname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
