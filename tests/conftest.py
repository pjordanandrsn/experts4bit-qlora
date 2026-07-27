# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Stop one missing heavy dependency from killing the whole suite at collection.

A test module that does ``import torch`` (or ``from experts4bit_qlora import
...``, which pulls ``bitsandbytes`` transitively) at *module scope* raises
during **collection** on a machine without that dependency. pytest aborts the
whole session on a collection error, so a single unguarded file takes down every
other test — including the pure-Python ones that would have run fine.

This has now bitten three times:

* three files at once, hiding 32 passing tests and 15 runtime failures behind
  ``Interrupted: 2 errors``;
* ``test_hot_residency_gptoss.py``, which arrived in ``main`` *after* that fix
  and reintroduced the pattern;
* and it will happen again, because nothing stops the next new file from doing
  it. Remembering to write ``pytest.importorskip`` is not a mechanism.

So the guard lives here instead. Each test module's **top-level** imports are
read statically; if one of them needs a dependency this machine does not have,
and the module does not already guard it, the file is skipped at collection
instead of exploding. Nested/function-level imports are ignored on purpose —
they do not execute during collection.

The files skipped this way are **reported**, never silent: a quiet skip is how
coverage rots. Adding ``pytest.importorskip`` to a file is still better than
relying on this — the module then reports as *skipped* rather than vanishing —
and ``test_import_hygiene.py`` nudges toward that. This is the backstop, not the
recommendation.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

#: Dependencies whose absence breaks collection rather than merely a test.
HEAVY = ("torch", "bitsandbytes", "triton", "transformers")

#: First-party packages that pull a heavy dependency at import time.
#: ``experts4bit_qlora/__init__`` reaches ``_vendor.experts``, which does
#: ``import bitsandbytes.functional`` at module scope.
TRANSITIVE = {"experts4bit_qlora": ("torch", "bitsandbytes")}


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


MISSING = frozenset(d for d in HEAVY if not _installed(d))

_skipped: list[tuple[str, str]] = []


def _toplevel_imports(path: Path):
    """(module, lineno) for every import executed at module scope."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out = []
    for node in tree.body:                      # body only == module scope
        if isinstance(node, ast.Import):
            out += [(a.name.split(".")[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module.split(".")[0], node.lineno))
    return out


def _guarded(path: Path):
    """{module: lineno} for each top-level ``pytest.importorskip("mod")``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            name = node.args[0].value.split(".")[0]
            found.setdefault(name, node.lineno)
    return found


def _blocking_dep(path: Path):
    """The missing dependency that would break collecting `path`, if any."""
    if not MISSING:
        return None
    guards = _guarded(path)
    # Any importorskip for a dependency this machine LACKS will raise Skipped
    # when the module executes, so everything after that line is unreachable —
    # the module skips cleanly and must NOT be ignored. Only a risky import
    # that runs BEFORE every such guard can actually break collection.
    #
    # Missing this is why an earlier version flagged 5 already-safe modules:
    # e.g. test_serve.py guards `fastapi` (absent) at line 15 and only then
    # imports experts4bit_qlora at 18. It never errors; it skips.
    # NB: effectiveness is decided by whether the guarded module is actually
    # installed -- NOT by whether it is in HEAVY. test_serve.py guards
    # `fastapi`, which is absent here and skips the module long before its
    # experts4bit_qlora import; restricting this to HEAVY missed that.
    first_effective_skip = min(
        (ln for dep, ln in guards.items() if not _installed(dep)), default=None
    )
    for mod, lineno in _toplevel_imports(path):
        needed = {mod} | set(TRANSITIVE.get(mod, ()))
        if not (needed & MISSING):
            continue
        if first_effective_skip is not None and first_effective_skip < lineno:
            continue                      # a guard fires first: clean skip
        return sorted(needed & MISSING)[0]
    return None


def pytest_ignore_collect(collection_path, config):
    """
    .. warning::
       Keep this signature EXACTLY as the hookspec declares it. pluggy matches
       hookimpls by **argument name**, so a permissive
       ``(collection_path=None, path=None, config=None, **_)`` registers as a
       plugin -- ``--trace-config`` even lists it -- and is then **never
       called**. That failure is silent: the suite simply keeps aborting as if
       no guard existed.
    """
    p = Path(str(collection_path))
    if p.suffix != ".py" or not p.name.startswith("test_"):
        return None
    dep = _blocking_dep(p)
    if dep:
        _skipped.append((p.name, dep))
        return True
    return None


def pytest_report_collectionfinish(config, items):
    """Say out loud what was skipped. A silent skip is how coverage rots."""
    if not _skipped:
        return None
    lines = [
        f"conftest: skipped {len(_skipped)} module(s) needing dependencies "
        f"this machine lacks ({', '.join(sorted(MISSING))}) — collecting them "
        f"would abort the whole session:"
    ]
    lines += [f"  - {name}  (needs {dep})" for name, dep in sorted(_skipped)]
    lines.append("  add `pytest.importorskip(...)` to these so they report as "
                 "SKIPPED rather than disappearing.")
    return lines
