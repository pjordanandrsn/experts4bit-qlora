#!/usr/bin/env python3
"""Documentation examples and links, offline. Standard library only.

The document set is the union of the docs/llms-bundle.json sources (globs
expanded), docs/solutions/*.md, docs/SOLUTIONS.md, AGENTS.md and llms.txt --
.md and .txt only. For each:
  * every fenced ```python block parses (ast) -- syntax is what CI can check;
    execution needs a GPU, the network or a model, and the page says so;
  * every fenced ```python block in docs/solutions/ carries an explicit
    "needs" token: within the 6 lines before the fence (``Needs: ...`` /
    ``Requires: ...``) or as the block's first comment line
    (``# CPU-only ...`` / ``# GPU ...`` / ``# needs ...``) -- no example
    pretends;
  * every relative Markdown link in the prose resolves to a file in the tree
    (links inside fenced blocks are code, not links);
  * every self-repository https link pinned to ``blob|tree/main/`` or the
    current ``v<project.version>`` resolves to a file in the tree. The
    repository slug comes from [project.urls] or the git remote; when neither
    says, this FAILS (exit 2) rather than skipping the self-repo checks.

``--run-cpu-blocks DIR`` additionally EXECUTES every python block whose first
line starts with ``# CPU-only``, with the current interpreter, cwd=DIR and
PYTHONPATH=DIR, reporting pass/fail per block and page; any failure exits 1.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import bundle_sources, load_pyproject, md_fences, md_links, read_text, self_slug  # noqa: E402

NEEDS = re.compile(r"(?i)\b(needs?|requires?)\b\s*:|#\s*(cpu-only|gpu|needs)")
CPU_BLOCK_PREFIX = "# CPU-only"
DOC_SUFFIXES = (".md", ".txt")
RUN_TIMEOUT_S = 600


def _doc_list(root: Path) -> list[str]:
    docs: list[str] = []
    bundle = root / "docs" / "llms-bundle.json"
    if bundle.is_file():
        docs += [it["path"] for it in bundle_sources(root, json.loads(read_text(bundle)))]
    docs += [f.relative_to(root).as_posix() for f in sorted((root / "docs" / "solutions").glob("*.md"))]
    docs += ["AGENTS.md", "llms.txt", "docs/SOLUTIONS.md"]
    return [d for d in dict.fromkeys(docs) if d.endswith(DOC_SUFFIXES)]


def _has_needs_note(lines: list[str], start_line: int, body: str) -> bool:
    """``start_line`` is the 1-based line of the opening fence: look at the 6
    lines above it, then at the block's first line when it is a comment."""
    before = lines[max(0, start_line - 7):start_line - 1]
    if any(NEEDS.search(ln) for ln in before):
        return True
    first = body.split("\n", 1)[0].strip()
    return first.startswith("#") and NEEDS.search(first) is not None


def _check_docs(root: Path, docs: list[str], slug: str, refs: set[str]) -> tuple[list[str], int]:
    self_link = re.compile(rf"https?://github\.com/{re.escape(slug)}/(?:blob|tree)/([^/]+)/([^#?]+)")
    errors: list[str] = []
    n_blocks = 0
    for rel in docs:
        f = root / rel
        if not f.is_file():
            errors.append(f"missing: {rel}")
            continue
        text = read_text(f)
        lines = text.splitlines()
        for lang, body, start in md_fences(text):
            if lang not in ("python", "py"):
                continue
            n_blocks += 1
            try:
                ast.parse(body)
            except SyntaxError as e:
                errors.append(f"{rel}:{start}: python block does not parse: {e}")
            if rel.startswith("docs/solutions/") and not _has_needs_note(lines, start, body):
                errors.append(f"{rel}:{start}: python block has no explicit 'needs' token "
                              "(Needs: / Requires: within 6 lines above, or a first-line '# CPU-only' / '# GPU' / '# needs' comment)")
        for href in md_links(text):
            if href.startswith(("#", "mailto:")):
                continue
            if href.startswith(("http://", "https://")):
                m = self_link.match(href)
                if m and m.group(1) in refs:
                    path = urllib.parse.unquote(m.group(2))
                    if not (root / path).exists():
                        errors.append(f"{rel}: self-repo link ({m.group(1)}) to a missing path: {path}")
                continue
            target = urllib.parse.unquote(href.split("#", 1)[0].split("?", 1)[0])
            if target and not (f.parent / target).resolve().exists():
                errors.append(f"{rel}: relative link does not resolve: {href}")
    return errors, n_blocks


def _run_cpu_blocks(root: Path, docs: list[str], run_dir: Path) -> tuple[int, int]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(run_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    n_ok = n_fail = 0
    for rel in docs:
        f = root / rel
        if not f.is_file():
            continue
        page_ok = page_fail = 0
        for lang, body, start in md_fences(read_text(f)):
            if lang not in ("python", "py") or not body.split("\n", 1)[0].startswith(CPU_BLOCK_PREFIX):
                continue
            with tempfile.TemporaryDirectory() as td:
                script = Path(td) / f"{Path(rel).stem}_L{start}.py"
                script.write_text(body, encoding="utf-8")
                t0 = time.monotonic()
                try:
                    r = subprocess.run([sys.executable, str(script)], cwd=str(run_dir), env=env,
                                       capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
                    code, err = r.returncode, r.stderr
                except subprocess.TimeoutExpired as e:
                    code, err = -1, f"timed out after {RUN_TIMEOUT_S}s\n{e.stderr or ''}"
            dt = time.monotonic() - t0
            if code == 0:
                page_ok += 1
                print(f"RUN ok   {rel}:{start} ({dt:.1f}s)")
            else:
                page_fail += 1
                print(f"RUN FAIL {rel}:{start} exit {code} ({dt:.1f}s)")
                for ln in (err or "").strip().splitlines()[-12:]:
                    print(f"         | {ln}")
        if page_ok or page_fail:
            print(f"RUN page {rel}: {page_ok} ok, {page_fail} failed")
        n_ok += page_ok
        n_fail += page_fail
    print(f"RUN total: {n_ok} CPU-only blocks ok, {n_fail} failed (cwd={run_dir}, {sys.executable})")
    return n_ok, n_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--run-cpu-blocks", metavar="DIR", default=None,
                    help="execute every '# CPU-only' python block with cwd=DIR and PYTHONPATH=DIR")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    py = load_pyproject(root)
    slug = self_slug(root, py)
    if slug is None:
        print("FAIL: self-repo slug unknown; self-repo links NOT checked "
              "(no github.com URL under [project.urls] Source/Repository/Homepage and no git origin)")
        return 2
    version = str(py.get("version", "")).strip()
    refs = {"main"} | ({f"v{version}"} if version else set())
    docs = _doc_list(root)
    errors, n_blocks = _check_docs(root, docs, slug, refs)
    for e in errors:
        print("FAIL:", e)
    if not errors:
        print(f"OK: {len(docs)} documents, {n_blocks} python blocks parse, every local and self-repo "
              f"({slug} @ {'/'.join(sorted(refs))}) link resolves")
    failed = bool(errors)
    if a.run_cpu_blocks is not None:
        run_dir = Path(a.run_cpu_blocks).resolve()
        if not run_dir.is_dir():
            print(f"FAIL: --run-cpu-blocks {a.run_cpu_blocks}: not a directory")
            return 2
        _n_ok, n_fail = _run_cpu_blocks(root, docs, run_dir)
        failed = failed or n_fail > 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
