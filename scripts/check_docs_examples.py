#!/usr/bin/env python3
"""Documentation examples and links, offline.

For docs/SOLUTIONS.md, docs/solutions/*.md, AGENTS.md and llms.txt:
  * every fenced ```python block parses (ast) -- syntax is what CI can
    check; execution needs a GPU, the network or a model and is marked as
    such in the page (the block's preceding "needs" note);
  * every fenced ```python block in docs/solutions/ is preceded, within
    the previous 6 lines, by a note saying what it needs (CPU-only, GPU,
    network, model download, storage) -- no example pretends;
  * every relative Markdown link resolves to a file in the tree;
  * every self-repository https link (blob/main or tree/main) resolves to a
    file in the tree.
Standard library only.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

FENCE = re.compile(r"```(\w+)?[^\n]*\n(.*?)```", re.S)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
NEEDS = re.compile(r"(needs|requires|runs on|cpu-only|cpu only|gpu|network|download|storage)", re.I)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    slug = None
    try:
        import subprocess
        url = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        slug = m.group(1) if m else None
    except Exception:  # noqa: BLE001
        pass
    files = [root / "docs" / "SOLUTIONS.md", root / "AGENTS.md", root / "llms.txt"] + sorted((root / "docs" / "solutions").glob("*.md"))
    errors = []
    n_blocks = 0
    for f in files:
        if not f.exists():
            errors.append(f"missing: {f.relative_to(root)}")
            continue
        text = f.read_text(errors="replace")
        lines = text.splitlines()
        for m in FENCE.finditer(text):
            lang = (m.group(1) or "").lower()
            if lang in ("python", "py"):
                n_blocks += 1
                try:
                    ast.parse(m.group(2))
                except SyntaxError as e:
                    errors.append(f"{f.relative_to(root)}: python block does not parse: {e}")
                if "docs/solutions" in str(f):
                    start_line = text[:m.start()].count("\n")
                    context = "\n".join(lines[max(0, start_line - 6):start_line + 1])
                    if not NEEDS.search(context):
                        errors.append(f"{f.relative_to(root)}:{start_line + 1}: python block has no 'needs' note (CPU-only / GPU / network / download)")
        for m in LINK.finditer(text):
            href = m.group(1)
            if href.startswith("#") or href.startswith("mailto:"):
                continue
            if href.startswith("http"):
                if slug and f"github.com/{slug}/" in href:
                    mm = re.search(rf"github\.com/{re.escape(slug)}/(?:blob|tree)/main/([^#?]+)", href)
                    if mm and not (root / mm.group(1)).exists():
                        errors.append(f"{f.relative_to(root)}: self-repo link to a missing path: {mm.group(1)}")
                continue
            target = (f.parent / href.split("#")[0]).resolve()
            if not target.exists():
                errors.append(f"{f.relative_to(root)}: relative link does not resolve: {href}")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OK: {len(files)} documents, {n_blocks} python blocks parse, every local and self-repo link resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
