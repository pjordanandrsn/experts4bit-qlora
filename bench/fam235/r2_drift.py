# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's copies of Q235's bash scripts against Q235's own, line for line: the declared substitutions, and nothing else
(bench/fam235/PREREG-r2-mixtral.md, Amendment 2's harness decision).

Each copy is compared with its Q235 source after the declared renames; the leading comment block (after the shebang) is the
copy's own; every other difference must be a one-for-one line replacement that a declared rule names on both sides.

- **STRICT** against the source at ``CITED``, the Q235 commit R2 cites (``git show``): any undeclared difference fails.
- **REPORTED** against the source in the working tree: a Q235 change after ``CITED`` (a fix R2 has not taken) is listed, not
  failed, so an open Q235 amendment never turns R2 red. R2 takes it by updating its copy and ``CITED`` together.

    python r2_drift.py        (exit 1 on a strict difference; the working-tree report is printed either way)
"""
from __future__ import annotations

import difflib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CITED = "eca70aff0807339cca706ec728964079baccd99d"   # e4b main when the copies were made; Q235's latest change in it is Amendment 5

PAIRS = {
    "bench/fam235/r2_run.sh": {
        "source": "bench/q235/q235_run.sh",
        "renames": [("q235.sha256", "r2.sha256"), ("q235_tiny.py", "r2_tiny.py"), ("] q235: ", "] r2: "),
                    ('"q235-run/1"', '"r2-run/1"'),
                    ("--only 'tokenizer*' vocab.json merges.txt", "--only 'tokenizer*' special_tokens_map.json")],
        "lines": [(r"^  science\) CARD_RE=", r"^  science\) say 'refusing: no box is registered ")],
    },
    "bench/fam235/r2_drive.sh": {
        "source": "bench/q235/q235_drive.sh",
        "renames": [("bench/q235/q235_fetch_gate.py", "bench/fam235/r2_fetch_gate.py"), ("q235_fetch_gate", "r2_fetch_gate"),
                    ("Q235_FETCH_GATE", "R2_FETCH_GATE"), ("Q235_ASSET_DIR", "R2_ASSET_DIR"), ("q235_stage.py", "r2_stage.py"),
                    ("TC1_RUNNER=q235_run.sh", "TC1_RUNNER=r2_run.sh"), ("Q235 refuses", "R2 refuses"), ("'Q235: ", "'R2: ")],
        "lines": [],
    },
}


def _body(lines):
    """The lines after the shebang and the leading comment block."""
    i = 1
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return lines[i:]


def undeclared(source: str, copy: str, rule: dict) -> list:
    """Every difference between the copy and its renamed source that no declared rule names."""
    for a, b in rule["renames"]:
        source = source.replace(a, b)
    s, c = source.splitlines(), copy.splitlines()
    out = [] if s[:1] == c[:1] else ["the shebang differs"]
    s, c = _body(s), _body(c)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        pairs = list(zip(s[i1:i2], c[j1:j2]))
        if tag == "replace" and i2 - i1 == j2 - j1 and all(
                any(re.match(src, x) and re.match(dst, y) for src, dst in rule["lines"]) for x, y in pairs):
            continue
        out.append(f"{tag}: Q235 {s[i1:i2][:2]} / R2 {c[j1:j2][:2]}")
    return out


def source_at(path: str, commit: str | None) -> str:
    if commit is None:
        return (REPO / path).read_text()
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cannot read {path} at {commit} (a shallow clone?): {r.stderr.strip()[:200]}")
    return r.stdout


def check(commit: str | None = CITED) -> dict:
    """{copy: [undeclared differences]} against the sources at ``commit`` (None: the working tree)."""
    return {copy: undeclared(source_at(rule["source"], commit), (REPO / copy).read_text(), rule) for copy, rule in PAIRS.items()}


def main():
    strict = check(CITED)
    for copy, diffs in strict.items():
        print(f"strict {copy} against {PAIRS[copy]['source']} at {CITED[:8]}: " + ("ok" if not diffs else f"{len(diffs)} undeclared"))
        for d in diffs:
            print("   ", d)
    for copy, diffs in check(None).items():
        if diffs:
            print(f"reported {copy}: Q235's working-tree source has moved since {CITED[:8]} ({len(diffs)} differences)")
    sys.exit(1 if any(strict.values()) else 0)


if __name__ == "__main__":
    main()
