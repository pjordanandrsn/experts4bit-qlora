# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's copies of Q235's bash scripts against Q235's own, line for line: the declared substitutions, and nothing else
(bench/fam235/PREREG-r2-mixtral.md, Amendment 2's harness decision).

Each copy is compared with its Q235 source after the declared renames; the leading comment block (after the shebang) is the
copy's own; every other difference must be a one-for-one line replacement that a declared rule names on both sides.

- **STRICT** against the source at ``CITED``, the Q235 commit R2 cites (``git show``): any undeclared difference fails.
- **The Python twin.** Every file a wrapped Q235 module (``WRAPPED``) reads through its ``HERE`` is classified; a new one fails.
- **Cited, not live.** ``cited_tree`` gives the Q235 files R2 runs and seals at ``CITED``; R2's wrappers and its CI read those.
- **REPORTED** against the source in the working tree: a Q235 change after ``CITED`` (a fix R2 has not taken) is listed, not
  failed, so an open Q235 amendment never turns R2 red. R2 takes it by updating its copy and ``CITED`` together.

    python r2_drift.py        (exit 1 on a strict difference; the working-tree report is printed either way)
"""
from __future__ import annotations

import atexit
import difflib
import functools
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CITED = "9480ed0386bd13c7666ae2b4d297633e522e78e7"   # e4b main at #1709's merge (Q235 Amendment 10), the commit R2 re-cites

PAIRS = {
    "bench/fam235/r2_run.sh": {
        "source": "bench/q235/q235_run.sh",
        "renames": [("q235.sha256", "r2.sha256"), ("q235_tiny.py", "r2_tiny.py"), ("] q235: ", "] r2: "),
                    ('"q235-run/1"', '"r2-run/1"'),
                    ("--only 'tokenizer*' vocab.json merges.txt", "--only 'tokenizer*' special_tokens_map.json")],
        "lines": [(r"^  science\) CARD_RE=",
                   r"^  science\) CARD_RE='\^NVIDIA GeForce RTX 5090'; MIN_VRAM_MIB=32000; MIN_RAM_GB=134; MIN_DISK_GB=350 ;; "),
                  # R2 Amendment 3: R2's own fit (its caps and its pull floor) and Q235's cold host-rate floors, each R2's line.
                  # R2 Amendment 4: Q235 Amendment 7's box refit comment (four lines) against R2's own four.
                  (r"^# Box: Amendment 7's refit ", r"^# Fitted \(r2_fit\.py, R2 Amendment 3\) "),
                  (r"^# warm against warm ", r"^# host listing at least 500 Mbps, box "),
                  (r"^# \(q235_fit rounds up to the second\)", r"^# R2 Amendment 4 re-cites Q235 and keeps these caps: "),
                  (r"^# the fetch \(7,935 s at 500 Mbps ", r"^# timed MoE pass, which R2's box does not run "),
                  (r"^CAP_GIT=", r"^CAP_GIT=240 CAP_VENV=120 CAP_INSTALL=360 CAP_HOSTRATE=600 CAP_FETCH=7080 CAP_TOKENS=300 CAP_BOX=3600 "
                                r"CAP_REPEAT=2040$"),
                  (r"^# The host-rate floors \(gate 2's premise\), ", r"^# The host-rate floors \(gate 2's premise\): Q235's sealed cold ones"),
                  (r"^# dominant part is the host-side", r"^# read its fresh slice warm\)"),
                  (r"^# O_DIRECT\) before the checkpoint", r"^# three \(the disk with O_DIRECT\) before the checkpoint"),
                  (r"^MIN_FETCH_GBS=", r"^MIN_FETCH_GBS=0\.063$"),
                  # The rehearsal's depth is R2's registration, not plumbing: a Q235 re-cite may move Q235's, never R2's.
                  (r"^  a2000\)   must fetch ", r"^  a2000\)   must fetch .* --slice 4 --slice-dir \"\$W/hf-cache/slice4\" "),
                  (r"^           MODEL=\"\$W/hf-cache/slice\d+\" ", r"^           MODEL=\"\$W/hf-cache/slice4\" DEVICE=cuda ENABLER=fast ;;$"),
                  # The box's shape is R2's registration: Q235 Amendment 6 runs shape B (--micro-batch 2); R2 keeps shape A.
                  (r"^  extra=\(.*\); cap=\$CAP_BOX; ",
                   r"^  extra=\(\); cap=\$CAP_BOX; \[ \"\$run\" = repeat \] && \{ extra=\(--repeat --plants\); cap=\$CAP_REPEAT; \}$")],
    },
    "bench/fam235/r2_drive.sh": {
        "source": "bench/q235/q235_drive.sh",
        "renames": [("bench/q235/q235_fetch_gate.py", "bench/fam235/r2_fetch_gate.py"), ("q235_fetch_gate", "r2_fetch_gate"),
                    ("Q235_FETCH_GATE", "R2_FETCH_GATE"), ("Q235_ASSET_DIR", "R2_ASSET_DIR"), ("q235_stage.py", "r2_stage.py"),
                    ("TC1_RUNNER=q235_run.sh", "TC1_RUNNER=r2_run.sh"), ("Q235 refuses", "R2 refuses"), ("'Q235: ", "'R2: ")],
        "lines": [],
    },
}

# The Q235 modules R2 runs unchanged through a wrapper, and every file each reads relative to its own directory (``HERE``), as of
# ``CITED``: "r2" (the wrapper maps it to R2's), "q235" (R2 reads Q235's on purpose) or "overridden" (only a function the wrapper
# replaces reads it). A name Q235 adds later is unclassified until it is declared here, so it cannot reach a box in R2's name.
WRAPPED = {
    "bench/q235/q235_fetch_gate.py": {
        "names": {"model_files.json": "r2", "wheels.json": "q235", "q235_bootstrap.py": "q235", "q235_stage.py": "overridden",
                  "q235.sha256": "overridden", "locked_inputs.json": "overridden", "q235_drive.sh": "overridden",
                  "PREREG-q235.md": "overridden", "AMENDMENT*.md": "overridden"},
        "attrs": {"parents", "glob"}},
    "bench/q235/q235_fit.py": {
        "names": {"model_files.json": "r2", "q235_run.sh": "r2", "wheels.json": "q235"},
        "attrs": set()},
    "bench/q235/q235_stage.py": {
        "names": {"q235.sha256": "overridden", "locked_inputs.json": "q235"},
        "attrs": {"parents"}},
}
FILE_NAME = re.compile(r"^[\w*.-]+\.[A-Za-z0-9]+$")


def here_reads(text: str) -> tuple:
    """The file names and attributes a module reaches through ``HERE``: every file-like string literal on a code line that uses
    ``HERE`` (``HERE / "x"``, ``HERE / n for n in (...)``, ``HERE.glob("x")``), and every ``HERE.<attr>``."""
    names, attrs = set(), set()
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if not re.search(r"\bHERE\b", code) or re.match(r"\s*HERE\s*=", code):
            continue
        names |= {x for x in re.findall(r'"([^"]+)"', code) if FILE_NAME.match(x)}
        attrs |= set(re.findall(r"\bHERE\.(\w+)", code))
    return names, attrs


def unclassified(text: str, rule: dict) -> list:
    names, attrs = here_reads(text)
    out = [f"reads {n!r} through HERE: not classified" for n in sorted(names - set(rule["names"]))]
    out += [f"declares {n!r}, which it no longer reads" for n in sorted(set(rule["names"]) - names)]
    out += [f"uses HERE.{a}: not declared" for a in sorted(attrs - rule["attrs"])]
    return out


def check_wrapped(commit: str | None = CITED) -> dict:
    return {src: unclassified(source_at(src, commit), rule) for src, rule in WRAPPED.items()}


# What R2 takes of Q235's at ``CITED``: the tools it runs through its wrappers and the files its closure seals, plus the shared files
# its closure carries. ``cited_tree`` extracts them from git, so a later Q235 change reaches R2 only when R2 re-cites.
CITED_PATHS = ("bench/q235", "bench/p130", "bench/tc1/tc1_arm.py", "bench/dq11/dq11_require_git.sh")


@functools.lru_cache(maxsize=None)
def cited_tree(commit: str = CITED) -> Path:
    """``CITED_PATHS`` at ``commit``, from ``git archive`` into a private temporary tree (removed at exit). A missing commit (a
    shallow clone) raises, naming it: R2 never falls back to the working tree."""
    arc = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "--", *CITED_PATHS], capture_output=True)
    if arc.returncode != 0:
        raise RuntimeError(f"cannot read Q235 at {commit} (a shallow clone, or not a git checkout?): "
                           f"{arc.stderr.decode(errors='replace').strip()[:200]}")
    root = Path(tempfile.mkdtemp(prefix=f"r2-cited-{commit[:8]}-"))
    atexit.register(shutil.rmtree, root, True)
    with tarfile.open(fileobj=io.BytesIO(arc.stdout)) as t:
        for m in t.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise RuntimeError(f"git archive at {commit} holds an unexpected member {m.name!r}")
        t.extractall(root, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))    # the stdlib's filter too, if any
    return root


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
    wrapped = check_wrapped(CITED)
    for src, diffs in wrapped.items():
        print(f"strict {src}'s HERE reads at {CITED[:8]}: " + ("ok" if not diffs else f"{len(diffs)} unclassified"))
        for d in diffs:
            print("   ", d)
    for copy, diffs in check(None).items():
        if diffs:
            print(f"reported {copy}: Q235's working-tree source has moved since {CITED[:8]} ({len(diffs)} differences)")
    sys.exit(1 if any(strict.values()) or any(wrapped.values()) else 0)


if __name__ == "__main__":
    main()
