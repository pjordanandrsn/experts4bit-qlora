#!/usr/bin/env python3
"""CHANGELOG.md's ``## Unreleased`` section, kept as one file per change in
``changelog.d/``. Standard library plus git; no network.

Why: on 2026-10-05 about ten pull requests an hour each inserted a ``### ``
section at the top of ``## Unreleased``. Every merge left every other open pull
request DIRTY on that one hunk, and one of them (#1136, commit a5f88f2a)
replaced the whole 8,187-line file with 31 lines. GitHub's mergeability check
ignores ``.gitattributes`` merge drivers, so ``merge=union`` does not help. A
new file never conflicts.

The rule:

  * A change adds ``changelog.d/<pr-or-slug>.md`` holding exactly the
    ``### Title`` and body it would have put under ``## Unreleased``. Nobody
    edits CHANGELOG.md's ``## Unreleased`` by hand; its body is ``POINTER``.
  * A release (``--release``) writes the fragments into a new versioned section
    below ``## Unreleased`` and deletes them. Order: newest first, by the
    first-parent commit that added each fragment; fragments one commit added
    are in reverse name order, so ``<pr>``-named fragments still read newest
    first; fragments not committed yet come first, in the same name order.
  * Released history is append-only. Every line of the merge base's released
    sections (from the first ``## X.Y.Z — date`` heading to the end) survives,
    in order, at the head, and a fragment that existed at the merge base leaves
    only by being released. A deliberate edit (a correction an owner asked for,
    an entry filed under the wrong release) passes with
    ``--allow-history-edit``, which CI sets only when the pull request carries
    the ``changelog-history-edit`` label.

Unreleased is assembled at release time only, never rendered in-tree: an
in-tree rendering would be the same shared hunk every pull request rewrites,
and a CI check of it would go stale on every open pull request after each
merge. ``--render`` prints what the section would say now.

    python scripts/changelog_fragments.py --check                       # fragments + CHANGELOG.md shape
    python scripts/changelog_fragments.py --check --base origin/main    # and history against the merge base
    python scripts/changelog_fragments.py --render                      # the Unreleased section, assembled
    python scripts/changelog_fragments.py --release "## 0.49.0 — 2026-10-06 — <title>" [--intro-file F]

Exit 0 when clean, 1 on any FAIL, 2 when the check itself cannot run (no
CHANGELOG.md, an unreadable base, a shallow clone without the merge base).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CHANGELOG = "CHANGELOG.md"
FRAGMENT_DIR = "changelog.d"
#: The one file in FRAGMENT_DIR that is not a fragment: it says what the directory is.
DIR_README = "README.md"
UNRELEASED = "## Unreleased"
#: The whole body of ``## Unreleased``. Changing it is a change to this script.
POINTER = ("Changes merged since the last release are one file each in [`changelog.d/`](changelog.d/); "
           "the release moves them into its section here. To add an entry, add "
           "`changelog.d/<pr-or-slug>.md`; never edit this section by hand.")
#: The release heading check_readme_links.py reads the released version from.
RELEASE_HEADING = re.compile(r"^## (?P<ver>\d+\.\d+\.\d+)\s+[—–-]\s+(?P<date>\d{4}-\d{2}-\d{2})")
FRAGMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")
HEADING = re.compile(r"^(#{1,6})\s")
FENCE = re.compile(r"^\s*(```|~~~)")
#: Never history: the conflict-marker guard refuses them, and removing one is a repair.
CONFLICT_MARKER = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )")


class CheckError(RuntimeError):
    """The check cannot run (exit 2): a missing file, an unreadable ref, no merge base."""


@dataclass(frozen=True)
class Fragment:
    name: str
    text: str

    @property
    def title(self) -> str:
        return self.text.split("\n", 1)[0]


# ----------------------------------------------------------------- parsing --

def headings(text: str) -> list[tuple[int, int, str]]:
    """``(line index, level, line)`` for each Markdown heading outside code fences."""
    out, fenced = [], False
    for i, line in enumerate(text.split("\n")):
        if FENCE.match(line):
            fenced = not fenced
            continue
        m = None if fenced else HEADING.match(line)
        if m:
            out.append((i, len(m.group(1)), line))
    return out


def split_changelog(text: str) -> tuple[str, str, str]:
    """``(head, unreleased_body, released)``: everything up to and including
    the ``## Unreleased`` line, the lines between it and the first release
    heading, and the rest from that heading on (byte-for-byte). A
    ``CheckError`` when either heading is missing or they are out of order."""
    lines = text.split("\n")
    first_release = next((i for i, ln in enumerate(lines) if RELEASE_HEADING.match(ln)), None)
    if first_release is None:
        raise CheckError(f"{CHANGELOG} has no '## X.Y.Z — YYYY-MM-DD' release heading")
    unrel = [i for i, ln in enumerate(lines) if ln.rstrip() == UNRELEASED]
    if len(unrel) != 1 or unrel[0] > first_release:
        raise CheckError(f"{CHANGELOG} needs exactly one '{UNRELEASED}' heading, above the first release heading "
                         f"(found {len(unrel)}{', below it' if unrel and unrel[0] > first_release else ''})")
    u = unrel[0]
    return ("\n".join(lines[:u + 1]), "\n".join(lines[u + 1:first_release]), "\n".join(lines[first_release:]))


def released_lines(text: str) -> list[str]:
    """The released sections' lines, from the first release heading to the end
    (none when there is no release heading: a file that lost them all)."""
    lines = text.split("\n")
    first = next((i for i, ln in enumerate(lines) if RELEASE_HEADING.match(ln)), len(lines))
    return lines[first:]


def lost_lines(old: list[str], new: list[str]) -> list[tuple[int, str]]:
    """Lines of ``old`` that do not survive, in order, in ``new``, as ``(index
    in old, line)``; empty exactly when ``old`` (conflict-marker lines aside) is
    a subsequence of ``new``. Additions anywhere are allowed; an edited or moved
    line is lost from where it was. The verdict is the exact two-pointer test;
    difflib only names the lost lines once the verdict is a loss."""
    keep = [(i, ln) for i, ln in enumerate(old) if not CONFLICT_MARKER.match(ln)]
    j = 0
    for _i, ln in keep:
        while j < len(new) and new[j] != ln:
            j += 1
        if j == len(new):
            break
        j += 1
    else:
        return []
    import difflib
    sm = difflib.SequenceMatcher(None, [ln for _i, ln in keep], new, autojunk=False)
    kept = {a + k for a, _b, size in sm.get_matching_blocks() for k in range(size)}
    return [keep[k] for k in range(len(keep)) if k not in kept]


# --------------------------------------------------------------- fragments --

def fragment_problems(name: str, text: str) -> list[str]:
    """Why ``changelog.d/<name>`` is not a valid fragment (empty when it is)."""
    probs = []
    if not FRAGMENT_NAME.match(name):
        probs.append("the name must match [A-Za-z0-9][A-Za-z0-9._-]*.md (e.g. 1150.md or tc1-amendment-40-read.md)")
    if not text.strip():
        return probs + ["it is empty"]
    if not text.startswith("### "):
        probs.append("its first line must be its '### Title' heading")
    hs = headings(text)
    titles = [h for h in hs if h[1] == 3]
    if len(titles) != 1:
        probs.append(f"it has {len(titles)} '### ' headings; one change is one fragment (split it, or make the "
                     "inner ones '####')")
    top = [h for h in hs if h[1] < 3]
    if top:
        probs.append(f"line {top[0][0] + 1} is a '{'#' * top[0][1]} ' heading; a fragment sits inside a release "
                     "section, so it may use '### ' once and '####' or deeper below it")
    return probs


def fragment_files(root: Path) -> tuple[list[Path], list[str]]:
    """``(fragment paths, problems)`` for everything in ``changelog.d/`` other than its README."""
    d = root / FRAGMENT_DIR
    if not d.is_dir():
        return [], []
    paths, probs = [], []
    for p in sorted(d.iterdir()):
        if p.name == DIR_README:
            continue
        if not p.is_file():
            probs.append(f"{FRAGMENT_DIR}/{p.name}: not a file; {FRAGMENT_DIR}/ holds one Markdown file per change")
        elif p.suffix != ".md":
            probs.append(f"{FRAGMENT_DIR}/{p.name}: not a .md fragment")
        else:
            paths.append(p)
    return paths, probs


def read_fragments(root: Path) -> tuple[list[Fragment], list[str]]:
    paths, probs = fragment_files(root)
    frags = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            probs.append(f"{FRAGMENT_DIR}/{p.name}: not UTF-8")
            continue
        probs += [f"{FRAGMENT_DIR}/{p.name}: {m}" for m in fragment_problems(p.name, text)]
        frags.append(Fragment(p.name, text.rstrip("\n").replace("\r\n", "\n")))
    return frags, probs


# --------------------------------------------------------------------- git --

def _git(root: Path, *args: str, ok=(0,)) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode not in ok:
        raise CheckError(f"git {' '.join(args)}: {p.stderr.strip() or 'exit ' + str(p.returncode)}")
    return p.stdout


def merge_order(root: Path) -> dict[str, int]:
    """``{fragment name: position}`` of the first-parent commit (of HEAD) that
    last added each file under changelog.d/, 0 = newest. Names never added in
    that history are absent (not committed yet)."""
    try:
        out = _git(root, "log", "--first-parent", "--diff-merges=first-parent", "--diff-filter=A",
                   "--name-only", "--format=%x00%H", "HEAD", "--", f"{FRAGMENT_DIR}/")
    except CheckError:
        return {}                                       # no commits yet: everything is uncommitted
    pos: dict[str, int] = {}
    for i, block in enumerate(b for b in out.split("\x00") if b.strip()):
        for path in block.split("\n")[1:]:
            path = path.strip()
            if path.startswith(FRAGMENT_DIR + "/"):
                pos.setdefault(path[len(FRAGMENT_DIR) + 1:], i)
    return pos


def ordered(root: Path, frags: list[Fragment]) -> list[Fragment]:
    """Newest first: by adding commit (uncommitted first), then reverse name."""
    pos = merge_order(root)
    by_name = sorted(frags, key=lambda f: f.name, reverse=True)
    return sorted(by_name, key=lambda f: pos.get(f.name, -1))


def render(frags: list[Fragment]) -> str:
    """The fragments as one block of sections, one blank line between them."""
    return "\n\n".join(f.text for f in frags)


def merge_base(root: Path, base: str) -> str:
    try:
        sha = _git(root, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
        return _git(root, "merge-base", sha, "HEAD").strip()
    except CheckError as e:
        shallow = _git(root, "rev-parse", "--is-shallow-repository", ok=(0, 128)).strip() == "true"
        raise CheckError(f"no merge base of {base} and HEAD: {e}"
                         + (" -- this checkout is shallow (actions/checkout: fetch-depth: 0)" if shallow else "")) from e


def base_file(root: Path, rev: str, rel: str) -> str | None:
    p = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{rel}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def base_fragments(root: Path, rev: str) -> dict[str, str]:
    names = _git(root, "ls-tree", "--name-only", rev, f"{FRAGMENT_DIR}/").split("\n")
    out = {}
    for path in (n.strip() for n in names):
        name = path[len(FRAGMENT_DIR) + 1:] if path.startswith(FRAGMENT_DIR + "/") else ""
        if name.endswith(".md") and name != DIR_README:
            out[name] = (base_file(root, rev, path) or "").replace("\r\n", "\n")
    return out


# ------------------------------------------------------------------- check --

def shape_problems(text: str) -> list[str]:
    """CHANGELOG.md's ``## Unreleased`` body must be exactly POINTER."""
    try:
        _head, body, _rel = split_changelog(text)
    except CheckError as e:
        return [str(e)]
    if body.split() == POINTER.split():                 # rewrapping is not an edit
        return []
    entries = [h for h in headings(body) if h[1] >= 3]
    if entries:
        return [f"{CHANGELOG}: '{UNRELEASED}' holds {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
                f"(first: {entries[0][2][:100]!r}). Move each '### ' section, unchanged, into its own "
                f"{FRAGMENT_DIR}/<pr-or-slug>.md and leave '{UNRELEASED}' as the one pointer paragraph"]
    return [f"{CHANGELOG}: the body of '{UNRELEASED}' is not the pointer paragraph; it is generated by this script, "
            f"and reads:\n\n{POINTER}"]


def history_problems(root: Path, base: str, head_text: str, frags: list[Fragment]) -> list[str]:
    """What the head lost relative to the merge base with ``base``: released
    lines, and unreleased fragments that left without being released."""
    mb = merge_base(root, base)
    old = base_file(root, mb, CHANGELOG)
    if old is None:
        raise CheckError(f"{CHANGELOG} is missing at the merge base {mb[:12]}")
    probs = []
    gone = lost_lines(released_lines(old), released_lines(head_text))
    if gone:
        shown = "\n".join(f"    base released line {i + 1}: {ln[:110]!r}" for i, ln in gone[:8])
        probs.append(f"{CHANGELOG}: {len(gone)} line(s) of the released sections at the merge base {mb[:12]} are gone "
                     f"or changed (released history is append-only; a correction is a new note, not an edit):\n"
                     f"{shown}" + (f"\n    ... and {len(gone) - 8} more" if len(gone) > 8 else ""))
    # A fragment leaves by being released: its '### ' heading appears in the
    # released sections more often than at the merge base. One that was renamed
    # keeps its heading in another fragment.
    released_gain = Counter(released_lines(head_text)) - Counter(released_lines(old))
    names, titles = {f.name for f in frags}, {f.title for f in frags}
    dropped = [n for n, t in sorted(base_fragments(root, mb).items())
               if n not in names and t.split("\n", 1)[0] not in titles and not released_gain[t.split("\n", 1)[0]]]
    if dropped:
        probs.append(f"{FRAGMENT_DIR}/: {len(dropped)} fragment(s) present at the merge base {mb[:12]} are gone and "
                     f"their '### ' heading is in no release section: {dropped[:8]}. An unreleased entry leaves "
                     "only by being released (--release)")
    return probs


def check(root: Path, base: str | None, allow_history_edit: bool) -> int:
    path = root / CHANGELOG
    if not path.is_file():
        print(f"FAIL: {CHANGELOG} is missing")
        return 2
    text = path.read_text(encoding="utf-8")
    try:
        frags, probs = read_fragments(root)
        probs = shape_problems(text) + probs
        history = history_problems(root, base, text, frags) if base else []
    except CheckError as e:
        print(f"FAIL: {e}")
        return 2
    for p in probs:
        print(f"FAIL: {p}")
    for p in history:
        print(f"{'NOTE (--allow-history-edit)' if allow_history_edit else 'FAIL'}: {p}")
    failed = len(probs) + (0 if allow_history_edit else len(history))
    if failed:
        print(f"FAIL: {failed} problem(s). Add changelog.d/<pr-or-slug>.md; never edit {CHANGELOG}'s "
              f"'{UNRELEASED}' by hand")
        return 1
    scope = f"history against the merge base with {base}" if base else "no --base: history not compared"
    print(f"OK: {len(frags)} fragment(s) in {FRAGMENT_DIR}/; '{UNRELEASED}' is the pointer; {scope}")
    return 0


# ----------------------------------------------------------------- release --

def release(root: Path, heading: str, intro: str | None) -> int:
    """Write the fragments into a new ``heading`` section below ``## Unreleased``,
    then delete them. The released part of the file is kept byte-for-byte."""
    m = RELEASE_HEADING.match(heading)
    if not m or "\n" in heading:
        print(f"FAIL: --release takes one '## X.Y.Z — YYYY-MM-DD — title' heading line, got {heading!r}")
        return 2
    path = root / CHANGELOG
    text = path.read_text(encoding="utf-8")
    try:
        head, _body, released = split_changelog(text)
    except CheckError as e:
        print(f"FAIL: {e}")
        return 2
    if any((h := RELEASE_HEADING.match(ln)) and h.group("ver") == m.group("ver") for ln in released.split("\n")):
        print(f"FAIL: {CHANGELOG} already has a {m.group('ver')} section")
        return 1
    frags, probs = read_fragments(root)
    probs = shape_problems(text) + probs
    if not frags:
        probs.append(f"{FRAGMENT_DIR}/ holds no fragment: nothing to release")
    if probs:
        for p in probs:
            print(f"FAIL: {p}")
        return 1
    frags = ordered(root, frags)
    parts = [heading.rstrip()] + ([intro.strip("\n")] if intro and intro.strip() else []) + [render(frags)]
    new = f"{head}\n\n{POINTER}\n\n" + "\n\n".join(parts) + "\n\n" + released
    if not new.endswith(released) or lost_lines(released_lines(text), released_lines(new)):
        print(f"FAIL: the assembled {CHANGELOG} would not keep the released sections byte-for-byte; nothing written")
        return 2
    path.write_text(new, encoding="utf-8")
    for f in frags:
        (root / FRAGMENT_DIR / f.name).unlink()
    print(f"OK: {len(frags)} fragment(s) written under {heading[:80]!r} and deleted, newest first:")
    for f in frags:
        print(f"  {FRAGMENT_DIR}/{f.name}: {f.title[:100]}")
    print(f"next: git add -A {FRAGMENT_DIR} {CHANGELOG}; then scripts/check_readme_claims.py --write-release-block")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="the repository root (default: .)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate the fragments and CHANGELOG.md")
    mode.add_argument("--render", action="store_true", help="print the Unreleased section as the fragments make it")
    mode.add_argument("--release", metavar="HEADING", help="write a release section from the fragments, delete them")
    ap.add_argument("--base", default=None, metavar="REF",
                    help="--check: also compare history against the merge base of REF and HEAD (empty: skipped)")
    ap.add_argument("--allow-history-edit", action="store_true",
                    help="--check: report history losses as NOTEs (CI: the changelog-history-edit label)")
    ap.add_argument("--intro-file", default=None, help="--release: the release's opening paragraphs")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.check:
        return check(root, a.base or None, a.allow_history_edit)
    if a.render:
        frags, probs = read_fragments(root)
        for p in probs:
            print(f"FAIL: {p}", file=sys.stderr)
        print(f"{UNRELEASED}\n\n" + (render(ordered(root, frags)) if frags else "(no fragments)"))
        return 1 if probs else 0
    intro = Path(a.intro_file).read_text(encoding="utf-8") if a.intro_file else None
    return release(root, a.release, intro)


if __name__ == "__main__":
    sys.exit(main())
