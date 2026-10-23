"""bench/common rule 8: a fetch never imports links (bench/common/README.md, "Fetch").

A rented box is a third-party host, and `rsync -a` (or any -l) copies a symlink AS a symlink. A box could leave a
relative link in its work dir that resolves, on the controller, to a file outside the run dir; a receipt commit that
followed it would push that file. The store's commit refuses links now; rule 8 is the second layer: the fetch itself
never brings one back (`--no-links` after -a, or flags without -a/-l, which make rsync skip links).

Closed drivers are not swept. Every shell script that keeps links today is frozen by its sha256 in
fixtures/rule8_legacy_fetch.json. A NEW script, or a frozen one whose bytes CHANGE, must pass. So a driver carries the
rule from its next edit, and a closed reading's driver stays exactly as it ran.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(__file__).parent / "fixtures" / "rule8_legacy_fetch.json"
SKIP_DIRS = {".git", "build", ".venv", "node_modules"}
# a word in command position: the start, after ; & | ( { ! or $( or a backtick, or after a keyword or wrapper word
_POS = re.compile(r"(?:^|[;&|({!`]|\$\(|\b(?:if|then|do|else|elif|while|until|nohup|exec|time|command|sudo)\s)\s*$")


def _mask_quotes(s: str) -> str:
    """The same string with every quoted span's contents replaced by 'x' (offsets kept), so a word inside quotes is
    never read as a command: say "rsync failed", --rsync-path="nice rsync"."""
    out, q = [], None
    for ch in s:
        if q:
            out.append(ch if ch == q else "x")
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _logical_lines(text: str) -> list[str]:
    """Comment-stripped lines, with backslash continuations joined."""
    lines, buf = [], ""
    for raw in text.splitlines():
        masked = _mask_quotes(raw)
        cut = re.search(r"(?:^|\s)#", masked)
        line = raw[:cut.start()] if cut else raw
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        lines.append(buf + line)
        buf = ""
    if buf:
        lines.append(buf)
    return lines


def rsync_calls(text: str) -> list[str]:
    """Each rsync invocation's command text, from `rsync` to the end of its simple command."""
    calls = []
    for line in _logical_lines(text):
        masked = _mask_quotes(line)
        for m in re.finditer(r"\brsync\b", masked):
            if not _POS.search(masked[:m.start()]):
                continue
            end = re.search(r"&&|\|\||[;|)&]", masked[m.end():])
            calls.append(line[m.start():m.end() + (end.start() if end else len(masked) - m.end())])
    return calls


def keeps_links(call: str) -> bool:
    """Whether this rsync copies symlinks as links: -a or -l (or --archive, --links) not undone by a LATER --no-links."""
    try:
        words = shlex.split(call, posix=True)
    except ValueError:
        words = call.split()
    links = False
    for w in words[1:]:
        if w in ("--no-links", "--no-l"):
            links = False
        elif w in ("--archive", "--links"):
            links = True
        elif re.fullmatch(r"-[A-Za-z0-9]+", w) and ("a" in w or "l" in w):
            links = True
    return links


def offending(text: str) -> list[str]:
    return [c for c in rsync_calls(text) if keeps_links(c)]


def _scripts() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.sh") if not SKIP_DIRS & set(p.relative_to(ROOT).parts))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_no_new_or_changed_script_imports_links_on_a_fetch():
    legacy = json.loads(LEGACY.read_text())
    bad = {}
    for p in _scripts():
        rel = p.relative_to(ROOT).as_posix()
        hits = offending(p.read_text(errors="replace"))
        if hits and legacy.get(rel) != _sha(p):
            bad[rel] = hits[0][:120]
    assert not bad, ("rule 8: these rsync calls keep links; add --no-links after -a (or fetch without -a/-l). A frozen "
                     f"script that changed carries the rule from that edit, and its fixture entry goes: {bad}")


def test_the_frozen_list_holds_only_unchanged_scripts_that_still_keep_links():
    legacy = json.loads(LEGACY.read_text())
    assert legacy == dict(sorted(legacy.items()))
    stale = [rel for rel, sha in legacy.items()
             if not (ROOT / rel).is_file() or _sha(ROOT / rel) != sha or not offending((ROOT / rel).read_text(errors="replace"))]
    assert not stale, f"remove these fixture entries (the script changed, was fixed, or is gone): {stale}"


def test_the_matcher_finds_every_command_form_and_reads_the_flags_in_order():
    keep = [
        'rsync -az -e "ssh -p $PORT" "root@$HOST:$W/" "$RUN_DIR/"',
        'if ! rsync -az -e "ssh" "${EXCL[@]}" "root@$HOST:$W/" "$D/"; then exit 22; fi',     # the tc1 form
        'if rsync -az root@h:/w/ d/; then :; fi',
        'box_fetch(){ rsync -az --exclude x root@h:/w/ d/; }',
        'nohup rsync -a --partial root@h:/w/ d/ &',
        'X=$(rsync -a root@h:/w/ d/)',
        'pull_box() { local -a low=(); rsync -az "${low[@]}" root@h:/w/ d/ || exit 22; }',
        'rsync -rltz root@h:/w/ d/',
        'rsync --no-links -a root@h:/w/ d/',                                                  # -a after re-enables links
        'rsync -az \\\n  --exclude x \\\n  root@h:/w/ d/',
    ]
    for text in keep:
        assert offending(text), text
    safe = [
        'rsync -az --no-links -e "ssh -p $PORT" "root@$HOST:$W/" "$RUN_DIR/"',
        'rsync -rtz root@h:/w/ d/',
        'rsync -az \\\n  --no-links \\\n  root@h:/w/ d/',
        'if ! rsync -az --no-links root@h:/w/ d/; then exit 22; fi',
        'say "fetch failed: rsync"; exit 22',
        'command -v rsync >/dev/null || apt-get install -y rsync',
        'for t in git cmake rsync; do command -v $t; done',
        'rsync -az --no-links --rsync-path="nice -n 19 rsync -a" root@h:/w/ d/',
        '# rsync -az root@h:/w/ d/',
        'echo "DRYRUN fetch: rsync -az root@h:/w/ d/"',
    ]
    for text in safe:
        assert not offending(text), text


def test_every_rsync_word_in_a_bench_script_is_a_call_or_a_known_mention():
    """The matcher's reach on the real tree: a line whose first word is rsync is always seen as a call."""
    missed = []
    for p in _scripts():
        for line in _logical_lines(p.read_text(errors="replace")):
            if re.match(r"\s*rsync\s", line) and not rsync_calls(line):
                missed.append(f"{p.relative_to(ROOT).as_posix()}: {line.strip()[:80]}")
    assert not missed, missed
