"""No source checkout lands inside a tc1 run's work dir ($W).

The fetch copies $W into the run's receipt dir, and a checkout's .git inside it once wedged the receipt store (a pre-check's clone under
$W). So every `git clone` target and every `git -C DIR checkout` in tc1_run.sh must resolve outside $W. A target held in a variable is
resolved from its nearest assignment above the command; one that cannot be resolved fails the test.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINES = (REPO / "bench" / "tc1" / "tc1_run.sh").read_text().splitlines()


def _word(s: str, i: int) -> str:
    """The shell word starting at s[i], keeping ${...}, $(...) and quotes whole."""
    depth, quote, j = 0, None, i
    while j < len(s):
        c = s[j]
        if quote:
            quote = None if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c in "({":
            depth += 1
        elif c in ")}":
            if depth == 0:
                break
            depth -= 1
        elif c.isspace() and depth == 0:
            break
        j += 1
    return s[i:j]


def _resolve(target: str, at: int) -> str:
    """A $NAME or ${NAME} target (plus any suffix) with NAME's nearest assignment above line `at` substituted."""
    m = re.fullmatch(r'"?\$\{?(\w+)\}?(.*?)"?', target)
    if not m or m.group(1) == "W":
        return target
    for ln in reversed(LINES[:at]):
        a = re.search(rf"(?:^|[\s;]){m.group(1)}=", ln)
        if a:
            return _word(ln, a.end()).strip('"') + m.group(2)
    raise AssertionError(f"tc1_run.sh:{at + 1}: no assignment of ${m.group(1)} above the checkout (target {target})")


def _checkouts() -> list[tuple[int, str]]:
    """(line index, target as written) for each clone and each checkout run in another dir."""
    out = []
    for i, ln in enumerate(LINES):
        if ln.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"git clone\b", ln):
            cmd = re.split(r"&&|\|\||;|>|\|", ln[m.end():])[0]
            out.append((i, shlex.split(cmd, posix=False)[-1]))
        for m in re.finditer(r"git -C (\S+) checkout\b", ln):
            out.append((i, m.group(1)))
    return out


def test_every_clone_and_checkout_in_tc1_run_resolves_outside_the_work_dir():
    found = _checkouts()
    assert len(found) >= 4, found                      # the decoded-route gate and amendment 76's pre-check, a clone and a checkout each
    inside = []
    for i, t in found:
        r = _resolve(t, i).strip('"')
        if re.match(r"\$\{?W\}?(/|$)", r):
            inside.append(f"tc1_run.sh:{i + 1}: {t} -> {r}")
    assert not inside, "a checkout inside $W reaches the fetched run dir:\n" + "\n".join(inside)


def test_the_resolver_catches_a_clone_under_the_work_dir():
    LINES.extend(['  local SRC=$W/gnf4-src', '  git clone -q https://example.invalid/x.git $SRC > log 2>&1'])
    try:
        found = _checkouts()[-1]
        assert _resolve(found[1], found[0]).startswith("$W/"), found
    finally:
        del LINES[-2:]
