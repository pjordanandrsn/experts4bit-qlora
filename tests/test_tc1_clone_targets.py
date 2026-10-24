"""No source checkout lands inside a tc1 run's work dir ($W).

The fetch copies $W into the run's receipt dir, and a checkout's .git inside it once wedged the receipt store (a pre-check's clone under
$W). So every `git clone` target, every `git -C DIR checkout` and every `git worktree add` path in tc1_run.sh must resolve outside $W.

How a target is judged:
- every quote character is removed first;
- each variable expands to its nearest assignment above the command, recursively (an alias such as `WD=$W; SRC=$WD/x` resolves);
- `${NAME:-X}` and `${NAME-X}` are judged both ways, as X and as NAME's own assignment when the script has one;
- `$(dirname "$W")` is $W's parent, outside it;
- a relative target counts as inside $W, since the runner runs from it (`cd $W` near its top);
- a target that cannot be resolved (an unassigned variable, another command substitution) fails the test.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LINES = (REPO / "bench" / "tc1" / "tc1_run.sh").read_text().splitlines()
W, PARENT = "<W>", "<PARENT_OF_W>"
VAR = re.compile(r"\$\{(\w+)(:?-)([^}]*)\}|\$\{(\w+)\}|\$(\w+)")


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


def _assignment(lines: list[str], name: str, at: int):
    """(the value, its line) of NAME's nearest assignment above line `at`, or None."""
    for k in range(at - 1, -1, -1):
        a = re.search(rf"(?:^|[\s;]){name}=", lines[k])
        if a and not lines[k].lstrip().startswith("#"):
            return _word(lines[k], a.end()), k
    return None


def _expand(lines: list[str], word: str, at: int, depth: int = 0) -> set[str]:
    """Every value the word can take, with $W kept as <W> and $(dirname "$W") as <PARENT_OF_W>."""
    assert depth < 12, f"tc1_run.sh:{at + 1}: {word} does not resolve within 12 expansions"
    w = word.replace('"', "").replace("'", "")
    w = re.sub(r"\$\{W\}|\$W(?!\w)", W, w)
    w = w.replace(f"$(dirname {W})", PARENT)
    m = VAR.search(w)
    if not m:
        assert "$(" not in w and "`" not in w, f"tc1_run.sh:{at + 1}: cannot judge the command substitution in {word}"
        return {w}
    pre, post = w[:m.start()], w[m.end():]
    if m.group(1):                                        # ${NAME:-X} or ${NAME-X}: the default, and NAME's own assignment
        alts = _expand(lines, m.group(3), at, depth + 1)
        own = _assignment(lines, m.group(1), at)
        if own:
            alts |= _expand(lines, own[0], own[1], depth + 1)
    else:
        name = m.group(4) or m.group(5)
        own = _assignment(lines, name, at)
        assert own, f"tc1_run.sh:{at + 1}: no assignment of ${name} above the checkout ({word})"
        alts = _expand(lines, own[0], own[1], depth + 1)
    return {r for alt in alts for r in _expand(lines, pre + alt + post, at, depth + 1)}


def _inside(value: str) -> bool:
    """Inside $W: under it, or relative (the runner runs from $W)."""
    return value == W or value.startswith(W + "/") or not value.startswith(("/", "~", PARENT))


def _checkouts(lines: list[str]) -> list[tuple[int, str]]:
    """(line index, target as written) for each clone, each checkout run in another dir, and each worktree add."""
    out = []
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"git clone\b", ln):
            cmd = re.split(r"&&|\|\||;|>|\|", ln[m.end():])[0]
            out.append((i, shlex.split(cmd, posix=False)[-1]))
        for m in re.finditer(r"git -C (\S+) checkout\b", ln):
            out.append((i, m.group(1)))
        for m in re.finditer(r"git(?: -C \S+)? worktree add\b", ln):
            args = shlex.split(re.split(r"&&|\|\||;|>|\|", ln[m.end():])[0], posix=False)
            pos, k = [], 0
            while k < len(args):
                if args[k] in ("-b", "-B", "--reason"):
                    k += 2
                    continue
                if not args[k].startswith("-"):
                    pos.append(args[k])
                k += 1
            out.append((i, pos[0]))
    return out


def _inside_targets(lines: list[str]) -> list[str]:
    bad = []
    for i, t in _checkouts(lines):
        for v in sorted(_expand(lines, t, i)):
            if _inside(v):
                bad.append(f"tc1_run.sh:{i + 1}: {t} -> {v}")
    return bad


def test_every_clone_checkout_and_worktree_in_tc1_run_resolves_outside_the_work_dir():
    found = _checkouts(LINES)
    assert len(found) >= 4, found                      # the decoded-route gate and amendment 76's pre-check, a clone and a checkout each
    assert any(re.search(r"(?:^|;)\s*cd \$W\b", ln) for ln in LINES[:80])   # why a relative target counts as inside
    bad = _inside_targets(LINES)
    assert not bad, "a checkout inside $W reaches the fetched run dir:\n" + "\n".join(bad)


@pytest.mark.parametrize("lines", [
    ['  local SRC=$W/gnf4-src', '  git clone -q https://example.invalid/x.git $SRC > log 2>&1'],
    ['  local PC=${POS76_PRECHECK_DIR:-$W/gnf4-pos76}', '  git clone -q https://example.invalid/x.git "$PC"'],     # the default
    ['  local PC=${POS76_PRECHECK_DIR-$W/gnf4-pos76}', '  git clone -q https://example.invalid/x.git "$PC"'],
    ['POS76_PRECHECK_DIR=$W/x', '  local PC=${POS76_PRECHECK_DIR:-$(dirname "$W")/y}', '  git clone -q URL "$PC"'],  # NAME's own
    ['  local SRC="$W"/gnf4-decgate', '  git clone -q https://example.invalid/x.git "$SRC"'],                        # inner quotes
    ['  git clone -q https://example.invalid/x.git gnf4-decgate'],                                                  # relative
    ['  git clone -q https://example.invalid/x.git ./gnf4-decgate'],
    ['  local WD=$W', '  local SRC=$WD/x', '  git clone -q https://example.invalid/x.git "$SRC"'],                  # an alias
    ['  git -C "$W"/x checkout -q abc'],
    ['  git worktree add "$W"/wt'],
    ['  git -C /repo worktree add -b br --detach $W/wt2 HEAD'],
])
def test_the_resolver_catches_a_checkout_under_the_work_dir(lines):
    assert _inside_targets(lines), lines


@pytest.mark.parametrize("lines", [
    ['  local PC=${POS76_PRECHECK_DIR:-$(dirname "$W")/gnf4-pos76}', '  git clone -q https://example.invalid/x.git "$PC"'],
    ['  local SRC=/opt/gnf4', '  git clone -q URL $SRC', '  git -C $SRC checkout -q abc'],
    ['  git worktree add -b br /tmp/wt HEAD'],
])
def test_the_resolver_passes_a_checkout_beside_the_work_dir(lines):
    assert _checkouts(lines) and not _inside_targets(lines), lines


def test_an_unresolvable_target_fails():
    with pytest.raises(AssertionError, match="no assignment"):
        _inside_targets(['  git clone -q URL "$NOPE/x"'])
    with pytest.raises(AssertionError, match="command substitution"):
        _inside_targets(['  git clone -q URL "$(mktemp -d)"'])
