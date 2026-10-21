"""Every public ``Re-derive:`` line still reproduces its committed verdict, byte for byte, from the committed files.

A receipt that a reader can re-check carries one line (CONTRIBUTING.md, "Re-derive lines")::

    Re-derive: `python bench/pNN/pNN_reduce.py --dir bench/pNN/receipts/<run> --out {out} ...` writes `bench/.../verdict.json`

A new receipt may carry the line in its README.md at its first commit; a committed receipt is never edited, so the line
for one goes in ``bench/<lane>/REDERIVE.md``. This test runs each command from the repository root with ``{out}`` replaced by a scratch file and
asserts the output equals the committed file. A later edit to a reducer that stops an old verdict reproducing fails here.

Forms the command may take, and nothing else (it runs without a shell):
- ``python <script> <args>``: the script is a ``bench/**/*_reduce.py`` path;
- ``<script>@<commit>``: the reducer as it was at that commit (``git show``), for a verdict an amendment's reducer no
  longer writes;
- a leading ``PYTHONPATH=bench/<dir>`` token, for a reducer that imports a sibling staged next to it on the box.

A ``Re-derive [torch]:`` line is skipped where torch does not import; every line today is stdlib-only and runs on CPU.

On Windows the comparison ignores carriage returns: a reducer that writes in text mode emits CRLF there, and a checkout
with ``core.autocrlf`` may rewrite the committed file. On Linux (CI) the bytes must be identical.
"""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"Re-derive(?P<torch> \[torch\])?: `(?P<cmd>[^`]+)` writes `(?P<target>[^`]+)`")
SCRIPT = re.compile(r"^(?P<path>bench/[\w./-]+_reduce\.py)(?:@(?P<commit>[0-9a-f]{7,40}))?$")
ENV = re.compile(r"^PYTHONPATH=(?P<path>bench/[\w./-]+)$")


def tracked_markdown(root: Path) -> list:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "bench"], capture_output=True, check=True).stdout
    return sorted(p for p in out.decode("utf-8").split("\0") if p.endswith(".md"))


def rederive_lines(root: Path) -> list:
    """[(where, needs_torch, command, target)] for every ``Re-derive:`` line in a tracked bench/**/*.md file."""
    found = []
    for rel in tracked_markdown(root):
        for n, ln in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
            m = LINE.search(ln)
            if m:
                found.append((f"{rel}:{n}", bool(m.group("torch")), m.group("cmd"), m.group("target")))
    return found


def parse(cmd: str, target: str) -> tuple:
    """(env, script path, commit or None, args) for one command, or ValueError naming what the convention refuses."""
    tok = shlex.split(cmd)
    env = {}
    while tok and "=" in tok[0] and not tok[0].startswith("-"):
        m = ENV.match(tok.pop(0))
        if not m:
            raise ValueError("only a PYTHONPATH=bench/<dir> prefix is allowed")
        env["PYTHONPATH"] = m.group("path")
    if len(tok) < 2 or tok[0] not in ("python", "python3"):
        raise ValueError("the command must be `python <bench/**/*_reduce.py> ...`")
    m = SCRIPT.match(tok[1])
    if not m:
        raise ValueError(f"{tok[1]} is not a bench/**/*_reduce.py path")
    args = tok[2:]
    if args.count("{out}") != 1:
        raise ValueError("the command must name {out} exactly once")
    if not target.startswith("bench/") or ".." in target.split("/"):
        raise ValueError(f"{target} is not a path under bench/")
    return env, m.group("path"), m.group("commit"), args


def rederive(root: Path, cmd: str, target: str, scratch: Path) -> bytes:
    """Run one line's command from ``root`` and return the bytes it wrote."""
    env, script, commit, args = parse(cmd, target)
    if commit:
        old = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{script}"], check=False,
                             capture_output=True)
        if old.returncode != 0:
            raise RuntimeError(f"git show {commit}:{script} failed (a shallow checkout needs fetch-depth: 0)")
        pinned = scratch / f"{Path(script).stem}_{commit[:12]}.py"
        pinned.write_bytes(old.stdout)
        script = str(pinned)
    out = scratch / "rederived.json"
    run_env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    if "PYTHONPATH" in env:
        run_env["PYTHONPATH"] = str(root / env["PYTHONPATH"])
    r = subprocess.run([sys.executable, script, *[str(out) if a == "{out}" else a for a in args]],
                       check=False, cwd=root, env=run_env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"rc {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")
    return out.read_bytes()


def same(got: bytes, want: bytes) -> bool:
    if os.name == "nt":
        got, want = got.replace(b"\r\n", b"\n"), want.replace(b"\r\n", b"\n")
    return got == want


LINES = rederive_lines(ROOT)


@pytest.mark.parametrize("where,needs_torch,cmd,target", LINES, ids=[w for w, *_ in LINES])
def test_every_rederive_line_reproduces_its_committed_verdict(where, needs_torch, cmd, target, tmp_path):
    if needs_torch:
        pytest.importorskip("torch")
    got = rederive(ROOT, cmd, target, tmp_path)
    assert same(got, (ROOT / target).read_bytes()), f"{where}: the output differs from {target}"


def late_receipt_lines(root: Path, lines: list) -> list:
    """Every ``Re-derive:`` line inside ``bench/**/receipts/**`` that its file did not carry at the file's first commit:
    a committed receipt is never edited. (On a shallow checkout the oldest visible commit stands in for the first, so
    this can miss an edit there; it never flags a line the first commit had.)"""
    late = []
    for where, _torch, cmd, target in lines:
        rel = where.rsplit(":", 1)[0]
        if "receipts" not in Path(rel).parts:
            continue
        log = subprocess.run(["git", "-C", str(root), "log", "--diff-filter=A", "--format=%H", "--", rel],
                             check=False, capture_output=True, text=True).stdout.split()
        first = subprocess.run(["git", "-C", str(root), "show", f"{log[-1]}:{rel}"], check=False,
                               capture_output=True).stdout.decode("utf-8", "replace") if log else ""
        if f"`{cmd}` writes `{target}`" not in first:
            late.append(where)
    return late


def test_no_committed_receipt_gains_a_rederive_line():
    assert late_receipt_lines(ROOT, LINES) == []


def test_a_line_added_to_a_committed_receipt_is_flagged(tmp_path):
    root = tmp_path / "repo"
    d = root / "bench" / "pX" / "receipts" / "pX-1"
    d.mkdir(parents=True)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    line = "Re-derive: `python bench/pX/pX_reduce.py --out {out}` writes `bench/pX/receipts/pX-1/v.json`\n"
    (d / "README.md").write_text("# pX-1\n", encoding="utf-8")
    (root / "bench" / "pX" / "REDERIVE.md").write_text(line, encoding="utf-8")
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "receipt"], check=True)
    assert late_receipt_lines(root, rederive_lines(root)) == []          # the lane file is outside receipts/
    (d / "README.md").write_text("# pX-1\n" + line, encoding="utf-8")
    subprocess.run(git + ["commit", "-q", "-am", "edit"], check=True)
    assert late_receipt_lines(root, rederive_lines(root)) == ["bench/pX/receipts/pX-1/README.md:2"]
    e = root / "bench" / "pX" / "receipts" / "pX-2"
    e.mkdir()
    (e / "README.md").write_text("# pX-2\n" + line.replace("pX-1", "pX-2"), encoding="utf-8")
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "a new receipt with its line"], check=True)
    assert late_receipt_lines(root, rederive_lines(root)) == ["bench/pX/receipts/pX-1/README.md:2"]


def test_the_check_reads_the_lines_it_guards():
    assert len(LINES) >= 11                                          # 11 when this check was written
    targets = [t for *_, t in LINES]
    assert len(set(targets)) == len(targets), "two Re-derive lines name the same committed file"
    tracked = set(subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                                 check=True).stdout.splitlines())
    assert [t for t in targets if t not in tracked] == []


def test_a_one_byte_edit_to_a_committed_verdict_fails(tmp_path):
    """The mutant: a real line's output against its committed verdict with one byte changed must not compare equal."""
    _, _, cmd, target = next(x for x in LINES if "p160" in x[0])
    got = rederive(ROOT, cmd, target, tmp_path)
    want = (ROOT / target).read_bytes()
    assert same(got, want)
    i = want.index(b"1")
    assert not same(got, want[:i] + b"2" + want[i + 1:])


@pytest.mark.parametrize("cmd,target,why", [
    ("bash -c 'cat x' {out}", "bench/x.json", "must be `python"),
    ("python tools/x_reduce.py --out {out}", "bench/x.json", "not a bench/**/*_reduce.py"),
    ("python bench/p1/p1_reduce.py --out v.json", "bench/x.json", "{out} exactly once"),
    ("python bench/p1/p1_reduce.py --out {out}", "docs/x.json", "not a path under bench/"),
    ("LD_PRELOAD=x.so python bench/p1/p1_reduce.py --out {out}", "bench/x.json", "only a PYTHONPATH"),
])
def test_the_convention_refuses_other_commands(cmd, target, why):
    with pytest.raises(ValueError, match=re.escape(why)):
        parse(cmd, target)
