"""Every lane driver hands the launcher the box's own exit code, and never reports success the box did not record.

The launcher writes ``command exited {rc}`` from the DRIVER's exit code into the receipt, and its
host-refusal classifier reads only that line: ``command exited (?:13|14|17|18)`` excludes the machine from later pins. So a driver
tail that maps every non-zero box code to one code of its own (``exit 23``) hides a box that refused for its disk (13), egress (14),
power cap (17) or host floor (18), and the next pin can buy the same machine again (e4b#2135's driver review). A tail that exits
``$RC`` when the success marker is missing exits 0 on a box rc 0 without the marker: a success the box never recorded (e4b#2137's
G3 kit). This test runs every driver's fetch-and-exit tail on synthetic fetched directories and asserts:

- (a) the host codes 13, 14, 17 and 18 exit unchanged, whatever the lane registers: they are the launcher's codes, the same in
  every lane (bench/common/drive_tail.sh's drive_finish always passes them through; a runner must emit them only for a genuine
  host fault, which is each lane's kit test);
- (b) rc 0 without the success marker never exits 0;
- (c) a missing exit file never exits 0;

plus a control: rc 0 WITH the marker exits 0. A tail the harness cannot drive (no fetched exit file, or a control that does not exit
0) fails as ``undrivable``. The tail is, in order of preference: the whole driver, when it exposes ``<LANE>_DRIVE_TAIL_DIR`` /
``<LANE>_DRIVE_TAIL_NONCE``; its ``drive_finish ...`` line (bench/common/drive_tail.sh, the shared tail every live lane can source);
or its own text from its ``RC=$(cat "...EXIT_CODE...")`` line to the end, run with ``RUN_DIR``, ``OUT`` and ``NONCE`` set. It runs
with only read-only tools on PATH (no rm, find, sed, awk, python3, ssh, scp, rsync or curl) in a scratch directory.

A WRAPPER reads no fetched exit file of its own: it delegates the run to another driver and checks that driver's records
(bench/sc2/sc2_qualify_drive.sh around bench/sc1/sc1_drive.sh, e4b#2139). Wrappers are listed by path in the allowlist's
``wrappers``; a listed wrapper must stay undrivable here (a drivable one is held to the rule like any driver), and no script may run
bench/sc1/sc1_drive.sh unless it is a listed wrapper.

Drivers that fail today are listed in tests/fixtures/drive_tail_allowlist.json: each with its sha256, the exact assertions it fails,
and either "closed lane: <reason>" or "live: fix due in the lane's next PR". A NEW driver, or a CHANGED one (its sha256 no longer
matches), is never allowlisted. A listed driver that starts to pass must leave the list, so the list only shrinks.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVERS = sorted((REPO / "bench").rglob("*_drive.sh"))
ALLOWLIST = REPO / "tests" / "fixtures" / "drive_tail_allowlist.json"
BASH = shutil.which("bash") or "/bin/bash"
pytestmark = pytest.mark.skipif(not Path(BASH).exists() or shutil.which("bash") is None, reason="needs bash")

RC_LINE = re.compile(r'RC=\$\(cat\s+"([^"]*_EXIT_CODE[^"]*)"')
SUCCESS = re.compile(r'"([^"]*_SUCCESS\.\$NONCE[^"]*)"')
TAIL_HOOK = re.compile(r"\b([A-Z0-9]+)_DRIVE_TAIL_DIR\b")
HOST_CODES = (13, 14, 17, 18)
NONCE = "n0nce"
#: the only programs a tail can run here, all read-only: no rm, no in-place editor or interpreter (find -delete, sed -i, awk
#: system(), python3), nothing that reaches the network. Today's tails use cat, grep, tail and cut beyond shell builtins.
SAFE_TOOLS = ("cat", "printf", "test", "[", "ls", "wc", "tr", "cut", "head", "tail", "grep", "basename", "dirname", "true", "false",
              "echo")
STATUS = re.compile(r"(closed lane: \S.*|live: \S.*)")
DRIVE_FINISH = re.compile(r"^\s*drive_finish\s+(.+)$", re.MULTILINE)
DRIVE_TAIL = REPO / "bench" / "common" / "drive_tail.sh"
SC1_DRIVER = "bench/sc1/sc1_drive.sh"
#: a shell line that RUNS sc1_drive.sh (bash/sh/exec/source/. or the path itself at command position), not one that names it
SC1_CALL = re.compile(r"(?:^|[;&|(]\s*|\b(?:bash|sh|exec|source)\s+|\.\s+)\"?[^\"\s;]*sc1_drive\.sh\b")


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO)) for p in paths]


def _env(tmp: Path) -> dict:
    safe = tmp / "safebin"
    safe.mkdir()
    for t in SAFE_TOOLS:
        src = shutil.which(t)
        if src:
            (safe / t).symlink_to(src)
    return {"PATH": str(safe), "HOME": str(tmp), "SSH": "false", "SCP": "false", "RSYNC": "false"}


def _expand(expr: str, env: dict) -> Path:
    out = subprocess.run([BASH, "-c", f'printf %s "{expr}"'], env=env, capture_output=True, text=True, check=True)
    return Path(out.stdout)


def run_case(driver: Path, tmp: Path, exit_code: int | None, marker: bool):
    """The driver's exit on one synthetic fetched dir (exit file content, or None for absent; the success marker or not).
    Returns None when the driver reads no fetched exit file."""
    text = driver.read_text(errors="replace")
    tmp.mkdir(parents=True)
    env = _env(tmp)
    hook = TAIL_HOOK.search(text)
    if hook:
        lane, d = hook.group(1), tmp / "fetched"
        d.mkdir()
        env.update({f"{lane}_DRIVE_TAIL_DIR": str(d), f"{lane}_DRIVE_TAIL_NONCE": NONCE})
        ec, sm, cmd = d / f"{lane}_EXIT_CODE.{NONCE}", d / f"{lane}_SUCCESS.{NONCE}", [BASH, str(driver)]
    elif DRIVE_FINISH.search(text):
        call = DRIVE_FINISH.search(text).group(1)
        env.update({"RUN_DIR": str(tmp / "run"), "OUT": str(tmp / "run" / "out"), "NONCE": NONCE, "E4B_RENT_RUN_DIR": str(tmp / "run")})
        args = subprocess.run([BASH, "-c", f'set -- {call}; printf "%s\\n" "$1" "$2" "$3"'], env=env, capture_output=True,
                              text=True, check=True).stdout.split("\n")
        ec, sm = Path(args[0]) / f"{args[2]}_EXIT_CODE.{args[1]}", Path(args[0]) / f"{args[2]}_SUCCESS.{args[1]}"
        script = tmp / "tail.sh"
        script.write_text(f'set +e\nsay(){{ :; }}\n. "{DRIVE_TAIL}"\ndrive_finish {call}\nexit 99\n')
        cmd = [BASH, str(script)]
    else:
        lines = text.splitlines()
        idx = next((i for i, ln in enumerate(lines) if RC_LINE.search(ln)), None)
        if idx is None:
            return None
        env.update({"RUN_DIR": str(tmp / "run"), "OUT": str(tmp / "run" / "out"), "NONCE": NONCE, "E4B_RENT_RUN_DIR": str(tmp / "run")})
        ec = _expand(RC_LINE.search(lines[idx]).group(1), env)
        m = SUCCESS.search("\n".join(lines[idx:]))
        sm = _expand(m.group(1), env) if m else None
        script = tmp / "tail.sh"
        script.write_text("set +e\nsay(){ :; }\n" + "\n".join(lines[idx:]) + "\nexit 99\n")
        cmd = [BASH, str(script)]
    ec.parent.mkdir(parents=True, exist_ok=True)
    if exit_code is not None:
        ec.write_text(f"{exit_code}\n")
    if marker and sm is not None:
        sm.parent.mkdir(parents=True, exist_ok=True)
        sm.write_text("")
    return subprocess.run(cmd, check=False, env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
                          cwd=tmp).returncode


def assess(driver: Path, tmp: Path) -> list[str]:
    """The assertions this driver's tail fails, as stable labels (an empty list passes)."""
    control = run_case(driver, tmp / "control", 0, True)
    if control is None:
        return ["undrivable: no fetched exit file"]
    fails = [] if control == 0 else [f"undrivable: rc 0 with the success marker exits {control}"]
    for c in HOST_CODES:
        r = run_case(driver, tmp / f"host{c}", c, False)
        if r != c:
            fails.append(f"(a) box rc {c} -> driver {r}")
    if run_case(driver, tmp / "nomarker", 0, False) == 0:
        fails.append("(b) rc 0 without the success marker exits 0")
    if run_case(driver, tmp / "nofile", None, False) == 0:
        fails.append("(c) a missing exit file exits 0")
    return fails


def _allowlist() -> dict:
    return json.loads(ALLOWLIST.read_text())["drivers"]


def _wrappers() -> dict:
    return json.loads(ALLOWLIST.read_text())["wrappers"]


def test_the_drivers_are_all_here():
    assert len(DRIVERS) >= 150, _ids(DRIVERS)       # a glob that matched nothing would make every test below vacuous


def test_every_allowlist_entry_names_a_driver_and_a_status():
    entries = _allowlist()
    known = set(_ids(DRIVERS))
    for path, e in entries.items():
        assert path in known, f"{path}: allowlisted but no such driver (remove the entry)"
        assert re.fullmatch(r"[0-9a-f]{64}", e.get("sha256") or ""), path
        assert e.get("fails"), f"{path}: an entry lists the assertions it fails"
        assert STATUS.fullmatch(e.get("status") or ""), f"{path}: status {e.get('status')!r}"
    for path, why in _wrappers().items():
        assert path in known and path not in entries and why.strip(), path


def judge(rel: str, fails: list[str], sha: str, entry: dict | None, is_wrapper: bool) -> str | None:
    """The rule over one driver: None when it holds, else why it does not."""
    if is_wrapper:
        return None if fails == ["undrivable: no fetched exit file"] else f"{rel} is drivable now: leave the wrappers list ({fails})"
    if not fails:
        return None if entry is None else f"{rel} now passes: remove its allowlist entry"
    if entry is None:
        return f"{rel} fails {fails} and is not allowlisted (a new or changed driver never is): fix its tail"
    if entry["sha256"] != sha:
        return f"{rel} changed since it was allowlisted, so it is held to the rule: fix its tail ({fails})"
    if entry["fails"] != fails:
        return f"{rel}: fails {fails}, allowlisted for {entry['fails']}"
    return None


@pytest.mark.parametrize("driver", DRIVERS, ids=_ids(DRIVERS))
def test_the_driver_tail_passes_the_box_code_through(driver: Path, tmp_path: Path):
    rel = str(driver.relative_to(REPO))
    why = judge(rel, assess(driver, tmp_path), hashlib.sha256(driver.read_bytes()).hexdigest(), _allowlist().get(rel),
                rel in _wrappers())
    assert why is None, why


F = ["(b) rc 0 without the success marker exits 0"]
E = {"sha256": "a" * 64, "fails": F, "status": "closed lane: x"}


@pytest.mark.parametrize("fails,sha,entry,wrapper,holds", [
    (F, "a" * 64, E, False, True),                     # allowlisted, unchanged, failing exactly as listed
    (F, "b" * 64, E, False, False),                    # CHANGED since it was allowlisted: held to the rule
    ([], "a" * 64, E, False, False),                   # passes now: its entry must leave
    (F, "a" * 64, None, False, False),                 # fails and is not allowlisted (a new driver)
    (F + ["(c) a missing exit file exits 0"], "a" * 64, E, False, False),   # fails more than its entry says
    ([], "a" * 64, None, False, True),                 # passes, no entry
    (["undrivable: no fetched exit file"], "z", None, True, True),           # a wrapper that stays undrivable
    ([], "z", None, True, False),                      # a wrapper that became drivable must leave the list
])
def test_the_rule_itself(fails, sha, entry, wrapper, holds):
    assert (judge("bench/x/x_drive.sh", fails, sha, entry, wrapper) is None) is holds


# -- the harness itself, on hand-written tails: each assertion fires, and a correct tail passes ---------------------------------
GOOD = '''RC=$(cat "$RUN_DIR/x/X_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) exit 23;; esac
if [ "$RC" = 0 ]; then [ -f "$RUN_DIR/x/X_SUCCESS.$NONCE" ] || exit 24; exit 0; fi
exit "$RC"
'''
COLLAPSE = GOOD.replace('exit "$RC"', "exit 23")                                    # P236's shape: every box code becomes 23
NO_MARKER = '''RC=$(cat "$RUN_DIR/x/X_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) exit 23;; esac
[ -f "$RUN_DIR/x/X_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || exit "$RC"
exit 0
'''                                                                                  # the exit "$RC" family: 0 without the marker
MISSING_OK = '''RC=$(cat "$RUN_DIR/x/X_EXIT_CODE.$NONCE" 2>/dev/null)
[ -f "$RUN_DIR/x/X_EXIT_CODE.$NONCE" ] || exit 0
if [ "$RC" = 0 ]; then [ -f "$RUN_DIR/x/X_SUCCESS.$NONCE" ] || exit 24; exit 0; fi
exit "$RC"
'''                                                                                  # a run that left no exit file reads as success


@pytest.mark.parametrize("name,tail,want", [
    ("good", GOOD, []),
    ("collapse", COLLAPSE, [f"(a) box rc {c} -> driver 23" for c in HOST_CODES]),
    ("no marker", NO_MARKER, ["(b) rc 0 without the success marker exits 0"]),
    ("missing file reads 0", MISSING_OK, ["(c) a missing exit file exits 0"]),
    ("no exit file", "exit 0\n", ["undrivable: no fetched exit file"]),
    ("control fails", GOOD.replace("exit 0; fi", "exit 26; fi"),                  # rc 0 WITH the marker does not exit 0
     ["undrivable: rc 0 with the success marker exits 26"]),
])
def test_the_harness_on_known_tails(tmp_path: Path, name, tail, want):
    d = tmp_path / "x_drive.sh"
    d.write_text("#!/bin/bash\n# synthetic driver: everything before the tail is skipped\nssh never-run\n" + tail)
    assert assess(d, tmp_path / "w") == want


def test_a_tail_cannot_delete_or_reach_the_network(tmp_path: Path):
    keep = tmp_path / "keep"
    keep.write_text("x")
    d = tmp_path / "x_drive.sh"
    d.write_text(f'RC=$(cat "$RUN_DIR/x/X_EXIT_CODE.$NONCE")\nrm -rf "{keep}"; ssh host true; rsync a b; exit "$RC"\n')
    assert run_case(d, tmp_path / "w", 13, False) == 13
    assert keep.exists()                                                              # rm is not on the tail's PATH


def test_only_a_listed_wrapper_runs_sc1_drive():
    """sc1_drive.sh is HOLD-pinned and exits rc 0 without its success marker (its last line); e4b#2139's wrapper checks the marker
    for it. A new script that runs it directly would not, so every caller must be a listed wrapper."""
    callers = []
    for p in sorted((REPO / "bench").rglob("*.sh")):
        rel = str(p.relative_to(REPO))
        if rel == SC1_DRIVER:
            continue
        for line in p.read_text(errors="replace").splitlines():
            code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
            if SC1_CALL.search(code) and not re.search(r"\b(?:say|echo|printf)\b", code.split("sc1_drive.sh")[0]):
                callers.append(rel)
                break
    assert "bench/sc2/sc2_qualify_drive.sh" in callers           # the pattern still finds the one known caller
    assert set(callers) <= set(_wrappers()), sorted(set(callers) - set(_wrappers()))


# -- bench/common/drive_tail.sh, driven directly ---------------------------------------------------------------------------------
@pytest.mark.parametrize("codes,content,marker,want", [
    ("", "0", NONCE, 0),
    ("", "0", None, 24),
    ("", "0", "other", 24),                                     # another nonce's marker never counts
    ("", None, None, 23),
    ("", "x1", None, 23),
    ("", "", None, 23),
    ("", "13", None, 13), ("", "14", None, 14), ("", "17", None, 17), ("", "18", None, 18),   # the launcher's codes, always
    ("", "9", None, 26),                                        # an empty list: every other non-zero is 26
    ("9 27", "9", None, 9), ("9 27", "27", None, 27), ("9 27", "41", None, 26),
    ("13 41", "13", None, 13),                                  # a list that also names a host code
])
def test_drive_finish(tmp_path: Path, codes, content, marker, want):
    d = tmp_path / "out"
    d.mkdir()
    if content is not None:
        (d / f"X_EXIT_CODE.{NONCE}").write_text(f"{content}\n")
    if marker is not None:
        (d / f"X_SUCCESS.{marker}").write_text("")
    script = f'. "{DRIVE_TAIL}"\ndrive_finish "{d}" {NONCE} X "{codes}"\nexit 99\n'
    r = subprocess.run([BASH, "-c", script], check=False, capture_output=True, text=True, timeout=30)
    assert r.returncode == want, r.stderr


def test_a_driver_that_sources_drive_tail_passes(tmp_path: Path):
    d = tmp_path / "x_drive.sh"
    d.write_text('#!/bin/bash\nssh never-run\n. "$REPO/bench/common/drive_tail.sh"\ndrive_finish "$OUT" "$NONCE" X "9 27"\n')
    assert assess(d, tmp_path / "w") == []
