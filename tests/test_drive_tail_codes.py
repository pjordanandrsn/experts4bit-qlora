"""Every lane driver hands the launcher the box's own exit code, and never reports success the box did not record.

The launcher writes ``command exited {rc}`` from the DRIVER's exit code into the receipt, and its
host-refusal classifier reads only that line: ``command exited (?:13|14|17|18)`` excludes the machine from later pins. So a driver
tail that maps every non-zero box code to one code of its own (``exit 23``) hides a box that refused for its disk (13), egress (14),
power cap (17) or host floor (18), and the next pin can buy the same machine again (e4b#2135's driver review). A tail that exits
``$RC`` when the success marker is missing exits 0 on a box rc 0 without the marker: a success the box never recorded (e4b#2137's
G3 kit). This test runs every driver's fetch-and-exit tail on synthetic fetched directories and asserts:

- (a) the host codes 13, 14, 17 and 18 pass through unchanged;
- (b) rc 0 without the success marker never exits 0;
- (c) a missing exit file never exits 0;

plus a control: rc 0 WITH the marker exits 0. A tail the harness cannot drive (no fetched exit file, or a control that does not exit
0) fails as ``undrivable``. The tail is the driver's own text from its ``RC=$(cat "...EXIT_CODE...")`` line to the end, run with
``RUN_DIR``, ``OUT`` and ``NONCE`` set; a driver that exposes ``<LANE>_DRIVE_TAIL_DIR`` / ``<LANE>_DRIVE_TAIL_NONCE`` is run whole
through them. It runs with only harmless tools on PATH (no rm, ssh, scp, rsync or curl) in a scratch directory.

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
#: the only tools a tail can run here: nothing that deletes or reaches the network
SAFE_TOOLS = ("cat", "printf", "test", "[", "ls", "wc", "tr", "cut", "date", "find", "mkdir", "head", "tail", "grep", "sed", "awk",
              "basename", "dirname", "sort", "true", "false", "env", "echo", "touch", "stat", "python3")
STATUS = re.compile(r"(closed lane: \S.*|live: fix due in the lane's next PR.*)")


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


@pytest.mark.parametrize("driver", DRIVERS, ids=_ids(DRIVERS))
def test_the_driver_tail_passes_the_box_code_through(driver: Path, tmp_path: Path):
    rel = str(driver.relative_to(REPO))
    fails = assess(driver, tmp_path)
    entry = _allowlist().get(rel)
    if not fails:
        assert entry is None, f"{rel} now passes: remove its allowlist entry"
        return
    assert entry is not None, f"{rel} fails {fails} and is not allowlisted (a new or changed driver never is): fix its tail"
    sha = hashlib.sha256(driver.read_bytes()).hexdigest()
    assert entry["sha256"] == sha, f"{rel} changed since it was allowlisted, so it is held to the rule: fix its tail ({fails})"
    assert entry["fails"] == fails, f"{rel}: fails {fails}, allowlisted for {entry['fails']}"


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
