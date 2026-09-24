"""The p66 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P66).

`bench/p66/p66_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p66/staged.sha256`; that
guard runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs
nothing. It mirrors the driver's own `case` statement rather than searching for files by name (there are several
`step_decomp.py` and `k8_bake.py` copies under bench/, and a search would pin a file the box never stages):
the three reused tools resolve to `bench/p39/k8_bake.py`, `bench/hybrid-g9/step_decomp.py` and
`bench/hybrid-g9/f1/step_budget.py`; everything else to `bench/p66/`.
"""
import hashlib
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p66"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p66_drive.sh's staging `case`, verbatim."""
    if name == "k8_bake.py":
        return REPO / "bench" / "p39" / name
    if name == "step_decomp.py":
        return REPO / "bench" / "hybrid-g9" / name
    if name == "step_budget.py":
        return REPO / "bench" / "hybrid-g9" / "f1" / name
    return LANE / name


def _entries():
    if not PIN.exists():                     # parametrize evaluates this at COLLECTION, before any skipif applies
        return
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="p66 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and v.endswith((".sh", ".py")) else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p66 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="p66 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (LANE / "p66_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split() if p.rstrip('"').endswith((".py", ".sh", ".json"))}
    named.discard("staged.sha256")           # the pin file itself rides along; it cannot pin itself
    pinned = {name for _w, name in _entries()}
    assert not (named - pinned), f"staged by the driver but not pinned: {sorted(named - pinned)}"
    assert not (pinned - named), f"pinned but never staged (a dead pin guards nothing): {sorted(pinned - named)}"


@pytest.mark.skipif(not PIN.exists(), reason="p66 lane not present")
def test_the_runner_checks_every_piece_it_runs():
    """The box's own `STAGE MISSING` loop must name every pinned file, or a missing piece surfaces as a Python
    ImportError halfway through a rented run instead of a named refusal before any install."""
    runner = (LANE / "p66_run.sh").read_text()
    m = re.search(r"for f in ([^;]+); do\s*\n\s*\[ -s \$W/\$f \]", runner)
    assert m, "p66_run.sh lost its staged-pieces check"
    checked = set(m.group(1).split())
    pinned = {name for _w, name in _entries()} - {"p66_run.sh"}     # the runner is what is running
    assert pinned <= checked, f"pinned but not checked on the box: {sorted(pinned - checked)}"


def test_the_driver_mapping_is_the_test_mapping():
    """The case statement in the driver and `_resolve` here must agree, or CI checks a different file."""
    driver = (LANE / "p66_drive.sh").read_text()
    for name, want in (("k8_bake.py", "bench/p39/"), ("step_decomp.py", "bench/hybrid-g9/"),
                       ("step_budget.py", "bench/hybrid-g9/f1/")):
        assert re.search(rf'{re.escape(name)}\) src="\$REPO/{re.escape(want)}\$name"', driver), name
        assert str(_resolve(name)).endswith(want + name), name


def test_the_driver_never_forwards_a_rehearsal_override():
    """p66_run.sh's P66_REHEARSAL_* knobs exist to exercise the runner on the NAS GPU. A rented run must use none
    of them, so the driver must not pass any to the box (and the runner stamps any it sees into summary.txt)."""
    driver = (LANE / "p66_drive.sh").read_text()
    assert "P66_REHEARSAL" not in driver
    runner = (LANE / "p66_run.sh").read_text()
    assert "OVERRIDE $v=${!v} -- this run is a REHEARSAL, not a reading" in runner
    knobs = set(re.findall(r"P66_REHEARSAL_[A-Z0-9_]+", runner.split("set -uo pipefail", 1)[1]))
    # the knob names carry digits (GNF4_SRC, NF4_ARENA): a letters-only pattern truncates them and passes vacuously
    assert {"P66_REHEARSAL_GNF4_SRC", "P66_REHEARSAL_NF4_ARENA"} <= knobs, sorted(knobs)
    stamp = next(ln for ln in runner.splitlines() if "OVERRIDE $v" in ln)
    assert "[A-Z0-9_]" in stamp, "the OVERRIDE stamp's name pattern must admit digits (A2000 rehearsal round 3)"
    header = runner.split("set -uo pipefail", 1)[0]
    undocumented = sorted(k for k in knobs if k[len("P66_REHEARSAL"):] not in header)
    assert not undocumented, f"rehearsal knobs the runner reads but its header does not name: {undocumented}"


def test_the_proving_mode_stops_before_any_measurement():
    """The compute rule: a rental whose guard exceeds 1 h needs a proving rental (<= $0.15, <= 10 min) first.
    `P66_MODE=prove` is that step. It must exit AFTER the box checks, install, tripwire and pin, and BEFORE
    calibration, the fetch of the full checkpoint, any bake and any arm -- or it is not a 10-minute rental."""
    runner = (LANE / "p66_run.sh").read_text()
    body = runner.split("set -uo pipefail", 1)[1]
    prove_exit = body.index(': > P66_PROVED.$NONCE')
    for before in ("REFUSED: card is", "sha256sum -c staged.sha256", "TRIPWIRE FAIL", "pin_memory()"):
        assert body.index(before) < prove_exit, before
    # the INVOCATIONS, not the names: both also appear in comments above the install block
    for after in ("python $GSRC/bench/calibrate.py", "import snapshot_download", "python $W/k8_bake.py",
                  "larm L_ref", "marm M_ref"):
        assert body.index(after) > prove_exit, after
    assert "case \"$MODE\" in full|prove)" in body, "an unknown mode must refuse, not fall through to the reading"


def test_the_driver_forwards_the_mode_and_requires_the_proved_marker():
    driver = (LANE / "p66_drive.sh").read_text()
    assert "P66_MODE=$MODE" in driver.split("PASS=", 1)[1].splitlines()[0]
    assert 'case "$MODE" in full|prove)' in driver
    assert "P66_PROVED.$NONCE" in driver, "a prove run that ends rc 0 without its marker is not a proving receipt"


def test_every_arm_expects_less_than_its_alarm():
    """can_run() is given the arm's EXPECTED time; the alarm is its cap. When the first draft passed the cap as the
    expectation, late arms on a 2 h guard were skipped as out of time while needing minutes."""
    runner = (LANE / "p66_run.sh").read_text()
    rows = re.findall(r"^\s*(larm|marm) (\S+)\s+(\d+) (\d+) ", runner, flags=re.M)
    assert len(rows) >= 10, rows
    for kind, name, need, cap in rows:
        assert int(need) < int(cap), (name, need, cap)
    assert re.search(r"larm\(\)\{ local n=\$1 need=\$2 cap=\$3;.*can_run \$need", runner)
    assert re.search(r"marm\(\)\{ local n=\$1 need=\$2 cap=\$3;.*can_run \$need", runner)


def _floor_fn(runner: str) -> str:
    a = runner.index("floor(){")
    return runner[a:runner.index("PROOF_MIN_DISK_GB=", a)]


@pytest.mark.parametrize("mode,want_rc,recorded", [("prove", 0, True), ("full", 13, False)])
def test_a_proof_records_reading_floors_and_a_reading_enforces_them(tmp_path, mode, want_rc, recorded):
    """P65 Amendment 1's lesson, applied here: the proving box is not the reading's box, so a floor only the reading
    needs is RECORDED in a proof and ENFORCED in the reading, with the reading's exit code."""
    import subprocess
    runner = (LANE / "p66_run.sh").read_text()
    script = (f"cd {tmp_path}\nMODE={mode}\nsay(){{ :; }}\nfinish(){{ exit $1; }}\n" + _floor_fn(runner)
              + 'floor 13 "host RAM available 40 GB"\nexit 0\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == want_rc, (r.returncode, r.stderr)
    summary = (tmp_path / "summary.txt").read_text() if (tmp_path / "summary.txt").exists() else ""
    assert ("floor_would_refuse_reading rc=13 host RAM available 40 GB" in summary) is recorded
    assert (tmp_path / "REFUSAL").exists() is (not recorded)


def test_which_checks_are_reading_floors_and_which_always_refuse():
    runner = (LANE / "p66_run.sh").read_text()
    body = runner.split("PROOF_MIN_DISK_GB=20", 1)[1]
    # reading-only floors go through floor()
    for msg in ("MiB VRAM free", "(two checkpoints + two arenas", "host RAM available", "cannot pin"):
        line = next(ln for ln in body.splitlines() if msg in ln)
        assert "floor 1" in line, line
    # the class and a dud box refuse whatever the mode; so does disk below the proof's own need
    assert re.search(r'REFUSED: card is .*finish 15', body)
    assert re.search(r'DUD BOX"; finish 10', body)
    assert re.search(r'\[ "\$MODE" = prove \] && \[ "\$\{FREE_GB:-0\}" -lt "\$PROOF_MIN_DISK_GB" \].*finish 13', body)
