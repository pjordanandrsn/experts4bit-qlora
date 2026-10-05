"""Lane P114 (bench/p114/PREREG-p114.md): the staged bytes are the pinned ones, the box script gates every energy pass on
correctness, the card check is exact, the rule's self-test passes, and the driver's dry run resolves.

`bench/p114/p114_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p114/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. These tests run the same checks in CI, where they cost nothing.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p114"
PIN = LANE / "staged.sha256"
SOURCES = {"p114_run.sh": LANE / "p114_run.sh", "p114_gate.py": LANE / "p114_gate.py",
           "p114_reduce.py": LANE / "p114_reduce.py",
           "bench_energy.py": REPO / "bench" / "_upstream" / "bench_energy.py",
           "bench_energy_excluded.py": REPO / "bench" / "bench_energy_excluded.py"}
#: the 2026-07-01 harness both energy rows were measured with (docs/METHODOLOGY.md section 10)
HARNESS_SHA = "ece6b5c81aedc4515f9d34dba3f51c1b7ab4a7856e8aa593400de9143aacf032"


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = SOURCES[name]
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, f"{name} changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}"


def test_every_staged_piece_is_pinned_and_the_harness_is_unchanged():
    pinned = {name for _w, name in _entries()}
    assert pinned == set(SOURCES), sorted(pinned ^ set(SOURCES))
    assert dict((n, w) for w, n in _entries())["bench_energy.py"] == HARNESS_SHA, "the energy harness is not the 2026-07-01 one"
    drive = (LANE / "p114_drive.sh").read_text()
    stage = next(ln for ln in drive.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].rstrip('"') for p in stage.split() if p.rstrip('"').endswith((".py", ".sh"))}
    assert named == set(SOURCES), sorted(named ^ set(SOURCES))


def test_correctness_gates_every_energy_pass_and_the_card_check_is_exact():
    run = (LANE / "p114_run.sh").read_text()
    card = run.index('[ "$GPU" = "NVIDIA GeForce RTX 5090" ]')
    power = run.index("finish 17")
    install = run.index("python -m pip install -q --no-input -c logs/constraints.txt")
    gate = run.index("python $W/p114_gate.py")
    selftest = run.index("python $W/p114_reduce.py --self-test")
    first_pass = run.index("one_pass bench_energy.py proj_$rep")
    excl = run.index("one_pass bench_energy_excluded.py excl_$rep")
    reduce = run.index("python $W/p114_reduce.py $W $W/p114_verdict.json")
    assert card < power < install < gate < selftest < first_pass < excl < reduce
    assert "finish 21; }" in run[gate:first_pass]
    assert "*5090*" not in run
    assert 'BNB_VER=0.50.2' in run


def test_every_time_left_check_fits_the_guard():
    """The guard registered in the PREREG must hold the install, the gate, six passes with their cool-downs and the
    600 s margin `left()` keeps (P109's lesson: a check that cannot fit its own guard is a lane that cannot run)."""
    prereg = (LANE / "PREREG-p114.md").read_text()
    m = re.search(r"guard \*\*([0-9.]+) h\*\*", prereg)
    assert m, "the PREREG states the reading's guard as 'guard **<h> h**'"
    guard_s = float(m.group(1)) * 3600
    run = (LANE / "p114_run.sh").read_text()
    caps = [int(c) for c in re.findall(r"one_pass bench_energy(?:_excluded)?\.py \w+_\$rep (\d+)", run)]
    assert caps == [900, 600], caps
    install_and_gate = 900 + 600            # the pip alarm and the gate's alarm, at most
    passes = 3 * (60 + 120) + 3 * (60 + 60)  # cool-down + a pass's expected length, both harnesses (~70 s and ~30 s)
    assert install_and_gate + passes + 600 < guard_s, (install_and_gate + passes + 600, guard_s)


def test_the_rule_self_test_passes():
    r = subprocess.run([sys.executable, str(LANE / "p114_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "10/10 cases" in r.stdout


def test_the_driver_dry_run_resolves():
    env = dict(os.environ, E4B_RENT_SSH_HOST="h", E4B_RENT_SSH_PORT="1", E4B_RENT_SSH_OPTS="-o X=1",
               E4B_RENT_RUN_DIR="/tmp/p114-dry", E4B_RENT_RUN_ID="p114-dry", E4B_RENT_DEADLINE_EPOCH="1",
               E4B_RENT_INSTANCE_ID="0", E4B_SHA="0" * 40, P114_DRIVE_DRYRUN="1")
    r = subprocess.run(["bash", str(LANE / "p114_drive.sh")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("DRYRUN stage -> root@h:/root/p114")
