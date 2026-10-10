# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SD2's proof harness (``bench/sd2/PREREG-sd2.md``, Amendment 1): the staged pins, the self-tests, the driver's
staging table and its dry run. The box path itself is the dry run's (``tests/test_sd2_dryrun.py``)."""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sd2"
SOURCES = {"sd2_run.sh": LANE, "sd2_box.py": LANE, "sd2_reduce.py": LANE, "sd1_box.py": REPO / "bench" / "sd1",
           "sd1_eagle3.py": REPO / "bench" / "sd1", "chat_prompts.json": REPO / "bench" / "sd1",
           "p109_box.py": REPO / "bench" / "p109", "k8_bake.py": REPO / "bench" / "p39", "calib.json": REPO / "bench" / "p39"}


def _pins():
    out = {}
    for line in (LANE / "staged.sha256").read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def test_every_staged_piece_matches_its_source():
    pins = _pins()
    assert set(pins) == set(SOURCES), sorted(set(pins) ^ set(SOURCES))
    for name, want in pins.items():
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: {got} != staged {want}"


def test_the_driver_stages_exactly_the_pinned_pieces():
    drive = (LANE / "sd2_drive.sh").read_text(encoding="utf-8")
    stage = re.search(r'^STAGE="([^"]+)"', drive, re.M).group(1).split()
    names = {pathlib.PurePosixPath(p).name for p in stage} - {"staged.sha256"}
    assert names == set(SOURCES)


def test_the_runner_pins_the_integration_commit_and_the_kernel_release():
    run = (LANE / "sd2_run.sh").read_text(encoding="utf-8")
    assert re.search(r"^E4B_T=539a2d2694096a81c6b272b335e26d818647cd95\b", run, re.M)
    assert re.search(r"^GNF4_T=724ccc454f006c1a46836e434e997f31f293747f\b", run, re.M)
    assert "E4B_PAGED_SPEC=eagle3 E4B_PAGED_SPEC_K=3 E4B_PAGED_SPEC_HEAD=$HEAD_DIR" in run


@pytest.mark.parametrize("script", ["sd2_reduce.py", "sd2_box.py"])
def test_the_self_tests_pass(script):
    out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "SD2_DRIVE_DRYRUN": "1",
           "E4B_SHA": "1" * 40, "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "sd2-prove-dry", "E4B_RENT_DEADLINE_EPOCH": "4102444800",
           "E4B_RENT_INSTANCE_ID": "0"}
    out = subprocess.run(["bash", str(LANE / "sd2_drive.sh")], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/sd2" in out.stdout and "bash sd2_run.sh" in out.stdout
