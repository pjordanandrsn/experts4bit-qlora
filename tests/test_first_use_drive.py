# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""T1 step 3's controller (bench/first_use/fu_drive.sh): it verifies the Loggetta kit at its pinned merge commit before any
spend, and its dry run exercises the refusals. The kit files are copies of Loggetta's bench/first_use/ at the pinned commit
(tests/fixtures/first_use/); the first test ties them to the driver's own pins, so they cannot drift. Each dry run runs
the driver from a clean temporary git repository, since the driver refuses a dirty tree. Bash only; no network, no box."""
import hashlib
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVE = REPO / "bench" / "first_use" / "fu_drive.sh"
TEXT = DRIVE.read_text(encoding="utf-8")
KIT = REPO / "tests" / "fixtures" / "first_use"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def pin(name):
    m = re.search(rf"^{name}=([0-9a-f]+)$", TEXT, re.M)
    assert m, name
    return m.group(1)


def test_the_fixture_kit_is_the_pinned_kit():
    assert len(pin("LOGGETTA_KIT_COMMIT")) == 40 and len(pin("KIT_SHA256")) == 64
    assert sha(KIT / "staged-first_use.sha256") == pin("KIT_SHA256")
    for line in (KIT / "staged-first_use.sha256").read_text().splitlines():
        want, name = line.split()
        assert sha(KIT / name) == want, name


def test_every_cell_the_driver_names_is_a_registered_cell():
    spec = importlib.util.spec_from_file_location("fu_reduce", KIT / "fu_reduce.py")
    fu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fu)
    named = set()
    for cells in re.findall(r'CELLS="([^"]+)"', TEXT):
        named |= set(cells.split())
    assert named == set(fu.CELLS) - {"no-gpu"}            # no-gpu runs on a host with no NVIDIA GPU, never a rental


def test_the_fetch_never_brings_back_a_venv_or_tensors():
    assert "--exclude 'venv*'" in TEXT and "--exclude '*.safetensors'" in TEXT and "--exclude '.cache'" in TEXT


def sandbox(tmp_path, kit=None):
    """A clean git repository holding the driver and the helper it sources, plus a kit copy and the RC's two files."""
    repo = tmp_path / "e4b"
    (repo / "bench" / "first_use").mkdir(parents=True)
    (repo / "bench" / "common").mkdir(parents=True)
    shutil.copy(DRIVE, repo / "bench" / "first_use" / "fu_drive.sh")
    shutil.copy(REPO / "bench" / "common" / "lane_liveness.sh", repo / "bench" / "common" / "lane_liveness.sh")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "sandbox"], check=True)
    k = tmp_path / "kit"
    shutil.copytree(kit or KIT, k)
    wheel = tmp_path / "loggetta-0.7.0rc1-py3-none-any.whl"
    wheel.write_bytes(b"a release candidate wheel stands in here")
    env = {**os.environ, "E4B_RENT_SSH_HOST": "203.0.113.7", "E4B_RENT_SSH_PORT": "22", "E4B_RENT_SSH_OPTS": "-o x=y",
           "E4B_RENT_RUN_DIR": str(tmp_path / "run"), "E4B_RENT_RUN_ID": "fu-below-1",
           "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "FU_DRIVE_DRYRUN": "1",
           "FU_KIT_SOURCE": str(k), "FU_RC_COMMIT": "a" * 40, "FU_RC_WHEEL": str(wheel),
           "FU_RC_FIRST_RUN": str(k / "first_run.sh"), "FU_LISTED_DRIVER": "575.64.05"}
    return repo / "bench" / "first_use" / "fu_drive.sh", k, env


def run(script, env):
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)


def test_the_dry_run_verifies_the_kit_and_names_the_cells(tmp_path):
    script, k, env = sandbox(tmp_path)
    r = run(script, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"DRYRUN kit {pin('KIT_SHA256')} at loggetta {pin('LOGGETTA_KIT_COMMIT')} verified" in r.stdout
    assert "FU_CELLS='below-default below-matched cpu-torch'" in r.stdout
    assert f"FU_WHEEL_SHA256={sha(env['FU_RC_WHEEL'])}" in r.stdout and f"FU_FIRST_RUN_SHA256={sha(k / 'first_run.sh')}" in r.stdout


@pytest.mark.parametrize("victim, said", [
    ("fu_reduce.py", "fu_reduce.py is "),                         # a staged file changed after review
    ("fu_run.sh", "fu_run.sh is "),
    ("staged-first_use.sha256", "the reviewed kit hash is"),      # the manifest itself changed
])
def test_a_tampered_kit_is_refused_at_usd_0(tmp_path, victim, said):
    script, k, env = sandbox(tmp_path)
    with open(k / victim, "a") as f:
        f.write("# changed after review\n")
    r = run(script, env)
    assert r.returncode == 78 and said in r.stdout, r.stdout + r.stderr
    assert "DRYRUN" not in r.stdout


@pytest.mark.parametrize("change, said", [
    ({"FU_DRIVE_DRYRUN": "0"}, "FU_KIT_SOURCE is for the dry run only"),    # a local kit never reaches a rental
    ({"E4B_RENT_RUN_ID": "fu-x-1"}, "refusing unregistered run ID"),
    ({"FU_RC_COMMIT": "main"}, "FU_RC_COMMIT is not a 40-char hex sha"),
    ({"FU_LISTED_DRIVER": "575; rm -rf /"}, "FU_LISTED_DRIVER is not a driver version"),
])
def test_the_driver_refuses_before_any_box(tmp_path, change, said):
    script, _k, env = sandbox(tmp_path)
    r = run(script, {**env, **change})
    assert r.returncode == 78 and said in r.stdout, r.stdout + r.stderr


def test_a_wheel_that_is_not_a_wheel_is_refused(tmp_path):
    script, _k, env = sandbox(tmp_path)
    r = run(script, {**env, "FU_RC_WHEEL": str(tmp_path / "loggetta.tar.gz")})
    assert r.returncode == 78 and "must be the RC's .whl" in r.stdout
