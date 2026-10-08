"""Evaluate the real controller's staging plan locally; no SSH, GPU, rental, or ledger writes."""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("dq9_stage_fixture", ROOT/"bench/dq9/dq9_stage.py")
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)


def test_every_checksum_subject_is_sent_by_the_actual_controller(tmp_path):
    env = dict(os.environ, TC1_DRIVE_DRYRUN="1", E4B_RENT_SSH_HOST="fixture.invalid", E4B_RENT_SSH_PORT="22",
               E4B_RENT_SSH_OPTS="-o BatchMode=yes", E4B_RENT_RUN_DIR=str(tmp_path), E4B_RENT_RUN_ID="fixture-stage-only",
               E4B_RENT_DEADLINE_EPOCH=str(int(time.time())+7200), E4B_RENT_INSTANCE_ID="fixture-no-instance",
               E4B_SHA=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    # This is TC1's existing local stage-plan branch, before any SSH call. It never invokes the compute launcher.
    result = subprocess.run(["bash", "-x", str(ROOT/"bench/dq9/dq9_drive.sh")], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    match = re.search(r"DRYRUN stage \[(.*?)\]", result.stdout)
    assert match, result.stdout
    staged = match.group(1).split()
    assert len(staged) == len(set(staged)), "flat staging must not overwrite a colliding basename"
    assignments = [token.split("=", 1)[1] for line in result.stderr.splitlines() if line.startswith("+ ")
                   for token in shlex.split(line[2:]) if token.startswith("TC1_EXTRA_STAGE=")]
    assert assignments
    subjects = {Path(path).name: Path(path) for path in shlex.split(assignments[-1])}
    manifest = ROOT/"bench/dq9/instrument.sha256"
    for line in manifest.read_text().splitlines():
        digest, filename = line.split()
        assert filename in staged, f"registered checksum file missing from actual stage: {filename}"
        assert hashlib.sha256(subjects[filename].read_bytes()).hexdigest() == digest
    assert "instrument.sha256" in staged


@pytest.mark.parametrize("mutation", ["missing", "changed", "duplicate", "path", "empty"])
def test_invalid_stage_contract_refuses_before_transport(tmp_path, mutation):
    here = tmp_path/"bench/dq9"
    here.mkdir(parents=True)
    subject = here/"subject.py"
    subject.write_bytes(b"registered bytes")
    line = hashlib.sha256(subject.read_bytes()).hexdigest()+"  subject.py\n"
    if mutation == "missing":
        subject.unlink()
    elif mutation == "changed":
        subject.write_bytes(b"different bytes")
    elif mutation == "duplicate":
        line += line
    elif mutation == "path":
        line = line.replace("subject.py", "../subject.py")
    elif mutation == "empty":
        line = ""
    (here/"instrument.sha256").write_text(line)
    with pytest.raises(ValueError):
        stager.stage(here)
