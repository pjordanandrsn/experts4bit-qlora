"""Mandatory CPU draft dry run: real stage/box scripts, no science or optional skips."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).parents[1]
LANE = ROOT / "bench/dq11"
PROTOCOL = json.loads((LANE / "dq11_protocol.json").read_text())


def load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


box = load("dq11_box")
stage = load("dq11_stage")


def stub(phase, arm):
    return {"stub": True, "phase": phase, "arm": arm}


def test_dry_protocol_runs_all_arms_and_quality_before_decision():
    result = box.dry_run(PROTOCOL, stub, nonce="fixture-dq11")
    assert result["status"] == "DRY_RUN_COMPLETE"
    assert result["scientific_evidence"] is False and result["launchable"] is False
    events = [(e["phase"], e["arm"]) for e in result["events"]]
    assert events[:4] == [("preflight", None), ("prepare_seal", None),
                          ("binding_precision_proof", None), ("initial_quality", None)]
    assert [arm for phase, arm in events if phase == "train"] == ["1-L", "1-U", "1-U0", "2-U0", "2-U", "2-L"]
    assert events[-3:] == [("quality_gate", None), ("decision", None), ("teardown_proof", None)]
    for tag in ("1-L", "1-U", "1-U0", "2-U0", "2-U", "2-L"):
        assert [phase for phase, arm in events if arm == tag] == ["audit_before", "train", "audit_after", "final_quality"]


@pytest.mark.parametrize("stop", ["preflight", "prepare_seal", "binding_precision_proof", "initial_quality",
                                  "audit_before", "train", "audit_after", "final_quality", "quality_gate", "teardown_proof"])
def test_phase_failure_stops_following_work_and_keeps_teardown(stop):
    calls = []

    def worker(phase, arm):
        calls.append((phase, arm))
        if phase == stop:
            raise RuntimeError("fixture refusal: " + stop)
        return stub(phase, arm)

    result = box.dry_run(PROTOCOL, worker, nonce="fixture-dq11")
    assert result["status"] == "DRY_RUN_FAILED" and result["scientific_evidence"] is False
    assert calls[-1] == ("teardown_proof", None)
    first = next(i for i, (phase, _) in enumerate(calls) if phase == stop)
    assert all(phase == "teardown_proof" for phase, _ in calls[first + 1:])
    if stop != "teardown_proof":
        assert ("decision", None) not in calls


@pytest.mark.parametrize("mutant", [{"stub": False}, {"stub": True, "phase": "wrong", "arm": None}, None])
def test_changed_or_non_stub_phase_cannot_pass(mutant):
    result = box.dry_run(PROTOCOL, lambda _phase, _arm: mutant, nonce="fixture-dq11")
    assert result["status"] == "DRY_RUN_FAILED" and result["events"] == []


@pytest.mark.parametrize("key,value", [("arms", ["L", "U", "U", "U0", "U", "L"]),
                                      ("updates", 41), ("status", "REGISTERED")])
def test_changed_registration_stops_before_any_training(key, value):
    protocol = copy.deepcopy(PROTOCOL)
    protocol[key] = value
    result = box.dry_run(protocol, stub, nonce="fixture-dq11")
    assert result["status"] == "DRY_RUN_FAILED"
    assert result["events"] == [stub("teardown_proof", None)]


def test_quality_budget_is_existing_k8_and_budget_below_maintainer_ceiling():
    from experts4bit_qlora.k8_gate import BUDGET

    assert PROTOCOL["quality"]["absolute_ppl_budget"] == BUDGET
    assert PROTOCOL["budget"]["max_total_usd"] == 2.8 < 15
    assert PROTOCOL["budget"]["wallclock_seconds"] == 7200


def test_actual_box_script_runs_with_verified_flat_closure_and_no_scientific_success(tmp_path):
    for path in stage.stage(LANE):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    # A Python checksum command verifies actual bytes on both macOS and Linux.
    commands = tmp_path / "bin"
    commands.mkdir()
    verifier = commands / "sha256sum"
    verifier.write_text(f"#!{sys.executable}\nimport hashlib,sys\nfrom pathlib import Path\n"
                        "assert sys.argv[1:] == ['-c','instrument.sha256']\n"
                        "for line in Path('instrument.sha256').read_text().splitlines():\n"
                        " digest,name=line.split()\n"
                        " assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest\n")
    verifier.chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
               DQ11_W=str(tmp_path), TC1_RUN_NONCE="fixture-dq11", DQ11_DRY_RUN="1")
    result = subprocess.run(["bash", str(LANE / "dq11_run.sh")], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "dq11-dry-run.fixture-dq11.json").read_text())
    assert receipt["status"] == "DRY_RUN_COMPLETE" and receipt["scientific_evidence"] is False
    assert (tmp_path / "TP_DONE.fixture-dq11").exists()
    assert (tmp_path / "TC1_EXIT_CODE.fixture-dq11").read_text().strip() == "0"
    assert not (tmp_path / "TC1_SUCCESS.fixture-dq11").exists()


def test_live_box_and_controller_refuse_without_gpu_network_or_receipt(tmp_path):
    result = subprocess.run([sys.executable, str(LANE / "dq11_box.py"), "--nonce", "fixture-dq11",
                             "--out", str(tmp_path / "receipt.json")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 78 and not (tmp_path / "receipt.json").exists()
    env = dict(os.environ, DQ11_W=str(tmp_path), TC1_RUN_NONCE="fixture-dq11", DQ11_DRY_RUN="0", TC1_DRIVE_DRYRUN="0")
    for script in ("dq11_run.sh", "dq11_drive.sh"):
        result = subprocess.run(["bash", str(LANE / script)], env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 78, result.stdout + result.stderr
    assert (tmp_path / "TC1_EXIT_CODE.fixture-dq11").read_text().strip() == "78"
    assert not (tmp_path / "TC1_SUCCESS.fixture-dq11").exists()


def test_actual_controller_dry_run_stages_every_registered_file(tmp_path):
    env = dict(os.environ, TC1_DRIVE_DRYRUN="1", E4B_RENT_SSH_HOST="fixture.invalid", E4B_RENT_SSH_PORT="22",
               E4B_RENT_SSH_OPTS="-o BatchMode=yes", E4B_RENT_RUN_DIR=str(tmp_path),
               E4B_RENT_RUN_ID="fixture-dq11", E4B_RENT_DEADLINE_EPOCH=str(int(time.time()) + 7200),
               E4B_RENT_INSTANCE_ID="fixture-no-instance",
               E4B_SHA=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    result = subprocess.run(["bash", str(LANE / "dq11_drive.sh")], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"DRYRUN stage \[(.*?)\]", result.stdout)
    assert match
    names = match.group(1).split()
    assert len(names) == len(set(names))
    assert {p.name for p in stage.stage(LANE)} <= set(names)
    assert "bash dq11_run.sh" in result.stdout


@pytest.mark.parametrize("mutation", ["missing", "changed", "duplicate", "path", "empty"])
def test_bad_flat_closure_refuses_before_transport(tmp_path, mutation):
    here = tmp_path / "bench/dq11"
    here.mkdir(parents=True)
    subject = here / "subject.py"
    subject.write_bytes(b"registered bytes")
    line = hashlib.sha256(subject.read_bytes()).hexdigest() + "  subject.py\n"
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
    (here / "instrument.sha256").write_text(line)
    with pytest.raises(ValueError):
        stage.stage(here)
