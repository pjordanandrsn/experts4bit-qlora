"""CPU-safe tests for the controller provenance gate (scripts/validate_job_provenance.py).

Pins the classification a doc claim depends on: complete+consistent -> claim_usable; missing
attribution/env or a manifest-vs-result mismatch -> debug_only; contradictory -> invalid; no
result -> missing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import validate_job_provenance as gate  # noqa: E402


def _job(tmp, job_id, result=None, status="pass", env=True, command=True):
    d = tmp / job_id
    os.makedirs(d)
    if result is not None:
        (d / "result.json").write_text(json.dumps(result))
    if status is not None:
        (d / "status.json").write_text(json.dumps({"status": status}))
    if env:
        (d / "env.json").write_text("{}")
    if command:
        (d / "command.sh").write_text("#!/bin/bash\n")
    return str(d)


_GOOD = {
    "job_id": "train_olmoe_nf4_resident_seed1337", "status": "pass", "commit": "abc123",
    "pod_id": "pod-a", "hostname": "h", "torch_version": "2.8.0", "started_at": "t0",
    "finished_at": "t1", "gpu_name": "RTX A5000", "cuda_version": "12.8",
    "bitsandbytes_version": "0.49.2", "storage_mode": "nf4", "offload": False, "seed": 1337,
}


def test_claim_usable(tmp_path):
    jid = _GOOD["job_id"]
    _job(tmp_path, jid, result=_GOOD)
    cls, reasons = gate.classify(jid, str(tmp_path / jid),
                                 {"storage_mode": "nf4", "offload": False, "seed": 1337})
    assert cls == "claim_usable" and reasons == []


def test_missing_env_is_debug_only(tmp_path):
    jid = _GOOD["job_id"]
    _job(tmp_path, jid, result=_GOOD, env=False)
    cls, reasons = gate.classify(jid, str(tmp_path / jid), {})
    assert cls == "debug_only" and any("env.json" in r for r in reasons)


def test_missing_versions_is_debug_only(tmp_path):
    jid = _GOOD["job_id"]
    bad = dict(_GOOD)
    del bad["bitsandbytes_version"]
    _job(tmp_path, jid, result=bad)
    cls, reasons = gate.classify(jid, str(tmp_path / jid), {})
    assert cls == "debug_only" and any("bitsandbytes_version" in r for r in reasons)


def test_manifest_mismatch_is_debug_only(tmp_path):
    jid = _GOOD["job_id"]
    _job(tmp_path, jid, result=_GOOD)
    # manifest says int8, result says nf4 -> integrity failure, not claim-usable.
    cls, reasons = gate.classify(jid, str(tmp_path / jid), {"storage_mode": "int8"})
    assert cls == "debug_only" and any("storage_mode" in r for r in reasons)


def test_job_id_mismatch_is_invalid(tmp_path):
    jid = _GOOD["job_id"]
    _job(tmp_path, jid, result=dict(_GOOD, job_id="something_else"))
    cls, reasons = gate.classify(jid, str(tmp_path / jid), {})
    assert cls == "invalid"


def test_no_result_but_ran_is_debug_only(tmp_path):
    jid = _GOOD["job_id"]
    _job(tmp_path, jid, result=None, status="running")  # status.json but no result.json
    cls, reasons = gate.classify(jid, str(tmp_path / jid), {})
    assert cls == "debug_only" and any("result.json" in r for r in reasons)


def test_never_ran_is_missing(tmp_path):
    cls, reasons = gate.classify("ghost_job", str(tmp_path / "ghost_job"), {})
    assert cls == "missing"


def test_failed_job_is_debug_only(tmp_path):
    jid = _GOOD["job_id"]
    _job(tmp_path, jid, result=dict(_GOOD, status="fail", fail_or_skip_reason="OOM"), status="fail")
    cls, reasons = gate.classify(jid, str(tmp_path / jid), {})
    assert cls == "debug_only" and any("OOM" in r for r in reasons)
