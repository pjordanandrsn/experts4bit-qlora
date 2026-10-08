"""CPU sensitivity checks for the registered release-anchor reducer."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ra_reduce", ROOT / "bench/ra/ra_reduce.py")
ra = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ra)


def test_registered_self_test():
    assert ra.self_test() == 0


def write_receipt(root):
    run, arms = ra.fixture()
    files = {"run.json": run, **{f"arm_{tag}.json": arms[tag] for tag in ra.TAGS}}
    sums = []
    for name, obj in files.items():
        data = json.dumps(obj, allow_nan=False).encode()
        (root / name).write_bytes(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def test_checksum_mutation_detected(tmp_path):
    write_receipt(tmp_path)
    ra.verified_files(tmp_path)
    (tmp_path / "arm_new_a.json").write_text("{}")
    with pytest.raises(ra.Invalid, match="checksum mismatch"):
        ra.verified_files(tmp_path)


def test_unchecksummed_raw_log_refused(tmp_path):
    write_receipt(tmp_path)
    (tmp_path / "install.log").write_text("unregistered bytes")
    with pytest.raises(ra.Invalid, match="unchecksummed raw"):
        ra.verified_files(tmp_path)


def test_checksum_escape_refused(tmp_path):
    write_receipt(tmp_path)
    with (tmp_path / "SHA256SUMS").open("a") as f:
        f.write("a" * 64 + "  ../foreign.log\n")
    with pytest.raises(ra.Invalid, match="escapes"):
        ra.verified_files(tmp_path)


def test_proof_cannot_clear_release():
    run, arms = ra.fixture()
    # A fully shaped READING is rejected when merely labelled a PROOF.
    run["kind"] = "PROOF"
    out = ra.reduce(run, arms)
    assert out["verdict"] == "VOID"
    assert out["release_clearance"] is False


def test_argmax_quality_regression_with_same_nll():
    run, arms = ra.fixture()
    for tag in ("new_a", "new_b"):
        for row in arms[tag]["quality_12"]["per_window"]["wikitext"]["R"]:
            row["argmax_ids"] = [2] * 128
    out = ra.reduce(run, arms)
    assert out["verdict"] == "REGRESSION"
    assert out["regression_blocks_next_release"] is True
    assert out["metrics"]["quality_group_12"]["pairs"][0]["bias_nats"] == 0
