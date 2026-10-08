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


def tradeoff_fixture():
    run, arms = ra.fixture()
    for tag in ("new_a", "new_b"):
        arms[tag]["training"]["step_ms"] = [1120.] * 20
    run["expected_tradeoffs"] = [{"metric": "train_wall_s", "claim_id": "e4b.synthetic.tradeoff",
                                  "package": "e4b", "release_commit": run["new"]["e4b"]["commit"],
                                  "direction": "worse", "unit": "new_over_old_cost_ratio", "interval": [1.10, 1.20],
                                  "claim_status": "measured", "changelog_quote": "Synthetic e4b.synthetic.tradeoff",
                                  "read_path": "synthetic/RESULTS.md", "read_quote": "Synthetic registered interval",
                                  "identity": run["identity"]}]
    return run, arms


def test_expected_tradeoff_does_not_remove_release_block():
    run, arms = tradeoff_fixture()
    out = ra.reduce(run, arms)
    assert out["verdict"] == "REGRESSION"
    assert out["metrics"]["train_wall_s"]["verdict"] == "REGRESSION"
    assert out["release_clearance"] is False
    assert out["regression_blocks_next_release"] is True
    assert out["expected_tradeoffs"][0]["changes_release_block"] is False
    assert out["expected_tradeoffs"][0]["provenance"] == "MANIFEST_CITATION_REQUIRES_MAINTAINER_VERIFICATION"


@pytest.mark.parametrize("field,value", [("changelog_quote", "uncited"), ("direction", "better"),
                                         ("interval", [1.13, 1.20]), ("claim_status", "retired"),
                                         ("release_commit", "c" * 40), ("unit", "absolute-seconds")])
def test_unsupported_tradeoff_stays_regression(field, value):
    run, arms = tradeoff_fixture()
    run["expected_tradeoffs"][0][field] = value
    out = ra.reduce(run, arms)
    assert out["verdict"] == "REGRESSION"
    assert out["regression_blocks_next_release"] is True
    assert not out["expected_tradeoffs"]
    assert out["rejected_tradeoff_annotations"]


def test_out_of_interval_one_pair_does_not_annotate():
    run, arms = tradeoff_fixture()
    arms["new_b"]["training"]["step_ms"] = [1160.] * 20
    run["expected_tradeoffs"][0]["interval"] = [1.10, 1.14]
    out = ra.reduce(run, arms)
    assert out["verdict"] == "REGRESSION"
    assert not out["expected_tradeoffs"]


def test_tradeoff_cannot_annotate_wrong_workload():
    run, arms = tradeoff_fixture()
    run["expected_tradeoffs"][0]["identity"] = {"workload": "different"}
    out = ra.reduce(run, arms)
    assert out["verdict"] == "REGRESSION"
    assert not out["expected_tradeoffs"]
