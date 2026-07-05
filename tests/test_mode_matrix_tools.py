"""CPU-safe unit tests for the train/query mode-matrix tooling (scripts/mode_matrix_common.py).

No torch, no CUDA, no model downloads: parser, sidecar metadata round-trip, mismatch detection,
result-row schema, and the summarizer over a fixture matrix with pass/fail/skip rows.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from mode_matrix_common import (  # noqa: E402
    REQUIRED_ROW_FIELDS,
    best_per,
    compute_mismatch,
    matrix_table,
    mode_label,
    parse_mode_label,
    parse_mode_list,
    read_metadata,
    transfer_summary,
    validate_row,
    write_metadata,
)


def test_parse_mode_label():
    assert parse_mode_label("nf4") == ("nf4", False)
    assert parse_mode_label("nf4-offload") == ("nf4", True)
    assert parse_mode_label(" INT8-OFFLOAD ") == ("int8", True)
    assert parse_mode_label("bf16") == ("bf16", False)
    assert mode_label("nf4", True) == "nf4-offload"
    for bad in ("int4", "nf4_offload", "offload-nf4", "", "nf4-resident"):
        with pytest.raises(ValueError, match="mode label|unknown mode"):
            parse_mode_label(bad)
    with pytest.raises(ValueError):
        parse_mode_label(4)


def test_parse_mode_list_dedups_and_validates():
    triples = parse_mode_list("nf4, NF4, int8-offload,nf4-offload")
    assert [t[0] for t in triples] == ["nf4", "int8-offload", "nf4-offload"]
    assert triples[1] == ("int8-offload", "int8", True)
    with pytest.raises(ValueError, match="unknown mode"):
        parse_mode_list("nf4,int4")
    with pytest.raises(ValueError, match="no modes"):
        parse_mode_list(" , ")


def test_metadata_roundtrip(tmp_path):
    meta = {"train_storage_mode": "nf4", "train_offload": True, "steps": 150, "seed": 0}
    p = write_metadata(str(tmp_path), meta)
    assert os.path.basename(p) == "expertsnbit_adapter_metadata.json"
    back = read_metadata(str(tmp_path))
    assert back["train_storage_mode"] == "nf4" and back["train_offload"] is True
    assert back["metadata_schema"] == 1 and "timestamp" in back  # stamped on write
    assert read_metadata(str(tmp_path / "nope")) is None  # absent -> None, not an exception


def test_compute_mismatch():
    same = compute_mismatch({"train_storage_mode": "nf4", "train_offload": False}, "nf4", False)
    assert same == {"storage_mode_mismatch": False, "offload_mismatch": False, "warnings": []}

    cross = compute_mismatch({"train_storage_mode": "nf4", "train_offload": False}, "int8", False)
    assert cross["storage_mode_mismatch"] is True and cross["offload_mismatch"] is False
    assert any("storage-mode mismatch" in w for w in cross["warnings"])

    off = compute_mismatch({"train_storage_mode": "nf4", "train_offload": True}, "nf4", False)
    assert off["storage_mode_mismatch"] is False and off["offload_mismatch"] is True

    unknown = compute_mismatch(None, "nf4", False)
    assert unknown["storage_mode_mismatch"] is None
    assert any("provenance unknown" in w for w in unknown["warnings"])


def _row(train, query, status="pass", loss=None, base=None, reason=None):
    t_storage, t_off = parse_mode_label(train)
    q_storage, q_off = parse_mode_label(query)
    return {
        "run_id": "t", "base_model": "m",
        "train_mode_label": train, "train_storage_mode": t_storage, "train_offload": t_off,
        "query_mode_label": query, "query_storage_mode": q_storage, "query_offload": q_off,
        "storage_mode_mismatch": t_storage != q_storage, "offload_mismatch": t_off != q_off,
        "eval_loss_with_adapter": loss, "eval_loss_base_query_mode_no_adapter": base,
        "delta_vs_base_query_mode": None if (loss is None or base is None) else loss - base,
        "status": status, "skip_or_fail_reason": reason, "timestamp": "2026-01-01T00:00:00Z",
    }


def test_validate_row():
    validate_row(_row("nf4", "int8", loss=1.0, base=1.1))
    with pytest.raises(ValueError, match="missing fields"):
        validate_row({"status": "pass"})
    with pytest.raises(ValueError, match="status must be"):
        validate_row(dict(_row("nf4", "nf4"), status="ok"))
    with pytest.raises(ValueError, match="skip_or_fail_reason"):
        validate_row(dict(_row("nf4", "nf4"), status="fail", skip_or_fail_reason=None))


FIXTURE = [
    _row("nf4", "nf4", loss=1.030, base=1.490),
    _row("nf4", "int8", loss=1.026, base=1.481),
    _row("nf4-offload", "nf4", loss=1.040, base=1.490),
    _row("int8", "int8", loss=1.025, base=1.481),
    _row("int8", "nf4", loss=1.033, base=1.490),
    _row("int8", "fp4", status="fail", reason="synthetic failure"),
    _row("nf4", "bf16", status="skip", reason="did not fit on host"),
]


def test_summarizer_handles_pass_fail_skip():
    passed = [r for r in FIXTURE if r["status"] == "pass"]
    table = matrix_table(FIXTURE, "eval_loss_with_adapter")
    assert "FAIL" in table and "SKIP" in table and "1.0300" in table

    lines = transfer_summary(passed)
    text = "\n".join(lines)
    assert any(line.startswith("same-mode: nf4 ->") for line in lines)
    assert "upward: nf4 -> int8" in text
    assert "downward: int8 -> nf4" in text
    # nf4-offload has no exact same-label query leg; its same-mode anchor is nf4 resident.
    assert any(line.startswith("same-mode: nf4-offload -> nf4") for line in lines)
    assert "symmetry: int8->nf4" in text or "symmetry: nf4->int8" in text

    best_t = best_per(passed, "train_mode_label")
    assert best_t["nf4"]["query_mode_label"] == "int8"  # 1.026 < 1.030 in the fixture
    best_q = best_per(passed, "query_mode_label")
    assert best_q["int8"]["train_mode_label"] == "int8"


def test_required_fields_pinned():
    # The docs and summarizer rely on these exact keys; renaming one is a breaking change.
    assert set(REQUIRED_ROW_FIELDS) <= set(_row("nf4", "nf4"))
