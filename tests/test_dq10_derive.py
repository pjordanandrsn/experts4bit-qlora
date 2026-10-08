"""Fitting input mutations cannot replace the archived DQ9 subject bytes."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "bench" / "dq10"
SPEC = importlib.util.spec_from_file_location("dq10_derive_fixture", ROOT / "dq10_derive.py")
derive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(derive)


def test_missing_fitting_rows_refuse(tmp_path):
    with pytest.raises(ValueError, match="missing source"):
        derive.derive(tmp_path)


def test_changed_fitting_receipt_refuses_before_parsing_or_refitting(tmp_path):
    (tmp_path / "read-llama31_8b-device-2048-baseline.json").write_bytes(b"different source bytes")
    with pytest.raises(ValueError, match="differs from the frozen"):
        derive.derive(tmp_path)
