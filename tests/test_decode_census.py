# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The lane 2 design note (bench/decode-census/README.md) is exactly decode_census.py's output over the committed
receipts, and its residuals close: every column's served step minus its eager classes, and every SC5 TPOT row."""
import importlib.util
import pathlib

LANE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "decode-census"


def _mod():
    spec = importlib.util.spec_from_file_location("decode_census", LANE / "decode_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_note_is_the_scripts_output():
    assert _mod().main(["--check"]) == 0


def test_the_breakdowns_close():
    g = _mod().gather()
    for rows, col in g["cols"].items():
        assert abs(sum(col["cls"].values()) - col["eager"]) < 1e-9
        for served in col["served"]:
            assert 0 < served - col["eager"] < 3.0, (rows, served, col["eager"])   # host + captured-vs-eager, positive
    for s in g["split"]:
        assert abs(s["tpot"] - s["step"] - s["stall"] - s["resid"]) < 1e-9
        assert abs(s["resid"]) < 0.5, s                                             # the TPOT split closes
