# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane 2 design note (a), bench/decode-census/PREFILL.md: it is exactly prefill_census.py's output over the committed
receipts, the class split sums to P119's device total, and the elementwise and routing sub-splits sum to their classes."""
import importlib.util
import pathlib

LANE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "decode-census"


def _mod():
    spec = importlib.util.spec_from_file_location("prefill_census", LANE / "prefill_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_note_is_the_scripts_output():
    assert _mod().main(["--check"]) == 0


def test_the_splits_close():
    g = _mod().gather()
    assert abs(sum(g["classes"].values()) - g["device"]) < 1e-3
    assert abs(sum(g["sub"].values()) - g["classes"]["elementwise"]) < 1e-3
    assert abs(sum(g["route"].values()) - g["classes"]["moe_route"]) < 1e-3
    for d in (1, 2):                                   # the served forward agrees with the eager profile within 1 ms
        assert abs(g["served"][d]["forward_device_ms_p50"] - g["device"]) < 1.0
