# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The 64-slot decode-step census (``bench/h2h-2026-10-02/sc2e/decode64/``) reproduces from its committed receipts, and
its breakdown closes: the parts of every served bucket-64 step add to the step, with the residual explicit."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "bench" / "h2h-2026-10-02" / "sc2e" / "decode64"


def _mod():
    spec = importlib.util.spec_from_file_location("decode64_census", DIR / "decode64_census.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_census_reproduces_from_its_receipts():
    pytest.importorskip("numpy")
    assert _mod().main(["--check"]) == 0


def test_the_breakdown_closes_with_an_explicit_residual():
    c = json.loads((DIR / "decode64_census.json").read_text(encoding="utf-8"))
    for d in ("d1", "d2"):
        k = c["closure"][d]
        parts = k["host_before_replay_ms"] + k["device_replay_ms"] + k["host_tail_ms"]
        assert abs(parts - k["step_ms"]) < 0.002, d
        classes = sum(k["device_by_class_ms"].values()) + k["device_residual_ms"]
        assert abs(classes - k["device_replay_ms"]) < 0.01, d


def test_the_levers_are_ranked_by_their_ceilings():
    c = json.loads((DIR / "decode64_census.json").read_text(encoding="utf-8"))
    ceil = [lv["ceiling_ms"] for lv in c["levers"]]
    assert ceil == sorted(ceil, reverse=True)
    assert [lv["lever"] for lv in c["levers"]] == ["c", "a", "d", "b"]
