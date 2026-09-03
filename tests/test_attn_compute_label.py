# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The K8 record must report the compute mode that RAN, never the
environment request: every P25 line said `compute=f32` while the mech
tally said fp8 245760 / f32 0."""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(__file__)


def _sd():
    spec = importlib.util.spec_from_file_location(
        "step_decomp", os.path.join(_HERE, "..", "bench", "hybrid-g9", "step_decomp.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step_decomp"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_kernel(monkeypatch, counts):
    m = types.ModuleType("fp8_paged_attn")
    m.compute_counts = lambda: dict(counts)
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", m)


def test_reports_the_mode_that_ran_not_the_request(monkeypatch):
    sd = _sd()
    monkeypatch.setenv("GNF4_ATTN_COMPUTE", "f32")     # a stale/typo'd request
    _fake_kernel(monkeypatch, {"f32": 0, "fp8": 245760})
    assert sd._attn_compute_ran() == "fp8"


def test_mixed_window_names_both(monkeypatch):
    sd = _sd()
    _fake_kernel(monkeypatch, {"f32": 12, "fp8": 100})
    got = sd._attn_compute_ran()
    assert got.startswith("mixed") and "f32 12" in got and "fp8 100" in got


def test_no_decode_attention_is_not_a_mode(monkeypatch):
    sd = _sd()
    _fake_kernel(monkeypatch, {"f32": 0, "fp8": 0})
    assert sd._attn_compute_ran() == "none (no decode attention entered)"


def test_missing_kernel_is_reported_not_guessed(monkeypatch):
    sd = _sd()
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", None)
    assert "unknown" in sd._attn_compute_ran()
