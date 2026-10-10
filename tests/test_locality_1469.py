# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""#1469 item 1, the expert-locality census (bench/locality-1469/): the summaries (union per window within sequences,
consecutive churn, the residency hit rate derived from the engine's own counters) and the capture (router discovery
and recording as grouped-nf4-gemm's capture_routing does it, decode and prefill traces, the npz / jsonl / manifest
outputs) on a tiny random Qwen3-MoE on the CPU -- through each module's self-test."""
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "locality-1469"


def _self_test(name):
    out = subprocess.run([sys.executable, str(LANE / name), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_summaries_self_test():
    pytest.importorskip("numpy")
    _self_test("locality_summary.py")


def test_the_capture_self_tests_on_a_tiny_qwen3_moe():
    pytest.importorskip("transformers")
    _self_test("locality_capture.py")


def test_every_e4b_call_in_the_census_matches_the_installed_signature():
    """loc-a2000-5 (2026-10-10) failed after a clean calibrate: enable_pipelined_residency requires k_slots, and the
    self-test never calls the real engine (it needs CUDA). Each e4b call the census makes is checked here against the
    real function's signature, so a mismatch fails in CI instead of on a rented box."""
    import ast
    import inspect
    pipelined = pytest.importorskip("experts4bit_qlora.engines.pipelined")
    profile = pytest.importorskip("experts4bit_qlora.engines.expert_profile")
    tree = ast.parse((LANE / "locality_capture.py").read_text(encoding="utf-8"))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)
    real = {"enable_pipelined_residency": pipelined.enable_pipelined_residency,
            "hot_sets_from_profile": profile.hot_sets_from_profile}
    for name, fn in real.items():
        assert calls.get(name), f"the census no longer calls {name}"
        sig = inspect.signature(fn)
        for c in calls[name]:
            bound = sig.bind(*[object()] * len(c.args), **{k.arg: object() for k in c.keywords if k.arg})
            for pname, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty or pname == "k_slots":
                    assert pname in bound.arguments, f"{name}(...) at line {c.lineno} does not pass {pname}"

