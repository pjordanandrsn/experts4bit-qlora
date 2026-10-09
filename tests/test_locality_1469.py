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
