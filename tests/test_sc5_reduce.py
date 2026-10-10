# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's reducer (bench/sc5/sc5_reduce.py): labels only when both ABBA pairs clear the bound in one direction, every
losing cell listed, and the VOIDs (not ready, off the lock, an invalid request, the wrong peak in flight, a missing cell, a
matched KV capacity off by more than its rounding, a reference sha256 mismatch, a quality parse refusal) -- through the
module's self-test."""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_reducer_self_tests():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_reduce.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_reduce self-test OK" in out.stdout, out.stdout + out.stderr
