# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's windows builder (bench/sc5/sc5_windows.py): the K8 / P117 corpus join, the window cut and its refusals, and
the digest the reference script checks -- through the module's self-test."""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_windows_builder_self_tests():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_windows.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_windows self-test OK" in out.stdout, out.stdout + out.stderr
