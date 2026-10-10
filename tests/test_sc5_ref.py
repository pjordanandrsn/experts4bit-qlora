# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's bf16 reference script (bench/sc5/sc5_ref.py), on a two-layer random Qwen3-MoE on the CPU: the position
convention (row j predicts ids[j+1], targets ids[prompt_len+1 .. prompt_len+steps]) against a hand computation, the
chunked order agreeing with the full forward in fp32, the windows' sha256 check, and the stored sha256."""
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
pytest.importorskip("transformers")


def test_the_reference_script_self_tests():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_ref.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_ref self-test OK" in out.stdout, out.stdout + out.stderr
