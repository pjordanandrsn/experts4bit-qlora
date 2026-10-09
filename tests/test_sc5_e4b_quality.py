# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's e4b scorer (bench/sc5/sc5_e4b_quality.py): the logit-row selection across served prefill pieces and the
(nll, argmax) pairing, which must equal the bf16 reference script's -- through the module's self-test. The passes
themselves need the card; the SC5 proof exercises them."""
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
pytest.importorskip("torch")


def test_the_e4b_scorer_self_tests():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_e4b_quality.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_e4b_quality self-test OK" in out.stdout, out.stdout + out.stderr
