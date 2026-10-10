# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's quality scorers (bench/sc5/sc5_quality.py): the pinned vLLM ``prompt_logprobs=1`` and SGLang
``top_logprobs_num=1`` response shapes, the refusals (including a full-vocabulary slot, vLLM's ``logprobs=-1`` V+1 trap),
and the comparison against the common reference -- through the module's own self-test."""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_quality_scorers_self_test():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_quality.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_quality self-test OK" in out.stdout, out.stdout + out.stderr
