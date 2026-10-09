# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""TC1's model-fetch guards (2026-10-09, tc1-5090-143: a box whose model fetch failed measured nothing and still ended OK).

- venv-e4b installs ``brotli>=1.2.0``. It sees the base image's site-packages, whose older brotli made huggingface_hub 2.x's httpx2
  fail on a brotli-encoded response (``process() takes no keyword arguments``).
- The e4b tripwire refuses a visible brotli older than 1.2.
- A fetch probe runs right after venv-e4b's tripwire and before any other venv is built, and refuses the box (rc 15) when it fails.
- A box on which no family's model staged ends rc 15 (HARNESS_ERROR), not OK with every arm a stub.
"""
from __future__ import annotations

import re
from pathlib import Path

RUN = (Path(__file__).resolve().parents[1] / "bench" / "tc1" / "tc1_run.sh").read_text()


def test_venv_e4b_installs_a_new_enough_brotli():
    block = RUN[RUN.index('"git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "git+'):RUN.index("> logs/pip_e4b.log")]
    assert '"brotli>=1.2.0"' in block


def test_the_tripwire_refuses_an_old_brotli():
    trip = RUN[RUN.index("> logs/tripwire_e4b.log"):RUN.index("\nPYT\n", RUN.index("> logs/tripwire_e4b.log"))]
    assert '_bv = _ver("brotli")' in trip and ">= (1, 2)" in trip


def test_the_fetch_probe_runs_before_the_other_venvs_and_refuses_with_rc_15():
    probe = RUN.index("> logs/fetch_probe.log")
    assert RUN.index("> logs/tripwire_e4b.log") < probe < RUN.index("venv-unsloth-t28: unsloth")
    tail = RUN[probe:RUN.index("\nfi\n", probe)]
    assert 'revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"' in RUN[RUN.index("<<'PYP'"):RUN.index("\nPYP\n")]
    assert "finish 15" in tail
    assert re.search(r'TC1_FETCH_PROBE:-1}" = 1 \] && \[ -z "\$\{TC1_LOCAL_SNAPSHOT:-\}" \]', RUN)


def test_a_box_with_no_model_staged_ends_rc_15():
    assert "FETCH_FAILED_N=$((FETCH_FAILED_N + 1)); return 1; fi\n  FETCH_OK_N=$((FETCH_OK_N + 1))" in RUN
    end = RUN[RUN.rindex("NO MODEL STAGED") - 200:]
    assert '[ "$FETCH_OK_N" = 0 ] && [ "$FETCH_FAILED_N" -gt 0 ]' in end and "finish 15" in end
    assert end.rstrip().endswith("finish 0")
