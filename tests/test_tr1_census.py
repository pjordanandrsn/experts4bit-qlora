# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-tr1-census CPU gates: the PhaseClock accounting (CPU
degrade mode) and the trainer's bracket wiring.

The wiring test guards the SHIPPED loop's instrumentation the same
way the packaging guard works: source-level assertions that every
registered phase is bracketed in train.py and that the whole thing is
flag-gated inert. The GPU semantics (event fencing) are exercised on
the census box."""

import json
import time

import pytest

torch = pytest.importorskip("torch")
census = pytest.importorskip("experts4bit_qlora.census")


def test_clock_accumulates_phases_cpu(tmp_path):
    c = census.PhaseClock(use_cuda=False)
    for _ in range(2):
        c.step_start()
        for phase, dt in (("data", 0.01), ("forward", 0.02),
                          ("data", 0.01)):
            c.start(phase)
            time.sleep(dt)
            c.stop()
        c.step_end()
    assert len(c.steps) == 2
    row = c.steps[0]
    assert row["data"] == pytest.approx(20.0, rel=0.5)      # ms, 2 brackets
    assert row["forward"] == pytest.approx(20.0, rel=0.5)
    assert row["step_wall_ms"] >= row["data"] + row["forward"] - 1.0
    out = c.write(tmp_path / "c.json", {"model": "m"})
    assert json.loads((tmp_path / "c.json").read_text()) == out


def test_clock_refuses_misuse():
    c = census.PhaseClock(use_cuda=False)
    c.step_start()
    c.start("data")
    with pytest.raises(AssertionError, match="still open"):
        c.start("forward")
    c.stop()
    with pytest.raises(AssertionError, match="no open"):
        c.stop()
    c.start("forward")
    with pytest.raises(AssertionError, match="left open"):
        c.step_end()


def test_trainer_brackets_are_wired_and_gated():
    # read as source, not import: importing the trainer pulls the full
    # model-loading dependency stack (accelerate etc.) for a test that
    # only guards bracket wiring
    from pathlib import Path

    import experts4bit_qlora
    src = (Path(experts4bit_qlora.__file__).parent / "train.py").read_text()
    # flag-gated: the clock exists only under census.enabled()
    assert "_census.enabled()" in src
    # every registered phase is bracketed in the shipped loop
    for phase in ("data", "forward", "backward", "loss_sync", "optim"):
        assert f'_clock.start("{phase}")' in src, f"{phase} not bracketed"
    # the OOM-discard path closes any open bracket
    assert "_clock._open is not None" in src
    # census output is written after the loop
    assert "TR1_CENSUS_OUT" in src


def test_composer_self_test_runs():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "bench/tr1-census/tr1_compose.py"),
         "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test: OK" in out.stdout
