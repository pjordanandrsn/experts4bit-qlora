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
    # kernel-budget window (instrument 2): step()s per training step,
    # table format carries the 'active window: N/M' header
    # step_budget.py requires
    assert "TR1_PROFILE_OUT" in src
    assert "_tprof.step()" in src
    assert "active window:" in src


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


def test_tr2_train_arena_wiring():
    """PREREG-tr2 wiring guards, source-level (importing the trainer
    pulls the model stack): engagement happens BEFORE the before-eval
    (both evals and training must run the same path or the quality
    gate mixes paths), refuses a partial engagement, and is env-gated
    inert."""
    from pathlib import Path

    import experts4bit_qlora
    src = (Path(experts4bit_qlora.__file__).parent / "train.py").read_text()
    assert 'os.environ.get("TRAIN_ARENA")' in src
    assert "refusing a partial treatment" in src
    # ordering: engagement text appears before the before-eval call
    assert src.index("enable_hybrid_train(") < src.index(
        "eval_before = eval_loss"), "hybrid must engage before eval"
    # all-VRAM manifest shape
    assert '"vram_frac": 1.0' in src
    # the bnb release: post-enable the loader's expert storage is dead
    # weight and keeping it doubles expert VRAM (Bugbot HIGH, #259) --
    # the wiring must shrink all four storage attrs and empty the cache
    for attr in ("gate_up_proj", "down_proj", "gate_up_absmax",
                 "down_absmax"):
        assert f'"{attr}"' in src, f"{attr} not released"
    assert "empty_cache()" in src
    # the release must PRECEDE the tier build: the build's peak (bnb +
    # arena stacks co-resident) is what OOMed the box, so a
    # post-enable release cannot help (hit live on hyb_a)
    assert src.index("_release_expert_storage(") < src.index(
        "enable_hybrid_train("), "release must precede engagement"


def test_release_swaps_shape_preserved_meta_twins():
    """The BEHAVIOR the wiring depends on, on a real module: after
    release, every packed attr is a meta tensor with the ORIGINAL
    shape (the tier init .view()s them and .float()s the absmax
    views -- both must stay legal), Parameters stay Parameters,
    buffers stay registered, and the freed byte count is the real
    storage. Parameter.data = meta is REJECTED by autograd
    ('incompatible tensor type', hit live on hyb_a attempt 2), which
    is why the helper REPLACES the objects."""
    import experts4bit_qlora.train as _tr
    from experts4bit_qlora._vendor.experts import ExpertsNbit

    m = ExpertsNbit(num_experts=4, hidden_dim=64, intermediate_dim=128,
                    quant_type="nf4")
    shapes = {a: getattr(m, a).shape
              for a in ("gate_up_proj", "down_proj",
                        "gate_up_absmax", "down_absmax")}
    want = sum(getattr(m, a).numel() * getattr(m, a).element_size()
               for a in shapes)
    holder = torch.nn.ModuleDict({"e": m})
    freed = _tr._release_expert_storage(holder, ExpertsNbit)
    assert freed == want, (freed, want)
    for a, shp in shapes.items():
        t = getattr(m, a)
        assert t.device.type == "meta", a
        assert t.shape == shp, (a, t.shape, shp)
        E = m.num_experts
        v = t.view(E, -1)               # the tier init's access pattern
        assert v.shape[0] == E
    assert isinstance(m.gate_up_proj, torch.nn.Parameter)
    assert "gate_up_absmax" in dict(m.named_buffers())
    # absmax .float() (the init materializes the cast) stays legal
    assert m.gate_up_absmax.float().device.type == "meta"
    # idempotent: second call frees nothing
    assert _tr._release_expert_storage(holder, ExpertsNbit) == 0
