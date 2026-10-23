"""E4B_TRAIN_DETERMINISTIC=1 (engines/train_determinism.py): the fused training path's deterministic mode.

Checked: the knob parses; it refuses BY NAME at trainer start without CUBLAS_WORKSPACE_CONFIG (or with a value that is not
deterministic) and with a disagreeing E4B_TRAIN_DET_GATHER; enabled, it turns the det gather on and the fill off; and
deterministic_backward scopes strict mode to the backward only: on inside, off after, off after a raise, refused when
already on at entry or when the mode was never enabled. The trainer and the layer-major step wrap their backward in it.
"""
import inspect

import pytest
import torch

from experts4bit_qlora.engines import train_determinism as td


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    import torch.utils.deterministic
    fill = torch.utils.deterministic.fill_uninitialized_memory
    monkeypatch.setitem(td.STATE, "enabled", False)
    for v in ("E4B_TRAIN_DETERMINISTIC", "E4B_TRAIN_DET_GATHER", "CUBLAS_WORKSPACE_CONFIG"):
        monkeypatch.delenv(v, raising=False)
    yield
    torch.utils.deterministic.fill_uninitialized_memory = fill
    torch.use_deterministic_algorithms(False)


def test_the_knob_is_off_unless_one(monkeypatch):
    for v, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False)):
        monkeypatch.setenv("E4B_TRAIN_DETERMINISTIC", v)
        assert td.deterministic_requested() is want


@pytest.mark.parametrize("cw", [None, "", ":0:0", "4096:8"])
def test_it_refuses_by_name_without_a_deterministic_cublas_config(monkeypatch, cw):
    if cw is not None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", cw)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        td.enable_training_determinism()
    assert td.STATE["enabled"] is False


def test_it_refuses_a_disagreeing_det_gather(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("E4B_TRAIN_DET_GATHER", "0")
    with pytest.raises(RuntimeError, match="E4B_TRAIN_DET_GATHER"):
        td.enable_training_determinism()


@pytest.mark.parametrize("cw", [":4096:8", ":16:8"])
def test_enabled_it_turns_the_gather_on_and_the_fill_off(monkeypatch, cw):
    import torch.utils.deterministic
    from experts4bit_qlora.engines import fast
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", cw)
    torch.utils.deterministic.fill_uninitialized_memory = True
    td.enable_training_determinism()
    assert td.STATE["enabled"] is True and fast.det_gather_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory is False


def _enabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("E4B_TRAIN_DETERMINISTIC", "1")
    td.enable_training_determinism()


def test_strict_mode_is_on_for_the_backward_only(monkeypatch):
    _enabled(monkeypatch)
    x = torch.randn(8, 4, requires_grad=True)
    assert not torch.are_deterministic_algorithms_enabled()               # the forward runs with it off
    loss = (x * 2).sum()
    with td.deterministic_backward():
        assert torch.are_deterministic_algorithms_enabled()
        loss.backward()
    assert not torch.are_deterministic_algorithms_enabled()               # and it is off again for the next step
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_strict_mode_is_off_after_a_raise(monkeypatch):
    _enabled(monkeypatch)
    with pytest.raises(ValueError):
        with td.deterministic_backward():
            raise ValueError("a backward that fails")
    assert not torch.are_deterministic_algorithms_enabled()


def test_it_refuses_when_strict_mode_is_already_on(monkeypatch):
    _enabled(monkeypatch)
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="already on"):
        with td.deterministic_backward():
            pass


def test_it_refuses_the_mode_without_enable(monkeypatch):
    monkeypatch.setenv("E4B_TRAIN_DETERMINISTIC", "1")
    with pytest.raises(RuntimeError, match="enable_training_determinism"):
        with td.deterministic_backward():
            pass


def test_off_it_does_nothing():
    with td.deterministic_backward():
        assert not torch.are_deterministic_algorithms_enabled()


def test_the_trainer_and_the_layer_major_step_wrap_their_backward():
    from experts4bit_qlora import train
    from experts4bit_qlora.engines import layer_major
    src = inspect.getsource(train.main)
    assert "enable_training_determinism()" in src and "with deterministic_backward():\n" in src
    i = src.index("with deterministic_backward():\n")
    assert "(out.loss / GRAD_ACCUM).backward()" in src[i:i + 120]
    assert src.index("enable_training_determinism()") < src.index("torch.manual_seed(")      # before any torch/CUDA work
    lm = inspect.getsource(layer_major)
    j = lm.index("with deterministic_backward():")
    assert ".backward()" in lm[j:j + 200]


def test_the_docstring_states_the_mode_is_the_knob_plus_an_in_run_self_identity_check():
    doc = " ".join((td.__doc__ or "").split())
    assert "THE DETERMINISTIC MODE IS THIS KNOB PLUS A SELF-IDENTITY CHECK IN THE GATE'S OWN RUN, NEVER THE FLAG ALONE." in doc
    assert "the same run with the fill ON" in doc
