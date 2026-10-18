"""The CLI trainer's FAST_TRAIN knob (bench/cli-fused/DESIGN.md): opt-in, implied by LAYER_MAJOR=1, refused with TRAIN_ARENA.

CPU only. Whether the trainer's fused step is the library call bit for bit is the A2000 leg's question, not this file's.
"""
import importlib

import pytest


def _reload(monkeypatch, **env):
    for k in ("FAST_TRAIN", "LAYER_MAJOR", "OFFLOAD_EXPERTS", "TRAIN_ARENA", "E4B_ABSMAX_DQ"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import experts4bit_qlora.train as train
    return importlib.reload(train)


@pytest.fixture(autouse=True)
def _restore_module():
    yield
    import experts4bit_qlora.train as train
    importlib.reload(train)                     # the module's constants back to the process environment


@pytest.mark.parametrize("env,want", [
    ({}, False),                                              # opt-in: no default flip
    ({"FAST_TRAIN": "1"}, True),
    ({"FAST_TRAIN": "0"}, False),
    ({"LAYER_MAJOR": "1", "OFFLOAD_EXPERTS": "1"}, True),     # layer-major means the fused path
    ({"LAYER_MAJOR": "1", "OFFLOAD_EXPERTS": "1", "FAST_TRAIN": "0"}, False),   # explicit 0 keeps the reference loop
], ids=["unset", "on", "off", "implied-by-layer-major", "explicit-off-under-layer-major"])
def test_fast_train_value(monkeypatch, env, want):
    assert _reload(monkeypatch, **env).FAST_TRAIN is want


def test_a_bad_value_refuses_before_the_load(monkeypatch):
    train = _reload(monkeypatch, FAST_TRAIN="yes")
    monkeypatch.setattr(train, "load_moe_4bit_streaming", lambda *a, **k: pytest.fail("loaded a model"))
    monkeypatch.setattr(train.sys, "argv", ["train"])
    with pytest.raises(SystemExit, match="FAST_TRAIN='yes'"):
        train.main()


def test_train_arena_refuses_before_the_load(monkeypatch):
    train = _reload(monkeypatch, FAST_TRAIN="1", TRAIN_ARENA="/nonexistent/arena")
    monkeypatch.setattr(train, "load_moe_4bit_streaming", lambda *a, **k: pytest.fail("loaded a model"))
    monkeypatch.setattr(train.sys, "argv", ["train"])
    with pytest.raises(SystemExit, match="FAST_TRAIN=1 cannot be combined with TRAIN_ARENA"):
        train.main()


def test_help_lists_the_knob(monkeypatch, capsys):
    train = _reload(monkeypatch)
    train._print_env_help("train")
    out = capsys.readouterr().out
    assert "FAST_TRAIN" in out and "enable_fast_train(dgrad=True)" in out and "implies FAST_TRAIN=1" in out


def test_checkpointing_is_enabled_before_the_fused_path_and_not_again():
    """enable_fast_train routes only already-checkpointed layers through e4b's reentrant checkpoint, and a second
    gradient_checkpointing_enable puts Hugging Face's back: under FAST_TRAIN the trainer enables it first and skips the later
    call (the A2000 leg reads the routing itself)."""
    import inspect

    import experts4bit_qlora.train as train
    src = inspect.getsource(train.main)
    block = src[src.index("    if FAST_TRAIN:\n"):src.index("    elif LAYER_MAJOR:")]
    assert block.index("model.gradient_checkpointing_enable(") < block.index("enable_fast_train(model")
    later = src[src.index("model.enable_input_require_grads()") - 400:src.index("model.enable_input_require_grads()")]
    assert "if not FAST_TRAIN:" in later and later.count("model.gradient_checkpointing_enable(") == 1


def test_the_arena_refusal_says_when_layer_major_implied_it(monkeypatch):
    train = _reload(monkeypatch, LAYER_MAJOR="1", OFFLOAD_EXPERTS="1", TRAIN_ARENA="/nonexistent/arena")
    monkeypatch.setattr(train, "load_moe_4bit_streaming", lambda *a, **k: pytest.fail("loaded a model"))
    monkeypatch.setattr(train.sys, "argv", ["train"])
    with pytest.raises(SystemExit, match=r"implied by LAYER_MAJOR=1"):
        train.main()
