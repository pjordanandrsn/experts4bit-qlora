"""P238's kit, on CPU: the per-load premise table (bench/p238/p238_bytes.py).

The table must say SAME on the two libraries' real quantizers (bitsandbytes' Params4bit with nested statistics, as Unsloth's loader
builds it; e4b's ExpertsNbit + compress_expert_absmax_, as enable_fast_train builds it), and must fail on every way a load can
differ: one packed byte, an fp32 absmax left uncompressed, a lost layer, an empty table.
"""
import importlib.util
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
bnb = pytest.importorskip("bitsandbytes")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("p238_bytes", "bench/p238/p238_bytes.py")
E, HID, INTER = 4, 256, 64                     # gate_up [E, 2*INTER, HID], down [E, HID, INTER]: whole 64-blocks per expert


def _stacks(seed):
    g = torch.Generator().manual_seed(seed)
    gu = (torch.randn(E, 2 * INTER, HID, generator=g) * 0.02).to(torch.bfloat16)
    dn = (torch.randn(E, HID, INTER, generator=g) * 0.02).to(torch.bfloat16)
    return gu, dn


def _e4b_model(n_layers=2, compress=True):
    from experts4bit_qlora import ExpertsLoRA, ExpertsNbit
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for L in range(n_layers):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = ExpertsLoRA(ExpertsNbit.from_float(*_stacks(L)), r=8, alpha=16)
        root.layers.append(layer)
    if compress:
        assert compress_expert_absmax_(root) == n_layers
    return root


class _PeftLike(torch.nn.Module):
    """A wrapper with ``base_layer``, as PEFT's ParamWrapper presents an expert parameter."""

    def __init__(self, base):
        super().__init__()
        self.base_layer = base


def _unsloth_model(n_layers=2, wrap=True):
    from bitsandbytes.nn import Params4bit
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for L in range(n_layers):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        ex = torch.nn.Module()
        for which, W in zip(("gate_up", "down"), _stacks(L)):
            p = Params4bit(W, requires_grad=False, blocksize=64, compress_statistics=True, quant_type="nf4",
                           quant_storage=torch.uint8).to("cpu")
            setattr(ex, f"_{which}_param", p)
            object.__setattr__(ex, f"{which}_proj", _PeftLike(p) if wrap else p)
        layer.mlp.experts = ex
        root.layers.append(layer)
    return root


def test_the_two_libraries_real_quantizers_read_same_on_cpu():
    a, b = B.e4b_table(_e4b_model()), B.unsloth_table(_unsloth_model())
    assert sorted(a) == [0, 1] and set(a[0]["gate_up"]) == set(B.FIELDS)
    assert B.compare(a, b, n_layers=2) == (True, "")


def test_one_flipped_packed_byte_differs():
    m = _e4b_model()
    m.layers[1].mlp.experts.base.down_proj.data[0, 7] ^= 1
    ok, why = B.compare(B.e4b_table(m), B.unsloth_table(_unsloth_model()), n_layers=2)
    assert not ok and "layer 1 down packed" in why


def test_an_uncompressed_fp32_absmax_never_equals_the_nested_table():
    ok, why = B.compare(B.e4b_table(_e4b_model(compress=False)), B.unsloth_table(_unsloth_model()), n_layers=2)
    assert not ok and "absmax" in why


def test_a_lost_layer_fails_even_where_the_kept_layers_agree():
    a, b = B.e4b_table(_e4b_model()), B.unsloth_table(_unsloth_model())
    del b[1]
    ok, why = B.compare(a, b, n_layers=2)
    assert not ok and "not 0..1" in why
    ok, why = B.compare(a, b)
    assert not ok and "layer sets differ" in why


def test_an_empty_table_never_passes():
    assert B.compare({}, {})[0] is False
    assert B.compare(B.e4b_table(_e4b_model()), {})[0] is False


def test_the_unsloth_side_unwraps_peft_and_skips_modules_without_both_stacks():
    a = B.unsloth_table(_unsloth_model(wrap=False))
    b = B.unsloth_table(_unsloth_model(wrap=True))
    assert a == b and sorted(a) == [0, 1]
