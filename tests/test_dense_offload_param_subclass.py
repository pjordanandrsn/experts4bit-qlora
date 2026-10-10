# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Dense offload's kept-small branch with a ``Parameter`` subclass: bitsandbytes' ``Params4bit``.

A frozen parameter under ``MIN_BYTES`` stays resident, moved to the offload device when it sits elsewhere. That branch
re-wrapped the moved tensor in ``torch.nn.Parameter``, which refuses a subclass whose ``detach()`` returns a plain tensor --
``Params4bit``'s does -- so a model whose 4-bit projections pack under ``MIN_BYTES`` (a small model, or small projections)
raised at ``enable_dense_offload``. ``_placed_param`` stores such a subclass as ``.to()`` returned it, with its quantization
state; a plain ``Parameter`` is re-wrapped as before, bit for bit.

- CPU (CI): the helper on a plain parameter (re-wrapped, values and ``requires_grad`` kept) and on a ``Params4bit`` (stored
  as returned, quantization state intact).
- CUDA: ``enable_dense_offload`` onto CUDA of a model whose small 4-bit projections sit on the CPU, packed already and
  quantizing on the move. The forward is bit for bit the same model moved with ``Module.to``.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

bnb = pytest.importorskip("bitsandbytes")

from experts4bit_qlora.engines import dense_offload as do  # noqa: E402
from experts4bit_qlora.engines.dense_offload import _DenseOffload, _placed_param, enable_dense_offload  # noqa: E402


@pytest.mark.parametrize("requires_grad", [False, True])
def test_a_plain_parameter_is_rewrapped_as_before(requires_grad):
    t = nn.Parameter(torch.randn(8, 4), requires_grad=requires_grad)
    moved = t.detach().clone()                         # what .to(other device) returns for a plain Parameter: a plain tensor
    p = _placed_param(t, moved)
    assert type(p) is nn.Parameter and p.requires_grad is requires_grad
    assert torch.equal(p.detach(), t.detach()) and p.data_ptr() == moved.data_ptr()


def _params4bit(packed: bool):
    w = torch.randn(128, 64, dtype=torch.bfloat16)
    p = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4", blocksize=64, compress_statistics=True)
    if packed:
        try:
            p = p.to("cpu")                            # bitsandbytes quantizes on this move
        except (RuntimeError, NotImplementedError) as e:
            pytest.skip(f"this bitsandbytes cannot quantize on the CPU: {e}")
        assert p.bnb_quantized
    return p


def test_a_params4bit_is_stored_as_returned_with_its_quant_state():
    t = _params4bit(packed=True)
    moved = t.to("cpu")                                # a Params4bit (the subclass), carrying the quant state
    assert isinstance(moved, bnb.nn.Params4bit)
    p = _placed_param(t, moved)
    assert p is moved and isinstance(p, bnb.nn.Params4bit)
    assert p.quant_state is t.quant_state and p.requires_grad is False and p.bnb_quantized


# --- CUDA: the branch itself ----------------------------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kept-small branch moves CPU tensors to CUDA")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = bnb.nn.Linear4bit(64, 128, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4", device="cpu")
        self.down = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4", device="cpu")
        self.norm = nn.Parameter(torch.ones(64, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x):
        return x + self.down(torch.relu(self.up(x * self.norm)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(2))

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


def _toy(packed: bool, seed=5):
    torch.manual_seed(seed)
    m = Toy()
    for lin in (mod for mod in m.modules() if isinstance(mod, bnb.nn.Linear4bit)):
        w = torch.randn(lin.out_features, lin.in_features, dtype=torch.bfloat16) * 0.05
        lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4", blocksize=64, compress_statistics=True)
        if packed:
            try:
                lin.weight = lin.weight.to("cpu")
            except (RuntimeError, NotImplementedError) as e:
                pytest.skip(f"this bitsandbytes cannot quantize on the CPU: {e}")
    return m.eval()


@pytest.fixture(autouse=True)
def _clean_class_state():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()


@needs_cuda
@pytest.mark.parametrize("packed", [True, False], ids=["packed-on-cpu", "quantized-on-the-move"])
def test_small_4bit_projections_are_placed_and_compute_as_module_to(packed):
    ref = _toy(packed).to("cuda")
    m = _toy(packed)
    sizes = [lin.weight.numel() * lin.weight.element_size() for lin in m.modules() if isinstance(lin, bnb.nn.Linear4bit)]
    assert sizes and max(sizes) < do.MIN_BYTES                     # every projection takes the kept-small branch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enable_dense_offload(m, "cuda", pin=False, prefetch=False)
    for lin in (mod for mod in m.modules() if isinstance(mod, bnb.nn.Linear4bit)):
        w = lin.weight
        assert isinstance(w, bnb.nn.Params4bit) and w.is_cuda and w.bnb_quantized
        assert w.quant_state.absmax.is_cuda and not w.requires_grad
    for blk_m, blk_r in zip(m.layers, ref.layers):
        assert type(blk_m.norm) is nn.Parameter and blk_m.norm.is_cuda and not blk_m.norm.requires_grad
        assert torch.equal(blk_m.norm, blk_r.norm)
        for name in ("up", "down"):
            assert torch.equal(getattr(blk_m, name).weight, getattr(blk_r, name).weight)
    x = torch.randn(5, 64, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        assert torch.equal(m(x), ref(x))
