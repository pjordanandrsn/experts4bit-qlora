"""Offloaded TRAINING through bitsandbytes ``Linear4bit`` actually frees the evicted weights (CUDA + bitsandbytes).

bitsandbytes 0.50's ``MatMul4Bit`` keeps the frozen packed weight on ``ctx`` (not ``save_for_backward``) whenever the
input needs grad, so checkpointing cannot drop it and every layer's weight stayed alive from forward to backward:
dense offload saved no VRAM in training, and DQ3's streamed arm OOMed above the resident arm (dq3-5090-3). Dense offload
now routes each offloaded ``Linear4bit``'s grad-mode matmul through a late-bound Function that keeps the module and
reads the weight bound at backward time. bnb 0.50.2 behaviour, worked around locally; not reported upstream.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

bnb = pytest.importorskip("bitsandbytes")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora.engines import dense_offload as do  # noqa: E402
from experts4bit_qlora.engines.dense_offload import _DenseOffload, enable_dense_offload  # noqa: E402

H, INTER, NL = 1024, 4096, 6          # each projection packs to 2 MiB (NF4), over MIN_BYTES


def _q(i, o):
    lin = nn.Linear(i, o, bias=False)
    q = bnb.nn.Linear4bit(i, o, bias=False, compute_dtype=torch.bfloat16, compress_statistics=True,
                          quant_type="nf4", device="cpu")
    q.weight = bnb.nn.Params4bit(lin.weight.data.to(torch.bfloat16), requires_grad=False, compress_statistics=True,
                                 quant_type="nf4", blocksize=64)
    return q


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = _q(H, INTER)
        self.down = _q(INTER, H)
        self.norm = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))     # trainable, small: stays resident

    def forward(self, x):
        return x + self.down(torch.relu(self.up(x * self.norm)))


class Toy(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(NL))
        self.ckpt = ckpt

    def forward(self, x):
        for lay in self.layers:
            x = checkpoint(lay, x, use_reentrant=False) if self.ckpt else lay(x)
        return x


@pytest.fixture(autouse=True)
def _clean_class_state():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()


def _model(ckpt, seed=3):
    torch.manual_seed(seed)
    m = Toy(ckpt).to("cuda")             # quantizes on the move
    m.train()
    return m


def _offload(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return enable_dense_offload(m, "cuda", pin=True, prefetch=False, **kw)


def _layer_bytes(m):
    lay = m.layers[0]
    return sum(p.numel() * p.element_size() for p in (lay.up.weight, lay.down.weight))


def _x(seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(16, 256, H, generator=g).to("cuda", torch.bfloat16)


def _step(m):
    x = _x().requires_grad_(True)
    loss = m(x).float().pow(2).mean()
    loss.backward()
    return loss.detach(), x.grad.detach().clone(), [lay.norm.grad.detach().clone() for lay in m.layers]


@pytest.mark.parametrize("ckpt", [True, False])
@pytest.mark.parametrize("train_prefetch", [False, True])
def test_offloaded_training_is_bitwise_identical_to_resident(ckpt, train_prefetch):
    ref = _step(_model(ckpt))
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    m = _model(ckpt)
    hs = _offload(m, train_prefetch=train_prefetch)
    assert all(getattr(lay.up, "_dense_offload_late_bound", False) for lay in m.layers), "not routed"
    for _ in range(2):                      # twice: the second step starts from a warmed, partly resident state
        for p in m.parameters():
            p.grad = None
        got = _step(m)
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
        assert all(torch.equal(a, b) for a, b in zip(ref[2], got[2]))
    assert hs


def _held_across_forward(m):
    """Bytes the autograd graph holds at the END of a forward: ``memory_allocated`` after minus before, each read after
    ``torch.cuda.synchronize()``. Static state (resident weights, staged layers carried over, the allocator's one-time
    allocations) cancels, so two models are compared by what their forwards keep alive -- not by baselines that the
    first version of this test confounded (a route-disabled control 'saved' more than every weight in the model)."""
    x = _x().requires_grad_(True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    y = m(x)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before
    y.float().pow(2).mean().backward()
    return held


def _measure(*, offload: bool, train_prefetch: bool = False, route: bool = True, monkeypatch=None, steps: int = 1):
    """Build, optionally offload (with the late-bound route on or off), take ONE warm step, then return the held bytes
    of each of ``steps`` further steps and the per-layer packed-weight bytes."""
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    m = _model(ckpt=True)
    if offload:
        if route:
            _offload(m, train_prefetch=train_prefetch)
        else:
            with monkeypatch.context() as mp:
                mp.setattr(do, "_install_late_bound_backward", lambda handles: 0)
                _offload(m, train_prefetch=train_prefetch)
    _held_across_forward(m)                                   # warm, identical on every side
    out = [_held_across_forward(m) for _ in range(steps)]
    lb = _layer_bytes(m)
    del m
    torch.cuda.empty_cache()
    return out, lb


@pytest.mark.parametrize("train_prefetch", [False, True])
def test_the_late_bound_route_frees_the_evicted_weights(train_prefetch, monkeypatch):
    """The quantity the lane exists to change, with its control in the SAME offloaded configuration: with the route
    off (stock bnb MatMul4Bit keeps each weight on ctx) the forward holds every streamed layer; with it on, at most the
    two the schedule keeps bound. The difference must be at least (L - 2) layers."""
    (on,), lb = _measure(offload=True, train_prefetch=train_prefetch)
    (off,), _ = _measure(offload=True, train_prefetch=train_prefetch, route=False, monkeypatch=monkeypatch)
    assert off - on >= (NL - 2) * lb - lb // 2, (off - on, (NL - 2) * lb, on, off)


@pytest.mark.parametrize("train_prefetch", [False, True])
def test_the_routed_forward_holds_no_more_than_two_layers_over_resident(train_prefetch):
    (res,), lb = _measure(offload=False)
    (on,), _ = _measure(offload=True, train_prefetch=train_prefetch)
    assert on <= res + 2 * lb + lb // 2, (on - res, 2 * lb, on, res)


def test_inference_takes_the_stock_forward_and_matches():
    m_ref = _model(ckpt=False)
    m_ref.eval()
    with torch.no_grad():
        ref = m_ref(_x())
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    m = _model(ckpt=False)
    _offload(m)
    m.eval()
    with torch.no_grad():
        got = m(_x())
    assert torch.equal(ref, got)


def test_the_mirrored_bnb_sources_are_the_pinned_ones():
    """The Function mirrors bnb 0.50.2's MatMul4Bit forward/backward, matmul_4bit and Linear4bit.forward. If an
    installed bnb's source differs, the route must not be used (the next test); on the pinned release it must match."""
    if bnb.__version__ != "0.50.2":
        pytest.skip(f"pins are for bitsandbytes 0.50.2, installed {bnb.__version__}")
    assert do._bnb_mirror_mismatches() == []


def test_a_source_mismatch_keeps_stock_bnb_with_a_warning(monkeypatch):
    monkeypatch.setitem(do._BNB_MIRRORED_SOURCES, "MatMul4Bit.backward", "0" * 64)
    m = _model(ckpt=True)
    with pytest.warns(UserWarning, match="keeping stock bnb"):
        enable_dense_offload(m, "cuda", pin=True, prefetch=False)
    assert not any(getattr(lay.up, "_dense_offload_late_bound", False) for lay in m.layers)
