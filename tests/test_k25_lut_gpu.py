# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_K25_LUT=prmt`` (K39's byte-permute decode, grouped-nf4-gemm#569) gives the served route the SAME BITS as the
default select tree, asserted with ``torch.equal`` on a GPU through the route's own call
(``hot_residency._fused_over_stack``) against the real kernel. This is the BITWISE tier lane P158's served read relies
on: the knob may move time, never a token.

Pinned, at Granite-3.1-3B-A800M's expert shapes and at Qwen3-30B-A3B's (the served read's model, with fewer experts to
keep the quantisation quick):
- a B = 16 step under the lean glue (the default), and with the expanded rows: tree == prmt;
- T == 1 under ``E4B_NF4_GROUPED_SMALLM=1``: tree == prmt;
- the kernel really receives each decode (a spy on K25's keyword ``lut``);
- a B = 16 step captured in a CUDA graph under prmt replays bit-equal to the eager tree on new inputs.

Skipped without a GPU, or with a kernel package that lacks K25 or K39's decode.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode's bits are the compiled kernel's")

DEV = "cuda"
TOP_K, B = 8, 16
SHAPES = {"granite": (40, 1536, 512), "qwen3-30b-a3b": (16, 2048, 768)}   # (experts, hidden, intermediate)


def _k25_with_prmt():
    sm = pytest.importorskip("nf4_smallm")
    if not hasattr(sm, "gemm_nf4_grouped_smallm"):
        pytest.skip("the installed grouped-nf4-gemm lacks K25")
    if "prmt" not in (getattr(sm, "_LUT_MODES", None) or {}):
        pytest.skip("the installed grouped-nf4-gemm lacks K39's decode (K25's lut='prmt', grouped-nf4-gemm#569)")
    return sm


_STACKS: dict = {}


def _stacks(shape):
    """Quantised once per shape per module."""
    if shape not in _STACKS:
        from nf4_pack_ref import quantize_pack_nf4
        E, H, INTER = SHAPES[shape]
        g = torch.Generator().manual_seed(0)

        def stack(N, K):
            ps, as_ = zip(*[quantize_pack_nf4(torch.randn(N, K, generator=g) * 0.02) for _ in range(E)])
            return torch.stack(ps).to(DEV), torch.stack(as_).to(DEV)
        _STACKS[shape] = (*stack(2 * INTER, H), *stack(H, INTER))
    return _STACKS[shape]


def _step(shape, seed):
    E, H, _ = SHAPES[shape]
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, H, generator=g) * 0.5).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(B)]).view(-1).to(DEV)
    return x, ids


def _setup(monkeypatch, lut, lean):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", lean)
    monkeypatch.setenv("E4B_K25_LUT", lut)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    return hr


def _call(hr, shape, st, T, xr, ids, x_tokens=None):
    _, H, INTER = SHAPES[shape]
    s, gr = hr._collapsed_grouping(T, None)
    return hr._fused_over_stack(xr, ids, *st, (2 * INTER, H, H, INTER), True, F.silu,
                                singleton_groups=s, device_grouping=gr, int4_stores=None, x_tokens=x_tokens)


def _spy(monkeypatch, sm, k25):
    """Record the decode each K25 call receives, calling the real kernel ``k25`` (never a previous spy)."""
    seen = []
    monkeypatch.setattr(sm, "gemm_nf4_grouped_smallm", lambda *a, **kw: (seen.append(kw.get("lut")), k25(*a, **kw))[1])
    return seen


@needs_cuda
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("lean", ["1", "0"], ids=["lean-glue", "expanded-rows"])
def test_a_b16_step_has_the_same_bits_under_prmt_as_under_the_tree(monkeypatch, shape, lean):
    sm = _k25_with_prmt()
    k25 = sm.gemm_nf4_grouped_smallm
    st = _stacks(shape)
    x, ids = _step(shape, 1)
    rt = torch.arange(B * TOP_K, device=DEV) // TOP_K
    outs = {}
    for lut in ("tree", "prmt"):
        hr = _setup(monkeypatch, lut, lean)
        seen = _spy(monkeypatch, sm, k25)
        if lean == "1":
            outs[lut] = _call(hr, shape, st, B, None, ids, x_tokens=(x, rt, TOP_K))
        else:
            outs[lut] = _call(hr, shape, st, B, x.index_select(0, rt), ids)
        torch.cuda.synchronize()
        assert seen == [lut, lut], (lut, seen)                       # gate_up and down, each on the asked decode
    assert torch.equal(outs["prmt"], outs["tree"]), \
        f"{shape} {lean}: maxabs {(outs['prmt'].float() - outs['tree'].float()).abs().max()}"


@needs_cuda
@pytest.mark.parametrize("shape", list(SHAPES))
def test_t1_has_the_same_bits_under_prmt_as_under_the_tree(monkeypatch, shape):
    sm = _k25_with_prmt()
    k25 = sm.gemm_nf4_grouped_smallm
    st = _stacks(shape)
    x, ids = _step(shape, 2)
    xr, ids1 = x[:1].repeat_interleave(TOP_K, 0), ids[:TOP_K]
    outs = {}
    for lut in ("tree", "prmt"):
        hr = _setup(monkeypatch, lut, "1")
        seen = _spy(monkeypatch, sm, k25)
        outs[lut] = _call(hr, shape, st, 1, xr, ids1)
        torch.cuda.synchronize()
        assert seen == [lut, lut], (lut, seen)
    assert torch.equal(outs["prmt"], outs["tree"])


@needs_cuda
def test_a_b16_step_captured_under_prmt_replays_bit_equal_to_the_eager_tree(monkeypatch):
    """The served decode loop is a captured graph: prmt must capture with no host sync, and a replay on NEW inputs
    must equal the default tree's eager call on those inputs, to the bit."""
    _k25_with_prmt()
    shape = "qwen3-30b-a3b"
    st = _stacks(shape)
    x, ids = _step(shape, 3)
    rt = torch.arange(B * TOP_K, device=DEV) // TOP_K
    x_s, id_s = x.clone(), ids.clone()
    hr = _setup(monkeypatch, "prmt", "1")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                                 # warm-up off the default stream (compiles K25)
        for _ in range(2):
            _call(hr, shape, st, B, None, id_s, x_tokens=(x_s, rt, TOP_K))
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = _call(hr, shape, st, B, None, id_s, x_tokens=(x_s, rt, TOP_K))
    for seed in (5, 12):
        xn, idn = _step(shape, seed)
        x_s.copy_(xn)
        id_s.copy_(idn)
        g.replay()
        torch.cuda.synchronize()
        hr = _setup(monkeypatch, "tree", "1")
        eager = _call(hr, shape, st, B, None, idn, x_tokens=(xn, rt, TOP_K))
        torch.cuda.synchronize()
        assert torch.equal(out_s, eager), f"seed {seed}: the captured prmt step differs from the eager tree"
        hr = _setup(monkeypatch, "prmt", "1")
