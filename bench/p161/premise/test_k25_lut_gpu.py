# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_K25_LUT=prmt`` (K39's byte-permute decode, grouped-nf4-gemm#569) gives the served route the SAME BITS as the
default select tree, asserted with ``torch.equal`` on a GPU through the route's own call
(``hot_residency._fused_over_stack``) against the real kernel. This is the BITWISE tier lane P158's served read relies
on: the knob may move time, never a token.

Pinned, at Granite-3.1-3B-A800M's expert shapes, at Qwen3-30B-A3B's (the served read's model, with fewer experts to
keep the quantisation quick) and at "tail-k" (K not a multiple of c1's KCs):
- a B = 16 step under the lean glue (the default), and with the expanded rows: tree == prmt;
- T == 1 under ``E4B_NF4_GROUPED_SMALLM=1``: tree == prmt;
- the kernel really receives each decode (a spy on K25's keyword ``lut``);
- a B = 16 step captured in a CUDA graph under prmt replays bit-equal to the eager tree on new inputs;
- ``E4B_K25_PLAN=c1`` (lane P160's plan, opt-in): tree == prmt == prmt+c1 at B = 16 (both glues) and T == 1, each
  projection on its own block shape, and a c1 step captured in a graph replays bit-equal to the eager tree.

Skipped without a GPU, or with a kernel package that lacks K25 or K39's decode.
"""
from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode's bits are the compiled kernel's")

DEV = "cuda"
TOP_K, B = 8, 16
#: (experts, hidden, intermediate). "tail-k" divides by neither of c1's KCs (1088 % 256, 320 % 128), so the kernel's
#: masked last chunk is read: an opt-in on another family's shapes still gives the tree's bits.
SHAPES = {"granite": (40, 1536, 512), "qwen3-30b-a3b": (16, 2048, 768), "tail-k": (8, 1088, 320)}


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
    """Record the decode each K25 call receives, calling the real kernel ``k25`` (never a previous spy). It keeps
    K25's signature (``functools.wraps``): the route reads it to decide the lean glue (``_lean_glue_supported``)."""
    seen = []

    @functools.wraps(k25)
    def spy(*a, **kw):
        seen.append(kw.get("lut"))
        return k25(*a, **kw)
    monkeypatch.setattr(sm, "gemm_nf4_grouped_smallm", spy)
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


# E4B_K25_PLAN=c1 (opt-in, lane P160): the plan moves time, never a bit. tree == prmt == prmt+c1 through the served call,
# with each projection on its own block shape.
C1_SEEN = [(64, 256, 4, 2), (128, 128, 8, 3)]
TODAY_SEEN = [(32, 64, 4, 3)] * 2


def _setup_plan(monkeypatch, lut, plan, lean):
    hr = _setup(monkeypatch, lut, lean)
    monkeypatch.setenv("E4B_K25_PLAN", plan)
    return hr


def _spy_plan(monkeypatch, sm, k25):
    """As :func:`_spy`, recording each call's ``(lut, (block_n, kc, warps, stages))``."""
    seen = []

    @functools.wraps(k25)
    def spy(*a, **kw):
        seen.append((kw.get("lut"), tuple(kw.get(k) for k in ("block_n", "kc", "warps", "stages"))))
        return k25(*a, **kw)
    monkeypatch.setattr(sm, "gemm_nf4_grouped_smallm", spy)
    return seen



ARMS = [("tree", "today", TODAY_SEEN), ("prmt", "today", TODAY_SEEN), ("prmt", "c1", C1_SEEN)]


@needs_cuda
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("lean", ["1", "0"], ids=["lean-glue", "expanded-rows"])
def test_a_b16_step_has_the_same_bits_under_c1_as_under_prmt_and_the_tree(monkeypatch, shape, lean):
    sm = _k25_with_prmt()
    k25 = sm.gemm_nf4_grouped_smallm
    st = _stacks(shape)
    x, ids = _step(shape, 1)
    rt = torch.arange(B * TOP_K, device=DEV) // TOP_K
    outs = {}
    for lut, plan, want in ARMS:
        hr = _setup_plan(monkeypatch, lut, plan, lean)
        seen = _spy_plan(monkeypatch, sm, k25)
        if lean == "1":
            outs[lut, plan] = _call(hr, shape, st, B, None, ids, x_tokens=(x, rt, TOP_K))
        else:
            outs[lut, plan] = _call(hr, shape, st, B, x.index_select(0, rt), ids)
        torch.cuda.synchronize()
        assert seen == [(lut, w) for w in want], (lut, plan, seen)       # gate_up's shape, then down's
    ref = outs["tree", "today"]
    for k, o in outs.items():
        assert torch.equal(o, ref), f"{shape} {lean} {k}: maxabs {(o.float() - ref.float()).abs().max()}"


@needs_cuda
@pytest.mark.parametrize("shape", list(SHAPES))
def test_t1_has_the_same_bits_under_c1_as_under_the_tree(monkeypatch, shape):
    sm = _k25_with_prmt()
    k25 = sm.gemm_nf4_grouped_smallm
    st = _stacks(shape)
    x, ids = _step(shape, 2)
    xr, ids1 = x[:1].repeat_interleave(TOP_K, 0), ids[:TOP_K]
    outs = {}
    for lut, plan, want in (ARMS[0], ARMS[2]):
        hr = _setup_plan(monkeypatch, lut, plan, "1")
        seen = _spy_plan(monkeypatch, sm, k25)
        outs[plan] = _call(hr, shape, st, 1, xr, ids1)
        torch.cuda.synchronize()
        assert seen == [(lut, w) for w in want], (lut, plan, seen)
    assert torch.equal(outs["c1"], outs["today"])


@needs_cuda
def test_a_b16_step_captured_under_c1_replays_bit_equal_to_the_eager_tree(monkeypatch):
    """The served decode loop is a captured graph: c1 must capture with no host sync, and a replay on NEW inputs must
    equal the tree's eager call at today's plan, to the bit."""
    _k25_with_prmt()
    shape = "qwen3-30b-a3b"
    st = _stacks(shape)
    x, ids = _step(shape, 3)
    rt = torch.arange(B * TOP_K, device=DEV) // TOP_K
    x_s, id_s = x.clone(), ids.clone()
    hr = _setup_plan(monkeypatch, "prmt", "c1", "1")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                                 # warm-up off the default stream (compiles c1's shapes)
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
        hr = _setup_plan(monkeypatch, "tree", "today", "1")
        eager = _call(hr, shape, st, B, None, idn, x_tokens=(xn, rt, TOP_K))
        torch.cuda.synchronize()
        assert torch.equal(out_s, eager), f"seed {seed}: the captured c1 step differs from the eager tree"
        hr = _setup_plan(monkeypatch, "prmt", "c1", "1")
