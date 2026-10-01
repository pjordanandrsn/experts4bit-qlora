# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K23's opt-in (``E4B_INT4_LEAN_GLUE=1``) leaves every output bit of K19's B = 16 rows where it was, asserted with
``torch.equal`` on a GPU against the real kernels.

The route folds the grouping glue into the kernels that bracket it:
- the tile table and the sorted ids come from one builder launch (``lean=True, sorted_ids=True``);
- handed the collapse's token rows, gate_up reads token row ``order[i] // top_k`` (``gather_div``), so the
  ``[T * top_k, H]`` expansion is never made;
- the down projection is stored straight into the caller's row order (``scatter=order``), with no ``index_copy_``.

None of that changes arithmetic: the table holds the same integers, and the same values are stored at other rows.
This file holds the route to it on the card it runs on, eager and inside a CUDA graph. The graph check matters
because the lean table is allocated with ``torch.empty``, so a padding slot left unwritten would read garbage from
the graph's pool.

Also pinned: the route really takes K23's builder options and K19's scatter (a spy on both), and the default
(unset) does not.

Random weights at Qwen3-30B-A3B's expert shapes, 128 experts, top-8, B = 16. Skipped when the installed kernel
package predates K23.
"""
from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the bit-equality is a property of the compiled kernels")

DEV = "cuda"
E, H, INTER, TOP_K, B = 128, 2048, 768, 8, 16          # Qwen3-30B-A3B


def _k23():
    sm = pytest.importorskip("int4_smallm")
    b32 = pytest.importorskip("int4_b32")
    if (not {"scatter", "gather_div"} <= set(inspect.signature(sm.gemm_int4_b32_grouped_smallm).parameters)
            or "lean" not in inspect.signature(b32.build_group_tiles_fused).parameters):
        pytest.skip("the installed grouped-nf4-gemm predates K23")
    return sm, b32


def _stores():
    from int4_pack_ref import pack_int4_b32
    g = torch.Generator().manual_seed(0)

    def stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk).to(DEV), torch.stack(sc).to(DEV)
    gp, gs = stack(torch.randn(E, 2 * INTER, H, generator=g) * 0.02)
    dp, ds = stack(torch.randn(E, H, INTER, generator=g) * 0.02)
    return {"gu": {"packed": gp, "scales": gs, "N": 2 * INTER, "K": H},
            "dn": {"packed": dp, "scales": ds, "N": H, "K": INTER}}


def _step(seed, tokens=False):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, H, generator=g) * 0.5).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(B)]).view(-1).to(DEV)
    return (x if tokens else x.repeat_interleave(TOP_K, 0)), ids



def _call(hr, stores, xr, ids, tokens=False):
    s, gr = hr._collapsed_grouping(B, stores)
    assert (s, gr) == (False, True)
    fg = torch.empty(0, 0, 0, dtype=torch.uint8, device=DEV)      # distinct objects: _mm names the slot by identity
    fd = torch.empty(0, 0, 0, dtype=torch.uint8, device=DEV)
    fa = torch.empty(0, 0, 0, device=DEV)
    if tokens:                                                    # the collapse's hand-over: token rows + map
        rt = torch.arange(B * TOP_K, device=DEV) // TOP_K
        return hr._fused_over_stack(None, ids, fg, fa, fd, fa, (2 * INTER, H, H, INTER), True, F.silu,
                                    singleton_groups=s, device_grouping=gr, int4_stores=stores,
                                    x_tokens=(xr, rt, TOP_K))
    return hr._fused_over_stack(xr, ids, fg, fa, fd, fa, (2 * INTER, H, H, INTER), True, F.silu,
                                singleton_groups=s, device_grouping=gr, int4_stores=stores)


def _setup(monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)     # the licensed default: K19 at T > 1
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    return hr


@needs_cuda
def test_lean_glue_is_bit_equal_to_the_default_at_b16(monkeypatch):
    sm, b32 = _k23()
    hr = _setup(monkeypatch)
    stores = _stores()
    seen = {"tiles": [], "k19": []}
    k19, tiles = sm.gemm_int4_b32_grouped_smallm, b32.build_group_tiles_fused

    def spy_k19(*a, scatter=None, gather_div=1, **kw):         # declares K23's options, as the route checks for
        seen["k19"].append(scatter is not None)
        return k19(*a, scatter=scatter, gather_div=gather_div, **kw)

    def spy_tiles(*a, lean=False, sorted_ids=False, **kw):
        seen["tiles"].append((lean, sorted_ids))
        return tiles(*a, lean=lean, sorted_ids=sorted_ids, **kw)
    monkeypatch.setattr(sm, "gemm_int4_b32_grouped_smallm", spy_k19)
    monkeypatch.setattr(b32, "build_group_tiles_fused", spy_tiles)
    for seed in (1, 2):
        xr, ids = _step(seed)
        monkeypatch.delenv("E4B_INT4_LEAN_GLUE", raising=False)
        off = _call(hr, stores, xr, ids)
        monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "1")
        on = _call(hr, stores, xr, ids)
        torch.cuda.synchronize()
        assert torch.equal(on, off), \
            f"seed {seed}: the lean glue moved an output bit (maxabs {(on.float() - off.float()).abs().max()})"
    assert seen["tiles"] == [(False, False), (True, True)] * 2, seen       # default, then K23's one launch
    assert seen["k19"] == [False, False, False, True] * 2, seen            # only the lean down call scatters


@needs_cuda
def test_lean_glue_captures_and_replays_bit_equal_to_the_eager_default(monkeypatch):
    """Captured with the lean route on one step's inputs, replayed on other steps': each replay must equal the eager
    DEFAULT route on those inputs, bit for bit. The lean table lives in the graph's pool uninitialised, so this also
    proves the builder writes every slot K19 reads."""
    _k23()
    hr = _setup(monkeypatch)
    stores = _stores()
    xr, ids = _step(3)
    x_s, id_s = xr.clone(), ids.clone()
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "1")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                                 # warm-up off the default stream (compiles)
        for _ in range(2):
            _call(hr, stores, x_s, id_s)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = _call(hr, stores, x_s, id_s)
    monkeypatch.delenv("E4B_INT4_LEAN_GLUE", raising=False)
    for seed in (4, 5):
        xn, idn = _step(seed)
        x_s.copy_(xn)
        id_s.copy_(idn)
        g.replay()
        torch.cuda.synchronize()
        eager = _call(hr, stores, xn, idn)
        torch.cuda.synchronize()
        assert torch.equal(out_s, eager), f"seed {seed}: the captured lean route differs from the eager default"


@needs_cuda
def test_lean_token_rows_are_bit_equal_to_the_expanded_default(monkeypatch):
    """The collapse's hand-over under "1": gate_up reads the [B, H] token rows through gather_div. Bit-equal to the
    default on the [B * TOP_K, H] expansion, eager and captured (replayed on new token rows)."""
    sm, _b32 = _k23()
    hr = _setup(monkeypatch)
    stores = _stores()
    seen = []
    k19 = sm.gemm_int4_b32_grouped_smallm

    def spy_k19(*a, scatter=None, gather_div=1, **kw):
        seen.append(gather_div)
        return k19(*a, scatter=scatter, gather_div=gather_div, **kw)
    monkeypatch.setattr(sm, "gemm_int4_b32_grouped_smallm", spy_k19)
    xt, ids = _step(6, tokens=True)
    monkeypatch.delenv("E4B_INT4_LEAN_GLUE", raising=False)
    off = _call(hr, stores, xt.repeat_interleave(TOP_K, 0), ids)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "1")
    seen.clear()
    on = _call(hr, stores, xt, ids, tokens=True)
    torch.cuda.synchronize()
    assert seen == [TOP_K, 1], seen                                # gate_up on token rows; down on the epilogue rows
    assert torch.equal(on, off), f"maxabs {(on.float() - off.float()).abs().max()}"
    x_s, id_s = xt.clone(), ids.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _call(hr, stores, x_s, id_s, tokens=True)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = _call(hr, stores, x_s, id_s, tokens=True)
    monkeypatch.delenv("E4B_INT4_LEAN_GLUE", raising=False)
    for seed in (7, 8):
        xn, idn = _step(seed, tokens=True)
        x_s.copy_(xn)
        id_s.copy_(idn)
        g.replay()
        torch.cuda.synchronize()
        eager = _call(hr, stores, xn.repeat_interleave(TOP_K, 0), idn)
        torch.cuda.synchronize()
        assert torch.equal(out_s, eager), f"seed {seed}: the captured token-row route differs from the eager default"
