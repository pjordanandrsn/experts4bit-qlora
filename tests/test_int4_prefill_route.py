# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_PREFILL (e4b#916) routes a uniform-int4 store's host-grouped T > 1 calls -- every prefill chunk with
DEVICE_GROUPING off, so every max_seqs == 1 server.

- ``auto`` (the default since lane P102): ``k19`` where K19 can run, else ``loop``;
- ``loop``: one ``dequant_int4_ref`` per routed expert per projection, as before;
- ``batched``: the same bf16 weights decoded in slices, so the output is BIT-IDENTICAL to ``loop`` and
  ``dequant_int4_ref`` is never called;
- ``k19`` / ``mtile``: ``_collapsed_grouping`` hands the call device grouping; ``k19`` serves it with K19 at every row
  count (refused without K19), ``mtile`` with the device-grouped int4 M-tile GEMM.

Pinned here on CPU (the host-grouped branch is pure torch; the device-grouped GEMM is stubbed to a pure-torch oracle,
as in test_int4_device_grouping.py):
- the knob's default, values and refusal;
- ``_collapsed_grouping`` under each value, for every store kind and T == 1;
- ``_dequant_int4_bf16`` bitwise against the reference;
- batched == loop bitwise through ``_fused_over_stack``, over more experts than one slice and under DECODE_A16's
  singleton rows, with the reference decode's call counts;
- ``k19`` reaching K19 and ``mtile`` the M-tile GEMM at prefill rows, neither running the host loop.

The kernels' own arithmetic on a GPU is tests/test_int4_prefill_route_gpu.py's.
"""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("nf4_grouped")
pytest.importorskip("int4_pack_ref")

import int4_pack_ref  # noqa: E402
from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

INTER, H = 32, 64


def _stores(E, seed=11):
    torch.manual_seed(seed)

    def pack_stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk), torch.stack(sc)
    gu_p, gu_s = pack_stack(torch.randn(E, 2 * INTER, H) * 0.1)
    dn_p, dn_s = pack_stack(torch.randn(E, H, INTER) * 0.1)
    return {"gu": {"packed": gu_p, "scales": gu_s, "N": 2 * INTER, "K": H},
            "dn": {"packed": dn_p, "scales": dn_s, "N": H, "K": INTER}}


def _call(stores, x, ids, **kw):
    freed = [torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8), torch.empty(0), torch.empty(0)]
    return hr._fused_over_stack(x, ids, freed[0], freed[2], freed[1], freed[3], (2 * INTER, H, H, INTER), True,
                                F.silu, int4_stores=stores, **kw)


def _counting_reference(monkeypatch):
    n = [0]
    real = int4_pack_ref.dequant_int4_ref

    def counted(*a, **k):
        n[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(int4_pack_ref, "dequant_int4_ref", counted)
    return n


def test_the_knob_defaults_to_auto_and_refuses_unknown_values(monkeypatch):
    """``auto`` (the default since lane P102) is ``k19`` where K19 can run, else ``loop``."""
    monkeypatch.delenv("E4B_INT4_PREFILL", raising=False)
    monkeypatch.setattr(hr, "_K19_PREFILL_OK", [True])
    assert hr._int4_prefill_mode_env() == "k19"
    monkeypatch.setenv("E4B_INT4_PREFILL", "")
    assert hr._int4_prefill_mode_env() == "k19"
    monkeypatch.setattr(hr, "_K19_PREFILL_OK", [False])
    assert hr._int4_prefill_mode_env() == "loop"
    monkeypatch.setenv("E4B_INT4_PREFILL", "AUTO")
    assert hr._int4_prefill_mode_env() == "loop"
    for v, want in (("loop", "loop"), (" Batched ", "batched"), ("K19", "k19"), ("mtile", "mtile")):
        monkeypatch.setenv("E4B_INT4_PREFILL", v)
        assert hr._int4_prefill_mode_env() == want
    assert hr.INT4_PREFILL_ROUTES == ("loop", "batched", "k19", "mtile")
    for bad in ("1", "grouped", "default"):
        monkeypatch.setenv("E4B_INT4_PREFILL", bad)
        with pytest.raises(ValueError, match="E4B_INT4_PREFILL"):
            hr._int4_prefill_mode_env()


@pytest.mark.parametrize("mode", ["loop", "batched", "k19", "mtile"])
def test_collapsed_grouping_under_each_route(monkeypatch, mode):
    monkeypatch.setenv("E4B_INT4_PREFILL", mode)
    for k in ("E4B_INT4_GROUPED_SMALLM", "E4B_MXFP4_GROUPED_SMALLM", "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    int4 = {"gu": {}, "dn": {}}
    mx = {"kind": "mxfp4", "gu": {}, "dn": {}}
    # the int4_b32 store's prefill call: device grouping only under the device-grouped routes
    assert hr._collapsed_grouping(512, int4) == (False, mode in ("k19", "mtile"))
    # T == 1 decode, the MXFP4 store and the NF4 store (no int4 store) are untouched by the knob
    assert hr._collapsed_grouping(1, int4) == (True, False)
    assert hr._collapsed_grouping(512, mx) == (False, False)
    assert hr._collapsed_grouping(512, None) == (False, False)
    # singleton groups, when forced, stay singleton
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [True])
    assert hr._collapsed_grouping(512, int4) == (True, False)
    # with DEVICE_GROUPING on, every route is already device-grouped
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    assert hr._collapsed_grouping(512, int4) == (False, True)


def test_the_batched_decode_is_bitwise_the_reference():
    st = _stores(20)
    for slot in ("gu", "dn"):
        s = st[slot]
        ids = torch.tensor([3, 0, 19, 7, 7])
        w = hr._dequant_int4_bf16(s["packed"], s["scales"], ids, s["N"], s["K"])
        assert w.dtype == torch.bfloat16 and w.shape == (5, s["N"], s["K"])
        for j, e in enumerate(ids.tolist()):
            ref = dequant_int4_ref(s["packed"][e], s["scales"][e], s["N"], s["K"]).to(torch.bfloat16)
            assert torch.equal(w[j], ref), (slot, e)


def test_batched_is_bitwise_the_loop_without_the_reference_decode(monkeypatch):
    E = 2 * hr._INT4_PREFILL_SLICE + 5            # three slices, the last one short
    st = _stores(E)
    torch.manual_seed(3)
    R = 640
    x = torch.randn(R, H, dtype=torch.bfloat16) * 0.3
    ids = torch.randint(0, E, (R,))
    distinct = int(ids.unique().numel())
    n = _counting_reference(monkeypatch)
    monkeypatch.setenv("E4B_INT4_PREFILL", "loop")
    y_loop = _call(st, x, ids, singleton_groups=False, device_grouping=False)
    assert n[0] == 2 * distinct                    # one decode per routed expert per projection
    monkeypatch.setenv("E4B_INT4_PREFILL", "batched")
    y_bat = _call(st, x, ids, singleton_groups=False, device_grouping=False)
    assert n[0] == 2 * distinct                    # batched never calls the reference decode
    assert y_loop.dtype == y_bat.dtype == torch.bfloat16 and y_loop.shape == (R, H)
    assert torch.equal(y_loop, y_bat)


def test_batched_is_bitwise_the_loop_on_decode_a16s_singleton_rows(monkeypatch):
    st = _stores(12, seed=5)
    torch.manual_seed(4)
    x = torch.randn(8, H, dtype=torch.bfloat16) * 0.3
    ids = torch.tensor([1, 4, 4, 9, 0, 11, 1, 6])     # one token's top-8 rows, repeats allowed across tokens
    monkeypatch.setattr(hr, "DECODE_A16", [True])
    outs = {}
    for mode in ("loop", "batched"):
        monkeypatch.setenv("E4B_INT4_PREFILL", mode)
        outs[mode] = _call(st, x, ids, singleton_groups=True, device_grouping=False)
    assert torch.equal(outs["loop"], outs["batched"])


def _oracle_tiles(x, packed, scales, t_row0, t_rows, t_grp, order=None):
    """Sorted rows out; ``order`` given = gather the unsorted input first (K19's and the M-tile's contract)."""
    E_, N_, kh = packed.shape
    xs = x.index_select(0, order) if order is not None else x
    out = torch.zeros(xs.shape[0], N_, dtype=torch.bfloat16)
    for g in range(t_row0.numel()):
        rows = int(t_rows[g])
        if rows:
            r0, e = int(t_row0[g]), int(t_grp[g])
            out[r0:r0 + rows] = (xs[r0:r0 + rows].float() @ dequant_int4_ref(packed[e], scales[e], N_, kh * 2).t()).to(torch.bfloat16)
    return out


def _prefill_rows(monkeypatch, mode, E=6):
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    monkeypatch.setenv("E4B_INT4_PREFILL", mode)
    st = _stores(E, seed=7)
    torch.manual_seed(8)
    T, k = 50, 6
    R = T * k                                                     # 300 > 256: not a decode route
    singleton, device = hr._collapsed_grouping(T, st)
    assert (singleton, device) == (False, True)
    x = torch.randn(R, H, dtype=torch.bfloat16) * 0.3
    ids = torch.randint(0, E, (R,))
    return st, x, ids, R


def _loop_reference(monkeypatch, st, x, ids):
    monkeypatch.setenv("E4B_INT4_PREFILL", "loop")
    return _call(st, x, ids, singleton_groups=False, device_grouping=False).float()


def test_mtile_reaches_the_device_grouped_int4_gemm(monkeypatch):
    calls = []
    stub = types.ModuleType("int4_b32")
    stub.quant_x_rows = lambda x: (x.float(), None)

    def gemm_int4_b32_grouped_captured(xq, xs, packed, scales, t_row0, t_rows, t_grp, **kw):
        calls.append(xq.shape[0])
        return _oracle_tiles(xq, packed, scales, t_row0, t_rows, t_grp)
    stub.gemm_int4_b32_grouped_captured = gemm_int4_b32_grouped_captured
    monkeypatch.setitem(sys.modules, "int4_b32", stub)
    st, x, ids, R = _prefill_rows(monkeypatch, "mtile")
    n = _counting_reference(monkeypatch)
    y = _call(st, x, ids, singleton_groups=False, device_grouping=True)
    assert calls == [R, R], calls                                 # gate_up, then down, through the M-tile GEMM
    assert n[0] == 0                                              # the host loop never ran
    ref = _loop_reference(monkeypatch, st, x, ids)
    assert float((y.float() - ref).abs().max() / ref.abs().max()) < 0.05


def test_k19_serves_prefill_rows_under_k19_only(monkeypatch):
    calls = []
    sm = types.ModuleType("int4_smallm")

    def gemm_int4_b32_grouped_smallm(x, packed, scales, t_row0, t_rows, t_group, order=None, **kw):
        assert kw.get("scatter") is None and kw.get("gather_div", 1) == 1   # no K23 glue on prefill rows
        calls.append(("gather" if order is not None else "sorted", x.shape[0]))
        return _oracle_tiles(x, packed, scales, t_row0, t_rows, t_group, order)
    sm.gemm_int4_b32_grouped_smallm = gemm_int4_b32_grouped_smallm
    monkeypatch.setitem(sys.modules, "int4_smallm", sm)
    m = types.ModuleType("int4_b32")                              # the M-tile must NOT run under k19
    m.quant_x_rows = lambda x: (_ for _ in ()).throw(AssertionError("M-tile quantise under k19"))
    m.gemm_int4_b32_grouped_captured = lambda *a, **k: (_ for _ in ()).throw(AssertionError("M-tile under k19"))
    monkeypatch.setitem(sys.modules, "int4_b32", m)
    st, x, ids, R = _prefill_rows(monkeypatch, "k19")
    n = _counting_reference(monkeypatch)
    y = _call(st, x, ids, singleton_groups=False, device_grouping=True)
    assert calls == [("gather", R), ("sorted", R)], calls         # gate_up gathers the unsorted rows; down is sorted
    assert n[0] == 0
    ref = _loop_reference(monkeypatch, st, x, ids)
    assert float((y.float() - ref).abs().max() / ref.abs().max()) < 0.05
    # without K19 in the kernel package the route is refused, never silently the M-tile
    monkeypatch.setitem(sys.modules, "int4_smallm", types.ModuleType("int4_smallm"))
    monkeypatch.setenv("E4B_INT4_PREFILL", "k19")
    with pytest.raises(RuntimeError, match="E4B_INT4_PREFILL=k19 needs"):
        _call(st, x, ids, singleton_groups=False, device_grouping=True)


@pytest.mark.parametrize("mode", ["loop", "batched", "k19", "mtile"])
def test_device_grouped_prefill_rows_keep_the_m_tile_unless_k19(monkeypatch, mode):
    """With DEVICE_GROUPING on (a max_seqs > 1 server), prefill rows above 256 have always taken the M-tile; only
    ``k19`` moves them."""
    seen = []
    b32 = types.ModuleType("int4_b32")
    b32.quant_x_rows = lambda x: (x.float(), None)

    def mtile(xq, xs, packed, scales, t_row0, t_rows, t_grp, **kw):
        seen.append("mtile")
        return _oracle_tiles(xq, packed, scales, t_row0, t_rows, t_grp)
    b32.gemm_int4_b32_grouped_captured = mtile
    monkeypatch.setitem(sys.modules, "int4_b32", b32)
    sm = types.ModuleType("int4_smallm")

    def k19(x, packed, scales, t_row0, t_rows, t_group, order=None, **kw):
        seen.append("k19")
        return _oracle_tiles(x, packed, scales, t_row0, t_rows, t_group, order)
    sm.gemm_int4_b32_grouped_smallm = k19
    monkeypatch.setitem(sys.modules, "int4_smallm", sm)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    monkeypatch.setenv("E4B_INT4_PREFILL", mode)
    st = _stores(6, seed=2)
    torch.manual_seed(1)
    x = torch.randn(300, H, dtype=torch.bfloat16) * 0.3
    ids = torch.randint(0, 6, (300,))
    assert hr._collapsed_grouping(50, st) == (False, True)
    _call(st, x, ids, singleton_groups=False, device_grouping=True)
    assert seen == (["k19", "k19"] if mode == "k19" else ["mtile", "mtile"]), seen
