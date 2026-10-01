# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_GROUPED_SMALLM=1 routes int4 decode rows through K19 (grouped-nf4-gemm#419), and only when asked.

K19 is `int4_smallm.gemm_int4_b32_grouped_smallm`: K16's arithmetic over the device tile table, gate_up's gather
folded into the kernel, outputs in sorted order. These tests stub it with a pure-torch oracle that obeys the same
contract (sorted rows out; `order` given = gather the unsorted input, None = input already sorted) and pin:
  - opted in: both projections go through K19 (gate_up WITH `order`, down WITHOUT), the GEMV never runs, the tile
    table is built with 16-row tiles, and the output equals the per-row dequant oracle in the caller's row order;
  - not opted in: the GEMV route is unchanged and K19 is never called;
  - opted in with a kernel side that lacks K19: a RuntimeError naming the requirement, not a silent GEMV fallback;
  - prefill rows (R > 256): K19 is not taken (the opt-in is a decode route).
Like test_int4_device_grouping.py, these run wherever the NF4 grouping helpers import (linux CI).
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

from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

E, K1, INTER = 4, 64, 32


def _stores():
    torch.manual_seed(9)
    gu_w = torch.randn(E, 2 * INTER, K1) * 0.1
    dn_w = torch.randn(E, K1, INTER) * 0.1

    def pack_stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk), torch.stack(sc)
    gu_p, gu_s = pack_stack(gu_w)
    dn_p, dn_s = pack_stack(dn_w)
    return {"gu": {"packed": gu_p, "scales": gu_s, "N": 2 * INTER, "K": K1},
            "dn": {"packed": dn_p, "scales": dn_s, "N": K1, "K": INTER}}


def _oracle(x, ids, stores):
    ref = torch.empty(x.shape[0], K1)
    for i in range(x.shape[0]):
        e = int(ids[i])
        w_gu = dequant_int4_ref(stores["gu"]["packed"][e], stores["gu"]["scales"][e], 2 * INTER, K1)
        w_dn = dequant_int4_ref(stores["dn"]["packed"][e], stores["dn"]["scales"][e], K1, INTER)
        g, u = (x[i].float() @ w_gu.t()).chunk(2)
        ref[i] = (F.silu(g) * u) @ w_dn.t()
    return ref


def _install_stubs(monkeypatch, with_k19=True):
    calls = {"k19": [], "gemv": 0, "tiles": []}
    b32 = types.ModuleType("int4_b32")

    def quant_x_rows(x):
        return x.float(), None

    def gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=None):
        calls["gemv"] += 1
        out = torch.zeros(xq.shape[0], N, dtype=torch.bfloat16)
        for i in range(xq.shape[0]):
            e = int(eids[i])
            out[i] = (xq[i] @ dequant_int4_ref(packed[e], scales[e], N, K).t()).to(torch.bfloat16)
        return out
    b32.quant_x_rows = quant_x_rows
    b32.gemv_int4_b32 = gemv_int4_b32
    monkeypatch.setitem(sys.modules, "int4_b32", b32)

    import nf4_grouped
    real_builder = nf4_grouped.build_group_tiles_device

    def recording_builder(ids, n_exp, block_m):
        calls["tiles"].append(block_m)
        return real_builder(ids, n_exp, block_m)
    monkeypatch.setattr(nf4_grouped, "build_group_tiles_device", recording_builder)

    sm = types.ModuleType("int4_smallm")
    if with_k19:
        def gemm_int4_b32_grouped_smallm(x, packed, scales, t_row0, t_rows, t_group, order=None, **kw):
            calls["k19"].append("gather" if order is not None else "sorted")
            E_, N_, kh = packed.shape
            xs = x.index_select(0, order) if order is not None else x
            out = torch.zeros(x.shape[0], N_, dtype=torch.bfloat16)
            for g in range(t_row0.numel()):
                rows = int(t_rows[g])
                if rows == 0:
                    continue
                assert rows <= 16, "K19 serves 16-row tiles"
                r0, e = int(t_row0[g]), int(t_group[g])
                w = dequant_int4_ref(packed[e], scales[e], N_, kh * 2)
                out[r0:r0 + rows] = (xs[r0:r0 + rows].float() @ w.t()).to(torch.bfloat16)
            return out
        sm.gemm_int4_b32_grouped_smallm = gemm_int4_b32_grouped_smallm
    monkeypatch.setitem(sys.modules, "int4_smallm", sm)
    return calls


def _run(R, monkeypatch):
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    stores = _stores()
    freed_gu = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_dn = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_a = torch.empty(0, 0, 0)
    x = torch.randn(R, K1, dtype=torch.bfloat16) * 0.2
    ids = torch.randint(0, E, (R,))
    out = _fused_over_stack(x, ids, freed_gu, freed_a, freed_dn, freed_a, (2 * INTER, K1, K1, INTER), True, F.silu,
                            device_grouping=True, int4_stores=stores)
    return out, _oracle(x, ids, stores)


def test_opted_in_decode_goes_through_k19_and_matches_the_oracle(monkeypatch):
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24, monkeypatch)
    assert calls["k19"] == ["gather", "sorted"], calls       # gate_up gathers in-kernel; down reads sorted rows
    assert calls["gemv"] == 0, "the GEMV must not run when K19 is asked for"
    assert calls["tiles"] == [16], calls                      # one tile table, K19's 16-row tiles
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 0.05, float(rel)


def test_not_opted_in_keeps_the_gemv_route(monkeypatch):
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24, monkeypatch)
    assert calls["k19"] == [] and calls["gemv"] == 2 and calls["tiles"] == [], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_opted_in_without_k19_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    _install_stubs(monkeypatch, with_k19=False)
    with pytest.raises(RuntimeError, match="needs grouped-nf4-gemm with K19"):
        _run(24, monkeypatch)


def test_prefill_rows_do_not_take_k19(monkeypatch):
    """R > 256 is the prefill route (K14's grouped GEMM): the opt-in names a DECODE route and does not touch it."""
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)

    def gemm_int4_b32_grouped_captured(xq, xs, packed, scales, t_row0, t_rows, t_grp, **kw):
        E_, N_, kh = packed.shape
        out = torch.zeros(xq.shape[0], N_, dtype=torch.bfloat16)
        for g in range(t_row0.numel()):
            rows = int(t_rows[g])
            if rows:
                r0, e = int(t_row0[g]), int(t_grp[g])
                out[r0:r0 + rows] = (xq[r0:r0 + rows] @ dequant_int4_ref(packed[e], scales[e], N_, kh * 2).t()).to(torch.bfloat16)
        return out
    sys.modules["int4_b32"].gemm_int4_b32_grouped_captured = gemm_int4_b32_grouped_captured
    sys.modules["int4_b32"].quant_x_rows_gathered = lambda x, order: (x.float().index_select(0, order), None)
    out, ref = _run(300, monkeypatch)
    assert calls["k19"] == [], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05
