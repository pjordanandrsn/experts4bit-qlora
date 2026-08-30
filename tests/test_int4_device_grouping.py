# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The int4 store must win the device-grouping dispatch.

Found live on the first composed bv3+int4 arm: the outer _mm chain
selected the NF4 captured path whenever device_grouping was true, so
the int4 sub-branch never ran -- the captured GEMM took the FREED
0-size stacks, read E as 0, and returned a silent [R, 0] that
detonated far away as a size-0 view. These tests pin the dispatch and
the wiring with the GEMM stubbed to a pure-torch reference, so they
run wherever the NF4 grouping helpers import (linux CI; macOS skips on
the triton import).
"""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("nf4_grouped")
pytest.importorskip("int4_pack_ref")   # gnf4 builds predating the int4 lane

from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

E, N1, K1, R = 4, 32, 64, 24     # gate_up N=2*N1? keep simple: gu N=64,K=64; dn N=64,K=32


def _stub_int4_b32(monkeypatch):
    """int4_b32 stub: quantise is identity-ish (fp32 carried), the GEMM
    dequantises the real packed bytes and matmuls -- a pure-torch oracle
    with the SAME tile-table contract (sorted rows in, sorted rows out)."""
    stub = types.ModuleType("int4_b32")

    def quant_x_rows(x):
        return x.float(), None            # (xq, xs) placeholder pair

    def gemm_int4_b32_grouped_captured(xq, xs, packed, scales,
                                       t_row0, t_rows, t_grp, **kw):
        E_, N_, kh = packed.shape
        K_ = kh * 2
        out = torch.zeros(xq.shape[0], N_, dtype=torch.bfloat16)
        for g in range(t_row0.numel()):
            rows = int(t_rows[g])
            if rows == 0:
                continue
            r0, e = int(t_row0[g]), int(t_grp[g])
            w = dequant_int4_ref(packed[e], scales[e], N_, K_)
            out[r0:r0 + rows] = (xq[r0:r0 + rows] @ w.t()).to(torch.bfloat16)
        return out

    stub.quant_x_rows = quant_x_rows
    stub.gemm_int4_b32_grouped_captured = gemm_int4_b32_grouped_captured
    monkeypatch.setitem(sys.modules, "int4_b32", stub)
    return stub


def _stores(gu_w, dn_w):
    def pack_stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk), torch.stack(sc)
    gu_p, gu_s = pack_stack(gu_w)
    dn_p, dn_s = pack_stack(dn_w)
    return {"gu": {"packed": gu_p, "scales": gu_s,
                   "N": gu_w.shape[1], "K": gu_w.shape[2]},
            "dn": {"packed": dn_p, "scales": dn_s,
                   "N": dn_w.shape[1], "K": dn_w.shape[2]}}


def test_int4_wins_device_grouping_dispatch(monkeypatch):
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    _stub_int4_b32(monkeypatch)
    torch.manual_seed(9)
    inter = 32
    gu_w = torch.randn(E, 2 * inter, K1) * 0.1        # [E, 64, 64]
    dn_w = torch.randn(E, K1, inter) * 0.1            # [E, 64, 32]
    stores = _stores(gu_w, dn_w)
    # DISTINCT freed sentinels per slot: the identity dispatch is
    # `pk is gu_p`, and production frees each attr to its own 0-tensor
    freed_gu = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_dn = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_a = torch.empty(0, 0, 0)
    x = torch.randn(R, K1, dtype=torch.bfloat16) * 0.2
    ids = torch.randint(0, E, (R,))

    out = _fused_over_stack(
        x, ids, freed_gu, freed_a, freed_dn, freed_a,
        (2 * inter, K1, K1, inter), True, F.silu,
        device_grouping=True, int4_stores=stores)
    assert out.shape == (R, K1)
    assert out.numel() > 0 and torch.isfinite(out.float()).all()

    # oracle: per-row dequant matmul + SwiGLU in the caller's row order
    ref = torch.empty(R, K1)
    for i in range(R):
        e = int(ids[i])
        w_gu = dequant_int4_ref(stores["gu"]["packed"][e],
                                stores["gu"]["scales"][e], 2 * inter, K1)
        w_dn = dequant_int4_ref(stores["dn"]["packed"][e],
                                stores["dn"]["scales"][e], K1, inter)
        gu = x[i].float() @ w_gu.t()
        g, u = gu.chunk(2)
        ref[i] = (F.silu(g) * u) @ w_dn.t()
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 0.05, float(rel)


def test_nf4_captured_refuses_freed_stacks(monkeypatch):
    """Belt for the braces: if the dispatch ever regresses, the NF4
    captured path must RAISE on freed stacks, not return [R, 0]."""
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    torch.manual_seed(9)
    freed_gu = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_dn = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_a = torch.empty(0, 0, 0)
    x = torch.randn(R, K1, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (R,))
    with pytest.raises(RuntimeError, match="freed"):
        _fused_over_stack(
            x, ids, freed_gu, freed_a, freed_dn, freed_a,
            (64, K1, K1, 32), True, F.silu,
            device_grouping=True, int4_stores=None)
