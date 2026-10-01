# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_MXFP4_GROUPED_SMALLM=1 routes the native MXFP4 store's device-grouped decode rows through K21
(grouped-nf4-gemm#422, `mxfp4_grouped.gemm_mxfp4_grouped_smallm`), opt-in.

K21 is K19's kernel on the MXFP4 bytes: the 16-row device tile table, gate_up's gather folded into the kernel,
outputs in sorted order. These tests stub it with a pure-torch MXFP4 dequant oracle obeying that contract, and pin:
  - "1" at T > 1 (device grouping): both projections go through K21 (gate_up WITH `order`, down WITHOUT) at the
    route's registered plan, neither the GEMV nor the v1 grouped GEMM runs, the tile table is 16-row, and the output
    equals the per-row dequant oracle in the caller's row order -- with gpt-oss's per-expert biases and clamped GLU too;
  - "1" moves T == 1 on an MXFP4 store to the device tile table (so the decode-shaped KL instrument reads K21);
  - the default ("0" / unset) is today's route: the GEMV at <= 16 rows, the v1 grouped GEMM above when the NF4 stacks
    are freed; K21 is never called;
  - "1" on a kernel side without K21, or without its masked K tail (#425), is a RuntimeError naming the requirement;
  - an unknown value is refused;
  - prefill rows (R > 256) do not take K21.
Runs wherever the NF4 grouping helpers import (linux CI).
"""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("nf4_grouped")

E, K1, INTER = 4, 64, 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequant(blocks, scales, N, K):
    """MXFP4: e2m1 nibbles, element 2j in the LOW nibble; e8m0 scale per 32-block = 2^(e - 127)."""
    lo, hi = (blocks & 0xF).long(), (blocks >> 4).long()
    nib = torch.stack([lo, hi], dim=-1).reshape(N, K)
    val = E2M1[nib & 7] * torch.where(nib & 8 > 0, -1.0, 1.0)
    return val * torch.pow(2.0, scales.float() - 127).repeat_interleave(32, dim=1)


def _stores():
    g = torch.Generator().manual_seed(21)

    def stack(N, K):
        b = torch.randint(0, 256, (E, N, K // 2), generator=g, dtype=torch.uint8)
        s = torch.randint(122, 126, (E, N, K // 32), generator=g, dtype=torch.uint8)
        return b, s
    gb, gs = stack(2 * INTER, K1)
    db, ds = stack(K1, INTER)
    return {"kind": "mxfp4",
            "gu": {"blocks": gb, "scales": gs, "N": 2 * INTER, "K": K1},
            "dn": {"blocks": db, "scales": ds, "N": K1, "K": INTER}}


def _gptoss():
    g = torch.Generator().manual_seed(5)
    return (torch.randn(E, 2 * INTER, generator=g) * 0.1, torch.randn(E, K1, generator=g) * 0.1, 1.702, 7.0)


def _oracle(x, ids, st, gptoss=None):
    ref = torch.empty(x.shape[0], K1)
    for i in range(x.shape[0]):
        e = int(ids[i])
        wg = _dequant(st["gu"]["blocks"][e], st["gu"]["scales"][e], 2 * INTER, K1)
        wd = _dequant(st["dn"]["blocks"][e], st["dn"]["scales"][e], K1, INTER)
        gu = x[i].float() @ wg.t()
        if gptoss is None:
            g_, u_ = gu.chunk(2)
            ref[i] = (F.silu(g_) * u_) @ wd.t()
        else:
            gb, db, alpha, limit = gptoss
            g_, u_ = (gu + gb[e]).chunk(2)
            g_, u_ = g_.clamp(max=limit), u_.clamp(min=-limit, max=limit)
            ref[i] = ((u_ + 1) * (g_ * torch.sigmoid(g_ * alpha))) @ wd.t() + db[e]
    return ref


def _install_stubs(monkeypatch, with_k21=True, masked_tail=True):
    calls = {"k21": [], "gemv": 0, "v1": 0, "tiles": []}
    b32 = types.ModuleType("int4_b32")
    b32.quant_x_rows = lambda x: (x.float(), None)
    monkeypatch.setitem(sys.modules, "int4_b32", b32)
    import nf4_grouped
    real_builder = nf4_grouped.build_group_tiles_device

    def recording_builder(ids, n_exp, block_m):
        calls["tiles"].append(block_m)
        return real_builder(ids, n_exp, block_m)
    monkeypatch.setattr(nf4_grouped, "build_group_tiles_device", recording_builder)

    mx = types.ModuleType("mxfp4_grouped")

    def gemv_mxfp4_b32(xq, xs, blocks, scales, eids, N, K):
        calls["gemv"] += 1
        return torch.stack([(xq[i] @ _dequant(blocks[int(eids[i])], scales[int(eids[i])], N, K).t())
                            for i in range(xq.shape[0])]).to(torch.bfloat16)

    def gemm_mxfp4_grouped(x, blocks, scales, sizes, eids):
        calls["v1"] += 1
        E_, N_, kh = blocks.shape
        return torch.stack([(x[i].float() @ _dequant(blocks[int(eids[i])], scales[int(eids[i])], N_, kh * 2).t())
                            for i in range(x.shape[0])]).to(torch.bfloat16)
    mx.gemv_mxfp4_b32 = gemv_mxfp4_b32
    mx.gemm_mxfp4_grouped = gemm_mxfp4_grouped
    if with_k21:
        def gemm_mxfp4_grouped_smallm(x, blocks, scales, t_row0, t_rows, t_group, order=None, **plan):
            calls["k21"].append(("gather" if order is not None else "sorted", plan))
            E_, N_, kh = blocks.shape
            xs = x.index_select(0, order) if order is not None else x
            out = torch.zeros(x.shape[0], N_, dtype=torch.bfloat16)
            for g in range(t_row0.numel()):
                rows = int(t_rows[g])
                if rows == 0:
                    continue
                assert rows <= 16, "K21 serves 16-row tiles"
                r0, e = int(t_row0[g]), int(t_group[g])
                out[r0:r0 + rows] = (xs[r0:r0 + rows].float() @ _dequant(blocks[e], scales[e], N_, kh * 2).t()
                                     ).to(torch.bfloat16)
            return out
        mx.gemm_mxfp4_grouped_smallm = gemm_mxfp4_grouped_smallm
        names = ["x_ptr", "K", "GATHER"] + (["EVEN_K"] if masked_tail else [])
        mx._gemm_mxfp4_grouped_smallm = types.SimpleNamespace(arg_names=names)
    monkeypatch.setitem(sys.modules, "mxfp4_grouped", mx)
    return calls


def _run(R, *, T1=False, gptoss=None, seed=3):
    from experts4bit_qlora.engines import hot_residency as hr
    st = _stores()
    torch.manual_seed(seed)
    x = torch.randn(R, K1, dtype=torch.bfloat16) * 0.3
    ids = torch.randint(0, E, (R,))
    freed = torch.empty(0, 0, 0, dtype=torch.uint8)
    freed_dn = torch.empty(0, 0, 0, dtype=torch.uint8)
    fa = torch.empty(0, 0, 0)
    s, gr = hr._collapsed_grouping(1, st) if T1 else (False, True)
    out = hr._fused_over_stack(x, ids, freed, fa, freed_dn, fa, (2 * INTER, K1, K1, INTER), True, F.silu,
                               gptoss=gptoss, singleton_groups=s, device_grouping=gr, int4_stores=st)
    return out, _oracle(x, ids, st, gptoss)


def _close(out, ref):
    return float((out.float() - ref).abs().max() / ref.abs().max()) < 0.05


@pytest.mark.parametrize("gptoss", [None, "gptoss"], ids=["silu-glu", "gptoss-epilogue"])
def test_opted_in_batched_rows_go_through_k21_and_match_the_oracle(monkeypatch, gptoss):
    from experts4bit_qlora.engines.hot_residency import _K21_PLAN
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24, gptoss=_gptoss() if gptoss else None)
    assert [c[0] for c in calls["k21"]] == ["gather", "sorted"], calls      # gate_up gathers in-kernel; down is sorted
    assert all(c[1] == _K21_PLAN for c in calls["k21"]), calls              # the registered plan, both projections
    assert calls["gemv"] == 0 and calls["v1"] == 0 and calls["tiles"] == [16], calls
    assert _close(out, ref)


def test_opted_in_t1_rows_take_the_device_table_and_k21(monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(4, T1=True, gptoss=_gptoss())
    assert [c[0] for c in calls["k21"]] == ["gather", "sorted"] and calls["gemv"] == 0, calls
    assert _close(out, ref)


@pytest.mark.parametrize("env", [None, "0", ""], ids=["unset", "zero", "empty"])
def test_the_default_keeps_todays_routes(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("E4B_MXFP4_GROUPED_SMALLM", raising=False)
    else:
        monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", env)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(8, T1=True)                                             # T == 1 shape: the split-K GEMV
    assert calls["gemv"] == 2 and calls["k21"] == [] and calls["tiles"] == [], calls
    assert _close(out, ref)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24)                                                     # 24 > 16 rows, NF4 freed: v1 grouped
    assert calls["v1"] == 2 and calls["k21"] == [] and calls["gemv"] == 0, calls
    assert _close(out, ref)


def test_opted_in_without_k21_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    _install_stubs(monkeypatch, with_k21=False)
    with pytest.raises(RuntimeError, match="needs grouped-nf4-gemm with K21"):
        _run(24)


def test_opted_in_without_the_masked_tail_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    _install_stubs(monkeypatch, masked_tail=False)
    with pytest.raises(RuntimeError, match="masked K tail"):
        _run(24)


def test_an_unknown_value_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "yes")
    _install_stubs(monkeypatch)
    with pytest.raises(ValueError, match="E4B_MXFP4_GROUPED_SMALLM='yes'"):
        _run(24)


def test_prefill_rows_do_not_take_k21(monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(300)
    assert calls["k21"] == [] and calls["v1"] == 2, calls
    assert _close(out, ref)


@pytest.mark.parametrize("opt_in", [False, True])
def test_collapsed_grouping_moves_only_mxfp4_t1(monkeypatch, opt_in):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    if opt_in:
        monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    else:
        monkeypatch.delenv("E4B_MXFP4_GROUPED_SMALLM", raising=False)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    assert hr._collapsed_grouping(1, {"kind": "mxfp4"}) == ((False, True) if opt_in else (True, False))
    assert hr._collapsed_grouping(1, {"gu": {}, "dn": {}}) == (True, False)       # the int4 store is K19's switch
    assert hr._collapsed_grouping(16, {"kind": "mxfp4"}) == (False, True)
