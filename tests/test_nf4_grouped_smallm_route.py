# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_NF4_GROUPED_SMALLM routes the NF4 store's device-grouped decode rows through K25 (grouped-nf4-gemm#429,
`nf4_smallm.gemm_nf4_grouped_smallm`): "0" (the default, also when unset) is today's route; "auto" takes K25 above
T == 1 when the kernel package carries it; "1" requires it and adds T == 1.

K25 is K19's kernel with the NF4 dequant: the 16-row device tile table, gate_up's gather folded into the kernel, outputs
in sorted order, K23's scatter / gather_div. These tests stub it, the served NF4 M-tile GEMM and the host-grouped NF4
GEMM with pure-torch oracles (`nf4_grouped.dequant_ref`) obeying those contracts, and pin:
  - unset / "0": the served M-tile GEMM serves the batched rows and the host path T == 1; K25 never runs;
  - "1" at T > 1: both projections go through K25 (gate_up WITH `order`, down WITHOUT) at the route's registered plan,
    the served GEMM does not run, the table is 16-row, and the output equals the per-row oracle in the caller's row
    order -- with gpt-oss's per-expert biases and clamped GLU too;
  - "auto" takes K25 for batched rows and leaves T == 1 on the singleton route; "1" moves T == 1 to the device table;
  - the decision never moves an int4 or MXFP4 store's T == 1;
  - "auto" on a kernel side without K25 is today's route, silently; "1" there is a RuntimeError naming K25;
  - an unknown value is refused;
  - prefill rows (R > 256) and a calibration sink do not take K25;
  - under the lean glue (K23), gate_up reads the token rows (gather_div) and down scatters, with the same bits as the
    expanded call.
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

E, K1, INTER = 4, 64, 64


def _stacks():
    from nf4_pack_ref import quantize_pack_nf4
    g = torch.Generator().manual_seed(25)

    def stack(N, K):
        ps, as_ = zip(*[quantize_pack_nf4(torch.randn(N, K, generator=g) * 0.1) for _ in range(E)])
        return torch.stack(ps), torch.stack(as_)
    gu_p, gu_a = stack(2 * INTER, K1)
    dn_p, dn_a = stack(K1, INTER)
    return gu_p, gu_a, dn_p, dn_a


def _w(p, a):
    from nf4_grouped import dequant_ref
    N, kh = p.shape
    return dequant_ref(p, a, N, kh * 2)


def _gptoss():
    g = torch.Generator().manual_seed(5)
    return (torch.randn(E, 2 * INTER, generator=g) * 0.1, torch.randn(E, K1, generator=g) * 0.1, 1.702, 7.0)


def _oracle(x, ids, st, gptoss=None):
    gu_p, gu_a, dn_p, dn_a = st
    ref = torch.empty(x.shape[0], K1)
    for i in range(x.shape[0]):
        e = int(ids[i])
        gu = x[i].float() @ _w(gu_p[e], gu_a[e]).t()
        wd = _w(dn_p[e], dn_a[e])
        if gptoss is None:
            g_, u_ = gu.chunk(2)
            ref[i] = (F.silu(g_) * u_) @ wd.t()
        else:
            gb, db, alpha, limit = gptoss
            g_, u_ = (gu + gb[e]).chunk(2)
            g_, u_ = g_.clamp(max=limit), u_.clamp(min=-limit, max=limit)
            ref[i] = ((u_ + 1) * (g_ * torch.sigmoid(g_ * alpha))) @ wd.t() + db[e]
    return ref


def _tiles_matmul(calls, tag, x, packed, absmax, t_row0, t_rows, t_group, order):
    """The 16-row-tile contract shared by the served M-tile GEMM and K25: sorted rows out."""
    calls[tag].append("gather" if order is not None else "sorted")
    xs = x.index_select(0, order) if order is not None else x
    out = torch.zeros(xs.shape[0], packed.shape[1], dtype=torch.bfloat16)
    for g in range(t_row0.numel()):
        rows = int(t_rows[g])
        if rows == 0:
            continue
        assert rows <= 16, "16-row tiles"
        r0, e = int(t_row0[g]), int(t_group[g])
        out[r0:r0 + rows] = (xs[r0:r0 + rows].float() @ _w(packed[e], absmax[e]).t()).to(torch.bfloat16)
    return out


def _install_stubs(monkeypatch, with_k25=True, k23=False, tree=False):
    calls = {"k25": [], "k25_plan": [], "captured": [], "host": 0, "tiles": [], "fused": []}
    b32 = types.ModuleType("int4_b32")
    monkeypatch.setitem(sys.modules, "int4_b32", b32)
    import nf4_grouped
    real_builder = nf4_grouped.build_group_tiles_device

    def recording_builder(ids, n_exp, block_m):
        calls["tiles"].append(block_m)
        return real_builder(ids, n_exp, block_m)
    monkeypatch.setattr(nf4_grouped, "build_group_tiles_device", recording_builder)

    def captured(a_sorted, B, absmax, t_row0, t_rows, t_group, block_m, **kw):
        assert block_m == 16
        return _tiles_matmul(calls, "captured", a_sorted, B, absmax, t_row0, t_rows, t_group, None)
    monkeypatch.setattr(nf4_grouped, "gemm_4bit_grouped_captured", captured)

    def host(x, B, absmax, sizes, eids):
        calls["host"] += 1
        e_l = eids.tolist() if torch.is_tensor(eids) else list(eids)
        out, r = torch.empty(x.shape[0], B.shape[1], dtype=torch.bfloat16), 0
        for m, e in zip(sizes, e_l):
            out[r:r + m] = (x[r:r + m].float() @ _w(B[int(e)], absmax[int(e)]).t()).to(torch.bfloat16)
            r += m
        return out
    monkeypatch.setattr(nf4_grouped, "gemm_4bit_grouped", host)
    if k23:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4):
            calls["fused"].append((block_m, lean, sorted_ids))
            out = real_builder(ids, n_exp, block_m)
            return out + (ids.index_select(0, out[3]),) if sorted_ids else out
        b32.build_group_tiles_fused = build_group_tiles_fused

    sm = types.ModuleType("nf4_smallm")
    if with_k25:
        def gemm_nf4_grouped_smallm(x, packed, absmax, t_row0, t_rows, t_group, order=None, *, scatter=None,
                                    gather_div=1, block_n=32, kc=256, warps=4, stages=2, lut="pair"):
            calls["k25_plan"].append({"block_n": block_n, "kc": kc, "warps": warps, "stages": stages, "lut": lut})
            if gather_div != 1:                                  # token rows: expand, as the kernel's order // k reads
                calls["k25"].append(f"tokens/{gather_div}")
                x = x.repeat_interleave(gather_div, 0)
            y = _tiles_matmul(calls, "k25", x, packed, absmax, t_row0, t_rows, t_group, order)
            if scatter is not None:
                calls["k25"][-1] += "+scatter"
                return torch.empty_like(y).index_copy_(0, scatter, y)
            return y
        sm.gemm_nf4_grouped_smallm = gemm_nf4_grouped_smallm
        if tree:                                                 # a kernel package that carries the select tree (K26)
            sm._LUT_MODES = {"load": 0, "pair": 2, "tree": 3}
    monkeypatch.setitem(sys.modules, "nf4_smallm", sm)
    return calls


def _run(R, *, T1=False, gptoss=None, seed=3, x_tokens=None):
    from experts4bit_qlora.engines import hot_residency as hr
    st = _stacks()
    torch.manual_seed(seed)
    x = torch.randn(R, K1, dtype=torch.bfloat16) * 0.3
    ids = torch.randint(0, E, (R,))
    s, gr = hr._collapsed_grouping(1, None) if T1 else (False, True)
    out = hr._fused_over_stack(x, ids, st[0], st[1], st[2], st[3], (2 * INTER, K1, K1, INTER), True, F.silu,
                               gptoss=gptoss, singleton_groups=s, device_grouping=gr, int4_stores=None)
    return out, _oracle(x, ids, st, gptoss)


def _close(out, ref):
    return float((out.float() - ref).abs().max() / ref.abs().max()) < 0.05


@pytest.mark.parametrize("env", [None, "0"], ids=["unset", "0"])
def test_the_default_is_todays_route(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("E4B_NF4_GROUPED_SMALLM", raising=False)
    else:
        monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", env)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24)
    assert calls["captured"] == ["sorted", "sorted"] and calls["k25"] == [] and calls["tiles"] == [16], calls
    assert _close(out, ref)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(4, T1=True)                                             # T == 1: the singleton host route
    assert calls["host"] == 2 and calls["k25"] == [] and calls["captured"] == [], calls
    assert _close(out, ref)


@pytest.mark.parametrize("gptoss", [None, "gptoss"], ids=["silu-glu", "gptoss-epilogue"])
@pytest.mark.parametrize("tree", [True, False], ids=["kernel-with-tree", "kernel-without-tree"])
def test_opted_in_batched_rows_go_through_k25_and_match_the_oracle(monkeypatch, gptoss, tree):
    """The plan is _K25_PLAN with the select-tree decode when the kernel package carries it, and the paired lookup
    (bit-identical, lane K26) when it does not."""
    from experts4bit_qlora.engines.hot_residency import _K25_PLAN
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch, tree=tree)
    out, ref = _run(24, gptoss=_gptoss() if gptoss else None)
    want = _K25_PLAN if tree else dict(_K25_PLAN, lut="pair")
    assert _K25_PLAN["lut"] == "tree"
    assert calls["k25"] == ["gather", "sorted"], calls                      # gate_up gathers in-kernel; down is sorted
    assert calls["k25_plan"] == [want, want], calls                         # the registered plan, both projections
    assert calls["captured"] == [] and calls["host"] == 0 and calls["tiles"] == [16], calls
    assert _close(out, ref)


def test_auto_takes_k25_above_t1_and_leaves_t1_alone(monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "auto")
    assert hr._collapsed_grouping(1, None) == (True, False)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24)
    assert calls["k25"] == ["gather", "sorted"] and calls["captured"] == [], calls
    assert _close(out, ref)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(4, T1=True)
    assert calls["host"] == 2 and calls["k25"] == [], calls
    assert _close(out, ref)


def test_opted_in_t1_rows_take_the_device_table_and_k25(monkeypatch):
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(4, T1=True, gptoss=_gptoss())
    assert calls["k25"] == ["gather", "sorted"] and calls["host"] == 0 and calls["captured"] == [], calls
    assert _close(out, ref)


@pytest.mark.parametrize("mode", ["auto", "1"])
def test_the_t1_decision_never_moves_another_store(monkeypatch, mode):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", mode)
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "0")
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "0")
    assert hr._collapsed_grouping(1, {"kind": "int4"}) == (True, False)
    assert hr._collapsed_grouping(1, {"kind": "mxfp4"}) == (True, False)


def test_auto_without_k25_is_todays_route_silently(monkeypatch):
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "auto")
    calls = _install_stubs(monkeypatch, with_k25=False)
    out, ref = _run(24)
    assert calls["captured"] == ["sorted", "sorted"], calls
    assert _close(out, ref)


def test_opted_in_without_k25_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    _install_stubs(monkeypatch, with_k25=False)
    with pytest.raises(RuntimeError, match="needs grouped-nf4-gemm with K25"):
        _run(24)


def test_an_unknown_value_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "yes")
    _install_stubs(monkeypatch)
    with pytest.raises(ValueError, match="E4B_NF4_GROUPED_SMALLM='yes'"):
        _run(24)


def test_prefill_rows_do_not_take_k25(monkeypatch):
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(300)
    assert calls["k25"] == [] and calls["captured"] == ["sorted", "sorted"], calls
    assert _close(out, ref)


def test_a_calibration_sink_does_not_take_k25(monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    seen = []
    monkeypatch.setattr(hr, "_CALIB_SINK", lambda gu_p, sorted_ids, x_sorted, h: seen.append(x_sorted.shape))
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24)
    assert calls["k25"] == [] and calls["captured"] == ["sorted", "sorted"] and seen == [(24, K1)], calls
    assert _close(out, ref)


@pytest.mark.parametrize("lean", ["0", "auto"])
def test_token_rows_under_the_lean_glue_have_the_expanded_calls_bits(monkeypatch, lean):
    """The collapse hands (x, row_token, top_k) with x_rows=None. Under the lean glue K25's gate_up reads the token
    rows (gather_div) and its down stores into the caller's order (scatter); without it the rows are expanded and the
    output unsorted afterwards. Same bits either way."""
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    T, k = 6, 4
    torch.manual_seed(31)
    st = _stacks()
    xt = torch.randn(T, K1, dtype=torch.bfloat16) * 0.2
    ids = torch.randint(0, E, (T * k,))
    rt = torch.arange(T * k) // k
    args = (*st, (2 * INTER, K1, K1, INTER), True, F.silu)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    _install_stubs(monkeypatch, k23=True)
    want = _fused_over_stack(xt.index_select(0, rt), ids, *args, device_grouping=True, int4_stores=None)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", lean)
    calls = _install_stubs(monkeypatch, k23=True)
    got = _fused_over_stack(None, ids, *args, device_grouping=True, int4_stores=None, x_tokens=(xt, rt, k))
    assert torch.equal(got, want)
    if lean == "auto":
        assert calls["k25"] == [f"tokens/{k}", "gather", "sorted+scatter"], calls
        assert calls["fused"] == [(16, True, True)], calls
    else:
        assert calls["k25"] == ["gather", "sorted"], calls
