# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_GROUPED_SMALLM routes int4 decode rows through K19 (grouped-nf4-gemm#419): by default (auto) at T > 1 when
the kernel package carries it, since lane P88 licensed it; "1" requires it and adds T == 1; "0" keeps the GEMV.

K19 is `int4_smallm.gemm_int4_b32_grouped_smallm`: K16's arithmetic over the device tile table, gate_up's gather
folded into the kernel, outputs in sorted order. These tests stub it with a pure-torch oracle that obeys the same
contract (sorted rows out; `order` given = gather the unsorted input, None = input already sorted) and pin:
  - "1", and the default (unset / auto) at T > 1: both projections go through K19 (gate_up WITH `order`, down
    WITHOUT), the GEMV never runs, the tile table is built with 16-row tiles, and the output equals the per-row
    dequant oracle in the caller's row order;
  - the default on a kernel side without K19, and "0": the GEMV route, unchanged, and K19 is never called;
  - "1" with a kernel side that lacks K19: a RuntimeError naming the requirement, not a silent GEMV fallback;
  - an unknown value: refused;
  - T == 1 takes K19 only under "1" (the default leaves it on the singleton GEMV: P88 read B=1 SLOWER);
  - prefill rows (R > 256): K19 is not taken (it is a decode route).
E4B_INT4_LEAN_GLUE (lane K23, opt-in) on K19's rows: "1" builds the table with ``lean=True, sorted_ids=True`` and
stores the down projection through ``scatter=order`` (no unsort), with the same bits as "0"; handed the collapse's
TOKEN rows (``x_tokens``), gate_up reads them through ``gather_div`` and the (token, slot) expansion is never made;
"1" on a kernel side without K23's options is refused; an unknown value is refused; off K19's rows it changes nothing.
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


def _install_stubs(monkeypatch, with_k19=True, k23=False):
    calls = {"k19": [], "gemv": 0, "tiles": [], "fused": []}
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
    if k23:
        # K23's builder: the chained table, plus ids[order] as a sixth output when asked
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4):
            calls["fused"].append((block_m, lean, sorted_ids))
            out = real_builder(ids, n_exp, block_m)
            return out + (ids.index_select(0, out[3]),) if sorted_ids else out
        b32.build_group_tiles_fused = build_group_tiles_fused

    sm = types.ModuleType("int4_smallm")
    if with_k19 and k23:
        def gemm_int4_b32_grouped_smallm(x, packed, scales, t_row0, t_rows, t_group, order=None, *, scatter=None,
                                         gather_div=1, **kw):
            if gather_div != 1:                                  # token rows: expand, as the kernel's order // k reads
                calls["k19"].append(f"tokens/{gather_div}")
                x = x.repeat_interleave(gather_div, 0)
            y = _k19_stub(calls, x, packed, scales, t_row0, t_rows, t_group, order, scatter is not None)
            return y if scatter is None else torch.empty_like(y).index_copy_(0, scatter, y)
        sm.gemm_int4_b32_grouped_smallm = gemm_int4_b32_grouped_smallm
    elif with_k19:
        def gemm_int4_b32_grouped_smallm(x, packed, scales, t_row0, t_rows, t_group, order=None, **kw):
            return _k19_stub(calls, x, packed, scales, t_row0, t_rows, t_group, order, False)
        sm.gemm_int4_b32_grouped_smallm = gemm_int4_b32_grouped_smallm
    monkeypatch.setitem(sys.modules, "int4_smallm", sm)
    return calls


def _k19_stub(calls, x, packed, scales, t_row0, t_rows, t_group, order, scatter):
    calls["k19"].append(("gather" if order is not None else "sorted") + ("+scatter" if scatter else ""))
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


def _run(R, monkeypatch, seed=None):
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    if seed is not None:
        torch.manual_seed(seed)
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


@pytest.mark.parametrize("env", [None, "auto", "AUTO", ""], ids=["unset", "auto", "AUTO", "empty"])
def test_the_default_routes_batched_decode_through_k19(monkeypatch, env):
    """P88 LICENSED K19 for these rows: unset / auto uses it when the kernel package carries it."""
    if env is None:
        monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    else:
        monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", env)
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24, monkeypatch)
    assert calls["k19"] == ["gather", "sorted"] and calls["gemv"] == 0 and calls["tiles"] == [16], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_the_default_without_k19_keeps_the_gemv_route_silently(monkeypatch):
    """auto on a kernel package that predates K19 is the split-K GEMV, as before -- not a refusal."""
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    calls = _install_stubs(monkeypatch, with_k19=False)
    out, ref = _run(24, monkeypatch)
    assert calls["k19"] == [] and calls["gemv"] == 2 and calls["tiles"] == [], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_zero_keeps_the_gemv_route(monkeypatch):
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "0")
    calls = _install_stubs(monkeypatch)
    out, ref = _run(24, monkeypatch)
    assert calls["k19"] == [] and calls["gemv"] == 2 and calls["tiles"] == [], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_an_unknown_value_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "yes")
    _install_stubs(monkeypatch)
    with pytest.raises(ValueError, match="expected 'auto', '0' or '1'"):
        _run(24, monkeypatch)


def test_opted_in_without_k19_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    _install_stubs(monkeypatch, with_k19=False)
    with pytest.raises(RuntimeError, match="needs grouped-nf4-gemm with K19"):
        _run(24, monkeypatch)


def test_prefill_rows_do_not_take_k19(monkeypatch):
    """R > 256 is the prefill route (K14's grouped GEMM): the opt-in names a DECODE route and does not touch it.
    The prefill route is pinned to the M-tile here (E4B_INT4_PREFILL, #916): its default ``auto`` sends these rows to
    K19 wherever K19 can run, which is that knob's business, not this opt-in's."""
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    monkeypatch.setenv("E4B_INT4_PREFILL", "mtile")
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


@pytest.mark.parametrize("opt_in", [False, True])
def test_collapsed_grouping_decisions(monkeypatch, opt_in):
    """The all-resident collapse's grouping. The opt-in moves only T == 1 on a uniform-int4 store, from singleton
    groups to the device tile table; T > 1 still follows DEVICE_GROUPING, and MXFP4 / no store keep T == 1 singleton.
    The prefill route is pinned to ``loop`` (E4B_INT4_PREFILL, #916), whose T > 1 calls follow DEVICE_GROUPING."""
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_INT4_PREFILL", "loop")
    if opt_in:
        monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    else:
        monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    int4 = {"gu": {}, "dn": {}}
    assert hr._collapsed_grouping(1, int4) == ((False, True) if opt_in else (True, False))
    assert hr._collapsed_grouping(16, int4) == (False, True)
    assert hr._collapsed_grouping(1, {"kind": "mxfp4"}) == (True, False)
    assert hr._collapsed_grouping(1, None) == (True, False)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    assert hr._collapsed_grouping(16, int4) == (False, False)


def test_opted_in_t1_rows_go_through_k19(monkeypatch):
    """T == 1: one token's top-k ids are distinct, one row per expert. With the opt-in the collapse's decision reaches K19 (one 1-row tile
    per expert), not the singleton GEMV, and the output matches the oracle."""
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    calls = _install_stubs(monkeypatch)
    stores = _stores()
    singleton, grouped = hr._collapsed_grouping(1, stores)
    freed_gu = torch.empty(0, 0, 0, dtype=torch.uint8)       # distinct objects: _mm tells the slots apart by identity
    freed_dn = torch.empty(0, 0, 0, dtype=torch.uint8)
    x = torch.randn(E, K1, dtype=torch.bfloat16) * 0.2
    ids = torch.randperm(E)                                   # one token's top-k ids are distinct
    out = hr._fused_over_stack(x, ids, freed_gu, torch.empty(0, 0, 0), freed_dn, torch.empty(0, 0, 0),
                               (2 * INTER, K1, K1, INTER), True, F.silu,
                               singleton_groups=singleton, device_grouping=grouped, int4_stores=stores)
    assert calls["k19"] == ["gather", "sorted"] and calls["gemv"] == 0 and calls["tiles"] == [16], calls
    ref = _oracle(x, ids, stores)
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_lean_glue_takes_k23s_builder_and_scatter_with_the_same_bits(monkeypatch):
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    calls = _install_stubs(monkeypatch, k23=True)
    off, ref = _run(24, monkeypatch, seed=5)
    assert calls["fused"] == [(16, False, False)] and calls["k19"] == ["gather", "sorted"], calls
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "1")
    calls = _install_stubs(monkeypatch, k23=True)
    on, _ = _run(24, monkeypatch, seed=5)
    assert calls["fused"] == [(16, True, True)], calls             # one launch: the table and the sorted ids
    assert calls["k19"] == ["gather", "sorted+scatter"], calls     # down stores straight into the caller's order
    assert torch.equal(on, off), "K23's route moved an output bit"
    assert (on.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_lean_glue_on_a_kernel_side_without_k23_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "1")
    calls = _install_stubs(monkeypatch, k23=True)
    import int4_smallm
    k19 = int4_smallm.gemm_int4_b32_grouped_smallm
    monkeypatch.setattr(int4_smallm, "gemm_int4_b32_grouped_smallm",
                        lambda x, packed, scales, t_row0, t_rows, t_group, order=None: k19(
                            x, packed, scales, t_row0, t_rows, t_group, order))   # pre-K23 K19: no scatter=
    with pytest.raises(RuntimeError, match="needs grouped-nf4-gemm with K23"):
        _run(24, monkeypatch)
    assert calls["k19"] == []


def test_lean_glue_unknown_value_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "yes")
    _install_stubs(monkeypatch, k23=True)
    with pytest.raises(ValueError, match="E4B_INT4_LEAN_GLUE='yes'"):
        _run(24, monkeypatch)


def test_lean_glue_off_k19s_rows_changes_nothing(monkeypatch):
    """With K19 off ("0"), the GEMV route builds no tile table; the opt-in has nothing to fold and must not refuse."""
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "0")
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "1")
    calls = _install_stubs(monkeypatch, k23=True)
    out, ref = _run(24, monkeypatch)
    assert calls["k19"] == [] and calls["gemv"] == 2 and calls["fused"] == [], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


@pytest.mark.parametrize("lean", ["0", "1"])
def test_token_rows_are_read_without_the_expansion_under_lean_glue(monkeypatch, lean):
    """The collapse hands (x, row_token, top_k) with x_rows=None. Under "1" gate_up reads the token rows through
    gather_div (no expansion); under "0" the rows are expanded inside, as the caller used to. Same bits either way,
    and the same bits as the expanded call."""
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    T, k = 6, 4
    torch.manual_seed(31)
    stores = _stores()
    xt = torch.randn(T, K1, dtype=torch.bfloat16) * 0.2
    ids = torch.randint(0, E, (T * k,))
    rt = torch.arange(T * k) // k
    freed = (torch.empty(0, 0, 0, dtype=torch.uint8), torch.empty(0, 0, 0), torch.empty(0, 0, 0, dtype=torch.uint8))
    args = (freed[0], freed[1], freed[2], freed[1], (2 * INTER, K1, K1, INTER), True, F.silu)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    _install_stubs(monkeypatch, k23=True)
    want = _fused_over_stack(xt.index_select(0, rt), ids, *args, device_grouping=True, int4_stores=stores)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", lean)
    calls = _install_stubs(monkeypatch, k23=True)
    got = _fused_over_stack(None, ids, *args, device_grouping=True, int4_stores=stores, x_tokens=(xt, rt, k))
    assert torch.equal(got, want)
    if lean == "1":
        assert calls["k19"] == [f"tokens/{k}", "gather", "sorted+scatter"], calls
    else:
        assert calls["k19"] == ["gather", "sorted"], calls


def test_fused_over_stack_needs_rows_or_tokens(monkeypatch):
    from experts4bit_qlora.engines.hot_residency import _fused_over_stack
    _install_stubs(monkeypatch, k23=True)
    with pytest.raises(ValueError, match="needs x_rows or x_tokens"):
        _fused_over_stack(None, torch.zeros(4, dtype=torch.long), None, None, None, None, (1, 1, 1, 1), True, F.silu)


@pytest.mark.parametrize("env", [None, "auto", "AUTO", ""], ids=["unset", "auto", "AUTO", "empty"])
def test_the_lean_glue_is_the_default_since_p89(monkeypatch, env):
    """P89 LICENSED it on K19's rows: unset / auto takes K23's builder and the down scatter when the kernel side
    carries them, with the same bits as 0."""
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    _install_stubs(monkeypatch, k23=True)
    off, _ = _run(24, monkeypatch, seed=7)
    if env is None:
        monkeypatch.delenv("E4B_INT4_LEAN_GLUE", raising=False)
    else:
        monkeypatch.setenv("E4B_INT4_LEAN_GLUE", env)
    calls = _install_stubs(monkeypatch, k23=True)
    on, _ = _run(24, monkeypatch, seed=7)
    assert calls["fused"] == [(16, True, True)] and calls["k19"] == ["gather", "sorted+scatter"], calls
    assert torch.equal(on, off)


def test_the_default_on_a_kernel_side_without_k23_keeps_the_separate_launches(monkeypatch):
    """auto on a kernel package that predates K23 is the old route, silently -- not a refusal (1 refuses)."""
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    monkeypatch.delenv("E4B_INT4_LEAN_GLUE", raising=False)
    calls = _install_stubs(monkeypatch, k23=True)
    import int4_smallm
    k19 = int4_smallm.gemm_int4_b32_grouped_smallm
    monkeypatch.setattr(int4_smallm, "gemm_int4_b32_grouped_smallm",
                        lambda x, packed, scales, t_row0, t_rows, t_group, order=None: k19(
                            x, packed, scales, t_row0, t_rows, t_group, order))   # pre-K23 K19: no scatter=
    out, ref = _run(24, monkeypatch)
    assert calls["fused"] == [(16, False, False)] and calls["k19"] == ["gather", "sorted"], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05
