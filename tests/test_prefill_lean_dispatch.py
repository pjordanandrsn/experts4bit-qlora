# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_PREFILL_LEAN_DISPATCH (bench/prefill-glue/DESIGN.md, P2), on CPU with a K19 stand-in that implements K23's
contract literally: under the knob, K19's PREFILL rows read the token rows themselves (``gather_div=top_k``) and the down
call stores into the caller's row order (``scatter=order``), over the same chained 16-row tile table, and the output is
``torch.equal`` to the knob-off route's. ``K19_DISPATCH_SEEN`` records the table and the dispatch, which is what the
on-card read asserts. Decode rows, a calibration sink and a K19 without the options keep (or refuse) as specified."""
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

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

INTER, H, E = 32, 64, 6


def _stores(seed=7):
    torch.manual_seed(seed)

    def pack_stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk), torch.stack(sc)
    gu_p, gu_s = pack_stack(torch.randn(E, 2 * INTER, H) * 0.1)
    dn_p, dn_s = pack_stack(torch.randn(E, H, INTER) * 0.1)
    return {"gu": {"packed": gu_p, "scales": gu_s, "N": 2 * INTER, "K": H},
            "dn": {"packed": dn_p, "scales": dn_s, "N": H, "K": INTER}}


def _k19_stub(calls, lean_options=True):
    """K19 by its contract: sorted rows out; ``order`` = gather the unsorted rows first; ``gather_div=g`` = sorted row i
    reads token row ``order[i] // g``; ``scatter=order`` = store sorted row i at ``order[i]``."""
    def tiles(xs, packed, scales, t_row0, t_rows, t_grp):
        n_out, kh = packed.shape[1], packed.shape[2]
        out = torch.zeros(xs.shape[0], n_out, dtype=torch.bfloat16)
        for g in range(t_row0.numel()):
            rows = int(t_rows[g])
            if rows:
                r0, e = int(t_row0[g]), int(t_grp[g])
                w = dequant_int4_ref(packed[e], scales[e], n_out, kh * 2)
                out[r0:r0 + rows] = (xs[r0:r0 + rows].float() @ w.t()).to(torch.bfloat16)
        return out

    def k19(x, packed, scales, t_row0, t_rows, t_group, order=None, gather_div=1, scatter=None):
        calls.append({"rows_in": x.shape[0], "order": order is not None, "gather_div": gather_div,
                      "scatter": scatter is not None})
        if gather_div != 1:
            xs = x.index_select(0, torch.div(order, gather_div, rounding_mode="floor"))
        elif order is not None:
            xs = x.index_select(0, order)
        else:
            xs = x
        out = tiles(xs, packed, scales, t_row0, t_rows, t_group)
        if scatter is None:
            return out
        res = torch.empty_like(out)
        res.index_copy_(0, scatter, out)
        return res

    if not lean_options:
        def k19_old(x, packed, scales, t_row0, t_rows, t_group, order=None):
            return k19(x, packed, scales, t_row0, t_rows, t_group, order)
        return k19_old
    return k19


def _install(monkeypatch, calls, lean_options=True):
    sm = types.ModuleType("int4_smallm")
    sm.gemm_int4_b32_grouped_smallm = _k19_stub(calls, lean_options)
    monkeypatch.setitem(sys.modules, "int4_smallm", sm)
    m = types.ModuleType("int4_b32")                  # no one-launch builder: the chained 16-row table, as on prefill
    m.quant_x_rows = lambda x: (_ for _ in ()).throw(AssertionError("M-tile quantise under k19"))
    monkeypatch.setitem(sys.modules, "int4_b32", m)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    monkeypatch.setattr(hr, "K19_DISPATCH_SEEN", {})
    monkeypatch.setenv("E4B_INT4_PREFILL", "k19")


def _call(st, T, k, seed=8):
    """The all-resident collapse's form: TOKEN rows plus the token-major (token, slot) -> token map."""
    torch.manual_seed(seed)
    x_t = torch.randn(T, H, dtype=torch.bfloat16) * 0.3
    row_token = torch.arange(T * k) // k
    ids = torch.randint(0, E, (T * k,))
    freed = [torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8), torch.empty(0), torch.empty(0)]
    return hr._fused_over_stack(None, ids, freed[0], freed[2], freed[1], freed[3], (2 * INTER, H, H, INTER), True,
                                F.silu, singleton_groups=False, device_grouping=True, int4_stores=st,
                                x_tokens=(x_t, row_token, k))


def test_the_knob_reads_zero_or_one_and_refuses_anything_else(monkeypatch):
    monkeypatch.delenv("E4B_PREFILL_LEAN_DISPATCH", raising=False)
    assert hr._prefill_lean_dispatch_env() is False
    for v, want in (("0", False), ("", False), (" 1 ", True)):
        monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", v)
        assert hr._prefill_lean_dispatch_env() is want, v
    for bad in ("auto", "yes", "2"):
        monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", bad)
        with pytest.raises(ValueError, match="E4B_PREFILL_LEAN_DISPATCH"):
            hr._prefill_lean_dispatch_env()


def test_prefill_rows_take_the_lean_dispatch_bitwise_over_the_chained_table(monkeypatch):
    st = _stores()
    T, k = 50, 6                                                  # 300 routed rows: K19's prefill route
    out, calls, seen = {}, {}, {}
    for env in ("0", "1"):
        calls[env] = []
        _install(monkeypatch, calls[env])
        monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", env)
        out[env] = _call(st, T, k)
        seen[env] = dict(hr.K19_DISPATCH_SEEN)
    # off: gate_up gathers the [T * k, H] rows through order; down is sorted, then unsorted by index_copy_
    assert calls["0"] == [{"rows_in": T * k, "order": True, "gather_div": 1, "scatter": False},
                          {"rows_in": T * k, "order": False, "gather_div": 1, "scatter": False}], calls["0"]
    # on: gate_up reads the T token rows itself; down stores straight into the caller's row order
    assert calls["1"] == [{"rows_in": T, "order": True, "gather_div": k, "scatter": False},
                          {"rows_in": T * k, "order": False, "gather_div": 1, "scatter": True}], calls["1"]
    assert torch.equal(out["0"], out["1"])                        # bit-identical by construction, here by test
    assert seen == {"0": {"chained|gather|gt256": 1}, "1": {"chained|lean|gt256": 1}}, seen


def test_decode_rows_and_a_calibration_sink_keep_the_gather(monkeypatch):
    st = _stores()
    calls = []
    _install(monkeypatch, calls)
    monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", "1")
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")                 # decode's own K23 stays out of this test
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    _call(st, 4, 6)                                               # 24 routed rows: a decode step, not prefill
    assert hr.K19_DISPATCH_SEEN == {"chained|gather|le256": 1}, hr.K19_DISPATCH_SEEN
    monkeypatch.setattr(hr, "K19_DISPATCH_SEEN", {})
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hr, "_CALIB_SINK", lambda *a: None)       # the sink reads x_sorted: the gather stays
    _call(st, 50, 6)
    assert hr.K19_DISPATCH_SEEN == {"chained|gather|gt256": 1}, hr.K19_DISPATCH_SEEN


def test_a_k19_without_the_options_is_refused_under_the_knob(monkeypatch):
    st = _stores()
    _install(monkeypatch, [], lean_options=False)
    monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", "1")
    with pytest.raises(RuntimeError, match="E4B_PREFILL_LEAN_DISPATCH=1 needs"):
        _call(st, 50, 6)
    monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", "0")          # off, the old K19 serves as before
    _call(st, 50, 6)
    assert hr.K19_DISPATCH_SEEN == {"chained|gather|gt256": 1}
