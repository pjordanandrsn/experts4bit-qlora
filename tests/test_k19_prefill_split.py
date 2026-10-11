# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_K19_PREFILL_BIG_M (opt-in, e4b#846): K19's prefill rows above K19_SPLIT_ROWS_MIN routed rows over a split tile
table (grouped-nf4-gemm#542), on CPU with stand-ins that implement the split contract literally: one ``order``, each
expert's full ``big_m``-row tiles plus 16-row remainders, the tall tiles launched first and the remainders stored into
the same result. Every row is computed as on the 16-row table, so the output is ``torch.equal`` to the knob-off route's,
and ``K19_DISPATCH_SEEN`` names the split calls ``split|...``; a 64-row decode step (512 routed rows) and every
smaller call keep the 16-row table; the knob composes with P130's lean dispatch; a kernel package without the split
is refused."""
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


def _tiles_into(out, xs, packed, scales, t_row0, t_rows, t_grp, cap):
    n_out, kh = packed.shape[1], packed.shape[2]
    for g in range(t_row0.numel()):
        rows = int(t_rows[g])
        if rows:
            assert rows <= cap, f"a {rows}-row tile on a {cap}-row launch"
            r0, e = int(t_row0[g]), int(t_grp[g])
            w = dequant_int4_ref(packed[e], scales[e], n_out, kh * 2)
            out[r0:r0 + rows] = (xs[r0:r0 + rows].float() @ w.t()).to(torch.bfloat16)
    return out


def _gathered(x, order, gather_div):
    if gather_div != 1:
        return x.index_select(0, torch.div(order, gather_div, rounding_mode="floor"))
    return x.index_select(0, order) if order is not None else x


def _unsort(out, scatter):
    if scatter is None:
        return out
    res = torch.empty_like(out)
    res.index_copy_(0, scatter, out)
    return res


def _install(monkeypatch, calls, split=True):
    """K19 and the split launch by their contracts; the split builder as grouped-nf4-gemm#542 states its rule."""
    def k19(x, packed, scales, t_row0, t_rows, t_group, order=None, gather_div=1, scatter=None):
        calls.append({"fn": "k19", "rows": int(t_rows.sum())})
        xs = _gathered(x, order, gather_div)
        out = torch.zeros(xs.shape[0], packed.shape[1], dtype=torch.bfloat16)
        return _unsort(_tiles_into(out, xs, packed, scales, t_row0, t_rows, t_group, 16), scatter)

    def k19_split(x, packed, scales, big, small, big_m, order=None, *, scatter=None, gather_div=1, **_kw):
        calls.append({"fn": "split", "big_m": big_m, "big_rows": int(big[1].sum()), "small_rows": int(small[1].sum()),
                      "scatter": scatter is not None, "gather_div": gather_div})
        xs = _gathered(x, order, gather_div)
        out = torch.zeros(xs.shape[0], packed.shape[1], dtype=torch.bfloat16)
        _tiles_into(out, xs, packed, scales, *big, big_m)
        _tiles_into(out, xs, packed, scales, *small, 16)
        return _unsort(out, scatter)

    def build_split(expert_ids, n_experts, big_m, small_m=16):
        ids = expert_ids.to(torch.int64)
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=n_experts)
        big, small, row = ([], [], []), ([], [], []), 0
        for e, c in enumerate(counts.tolist()):
            nb = c // big_m
            for i in range(nb):
                for t, v in zip(big, (row + i * big_m, big_m, e)):
                    t.append(v)
            r0, left = row + nb * big_m, c - nb * big_m
            while left > 0:
                take = min(small_m, left)
                for t, v in zip(small, (r0, take, e)):
                    t.append(v)
                r0, left = r0 + take, left - take
            row += c
        as_t = [tuple(torch.tensor(v or [0], dtype=torch.int32) * (1 if v else 0) for v in tab) for tab in (big, small)]
        return as_t[0], as_t[1], order, counts

    sm = types.ModuleType("int4_smallm")
    sm.gemm_int4_b32_grouped_smallm = k19
    if split:
        sm.gemm_int4_b32_grouped_split = k19_split
    monkeypatch.setitem(sys.modules, "int4_smallm", sm)
    m = types.ModuleType("int4_b32")                  # no one-launch builder: the chained table, as on prefill
    m.quant_x_rows = lambda x: (_ for _ in ()).throw(AssertionError("M-tile quantise under k19"))
    monkeypatch.setitem(sys.modules, "int4_b32", m)
    import nf4_grouped
    if split:
        monkeypatch.setattr(nf4_grouped, "build_group_tiles_split", build_split, raising=False)
    else:
        monkeypatch.delattr(nf4_grouped, "build_group_tiles_split", raising=False)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    monkeypatch.setattr(hr, "K19_DISPATCH_SEEN", {})
    monkeypatch.setenv("E4B_INT4_PREFILL", "k19")
    monkeypatch.setenv("E4B_INT4_WIDE_TILES", "0")


def _call(st, T, k, seed=8, heavy=True):
    """Token rows plus the token-major (token, slot) map; expert 0 heavy when ``heavy`` (the census's prefill shape)."""
    torch.manual_seed(seed)
    x_t = torch.randn(T, H, dtype=torch.bfloat16) * 0.3
    row_token = torch.arange(T * k) // k
    ids = torch.randint(0, E, (T * k,))
    if heavy:
        ids[::2] = 0
    freed = [torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8), torch.empty(0), torch.empty(0)]
    return hr._fused_over_stack(None, ids, freed[0], freed[2], freed[1], freed[3], (2 * INTER, H, H, INTER), True,
                                F.silu, singleton_groups=False, device_grouping=True, int4_stores=st,
                                x_tokens=(x_t, row_token, k))


def test_the_knob_defaults_to_0_and_refuses_anything_else(monkeypatch):
    monkeypatch.delenv("E4B_K19_PREFILL_BIG_M", raising=False)
    assert hr._k19_prefill_big_m_env() == 0
    for v, want in (("0", 0), ("", 0), (" 32 ", 32), ("64", 64)):
        monkeypatch.setenv("E4B_K19_PREFILL_BIG_M", v)
        assert hr._k19_prefill_big_m_env() == want, v
    for bad in ("16", "128", "auto", "1", "yes"):
        monkeypatch.setenv("E4B_K19_PREFILL_BIG_M", bad)
        with pytest.raises(ValueError, match="E4B_K19_PREFILL_BIG_M"):
            hr._k19_prefill_big_m_env()
    assert hr.K19_SPLIT_ROWS_MIN == 512


@pytest.mark.parametrize("lean", ["auto", "0"])
@pytest.mark.parametrize("big_m", [32, 64])
def test_prefill_rows_split_bitwise(monkeypatch, big_m, lean):
    """A 100-token chunk at top-6 (600 routed rows, expert 0 heavy): the split table, the same bits as the 16-row table,
    with P130's lean dispatch on (auto) or off."""
    st = _stores()
    out, seen, calls = {}, {}, {}
    monkeypatch.setenv("E4B_PREFILL_LEAN_DISPATCH", lean)
    for knob in ("0", str(big_m)):
        calls[knob] = []
        _install(monkeypatch, calls[knob])
        monkeypatch.setenv("E4B_K19_PREFILL_BIG_M", knob)
        out[knob] = _call(st, 100, 6)
        seen[knob] = dict(hr.K19_DISPATCH_SEEN)
    disp = "lean" if lean == "auto" else "gather"
    assert seen["0"] == {f"chained|{disp}|gt256": 1} and seen[str(big_m)] == {f"split|{disp}|gt256": 1}
    assert [c["fn"] for c in calls["0"]] == ["k19", "k19"] and [c["fn"] for c in calls[str(big_m)]] == ["split", "split"]
    gu = calls[str(big_m)][0]
    assert gu["big_m"] == big_m and gu["big_rows"] > 0 and gu["big_rows"] + gu["small_rows"] == 600
    assert gu["gather_div"] == (6 if lean == "auto" else 1) and calls[str(big_m)][1]["scatter"] == (lean == "auto")
    assert torch.equal(out[str(big_m)], out["0"])


def test_decode_steps_keep_the_16_row_table(monkeypatch):
    """A 64-row decode step at top-8 routes exactly 512 rows: not above K19_SPLIT_ROWS_MIN, so the chained table."""
    st = _stores()
    calls = []
    _install(monkeypatch, calls)
    monkeypatch.setenv("E4B_K19_PREFILL_BIG_M", "64")
    _call(st, 64, 8)
    assert {c["fn"] for c in calls} == {"k19"} and set(hr.K19_DISPATCH_SEEN) == {"chained|lean|gt256"}
    calls.clear()
    monkeypatch.setattr(hr, "K19_DISPATCH_SEEN", {})
    _call(st, 65, 8)                                   # 520 routed rows: above it
    assert {c["fn"] for c in calls} == {"split"} and set(hr.K19_DISPATCH_SEEN) == {"split|lean|gt256"}


def test_a_kernel_package_without_the_split_is_refused(monkeypatch):
    st = _stores()
    _install(monkeypatch, [], split=False)
    monkeypatch.setenv("E4B_K19_PREFILL_BIG_M", "64")
    with pytest.raises(RuntimeError, match="E4B_K19_PREFILL_BIG_M needs grouped-nf4-gemm"):
        _call(st, 100, 6)
    monkeypatch.setenv("E4B_K19_PREFILL_BIG_M", "0")         # off: the same package serves as before
    _call(st, 100, 6)
    assert set(hr.K19_DISPATCH_SEEN) == {"chained|lean|gt256"}
