# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_NF4_SMALLM_BW`` (serving roadmap lever B, step B0; opt-in): on the NF4 store's all-resident collapse, a call of
2..8 tokens takes singleton groups, every (token, slot) row its own group, so grouped-nf4-gemm's ``gemm_4bit_grouped``
serves it through K33's bandwidth GEMV as it serves T == 1, instead of the device tile table and K25.

On CPU with the route tests' kernel stubs (``test_nf4_grouped_smallm_route.py``), these pin:
  - unset and ``0`` change no grouping at any T; ``1`` sends T = 2..8 to singleton groups with or without
    ``DEVICE_GROUPING``, and leaves T == 1 and T = 9 as they were; anything else is refused;
  - an int4 or MXFP4 store, and ``E4B_NF4_GROUPED_SMALLM=1`` (which names the same rows for K25), are refused, never
    dropped, and only where the knob would route;
  - on the singleton route the gate_up call reads the TOKEN rows (``gather_div=top_k``) with one group per row, down
    reads its own rows, K25 never runs, and the output is the per-row oracle's;
  - :func:`hot_residency.resolve_nf4_smallm_bw` reports the knob off, and refuses by name where it cannot engage: K25
    forced, an int4 store, a layer off the all-resident collapse, a kernel package without K33, a shape where K33 does
    not engage. Only a resolution that passes lets the routing take the knob; an unresolved process (a harness enabling
    the collapse itself) is refused at its first 2..8-token call, and a refused resolution clears an earlier pass.
The kernel's bits (each row at T = 2..8 equal to the same row at T == 1) are ``test_nf4_smallm_bw_gpu.py``'s.
"""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("nf4_grouped")

from test_nf4_grouped_smallm_route import E, INTER, K1, _install_stubs, _oracle, _stacks, _w  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

KNOBS = ("E4B_NF4_SMALLM_BW", "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_K25_LUT", "E4B_K25_PLAN",
         "GNF4_GEMV_BW")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hr, "NF4_SMALLM_BW_SEEN", {})
    monkeypatch.setattr(hr, "ROUTE_SEEN", {})
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_RESOLVED", [True])      # the routing tests' process resolved it
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])


@pytest.mark.parametrize("value", [None, "0", ""], ids=["unset", "zero", "empty"])
def test_off_changes_no_grouping(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("E4B_NF4_SMALLM_BW", value)
    assert hr._collapsed_grouping(1, None) == (True, False)
    for T in (2, 4, 8, 9, 64):
        assert hr._collapsed_grouping(T, None) == (False, True), T


@pytest.mark.parametrize("device_grouping", [True, False])
def test_on_sends_2_to_8_tokens_to_singleton_groups_and_nothing_else(monkeypatch, device_grouping):
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [device_grouping])
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    assert hr._collapsed_grouping(1, None) == (True, False)
    for T in range(2, hr.NF4_SMALLM_BW_MAX_T + 1):
        assert hr._collapsed_grouping(T, None) == (True, False), T
    assert hr._collapsed_grouping(9, None) == (False, device_grouping)
    assert hr.NF4_SMALLM_BW_MAX_T == 8


@pytest.mark.parametrize("value", ["2", "yes", "auto", "on"])
def test_an_unknown_value_is_refused(monkeypatch, value):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", value)
    with pytest.raises(ValueError, match=r"E4B_NF4_SMALLM_BW=.*expected '0' or '1'"):
        hr._collapsed_grouping(4, None)


@pytest.mark.parametrize("kind", ["int4_b32", "mxfp4"])
def test_an_int4_or_mxfp4_store_is_refused_where_the_knob_routes(monkeypatch, kind):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    with pytest.raises(RuntimeError, match=rf"E4B_NF4_SMALLM_BW=1 .* serves a {kind} expert store"):
        hr._collapsed_grouping(4, {"kind": kind})
    hr._collapsed_grouping(9, {"kind": kind})                         # outside 2..8: not the knob's, no refusal


def test_k25_forced_on_the_same_rows_is_refused(monkeypatch):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    with pytest.raises(RuntimeError, match=r"E4B_NF4_SMALLM_BW=1 and E4B_NF4_GROUPED_SMALLM=1 both name"):
        hr._collapsed_grouping(2, None)


def _tokens_run(monkeypatch, T, top_k, *, knob):
    """One collapse-shaped call (token rows, token-major (token, slot) rows) on the singleton route, through a
    ``gemm_4bit_grouped`` stub that takes ``gather_div`` and records each call."""
    calls = _install_stubs(monkeypatch)
    seen = []

    def gemm_4bit_grouped(x, B, absmax, sizes, eids, gather_div=1):
        seen.append({"rows": x.shape[0], "sizes": list(sizes), "gather_div": gather_div})
        xr = x.repeat_interleave(gather_div, 0)
        e_l = eids.tolist() if torch.is_tensor(eids) else list(eids)
        assert len(sizes) == len(e_l) == xr.shape[0] and set(sizes) == {1}
        return torch.stack([(xr[r].float() @ _w(B[int(e)], absmax[int(e)]).t()) for r, e in enumerate(e_l)]
                           ).to(torch.bfloat16)
    import nf4_grouped
    monkeypatch.setattr(nf4_grouped, "gemm_4bit_grouped", gemm_4bit_grouped)
    if knob:
        monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    st = _stacks()
    torch.manual_seed(7)
    x_t = torch.randn(T, K1, dtype=torch.bfloat16) * 0.3
    ids = torch.randint(0, E, (T * top_k,))
    rt = torch.arange(T * top_k) // top_k
    s, g = hr._collapsed_grouping(T, None) if knob else (True, False)
    out = hr._fused_over_stack(None, ids, st[0], st[1], st[2], st[3], (2 * INTER, K1, K1, INTER), True, F.silu,
                               singleton_groups=s, device_grouping=g, int4_stores=None, x_tokens=(x_t, rt, top_k))
    return out, _oracle(x_t.index_select(0, rt), ids, st), seen, calls


@pytest.mark.parametrize("T", [2, 5, 8])
def test_on_the_gate_up_call_reads_token_rows_one_group_per_row(monkeypatch, T):
    top_k = 2
    out, ref, seen, calls = _tokens_run(monkeypatch, T, top_k, knob=True)
    R = T * top_k
    assert seen == [{"rows": T, "sizes": [1] * R, "gather_div": top_k},          # gate_up: the token rows
                    {"rows": R, "sizes": [1] * R, "gather_div": 1}]              # down: its own rows
    assert calls["k25"] == [] and calls["captured"] == [] and calls["tiles"] == []
    assert hr.ROUTE_SEEN == {"nf4_singleton|le256": 1}
    assert (out.float() - ref).abs().max() < 0.05


def test_without_the_knob_a_multi_token_singleton_call_keeps_the_expanded_rows(monkeypatch):
    """The gather_div widening is the knob's: FORCE_SINGLETON_GROUPS' T > 1 calls still hand over expanded rows."""
    out, ref, seen, _ = _tokens_run(monkeypatch, 4, 2, knob=False)
    assert [c["gather_div"] for c in seen] == [1, 1] and seen[0]["rows"] == 8
    assert (out.float() - ref).abs().max() < 0.05


class _State:
    def __init__(self, *, stores=None, collapse=True, all_hot=True):
        self.shapes = (2 * INTER, K1, K1, INTER)
        self._int4_stores = stores
        self.collapse_resident = collapse
        self._hot = all_hot

    def _all_hot(self):
        return self._hot


def _mods(n=3, **kw):
    return [types.SimpleNamespace(_hot_residency=_State(**kw)) for _ in range(n)]


@pytest.fixture
def engages(monkeypatch):
    import nf4_grouped
    asked = []

    def _bw_engages(N, K, device):
        asked.append((N, K, str(device)))
        return True
    monkeypatch.setattr(nf4_grouped, "_bw_engages", _bw_engages, raising=False)
    return asked


def test_resolve_reports_the_knob_off_without_reading_anything(monkeypatch):
    """Unset with no family the default lists (here none at all), and set to 0: off, read from nothing."""
    r = hr.resolve_nf4_smallm_bw([], "cpu")
    assert r == {"env": None, "on": False, "max_tokens": 0, "why": "no read licenses model_type None",
                 "source": "default-off: no read licenses model_type None"}
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "0")
    assert hr.resolve_nf4_smallm_bw([], "cpu") == {"env": "0", "on": False, "max_tokens": 8, "source": "explicit",
                                                   "why": "E4B_NF4_SMALLM_BW=0"}


def test_resolve_on_names_every_expert_shape_k33_engages_at(monkeypatch, engages):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    r = hr.resolve_nf4_smallm_bw(_mods(), "cpu")
    assert r == {"env": "1", "on": True, "max_tokens": 8, "why": None, "source": "explicit",
                 "shapes": [[K1, INTER], [2 * INTER, K1]]}
    assert sorted(set(engages)) == [(K1, INTER, "cpu"), (2 * INTER, K1, "cpu")]


def test_resolve_refuses_a_shape_where_k33_does_not_engage(monkeypatch):
    import nf4_grouped
    monkeypatch.setattr(nf4_grouped, "_bw_engages", lambda N, K, dev: (N, K) != (K1, INTER), raising=False)
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    with pytest.raises(RuntimeError, match=rf"does not engage at \(N, K\) \[\({K1}, {INTER}\)\].*GNF4_GEMV_BW=auto"):
        hr.resolve_nf4_smallm_bw(_mods(), "cpu")


def test_resolve_refuses_a_kernel_package_without_k33(monkeypatch):
    import nf4_grouped
    monkeypatch.delattr(nf4_grouped, "_bw_engages", raising=False)
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    with pytest.raises(RuntimeError, match=r"needs grouped-nf4-gemm with K33's bandwidth GEMV"):
        hr.resolve_nf4_smallm_bw(_mods(), "cpu")


@pytest.mark.parametrize("mods, why", [
    (lambda: _mods(stores={"kind": "int4_b32"}), r"3/3 MoE layers serve an int4 or MXFP4 expert store"),
    (lambda: _mods(collapse=False), r"all-resident collapse only, and 3/3 MoE layers are not on it"),
    (lambda: _mods(all_hot=False), r"all-resident collapse only, and 3/3 MoE layers are not on it"),
    (lambda: _mods() + [types.SimpleNamespace()], r"hot-residency engine on every MoE layer; 1/4 have none"),
], ids=["int4-store", "no-collapse", "not-all-hot", "no-engine"])
def test_resolve_refuses_where_the_knob_cannot_engage(monkeypatch, engages, mods, why):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    with pytest.raises(RuntimeError, match=why):
        hr.resolve_nf4_smallm_bw(mods(), "cpu")


def test_resolve_refuses_k25_forced(monkeypatch, engages):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    with pytest.raises(RuntimeError, match=r"both name the rows of 2..8 tokens"):
        hr.resolve_nf4_smallm_bw(_mods(), "cpu")


def test_the_knob_without_a_resolution_is_refused_at_the_first_routed_call(monkeypatch):
    """A harness that enables the collapse itself (enable_hybrid_tier, hybrid_train, step_decomp) never resolved the
    knob: where K33 does not engage it would run dot-pad, so its 2..8-token calls are refused, by name."""
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_RESOLVED", [False])
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    assert hr._collapsed_grouping(1, None) == (True, False)              # T == 1 is not the knob's
    assert hr._collapsed_grouping(9, None) == (False, True)
    with pytest.raises(RuntimeError, match=r"only after hot_residency.resolve_nf4_smallm_bw\(modules, device\)"):
        hr._collapsed_grouping(4, None)


def test_a_passing_resolution_enables_the_routing_and_a_refused_one_clears_it(monkeypatch, engages):
    import nf4_grouped
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_RESOLVED", [False])
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    hr.resolve_nf4_smallm_bw(_mods(), "cpu")
    assert hr._NF4_SMALLM_BW_RESOLVED == [True] and hr._collapsed_grouping(4, None) == (True, False)
    monkeypatch.setattr(nf4_grouped, "_bw_engages", lambda N, K, dev: False, raising=False)
    with pytest.raises(RuntimeError, match=r"does not engage"):
        hr.resolve_nf4_smallm_bw(_mods(), "cpu")
    assert hr._NF4_SMALLM_BW_RESOLVED == [False]
    with pytest.raises(RuntimeError, match=r"only after"):
        hr._collapsed_grouping(4, None)
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "0")
    hr.resolve_nf4_smallm_bw([], "cpu")
    assert hr._NF4_SMALLM_BW_RESOLVED == [False]
