# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_NF4_SMALLM_BW``'s per-family default (lane P234, ``bench/p234/RESULTS-p234.md``): unset, Mixtral's NF4 store
routes calls of 2 to 4 tokens (buckets 2 and 4) to K33's per-row GEMV; nothing else moves.

On CPU, with K33's engagement and the device's part stubbed (CUDA questions; P234's box read them on the card), these
pin the scope:
  - Mixtral, unset, on the part P234 read (sm_120 at 170 SMs, the RTX 5090): on at 2, 3 and 4 tokens; off at 5 to 8
    (bucket 8 stays opt-in), at 1 (bucket 1, where the switch never acts) and at 9 and above;
  - any other part (a 188-SM sm_120 included, where K33 also engages) or no CUDA device: off, the part named;
  - every other family, unset: off at every token count;
  - ``0`` is the way back: off on Mixtral too; ``1`` is unchanged: 2 to 8 on any family;
  - the default never refuses a build: where an explicit ``1`` would refuse (an int4 or MXFP4 store, K25 forced, a
    kernel package without K33, a shape where K33 does not engage, a layer off the all-resident collapse), it stays off
    and names the reason; at call time it yields to an int4 store and to K25 forced, without raising;
  - each resolution replaces the last (a later family's build clears Mixtral's default);
  - the server passes the model's family to the resolver.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("nf4_grouped")

from test_nf4_grouped_smallm_route import INTER, K1  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

KNOBS = ("E4B_NF4_SMALLM_BW", "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING", "GNF4_GEMV_BW")
OTHER_FAMILIES = ("qwen3_moe", "olmoe", "granitemoe", "gpt_oss", "qwen2_moe", None)
REAL_PART = hr._nf4_smallm_bw_part                     # before the fixture stubs it


class _State:
    def __init__(self, *, stores=None, collapse=True):
        self.shapes = (2 * INTER, K1, K1, INTER)
        self._int4_stores = stores
        self.collapse_resident = collapse

    def _all_hot(self):
        return True


def _mods(n=3, **kw):
    return [types.SimpleNamespace(_hot_residency=_State(**kw)) for _ in range(n)]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    import nf4_grouped
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_RESOLVED", [False])
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_DEFAULT_MAX", [0])
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    engages = {"on": True}
    monkeypatch.setattr(nf4_grouped, "_bw_engages", lambda N, K, dev: engages["on"], raising=False)
    monkeypatch.setattr(hr, "_nf4_smallm_bw_part", lambda dev: ((12, 0), 170))   # the part P234 read (a CUDA question)
    return engages


def _routed():
    return {T: hr._nf4_smallm_bw_takes(T, None) for T in range(1, 17)}


def test_the_default_table_is_mixtral_at_four_tokens():
    assert hr.NF4_SMALLM_BW_DEFAULT_FAMILIES == {"mixtral": 4} and hr.NF4_SMALLM_BW_MAX_T == 8
    assert hr.NF4_SMALLM_BW_DEFAULT_PART == ((12, 0), 170)


@pytest.mark.parametrize("part,name", [(((12, 0), 188), "sm_120 at 188 SMs"), (((9, 0), 132), "sm_90 at 132 SMs"),
                                       (((8, 9), 128), "sm_89 at 128 SMs"), (((12, 0), 160), "sm_120 at 160 SMs"),
                                       (None, "no CUDA device")])
def test_only_the_part_p234_read_takes_the_default(monkeypatch, part, name):
    """K33 engages on any part of 160 or more SMs; the default is the READ's scope: sm_120 at 170 SMs (the RTX 5090).
    Any other part, a 188-SM sm_120 included, stays off with the part named, and an explicit 1 still routes there."""
    monkeypatch.setattr(hr, "_nf4_smallm_bw_part", lambda dev: part)
    r = hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="mixtral")
    assert r["on"] is False and r["source"] == f"default-off: no P234 read on this part ({name}; read on sm_120 at 170 SMs)"
    assert not any(_routed().values())
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    assert hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="mixtral")["on"] is True


def test_the_real_part_probe_names_no_part_off_cuda():
    assert REAL_PART("cpu") is None and REAL_PART("not a device") is None


def test_mixtral_unset_routes_buckets_2_and_4_only():
    r = hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="mixtral")
    assert r == {"env": None, "on": True, "max_tokens": 4, "why": None, "source": "default-allowlisted",
                 "shapes": [[K1, INTER], [2 * INTER, K1]]}
    assert {T for T, on in _routed().items() if on} == {2, 3, 4}
    for T in (2, 3, 4):
        assert hr._collapsed_grouping(T, None) == (True, False), T            # singleton groups: K33 per row
    for T in (5, 6, 7, 8, 9, 16):
        assert hr._collapsed_grouping(T, None) == (False, True), T            # the device tile table, as before
    assert hr._collapsed_grouping(1, None) == (True, False)                   # bucket 1 as it always was
    assert hr._NF4_SMALLM_BW_RESOLVED == [False], "the default never stands in for an explicit 1's resolution"


@pytest.mark.parametrize("family", OTHER_FAMILIES)
def test_every_other_family_stays_off(family):
    r = hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type=family)
    assert r["on"] is False and r["source"] == f"default-off: no read licenses model_type {family!r}"
    assert not any(_routed().values())
    assert hr._collapsed_grouping(4, None) == (False, True)


def test_zero_is_the_way_back_on_mixtral(monkeypatch):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "0")
    r = hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="mixtral")
    assert r["on"] is False and r["source"] == "explicit" and r["why"] == "E4B_NF4_SMALLM_BW=0"
    assert not any(_routed().values())


@pytest.mark.parametrize("family", ["mixtral", "qwen3_moe"])
def test_one_is_unchanged_two_to_eight_on_any_family(monkeypatch, family):
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    r = hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type=family)
    assert r["on"] is True and r["max_tokens"] == 8 and r["source"] == "explicit"
    assert {T for T, on in _routed().items() if on} == set(range(2, 9))


@pytest.mark.parametrize("case", ["int4-store", "k25-forced", "no-k33", "k33-off-at-a-shape", "off-collapse",
                                  "no-engine"])
def test_the_default_never_refuses_a_build(monkeypatch, _fresh, case):
    import nf4_grouped
    mods = _mods()
    if case == "int4-store":
        mods = _mods(stores={"kind": "int4_b32"})
    elif case == "k25-forced":
        monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    elif case == "no-k33":
        monkeypatch.delattr(nf4_grouped, "_bw_engages", raising=False)
    elif case == "k33-off-at-a-shape":
        _fresh["on"] = False
    elif case == "off-collapse":
        mods = _mods(collapse=False)
    else:
        mods = _mods() + [types.SimpleNamespace()]
    r = hr.resolve_nf4_smallm_bw(mods, "cpu", model_type="mixtral")             # never raises
    assert r["on"] is False and r["source"].startswith("default-off: ") and r["why"], r
    assert not any(_routed().values())
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    with pytest.raises(RuntimeError):                                           # where an explicit 1 refuses
        hr.resolve_nf4_smallm_bw(mods, "cpu", model_type="mixtral")


def test_the_resolved_default_yields_at_call_time_without_raising(monkeypatch):
    hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="mixtral")
    assert hr._nf4_smallm_bw_takes(2, None)
    assert not hr._nf4_smallm_bw_takes(2, {"kind": "mxfp4"})                  # an MXFP4 layer: not the NF4 store
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    assert not hr._nf4_smallm_bw_takes(2, None)                                 # K25 forced names these rows


def test_each_resolution_replaces_the_last():
    hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="mixtral")
    assert hr._NF4_SMALLM_BW_DEFAULT_MAX == [4]
    hr.resolve_nf4_smallm_bw(_mods(), "cpu", model_type="qwen3_moe")
    assert hr._NF4_SMALLM_BW_DEFAULT_MAX == [0] and not any(_routed().values())


def test_an_unresolved_process_routes_nothing_by_default():
    assert not any(_routed().values())


def test_the_server_passes_the_family():
    src = open(os.path.join(os.path.dirname(__file__), "..", "experts4bit_qlora", "serve_paged.py"),
               encoding="utf-8").read()
    i = src.index("nf4_smallm_bw = resolve_nf4_smallm_bw(mods, cfg.device, model_type=(")
    assert 'getattr(model_config, "model_type", None)' in src[i:i + 300]
