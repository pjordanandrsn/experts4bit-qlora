# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_TILE_PROGRAMS (e4b#846, opt-in): the one-launch cumsum tile table above 256 routed rows, split over programs.

grouped-nf4-gemm #524's ``build_group_tiles_fused(..., rank="cumsum", programs=P)`` gives each of P programs a slice of
the experts; the tables are the same integers at every P. With the kernel side stubbed as
``tests/test_int4_wide_tiles.py`` stubs it, these pin:
  - ``1`` (the default, also unset or empty) calls the builder exactly as before: no ``programs=`` keyword at all;
  - an integer from 2 to 64 passes ``programs=P`` where the cumsum table is built, with the chained builder's bits;
  - ``P > 1`` on a kernel package without ``programs=`` is refused there; calls that take no cumsum table (at most 256
    rows, ``E4B_INT4_WIDE_TILES=0``, prefill chunks) are unchanged and never refused;
  - anything that is not an integer from 1 to 64 is refused.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
wide = pytest.importorskip("test_int4_wide_tiles")      # its stubs and runner (nf4_grouped needed)


def _stubs(monkeypatch, programs=True):
    calls = wide.route._install_stubs(monkeypatch, k23=True)
    import int4_b32
    import nf4_grouped
    chained = nf4_grouped.build_group_tiles_device
    calls["kw"] = []

    def _table(ids, n_exp, block_m, sorted_ids, kw):
        calls["fused"].append((block_m, kw.get("lean", False), sorted_ids, kw.get("rank")))
        calls["kw"].append({k: v for k, v in kw.items() if k in ("rank", "programs")})
        out = chained(ids, n_exp, block_m)                # the kernel's contract: the chained builder's integers
        return out + (ids.index_select(0, out[3]),) if sorted_ids else out

    if programs:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4,
                                    rank="pairwise", rchunk=None, programs=None):
            kw = {"lean": lean, "rank": rank}
            if programs is not None:
                kw["programs"] = programs
            return _table(ids, n_exp, block_m, sorted_ids, kw)
    else:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4,
                                    rank="pairwise", rchunk=None):
            return _table(ids, n_exp, block_m, sorted_ids, {"lean": lean, "rank": rank})
    int4_b32.build_group_tiles_fused = build_group_tiles_fused
    return calls


def _run(R, monkeypatch, programs, wide_tiles=None, seed=7):
    if programs is None:
        monkeypatch.delenv("E4B_INT4_TILE_PROGRAMS", raising=False)
    else:
        monkeypatch.setenv("E4B_INT4_TILE_PROGRAMS", programs)
    return wide._wide_run(R, monkeypatch, wide_tiles, seed=seed)


@pytest.mark.parametrize("programs", [None, "", "1"], ids=["unset", "empty", "one"])
def test_one_program_calls_the_builder_exactly_as_before(monkeypatch, programs):
    calls = _stubs(monkeypatch)
    _run(512, monkeypatch, programs)
    assert calls["kw"] == [{"rank": "cumsum"}], calls["kw"]                       # no programs= keyword at all


@pytest.mark.parametrize("P", ["2", "4", "8", "64"])
@pytest.mark.parametrize("R", [257, 512, 1024])
def test_a_program_count_reaches_the_cumsum_table_with_the_same_bits(monkeypatch, P, R):
    calls = _stubs(monkeypatch)
    one, ref = _run(R, monkeypatch, "1", seed=R)
    calls = _stubs(monkeypatch)
    split, _ = _run(R, monkeypatch, P, seed=R)
    assert calls["kw"] == [{"rank": "cumsum", "programs": int(P)}], calls["kw"]
    assert torch.equal(split, one), "the split table moved an output bit"
    assert (split.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_more_than_one_program_on_an_older_kernel_is_refused(monkeypatch):
    calls = _stubs(monkeypatch, programs=False)
    with pytest.raises(RuntimeError, match="E4B_INT4_TILE_PROGRAMS > 1 needs grouped-nf4-gemm"):
        _run(512, monkeypatch, "4")
    assert calls["k19"] == []
    calls = _stubs(monkeypatch, programs=False)
    _run(512, monkeypatch, "1")                                                     # the default never asks
    assert calls["kw"] == [{"rank": "cumsum"}]


@pytest.mark.parametrize("case", ["at most 256 rows", "wide tiles off", "prefill chunk"])
def test_calls_without_a_cumsum_table_are_unchanged(monkeypatch, case):
    calls = _stubs(monkeypatch, programs=False)                                     # even an older kernel: no refusal
    if case == "at most 256 rows":
        monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
        _run(200, monkeypatch, "8")
        assert calls["kw"] == [{"rank": "pairwise"}], calls["kw"]
    elif case == "wide tiles off":
        _run(512, monkeypatch, "8", wide_tiles="0")
        assert calls["kw"] == [] and calls["tiles"] == [16], calls
    else:
        _run(1025, monkeypatch, "8")
        assert calls["kw"] == [] and calls["tiles"] == [16], calls


@pytest.mark.parametrize("bad", ["0", "65", "x", "2.0", "-4", "04"])
def test_anything_but_an_integer_from_1_to_64_is_refused(monkeypatch, bad):
    _stubs(monkeypatch)
    with pytest.raises(ValueError, match="E4B_INT4_TILE_PROGRAMS=.*expected an integer from 1 to 64"):
        _run(512, monkeypatch, bad)
