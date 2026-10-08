# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_WIDE_TILES (e4b#846): the one-launch tile table above 256 routed rows.

A decode step above 32 rows at top-k 8 routes more than 256 rows and takes K19's prefill route, whose tile table the
chained builder makes (lane P119: 2.87 ms of a 15.64 ms 64-row step). ``1`` builds it in one launch with
grouped-nf4-gemm's cumsum rank (``build_group_tiles_fused(..., rank="cumsum")``), whose integers are the chained
builder's. With the kernel side stubbed as ``tests/test_int4_grouped_smallm_route.py`` stubs it, these pin:
  - the default (unset / ``0``) keeps the chained builder;
  - ``1`` at 257, 512 and 1024 rows calls the fused builder with ``rank="cumsum"`` and gives the default's bits;
  - wider calls (prefill chunks, > 1024 rows) and calls of at most 256 rows are unchanged under ``1``;
  - ``1`` on a kernel side whose builder has no ``rank=`` is refused, and an unknown value is refused.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
route = pytest.importorskip("test_int4_grouped_smallm_route")     # its stubs, oracle and runner (nf4_grouped needed)


def _stubs(monkeypatch, with_rank=True):
    calls = route._install_stubs(monkeypatch, k23=True)
    import int4_b32
    import nf4_grouped
    chained = nf4_grouped.build_group_tiles_device

    if with_rank:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4,
                                    rank="pairwise"):
            calls["fused"].append((block_m, lean, sorted_ids, rank))
            out = chained(ids, n_exp, block_m)          # the cumsum kernel's contract: the chained builder's integers
            return out + (ids.index_select(0, out[3]),) if sorted_ids else out
    else:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4):
            calls["fused"].append((block_m, lean, sorted_ids, None))
            out = chained(ids, n_exp, block_m)
            return out + (ids.index_select(0, out[3]),) if sorted_ids else out
    int4_b32.build_group_tiles_fused = build_group_tiles_fused
    return calls


def _wide_run(R, monkeypatch, wide, seed=7):
    monkeypatch.setenv("E4B_INT4_PREFILL", "k19")               # K19 serves the rows above 256 (no CUDA needed here)
    if wide is None:
        monkeypatch.delenv("E4B_INT4_WIDE_TILES", raising=False)
    else:
        monkeypatch.setenv("E4B_INT4_WIDE_TILES", wide)
    return route._run(R, monkeypatch, seed=seed)


@pytest.mark.parametrize("wide", [None, "0", ""], ids=["unset", "0", "empty"])
def test_the_default_keeps_the_chained_builder_above_256_rows(monkeypatch, wide):
    calls = _stubs(monkeypatch)
    out, ref = _wide_run(300, monkeypatch, wide)
    assert calls["fused"] == [] and calls["tiles"] == [16], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


@pytest.mark.parametrize("R", [257, 512, 1024])
def test_wide_tiles_take_the_cumsum_builder_with_the_same_bits(monkeypatch, R):
    calls = _stubs(monkeypatch)
    off, ref = _wide_run(R, monkeypatch, "0", seed=R)
    assert calls["fused"] == [], calls
    calls = _stubs(monkeypatch)
    on, _ = _wide_run(R, monkeypatch, "1", seed=R)
    assert calls["fused"] == [(16, False, False, "cumsum")] and calls["tiles"] == [16], calls
    assert torch.equal(on, off), "the one-launch table moved an output bit"
    assert (on.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_wide_tiles_leave_prefill_chunks_on_the_chained_builder(monkeypatch):
    calls = _stubs(monkeypatch)
    _wide_run(1025, monkeypatch, "1")
    assert calls["fused"] == [] and calls["tiles"] == [16], calls


def test_wide_tiles_change_nothing_at_or_below_256_rows(monkeypatch):
    """At most 256 rows the decode route's own one-launch table runs (pairwise rank), as before."""
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    calls = _stubs(monkeypatch)
    _wide_run(200, monkeypatch, "1")
    assert calls["fused"] == [(16, False, False, "pairwise")], calls


def test_wide_tiles_on_a_kernel_side_without_rank_is_refused(monkeypatch):
    calls = _stubs(monkeypatch, with_rank=False)
    with pytest.raises(RuntimeError, match="E4B_INT4_WIDE_TILES=1 needs grouped-nf4-gemm"):
        _wide_run(300, monkeypatch, "1")
    assert calls["k19"] == []


def test_wide_tiles_unknown_value_is_refused(monkeypatch):
    _stubs(monkeypatch)
    with pytest.raises(ValueError, match="E4B_INT4_WIDE_TILES='auto'"):
        _wide_run(300, monkeypatch, "auto")
