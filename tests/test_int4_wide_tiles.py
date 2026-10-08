# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_WIDE_TILES (e4b#846): the one-launch tile table above 256 routed rows.

A decode step above 32 rows at top-k 8 routes more than 256 rows and takes K19's prefill route, whose tile table the
chained builder makes. grouped-nf4-gemm's cumsum rank (``build_group_tiles_fused(..., rank="cumsum")``, #515) builds
the same integers in one launch, and its chunked build (``rchunk=``, #519) made Qwen3-30B-A3B's 64-row step 4.3 %
faster (lane P122, ``e4b.serve.p122.wide-tiles-chunked.qwen3-int4.5090.2026-10-08``); without the chunks it was 1.44x
slower (lane P120). With the kernel side stubbed as ``tests/test_int4_grouped_smallm_route.py`` stubs it, these pin:
  - ``auto`` (the default, also unset or empty) takes the one-launch table when the builder has ``rank=`` and
    ``rchunk=`` and the table is inside P122's bound, with the chained builder's bits;
  - ``auto`` keeps the chained builder on a kernel package without ``rchunk=`` (between #515 and #519) or without
    ``rank=`` (released grouped-nf4-gemm up to 0.43.0), and on a table larger than the one read;
  - ``1`` forces the one-launch table at any size and without ``rchunk=``, and is refused without ``rank=``;
  - ``0`` keeps the chained builder; calls of at most 256 rows and prefill chunks are unchanged; an unknown value is
    refused.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
route = pytest.importorskip("test_int4_grouped_smallm_route")     # its stubs, oracle and runner (nf4_grouped needed)


def _stubs(monkeypatch, rank=True, rchunk=True):
    calls = route._install_stubs(monkeypatch, k23=True)
    import int4_b32
    import nf4_grouped
    chained = nf4_grouped.build_group_tiles_device

    def _table(ids, n_exp, block_m, sorted_ids, tag):
        calls["fused"].append(tag)
        out = chained(ids, n_exp, block_m)          # the cumsum kernel's contract: the chained builder's integers
        return out + (ids.index_select(0, out[3]),) if sorted_ids else out

    if rank and rchunk:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4,
                                    rank="pairwise", rchunk=None):
            return _table(ids, n_exp, block_m, sorted_ids, (block_m, lean, sorted_ids, rank))
    elif rank:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4,
                                    rank="pairwise"):
            return _table(ids, n_exp, block_m, sorted_ids, (block_m, lean, sorted_ids, rank))
    else:
        def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4):
            return _table(ids, n_exp, block_m, sorted_ids, (block_m, lean, sorted_ids, None))
    int4_b32.build_group_tiles_fused = build_group_tiles_fused
    return calls


def _wide_run(R, monkeypatch, wide, seed=7):
    monkeypatch.setenv("E4B_INT4_PREFILL", "k19")               # K19 serves the rows above 256 (no CUDA needed here)
    if wide is None:
        monkeypatch.delenv("E4B_INT4_WIDE_TILES", raising=False)
    else:
        monkeypatch.setenv("E4B_INT4_WIDE_TILES", wide)
    return route._run(R, monkeypatch, seed=seed)


@pytest.mark.parametrize("wide", [None, "", "auto"], ids=["unset", "empty", "auto"])
@pytest.mark.parametrize("R", [257, 512, 1024])
def test_auto_takes_the_chunked_cumsum_table_with_the_chained_bits(monkeypatch, wide, R):
    calls = _stubs(monkeypatch)
    off, ref = _wide_run(R, monkeypatch, "0", seed=R)
    assert calls["fused"] == [] and calls["tiles"] == [16], calls
    calls = _stubs(monkeypatch)
    on, _ = _wide_run(R, monkeypatch, wide, seed=R)
    assert calls["fused"] == [(16, False, False, "cumsum")] and calls["tiles"] == [16], calls
    assert torch.equal(on, off), "the one-launch table moved an output bit"
    assert (on.float() - ref).abs().max() / ref.abs().max() < 0.05


@pytest.mark.parametrize("rank,rchunk", [(True, False), (False, False)], ids=["no-rchunk (#515 only)", "no-rank (<=0.43.0)"])
def test_auto_keeps_the_chained_builder_without_the_chunked_build(monkeypatch, rank, rchunk):
    calls = _stubs(monkeypatch, rank=rank, rchunk=rchunk)
    out, ref = _wide_run(512, monkeypatch, None)
    assert calls["fused"] == [] and calls["tiles"] == [16], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


def test_auto_keeps_the_chained_builder_above_the_size_read(monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setattr(hr, "_WIDE_TILES_AUTO_MAX_TILE", 4 * 256)  # the harness has 4 experts: admit R <= 256 only
    calls = _stubs(monkeypatch)
    _wide_run(300, monkeypatch, None)                              # next_pow2(4) * next_pow2(300) = 2048 > 1024
    assert calls["fused"] == [] and calls["tiles"] == [16], calls
    calls = _stubs(monkeypatch)
    _wide_run(300, monkeypatch, "1")                               # 1 forces it at any size
    assert calls["fused"] == [(16, False, False, "cumsum")], calls


def test_the_shape_bound_is_the_table_p122_read():
    from experts4bit_qlora.engines import hot_residency as hr
    assert hr._WIDE_TILES_AUTO_MAX_TILE == 128 * 512
    assert [hr._next_pow2(n) for n in (1, 2, 3, 40, 128, 129, 300, 512, 513, 1024)] == [1, 2, 4, 64, 128, 256, 512, 512,
                                                                                         1024, 1024]
    assert hr._wide_tiles_auto_takes(128, 512) and hr._wide_tiles_auto_takes(128, 257)       # Qwen3's 64-row step
    assert hr._wide_tiles_auto_takes(40, 1024) and hr._wide_tiles_auto_takes(64, 1024)       # 64 x 1024: the same lanes
    assert not hr._wide_tiles_auto_takes(128, 513) and not hr._wide_tiles_auto_takes(256, 300)
    assert "e4b.serve.p122.wide-tiles-chunked.qwen3-int4.5090.2026-10-08" in open(hr.__file__, encoding="utf-8").read()


def test_one_forces_the_table_without_the_chunked_build(monkeypatch):
    calls = _stubs(monkeypatch, rchunk=False)
    _wide_run(512, monkeypatch, "1")
    assert calls["fused"] == [(16, False, False, "cumsum")], calls


def test_one_on_a_kernel_side_without_rank_is_refused(monkeypatch):
    calls = _stubs(monkeypatch, rank=False, rchunk=False)
    with pytest.raises(RuntimeError, match="E4B_INT4_WIDE_TILES=1 needs grouped-nf4-gemm"):
        _wide_run(300, monkeypatch, "1")
    assert calls["k19"] == []


def test_zero_keeps_the_chained_builder(monkeypatch):
    calls = _stubs(monkeypatch)
    out, ref = _wide_run(300, monkeypatch, "0")
    assert calls["fused"] == [] and calls["tiles"] == [16], calls
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.05


@pytest.mark.parametrize("wide", [None, "1"])
def test_prefill_chunks_keep_the_chained_builder(monkeypatch, wide):
    calls = _stubs(monkeypatch)
    _wide_run(1025, monkeypatch, wide)
    assert calls["fused"] == [] and calls["tiles"] == [16], calls


@pytest.mark.parametrize("wide", [None, "1"])
def test_at_most_256_rows_nothing_changes(monkeypatch, wide):
    """At most 256 rows the decode route's own one-launch table runs (pairwise rank), as before."""
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    calls = _stubs(monkeypatch)
    _wide_run(200, monkeypatch, wide)
    assert calls["fused"] == [(16, False, False, "pairwise")], calls


def test_an_unknown_value_is_refused(monkeypatch):
    _stubs(monkeypatch)
    with pytest.raises(ValueError, match="E4B_INT4_WIDE_TILES='2': expected 'auto', '0' or '1'"):
        _wide_run(300, monkeypatch, "2")
