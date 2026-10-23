# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""serve_paged at the solver placement, the capacity frontier's first fact (bench/serve-hybrid/DESIGN-HC.md). CPU only.

Pinned: the startup matrix for decode graphs at each placement; the all-resident collapse runs only when every expert
is resident AND the hot stack is in identity order (``_all_hot``, driven on CPU); and the baseline (tiered) forward
reaches its host sync, ``hot_row.nonzero()``, before any branch on the routing, which is why a decode-graph capture of
a tiered layer raises on every call and falls back to eager instead of capturing a routing that skips cold experts.
"""
import inspect
import re
import types

import pytest
import torch

pytest.importorskip("fastapi")
from experts4bit_qlora import serve_paged as serve_paged_mod  # noqa: E402
from experts4bit_qlora.engines import hot_residency  # noqa: E402
from experts4bit_qlora.serve_paged import PagedServeConfig, _batched_graph_grouping, _graphs_env  # noqa: E402


@pytest.fixture
def _flags(monkeypatch):
    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hot_residency, "FORCE_SINGLETON_GROUPS", [True])
    return hot_residency


def test_the_startup_matrix_at_the_solver_placement(_flags):
    """auto decodes eagerly at solver; a forced 1 is refused with more than one sequence and a bucket above 1, and
    accepted at one sequence (where every tiered capture then falls back to eager), leaving grouping at its defaults."""
    assert _graphs_env("auto", "cuda", "solver") is False and _graphs_env("", "cuda", "solver") is False
    assert _graphs_env("auto", "cuda", "all-vram") is True
    assert _graphs_env("1", "cuda", "solver") is True
    with pytest.raises(ValueError, match="all-resident"):
        _batched_graph_grouping(PagedServeConfig(model="m", graphs=True, max_seqs=4, buckets=(1, 2, 4), placement="solver"))
    for cfg in (PagedServeConfig(model="m", graphs=True, max_seqs=1, buckets=(1, 2, 4), placement="solver"),
                PagedServeConfig(model="m", graphs=True, max_seqs=4, buckets=(1,), placement="solver")):
        assert _batched_graph_grouping(cfg) == {"device_grouping": False, "force_singleton_groups": True}
    assert _flags.DEVICE_GROUPING == [False] and _flags.FORCE_SINGLETON_GROUPS == [True]


def _state(is_hot, g2h):
    return types.SimpleNamespace(is_hot=torch.tensor(is_hot), g2h=torch.tensor(g2h), _all_hot_cache=None)


def test_the_collapse_needs_every_expert_resident_in_identity_order():
    all_hot = hot_residency._HotResidency._all_hot
    assert all_hot(_state([True] * 4, [0, 1, 2, 3])) is True
    assert all_hot(_state([True, True, False, True], [0, 1, -1, 2])) is False       # one expert off the GPU: tiered
    assert all_hot(_state([True] * 4, [1, 0, 2, 3])) is False                        # resident but reordered
    s = _state([True, False], [0, -1])
    assert all_hot(s) is False and s._all_hot_cache is False                        # read once per placement


def _forward_src():
    return inspect.getsource(hot_residency._HotResidency.forward)


def test_the_collapse_is_gated_on_all_hot():
    src = _forward_src()
    gate = src.index("self._all_hot()")
    assert src.index("return self._forward_collapsed(") > gate
    assert "self.collapse_resident" in src[src.rindex("if", 0, gate):gate]


def test_a_tiered_layer_syncs_before_any_branch_on_the_routing():
    """Between the routing mask (``hot_row = self.is_hot[flat]``) and the first ``nonzero`` there is no branch, so the
    sync runs on every tiered call whatever the tokens route to: a capture raises there, never past it."""
    src = _forward_src()
    start = src.index("hot_row = self.is_hot[flat]")
    sync = src.index("hot_row.nonzero(")
    assert start < sync
    between = src[start:sync]
    assert not re.search(r"^\s*(if|elif|while|for)\b", between, re.M), between
    # the only earlier exits are the collapse, the residual re-entry and the diet; none of them is on the tiered path
    # that serve_paged builds (dispatch_diet=False), and the residual re-entry calls this same forward without it
    assert src.index("if self.dispatch_diet:") < start


def test_serve_paged_builds_the_tier_without_the_diet_or_prefetch():
    src = inspect.getsource(serve_paged_mod.build_engine)
    call = src[src.index("enable_hybrid_tier(model"):]
    call = call[:call.index(")") + 1]
    assert "dispatch_diet=False" in call and "collapse_resident=True" in call and "prefetch" not in call
