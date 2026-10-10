# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``/health``'s ``prefill_routes.seen``: what the forward RAN, not what the environment resolves.

sc2g-prove-2 served gpt-oss-20b with ``prefill_routes`` reading ``k19`` / ``k19`` / ``flash``, and no call took any of
them. The MXFP4 store's rows above the GEMV bound fall to the kept NF4 stacks (K21 below 256 rows under device
grouping), and every gpt-oss layer has sinks, so it keeps the explicit mask. The counters are set where the route is
chosen. These tests tie each label to the function that was actually CALLED (mocked GEMMs record their calls), on CPU:

- the NF4 host-grouped and singleton routes, by row class;
- the MXFP4 store with its NF4 stacks kept takes the NF4 route above the GEMV bound, and with them freed stays on the
  MXFP4 kernel (the case /health could not tell apart before);
- the prefill attention path: ``flash`` on a plain layer, the explicit mask for sinks, a window, or ``math``;
- ``prefill_routes()`` carries both counters.

The device-grouped routes (K19, K21, K25, the captured M-tiles) need CUDA builders and are labelled in the same place.
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402
from experts4bit_qlora.engines import paged_attention as pa  # noqa: E402

SHAPES = (8, 8, 8, 8)


@pytest.fixture
def calls(monkeypatch):
    """Mock both GEMM families (per test, never module-level; see tests/test_singleton_groups.py) and record calls."""
    seen = []
    nf4 = sys.modules.get("nf4_grouped")
    if nf4 is None:
        nf4 = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "nf4_grouped", nf4)

    def _nf4(x, p, a, sizes, eids):
        seen.append("gemm_4bit_grouped")
        return x * 2.0

    monkeypatch.setattr(nf4, "gemm_4bit_grouped", _nf4, raising=False)
    mx = types.SimpleNamespace()                      # no gemv_mxfp4_b32: the v1 grouped GEMM is the route

    def _mx(x, blocks, scales, sizes, eids):
        seen.append("gemm_mxfp4_grouped")
        return x * 3.0

    mx.gemm_mxfp4_grouped = _mx
    monkeypatch.setitem(sys.modules, "mxfp4_grouped", mx)
    monkeypatch.setattr(hr, "ROUTE_SEEN", {})
    return seen


def _run(rows, **kw):
    g = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, 8, generator=g)
    ids = torch.randint(0, 4, (rows,), generator=g)
    return hr._fused_over_stack(x, ids, kw.pop("gu_p", None), None, kw.pop("dn_p", None), None, SHAPES,
                                has_gate=False, act_fn=torch.nn.functional.silu, **kw)


def _mxfp4_store():
    st = {"blocks": torch.zeros(4, 8, 4, dtype=torch.uint8), "scales": torch.zeros(4, 8, 1, dtype=torch.uint8),
          "N": 8, "K": 8}
    return {"kind": "mxfp4", "gu": dict(st), "dn": dict(st)}


def test_nf4_host_grouped_and_singleton_routes_by_row_class(calls):
    _run(12)
    _run(300)
    _run(5, singleton_groups=True)
    assert hr.ROUTE_SEEN == {"nf4_mtile_host|le256": 1, "nf4_mtile_host|gt256": 1, "nf4_singleton|le256": 1}
    assert calls == ["gemm_4bit_grouped"] * 6                       # gate_up + down per call, all on the NF4 GEMM


def test_mxfp4_store_with_nf4_kept_takes_the_nf4_route_above_the_gemv_bound(calls, monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GEMV", "0")
    kept = torch.ones(4)                                              # the NF4 stacks are still resident
    _run(hr._MXFP4_GEMV_ROWS + 4, gu_p=kept, dn_p=kept, int4_stores=_mxfp4_store())
    assert hr.ROUTE_SEEN == {"nf4_mtile_host|le256": 1}
    assert calls == ["gemm_4bit_grouped"] * 2 and "gemm_mxfp4_grouped" not in calls


def test_mxfp4_store_with_nf4_freed_stays_on_the_mxfp4_kernel(calls, monkeypatch):
    monkeypatch.setenv("E4B_MXFP4_GEMV", "0")
    freed = torch.empty(0)                                            # E4B_INT4_KEEP_NF4 unset: sentinels
    _run(hr._MXFP4_GEMV_ROWS + 4, gu_p=freed, dn_p=freed, int4_stores=_mxfp4_store())
    _run(300, gu_p=freed, dn_p=freed, int4_stores=_mxfp4_store())
    assert hr.ROUTE_SEEN == {"mxfp4_grouped_v1|le256": 1, "mxfp4_grouped_v1|gt256": 1}
    assert calls == ["gemm_mxfp4_grouped"] * 4


H_Q, H_KV, D = 4, 2, 16


class _Mod(torch.nn.Module):
    def __init__(self, sliding_window=None, sinks=None):
        super().__init__()
        self.layer_idx = 0
        self.num_key_value_groups = H_Q // H_KV
        self.is_causal = True
        self.sliding_window = sliding_window
        if sinks is not None:
            self.sinks = sinks


def _prefill(mod, T=6):
    pytest.importorskip("transformers")
    g = torch.Generator().manual_seed(T)
    q = torch.randn(1, H_Q, T, D, generator=g)
    k = torch.randn(1, H_KV, T, D, generator=g)
    v = torch.randn(1, H_KV, T, D, generator=g)
    pa.set_context(pa.PagedAttentionContext(kv=None, slots=[0], mode="prefill"))
    try:
        pa.paged_attention_forward(mod, q, k, v, None, scaling=D ** -0.5)
    finally:
        pa.set_context(None)


def test_prefill_attention_path_is_counted_by_what_ran(monkeypatch):
    monkeypatch.setattr(pa, "ATTN_SEEN", {})
    monkeypatch.delenv("E4B_PAGED_PREFILL_ATTN", raising=False)      # flash, the default
    _prefill(_Mod())
    _prefill(_Mod(sinks=torch.zeros(H_Q)))                            # gpt-oss: sinks keep the explicit mask
    _prefill(_Mod(sliding_window=3))
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", "math")
    _prefill(_Mod())
    assert pa.ATTN_SEEN == {"flash": 1, "explicit_mask:sinks": 1, "explicit_mask:window": 1,
                            "explicit_mask:env": 1}


def test_prefill_routes_reports_both_counters(monkeypatch):
    from experts4bit_qlora import serve_paged
    from experts4bit_qlora.engines import glue_fuse
    monkeypatch.setattr(hr, "ROUTE_SEEN", {"mxfp4_k21|le256": 2, "nf4_mtile_captured|gt256": 1})
    monkeypatch.setattr(pa, "ATTN_SEEN", {"explicit_mask:sinks": 24})
    monkeypatch.setattr(hr, "K19_DISPATCH_SEEN", {"chained|lean|gt256": 48})
    monkeypatch.setattr(glue_fuse, "PREFILL_FOLD_SEEN", {"norm": 97, "layer": 48})
    r = serve_paged.prefill_routes()
    assert r["seen"] == {"moe": {"mxfp4_k21|le256": 2, "nf4_mtile_captured|gt256": 1},
                         "prefill_attn": {"explicit_mask:sinks": 24},
                         "moe_k19_dispatch": {"chained|lean|gt256": 48},     # E4B_PREFILL_LEAN_DISPATCH's evidence
                         "prefill_folds": {"layer": 48, "norm": 97}}         # E4B_FUSE_PREFILL_GLUE's
    assert {"int4_prefill", "prefill_attn", "device_grouping"} <= set(r)   # the env-resolved fields stay
