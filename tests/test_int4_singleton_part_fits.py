# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The int4 singleton branch hands the GEMV its preallocated partials buffer only when it fits.

``enable_serve_experts_int4`` sizes ``st["part"]`` for ONE token's ``top_k`` rows. The singleton
branch of ``_fused_over_stack`` also runs at T > 1 (``FORCE_SINGLETON_GROUPS``, the S2 verify
default), where the call has ``T * top_k`` rows. Lane P63 measured the old code on an A2000
writing 16,384 fp32 elements past a 128-row buffer at T = 17
(``bench/p63/rehearsal-a2000/part_oob.json``). These tests pin the choice with the kernel module
stubbed, so they run on any CPU: T = 1 keeps the store's buffer (the capture pattern, unchanged),
a call that needs more rows gets None and the wrapper allocates its own.
"""
import sys
import types

import pytest
import torch

from experts4bit_qlora.engines import hot_residency as hr

N, K, TOP_K, SK = 64, 64, 4, 8


def _stub(monkeypatch, sk_for_rows):
    seen = []
    stub = types.ModuleType("int4_b32")
    stub._sm_count = lambda device: 170
    stub._plan = lambda n, k, r=1, sm_count=128: (128, 4, sk_for_rows(r), 1)
    stub.quant_x_rows = lambda x: (x.float(), None)

    def gemv_int4_b32(xq, xs, packed, scales, eids, n, k, part=None):
        seen.append(part)
        return torch.zeros(xq.shape[0], n, dtype=torch.bfloat16)
    stub.gemv_int4_b32 = gemv_int4_b32
    monkeypatch.setitem(sys.modules, "int4_b32", stub)
    nf4 = types.ModuleType("nf4_grouped")
    nf4.gemm_4bit_grouped = lambda *a, **k: pytest.fail("the NF4 path must not run on the int4 store")
    monkeypatch.setitem(sys.modules, "nf4_grouped", nf4)
    return seen


def _stores(rows_for):
    def one(n, k):
        return {"packed": torch.zeros(2, n, k // 2, dtype=torch.uint8),
                "scales": torch.zeros(2, n, k // 32, dtype=torch.float16), "N": n, "K": k,
                "part": torch.empty(SK * rows_for, n, dtype=torch.float32)}
    return {"gu": one(2 * N, K), "dn": one(K, N)}


def _run(stores, rows):
    x = torch.randn(rows, K)
    ids = torch.zeros(rows, dtype=torch.long)
    # DISTINCT freed sentinels: the branch picks the store by identity (`pk is gu_p`)
    gu_p, gu_a, dn_p, dn_a = (torch.empty(0) for _ in range(4))
    return hr._fused_over_stack(x, ids, gu_p, gu_a, dn_p, dn_a, (2 * N, K, K, N),
                                has_gate=True, act_fn=torch.nn.functional.silu,
                                singleton_groups=True, int4_stores=stores)


def test_one_token_keeps_the_store_buffer(monkeypatch):
    seen = _stub(monkeypatch, lambda r: SK)
    stores = _stores(TOP_K)
    _run(stores, TOP_K)
    assert len(seen) == 2
    assert seen[0] is stores["gu"]["part"] and seen[1] is stores["dn"]["part"]


def test_a_verify_shaped_call_does_not_get_a_buffer_it_would_overrun(monkeypatch):
    seen = _stub(monkeypatch, lambda r: SK)
    _run(_stores(TOP_K), 17 * TOP_K)
    assert seen == [None, None]


def test_a_larger_buffer_is_kept_when_the_plan_shrinks(monkeypatch):
    """A call whose own plan needs FEWER rows than the buffer holds uses the buffer's head, as the
    old code did (the A2000's R term collapses sk at 16+ rows); only a shortfall is replaced."""
    seen = _stub(monkeypatch, lambda r: 1 if r >= 16 else SK)
    stores = _stores(TOP_K)                     # SK * TOP_K = 32 rows
    _run(stores, 2 * TOP_K * 2)                 # 16 rows, sk 1 -> needs 16: fits
    assert seen[0] is stores["gu"]["part"]
    seen.clear()
    _run(stores, 17 * TOP_K)                    # 68 rows, sk 1 -> needs 68: does not
    assert seen == [None, None]


def test_a_floor_cut_without_the_row_aware_plan_uses_its_n_only_plan(monkeypatch):
    """grouped-nf4-gemm 0.30.x (the [fast] floor) has ``_plan(N, K)`` and no ``_sm_count``; its GEMV wrapper plans sk
    from (N, K) alone, and the fit test must use that plan, not fail on the import."""
    seen = _stub(monkeypatch, lambda r: SK)
    stub = sys.modules["int4_b32"]
    del stub._sm_count
    stub._plan = lambda n, k: (128, 4, SK, 1)
    stores = _stores(TOP_K)
    _run(stores, TOP_K)
    assert seen[0] is stores["gu"]["part"]
    seen.clear()
    _run(stores, 17 * TOP_K)
    assert seen == [None, None]


def test_the_real_plan_decides_the_same_way_where_the_kernel_imports():
    """Against the shipped planner (skips where triton does not import): the store's buffer, sized the
    way enable_serve_experts_int4 sizes it, fits one token's rows and not seventeen tokens' rows."""
    real = pytest.importorskip("int4_b32")
    for n, k in ((1536, 2048), (2048, 768), (2048, 2048)):
        _b, _w, sk_store, _k = real._plan(n, k)
        st = {"N": n, "K": k, "part": torch.empty(sk_store * 8, n)}
        for sm in (26, 170):
            real._SM_CACHE["cpu"] = sm
            assert hr._int4_part_or_none(st, 8, "cpu") is st["part"]
            assert hr._int4_part_or_none(st, 17 * 8, "cpu") is None
        real._SM_CACHE.pop("cpu", None)
