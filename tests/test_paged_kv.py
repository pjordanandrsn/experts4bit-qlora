# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Tiered paged KV (Phase 6): byte identity with a plain concatenated
reference across prefill + decode + block boundaries, zero-copy views in
the everything-fits regime (the G6 ≤2% clause's mechanism), byte identity
through demotion, and structural freedom when the window is unset
(invariant 9)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.paged_kv import TieredPagedKV  # noqa: E402

L, H, D = 3, 2, 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _steps(seed=3):
    """A prefill of 5 tokens then decodes crossing the 16-token block
    boundary twice: token counts 5,1,1,...  (total 37 tokens)."""
    g = torch.Generator().manual_seed(seed)
    steps = []
    for t_new in [5] + [1] * 32:
        ks = torch.randn(1, H, t_new, D, generator=g, dtype=torch.bfloat16)
        vs = torch.randn(1, H, t_new, D, generator=g, dtype=torch.bfloat16)
        steps.append((ks.to(DEV), vs.to(DEV)))
    return steps


def _run_both(cache):
    ref_k = [torch.zeros(1, H, 0, D, dtype=torch.bfloat16, device=DEV)
             for _ in range(L)]
    ref_v = [t.clone() for t in ref_k]
    for ks, vs in _steps():
        for layer in range(L):
            got_k, got_v = cache.update(ks, vs, layer)
            ref_k[layer] = torch.cat([ref_k[layer], ks], dim=2)
            ref_v[layer] = torch.cat([ref_v[layer], vs], dim=2)
            assert torch.equal(got_k, ref_k[layer]), \
                f"K mismatch layer {layer} at t={ref_k[layer].shape[2]}"
            assert torch.equal(got_v, ref_v[layer])
    return ref_k


def test_paged_equals_contiguous_reference_across_block_boundaries():
    cache = TieredPagedKV(L, H, D, device=DEV, max_tokens=64)
    ref = _run_both(cache)
    assert cache.get_seq_length() == ref[0].shape[2] == 37
    assert cache.stats()["gather_returns"] == 0


def test_everything_fits_returns_are_views_of_the_pool():
    cache = TieredPagedKV(L, H, D, device=DEV, max_tokens=64)
    ks = torch.randn(1, H, 4, D, dtype=torch.bfloat16, device=DEV)
    vs = torch.randn_like(ks)
    got_k, _ = cache.update(ks, vs, 0)
    pool_start = cache.pool.dev.data_ptr()
    pool_end = pool_start + cache.pool.dev.numel()
    assert pool_start <= got_k.data_ptr() < pool_end, \
        "everything-fits K must be a VIEW of the device pool, not a copy"


def test_demotion_keeps_bytes_identical_and_counts_gathers():
    cache = TieredPagedKV(L, H, D, device=DEV, max_tokens=64,
                          hot_window=16, host_tokens=64)
    _run_both(cache)                     # byte-equality asserted inside
    s = cache.stats()
    assert s["demotions"] > 0, "window=16 over 37 tokens must demote"
    assert s["settled"] > 0
    assert s["gather_returns"] > 0, "demoted context must stream back"


def test_free_when_unused(two=None):
    cache = TieredPagedKV(L, H, D, device=DEV, max_tokens=64)
    _run_both(cache)
    s = cache.stats()
    assert s["demotions"] == 0 and s["host_reads"] == 0
    assert s["gather_returns"] == 0
    assert cache._side is None, "no side stream without a window"


def test_batched_input_is_refused_loudly():
    cache = TieredPagedKV(L, H, D, device=DEV, max_tokens=64)
    ks = torch.randn(2, H, 1, D, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="batch-1"):
        cache.update(ks, ks, 0)
