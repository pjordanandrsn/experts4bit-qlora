# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""T5 dispatch diet: the diet's index algebra must be BIT-IDENTICAL to the
baseline's ``nonzero``/boolean-indexing algebra — same indices, same order —
across all regimes (all-hot, all-cold, mixed; static and data-dependent
cold splits). The on-box gate is cross-arm token identity; these tests pin
the same claim at the tensor level so a violation fails in CI, not on a
rented box. CPU-only."""

import torch

from experts4bit_qlora.engines.hot_residency import _partition_by_mask
from experts4bit_qlora.engines.hybrid import _HybridTier


def _baseline_split(hot_row):
    hr = hot_row.nonzero(as_tuple=False).view(-1)
    cr = (~hot_row).nonzero(as_tuple=False).view(-1)
    return cr, hr


def test_partition_matches_nonzero_all_regimes():
    g = torch.Generator().manual_seed(7)
    for n in (1, 2, 3, 7, 32, 257):
        for p in (0.0, 0.15, 0.5, 0.9, 1.0):
            mask = torch.rand(n, generator=g) < p
            n_cold, cr, hr = _partition_by_mask(mask)
            cr0, hr0 = _baseline_split(mask)
            assert n_cold == cr0.numel()
            assert torch.equal(cr, cr0)
            assert torch.equal(hr, hr0)
            assert cr.dtype == cr0.dtype and hr.dtype == hr0.dtype


def _stub_tier(is_hot, is_dram):
    st = _HybridTier.__new__(_HybridTier)   # split logic needs only these
    st.is_hot = is_hot
    st.is_dram = is_dram
    st.dispatch_diet = True
    st._cold_static = None
    return st


def _baseline_cold_split(is_dram, flat, cr):
    dmask = is_dram[flat.index_select(0, cr)]
    return cr[~dmask], cr[dmask]


def test_cold_split_static_dram_covers_swappable_shape():
    E = 16
    is_hot = torch.zeros(E, dtype=torch.bool)
    is_hot[:4] = True
    is_dram = torch.ones(E, dtype=torch.bool)   # controller mode: all-true
    st = _stub_tier(is_hot, is_dram)
    flat = torch.randint(0, E, (40,), generator=torch.Generator().manual_seed(1))
    cr = (~is_hot[flat]).nonzero(as_tuple=False).view(-1)
    nr, dr = st._split_cold_diet(flat, cr)
    assert st._cold_static == "dram"
    nr0, dr0 = _baseline_cold_split(is_dram, flat, cr)
    assert torch.equal(nr, nr0) and torch.equal(dr, dr0)
    assert nr.numel() == 0 and torch.equal(dr, cr)


def test_cold_split_static_nvme_when_no_dram_tier():
    E = 12
    is_hot = torch.zeros(E, dtype=torch.bool)
    is_hot[:3] = True
    is_dram = torch.zeros(E, dtype=torch.bool)
    st = _stub_tier(is_hot, is_dram)
    flat = torch.randint(0, E, (30,), generator=torch.Generator().manual_seed(2))
    cr = (~is_hot[flat]).nonzero(as_tuple=False).view(-1)
    nr, dr = st._split_cold_diet(flat, cr)
    assert st._cold_static == "nvme"
    nr0, dr0 = _baseline_cold_split(is_dram, flat, cr)
    assert torch.equal(nr, nr0) and torch.equal(dr, dr0)


def test_cold_split_mixed_matches_boolean_indexing():
    g = torch.Generator().manual_seed(3)
    E = 24
    seen_mixed = 0
    for _ in range(40):
        is_hot = torch.rand(E, generator=g) < 0.25
        is_dram = (~is_hot) & (torch.rand(E, generator=g) < 0.5)
        st = _stub_tier(is_hot, is_dram)
        flat = torch.randint(0, E, (64,), generator=g)
        cr = (~is_hot[flat]).nonzero(as_tuple=False).view(-1)
        nr0, dr0 = _baseline_cold_split(is_dram, flat, cr)
        nr, dr = st._split_cold_diet(flat, cr)
        assert torch.equal(nr, nr0) and torch.equal(dr, dr0)
        # and the cached static answer stays correct on a SECOND call
        nr2, dr2 = st._split_cold_diet(flat, cr)
        assert torch.equal(nr2, nr0) and torch.equal(dr2, dr0)
        seen_mixed += st._cold_static == "mixed"
    assert seen_mixed >= 10   # the sweep genuinely exercised the sort path


def test_weight_flatten_identity():
    # the all-hot fast path replaces top_k_weights[row_token, row_slot]
    # with reshape(-1); they must be the same tensor content exactly
    g = torch.Generator().manual_seed(11)
    for T, k in ((1, 8), (16, 8), (5, 3)):
        w = torch.randn(T, k, generator=g)
        rt = torch.arange(T * k) // k
        rs = torch.arange(T * k) - rt * k
        assert torch.equal(w[rt, rs], w.reshape(-1))
