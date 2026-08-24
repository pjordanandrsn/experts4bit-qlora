# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""B1c collapse: the placement-static predicate and its algebra
(PREREG-b1c). The bitwise forward equality is the on-box G1 gate; CI
pins the parts that decide WHETHER the collapse may fire and the index
identities it relies on. CPU-only."""

import torch

from experts4bit_qlora.engines.hot_residency import (_HotResidency,
                                                     _partition_by_mask)


def _stub(E, hot_ids, device="cpu"):
    st = _HotResidency.__new__(_HotResidency)
    st.is_hot = torch.zeros(E, dtype=torch.bool)
    st.is_hot[hot_ids] = True
    g2h = torch.full((E,), -1, dtype=torch.long)
    g2h[hot_ids] = torch.arange(len(hot_ids))
    st.g2h = g2h
    st._all_hot_cache = None
    return st


def test_all_hot_predicate_true_on_full_identity():
    E = 16
    st = _stub(E, torch.arange(E))
    assert st._all_hot() is True
    assert st._all_hot_cache is True          # cached, no re-inspection


def test_all_hot_predicate_false_on_subset():
    E = 16
    st = _stub(E, torch.arange(6))
    assert st._all_hot() is False


def test_all_hot_predicate_false_on_reordered_hot_stack():
    # every expert hot but the stack NOT in identity order: the
    # collapse must refuse (flat would mis-index the stack)
    E = 8
    st = _stub(E, torch.arange(E))
    st.g2h = torch.flip(torch.arange(E), [0])
    assert st._all_hot() is False


def test_finish_ids_builds_identity_g2h_when_all_hot():
    # the real construction path: _finish_ids with hot = all experts
    # must produce the identity map the collapse relies on
    st = _HotResidency.__new__(_HotResidency)
    st.device = torch.device("cpu")
    E = 12
    st._finish_ids(E, torch.arange(E), torch.tensor([], dtype=torch.long))
    st._all_hot_cache = None
    assert st._all_hot() is True


def test_row_major_cell_enumeration():
    # the collapse's reshape-scatter identity: rt/rs enumerate (T, k)
    # cells row-major, so writing dn in input order IS the [T, k, :]
    # layout the baseline's index_put_ produces
    for T, k in ((1, 8), (3, 4)):
        rt = torch.arange(T * k) // k
        rs = torch.arange(T * k) - rt * k
        ref = torch.zeros(T, k, dtype=torch.long)
        vals = torch.arange(T * k)
        ref.index_put_((rt, rs), vals)
        assert torch.equal(ref.view(-1), vals)


def test_partition_helper_still_matches_nonzero():
    # regression guard: the collapse coexists with the diet's helper
    g = torch.Generator().manual_seed(5)
    mask = torch.rand(64, generator=g) < 0.5
    n_cold, cr, hr = _partition_by_mask(mask)
    assert torch.equal(hr, mask.nonzero(as_tuple=False).view(-1))
    assert torch.equal(cr, (~mask).nonzero(as_tuple=False).view(-1))
