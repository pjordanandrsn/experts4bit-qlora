# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""B1d graph-mode KV append: the device-addressed T=1 path must produce
byte-identical pool state to the host-addressed production append —
including across a block boundary, the case where a baked write offset
(the e4b#227 finding) would silently overwrite one slot. CPU-only."""

import pytest
import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV


def make(device="cpu"):
    try:
        return Fp8PagedKV(3, 2, 64, batch=4, max_tokens_per_seq=48,
                          k_groups=4, device=device)
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"Fp8PagedKV unavailable on {device}: {e}")


def test_graph_append_matches_host_append_across_boundary():
    torch.manual_seed(11)
    ref, gr = make(), make()
    seq = 1
    gr.graph_mode_init(seq=seq, upto_tokens=40)
    assert gr.graph_t1 is True
    steps = 20                                  # crosses the 16-token block
    for _ in range(steps):
        for layer in range(ref.L):
            k = torch.randn(1, ref.H, ref.D)
            v = torch.randn(1, ref.H, ref.D)
            ref.append(layer, seq, k, v)
            gr.append_graph_t1(layer, k.clone(), v.clone())
    assert torch.equal(ref.seq_lens, gr.seq_lens)
    for layer in range(ref.L):
        # compare only the rows this sequence owns -- the ref engine
        # allocated the same rows in the same order (same allocator)
        for pool_r, pool_g in ((ref.kp, gr.kp), (ref.vp, gr.vp)):
            assert torch.equal(pool_r.dev[layer], pool_g.dev[layer])


def test_graph_append_advances_not_bakes():
    # the failure the finding described: writes landing at a FIXED
    # position. After N appends, every block the sequence owns must be
    # non-uniform (a same-slot overwrite leaves later blocks untouched)
    torch.manual_seed(7)
    gr = make()
    seq = 0
    gr.graph_mode_init(seq=seq, upto_tokens=40)
    for t in range(18):
        for layer in range(gr.L):
            gr.append_graph_t1(layer, torch.full((1, gr.H, gr.D), 1.0 + t),
                               torch.full((1, gr.H, gr.D), 2.0 + t))
    assert int(gr.seq_lens[0, seq]) == 18
    rows = gr._rows[(0, seq)]
    assert len(rows) >= 2                       # boundary genuinely crossed
    b0 = gr.vp.dev[0, rows[0]]
    b1 = gr.vp.dev[0, rows[1]]
    assert b0.any() and b1.any()                # BOTH blocks got bytes


def test_graph_mode_requires_pristine_arena():
    gr = make()
    gr.kp.head[0] = 1                           # simulate a moved ring
    with pytest.raises(AssertionError):
        gr.graph_mode_init(seq=0)
