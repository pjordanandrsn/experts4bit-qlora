# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""B1d graph-mode KV append: the device-addressed T=1 path must produce
byte-identical pool state to the host-addressed production append —
including across a block boundary, the case where a baked write offset
(the e4b#227 finding) would silently overwrite one slot. CPU-only."""

import pytest
import torch

pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402


def make(device="cpu"):
    return Fp8PagedKV(3, 2, 64, batch=4, max_tokens_per_seq=48,
                      k_groups=4, device=device)


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


def test_batch_graph_append_matches_per_slot(make=None):
    """PREREG-bv3: append_graph_bt1 over B slots must leave pool bytes
    and seq_lens exactly equal to B separate single-slot appends --
    the batch form is a loop of the certified path, and this pins it."""
    import experts4bit_qlora.engines.fp8_paged_kv as mod
    torch.manual_seed(11)
    B, H, D, steps = 3, 2, 64, 5

    def _mk():
        kv = mod.Fp8PagedKV(n_layers=1, n_kv_heads=H, head_dim=D,
                            batch=B, max_tokens_per_seq=64,
                            device="cpu")
        for s in range(B):
            kv.append(0, s, torch.randn(4, H, D), torch.randn(4, H, D))
        return kv

    torch.manual_seed(11)
    a = _mk()
    torch.manual_seed(11)
    b = _mk()
    a.graph_mode_init_batch(list(range(B)), upto_tokens=32)
    b.graph_mode_init(seq=0, upto_tokens=32)
    for s in range(1, B):
        for layer in range(1):
            b._ensure_blocks(layer, s, (32 - 1) // b.bt)
    for t in range(steps):
        torch.manual_seed(100 + t)
        k = torch.randn(B, H, D)
        v = torch.randn(B, H, D)
        a.append_graph_bt1(0, k, v)
        for s in range(B):
            b._g_seq = s
            b.append_graph_t1(0, k.narrow(0, s, 1), v.narrow(0, s, 1))
    assert torch.equal(a.seq_lens, b.seq_lens)
    assert torch.equal(a.kp.dev[0], b.kp.dev[0]), "K pool bytes differ"
    assert torch.equal(a.vp.dev[0], b.vp.dev[0]), "V pool bytes differ"
    # every slot advanced by exactly `steps`
    assert a.seq_lens[0].tolist() == [4 + steps] * B


def test_batch_graph_append_row_count_mismatch_refuses():
    import experts4bit_qlora.engines.fp8_paged_kv as mod
    kv = mod.Fp8PagedKV(n_layers=1, n_kv_heads=1, head_dim=64,
                        batch=2, max_tokens_per_seq=32, device="cpu")
    for s in range(2):
        kv.append(0, s, torch.randn(2, 1, 64), torch.randn(2, 1, 64))
    kv.graph_mode_init_batch([0, 1], upto_tokens=16)
    with pytest.raises(AssertionError):
        kv.append_graph_bt1(0, torch.randn(3, 1, 64),
                            torch.randn(3, 1, 64))
