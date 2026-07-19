# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hot-residency correctness: partition experts into a resident GPU hot-stack +
a streamed CPU cold-stack, and prove the split forward reproduces the reference
all-experts forward. Skips unless CUDA + grouped-nf4-gemm are present.

The two paths decode the same NF4 values through the same fused kernel; the
partition and cross-device (CPU-stream) recombine must not change the result
beyond bf16 epilogue noise.
"""
import pytest
import torch

pytest.importorskip("nf4_grouped")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    disable_hot_residency,
    enable_hot_residency,
    hot_residency_available,
)


def _make(E=8, H=128, inter=64, k=3, tokens=24, seed=0, has_gate=True):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * inter if has_gate else inter, H)
    down = torch.randn(E, H, inter)
    mod = Experts4bit.from_float(gate_up_proj=gate_up.cuda(), down_proj=down.cuda(),
                                 compute_dtype=torch.bfloat16, has_gate=has_gate)
    hs = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
    tw, ti = torch.topk(torch.softmax(torch.randn(tokens, E, device="cuda"), -1), k=k, dim=-1)
    return mod, hs, ti, tw.to(torch.bfloat16)


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def test_hot_residency_available():
    assert hot_residency_available()


def test_split_matches_reference():
    mod, hs, ti, tw = _make()
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    # hot = experts {0,1,2,3} resident on GPU; {4,5,6,7} streamed from CPU
    n = enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
    assert n == 1
    st = mod._hot_residency
    assert st.h_gu_p.is_cuda and not st.c_gu_p.is_cuda  # hot resident, cold on host
    with torch.no_grad():
        got = mod(hs, ti, tw)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert _b_rel(got, ref) < 1.5e-2, _b_rel(got, ref)
    assert disable_hot_residency(mod) == 1
    with torch.no_grad():
        back = mod(hs, ti, tw)
    torch.testing.assert_close(back.float(), ref.float(), rtol=0, atol=0)


def test_all_hot_equals_all_cold():
    # extremes must both equal the reference: everything resident, or everything streamed
    mod, hs, ti, tw = _make(seed=2)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    for hot in (torch.arange(8), torch.tensor([], dtype=torch.long)):
        enable_hot_residency(mod, [hot], device="cuda")
        with torch.no_grad():
            got = mod(hs, ti, tw)
        assert _b_rel(got, ref) < 1.5e-2, (hot.numel(), _b_rel(got, ref))
        disable_hot_residency(mod)


def test_expert_with_no_tokens_and_uneven_split():
    # route only to a mix that leaves some hot and some cold experts unused
    mod, hs, ti, tw = _make(E=8, k=2, tokens=16, seed=5)
    ti = torch.stack([torch.zeros(16, dtype=torch.long),           # all hit expert 0 (hot)
                      torch.full((16,), 5, dtype=torch.long)], 1).cuda()  # and expert 5 (cold)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
    with torch.no_grad():
        got = mod(hs, ti, tw)
    assert _b_rel(got, ref) < 1.5e-2, _b_rel(got, ref)
    disable_hot_residency(mod)


def test_training_falls_back():
    mod, hs, ti, tw = _make(seed=7)
    enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
    x = hs.clone().requires_grad_(True)
    out = mod(x, ti, tw)            # grad required -> reference path (no kernel backward)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    disable_hot_residency(mod)


# ---- protection tests for the review-found failure modes ----

def test_hot_sets_alignment_survives_skipped_module():
    # two modules; make the FIRST ineligible (blocksize) — the SECOND must still
    # receive hot_sets[1], not hot_sets[0] (alignment is by module order)
    import torch.nn as nn
    m1, *_ = _make(seed=11)
    m2, hs, ti, tw = _make(seed=12)
    seq = nn.Sequential(m1, m2)
    m1.blocksize = 128  # ineligible
    with torch.no_grad():
        ref = m2(hs, ti, tw)
    n = enable_hot_residency(seq, [torch.tensor([0]), torch.tensor([0, 1, 2, 3])], device="cuda")
    assert n == 1
    assert not hasattr(m1, "_hot_residency")
    assert m2._hot_residency.hot_ids.numel() == 4  # got ITS entry, not m1's 1-id set
    with torch.no_grad():
        got = m2(hs, ti, tw)
    assert _b_rel(got, ref) < 1.5e-2
    disable_hot_residency(seq)
    m1.blocksize = 64


def test_short_hot_sets_raises():
    import torch.nn as nn
    m1, *_ = _make(seed=13)
    m2, *_ = _make(seed=14)
    seq = nn.Sequential(m1, m2)
    with pytest.raises(ValueError, match="hot_sets has 1"):
        enable_hot_residency(seq, [torch.tensor([0])], device="cuda")


def test_out_of_range_hot_id_raises():
    mod, *_ = _make(seed=15)
    with pytest.raises(ValueError, match="hot ids must lie"):
        enable_hot_residency(mod, [torch.tensor([-1, 2])], device="cuda")
    with pytest.raises(ValueError, match="hot ids must lie"):
        enable_hot_residency(mod, [torch.tensor([0, 8])], device="cuda")  # E=8 -> max valid 7


def test_compute_dtype_guard_reads_live_not_snapshot():
    mod, hs, ti, tw = _make(seed=16)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda")
    mod.compute_dtype = torch.float32                    # changed AFTER enable
    with torch.no_grad():
        got = mod(hs, ti, tw)                            # live-read must trigger the fallback
    assert got.device == hs.device                        # output on the input's device
    assert _b_rel(got, ref) < 1.5e-2, _b_rel(got, ref)
    mod.compute_dtype = torch.bfloat16
    disable_hot_residency(mod)


def test_output_on_input_device():
    mod, hs, ti, tw = _make(seed=20)
    enable_hot_residency(mod, [torch.tensor([0, 1, 2, 3])], device="cuda")
    with torch.no_grad():
        got = mod(hs, ti, tw)
    assert got.device == hs.device and got.dtype == hs.dtype
    disable_hot_residency(mod)


def test_longer_hot_sets_raises():
    import torch.nn as nn
    m1, *_ = _make(seed=21)
    seq = nn.Sequential(m1)                              # 1 module
    with pytest.raises(ValueError, match="hot_sets has 2"):
        enable_hot_residency(seq, [torch.tensor([0]), torch.tensor([1])], device="cuda")


def test_mutual_exclusion_with_fast_both_orders():
    from experts4bit_qlora import disable_fast, enable_fast
    mod, hs, ti, tw = _make(seed=17)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    # fast first -> hot refuses; disable fast -> hot proceeds
    assert enable_fast(mod) == 1
    assert enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda") == 0
    assert disable_fast(mod) == 1
    assert enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda") == 1
    # hot active -> fast refuses; full unwind restores the stock forward exactly
    assert enable_fast(mod) == 0
    assert disable_hot_residency(mod) == 1
    with torch.no_grad():
        back = mod(hs, ti, tw)
    torch.testing.assert_close(back.float(), ref.float(), rtol=0, atol=0)


def test_enable_idempotent_and_disable_twice():
    mod, hs, ti, tw = _make(seed=18)
    assert enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda") == 1
    assert enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda") == 0  # already on
    assert disable_hot_residency(mod) == 1
    assert disable_hot_residency(mod) == 0


def test_reenable_with_new_hot_sets_retunes():
    mod, hs, ti, tw = _make(seed=19)
    with torch.no_grad():
        ref = mod(hs, ti, tw)
    assert enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda") == 1
    # same set -> idempotent no-op; different set -> partition rebuilt in place
    assert enable_hot_residency(mod, [torch.tensor([0, 1])], device="cuda") == 0
    assert enable_hot_residency(mod, [torch.tensor([4, 5, 6])], device="cuda") == 1
    assert mod._hot_residency.hot_ids.tolist() == [4, 5, 6]
    with torch.no_grad():
        got = mod(hs, ti, tw)
    assert _b_rel(got, ref) < 1.5e-2
    assert disable_hot_residency(mod) == 1
