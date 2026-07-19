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
