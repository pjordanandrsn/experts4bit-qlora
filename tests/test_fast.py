# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""[fast] extra: fused grouped-GEMM forward vs the reference expert loop.

Skips cleanly unless CUDA and grouped-nf4-gemm are both present (the extra is
optional by design). Parity bar: the two paths dequantize identical NF4
values; the fused path accumulates in fp32, so agreement is bounded by the
reference path's own bf16 materialization noise.
"""
import pytest
import torch

nf4_grouped = pytest.importorskip("nf4_grouped")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    disable_fast,
    enable_fast,
    fast_available,
)


def _make_experts(E=4, H=128, inter=64, k=2, tokens=16, seed=0, has_gate=True):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * inter if has_gate else inter, H, dtype=torch.float32)
    down = torch.randn(E, H, inter, dtype=torch.float32)
    mod = Experts4bit.from_float(
        gate_up_proj=gate_up.cuda(),
        down_proj=down.cuda(),
        compute_dtype=torch.bfloat16,
        has_gate=has_gate,
    )
    hs = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(tokens, E, device="cuda")
    top_w, top_i = torch.topk(torch.softmax(logits, -1), k=k, dim=-1)
    return mod, hs, top_i, top_w.to(torch.bfloat16)


def test_fast_available():
    assert fast_available()


def _b_rel(a, b):
    """Norm-relative error — per-element rel false-FAILs near-zero cells
    (the hw_contract lesson); magnitude-relative is the honest metric."""
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _oracle(mod, hs, top_i, top_w):
    """fp32 forward over the same dequantized NF4 values — the fidelity anchor."""
    E = mod.num_experts
    x = hs.float()
    out = torch.zeros_like(x)
    for e in range(E):
        w_gu = mod._dequantize_expert(mod.gate_up_proj, mod.gate_up_absmax, mod._gate_up_shape, e, torch.float32)
        w_dn = mod._dequantize_expert(mod.down_proj, mod.down_absmax, mod._down_shape, e, torch.float32)
        tok, pos = torch.where(top_i == e)
        if not len(tok):
            continue
        h = torch.nn.functional.linear(x[tok], w_gu)
        if mod.has_gate:
            g, u = h.chunk(2, -1)
            h = mod.act_fn(g) * u
        else:
            h = mod.act_fn(h)
        h = torch.nn.functional.linear(h, w_dn)
        out.index_add_(0, tok, h * top_w[tok, pos, None].float())
    return out


def test_fused_matches_reference():
    mod, hs, top_i, top_w = _make_experts()
    with torch.no_grad():
        ref = mod(hs, top_i, top_w)
    n = enable_fast(mod)
    assert n == 1
    with torch.no_grad():
        fast = mod(hs, top_i, top_w)
        oracle = _oracle(mod, hs, top_i, top_w)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    # both paths share the exact NF4 grid values; agreement is bf16-noise-bounded
    assert _b_rel(fast, ref) < 1.5e-2
    # P-fid ordering: the fp32-accumulating fused path may not be meaningfully
    # LESS faithful to the fp32 oracle than the bf16-materializing reference
    err_ref = _b_rel(ref, oracle)
    err_fast = _b_rel(fast, oracle)
    assert err_fast <= max(2 * err_ref, 5e-3), (err_fast, err_ref)
    assert disable_fast(mod) == 1
    with torch.no_grad():
        back = mod(hs, top_i, top_w)
    torch.testing.assert_close(back.float(), ref.float(), rtol=0, atol=0)


def test_training_path_falls_back_and_backprops():
    mod, hs, top_i, top_w = _make_experts()
    enable_fast(mod)
    x = hs.clone().requires_grad_(True)
    out = mod(x, top_i, top_w)          # grad required -> reference path
    out.sum().backward()                # fused kernel has no backward; must not crash
    assert x.grad is not None and torch.isfinite(x.grad).all()
    disable_fast(mod)


def test_skips_ineligible_blocksize():
    mod, *_ = _make_experts()
    mod.blocksize = 128                 # simulate a non-64 blocksize module
    assert enable_fast(mod) == 0


def test_uneven_routing_and_empty_experts():
    # force an expert with zero tokens: route everything to experts {0,1}
    mod, hs, top_i, top_w = _make_experts(E=4, k=2, tokens=8, seed=3)
    top_i = torch.stack([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)], 1).cuda()
    with torch.no_grad():
        ref = mod(hs, top_i, top_w)
    enable_fast(mod)
    with torch.no_grad():
        fast = mod(hs, top_i, top_w)
    assert _b_rel(fast, ref) < 1.5e-2
    disable_fast(mod)
