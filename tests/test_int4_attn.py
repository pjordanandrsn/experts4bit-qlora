# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Serving attention on the int4-b32 grid: swap structure, refusal
paths, and (CUDA) forward parity of both branches against the dequant
reference. Skips wholesale until the grouped-nf4-gemm cut carrying
``int4_b32`` is installed; the enable-path arms also skip off-linux,
where that package does not ship triton."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("int4_pack_ref",
                    reason="needs grouped-nf4-gemm with int4_b32")

from torch import nn  # noqa: E402

from experts4bit_qlora.engines.int4_attn import (  # noqa: E402
    Int4Linear,
    enable_serve_attn_int4,
)

# The enable-path arms call enable_serve_attn_int4, which imports
# ``int4_b32`` -- and its module scope needs triton, declared linux-only
# by grouped-nf4-gemm. Gate on triton's absence, NOT importorskip on
# int4_b32: where triton exists, a broken kernel module must FAIL these
# arms, never skip them.
needs_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="enable path imports int4_b32, which needs triton (linux-only)",
)


class _Attention(nn.Module):          # structural: name ends in Attention
    def __init__(self, bias=False):
        super().__init__()
        self.qkv_proj = nn.Linear(64, 160, bias=bias)
        self.o_proj = nn.Linear(128, 64, bias=bias)
        self.q_norm = nn.LayerNorm(64)   # non-Linear child: untouched


class _Model(nn.Module):
    def __init__(self, n=2, bias=False):
        super().__init__()
        self.layers = nn.ModuleList(_Attention(bias) for _ in range(n))
        self.lm_head = nn.Linear(64, 512, bias=False)


@needs_triton
def test_swap_counts_and_lm_head_untouched():
    m = _Model(n=3)
    n = enable_serve_attn_int4(m)
    assert n == 6
    for lyr in m.layers:
        assert isinstance(lyr.qkv_proj, Int4Linear)
        assert isinstance(lyr.o_proj, Int4Linear)
        assert isinstance(lyr.q_norm, nn.LayerNorm)
    assert type(m.lm_head) is nn.Linear      # NEVER on this grid (+0.18 ppl)


@needs_triton
def test_bias_is_carried_not_refused():
    """gpt-oss's q/k/v/o carry biases: the int4 store keeps the weight on
    the grid and the bias in bf16, added after the GEMV / matmul. The
    swap must count the biased projections and the module must expose
    the bias it carries."""
    m = _Model(bias=True)
    n = enable_serve_attn_int4(m)
    assert n == 4
    for layer in m.layers:
        assert layer.qkv_proj.bias is not None and layer.qkv_proj.bias.dtype == torch.bfloat16
        assert layer.o_proj.bias is not None


def _cpu_kernel_stubs(monkeypatch, calls):
    """A CPU reference of the kernel package so the ROUTE (which path a
    given row count takes) is tested where CI runs, not skipped."""
    import sys
    import types
    pack_ref = types.ModuleType("int4_pack_ref")

    def pack_int4_b32(w):
        N, K = w.shape
        b = w.float().reshape(N, K // 32, 32)
        s = b.abs().amax(-1).clamp_min(1e-12) / 7.0
        q = ((b / s[..., None]).round().clamp(-8, 7) + 8).to(torch.uint8)
        q = q.reshape(N, K)
        return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous(), s.half()

    def dequant_int4_ref(packed, scales, N, K):
        lo = (packed & 0xF).to(torch.int16) - 8
        hi = ((packed >> 4) & 0xF).to(torch.int16) - 8
        q = torch.stack([lo, hi], -1).reshape(N, K).float()
        return (q.reshape(N, K // 32, 32) * scales.float()[..., None]).reshape(N, K)
    pack_ref.pack_int4_b32 = pack_int4_b32
    pack_ref.dequant_int4_ref = dequant_int4_ref

    k = types.ModuleType("int4_b32")

    def gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=None):
        calls.append(("gemv", int(eids.numel()), tuple(part.shape)))
        w = dequant_int4_ref(packed[0].reshape(N, K // 2),
                             scales[0].reshape(N, K // 32), N, K)
        return (xq.float() @ w.t()).to(torch.bfloat16)

    def quant_x_rows(x):
        return x, torch.ones(x.shape[0])

    def _plan(N, K):
        return 128, 4, 2, 1
    k.gemv_int4_b32 = gemv_int4_b32
    k.quant_x_rows = quant_x_rows
    k._plan = _plan
    for name, mod in (("int4_pack_ref", pack_ref), ("int4_b32", k)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_one_row_on_the_gemv_batches_on_cached_bf16(monkeypatch):
    """rows == 1 takes the GEMV; rows > 1 takes a dequantised bf16 weight
    built ONCE and reused, never the GEMV (which re-streams the weight per
    row: x0.90 at B=16 on a 5090) and never a per-call dequant (x0.52)."""
    calls = []
    _cpu_kernel_stubs(monkeypatch, calls)
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    torch.manual_seed(3)
    lin = nn.Linear(64, 96, bias=False, dtype=torch.bfloat16)
    m = Int4Linear(lin)
    x = torch.randn(16, 64, dtype=torch.bfloat16)
    assert m._bf16_cache is None
    y16 = m(x)
    assert calls == []                                   # no GEMV for 16 rows
    w = m._bf16_cache
    assert w is not None and w.shape == (96, 64) and w.dtype == torch.bfloat16
    m(x)
    assert m._bf16_cache is w                            # built once, reused
    assert calls == []
    # the cached weight is the same int4 values the GEMV serves
    import int4_pack_ref
    ref = int4_pack_ref.dequant_int4_ref(m.packed[0], m.scales[0], 96, 64)
    assert torch.equal(w.float(), ref.to(torch.bfloat16).float())
    assert torch.allclose(y16.float(), (x.float() @ ref.t()), rtol=2e-2, atol=2e-2)
    # one row still takes the GEMV with the construction-time buffers
    m(x[:1])
    assert calls[-1] == ("gemv", 1, (2, 96))


def test_bias_is_added_after_the_gemv_and_the_matmul(monkeypatch):
    """A biased projection: the weight goes on the grid, the bias rides in
    bf16 and is added on BOTH routes (one row -> GEMV, many rows -> the
    cached bf16 matmul), so the swapped module equals the Linear it
    replaced up to the grid's own rounding."""
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    calls = []
    _cpu_kernel_stubs(monkeypatch, calls)
    torch.manual_seed(3)
    lin = nn.Linear(64, 96, bias=True)
    with torch.no_grad():
        lin.weight.mul_(0.05)
        lin.bias.copy_(torch.randn(96))
    q = Int4Linear(lin)
    assert q.bias is not None and q.bias.dtype == torch.bfloat16
    x1 = torch.randn(1, 64).to(torch.bfloat16)
    ref1 = lin(x1.float())
    got1 = q(x1)
    assert calls and calls[-1][0] == "gemv"
    assert torch.allclose(got1.float(), ref1, rtol=2 ** -4, atol=2 ** -3), (got1.float() - ref1).abs().max()
    # the bias is the whole difference between a biased and an unbiased store
    lin0 = nn.Linear(64, 96, bias=False)
    with torch.no_grad():
        lin0.weight.copy_(lin.weight)
    q0 = Int4Linear(lin0)
    assert q0.bias is None
    assert torch.allclose((q(x1) - q0(x1)).float(), lin.bias.to(torch.bfloat16).float().expand(1, 96), rtol=2 ** -6, atol=2 ** -6)
    xb = torch.randn(4, 64).to(torch.bfloat16)
    gotb = q(xb)
    assert torch.allclose(gotb.float(), lin(xb.float()), rtol=2 ** -4, atol=2 ** -3)
    assert torch.allclose((q(xb) - q0(xb)).float(), lin.bias.to(torch.bfloat16).float().expand(4, 96), rtol=2 ** -6, atol=2 ** -6)
