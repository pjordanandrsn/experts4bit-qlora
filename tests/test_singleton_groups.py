# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""AMENDMENT-b1d-capture: the singleton-groups dispatch must be
value-identical to the grouped dispatch — the whole sort/group/unsort
algebra collapses away without changing any row's arithmetic. Tested
with DUPLICATE ids (harder than the exact T=1 case) through a mocked
GEMM so CI needs no CUDA.

The mock is installed per-test via monkeypatch, never module-level:
`sys.modules.setdefault` would silently NOT install under a full
.[test] install where an earlier test already imported the real
`nf4_grouped` (the mock loses), and a module-level insert would LEAK
the fake into later real-kernel tests in the other order (Bugbot,
e4b#229). monkeypatch covers both directions and auto-restores."""

import sys
import types

import pytest
import torch

from experts4bit_qlora.engines.hot_residency import _fused_over_stack

SHAPES = (8, 16, 8, 8)


def _fake_gemm(x, p, a, sizes, eids):
    ids = eids if torch.is_tensor(eids) else torch.as_tensor(eids)
    ids = ids.to(torch.float32)
    per_row = ids.repeat_interleave(torch.as_tensor(sizes))
    return x * (per_row.unsqueeze(1) + 2.0)


@pytest.fixture
def mock_gemm(monkeypatch):
    mod = sys.modules.get("nf4_grouped")
    if mod is None:
        mod = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "nf4_grouped", mod)
    monkeypatch.setattr(mod, "gemm_4bit_grouped", _fake_gemm,
                        raising=False)
    return mod


def test_singleton_matches_grouped_plain(mock_gemm):
    torch.manual_seed(3)
    x = torch.randn(8, 8)
    ids = torch.tensor([3, 1, 3, 0, 2, 1, 5, 5])
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=False, act_fn=torch.nn.functional.silu)
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=False, act_fn=torch.nn.functional.silu,
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_singleton_matches_grouped_gated(mock_gemm):
    torch.manual_seed(4)
    x = torch.randn(6, 8)
    ids = torch.tensor([2, 2, 0, 4, 4, 4])
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu)
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_singleton_matches_grouped_gptoss_epilogue(mock_gemm):
    torch.manual_seed(5)
    # widths matter here: the mock GEMM preserves width, and the gptoss
    # branch chunks gu in half before the down GEMM -- so gu_bias is 2x
    # the dn width or the bias add fails on shape, masking the equality
    x = torch.randn(5, 16)
    ids = torch.tensor([1, 0, 1, 3, 0])
    gu_b = torch.randn(6, 16)
    dn_b = torch.randn(6, 8)
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          gptoss=(gu_b, dn_b, 1.702, 7.0))
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          gptoss=(gu_b, dn_b, 1.702, 7.0),
                          singleton_groups=True)
    assert torch.equal(g, s)


def _mxfp4_reference_gemm(a_cat, blocks, scales, sizes, expert_ids):
    """The kernel side's contract, in torch: ``out[t] = a[t] @ dequant(B[e(t)]).T``
    with blocks [E, N, K//2] u8 + scales [E, N, K//32] e8m0, bf16 out."""
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4
    E, N, half = blocks.shape
    K = half * 2
    ids = expert_ids if torch.is_tensor(expert_ids) else torch.as_tensor(expert_ids)
    per_row = ids.repeat_interleave(torch.as_tensor(sizes))
    w_kn = dequantize_mxfp4(blocks.reshape(E, N, K // 32, 16), scales, dtype=torch.float32)  # [E, K, N]
    out = torch.stack([a_cat[t].float() @ w_kn[int(per_row[t])] for t in range(a_cat.shape[0])])
    return out.to(torch.bfloat16)


def test_mxfp4_store_serves_gptoss_through_the_grouped_mxfp4_gemm(mock_gemm, monkeypatch):
    """A native-MXFP4 store (gpt-oss) routes every projection through the
    kernel side's grouped MXFP4 GEMM on the singleton contract, and the
    forward -- store bytes + bias epilogue + clamped GLU -- equals gpt-oss's
    own expert math on the dequantised weights."""
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4
    mx = sys.modules.get("mxfp4_grouped")
    if mx is None:
        mx = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "mxfp4_grouped", mx)
    monkeypatch.setattr(mx, "gemm_mxfp4_grouped", _mxfp4_reference_gemm, raising=False)
    torch.manual_seed(11)
    E, H, inter = 3, 64, 32
    g = torch.Generator().manual_seed(2)
    gu_blocks = torch.randint(0, 256, (E, 2 * inter, H // 32, 16), generator=g, dtype=torch.uint8)
    gu_scales = torch.randint(122, 128, (E, 2 * inter, H // 32), generator=g, dtype=torch.uint8)
    dn_blocks = torch.randint(0, 256, (E, H, inter // 32, 16), generator=g, dtype=torch.uint8)
    dn_scales = torch.randint(122, 128, (E, H, inter // 32), generator=g, dtype=torch.uint8)
    gu_bias_i = torch.randn(E, 2 * inter) * 0.1                       # interleaved, as released
    dn_bias = torch.randn(E, H) * 0.1
    # the store: rows de-interleaved (gate first), flattened
    gub = torch.cat([gu_blocks[:, 0::2], gu_blocks[:, 1::2]], 1).reshape(E, 2 * inter, H // 2)
    gus = torch.cat([gu_scales[:, 0::2], gu_scales[:, 1::2]], 1)
    stores = {"kind": "mxfp4",
              "gu": {"blocks": gub, "scales": gus, "N": 2 * inter, "K": H},
              "dn": {"blocks": dn_blocks.reshape(E, H, inter // 2), "scales": dn_scales, "N": H, "K": inter}}
    gu_b = torch.cat([gu_bias_i[:, 0::2], gu_bias_i[:, 1::2]], 1)   # as the loader stores it
    x = (torch.randn(5, H) * 0.5).to(torch.bfloat16)
    ids = torch.tensor([1, 0, 2, 2, 0])
    freed_gu, freed_dn = torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8)
    got = _fused_over_stack(x, ids, freed_gu, None, freed_dn, None, (2 * inter, H, H, inter),
                            has_gate=True, act_fn=torch.nn.functional.silu,
                            gptoss=(gu_b, dn_bias, 1.702, 7.0), int4_stores=stores)
    # gpt-oss's own math on the dequantised (module-layout) weights
    W_gu = dequantize_mxfp4(gu_blocks, gu_scales, dtype=torch.float32)   # [E, H, 2I] interleaved
    W_dn = dequantize_mxfp4(dn_blocks, dn_scales, dtype=torch.float32)   # [E, I, H]
    # the same dtype path the kernel contract imposes: fp32 accumulate,
    # bf16 out of each GEMM, the bias add and the GLU in bf16 (as the
    # fused forward and _GptOssForwardMixin both run them at compute dtype)
    want = []
    for t in range(x.shape[0]):
        e = int(ids[t])
        gu = (x[t].float() @ W_gu[e]).to(torch.bfloat16) + gu_bias_i[e].to(torch.bfloat16)
        gate, up = gu[0::2], gu[1::2]
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        h = (up + 1) * (gate * torch.sigmoid(gate * 1.702))
        dn = (h.float() @ W_dn[e]).to(torch.bfloat16) + dn_bias[e].to(torch.bfloat16)
        want.append(dn)
    want = torch.stack(want)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    assert torch.allclose(got.float(), want.float(), rtol=2 ** -7, atol=2 ** -6), \
        (got.float() - want.float()).abs().max()


def _ref_quant_x_rows(x):
    """int4-b32 activation quantisation, reference: int8 per 32-wide block."""
    R, K = x.shape
    xb = x.float().reshape(R, K // 32, 32)
    s = xb.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    xq = torch.round(xb / s).clamp(-127, 127).to(torch.int8).reshape(R, K)
    return xq, s.reshape(R, K // 32).contiguous()


def _ref_gemv_mxfp4_b32(xq, xs, blocks, scales, eids, N, K, part=None):
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4
    E = blocks.shape[0]
    w_kn = dequantize_mxfp4(blocks.reshape(E, N, K // 32, 16), scales, dtype=torch.float32)  # [E, K, N]
    a = xq.float() * xs.repeat_interleave(32, dim=1)
    return torch.stack([a[r] @ w_kn[int(eids[r])] for r in range(xq.shape[0])]).to(torch.bfloat16)


def test_mxfp4_store_decode_rows_take_the_b32_gemv(mock_gemm, monkeypatch):
    """With the kernel side's decode-grade GEMV present, decode rows on
    the MXFP4 store go through it (quantised int8 rows, split-K), not
    the v1 grouped GEMM; E4B_MXFP4_GEMV=0 forces v1 (the A/B arm), and
    the two routes agree to the activation-quantisation tolerance."""
    mx = sys.modules.get("mxfp4_grouped")
    if mx is None:
        mx = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "mxfp4_grouped", mx)
    calls = {"gemv": 0, "gemm": 0}

    def gemm(*a, **k):
        calls["gemm"] += 1
        return _mxfp4_reference_gemm(*a, **k)

    def gemv(*a, **k):
        calls["gemv"] += 1
        return _ref_gemv_mxfp4_b32(*a, **k)
    monkeypatch.setattr(mx, "gemm_mxfp4_grouped", gemm, raising=False)
    monkeypatch.setattr(mx, "gemv_mxfp4_b32", gemv, raising=False)
    monkeypatch.setitem(sys.modules, "int4_b32",
                        types.SimpleNamespace(quant_x_rows=_ref_quant_x_rows))
    g = torch.Generator().manual_seed(5)
    E, H, inter = 3, 64, 32
    gub = torch.randint(0, 256, (E, 2 * inter, H // 2), generator=g, dtype=torch.uint8)
    gus = torch.randint(122, 128, (E, 2 * inter, H // 32), generator=g, dtype=torch.uint8)
    dnb = torch.randint(0, 256, (E, H, inter // 2), generator=g, dtype=torch.uint8)
    dns = torch.randint(122, 128, (E, H, inter // 32), generator=g, dtype=torch.uint8)
    stores = {"kind": "mxfp4",
              "gu": {"blocks": gub, "scales": gus, "N": 2 * inter, "K": H},
              "dn": {"blocks": dnb, "scales": dns, "N": H, "K": inter}}
    gu_b, dn_b = torch.randn(E, 2 * inter) * 0.1, torch.randn(E, H) * 0.1
    x = (torch.randn(4, H) * 0.5).to(torch.bfloat16)
    ids = torch.tensor([2, 0, 1, 2])
    # distinct freed stacks: identity dispatch (`pk is gu_p`) names the slot
    freed_gu, freed_dn = torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8)
    kw = dict(has_gate=True, act_fn=torch.nn.functional.silu,
              gptoss=(gu_b, dn_b, 1.702, 7.0), int4_stores=stores)
    shapes = (2 * inter, H, H, inter)
    monkeypatch.delenv("E4B_MXFP4_GEMV", raising=False)
    got = _fused_over_stack(x, ids, freed_gu, None, freed_dn, None, shapes, **kw)
    assert calls == {"gemv": 2, "gemm": 0}, calls
    monkeypatch.setenv("E4B_MXFP4_GEMV", "0")
    v1 = _fused_over_stack(x, ids, freed_gu, None, freed_dn, None, shapes, **kw)
    assert calls == {"gemv": 2, "gemm": 2}, calls
    # the two routes differ by the int8 per-32 activation quantisation of
    # the decode GEMV (the kernel's own exactness is the kernel side's
    # interpreter gate); here the check is that they compute the same
    # projection, not the same rounding
    rel = ((got.float() - v1.float()).norm() / v1.float().norm()).item()
    assert rel < 0.05, rel
    # a kernel cut without the GEMV falls back to v1 without a flag
    monkeypatch.delenv("E4B_MXFP4_GEMV", raising=False)
    monkeypatch.delattr(mx, "gemv_mxfp4_b32", raising=False)
    _fused_over_stack(x, ids, freed_gu, None, freed_dn, None, shapes, **kw)
    assert calls == {"gemv": 2, "gemm": 4}, calls
