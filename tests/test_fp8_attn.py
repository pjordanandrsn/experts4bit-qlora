# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The fp8 attention swap patches structurally, refuses vacuously, and
keeps the projection's numerics inside the format's own error budget."""
import os
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GROUP = 128


def _stub(monkeypatch, calls):
    """Reference implementations of the two kernels, so these tests
    check the MODULE (swap, refusals, shapes, prefill/decode routing)
    without a GPU; the kernels have their own tests in the kernel
    package."""
    stub = types.ModuleType("int4_b32")

    def quant_fp8_rows(w, group=GROUP):
        N, K = w.shape
        wf = w.float().reshape(N, K // group, group)
        s = wf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
        q = (wf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q.reshape(N, K), s.reshape(N, K // group).float()

    def gemv_fp8_rows(x, q, scales, group=GROUP):
        calls["gemv"] += 1
        N, K = q.shape
        deq = (q.float().reshape(N, K // group, group)
               * scales.float()[..., None]).reshape(N, K)
        return (x.float() @ deq.t()).to(torch.bfloat16)

    stub.quant_fp8_rows = quant_fp8_rows
    stub.gemv_fp8_rows = gemv_fp8_rows
    monkeypatch.setitem(sys.modules, "int4_b32", stub)


class ToyAttention(nn.Module):
    def __init__(self, hidden=256, heads=2, d=128):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden, 3 * heads * d, bias=False,
                                  dtype=torch.bfloat16)
        self.o_proj = nn.Linear(heads * d, hidden, bias=False,
                                dtype=torch.bfloat16)


def test_swaps_and_routes_decode_through_the_gemv(monkeypatch):
    from experts4bit_qlora.engines.fp8_attn import (Fp8Linear,
                                                    enable_serve_attn_fp8)
    calls = {"gemv": 0}
    _stub(monkeypatch, calls)
    m = nn.Module()
    m.attn = ToyAttention()
    ref_w = m.attn.qkv_proj.weight.detach().clone()
    assert enable_serve_attn_fp8(m) == 2
    assert isinstance(m.attn.qkv_proj, Fp8Linear)

    x = torch.randn(1, 1, 256, dtype=torch.bfloat16)
    got = m.attn.qkv_proj(x)
    assert calls["gemv"] == 1, "decode row must take the GEMV path"
    assert got.shape == (1, 1, 768) and got.dtype == torch.bfloat16
    ref = (x.float() @ ref_w.float().t())
    # inside the format's own error budget, not bitwise: e4m3 with
    # per-group scales is a lossy store and that loss is the point
    assert (got.float() - ref).abs().max() <= ref.abs().max() * 2 ** -5

    # prefill routes around the GEMV to the dequant-then-matmul side
    big = torch.randn(1, 8, 256, dtype=torch.bfloat16)
    out = m.attn.qkv_proj(big)
    assert calls["gemv"] == 1 and out.shape == (1, 8, 768)


def test_refuses_a_biased_projection(monkeypatch):
    from experts4bit_qlora.engines.fp8_attn import Fp8Linear
    _stub(monkeypatch, {"gemv": 0})
    lin = nn.Linear(256, 256, bias=True, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="bias"):
        Fp8Linear(lin)


def test_refuses_a_group_misaligned_projection(monkeypatch):
    from experts4bit_qlora.engines.fp8_attn import Fp8Linear
    _stub(monkeypatch, {"gemv": 0})
    lin = nn.Linear(100, 256, bias=False, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiple of the scale group"):
        Fp8Linear(lin)


def test_refuses_a_vacuous_enable(monkeypatch):
    from experts4bit_qlora.engines.fp8_attn import enable_serve_attn_fp8
    _stub(monkeypatch, {"gemv": 0})
    m = nn.Module()
    m.mlp = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="vacuous"):
        enable_serve_attn_fp8(m)


def test_lm_head_is_never_touched(monkeypatch):
    """lm_head was ineligible on quality even for int4 (+0.18 ppl); this
    module must leave it alone however it is named."""
    from experts4bit_qlora.engines.fp8_attn import enable_serve_attn_fp8
    _stub(monkeypatch, {"gemv": 0})
    m = nn.Module()
    m.attn = ToyAttention()
    m.lm_head = nn.Linear(256, 512, bias=False, dtype=torch.bfloat16)
    assert enable_serve_attn_fp8(m) == 2
    assert type(m.lm_head) is nn.Linear
