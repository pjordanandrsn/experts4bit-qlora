# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The RMSNorm fusion patches structurally and falls through safely."""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue  # noqa: E402


class ToyRMSNorm(torch.nn.Module):
    def __init__(self, H=32, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)
                                 + self.variance_epsilon)
                * self.weight.float()).to(x.dtype)


def _stub(monkeypatch, calls):
    stub = types.ModuleType("int4_b32")

    def rmsnorm_rows(x, w, eps):
        calls["fused"] += 1
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                * w.float()).to(torch.bfloat16)
    stub.rmsnorm_rows = rmsnorm_rows
    monkeypatch.setitem(sys.modules, "int4_b32", stub)


def test_patches_and_falls_through(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    calls = {"fused": 0}
    _stub(monkeypatch, calls)
    m = torch.nn.Module()
    m.n1 = ToyRMSNorm()
    m.n2 = ToyRMSNorm()
    assert fuse_t1_glue(m) == 2
    x = torch.randn(1, 1, 32, dtype=torch.bfloat16)
    y = m.n1(x)
    assert calls["fused"] == 1 and y.dtype == torch.bfloat16
    ref = ToyRMSNorm()(x)
    assert torch.allclose(y.float(), ref.float(), rtol=2 ** -6, atol=2 ** -8)
    # prefill-size input falls through
    big = torch.randn(1, 4096, 32, dtype=torch.bfloat16)
    m.n1(big)
    assert calls["fused"] == 1
    # fp32 input falls through
    m.n2(torch.randn(1, 1, 32))
    assert calls["fused"] == 1


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("E4B_FUSE_T1_GLUE", raising=False)
    m = torch.nn.Module()
    m.n = ToyRMSNorm()
    assert fuse_t1_glue(m) == 0


def test_vacuous_refused(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    _stub(monkeypatch, {"fused": 0})
    with pytest.raises(RuntimeError, match="matched no RMSNorm"):
        fuse_t1_glue(torch.nn.Module())
