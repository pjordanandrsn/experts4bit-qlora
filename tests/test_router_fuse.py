# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The T=1 router fusion patches structurally and falls through safely."""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts4bit_qlora.engines.router_fuse import fuse_t1_router  # noqa: E402


class _Router(torch.nn.Module):
    def __init__(self, E=8, H=16, top_k=2, norm=True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(E, H, dtype=torch.bfloat16) * 0.1)
        self.top_k = top_k
        self.norm_topk_prob = norm
        self.hidden_dim = H

    def forward(self, hidden_states):
        hs = hidden_states.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(hs, self.weight)
        p = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        v, i = torch.topk(p, self.top_k, dim=-1)
        if self.norm_topk_prob:
            v = v / v.sum(dim=-1, keepdim=True)
        return logits, v.to(logits.dtype), i


def _stub_kernel(monkeypatch, calls):
    stub = types.ModuleType("int4_b32")

    def router_topk_t1(x, weight, top_k, norm_topk_prob):
        calls["fused"] += 1
        logits = torch.nn.functional.linear(x, weight)
        p = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        v, i = torch.topk(p, top_k, dim=-1)
        if norm_topk_prob:
            v = v / v.sum(dim=-1, keepdim=True)
        return logits, v.to(logits.dtype), i
    stub.router_topk_t1 = router_topk_t1
    monkeypatch.setitem(sys.modules, "int4_b32", stub)


def test_patches_and_routes_t1(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_ROUTER", "1")
    calls = {"fused": 0}
    _stub_kernel(monkeypatch, calls)
    m = torch.nn.Module()
    m.r = _Router()
    assert fuse_t1_router(m) == 1
    x = torch.randn(1, 1, 16, dtype=torch.bfloat16)
    logits, w, i = m.r(x)
    assert calls["fused"] == 1
    assert i.shape == (1, 2) and w.dtype == torch.bfloat16
    # T > 1 falls through to the original chain
    x8 = torch.randn(1, 8, 16, dtype=torch.bfloat16)
    m.r(x8)
    assert calls["fused"] == 1


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("E4B_FUSE_T1_ROUTER", raising=False)
    m = torch.nn.Module()
    m.r = _Router()
    assert fuse_t1_router(m) == 0


def test_vacuous_enable_refused(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_ROUTER", "1")
    _stub_kernel(monkeypatch, {"fused": 0})
    with pytest.raises(RuntimeError, match="matched no router"):
        fuse_t1_router(torch.nn.Module())


def test_missing_kernel_refused(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_ROUTER", "1")
    monkeypatch.setitem(sys.modules, "int4_b32", None)
    m = torch.nn.Module()
    m.r = _Router()
    with pytest.raises((RuntimeError, ImportError)):
        fuse_t1_router(m)
