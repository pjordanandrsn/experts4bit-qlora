# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The router-epilogue fusion patches only routers whose own forward
agrees with the reference epilogue, and never mis-routes."""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts4bit_qlora.engines.router_epilogue import (  # noqa: E402
    fuse_router_epilogue)

E, K, HID = 16, 4, 32


class ToyRouter(torch.nn.Module):
    """softmax over all experts, then top-k, then renormalise."""

    def __init__(self, norm=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, HID))
        self.top_k, self.num_experts = K, E
        self.norm_topk_prob, self.hidden_dim = norm, HID

    def forward(self, x):
        logits = F.linear(x.reshape(-1, HID), self.weight)
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            v = v / v.sum(dim=-1, keepdim=True)
        return probs, v, i


class BiasedRouter(ToyRouter):
    """Adds a per-expert bias before selection (the gpt-oss shape). The
    class name and every structural attribute match, but it SELECTS a
    different expert set -- the case a name match cannot catch."""

    def __init__(self):
        super().__init__(norm=True)
        self.e_score_bias = torch.nn.Parameter(torch.randn(E) * 4)

    def forward(self, x):
        logits = F.linear(x.reshape(-1, HID), self.weight) + self.e_score_bias
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, self.top_k, dim=-1)
        return probs, v / v.sum(dim=-1, keepdim=True), i


class UnnormalisedTopkSoftmaxRouter(ToyRouter):
    """Softmaxes over the SELECTED logits only. With norm_topk_prob on
    this is algebraically identical to the reference (the partition
    function cancels), so it is only distinguishable -- and only wrong
    -- with the renormalisation off."""

    def __init__(self):
        super().__init__(norm=False)

    def forward(self, x):
        logits = F.linear(x.reshape(-1, HID), self.weight)
        v, i = torch.topk(logits.float(), self.top_k, dim=-1)
        return logits, torch.softmax(v, dim=-1), i


def _stub(monkeypatch, calls):
    stub = types.ModuleType("int4_b32")

    def router_epilogue(logits, k, norm):
        calls["fused"] += 1
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        if norm:
            v = v / v.sum(dim=-1, keepdim=True)
        return probs, v, i
    stub.router_epilogue = router_epilogue
    monkeypatch.setitem(sys.modules, "int4_b32", stub)


def test_patches_and_matches_the_unpatched_router(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "1")
    calls = {"fused": 0}
    _stub(monkeypatch, calls)
    torch.manual_seed(7)
    m = torch.nn.Module()
    m.gate = ToyRouter()
    ref = ToyRouter()
    ref.load_state_dict(m.gate.state_dict())

    assert fuse_router_epilogue(m) == 1
    x = torch.randn(1, 1, HID)
    p, w, i = m.gate(x)
    assert calls["fused"] == 1
    rp, rw, ri = ref(x)
    assert torch.equal(i, ri), "selected experts must be identical"
    assert torch.allclose(w, rw, rtol=2 ** -12, atol=2 ** -14)
    assert torch.allclose(p, rp, rtol=2 ** -12, atol=2 ** -14)

    big = torch.randn(1, 4096, HID)      # prefill falls through
    m.gate(big)
    assert calls["fused"] == 1


@pytest.mark.parametrize("cls", [BiasedRouter, UnnormalisedTopkSoftmaxRouter])
def test_refuses_routers_with_different_math(monkeypatch, cls):
    """Both name-match and are structurally identical to the reference
    router; one selects a different expert SET, the other returns
    different WEIGHTS. Patching either would change routing, which no
    perplexity gate downstream would forgive."""
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "1")
    _stub(monkeypatch, {"fused": 0})
    torch.manual_seed(8)
    m = torch.nn.Module()
    m.gate = cls()
    with pytest.raises(RuntimeError, match="vacuous"):
        fuse_router_epilogue(m)


def test_topk_then_softmax_is_the_same_function_when_renormalising():
    """Worth pinning as a fact rather than leaving as a trap: with
    norm_topk_prob on, softmaxing the selected logits equals gathering
    the full softmax and renormalising -- the partition function
    cancels. The probe therefore cannot distinguish those two routers
    there, and does not need to."""
    torch.manual_seed(9)
    logits = torch.randn(4, E)
    v_sel, i_sel = torch.topk(logits, K, dim=-1)
    a = torch.softmax(v_sel, dim=-1)
    p = torch.softmax(logits, dim=-1)
    b = p.gather(-1, i_sel)
    b = b / b.sum(dim=-1, keepdim=True)
    assert torch.allclose(a, b, rtol=2 ** -18, atol=2 ** -20)


def test_env_gate_off_is_a_noop(monkeypatch):
    monkeypatch.delenv("E4B_FUSE_ROUTER_EPI", raising=False)
    m = torch.nn.Module()
    m.gate = ToyRouter()
    assert fuse_router_epilogue(m) == 0
    assert m.gate.forward.__self__ is m.gate


def test_missing_kernel_refuses_loudly(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "1")
    monkeypatch.setitem(sys.modules, "int4_b32", None)
    m = torch.nn.Module()
    m.gate = ToyRouter()
    with pytest.raises(RuntimeError, match="router_epilogue"):
        fuse_router_epilogue(m)
