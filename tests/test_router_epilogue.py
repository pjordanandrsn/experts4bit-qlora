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


def _stub(monkeypatch, calls, legacy=False):
    """A torch stand-in for the kernel epilogue. ``legacy=True`` mimics a
    grouped-nf4-gemm whose router_epilogue predates select_on_logits."""
    stub = types.ModuleType("int4_b32")

    def _softmax_topk(logits, k, norm):
        calls["fused"] += 1
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        if norm:
            v = v / v.sum(dim=-1, keepdim=True)
        return probs, v, i

    if legacy:
        stub.router_epilogue = _softmax_topk
    else:
        def router_epilogue(logits, k, norm, *, select_on_logits=False, bias=None):
            if not select_on_logits:
                return _softmax_topk(logits, k, norm)
            calls["fused"] += 1
            x = logits.float() + (bias.float() if bias is not None else 0.0)
            top, i = torch.topk(x, k, dim=-1)
            return x, torch.softmax(top, dim=-1), i
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


def test_refuses_a_router_whose_math_matches_no_kind(monkeypatch):
    """Bias added BEFORE a softmax over all experts (then renormalised) is
    neither the softmax-topk kind nor the select-on-logits kind: no
    family does it, and its structure looks like the softmax-topk one, so
    only the semantic probe catches it. Patching it would change routing,
    which no perplexity gate downstream would forgive."""
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "1")
    _stub(monkeypatch, {"fused": 0})
    torch.manual_seed(8)
    m = torch.nn.Module()
    m.gate = BiasedRouter()
    with pytest.raises(RuntimeError, match="vacuous"):
        fuse_router_epilogue(m)


class GptOssLikeRouter(torch.nn.Module):
    """top-k on the biased logits, softmax over the k, returns
    (logits, weights, index) -- GptOssTopKRouter."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, HID))
        self.bias = torch.nn.Parameter(torch.randn(E) * 2)
        self.top_k, self.num_experts, self.hidden_dim = K, E, HID

    def forward(self, x):
        logits = F.linear(x.reshape(-1, HID), self.weight, self.bias)
        top, i = torch.topk(logits, self.top_k, dim=-1)
        return logits, torch.softmax(top, dim=-1, dtype=top.dtype), i


class GraniteLikeRouter(torch.nn.Module):
    """No bias; returns (index, weights, logits) -- GraniteMoeTopKRouter's
    order, which position-based hooks would misread."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, HID))
        self.top_k = K

    def forward(self, x):
        logits = F.linear(x.reshape(-1, HID), self.weight)
        top, i = torch.topk(logits, self.top_k, dim=-1)
        return i, torch.softmax(top, dim=-1), logits


class MixtralLikeRouter(torch.nn.Module):
    """MixtralTopKRouter (transformers 5.16): softmax over all experts,
    top-k, always renormalised; carries num_experts/hidden_dim but NO
    norm_topk_prob. Returns (logits, scores, index)."""
    def __init__(self):
        super().__init__()
        self.top_k = K
        self.num_experts = E
        self.hidden_dim = HID
        self.weight = torch.nn.Parameter(torch.randn(E, HID))

    def forward(self, x):
        x = x.reshape(-1, HID)
        logits = F.linear(x, self.weight)
        probs = torch.softmax(logits.float(), dim=-1)
        top, i = torch.topk(probs, self.top_k, dim=-1)
        top = top / top.sum(dim=-1, keepdim=True)
        return logits, top, i


class _NoScaleRMSNorm(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class Gemma4TextRouter(torch.nn.Module):
    """Gemma-4's router: unscaled RMSNorm, per-channel scale and
    hidden**-0.5 before the projection; softmax -> top-k -> renormalise ->
    times a learned per-expert scale. Named as upstream names it."""
    def __init__(self):
        super().__init__()
        self.norm = _NoScaleRMSNorm()
        self.scale = torch.nn.Parameter(torch.rand(HID) + 0.5)
        self.scalar_root_size = HID ** -0.5
        self.proj = torch.nn.Linear(HID, E, bias=False)
        self.per_expert_scale = torch.nn.Parameter(torch.rand(E) + 0.5)
        self.config = types.SimpleNamespace(top_k_experts=K)

    def forward(self, x):
        h = self.norm(x.reshape(-1, HID)) * self.scale * self.scalar_root_size
        probs = torch.softmax(self.proj(h), dim=-1, dtype=torch.float32)
        w, i = torch.topk(probs, self.config.top_k_experts, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        return probs, w * self.per_expert_scale[i], i


@pytest.mark.parametrize("cls,order", [(UnnormalisedTopkSoftmaxRouter, (0, 1, 2)), (GptOssLikeRouter, (0, 1, 2)),
                                       (GraniteLikeRouter, (2, 1, 0)), (Gemma4TextRouter, (0, 1, 2)),
                                       (MixtralLikeRouter, (0, 1, 2))])
def test_other_router_kinds_are_licensed_and_match_their_own_forward(monkeypatch, cls, order):
    """select-on-logits (with and without a bias, in either output order)
    and Gemma-4's normed/scaled router are patched and reproduce the
    module's own forward: same expert set, same weights, same tuple order."""
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "1")
    calls = {"fused": 0}
    _stub(monkeypatch, calls)
    torch.manual_seed(9)
    m = torch.nn.Module()
    m.gate = cls()
    ref = cls()
    ref.load_state_dict(m.gate.state_dict())
    assert fuse_router_epilogue(m) == 1
    x = torch.randn(1, 1, HID)
    got = m.gate(x)
    assert calls["fused"] == 1
    want = ref(x)
    fpos, wpos, ipos = order
    assert torch.equal(got[ipos], want[ipos]), "selected experts must be identical"
    assert torch.allclose(got[wpos].float(), want[wpos].float(), rtol=2 ** -10, atol=2 ** -12)
    assert got[fpos].shape == want[fpos].shape
    big = torch.randn(1, 4096, HID)      # prefill falls through
    m.gate(big)
    assert calls["fused"] == 1


def test_select_on_logits_kinds_need_the_kernel_mode(monkeypatch):
    """With a kernel whose router_epilogue predates select_on_logits, the
    select-on-logits routers are skipped -- counted in the refusal, never
    served by the wrong epilogue -- while the softmax-topk kind still folds."""
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "1")
    _stub(monkeypatch, {"fused": 0}, legacy=True)
    torch.manual_seed(10)
    m = torch.nn.Module()
    m.gate = GptOssLikeRouter()
    with pytest.raises(RuntimeError, match="select_on_logits"):
        fuse_router_epilogue(m)
    m2 = torch.nn.Module()
    m2.a = GptOssLikeRouter()
    m2.b = ToyRouter()
    assert fuse_router_epilogue(m2) == 1


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


def test_mixtral_router_is_offered_both_renormalisations_and_the_probe_picks():
    """No norm_topk_prob on the module: the matcher offers softmax_topk
    with and without renormalisation and the probe keeps the one the
    forward computes (Mixtral renormalises)."""
    from experts4bit_qlora.engines.router_epilogue import _probe_matches, _structural
    torch.manual_seed(3)
    mod = MixtralLikeRouter()
    cands = _structural(mod)
    assert [(k, s["norm"]) for k, s in cands if k == "softmax_topk"] == [("softmax_topk", True), ("softmax_topk", False)]
    picked = [s["norm"] for k, s in cands if k == "softmax_topk" and _probe_matches(mod, k, s)]
    assert picked == [True]
