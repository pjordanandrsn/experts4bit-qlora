# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Round-2 decode folds patch structurally, license semantically, and
fall through off decode shapes."""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts4bit_qlora.engines.glue_r2 import fuse_t1_glue_r2  # noqa: E402

H = 32


class ToyRMSNorm(torch.nn.Module):
    def __init__(self, width=H, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width,
                                                    dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)
                                 + self.variance_epsilon)
                * self.weight.float()).to(x.dtype)


class CenteredRMSNorm(ToyRMSNorm):
    """Shares the name and the structure, computes ``x * (1 + w)``."""

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)
                                 + self.variance_epsilon)
                * (1.0 + self.weight.float())).to(x.dtype)


class ToyAttn(torch.nn.Module):
    def forward(self, hidden_states=None, **kw):
        return torch.zeros_like(hidden_states), None


class ToyDecoderLayer(torch.nn.Module):
    def __init__(self, norm_cls=ToyRMSNorm):
        super().__init__()
        self.input_layernorm = norm_cls()
        self.post_attention_layernorm = norm_cls()
        self.self_attn = ToyAttn()
        self.mlp = torch.nn.Identity()

    def forward(self, hidden_states, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                position_embeddings=None, **kw):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def _stub(monkeypatch, calls):
    stub = types.ModuleType("int4_b32")

    def rmsnorm_resid_rows(x, resid, w, eps):
        calls["resid"] += 1
        s = (x.float() + resid.float()).to(torch.bfloat16)
        sf = s.float()
        out = (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps)
               * w.float()).to(torch.bfloat16)
        return out, s

    def rope_norm_heads(x, w, cos, sin, eps):
        calls["rope"] += 1
        return torch.zeros_like(x)

    stub.rmsnorm_resid_rows = rmsnorm_resid_rows
    stub.rope_norm_heads = rope_norm_heads
    monkeypatch.setitem(sys.modules, "int4_b32", stub)


def test_layer_fold_matches_and_falls_through(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    m = torch.nn.Module()
    m.layer = ToyDecoderLayer()
    assert fuse_t1_glue_r2(m) == (1, 0)

    x = (torch.randn(1, 1, H) * 2).to(torch.bfloat16)
    got = m.layer(x)
    assert calls["resid"] == 1
    ref = ToyDecoderLayer().forward(x)      # unpatched reference
    assert torch.allclose(got.float(), ref.float(),
                          rtol=2 ** -6, atol=2 ** -8)

    big = (torch.randn(1, 4096, H)).to(torch.bfloat16)   # prefill
    m.layer(big)
    assert calls["resid"] == 1
    m.layer(torch.randn(1, 1, H))                        # fp32
    assert calls["resid"] == 1


def test_centered_norm_is_refused(monkeypatch):
    """A centered variant name-matches but must not be patched: folding
    it would compute a different residual stream entirely."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    m = torch.nn.Module()
    m.layer = ToyDecoderLayer(norm_cls=CenteredRMSNorm)
    with pytest.raises(RuntimeError, match="vacuous"):
        fuse_t1_glue_r2(m)


def test_unfused_attention_is_refused(monkeypatch):
    """Round 2 replaces this package's fused attention forward; an
    attention without qkv_proj keeps its own."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)

    class BareAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_norm = ToyRMSNorm()
            self.k_norm = ToyRMSNorm()
            self.head_dim = H

        def forward(self, hidden_states=None, **kw):
            return hidden_states, None

    m = torch.nn.Module()
    m.attn = BareAttention()
    with pytest.raises(RuntimeError, match="vacuous"):
        fuse_t1_glue_r2(m)


def test_env_gate_off_is_a_noop(monkeypatch):
    monkeypatch.delenv("E4B_FUSE_T1_GLUE_R2", raising=False)
    m = torch.nn.Module()
    m.layer = ToyDecoderLayer()
    assert fuse_t1_glue_r2(m) == (0, 0)
    assert m.layer.forward.__self__ is m.layer   # untouched bound method


def test_missing_kernel_refuses_loudly(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    monkeypatch.setitem(sys.modules, "int4_b32", None)
    m = torch.nn.Module()
    m.layer = ToyDecoderLayer()
    with pytest.raises(RuntimeError, match="rmsnorm_resid_rows"):
        fuse_t1_glue_r2(m)
