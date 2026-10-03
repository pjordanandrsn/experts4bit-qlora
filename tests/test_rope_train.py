"""The fused rotary embedding (on by default in enable_fast_train; E4B_FUSED_ROPE=0 turns it off) is BIT-IDENTICAL to Hugging
Face's apply_rotary_pos_emb, forward and both gradients; calls outside its contract go to the original; the patch is scoped to
the model's own attention modules and unwinds."""
import sys
import types

import pytest
import torch

CUDA = torch.cuda.is_available()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply(q, k, cos, sin, unsqueeze_dim=1):              # transformers' function, verbatim
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _qkcs(B, L, Hq, Hk, D, cos_batch=None):
    q = torch.randn(B, L, Hq, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)   # as the attention builds it
    k = torch.randn(B, L, Hk, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    fr = torch.outer(torch.arange(L, device="cuda").float(), inv)
    emb = torch.cat((fr, fr), -1)
    cb = cos_batch or B
    return q, k, emb.cos()[None].expand(cb, L, D).to(torch.bfloat16), emb.sin()[None].expand(cb, L, D).to(torch.bfloat16)


@pytest.mark.skipif(not CUDA, reason="the fused kernel needs CUDA")
@pytest.mark.parametrize("B,L,Hq,Hk,D,cb", [(2, 190, 32, 4, 128, None), (1, 7, 4, 2, 64, None), (2, 33, 8, 8, 128, 1)])
def test_fused_rope_is_bit_identical_forward_and_backward(B, L, Hq, Hk, D, cb):
    from experts4bit_qlora.engines.rope_train import rope_qk
    torch.manual_seed(0)
    q, k, cos, sin = _qkcs(B, L, Hq, Hk, D, cb)
    qa, ka = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True)
    qb, kb = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True)
    ya, yb = hf_apply(qa, ka, cos, sin), rope_qk(qb, kb, cos, sin)
    gq, gk = torch.randn_like(ya[0]), torch.randn_like(ya[1])
    torch.autograd.backward(ya, (gq, gk))
    torch.autograd.backward(yb, (gq, gk))
    for a, b in ((ya[0], yb[0]), (ya[1], yb[1]), (qa.grad, qb.grad), (ka.grad, kb.grad)):
        assert torch.equal(a, b)


def _fake_model():
    """A model whose attention class lives in a throwaway module that defines apply_rotary_pos_emb, like transformers'."""
    mod = types.ModuleType("e4b_test_fake_modeling")
    mod.apply_rotary_pos_emb = hf_apply
    exec("import torch.nn as nn\nclass FakeAttention(nn.Module):\n    pass\n", mod.__dict__)
    sys.modules[mod.__name__] = mod
    m = torch.nn.Sequential(mod.FakeAttention())
    return m, mod


def test_patch_is_scoped_falls_through_and_unwinds():
    from experts4bit_qlora.engines import rope_train as rt
    m, mod = _fake_model()
    try:
        n = rt.enable_fused_rope(m)
        if rt.triton is None:
            assert n == 0
            return
        assert n == 1 and mod.apply_rotary_pos_emb is not hf_apply and mod.apply_rotary_pos_emb._e4b_orig is hf_apply
        q = torch.randn(1, 2, 3, 8); k = torch.randn(1, 2, 3, 8); c = torch.randn(1, 3, 8); s = torch.randn(1, 3, 8)
        before = rt.ROPE_TRAIN_STATS["calls"]
        out = mod.apply_rotary_pos_emb(q, k, c, s)                   # CPU fp32: outside the contract -> the original
        assert rt.ROPE_TRAIN_STATS["calls"] == before and torch.equal(out[0], hf_apply(q, k, c, s)[0])
        assert rt.enable_fused_rope(m) == 0                          # idempotent
        assert rt.disable_fused_rope(m) == 1 and mod.apply_rotary_pos_emb is hf_apply
    finally:
        sys.modules.pop(mod.__name__, None)


def test_requested_defaults_on(monkeypatch):
    from experts4bit_qlora.engines.rope_train import fused_rope_requested
    monkeypatch.delenv("E4B_FUSED_ROPE", raising=False)
    assert fused_rope_requested()
    monkeypatch.setenv("E4B_FUSED_ROPE", "0")
    assert not fused_rope_requested()
