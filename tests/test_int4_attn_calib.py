# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The calibrated int4 attention lane: hooks see every projection input,
Hessians reach the packer, and nothing is packed uncalibrated under the
calibrated banner."""
import os
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

H, HEADS, D = 64, 2, 32


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(H, 3 * HEADS * D, bias=False,
                                  dtype=torch.bfloat16)
        self.o_proj = nn.Linear(HEADS * D, H, bias=False,
                                dtype=torch.bfloat16)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, _, _ = qkv.split(HEADS * D, dim=-1)
        return self.o_proj(q)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, H)
        self.attn = ToyAttention()

    def forward(self, ids):
        return self.attn(self.emb(ids).to(torch.bfloat16))


def _stubs(monkeypatch, seen):
    """Reference kernel package: the real quantiser math on CPU, plus a
    record of which Hessian each pack call received."""
    from experts4bit_qlora import _vendor  # noqa: F401  (package import)
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
    def _no_gpu(*a, **kw):
        raise RuntimeError("no gpu in the CPU test")

    def _quant(x):
        return x, x

    def _plan(N, K):
        return 128, 4, 1, 1
    k.gemv_int4_b32 = _no_gpu
    k.quant_x_rows = _quant
    k._plan = _plan

    gp = types.ModuleType("gptq_pack")

    class HessianAccumulator:
        """Mirrors grouped-nf4-gemm 0.23: Gram on the activation device,
        storage on `device`, allocated on the first batch."""

        def __init__(self, k, device=None):
            self.k, self.device, self.H, self.n = k, device, None, 0

        def add(self, x):
            rows = x.reshape(-1, x.shape[-1]).float()
            b = rows.shape[0]
            gram = rows.t() @ rows
            if self.H is None:
                self.H = torch.zeros(self.k, self.k,
                                     device=self.device or gram.device)
            self.H *= self.n / (self.n + b)
            self.n += b
            self.H.add_(gram.to(self.H.device), alpha=2.0 / self.n)

    def gptq_pack_int4_b32(w, hessian, **kw):
        seen.append(hessian.shape)
        return pack_int4_b32(w)
    gp.HessianAccumulator = HessianAccumulator
    gp.gptq_pack_int4_b32 = gptq_pack_int4_b32
    for name, mod in (("int4_pack_ref", pack_ref), ("int4_b32", k),
                      ("gptq_pack", gp)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_calibration_sees_every_projection_input(monkeypatch):
    seen = []
    _stubs(monkeypatch, seen)
    from experts4bit_qlora.engines.int4_attn_calib import (
        calibrate_attention_hessians)
    torch.manual_seed(1)
    m = ToyModel()
    batches = [torch.randint(0, 50, (2, 8)) for _ in range(3)]
    hs = calibrate_attention_hessians(m, batches, device="cpu")
    assert set(hs) == {"attn.qkv_proj", "attn.o_proj"}
    # Hessians live on the requested device (CPU by default), not the card
    assert all(v.device.type == "cpu" for v in hs.values())
    assert hs["attn.qkv_proj"].shape == (H, H)
    assert hs["attn.o_proj"].shape == (HEADS * D, HEADS * D)
    # a Gram matrix over real inputs is symmetric PSD and non-trivial
    for Hm in hs.values():
        assert torch.allclose(Hm, Hm.t())
        assert torch.linalg.eigvalsh(Hm).min() >= -1e-4
        assert Hm.diag().sum() > 0
    # hooks are gone: another forward must not grow the Hessians
    before = hs["attn.qkv_proj"].clone()
    m(batches[0])
    assert torch.equal(before, hs["attn.qkv_proj"])


def test_enable_packs_with_each_projections_hessian(monkeypatch):
    seen = []
    _stubs(monkeypatch, seen)
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    from experts4bit_qlora.engines.int4_attn_calib import (
        calibrate_attention_hessians, enable_serve_attn_int4_calib)
    m = ToyModel()
    hs = calibrate_attention_hessians(m, [torch.randint(0, 50, (2, 8))],
                                      device="cpu")
    assert enable_serve_attn_int4_calib(m, hs) == 2
    assert isinstance(m.attn.qkv_proj, Int4Linear)
    assert isinstance(m.attn.o_proj, Int4Linear)
    # the packer was handed a Hessian of the right width for each
    assert sorted(seen) == sorted([(H, H), (HEADS * D, HEADS * D)])
    # Hessians are released as each projection is packed
    assert all(v is None for v in hs.values())


def test_refuses_to_pack_a_projection_without_a_hessian(monkeypatch):
    """Two quantisers under one banner would make the K8 verdict
    ambiguous; a missing Hessian must refuse, never fall back."""
    seen = []
    _stubs(monkeypatch, seen)
    from experts4bit_qlora.engines.int4_attn_calib import (
        enable_serve_attn_int4_calib)
    m = ToyModel()
    with pytest.raises(RuntimeError, match="no Hessian"):
        enable_serve_attn_int4_calib(m, {"attn.qkv_proj": torch.eye(H)})
    assert seen == []                      # nothing was packed


def test_env_gate_off_is_a_noop(monkeypatch):
    seen = []
    _stubs(monkeypatch, seen)
    monkeypatch.delenv("E4B_SERVE_ATTN_INT4_CALIB", raising=False)
    from experts4bit_qlora.engines.int4_attn_calib import enable_from_env
    m = ToyModel()
    assert enable_from_env(m, [torch.randint(0, 50, (1, 4))]) == 0
    assert type(m.attn.qkv_proj) is nn.Linear


def test_uncalibrated_int4linear_is_unchanged(monkeypatch):
    """The packer hook defaults to the shipped packer, so the existing
    uncalibrated lane produces byte-identical stores."""
    seen = []
    _stubs(monkeypatch, seen)
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    torch.manual_seed(2)
    lin = nn.Linear(H, 2 * H, bias=False, dtype=torch.bfloat16)
    a = Int4Linear(lin)
    import int4_pack_ref
    p, s = int4_pack_ref.pack_int4_b32(lin.weight.detach().float())
    assert torch.equal(a.packed.reshape(-1), p.reshape(-1))
    assert torch.equal(a.scales.reshape(-1), s.reshape(-1))
    assert seen == []


class ToyLM(ToyModel):
    """A causal-LM-shaped toy: the attention plus a bias-free output
    head TIED to the embedding table (Gemma-style), found through
    ``get_output_embeddings``."""
    def __init__(self, tied=True, bias=False):
        super().__init__()
        self.lm_head = nn.Linear(H, 50, bias=bias, dtype=torch.bfloat16)
        if tied:
            self.emb.weight = self.lm_head.weight = nn.Parameter(
                torch.randn(50, H).to(torch.bfloat16))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, ids):
        h = self.attn(self.emb(ids).to(torch.bfloat16))
        return self.lm_head(h)


def test_output_head_is_packed_only_when_opted_in(monkeypatch):
    """Without the flag the head is untouched (the attention lane's
    standing contract); with it the head is calibrated with its own
    Hessian and swapped, and the tied embedding table is NOT changed --
    the swap replaces the Linear module, not the shared Parameter."""
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    from experts4bit_qlora.engines.int4_attn_calib import (
        calibrate_attention_hessians, enable_serve_attn_int4_calib)
    seen = []
    _stubs(monkeypatch, seen)
    torch.manual_seed(1)
    m = ToyLM()
    ids = torch.randint(0, 50, (2, 8))
    hess = calibrate_attention_hessians(m, [ids])
    assert "lm_head" not in hess
    assert enable_serve_attn_int4_calib(m, hess) == 2
    assert type(m.lm_head) is nn.Linear, "no flag: the head stays"
    m2 = ToyLM()
    emb_param = m2.emb.weight
    emb_before = emb_param.detach().clone()
    assert m2.lm_head.weight is emb_param, "fixture must be tied"
    hess = calibrate_attention_hessians(m2, [ids], include_head=True)
    assert "lm_head" in hess and tuple(hess["lm_head"].shape) == (H, H)
    seen.clear()
    assert enable_serve_attn_int4_calib(m2, hess, include_head=True) == 3
    assert isinstance(m2.lm_head, Int4Linear)
    assert (H, H) in seen and len(seen) == 3
    # the tied table is the SAME Parameter object, bitwise unchanged, still bf16
    assert m2.emb.weight is emb_param
    assert torch.equal(m2.emb.weight.detach(), emb_before)
    assert m2.emb.weight.dtype == torch.bfloat16 and m2.emb.weight.shape == (50, H)


def test_output_head_alone(monkeypatch):
    """Head-only mode packs the head and nothing else."""
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    from experts4bit_qlora.engines.int4_attn_calib import (
        calibrate_attention_hessians, enable_serve_attn_int4_calib)
    _stubs(monkeypatch, [])
    m = ToyLM(tied=False)
    ids = torch.randint(0, 50, (2, 8))
    hess = calibrate_attention_hessians(m, [ids], include_attention=False, include_head=True)
    assert list(hess) == ["lm_head"]
    assert enable_serve_attn_int4_calib(m, hess, include_attention=False, include_head=True) == 1
    assert isinstance(m.lm_head, Int4Linear)
    assert type(m.attn.qkv_proj) is nn.Linear and type(m.attn.o_proj) is nn.Linear


def test_output_head_refusals(monkeypatch):
    from experts4bit_qlora.engines.int4_attn_calib import _int4_targets
    _stubs(monkeypatch, [])
    with pytest.raises(RuntimeError, match="carries a bias"):
        _int4_targets(ToyLM(tied=False, bias=True), include_head=True)
    with pytest.raises(RuntimeError, match="no nn.Linear output head"):
        _int4_targets(ToyModel(), include_head=True)


def test_env_flags_for_the_head(monkeypatch):
    """The head flag alone is a head-only enable, never a silent ignore;
    both off is a no-op."""
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    from experts4bit_qlora.engines.int4_attn_calib import enable_from_env
    _stubs(monkeypatch, [])
    ids = torch.randint(0, 50, (2, 8))
    monkeypatch.delenv("E4B_SERVE_ATTN_INT4_CALIB", raising=False)
    monkeypatch.delenv("E4B_SERVE_LMHEAD_INT4_CALIB", raising=False)
    m = ToyLM(tied=False)
    assert enable_from_env(m, [ids]) == 0
    monkeypatch.setenv("E4B_SERVE_LMHEAD_INT4_CALIB", "1")
    m = ToyLM(tied=False)
    assert enable_from_env(m, [ids]) == 1
    assert isinstance(m.lm_head, Int4Linear) and type(m.attn.o_proj) is nn.Linear
    monkeypatch.setenv("E4B_SERVE_ATTN_INT4_CALIB", "1")
    m = ToyLM(tied=False)
    assert enable_from_env(m, [ids]) == 3


class ToyDenseMLP(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(H, 2 * H, bias=bias, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(H, 2 * H, bias=bias, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(2 * H, H, bias=bias, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))


class ToySparseMoeBlock(nn.Module):
    """A routed block under the same attribute name: a router Linear and
    fused experts -- NOT a dense MLP, so nothing in it is selected."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, 4, bias=False, dtype=torch.bfloat16)
        self.experts = nn.Identity()

    def forward(self, x):
        return x


class ToyDecoderLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class ToyGemmaLikeModel(nn.Module):
    """A Gemma-4-shaped tree: a dense MLP beside the routed block in one
    layer, a routed block alone in another."""
    def __init__(self, dense_bias=False):
        super().__init__()
        self.emb = nn.Embedding(50, H)
        self.layers = nn.ModuleList([ToyDecoderLayer(ToyDenseMLP(dense_bias)),
                                     ToyDecoderLayer(ToySparseMoeBlock())])

    def forward(self, ids):
        x = self.emb(ids).to(torch.bfloat16)
        for layer in self.layers:
            x = layer(x)
        return x


def test_dense_mlp_target_selects_the_dense_mlp_only(monkeypatch):
    """The dense MLP's three projections are packed under their flag; the
    routed block's router Linear (same attribute name) is not; attention
    is untouched when its flag is off."""
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    from experts4bit_qlora.engines.int4_attn_calib import (
        _dense_mlp_linears, calibrate_attention_hessians, enable_serve_attn_int4_calib)
    seen = []
    _stubs(monkeypatch, seen)
    torch.manual_seed(2)
    m = ToyGemmaLikeModel()
    names = sorted(_dense_mlp_linears(m))
    assert names == ["layers.0.mlp.down_proj", "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"]
    ids = torch.randint(0, 50, (2, 8))
    hess = calibrate_attention_hessians(m, [ids], include_attention=False, include_dense_mlp=True)
    assert sorted(hess) == names
    assert tuple(hess["layers.0.mlp.down_proj"].shape) == (2 * H, 2 * H)
    n = enable_serve_attn_int4_calib(m, hess, include_attention=False, include_dense_mlp=True)
    assert n == 3
    assert isinstance(m.layers[0].mlp.gate_proj, Int4Linear) and isinstance(m.layers[0].mlp.down_proj, Int4Linear)
    assert type(m.layers[1].mlp.gate) is nn.Linear, "the router Linear is never a dense-MLP target"
    for layer in m.layers:
        assert type(layer.self_attn.qkv_proj) is nn.Linear


def test_dense_mlp_refusals(monkeypatch):
    from experts4bit_qlora.engines.int4_attn_calib import _int4_targets
    _stubs(monkeypatch, [])
    with pytest.raises(RuntimeError, match="carries a bias"):
        _int4_targets(ToyGemmaLikeModel(dense_bias=True), include_attention=False, include_dense_mlp=True)
    m = ToyModel()                                   # no decoder layers at all
    with pytest.raises(RuntimeError, match="no decoder layer carries a dense MLP"):
        _int4_targets(m, include_attention=False, include_dense_mlp=True)


def test_env_flag_for_the_dense_mlp(monkeypatch):
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    from experts4bit_qlora.engines.int4_attn_calib import enable_from_env
    _stubs(monkeypatch, [])
    for k in ("E4B_SERVE_ATTN_INT4_CALIB", "E4B_SERVE_LMHEAD_INT4_CALIB", "E4B_SERVE_DENSE_INT4_CALIB"):
        monkeypatch.delenv(k, raising=False)
    ids = torch.randint(0, 50, (2, 8))
    monkeypatch.setenv("E4B_SERVE_DENSE_INT4_CALIB", "1")
    m = ToyGemmaLikeModel()
    assert enable_from_env(m, [ids]) == 3
    assert isinstance(m.layers[0].mlp.up_proj, Int4Linear)
    monkeypatch.setenv("E4B_SERVE_ATTN_INT4_CALIB", "1")
    m = ToyGemmaLikeModel()
    assert enable_from_env(m, [ids]) == 3 + 4
