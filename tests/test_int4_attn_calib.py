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
        def __init__(self, k, device=None):
            self.H = torch.zeros(k, k)
            self.n = 0

        def add(self, x):
            rows = x.reshape(-1, x.shape[-1]).float()
            b = rows.shape[0]
            self.H *= self.n / (self.n + b)
            self.n += b
            self.H += (2.0 / self.n) * (rows.t() @ rows)

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
