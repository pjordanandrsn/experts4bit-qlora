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


def _stub(monkeypatch, calls, legacy=False):
    """A kernel-side stand-in. ``legacy=True`` is the pre-0.27 cut:
    no ``scale`` on the residual fold and no scaled residual add."""
    stub = types.ModuleType("int4_b32")

    def rmsnorm_resid_rows_legacy(x, resid, w, eps):
        calls["resid"] += 1
        s = (x.float() + resid.float()).to(torch.bfloat16)
        sf = s.float()
        out = (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps)
               * w.float()).to(torch.bfloat16)
        return out, s

    def rmsnorm_resid_rows(x, resid, w, eps, scale=1.0):
        if scale != 1.0:
            calls["resid_scaled"] = calls.get("resid_scaled", 0) + 1
            # upstream's two roundings: bf16 product, then bf16 sum
            x = (x * scale)
            assert x.dtype == torch.bfloat16
        return rmsnorm_resid_rows_legacy(x, resid, w, eps)

    def scaled_resid_add_rows(x, resid, scale):
        calls["scaled_add"] = calls.get("scaled_add", 0) + 1
        return resid + x * scale

    def rope_norm_heads(x, w, cos, sin, eps):
        # real formula, so a mis-paired cos/sin row shows up as a
        # numeric mismatch rather than passing on zeros
        calls["rope"] += 1
        assert cos.shape[0] == x.shape[0], "one cos row per input row"
        calls.setdefault("rope_out", [])
        xf = x.float()
        xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
              * w.float()).to(torch.bfloat16).float()
        half = x.shape[-1] // 2
        rot = torch.cat([-xn[..., half:], xn[..., :half]], dim=-1)
        out = (xn * cos.float().unsqueeze(1)
               + rot * sin.float().unsqueeze(1)).to(torch.bfloat16)
        calls["rope_out"].append(out)
        return out

    if legacy:
        stub.rmsnorm_resid_rows = rmsnorm_resid_rows_legacy
    else:
        stub.rmsnorm_resid_rows = rmsnorm_resid_rows
        stub.scaled_resid_add_rows = scaled_resid_add_rows
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


class ToyFusedAttention(torch.nn.Module):
    """Mirrors this package's qkv-fused attention closely enough to
    exercise the rotary fold: one packed projection, per-head norms,
    norm-then-rotate, and an identity output projection."""

    def __init__(self, heads=2, d=H):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(H, 3 * heads * d, bias=False,
                                        dtype=torch.bfloat16)
        self.q_norm = ToyRMSNorm(d)
        self.k_norm = ToyRMSNorm(d)
        self.o_proj = torch.nn.Identity()
        self.head_dim = d
        self._fused_nq = self._fused_nk = self._fused_nv = heads * d
        self.scaling = 1.0
        self.sliding_window = None
        self.layer_idx = 0
        self.num_key_value_groups = 1
        self.attention_dropout = 0.0
        self.config = types.SimpleNamespace(_attn_implementation="eager")
        self.captured = None

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None, **kw):
        """Unfused reference: norm each head then apply rotary."""
        d = self.head_dim
        rows = hidden_states.numel() // hidden_states.shape[-1]
        qkv = self.qkv_proj(hidden_states)
        q, k, _ = qkv.split([self._fused_nq, self._fused_nk,
                             self._fused_nv], dim=-1)
        cos, sin = position_embeddings
        out = []
        for t, norm in ((q, self.q_norm), (k, self.k_norm)):
            xn = norm(t.reshape(rows, -1, d)).float()
            half = d // 2
            rot = torch.cat([-xn[..., half:], xn[..., :half]], dim=-1)
            c = cos.reshape(-1, d).float()
            s = sin.reshape(-1, d).float()
            if c.shape[0] == 1 and rows > 1:
                c, s = c.expand(rows, d), s.expand(rows, d)
            out.append((xn * c.unsqueeze(1)
                        + rot * s.unsqueeze(1)).to(torch.bfloat16))
        self.captured = out
        return hidden_states, None


@pytest.mark.parametrize("batch,cos_batch", [(1, 1), (4, 4), (4, 1)])
def test_rotary_fold_handles_broadcast_cos(monkeypatch, batch, cos_batch):
    """A batch-1 cos/sin that upstream broadcasts must be materialised
    per row, never paired positionally with row 0 (review finding)."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    torch.manual_seed(5)
    m = torch.nn.Module()
    m.attn = ToyFusedAttention()
    assert fuse_t1_glue_r2(m) == (0, 1)

    x = (torch.randn(batch, 1, H) * 2).to(torch.bfloat16)
    ang = torch.rand(cos_batch, 1, H // 2) * 6.28
    ang = torch.cat([ang, ang], dim=-1)
    cos = ang.cos().to(torch.bfloat16)
    sin = ang.sin().to(torch.bfloat16)

    ref = ToyFusedAttention()
    ref.load_state_dict(m.attn.state_dict())
    ref.forward(x, position_embeddings=(cos, sin))
    q_ref, k_ref = ref.captured

    with torch.no_grad():
        m.attn(x, position_embeddings=(cos, sin))
    assert calls["rope"] == 2, "both projections must take the fused path"
    q_got, k_got = calls["rope_out"]
    for got, want in ((q_got, q_ref), (k_got, k_ref)):
        assert got.shape == want.shape
        assert torch.allclose(got.float(), want.float(),
                              rtol=2 ** -6, atol=2 ** -8)


class Gemma4ShapedDecoderLayer(ToyDecoderLayer):
    """Same four attribute names as the plain layer, a different body:
    two more norms, a routed-expert branch beside the dense MLP and a
    layer scalar -- Gemma-4's decoder layer. The fold's re-implemented
    forward would silently drop all of that."""
    def __init__(self):
        super().__init__()
        self.pre_feedforward_layernorm = ToyRMSNorm()
        self.post_feedforward_layernorm = ToyRMSNorm()
        self.router = torch.nn.Linear(H, 4, bias=False)
        self.experts = torch.nn.Identity()
        self.layer_scalar = torch.nn.Buffer(torch.ones(1))


class GraniteShapedDecoderLayer(ToyDecoderLayer):
    """Plain children, but the layer scales its residual adds."""
    def __init__(self):
        super().__init__()
        self.residual_multiplier = torch.nn.Parameter(torch.full((1,), 0.22))


def test_layer_with_extra_structure_is_refused(monkeypatch):
    """Name presence is not structure: a layer whose children exceed the
    four the fold re-implements, or that carries parameters or buffers of
    its own, keeps its own forward (Gemma-4, GraniteMoe)."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    for cls in (Gemma4ShapedDecoderLayer, GraniteShapedDecoderLayer):
        m = torch.nn.Module()
        m.layer = cls()
        with pytest.raises(RuntimeError, match="patched nothing"):
            fuse_t1_glue_r2(m)
        x = (torch.randn(1, 1, H) * 2).to(torch.bfloat16)
        m.layer(x)
        assert calls["resid"] == 0, cls.__name__
    # the plain layer beside them still folds
    m = torch.nn.Module()
    m.plain = ToyDecoderLayer()
    m.gemma = Gemma4ShapedDecoderLayer()
    assert fuse_t1_glue_r2(m) == (1, 0)
    assert calls["resid"] == 0
    m.plain(torch.randn(1, 1, H).to(torch.bfloat16))
    m.gemma(torch.randn(1, 1, H).to(torch.bfloat16))
    assert calls["resid"] == 1, "only the plain layer folded"


def test_structural_guard_is_what_refuses(monkeypatch):
    """The guard itself, and proof the end-to-end refusal goes through it:
    with the guard forced permissive, the Gemma-shaped layer WOULD fold
    (Bugbot, #366 -- the earlier toy names did not end in DecoderLayer,
    so the fold never looked at them)."""
    from experts4bit_qlora.engines import glue_r2
    assert glue_r2._layer_is_plain(ToyDecoderLayer())
    assert not glue_r2._layer_is_plain(Gemma4ShapedDecoderLayer())
    assert not glue_r2._layer_is_plain(GraniteShapedDecoderLayer())
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    _stub(monkeypatch, {"resid": 0, "rope": 0})
    monkeypatch.setattr(glue_r2, "_layer_is_plain", lambda mod: True)
    m = torch.nn.Module()
    m.layer = Gemma4ShapedDecoderLayer()
    assert fuse_t1_glue_r2(m) == (1, 0), "with the guard gone the fold reaches the Gemma-shaped layer"



class GraniteMoeShapedDecoderLayer(torch.nn.Module):
    """GraniteMoe's actual layer shape (transformers 5.x): the four
    pre-norm children with the MoE under ``block_sparse_moe``, and a
    Python-float ``residual_multiplier`` scaling both residual adds.
    The forward is upstream's, line for line."""
    def __init__(self, multiplier=0.22, norm_cls=ToyRMSNorm):
        super().__init__()
        self.hidden_size = H
        self.self_attn = ToyAttn()
        self.input_layernorm = norm_cls()
        self.post_attention_layernorm = norm_cls()
        self.block_sparse_moe = torch.nn.Linear(H, H, bias=False,
                                                dtype=torch.bfloat16)
        if multiplier is not None:
            self.residual_multiplier = multiplier

    def forward(self, hidden_states, attention_mask=None,
                past_key_values=None, position_embeddings=None, **kw):
        m = getattr(self, "residual_multiplier", 1.0)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states * m
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states * m
        return hidden_states


class NoisyAttn(torch.nn.Module):
    """Deterministic non-zero attention so the scaled add is exercised
    on real values (a zero attention output makes any scale pass)."""
    def forward(self, hidden_states=None, **kw):
        return torch.tanh(hidden_states.float() * 1.7).to(hidden_states.dtype), None


def test_granite_shaped_layer_folds_and_matches(monkeypatch):
    """The scaled body folds: residual + attn * m into the post-attention
    norm, the tail into one scaled add; the result is the layer's own
    forward to the bit (the stub reproduces upstream's two roundings),
    and both fold calls are counted."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    torch.manual_seed(5)
    m = torch.nn.Module()
    m.layer = GraniteMoeShapedDecoderLayer(0.22)
    m.layer.self_attn = NoisyAttn()
    x = (torch.randn(1, 1, H) * 2).to(torch.bfloat16)
    want = m.layer(x)
    assert fuse_t1_glue_r2(m) == (1, 0)
    got = m.layer(x)
    assert torch.equal(got, want), "scaled fold must reproduce upstream's roundings"
    assert calls["resid"] == 1 and calls["resid_scaled"] == 1
    assert calls["scaled_add"] == 1
    # off decode shapes the original chain runs
    big = (torch.randn(1, 65, H)).to(torch.bfloat16)
    assert torch.equal(m.layer(big), GraniteMoeShapedDecoderLayer(0.22).__class__.forward(m.layer, big))
    assert calls["resid"] == 1 and calls["scaled_add"] == 1


def test_granite_shape_without_multiplier_is_the_plain_fold(monkeypatch):
    """The same body with no multiplier (older Mixtral cuts) is the plain
    fold under another child name: no scaled kernels needed, and the
    legacy kernel cut serves it."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls, legacy=True)
    torch.manual_seed(6)
    m = torch.nn.Module()
    m.layer = GraniteMoeShapedDecoderLayer(None)
    m.layer.self_attn = NoisyAttn()
    x = (torch.randn(1, 1, H) * 2).to(torch.bfloat16)
    want = m.layer(x)
    assert fuse_t1_glue_r2(m) == (1, 0)
    assert torch.equal(m.layer(x), want)
    assert calls["resid"] == 1 and "scaled_add" not in calls


def test_scaled_layer_on_a_legacy_kernel_refuses_loudly(monkeypatch):
    """A residual-scaled body on a kernel cut without the scaled fold is
    an error naming the cut, never a quiet skip: the arm asked for the
    fusion."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    _stub(monkeypatch, {"resid": 0, "rope": 0}, legacy=True)
    m = torch.nn.Module()
    m.layer = GraniteMoeShapedDecoderLayer(0.22)
    with pytest.raises(RuntimeError, match="scaled residual fold"):
        fuse_t1_glue_r2(m)


def test_scaled_layer_guard_reads_structure(monkeypatch):
    """Only a Python-float multiplier on the exact GraniteMoe child set is
    licensed: a tensor multiplier, a Parameter (the older toy), an
    integer, or an extra child all keep their own forward."""
    from experts4bit_qlora.engines import glue_r2
    assert glue_r2._layer_scale(GraniteMoeShapedDecoderLayer(0.22)) == 0.22
    assert glue_r2._layer_scale(GraniteMoeShapedDecoderLayer(None)) == 1.0
    assert glue_r2._layer_scale(GraniteShapedDecoderLayer()) is None
    assert glue_r2._layer_scale(ToyDecoderLayer()) is None
    for bad in (torch.tensor(0.22), 1, True):
        layer = GraniteMoeShapedDecoderLayer(None)
        layer.residual_multiplier = bad
        assert glue_r2._layer_scale(layer) is None, repr(bad)
    extra = GraniteMoeShapedDecoderLayer(0.22)
    extra.pre_feedforward_layernorm = ToyRMSNorm()
    assert glue_r2._layer_scale(extra) is None
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    m = torch.nn.Module()
    m.tensor_scaled = GraniteMoeShapedDecoderLayer(None)
    m.tensor_scaled.residual_multiplier = torch.tensor(0.22)
    with pytest.raises(RuntimeError, match="patched nothing"):
        fuse_t1_glue_r2(m)


def test_scaled_fold_keeps_the_centered_norm_refusal(monkeypatch):
    """The semantic norm probe applies to the scaled body too."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    _stub(monkeypatch, {"resid": 0, "rope": 0})
    m = torch.nn.Module()
    m.layer = GraniteMoeShapedDecoderLayer(0.22, norm_cls=CenteredRMSNorm)
    with pytest.raises(RuntimeError, match="patched nothing"):
        fuse_t1_glue_r2(m)


class TupleMoE(torch.nn.Module):
    """gpt-oss's MoE block shape: returns ``(hidden, router_scores)``."""
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(H, H, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.proj(x), torch.zeros(x.shape[0], 4)


class GptOssShapedDecoderLayer(ToyDecoderLayer):
    """Plain four children, but the MoE child returns a tuple and the
    layer unpacks it -- gpt-oss's decoder layer."""
    def __init__(self):
        super().__init__()
        self.mlp = TupleMoE()

    def forward(self, hidden_states, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                position_embeddings=None, **kw):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        return residual + hidden_states


def test_plain_fold_unpacks_a_tuple_returning_moe(monkeypatch):
    """The gpt-oss layer is structurally plain; its MoE block returns
    ``(hidden, scores)``. The fold must mirror the layer's unpack, not
    add a tuple to the residual (validation lane, TypeError)."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    calls = {"resid": 0, "rope": 0}
    _stub(monkeypatch, calls)
    torch.manual_seed(9)
    m = torch.nn.Module()
    m.layer = GptOssShapedDecoderLayer()
    m.layer.self_attn = NoisyAttn()
    x = (torch.randn(1, 1, H) * 2).to(torch.bfloat16)
    want = m.layer(x)
    assert fuse_t1_glue_r2(m) == (1, 0)
    got = m.layer(x)
    assert torch.equal(got, want)
    assert calls["resid"] == 1
