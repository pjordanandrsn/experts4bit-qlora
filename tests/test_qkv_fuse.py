# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-f2-tail T2 gate: fused-QKV attention must match stock within
the fp-reorder class -- per projection and through the full forward.

NOT a bitwise gate, deliberately: BLAS kernel selection depends on the
output dim, so fusing three [N_i, K] weights into one [sum N_i, K]
weight can change a row's accumulation ORDER (first observed right
here, on CPU sgemm at the tiny shape: rel 2.9e-7). The mechanism
guarantees same-operands/same-dots, so the bound is the K6 relative
frame: max|delta| <= max|ref| * 2^-7. Structural drift in the mirrored
forward (norm or rotary misplacement) is an O(1) error and fails this
tolerance immediately -- that is what the full-forward case is for.
Runs the REAL transformers class on CPU so drift fails before a box is
rented."""

import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")
qmod = pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe")

_qf = pytest.importorskip("experts4bit_qlora.engines.qkv_fuse")
fuse_qkv = _qf.fuse_qkv

REL_BAR = 2.0 ** -7


def _close(want, got, tag):
    ref = want.abs().max().item()
    d = (want - got).abs().max().item()
    assert d <= ref * REL_BAR, (
        f"{tag}: max|delta|={d:g} exceeds max|ref|*2^-7={ref * REL_BAR:g}"
        " -- structural drift, not reorder noise")


def _tiny():
    cfg = qmod.Qwen3MoeConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, moe_intermediate_size=32,
        num_experts=4, num_experts_per_tok=2, num_hidden_layers=2,
        attention_bias=False, rms_norm_eps=1e-6)
    cfg._attn_implementation = "eager"
    torch.manual_seed(3)
    attn = qmod.Qwen3MoeAttention(cfg, layer_idx=0)
    attn.eval()
    return cfg, attn


def _inputs(cfg, t=5):
    torch.manual_seed(9)
    x = torch.randn(1, t, cfg.hidden_size)
    rot = qmod.Qwen3MoeRotaryEmbedding(cfg)
    pos = torch.arange(t)[None]
    cos, sin = rot(x, pos)
    return x, (cos, sin)


def test_projections_and_forward_within_reorder_class():
    cfg, attn = _tiny()
    x, pe = _inputs(cfg)
    with torch.no_grad():
        want_q = attn.q_proj(x)
        want_k = attn.k_proj(x)
        want_v = attn.v_proj(x)
        want_out, _ = attn(x, pe, None)
    n = fuse_qkv(attn)
    assert n == 1
    assert not hasattr(attn, "q_proj"), "old path must be gone"
    with torch.no_grad():
        qkv = attn.qkv_proj(x)
        got_q, got_k, got_v = qkv.split(
            [attn._fused_nq, attn._fused_nk, attn._fused_nv], dim=-1)
        got_out, _ = attn(x, pe, None)
    _close(want_q, got_q, "q_proj")
    _close(want_k, got_k, "k_proj")
    _close(want_v, got_v, "v_proj")
    _close(want_out, got_out, "full forward")


def test_the_bar_could_fail():
    """Verdict-calculator discipline for the test itself: a genuine
    structural drift (k_norm applied to q) must FAIL the tolerance --
    otherwise the full-forward case can't catch what it exists for."""
    cfg, attn = _tiny()
    x, pe = _inputs(cfg)
    with torch.no_grad():
        want_out, _ = attn(x, pe, None)
        # simulate norm-misplacement drift: swap q_norm/k_norm weights
        with torch.no_grad():
            qw = attn.q_norm.weight.clone()
            attn.q_norm.weight.copy_(attn.k_norm.weight * 3.7 + 0.5)
            attn.k_norm.weight.copy_(qw * 0.2 - 1.1)
        drift_out, _ = attn(x, pe, None)
    with pytest.raises(AssertionError, match="structural drift"):
        _close(want_out, drift_out, "full forward")


def test_fuse_counts_modules_in_a_model_tree():
    cfg, attn = _tiny()
    holder = torch.nn.ModuleDict({"a": attn,
                                  "b": qmod.Qwen3MoeAttention(cfg, 1)})
    assert fuse_qkv(holder) == 2


def test_biased_projection_refuses():
    cfg, attn = _tiny()
    attn.q_proj.bias = torch.nn.Parameter(
        torch.zeros(attn.q_proj.out_features))
    with pytest.raises(RuntimeError, match="bias"):
        fuse_qkv(attn)


def test_missing_attr_refuses_not_half_fuses():
    cfg, attn = _tiny()
    del attn.k_norm
    with pytest.raises(RuntimeError, match="k_norm"):
        fuse_qkv(attn)
    assert hasattr(attn, "q_proj"), "refusal must leave the module intact"
