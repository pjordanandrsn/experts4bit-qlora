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

import importlib.util
from pathlib import Path

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


def _int4_cpu_stubs():
    """test_int4_attn's CPU stand-ins for the int4 kernel package, loaded by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_t_int4_attn", Path(__file__).with_name("test_int4_attn.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # its own importorskip guards apply
    return mod._cpu_kernel_stubs


def test_fuse_qkv_after_the_int4_swap_fuses_the_packed_store(monkeypatch):
    """Lane order in every int4 census: ``enable_serve_attn_int4`` at load, ``fuse_qkv`` after. Until P54 the
    second refused the first's output (no ``.weight`` on an Int4Linear), so every int4 arm ran unfused and paid
    q, k, v and o as four attention launches per layer. Fused: two (qkv, o), same function."""
    calls = []
    _int4_cpu_stubs()(monkeypatch, calls)
    from experts4bit_qlora.engines.int4_attn import Int4Linear, enable_serve_attn_int4
    cfg, attn = _tiny()
    x, pe = _inputs(cfg)
    assert enable_serve_attn_int4(attn, smallm=True) == 4                       # q, k, v, o
    with torch.no_grad():
        want_out, _ = attn(x, pe, None)                                          # int4, unfused
    launches_unfused = [c[0] for c in calls]
    calls.clear()
    assert fuse_qkv(attn) == 1
    assert isinstance(attn.qkv_proj, Int4Linear) and not hasattr(attn, "q_proj")
    n_q = cfg.num_attention_heads * cfg.head_dim
    n_kv = cfg.num_key_value_heads * cfg.head_dim
    assert (attn._fused_nq, attn._fused_nk, attn._fused_nv) == (n_q, n_kv, n_kv)
    assert attn.qkv_proj.N == n_q + 2 * n_kv and attn.qkv_proj._smallm is not None
    with torch.no_grad():
        got_out, _ = attn(x, pe, None)
    launches_fused = [c[0] for c in calls]
    assert launches_unfused == ["smallm"] * 4 and launches_fused == ["smallm"] * 2
    _close(want_out, got_out, "full forward, int4: fused vs unfused")


def test_fuse_qkv_refuses_a_mixed_int4_and_dense_triple(monkeypatch):
    calls = []
    _int4_cpu_stubs()(monkeypatch, calls)
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    cfg, attn = _tiny()
    attn.q_proj = Int4Linear(attn.q_proj)                                        # only q on the grid
    with pytest.raises(RuntimeError, match="mix of Int4Linear and dense"):
        fuse_qkv(attn)
    assert hasattr(attn, "k_proj") and hasattr(attn, "v_proj"), "refusal must leave the module intact"
