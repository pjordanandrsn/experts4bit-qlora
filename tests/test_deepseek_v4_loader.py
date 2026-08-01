"""The deepseek_v4 loader wiring: registry entries and the FP8 dense installer.

Does NOT load a real checkpoint — that needs the 161 GB Flash shards. What it pins is
the wiring that decides *how* those shards are read, including the one trap the key-map
tests surfaced: `.scale` is an overloaded suffix and pairing must go by sibling presence.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from experts4bit_qlora.fp8_blocks import convert_to_fp8_blocks  # noqa: E402
from experts4bit_qlora.loader import (  # noqa: E402
    CKPT_KEY_REWRITERS,
    DEEPSEEK_V4_FP8_DENSE,
    K3_PER_EXPERT_MXFP4,
    SUPPORTED_ARCHITECTURES,
    _install_fp8_block_linears,
)


def test_registry_entries():
    assert SUPPORTED_ARCHITECTURES["deepseek_v4"] == "mlp.experts"
    assert K3_PER_EXPERT_MXFP4["deepseek_v4"] == (("w1", "w3", "w2"), "weight", "scale")
    assert "deepseek_v4" in DEEPSEEK_V4_FP8_DENSE
    assert "deepseek_v4" in CKPT_KEY_REWRITERS
    # K3's own entry must be untouched — it uses different suffixes.
    assert K3_PER_EXPERT_MXFP4["kimi_k3"] == (("w1", "w3", "w2"), "weight_packed", "weight_scale")


def test_rewriter_adds_the_model_prefix_the_strip_branch_cannot():
    rewrite = CKPT_KEY_REWRITERS["deepseek_v4"]
    assert rewrite("layers.0.attn.wq_a.weight") == "model.layers.0.self_attn.q_a_proj.weight"
    assert rewrite("embed.weight") == "model.embed_tokens.weight"


class _Tiny(nn.Module):
    """Two FP8 projections plus a hyper-connection whose parameter is *named* `scale`."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_a_proj = nn.Linear(128, 256, bias=False)
        self.self_attn.kv_proj = nn.Linear(128, 128, bias=False)
        self.attn_hc = nn.Module()
        self.attn_hc.scale = nn.Parameter(torch.zeros(3))
        self.attn_hc.fn = nn.Parameter(torch.zeros(24, 512))


def _fake_shards():
    def fp8(o, i):
        return (torch.randn(o, i) / 8).to(torch.float8_e4m3fn)

    tensors = {
        "self_attn.q_a_proj.weight": fp8(256, 128),
        "self_attn.q_a_proj.scale": torch.full((2, 1), 122, dtype=torch.uint8),
        "self_attn.kv_proj.weight": fp8(128, 128),
        "self_attn.kv_proj.scale": torch.full((1, 1), 121, dtype=torch.uint8),
        # standalone HC params — `scale` here is a real parameter, not an FP8 companion
        "attn_hc.scale": torch.randn(3),
        "attn_hc.fn": torch.randn(24, 512),
    }
    return tensors, {k: "shard0" for k in tensors}


def test_installer_pairs_by_sibling_not_by_suffix():
    model = _Tiny()
    tensors, weight_map = _fake_shards()
    consumed = _install_fp8_block_linears(
        model, weight_map, tensors.__getitem__, set(), torch.float32, "cpu")

    assert consumed == {
        "self_attn.q_a_proj.weight", "self_attn.q_a_proj.scale",
        "self_attn.kv_proj.weight", "self_attn.kv_proj.scale",
    }
    # the trap: a standalone `.scale` with no `.weight` sibling must be left alone,
    # so the ordinary _assign pass can still place it
    assert "attn_hc.scale" not in consumed
    assert "attn_hc.fn" not in consumed


def test_installer_converts_in_place_and_keeps_fp8_resident():
    model = _Tiny()
    tensors, weight_map = _fake_shards()
    _install_fp8_block_linears(
        model, weight_map, tensors.__getitem__, set(), torch.float32, "cpu")

    q = model.self_attn.q_a_proj
    assert isinstance(q, nn.Linear), "must stay an nn.Linear subclass, not be replaced"
    assert q.fp8_weight.dtype == torch.float8_e4m3fn      # storage stayed FP8
    assert "weight" not in q._parameters                  # dense copy is gone
    assert q.weight.dtype == torch.float32                # decoded on access
    assert (q.out_features, q.in_features) == (256, 128)
    out = q(torch.randn(3, 128))
    assert out.shape == (3, 256) and torch.isfinite(out).all()
    # the untouched HC parameters are still plain parameters
    assert isinstance(model.attn_hc.scale, nn.Parameter)


class _GroupedLinear(nn.Linear):
    """Stand-in for DeepseekV4GroupedLinear: an nn.Linear SUBCLASS with a
    block-diagonal forward. Duck-typing cannot tell it from a Linear."""

    def __init__(self, in_per_group, out_features, n_groups):
        super().__init__(in_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x):
        w = self.weight.view(self.n_groups, -1, self.in_features).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, self.in_features).transpose(0, 1)
        return torch.bmm(x, w).transpose(0, 1).reshape(x.shape[1], -1)


def test_conversion_preserves_a_custom_forward():
    """The o_a_proj trap: replacing the module would silently make it dense."""
    g = _GroupedLinear(64, 256, n_groups=4)
    w = (torch.randn(256, 64) / 8).to(torch.float8_e4m3fn)
    scale = torch.full((2, 1), 122, dtype=torch.uint8)
    x = torch.randn(5, 4 * 64)

    dense_ref = torch.nn.functional.linear     # what a plain-Linear swap would do
    convert_to_fp8_blocks(g, w, scale, compute_dtype=torch.float32)

    assert type(g).__name__ == "Fp8_GroupedLinear"
    assert isinstance(g, _GroupedLinear), "class identity must survive the conversion"
    out = g(x)
    assert out.shape == (5, 4 * 64), out.shape       # grouped, not dense
    assert torch.isfinite(out).all()
    # and it matches the same grouped math on the decoded weight
    ref = _GroupedLinear(64, 256, n_groups=4)
    with torch.no_grad():
        ref.weight.copy_(g.weight)
    assert torch.allclose(out, ref(x), atol=1e-5)
    assert dense_ref is torch.nn.functional.linear


def test_installer_skips_expert_keys():
    """Expert weight/scale pairs are consumed by the expert branch, not here."""
    model = _Tiny()
    tensors, weight_map = _fake_shards()
    already = {"self_attn.q_a_proj.weight", "self_attn.q_a_proj.scale"}
    consumed = _install_fp8_block_linears(
        model, weight_map, tensors.__getitem__, already, torch.float32, "cpu")
    assert consumed == {"self_attn.kv_proj.weight", "self_attn.kv_proj.scale"}
    assert isinstance(model.self_attn.q_a_proj, nn.Linear)   # left for the expert path


def test_installer_ignores_a_non_fp8_weight_with_a_scale_sibling():
    model = _Tiny()
    tensors, weight_map = _fake_shards()
    tensors["self_attn.kv_proj.weight"] = torch.randn(128, 128)   # bf16-ish, not FP8
    consumed = _install_fp8_block_linears(
        model, weight_map, tensors.__getitem__, set(), torch.float32, "cpu")
    assert "self_attn.kv_proj.weight" not in consumed
    assert isinstance(model.self_attn.kv_proj, nn.Linear)
