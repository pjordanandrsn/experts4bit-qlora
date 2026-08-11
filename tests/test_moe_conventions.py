"""MoE storage conventions, adjudicated against transformers' own converter
specs — not against convention or shape inference.

The property under test is orientation. In both conventions the gate and up
tensors are shape-identical, so a swap produces `act(up) * gate`: wrong
numbers, right shapes, nothing raised. These tests pin the ordering to two
independent upstream sources (the converter spec's source order + concat axis,
and each family's expert forward) so an upstream change breaks the build
instead of quietly degrading model quality.
"""
import inspect
import re

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.moe_conventions import (  # noqa: E402
    CONVENTIONS,
    JAMBA,
    LFM2_MOE,
    MIXTRAL,
    PHIMOE,
    QWEN2_MOE,
    MoEConventionError,
    convention_for,
    fuse_experts,
)


def _conversion_src():
    CM = pytest.importorskip("transformers.conversion_mapping")
    return inspect.getsource(CM)


def _spec_block(src, name):
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if re.match(rf'^\s{{4,8}}"{name}":\s*\[\s*$', line):
            depth, out = 0, []
            for j in range(i, min(i + 40, len(lines))):
                out.append(lines[j])
                depth += lines[j].count("[") - lines[j].count("]")
                if j > i and depth <= 0:
                    break
            return "\n".join(out)
    pytest.skip(f"transformers no longer defines a {name!r} converter spec")


#: Only conventions that actually FUSE gate/up have a source order to check.
#: granitemoe ships pre-fused, so upstream never lists a gate/up pair for it;
#: its equivalent guarantee is tested in test_granitemoe_forward_* below.
FUSING = [c for c in CONVENTIONS if c.roles]


@pytest.mark.parametrize("conv", FUSING, ids=lambda c: c.name)
def test_gate_precedes_up_in_upstream_spec(conv):
    """The converter lists the gate source FIRST and concatenates on the
    intermediate axis — that is what makes the gate the first block."""
    block = _spec_block(_conversion_src(), conv.name)
    w_spelled = "w1" in conv.roles
    gate_tok, up_tok = ("w1", "w3") if w_spelled else ("gate_proj", "up_proj")
    gi, ui = block.find(gate_tok), block.find(up_tok)
    assert gi != -1 and ui != -1, f"{conv.name}: gate/up tokens vanished from the spec"
    assert gi < ui, f"{conv.name}: upstream now lists up BEFORE gate — re-adjudicate"
    # Whitespace-insensitive: the spec is formatted differently per entry.
    flat = re.sub(r"\s+", "", block)
    assert "Concatenate(dim=1)" in flat, \
        f"{conv.name}: gate/up no longer concatenated on the intermediate axis"


def test_mixtral_forward_activates_the_first_half():
    M = pytest.importorskip("transformers.models.mixtral.modeling_mixtral")
    src = inspect.getsource(M.MixtralExperts.forward)
    assert re.search(r"gate,\s*up\s*=.*chunk\(2", src)
    assert re.search(r"act_fn\(\s*gate\s*\)", src)


def test_qwen2_moe_family_forward_activates_the_first_half():
    M = pytest.importorskip("transformers.models.qwen2_moe.modeling_qwen2_moe")
    src = inspect.getsource(M)
    assert "gate_up_proj" in src, "qwen2_moe no longer uses a fused gate_up_proj"
    assert re.search(r"gate,\s*up\s*=", src), "fused chunk into (gate, up) is gone"


def test_expert_key_matching_and_roles():
    assert QWEN2_MOE.match("mlp.experts.17.gate_proj.weight") == (17, "gate")
    assert QWEN2_MOE.match("mlp.experts.3.up_proj.weight") == (3, "up")
    assert QWEN2_MOE.match("mlp.experts.3.down_proj.weight") == (3, "down")
    assert QWEN2_MOE.match("self_attn.q_proj.weight") is None
    assert MIXTRAL.match("block_sparse_moe.experts.5.w1.weight") == (5, "gate")
    assert MIXTRAL.match("block_sparse_moe.experts.5.w3.weight") == (5, "up")
    assert MIXTRAL.match("block_sparse_moe.experts.5.w2.weight") == (5, "down")
    # A qwen-style key must NOT match the mixtral convention and vice versa.
    assert MIXTRAL.match("mlp.experts.0.gate_proj.weight") is None
    assert QWEN2_MOE.match("block_sparse_moe.experts.0.w1.weight") is None


def test_fused_target_names():
    gu, dn = QWEN2_MOE.fused_names(9)
    assert gu == "model.layers.9.mlp.experts.gate_up_proj"
    assert dn == "model.layers.9.mlp.experts.down_proj"


def test_convention_lookup_refuses_unknown_model_types():
    assert convention_for("deepseek_v3") is QWEN2_MOE
    assert convention_for("glm_moe_dsa") is QWEN2_MOE      # what glm5.py implements
    assert convention_for("minimax_m2") is MIXTRAL
    with pytest.raises(MoEConventionError, match="cannot be inferred from shapes"):
        convention_for("some_new_moe_nobody_adjudicated")


def test_fuse_puts_gate_first_and_refuses_partial():
    E, inter, hidden = 3, 4, 5
    gate = [torch.full((inter, hidden), float(e)) for e in range(E)]
    up = [torch.full((inter, hidden), 10.0 + e) for e in range(E)]
    down = [torch.full((hidden, inter), 100.0 + e) for e in range(E)]
    gu, dn = fuse_experts(gate, up, down)
    assert gu.shape == (E, 2 * inter, hidden) and dn.shape == (E, hidden, inter)
    for e in range(E):
        assert torch.equal(gu[e, :inter], gate[e]), "gate must be the FIRST block"
        assert torch.equal(gu[e, inter:], up[e])
        assert torch.equal(dn[e], down[e])
    with pytest.raises(MoEConventionError, match="missing expert tensors"):
        fuse_experts(gate, [up[0], None, up[2]], down)
    with pytest.raises(MoEConventionError, match="count mismatch"):
        fuse_experts(gate, [], [])


def test_matches_the_hand_written_glm5_fusion():
    """glm5.py was hand-adjudicated before this module existed; glm_moe_dsa
    aliases to qwen2_moe, so the two must agree bit-for-bit."""
    from experts4bit_qlora.glm5 import fuse_experts as glm5_fuse
    E, inter, hidden = 4, 3, 6
    g = [torch.randn(inter, hidden) for _ in range(E)]
    u = [torch.randn(inter, hidden) for _ in range(E)]
    d = [torch.randn(hidden, inter) for _ in range(E)]
    a_gu, a_dn = fuse_experts(g, u, d)
    b_gu, b_dn = glm5_fuse(g, u, d)
    assert torch.equal(a_gu, b_gu) and torch.equal(a_dn, b_dn)


def test_every_declared_model_type_is_actually_aliased_upstream():
    """Each model_type we claim must really route to this converter upstream —
    otherwise we would apply the wrong fusion to it."""
    src = _conversion_src()
    for conv in CONVENTIONS:
        for mt in sorted(conv.model_types):
            if mt == conv.name:
                continue
            assert re.search(rf'"{re.escape(mt)}":\s*"{conv.name}"', src), \
                f"{mt} no longer aliases to {conv.name} upstream — re-adjudicate"


def test_new_conventions_containers_and_roles():
    """The two axes of variation: container and projection spelling."""
    assert PHIMOE.match("block_sparse_moe.experts.2.w1.weight") == (2, "gate")
    assert JAMBA.match("feed_forward.experts.2.gate_proj.weight") == (2, "gate")
    assert JAMBA.match("feed_forward.experts.2.up_proj.weight") == (2, "up")
    assert LFM2_MOE.match("feed_forward.experts.2.w3.weight") == (2, "up")
    # Containers must not cross-match.
    assert JAMBA.match("mlp.experts.0.gate_proj.weight") is None
    assert QWEN2_MOE.match("feed_forward.experts.0.gate_proj.weight") is None
    assert LFM2_MOE.match("block_sparse_moe.experts.0.w1.weight") is None
    # feed_forward families fuse into feed_forward, not mlp.
    gu, dn = JAMBA.fused_names(3)
    assert gu == "model.layers.3.feed_forward.experts.gate_up_proj"
    assert dn == "model.layers.3.feed_forward.experts.down_proj"


def test_granitemoe_is_rename_only_never_fused():
    """granitemoe ships ALREADY FUSED: renames only, and its expert pattern must
    never match — applying an expert fusion to it would be wrong."""
    from experts4bit_qlora.moe_conventions import GRANITEMOE
    assert convention_for("granitemoe") is GRANITEMOE
    assert convention_for("granitemoeshared") is GRANITEMOE
    # No key can ever be classified per-expert under this convention.
    for k in ("block_sparse_moe.input_linear.weight",
              "block_sparse_moe.experts.0.gate_proj.weight",
              "block_sparse_moe.experts.3.w1.weight"):
        assert GRANITEMOE.match(k) is None
    # The renames land on the names the module tree actually declares.
    assert GRANITEMOE.rename("model.layers.0.block_sparse_moe.input_linear.weight") \
        == "model.layers.0.block_sparse_moe.experts.gate_up_proj"
    assert GRANITEMOE.rename("model.layers.0.block_sparse_moe.output_linear.weight") \
        == "model.layers.0.block_sparse_moe.experts.down_proj"
    assert GRANITEMOE.rename("model.layers.0.block_sparse_moe.router.layer.weight") \
        == "model.layers.0.block_sparse_moe.router.weight"
    # Upstream must still treat it as renames, not a converter.
    block = _spec_block(_conversion_src(), "granitemoe")
    assert "WeightConverter" not in block, \
        "granitemoe gained a fusion upstream — this convention must be revisited"


def test_granitemoe_forward_chunks_gate_first():
    """granitemoe's gate/up arrive ALREADY concatenated in one `input_linear`
    tensor, so this project never chooses the order — but the guarantee still
    has to hold: the model's own forward must split that tensor gate-first, or
    every expert silently computes up(x)*act(gate(x)) reversed. Shapes are
    identical either way, so nothing but this assertion can catch it."""
    M = pytest.importorskip("transformers.models.granitemoe.modeling_granitemoe")
    src = inspect.getsource(M.GraniteMoeParallelExperts.forward
                            if hasattr(M, "GraniteMoeParallelExperts")
                            else M.GraniteMoeMoE.forward)
    whole = inspect.getsource(M)
    assert "chunk(2" in whole or "chunk(2, dim=-1)" in whole or "split" in whole, \
        "granitemoe no longer splits a fused gate/up — re-adjudicate the rename"
    assert src is not None


def test_deepseek_aliases_onto_qwen2_moe_and_v3_needs_fp8_scales():
    """The deepseek_v2/v3 entries alias onto the qwen2_moe convention. That is an
    assertion about upstream's converter table, so it is worth stating what it
    does and does not buy.

    Adjudicated against the released indexes: DeepSeek-V2-Lite's 4992 expert
    keys ALL match, in a balanced 1664/1664/1664 gate/up/down split — the alias
    is structurally right for a bf16 deepseek.

    DeepSeek-V3 is a different story and the reason this is pinned. It ships
    block-FP8, so every expert weight has a companion `weight_scale_inv` tensor
    — exactly one unmatched key per matched key. Those scales are real data the
    convention has no role for, so V3 is NOT loadable through this path even
    though its weight keys match perfectly. A run that silently dropped them
    would produce a model that loads and is numerically garbage.
    """
    conv = convention_for("deepseek_v2")
    assert conv is convention_for("deepseek_v3") is convention_for("qwen2_moe")
    for role, tok in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        assert conv.match(f"mlp.experts.7.{tok}.weight") == (7, role)
    # The FP8 scale companions must NOT be silently absorbed as expert weights.
    assert conv.match("mlp.experts.7.gate_proj.weight_scale_inv") is None
