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

from experts4bit_qlora.arch.moe_conventions import (  # noqa: E402
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


#: Only conventions that fuse a GATE with up have a gate/up source order to
#: check. granitemoe ships pre-fused (no gate/up pair upstream); nemotron_h is
#: non-gated (up/down only, its stacking is tested separately). Both are
#: excluded here and pinned by their own dedicated tests.
FUSING = [c for c in CONVENTIONS if c.roles and c.gated]


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
    from experts4bit_qlora.arch.glm5 import fuse_experts as glm5_fuse
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
        # A convention that never fuses (granitemoe, gptoss: pre-fused on disk)
        # cannot misapply a fusion, so the upstream-alias requirement does not
        # apply — its correctness is pinned by its own dedicated test. This
        # check exists to catch a FUSING convention drifting off its converter.
        if not conv.roles:
            continue
        for mt in sorted(conv.model_types):
            if mt == conv.name:
                continue
            # Types adjudicated from a released checkpoint may be NEWER than the
            # installed transformers' explicit alias list — they resolve through
            # the converter API but never appear in this source text. For those
            # the authoritative check is the converter-API pair
            # (test_alias_set_matches_transformers_converter_table +
            # test_adjudicated_released_type_resolves_to_its_convention), so this
            # source-text scan skips them rather than failing on their absence.
            if mt in ADJUDICATED_RELEASED:
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
    from experts4bit_qlora.arch.moe_conventions import GRANITEMOE
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


# --- coverage drift vs transformers' own converter table --------------------
# The conventions here alias many model_types onto a few converter specs. That
# aliasing is only correct as long as transformers agrees those model_types
# really share the converter. transformers can add a model_type to a converter
# (a coverage gap we should close) or change a converter out from under an alias
# (a correctness bug). Both are invisible to every other test, because they
# happen upstream. These pin the alias sets to transformers' OWN authority —
# `get_checkpoint_conversion_mapping` — so a drift breaks CI with a message that
# names exactly which model_type moved.

def _converter_signature(model_type):
    """A hashable signature of transformers' checkpoint converter for a type,
    or None if it exposes none. Compares source->target patterns and the op
    sequence, so two model_types with byte-identical conversion rules compare
    equal regardless of how they are registered."""
    gc = pytest.importorskip(
        "transformers.conversion_mapping").get_checkpoint_conversion_mapping
    try:
        spec = gc(model_type)
    except Exception:
        return None
    if not spec:
        return ()
    parts = []
    for w in spec:
        tgt = str(getattr(w, "target_patterns", None))
        # A convention governs how the EXPERT stacks are built. A model may add
        # non-expert renames (ernie/exaone rename a router e_score bias) without
        # changing that; those must not count as a different convention. Filter
        # to converters whose target is an expert tensor.
        if "expert" not in tgt:
            continue
        ops = tuple(type(o).__name__ for o in getattr(w, "operations", []) or [])
        parts.append((str(getattr(w, "source_patterns", None)), tgt, ops))
    return tuple(sorted(parts))


def _types_sharing_converter(root_type):
    """Every model_type transformers maps to the SAME converter as root_type."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    target = _converter_signature(root_type)
    return {mt for mt in CONFIG_MAPPING_NAMES
            if _converter_signature(mt) == target}


#: Types that transformers maps to a covered EXPERT converter but which e4b
#: intentionally does not alias into THAT convention — each with the reason it
#: is not a coverage bug. New entries appearing outside this set mean
#: transformers shipped a MoE family worth adjudicating.
_KNOWN_UNCLAIMED = {
    # Covered by its OWN convention (same expert fusion, different router/attn):
    "phimoe",
    # Huge, served through the NVMe arena path, not the in-VRAM planner:
    "kimi_k25",
    # OCR / multimodal composites whose text tower is not separately validated;
    # and pre-release / obscure types not yet checked against a real checkpoint:
    "deepseek_ocr2", "qwen3_5_moe_text", "axk1", "axk2", "mimo_v2_flash",
    # Appeared in transformers between 2026-08-26 10:35 and 13:34 UTC,
    # mapped to the qwen2_moe converter. NOT aliased: sharing a
    # converter is not evidence of sharing e4b's expert-fusion or
    # router layout, and aliasing on that assumption is how a wrong
    # convention reaches a real checkpoint silently. Named here as
    # UNVERIFIED until it can be adjudicated against one.
    "qwen4_exp_text",
    # Appeared 2026-08-26, also mapped to the qwen2_moe converter.
    # NOT aliased, and the NAME is the trap: e4b HAS a glm5 lane
    # (arch/glm5.py) but it covers `glm_moe_dsa` -- DeepSeek-V3
    # lineage, MLA attention, DSA lightning-indexer. A model routed
    # to the qwen2_moe converter is a different family, so neither
    # alias is defensible. arch/glm5.py's own standard is adjudication
    # against a real released index.json AND the instantiated tree,
    # "not convention"; that standard cannot be met from a converter
    # mapping alone.
    "glm5_next",
}


def _covered_by_any_convention(model_type):
    try:
        convention_for(model_type)
        return True
    except MoEConventionError:
        return False


@pytest.mark.parametrize("conv_name, root", [("qwen2_moe", "qwen2_moe"),
                                             ("mixtral", "mixtral")])
def test_alias_set_matches_transformers_converter_table(conv_name, root):
    """Two guarantees against upstream drift.

    Correctness (hard): every model_type e4b aliases into this convention must
    still share this converter in transformers. A stale alias is a wrong load.

    Coverage (tracked): every model_type transformers maps to this converter is
    either covered by SOME e4b convention or listed in _KNOWN_UNCLAIMED with a
    reason. A new name here means transformers shipped a MoE family to look at —
    the test names it rather than letting it fail silently at load time."""
    from experts4bit_qlora.arch.moe_conventions import MoEConventionError  # noqa: F401
    conv = next(c for c in CONVENTIONS if c.name == conv_name)
    ours = set(conv.model_types)
    theirs = _types_sharing_converter(root)

    stale = (ours - theirs) - set(ADJUDICATED_RELEASED)
    assert not stale, (
        f"{conv_name}: e4b aliases {sorted(stale)} but transformers no longer "
        f"maps them to this converter — the alias may now be wrong")

    uncovered = {mt for mt in (theirs - ours)
                 if not _covered_by_any_convention(mt)} - _KNOWN_UNCLAIMED
    assert not uncovered, (
        f"{conv_name}: transformers maps {sorted(uncovered)} to this converter "
        f"and no e4b convention covers them — adjudicate against a real "
        f"checkpoint and either alias or add to _KNOWN_UNCLAIMED with a reason")


def test_prefused_transpose_family_has_its_own_convention_not_qwen2_moe():
    """qwen3_vl_moe ships experts PRE-FUSED with a Transpose(1,2), a different
    checkpoint layout from qwen2_moe's per-expert MergeModulelist. It gets its
    OWN convention (QWEN3_VL_MOE, transpose-only) — it must never resolve to
    qwen2_moe, which would gather keys that are already stacked."""
    from experts4bit_qlora.arch.moe_conventions import QWEN2_MOE, QWEN3_VL_MOE
    assert _converter_signature("qwen3_vl_moe") != _converter_signature("qwen2_moe")
    assert convention_for("qwen3_vl_moe") is QWEN3_VL_MOE
    assert convention_for("qwen3_vl_moe") is not QWEN2_MOE
    # transpose-only: never per-expert, and it marks the two pre-fused stacks.
    assert not QWEN3_VL_MOE.roles
    assert QWEN3_VL_MOE.needs_transpose("x.mlp.experts.gate_up_proj")
    assert QWEN3_VL_MOE.needs_transpose("x.mlp.experts.down_proj")
    assert not QWEN3_VL_MOE.needs_transpose("x.mlp.experts.gate_up_proj_bias")


def test_gptoss_is_prefused_mxfp4_passthrough_never_per_expert():
    """gpt-oss ships experts PRE-FUSED and MXFP4-quantized. The convention must
    never treat a key as per-expert (the planner pairs blocks+scales and
    dequantizes upstream of it); everything reaches it as a passthrough."""
    from experts4bit_qlora.arch.moe_conventions import GPTOSS
    assert convention_for("gpt_oss") is GPTOSS
    assert not GPTOSS.roles                      # never fuses
    for k in ("mlp.experts.gate_up_proj_blocks", "mlp.experts.0.gate_proj.weight",
              "mlp.experts.down_proj"):
        assert GPTOSS.match(k) is None
    # No renames: the synthesized base name already equals the tree target.
    assert GPTOSS.rename("model.layers.0.mlp.experts.gate_up_proj") == \
        "model.layers.0.mlp.experts.gate_up_proj"


def test_jetmoe_dual_moe_is_native_passthrough_never_drops_attention_experts():
    """JetMoE is MoE in both MLP and attention. Because its converter is native
    (pre-fused, no conversion), both pass straight through — the planner never
    fuses or drops. Pinned so a future 'optimization' that treats it as
    per-expert trips here."""
    from experts4bit_qlora.arch.moe_conventions import JETMOE
    assert convention_for("jetmoe") is JETMOE
    assert not JETMOE.roles                         # never per-expert
    for k in ("mlp.input_linear.weight", "self_attention.experts.input_linear.weight",
              "mlp.experts.0.gate_proj.weight"):
        assert JETMOE.match(k) is None
    assert JETMOE.transpose_re is None              # nothing transposed
    assert JETMOE.rename("self_attention.experts.input_linear.weight") == \
        "self_attention.experts.input_linear.weight"


def test_dbrx_is_flat_native_passthrough_with_pinned_roles():
    """DBRX stores each projection as one flat [E*inter, hidden] tensor that the
    transformers tree declares identically, so it loads as pure passthrough.
    The w1/v1/w2 -> gate/up/down roles are pinned from the forward for a future
    fusion path but unused by the loader; the expert pattern never matches."""
    from experts4bit_qlora.arch.moe_conventions import DBRX
    assert convention_for("dbrx") is DBRX
    assert not DBRX.roles                        # passthrough, never per-expert
    assert DBRX.transpose_re is None
    for k in ("ffn.experts.mlp.w1", "ffn.experts.mlp.v1", "ffn.experts.mlp.w2",
              "ffn.experts.mlp_experts.0.w1.weight"):
        assert DBRX.match(k) is None
    assert DBRX.rename("transformer.blocks.0.ffn.experts.mlp.w1") == \
        "transformer.blocks.0.ffn.experts.mlp.w1"


#: Model types adjudicated against a REAL released checkpoint on a given date,
#: mapped to the convention they must resolve to. This is the durable half of
#: coverage: the automatic drift test only sees types the installed transformers
#: enumerates in CONFIG_MAPPING_NAMES, so a family HF has shipped but
#: transformers has not registered yet is invisible to it. These are pinned by
#: hand from the checkpoint's own keys + the converter API, so a dropped alias
#: trips here regardless of the installed transformers version.
ADJUDICATED_RELEASED = {
    # per-expert gate/up/down + block-FP8, confirmed on the real index 2026-08-11
    "axk2": "qwen2_moe",             # skt/A.X-K2, 256 experts
    "mimo_v2_flash": "qwen2_moe",    # XiaomiMiMo/MiMo-V2-Flash, 256 experts
    # MiniMax-M3-VL: w1/w3/w2 block_sparse_moe under a language_model.model.
    # prefix. Validated against the 23416-key index (57 layers, complete
    # 128-expert stacks); its converter signature differs from bare mixtral only
    # cosmetically (source-pattern spelling), so the drift check exempts it.
    "minimax_m3_vl": "mixtral",
    # Kimi-K2-Instruct: per-expert qwen2_moe + block-FP8 at scale. Its converter
    # is EMPTY in current transformers (incomplete support), so this is
    # adjudicated purely from the 139644-key released index: 69120 expert keys,
    # perfectly balanced gate/up/down, 0 unmatched.
    "kimi_k2": "qwen2_moe",
    # Kimi-K2.5: per-expert qwen2_moe + compressed-tensors int4, under a
    # language_model.model. prefix. 69120 triples in the real index, 0 leftover.
    "kimi_k25": "qwen2_moe",
}


@pytest.mark.parametrize("model_type, expected", sorted(ADJUDICATED_RELEASED.items()))
def test_adjudicated_released_type_resolves_to_its_convention(model_type, expected):
    """A type verified against a real checkpoint must keep resolving to the
    convention it was adjudicated into — even if the installed transformers does
    not enumerate it, which is exactly when the automatic drift test goes blind."""
    # The guarantee is resolution: a type adjudicated against a real checkpoint
    # must keep resolving to the convention it was validated into. The converter
    # SIGNATURE is deliberately not asserted equal — some real families (minimax_
    # m3_vl) share a convention's expert LAYOUT while their converter spec differs
    # cosmetically (source-pattern spelling, VL prefix). The adjudication basis is
    # the real key structure, recorded per entry above, not string equality here.
    assert convention_for(model_type).name == expected


def test_qwen3_5_moe_is_native_prefused_passthrough_no_transpose():
    """qwen3_5_moe's converter is EMPTY (native) — unlike qwen3_vl_moe it has no
    Transpose, so the pre-fused stacks match the tree as-is. Never per-expert,
    nothing transposed. Pinned so it is not confused with its transposed sibling."""
    from experts4bit_qlora.arch.moe_conventions import QWEN3_5_MOE, QWEN3_VL_MOE
    assert convention_for("qwen3_5_moe") is QWEN3_5_MOE
    assert QWEN3_5_MOE is not QWEN3_VL_MOE
    assert not QWEN3_5_MOE.roles
    assert QWEN3_5_MOE.transpose_re is None       # native: no transpose (the sibling has one)
    for k in ("mlp.experts.gate_up_proj", "mlp.experts.0.gate_proj.weight"):
        assert QWEN3_5_MOE.match(k) is None


def test_nemotron_h_is_non_gated_up_down_in_the_mixer_container():
    """Nemotron-H has NO gate: experts are up_proj + down_proj only, in a mixer
    block. It must declare roles {up, down}, be non-gated, and target up_proj
    (not gate_up_proj). A gate token must never match."""
    from experts4bit_qlora.arch.moe_conventions import NEMOTRON_H
    assert convention_for("nemotron_h") is NEMOTRON_H
    assert NEMOTRON_H.gated is False
    assert set(NEMOTRON_H.roles.values()) == {"up", "down"}
    assert NEMOTRON_H.match("mixer.experts.3.up_proj.weight") == (3, "up")
    assert NEMOTRON_H.match("mixer.experts.3.down_proj.weight") == (3, "down")
    assert NEMOTRON_H.match("mixer.experts.3.gate_proj.weight") is None
    # shared_expert is passthrough, not a per-expert match
    assert NEMOTRON_H.match("mixer.shared_experts.up_proj.weight") is None
    first, down = NEMOTRON_H.fused_names(2)
    assert first == "model.layers.2.mixer.experts.up_proj"     # NOT gate_up_proj
    assert down == "model.layers.2.mixer.experts.down_proj"


def test_stack_experts_stacks_without_concatenating_a_gate():
    """The non-gated fusion stacks up and down each on their own — no gate to
    concatenate, so up_proj keeps [E, inter, hidden] not [E, 2*inter, hidden]."""
    from experts4bit_qlora.arch.moe_conventions import stack_experts
    E, inter, hidden = 3, 4, 5
    up = [torch.full((inter, hidden), float(e)) for e in range(E)]
    down = [torch.full((hidden, inter), 100.0 + e) for e in range(E)]
    up_proj, down_proj = stack_experts(up, down)
    assert up_proj.shape == (E, inter, hidden)      # NOT 2*inter
    assert down_proj.shape == (E, hidden, inter)
    for e in range(E):
        assert torch.equal(up_proj[e], up[e])
    with pytest.raises(MoEConventionError, match="missing expert tensors"):
        stack_experts([up[0], None, up[2]], down)


def test_loader_expert_paths_agree_with_conventions():
    """The quantized streaming loader (loader.SUPPORTED_ARCHITECTURES) and the
    convention system must not disagree about where a family's experts live.
    Where both cover a model_type, the loader's path must equal the convention's
    fused_prefix — otherwise the two source-of-truth systems would quantize and
    plan against different submodules. expert_layout_for() makes the convention
    authoritative; this pins that the migration stays consistent."""
    loader = pytest.importorskip("experts4bit_qlora.loader")
    for mt, path in loader.SUPPORTED_ARCHITECTURES.items():
        try:
            conv = convention_for(mt)
        except MoEConventionError:
            continue                        # loader-only dedicated-quant special
        assert conv.fused_prefix == path, (
            f"{mt}: loader path {path!r} != convention fused_prefix "
            f"{conv.fused_prefix!r} — the two systems disagree")


def test_expert_layout_for_sources_gate_from_the_convention():
    """has_gate must come from the convention, so a non-gated family (nemotron_h)
    is not quantized as if it were SwiGLU. Previously the loader hardcoded
    has_gate=True."""
    loader = pytest.importorskip("experts4bit_qlora.loader")
    assert loader.expert_layout_for("olmoe") == ("mlp.experts", True)
    assert loader.expert_layout_for("nemotron_h") == ("mixer.experts", False)
    # a dedicated-quant special with no convention still resolves via the map
    assert loader.expert_layout_for("gemma4") == ("experts", True)


def test_gemma4_is_a_prefused_convention_in_the_module_layout():
    """Adjudicated against the released google/gemma-4-26B-A4B-it index and
    Gemma4TextExperts.forward (P30): one stacked tensor per projection under
    ``layer.experts``, gate rows first, no transpose, no per-expert keys."""
    from experts4bit_qlora.arch.moe_conventions import convention_for
    for mt in ("gemma4_text", "gemma4"):
        c = convention_for(mt)
        assert c.name == "gemma4" and c.fused_prefix == "experts"
        assert c.transpose_re is None and c.roles == {} and c.renames == ()
        assert c.expert_re.search("model.language_model.layers.0.experts.gate_up_proj") is None
    # the int4 lane's pre-fused matcher sees these keys as layer 0's stacks
    from experts4bit_qlora.engines.int4_experts import _FUSED_TARGET
    m = _FUSED_TARGET.match("model.language_model.layers.0.experts.gate_up_proj")
    assert m and m.group("layer") == "0" and m.group("role") == "gate_up_proj"
    m = _FUSED_TARGET.match("model.language_model.layers.7.experts.down_proj")
    assert m and m.group("layer") == "7" and m.group("role") == "down_proj"
    # the DENSE mlp beside the experts is not an expert stack
    assert _FUSED_TARGET.match("model.language_model.layers.0.mlp.gate_proj.weight") is None

