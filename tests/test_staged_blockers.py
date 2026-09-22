# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""e4b#648 — WHY each remaining staged family is not wired, as checks rather than prose.

``STAGED_NOT_WIRED`` used to say of four of these: *"convention present, neither
admission route open. No reason recorded here because none was found; that absence
is itself the thing to resolve."* For ``jamba`` and ``lfm2_moe`` the absence turned
out to be that nobody had tried — both were wired on real-checkpoint rows. For the
five left, a reason was found, and this file is it.

Each test asserts the CURRENT upstream or checkpoint fact that blocks admission, so
it **fails when the blocker lifts**. That is the point: a blocker recorded as a
comment rots silently into a stale excuse, while a blocker recorded as a test tells
you the day it stops being true and the family becomes wirable. It is the same
both-directions discipline as :mod:`tests.test_staged_not_wired`, one level down —
that file checks *whether* a family is admitted, this one checks *why not*.

Nothing here asserts that a family SHOULD stay unwired. It asserts what would have
to change first.
"""
from __future__ import annotations

import inspect

import pytest

from experts4bit_qlora.arch.moe_conventions import STAGED_NOT_WIRED


# ─────────────────────────────────────────────────────────────────────────────
# qwen3_vl_moe / qwen3_vl_moe_text — transformers publishes no CausalLM class.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_type", ("qwen3_vl_moe", "qwen3_vl_moe_text"))
def test_qwen3_vl_moe_has_no_causal_lm_class_to_build(model_type):
    """The loader's ONLY build path is ``AutoModelForCausalLM.from_config``.

    transformers ships `Qwen3VLMoeForConditionalGeneration` and no
    `...ForCausalLM`, and neither config class is registered in
    `MODEL_FOR_CAUSAL_LM_MAPPING` — so admitting either model_type would get past
    the architecture gate and then raise while BUILDING the tree, before a single
    weight is read. That is a transformers-side gap, not an e4b decision, and it
    is why #637/#639's conditional-transpose work could adjudicate the family's
    expert LAYOUT (which the int4 planner uses) while the loader still refuses it.

    Wiring needs a build path that is not `AutoModelForCausalLM` — the same kind
    of step `MULTIMODAL_CKPT_PREFIX` represents for a composite checkpoint, but
    for the MODEL rather than the keys.
    """
    transformers = pytest.importorskip("transformers")
    assert model_type in STAGED_NOT_WIRED

    cfg_cls = transformers.CONFIG_MAPPING[model_type]
    assert cfg_cls not in transformers.MODEL_FOR_CAUSAL_LM_MAPPING, (
        f"{model_type} now has a CausalLM class upstream — the build blocker is "
        f"gone and the family should be re-evaluated for wiring (e4b#648)")

    import transformers.models.qwen3_vl_moe as mod
    assert not [n for n in dir(mod) if n.endswith("ForCausalLM")], (
        "qwen3_vl_moe now exports a ForCausalLM class — re-evaluate wiring")


# ─────────────────────────────────────────────────────────────────────────────
# axk1 — the keymap does not fit the registry that would have to hold it.
# ─────────────────────────────────────────────────────────────────────────────

def test_axk1s_keymap_does_not_fit_the_ckpt_key_rewriter_contract():
    """#509 said admission and the rewriter must land together. They cannot yet.

    ``CKPT_KEY_REWRITERS`` maps model_type -> a callable the loader applies
    ONE KEY AT A TIME (``rewrite(k)`` -> new name, or ``None`` to drop it);
    ``rename_deepseek_v4_key``, its only entry, has exactly that shape.
    ``rewrite_axk1_keys`` takes ``(checkpoint_keys, first_k_dense_replace)`` — a
    whole key LIST plus a config value — and returns ``(kept, dropped)``. It
    cannot be registered as-is, and calling it with a single key raises.

    So "wire axk1" is not an admission row plus a dict entry. It is: read
    ``first_k_dense_replace`` off the config, close over it, and adapt the
    list-shaped keymap to the per-key contract (or widen the contract). Naming
    that is the point — the record used to imply the two were the same size of
    change.
    """
    from experts4bit_qlora.arch.axk1 import rewrite_axk1_keys
    from experts4bit_qlora.loader import CKPT_KEY_REWRITERS

    params = list(inspect.signature(rewrite_axk1_keys).parameters)
    assert params == ["checkpoint_keys", "first_k_dense_replace"], params
    with pytest.raises(TypeError):
        rewrite_axk1_keys("model.layers.0.post_mlp_layernorm.weight")

    # Every registered rewriter IS per-key, which is the contract axk1 misses.
    for model_type, fn in CKPT_KEY_REWRITERS.items():
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 1, (
            f"{model_type}'s rewriter takes {len(sig.parameters)} arguments; the "
            f"loader calls rewrite(key)")

    assert "axk1" not in CKPT_KEY_REWRITERS


def test_axk1s_layer_conditional_rename_is_real_on_the_released_checkpoint():
    """Why the keymap needs per-layer knowledge at all, pinned to the release.

    ``skt/A.X-K1`` ships ``post_mlp_layernorm`` on BOTH its dense layer 0
    (``first_k_dense_replace=1``) and its MoE layers, but the module only exists
    on MoE layers — so the key must be RENAMED on one and DROPPED on the other. A
    single global substring rename cannot express that, which is the whole reason
    ``arch/axk1.py`` exists and the reason its signature is list-shaped.

    Recorded as the released index's own key list rather than re-downloading
    1.04 TB: the facts below were read from
    ``model.safetensors.index.json`` (976 keys) on 2026-09-21.
    """
    from experts4bit_qlora.arch.axk1 import AXK1_IGNORE_PARAM_PATTERNS, rewrite_axk1_keys

    released = [
        "model.layers.0.post_mlp_layernorm.weight",   # dense layer: present on disk
        "model.layers.1.post_mlp_layernorm.weight",   # MoE layer: present on disk
        "model.layers.1.mlp.experts.gate_up_proj",    # pre-fused, present
        "model.layers.1.mlp.experts.down_proj",       # pre-fused, present
    ]
    kept, dropped = rewrite_axk1_keys(released, first_k_dense_replace=1)
    assert dropped == ["model.layers.0.post_mlp_layernorm.weight"]
    assert "model.layers.1.mlp.post_mlp_layernorm.weight" in kept
    # The router bias the checkpoint does NOT ship stays an ignore-pattern, not a
    # missing weight (verified absent from the released index).
    assert AXK1_IGNORE_PARAM_PATTERNS


# ─────────────────────────────────────────────────────────────────────────────
# jetmoe — the module tree has no `experts` submodule to replace.
# ─────────────────────────────────────────────────────────────────────────────

def test_jetmoes_tree_has_no_experts_submodule_for_the_loader_to_replace():
    """The loader installs its stack with ``setattr(parent, leaf, Experts4bit(...))``
    at ``model.layers.{i}.{fused_prefix}``. JetMoE has no such module.

    Its built tree puts the fused stacks DIRECTLY on the MoE block as
    ``mlp.input_linear`` [E, 2I, H] and ``mlp.output_linear`` [E, H, I] — the
    granitemoe shape, but granitemoe's TREE declares
    ``block_sparse_moe.experts.gate_up_proj``, so a checkpoint-side rename is
    enough there. Here both sides say ``input_linear``, so there is nothing named
    ``experts`` to replace and nothing named ``gate_up_proj`` to fill.

    JetMoE is also a DUAL MoE: ``self_attention.experts`` is a second expert
    stack (``JetMoeMoA``). e4b has no representation for attention experts, so a
    naive admission would quantize the MLP experts, leave the attention experts
    in bf16, and still report a successful load — the kind of quietly-partial
    result the zero-expert guard exists to prevent at the other extreme.
    """
    pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    from experts4bit_qlora.arch.moe_conventions import convention_for

    assert "jetmoe" in STAGED_NOT_WIRED
    with accelerate.init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(
            transformers.CONFIG_MAPPING["jetmoe"]())

    mlp = model.get_submodule("model.layers.0.mlp")
    assert not hasattr(mlp, "experts"), (
        "jetmoe's MoE block now has an `experts` submodule — the structural "
        "blocker is gone and the family should be re-evaluated (e4b#648)")
    declared = dict(mlp.named_parameters(recurse=True))
    assert "input_linear.weight" in declared and "output_linear.weight" in declared
    assert declared["input_linear.weight"].dim() == 3

    # The convention's fused_prefix names a module path that does not exist.
    prefix = convention_for("jetmoe").fused_prefix
    with pytest.raises(AttributeError):
        model.get_submodule(f"model.layers.0.{prefix}")

    # The second, attention-side MoE that e4b cannot represent.
    moa = model.get_submodule("model.layers.0.self_attention.experts")
    assert "input_linear.weight" in dict(moa.named_parameters(recurse=True))


# ─────────────────────────────────────────────────────────────────────────────
# dbrx — flat 2-D stacks, not the 3-D the expert primitive takes.
# ─────────────────────────────────────────────────────────────────────────────

def test_dbrx_expert_stacks_are_flat_2d_not_the_3d_experts4bit_requires():
    """``Experts4bit.from_float`` refuses anything that is not 3-D.

    DBRX stores each projection as ONE flat ``[E * ffn_hidden, hidden]`` tensor
    (``ffn.experts.mlp.w1`` / ``v1`` / ``w2``) and the module declares them the
    same flat way, so loading is pure passthrough — which is exactly why the
    convention records ``roles={}``. But e4b's primitive wants
    ``[E, out, in]``, and reshaping a flat stack into it is a real change with a
    real orientation decision (the convention already pins w1=gate, v1=up,
    w2=down from ``DbrxExpertGLU.forward`` so that decision is not re-guessed
    later), not an admission row.
    """
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.dbrx.modeling_dbrx")
    from experts4bit_qlora._vendor.experts import ExpertsNbit

    assert "dbrx" in STAGED_NOT_WIRED
    # Same construction tests/test_fused_layout_probe.py uses: a duck-typed config,
    # because transformers' own default DbrxConfig is incomplete in 5.17.
    import types

    glu = modeling.DbrxExpertGLU(types.SimpleNamespace(
        hidden_size=6, ffn_hidden_size=4, moe_num_experts=2,
        ffn_act_fn={"name": "silu"}))
    shapes = {n: tuple(p.shape) for n, p in glu.named_parameters(recurse=False)}
    assert set(shapes) == {"w1", "v1", "w2"}, shapes
    for name, shape in shapes.items():
        assert len(shape) == 2, (
            f"dbrx's {name} is now {len(shape)}-D — if it became [E, out, in] the "
            f"flat-stack blocker is gone and the family should be re-evaluated")

    # The primitive's own refusal, so the claim is the code's and not this test's.
    with pytest.raises(ValueError, match="3D"):
        ExpertsNbit.from_float(torch.zeros(4, 8), torch.zeros(4, 8))


# ─────────────────────────────────────────────────────────────────────────────
# Every staged family must have a reason here.
# ─────────────────────────────────────────────────────────────────────────────

def test_every_staged_family_is_named_by_a_blocker_test_in_this_file():
    """The gap this file closes must not reopen for a family added later.

    If a new convention arrives that no loader admits, ``test_staged_not_wired``
    forces it into ``STAGED_NOT_WIRED``; this forces someone to say WHY here.
    """
    import pathlib

    src = pathlib.Path(__file__).read_text()
    missing = [mt for mt in STAGED_NOT_WIRED if mt not in src]
    assert not missing, (
        f"{sorted(missing)} are staged but no blocker is recorded in this file — "
        f"'no reason recorded' is the state e4b#648 exists to end")
