# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The loader walks raw checkpoint keys, so it must apply transformers' renamings.

Both released ERNIE 4.5 checkpoints (21B-A3B, 300B-A47B) store
`model.layers.N.mlp.moe_statics.e_score_correction_bias` at the MoE-block level —
27 and 51 keys respectively — while the module tree carries it under `mlp.gate.`.
Walking the disk name produced `Ernie4_5_MoeSparseMoeBlock has no attribute
'moe_statics'`, naming neither the architecture nor the problem.

transformers reconciles this through a per-model_type table, so these pin that we
read it rather than keep a second copy that can drift. Pure string work, so it runs
on CPU where a regression is actually caught.
"""
import pytest

pytest.importorskip("torch")

from experts4bit_qlora.loader import _checkpoint_key_renamings, _rename_checkpoint_key  # noqa: E402


def test_suffix_rename_rewrites_only_the_matching_tail():
    r = [("mlp.moe_statics.e_score_correction_bias",
          "mlp.gate.moe_statics.e_score_correction_bias")]
    got = _rename_checkpoint_key("model.layers.7.mlp.moe_statics.e_score_correction_bias", r)
    assert got == "model.layers.7.mlp.gate.moe_statics.e_score_correction_bias"


def test_unrelated_keys_pass_through_untouched():
    """Arming: a rename that fired on everything would satisfy the test above."""
    r = [("mlp.moe_statics.e_score_correction_bias",
          "mlp.gate.moe_statics.e_score_correction_bias")]
    for k in ("model.layers.7.self_attn.q_proj.weight",
              "model.embed_tokens.weight",
              "model.layers.7.mlp.gate.weight"):
        assert _rename_checkpoint_key(k, r) == k
    assert _rename_checkpoint_key("model.layers.0.mlp.moe_statics.other", r) == \
        "model.layers.0.mlp.moe_statics.other"


def test_a_missing_or_changed_transformers_table_degrades_to_no_renamings():
    """It is a transformers-internal table; an import failure must not break loading."""
    assert _checkpoint_key_renamings("a_model_type_that_does_not_exist") == []


def test_the_ernie_renaming_is_actually_present_upstream():
    """If transformers drops or renames this entry, we want a red test, not a silent
    return to the opaque AttributeError."""
    got = _checkpoint_key_renamings("ernie4_5_moe")
    if not got:
        pytest.skip("this transformers build exposes no conversion mapping for ernie4_5_moe")
    assert any(s.endswith("mlp.moe_statics.e_score_correction_bias")
               and t.endswith("mlp.gate.moe_statics.e_score_correction_bias")
               for s, t in got), got
