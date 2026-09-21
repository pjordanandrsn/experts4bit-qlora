"""e4b#648 / #509 -- the staged-not-wired families, and the activation that
silently defaulted.

Two families have a convention, (for axk1) a keymap, and passing unit tests, and
no loader path admits either. Everything a reader would check says "supported";
the one table that decides says nothing at all. ``STAGED_NOT_WIRED`` states it,
and this file makes the statement a CHECK rather than a comment -- it fails in
both directions, so wiring a family without delisting it fails, and adding a
convention nothing admits without listing it fails too.

The activation half is a real bug rather than a documentation gap: nemotron_h
names its activation ``mlp_hidden_act`` (``relu2``) and declares neither field the
loader used to read, so it resolved the ``"silu"`` DEFAULT -- and nemotron_h is
non-gated, so ``down(act(up(x)))`` with SiLU instead of ReLU squared is a
different function with every shape agreeing.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")

from experts4bit_qlora import loader  # noqa: E402
from experts4bit_qlora.arch.moe_conventions import (  # noqa: E402
    CONVENTIONS,
    STAGED_NOT_WIRED,
    MoEConventionError,
)


def _admitted(model_type: str) -> bool:
    """Exactly the loader's own admission rule."""
    return (model_type in loader.SUPPORTED_ARCHITECTURES
            or bool(loader._read_compatible_convention(model_type)))


# ─────────────────────────────────────────────────────────────────────────────
# STAGED_NOT_WIRED must describe the loader, in both directions.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_type", sorted(STAGED_NOT_WIRED))
def test_a_staged_family_is_still_not_admitted(model_type):
    """If one gets wired, this fails and the entry must be removed in that change."""
    assert not _admitted(model_type), (
        f"{model_type} is now admitted by the loader, so it is no longer staged — "
        f"remove it from STAGED_NOT_WIRED and from README-LAYOUT.md's note (e4b#648)"
    )


def test_no_reachable_convention_is_quietly_unadmitted():
    """The other direction: a family nothing admits must SAY so.

    Without this, adding a convention and forgetting the loader row reproduces
    exactly the state #509 filed — implemented, exported, documented, tested, and
    refused.
    """
    unadmitted = {mt for conv in CONVENTIONS for mt in conv.model_types
                  if not _admitted(mt)}
    undeclared = unadmitted - set(STAGED_NOT_WIRED)
    assert not undeclared, (
        f"conventions exist for {sorted(undeclared)} but no loader path admits them, "
        f"and they are not listed in STAGED_NOT_WIRED. Either wire them or say so — "
        f"an unreachable family that reads as supported is the state #509 is about."
    )


def test_axk2_is_admitted_and_is_not_staged():
    """The contrast that makes the distinction real.

    axk2 has no SUPPORTED_ARCHITECTURES entry either, but it aliases onto
    qwen2_moe, which IS read-compatible. "Has a convention" and "is loadable" are
    different questions, and #509 flagged axk2 as possibly staged without checking.
    """
    assert _admitted("axk2")
    assert "axk2" not in STAGED_NOT_WIRED


def test_axk1_is_unreachable_twice_over():
    """#509's core point, pinned: admission and the rewriter must land together.

    Wiring only the admission would admit the architecture and never apply
    ``rewrite_axk1_keys`` — a silent wrong-key-mapping path, worse than the
    refusal it replaced.
    """
    assert "axk1" not in loader.SUPPORTED_ARCHITECTURES
    assert not loader._read_compatible_convention("axk1")
    assert "axk1" not in getattr(loader, "CKPT_KEY_REWRITERS", {}), (
        "rewrite_axk1_keys is registered but axk1 is still not admitted, or vice "
        "versa — the two must be wired in the same change (#509)"
    )


def test_the_layout_doc_points_at_the_registry():
    """Prose and data must not drift; the doc names the authority."""
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[1]
           / "experts4bit_qlora" / "README-LAYOUT.md").read_text()
    assert "STAGED_NOT_WIRED" in doc
    for model_type in STAGED_NOT_WIRED:
        assert model_type in doc, f"{model_type} is staged but README-LAYOUT.md omits it"


def test_the_stated_counts_match_the_set():
    """A hand-written count is prose about data, so it is checked against the data.

    Both the registry docstring and README-LAYOUT.md say how many model_types are
    staged and across how many conventions. The first version of that sentence said
    "ten model_types across seven conventions" when the conventions were eight --
    wrong the day it was written, and invisible because nothing read it. Now the
    numbers come from the same frozenset the loader is asserted against.
    """
    import pathlib
    import experts4bit_qlora.arch.moe_conventions as mc

    n_types = len(STAGED_NOT_WIRED)
    n_conventions = len({conv.name for conv in CONVENTIONS
                         if conv.model_types & STAGED_NOT_WIRED})
    phrase = f"{n_types} model_types across {n_conventions} conventions"
    doc = (pathlib.Path(__file__).resolve().parents[1]
           / "experts4bit_qlora" / "README-LAYOUT.md").read_text()
    assert phrase in doc, (
        f"README-LAYOUT.md must state {phrase!r} — it is derived from "
        f"STAGED_NOT_WIRED, not written by hand")
    assert phrase in mc.__doc__ or phrase in pathlib.Path(mc.__file__).read_text(), (
        f"the STAGED_NOT_WIRED record must state {phrase!r}")


# ─────────────────────────────────────────────────────────────────────────────
# The activation the loader used to default.
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(**fields):
    return types.SimpleNamespace(**fields)


def test_nemotron_hs_activation_field_is_read_not_defaulted():
    """The bug, on the real field naming.

    inference-optimization/NemotronH-0.3B-A0.3B declares `mlp_hidden_act: relu2`
    and NEITHER `hidden_act` nor `hidden_activation`. Before #648 this resolved
    to the "silu" default.
    """
    name, field = loader._expert_activation_name(
        _cfg(mlp_hidden_act="relu2", mamba_hidden_act="silu"), "nemotron_h")
    assert (name, field) == ("relu2", "mlp_hidden_act")


def test_an_undeclared_activation_is_refused_not_defaulted_to_silu():
    """"No declaration" and "declared silu" must not look the same.

    That equivalence is exactly what hid the nemotron_h case: the pre-existing
    guard warns on an unknown activation NAME, and a defaulted "silu" is both
    known and explicitly excluded from the warning, so nothing fired.
    """
    with pytest.raises(MoEConventionError, match="declares none of"):
        loader._expert_activation_name(_cfg(num_experts=4), "a_family_nobody_checked")


def test_a_declared_silu_still_resolves():
    """The refusal is scoped to ABSENCE; an explicit silu is unaffected."""
    assert loader._expert_activation_name(_cfg(hidden_act="silu"), "qwen3_moe") == \
        ("silu", "hidden_act")
    assert loader._expert_activation_name(
        _cfg(hidden_activation="gelu_pytorch_tanh"), "gemma4_text") == \
        ("gelu_pytorch_tanh", "hidden_activation")


def test_field_priority_is_most_specific_first():
    """A config carrying several must resolve the one the expert path uses."""
    name, field = loader._expert_activation_name(
        _cfg(hidden_activation="gelu_pytorch_tanh", hidden_act="silu",
             mlp_hidden_act="relu2"), "probe")
    assert (name, field) == ("gelu_pytorch_tanh", "hidden_activation")


@pytest.mark.parametrize("model_type", sorted(loader._ACTIVATION_UNDECLARED_OK))
def test_an_exempt_family_resolves_to_silu_with_its_reason(model_type):
    """The exemptions are narrow, named, and carry their evidence."""
    name, field = loader._expert_activation_name(_cfg(num_experts=4), model_type)
    assert name == "silu"
    assert loader._ACTIVATION_UNDECLARED_OK[model_type] in field


@pytest.mark.parametrize("model_type", sorted(loader._ACTIVATION_UNDECLARED_OK))
def test_every_undeclared_exemption_is_still_true(model_type):
    """An exemption must keep describing upstream, not merely excuse a gap.

    If a family starts declaring one of the fields, the exemption is dead weight
    and the real declaration should be used instead.
    """
    CONFIG_MAPPING = pytest.importorskip("transformers").CONFIG_MAPPING
    try:
        cfg = CONFIG_MAPPING[model_type]()
    except Exception:  # noqa: BLE001 - family absent from this transformers
        pytest.skip(f"{model_type} has no config class in this transformers")
    lm = getattr(cfg, "text_config", None) or cfg
    declared = [f for f in loader._ACTIVATION_FIELDS if getattr(lm, f, None) is not None]
    assert not declared, (
        f"{model_type} now declares {declared} — drop it from "
        f"_ACTIVATION_UNDECLARED_OK and let the declaration be read (e4b#648)"
    )


def _admitted_model_types():
    """Every model_type the loader admits today, by its own rule."""
    return sorted({mt for conv in CONVENTIONS for mt in conv.model_types if _admitted(mt)}
                  | set(loader.SUPPORTED_ARCHITECTURES))


@pytest.mark.parametrize("model_type", _admitted_model_types())
def test_no_admitted_family_is_refused_by_its_default_config(model_type):
    """The refusal must stop at the families the loader admits.

    ``_expert_activation_name`` raises where the old lookup defaulted, so the
    regression to guard against is an ADMITTED family whose config declares none
    of ``_ACTIVATION_FIELDS`` and is not exempt: it loaded yesterday and would
    refuse today. transformers' own default config for the family is the
    cheapest stand-in for a released one. A family absent from this transformers
    is skipped, not passed (kimi_k2 / kimi_k3 ship their config as remote code).
    """
    CONFIG_MAPPING = pytest.importorskip("transformers").CONFIG_MAPPING
    try:
        cfg = CONFIG_MAPPING[model_type]()
    except Exception:  # noqa: BLE001 - family absent from this transformers
        pytest.skip(f"{model_type} has no config class in this transformers")
    lm = getattr(cfg, "text_config", None) or cfg
    name, field = loader._expert_activation_name(lm, model_type)
    assert isinstance(name, str) and name, (model_type, name, field)
    assert field, (model_type, name, field)
