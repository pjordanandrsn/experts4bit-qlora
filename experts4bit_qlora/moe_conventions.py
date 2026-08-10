"""MoE storage *conventions* — the small set of on-disk shapes that the large
set of MoE models actually use.

There are far fewer MoE conventions than MoE models. transformers encodes this
directly: its ``conversion_mapping`` defines a handful of converter specs and
aliases dozens of ``model_type``s onto them. Mapping a convention therefore
lights up every model that follows it, and — more importantly — means the
error-prone part (expert fusion orientation) is adjudicated ONCE per
convention instead of once per model.

Two conventions are implemented here. Both fuse per-expert checkpoint tensors
into the stacked tensors transformers >= 5 builds, and both put the **gate
block first**; they differ only in the on-disk spelling:

All of them vary on only two axes — the CONTAINER the experts live under, and
how the projections are SPELLED:

===============  =================  ======================  ==================
convention       container          projection spelling     model_types
===============  =================  ======================  ==================
``QWEN2_MOE``    ``mlp``            ``{gate,up,down}_proj``  22
``MIXTRAL``      ``block_sparse_moe``  ``w1``/``w3``/``w2``  3
``PHIMOE``       ``block_sparse_moe``  ``w1``/``w3``/``w2``  1
``JAMBA``        ``feed_forward``   ``{gate,up,down}_proj``  1
``LFM2_MOE``     ``feed_forward``   ``w1``/``w3``/``w2``     1
===============  =================  ======================  ==================

``QWEN2_MOE`` alone covers deepseek_v2/v3/v32, every GLM MoE (glm4_moe,
glm4_moe_lite, glm4v_moe, glm_moe_dsa), olmoe, qwen3_moe, qwen3_next, dots1,
ernie4_5_moe, hunyuan_v1_moe, longcat_flash, exaone_moe, cohere2_moe,
flex_olmo, afmoe, mellum and solar_open.

``w1``=gate, ``w3``=up, ``w2``=down in every ``w``-spelled family — pinned to
upstream, never inferred.

Deliberately NOT here: ``granitemoe`` (and its granitemoehybrid /
granitemoeshared aliases) ships ALREADY-FUSED on disk and needs pure renames,
not a fusion — that lives in :mod:`experts4bit_qlora.loader`'s
``LEGACY_KEY_RENAMES``. A fusion applied to it would be wrong.

**Why the orientation is the whole point.** In both conventions the gate and up
tensors are SHAPE-IDENTICAL, so no amount of shape inspection can tell them
apart. Ordering them wrong yields ``act(up) * gate`` — a wrong activation with
every shape agreeing, nothing raising, and only a quality regression to show
for it. The orientation below is taken from transformers' own converter specs
(source order + ``Concatenate(dim=1)``) and cross-checked against each family's
expert ``forward``; :mod:`tests.test_moe_conventions` asserts both so an
upstream change fails loudly instead of silently swapping halves.

This module owns the EXPERT surface only. Each model's non-expert surface
(attention flavour, shared experts, router bias, dense-layer prefix, extra
heads) still needs its own map — see :mod:`experts4bit_qlora.glm5` for a worked
example that layers a model's quirks on top of the QWEN2_MOE expert
convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MoEConvention:
    """How one family stores per-expert weights on disk.

    ``expert_re`` captures (expert_index, projection_token) from the part of a
    checkpoint key that follows ``model.layers.{L}.``. ``roles`` maps the
    projection token to its role. ``fused_prefix`` is where the stacked
    tensors live in the module tree.
    """

    name: str
    expert_re: re.Pattern
    roles: dict
    fused_prefix: str
    model_types: frozenset = field(default_factory=frozenset)

    def match(self, layer_suffix: str):
        """-> (expert_index, role) or None."""
        m = self.expert_re.match(layer_suffix)
        if not m:
            return None
        token = m.group(2)
        if token not in self.roles:
            return None
        return int(m.group(1)), self.roles[token]

    def fused_names(self, layer: int) -> tuple[str, str]:
        base = f"model.layers.{layer}.{self.fused_prefix}"
        return f"{base}.gate_up_proj", f"{base}.down_proj"


QWEN2_MOE = MoEConvention(
    name="qwen2_moe",
    expert_re=re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"),
    roles={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    fused_prefix="mlp.experts",
    model_types=frozenset({
        "qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_omni_moe",
        "qwen3_omni_moe_thinker", "olmoe", "flex_olmo", "afmoe", "cohere2_moe",
        "deepseek_v2", "deepseek_v3", "deepseek_v32", "dots1", "ernie4_5_moe",
        "exaone_moe", "glm4_moe", "glm4_moe_lite", "glm4v_moe", "glm_moe_dsa",
        "hunyuan_v1_moe", "longcat_flash", "mellum", "solar_open",
    }),
)

MIXTRAL = MoEConvention(
    name="mixtral",
    # w1 = gate, w3 = up, w2 = down — pinned by transformers' converter source
    # order and MixtralExperts.forward, never inferred (w1/w3 are same-shaped).
    expert_re=re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"),
    roles={"w1": "gate", "w3": "up", "w2": "down"},
    fused_prefix="mlp.experts",
    model_types=frozenset({"mixtral", "minimax", "minimax_m2"}),
)

PHIMOE = MoEConvention(
    name="phimoe",
    expert_re=re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"),
    roles={"w1": "gate", "w3": "up", "w2": "down"},
    fused_prefix="mlp.experts",
    model_types=frozenset({"phimoe"}),
)

JAMBA = MoEConvention(
    name="jamba",
    expert_re=re.compile(
        r"^feed_forward\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"),
    roles={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    fused_prefix="feed_forward.experts",
    model_types=frozenset({"jamba"}),
)

LFM2_MOE = MoEConvention(
    name="lfm2_moe",
    expert_re=re.compile(r"^feed_forward\.experts\.(\d+)\.(w1|w2|w3)\.weight$"),
    roles={"w1": "gate", "w3": "up", "w2": "down"},
    fused_prefix="feed_forward.experts",
    model_types=frozenset({"lfm2_moe"}),
)

CONVENTIONS = (QWEN2_MOE, MIXTRAL, PHIMOE, JAMBA, LFM2_MOE)
_BY_MODEL_TYPE = {mt: c for c in CONVENTIONS for mt in c.model_types}


class MoEConventionError(ValueError):
    """An expert stack could not be built. Never downgrade to a warning: a
    partial or mis-ordered stack computes wrong numbers without raising."""


def convention_for(model_type: str) -> MoEConvention:
    """The storage convention for a model_type, or raise.

    Raising on an unknown type is deliberate. Guessing a convention is exactly
    the failure this module exists to prevent — a wrong guess still loads.
    """
    try:
        return _BY_MODEL_TYPE[model_type]
    except KeyError:
        raise MoEConventionError(
            f"no adjudicated MoE convention for model_type {model_type!r}. Add "
            f"one only after reading its converter spec in transformers' "
            f"conversion_mapping AND its expert forward — the gate/up "
            f"orientation cannot be inferred from shapes."
        ) from None


def fuse_experts(gate: list, up: list, down: list):
    """Per-expert lists -> ``(gate_up_proj [E, 2*inter, hidden],
    down_proj [E, hidden, inter])``.

    ``gate[e]``/``up[e]`` are ``[inter, hidden]``; ``down[e]`` is
    ``[hidden, inter]``. Equivalent to transformers'
    ``MergeModulelist(dim=0)`` + ``Concatenate(dim=1)``: stack each projection
    across experts, then join gate and up along the intermediate axis with the
    **gate first**. Missing experts raise rather than silently shrinking the
    stack — a short stack would route some tokens to the wrong expert.
    """
    import torch

    n = len(gate)
    if n == 0 or not (len(up) == len(down) == n):
        raise MoEConventionError(
            f"expert count mismatch: gate={len(gate)} up={len(up)} down={len(down)}")
    missing = [i for i, t in enumerate(gate) if t is None]
    missing += [i for i, t in enumerate(up) if t is None]
    missing += [i for i, t in enumerate(down) if t is None]
    if missing:
        raise MoEConventionError(
            f"missing expert tensors at indices {sorted(set(missing))[:5]} — "
            f"refusing to build a partial expert stack")
    gate_up = torch.stack([torch.cat([gate[e], up[e]], dim=0) for e in range(n)])
    return gate_up, torch.stack(list(down))
