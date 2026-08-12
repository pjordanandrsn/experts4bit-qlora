"""Mixtral-convention MoE checkpoints — key map and expert fusion.

Covers the family transformers itself groups under the ``"mixtral"`` conversion:
``mixtral``, ``minimax``, ``minimax_m2``. They share one on-disk shape, so they
share one map — which is the point. There are far fewer MoE *conventions* than
MoE models, and mapping a convention lights up every model that follows it.

The divergence this module exists to bridge: released checkpoints ship
**per-expert** tensors under a ``block_sparse_moe`` block::

    model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight   [inter, hidden]
    model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight   [inter, hidden]
    model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight   [hidden, inter]
    model.layers.{L}.block_sparse_moe.gate.weight             [E, hidden]

while transformers >= 5 builds a **fused** tree::

    model.layers.{L}.mlp.experts.gate_up_proj   [E, 2*inter, hidden]
    model.layers.{L}.mlp.experts.down_proj      [E, hidden, inter]
    model.layers.{L}.mlp.gate.weight            [E, hidden]

``from_pretrained`` applies that conversion itself; this loader reads shards
directly, so it must apply it here.

**w1 is the gate and it comes FIRST — and that is not a guess.** w1 and w3 have
identical shapes, so shape inference cannot disambiguate them; a coin-flip here
yields ``up * act(gate)`` swapped, which computes a wrong activation with every
shape agreeing and nothing raising. Two independent sources pin it:

* ``MixtralExperts.forward`` does
  ``gate, up = linear(x, gate_up_proj[e]).chunk(2, -1)`` then
  ``act_fn(gate) * up`` — so rows ``[0:inter]`` must be the gate;
* transformers' own converter (``conversion_mapping.py``, key ``"mixtral"``)
  lists sources ``[w1, w3]`` in that order, ``MergeModulelist(dim=0)`` to stack
  each across experts, then ``Concatenate(dim=1)`` — w1's rows land first.

Note this is *block* concatenation (gate block, then up block), NOT the
interleave GPT-OSS uses. See :mod:`experts4bit_qlora.arch.gptoss` for that one.
"""
from __future__ import annotations

import re

# model_type -> True. All share the on-disk convention below; transformers
# aliases minimax/minimax_m2 onto "mixtral" in its own conversion table.
MIXTRAL_CONVENTION_MODEL_TYPES = {"mixtral", "minimax", "minimax_m2"}

_GLOBAL = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

# Per-layer non-expert keys: same spelling either side, except the MoE block
# rename handled separately.
_PASSTHROUGH_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
}

# The router: block_sparse_moe.gate -> mlp.gate
_ROUTER_SUFFIX = "block_sparse_moe.gate.weight"
_ROUTER_TARGET = "mlp.gate.weight"

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT = re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")

# w1 = gate, w3 = up, w2 = down. Adjudicated against the forward AND
# transformers' converter (see module docstring) — never assumed from shapes,
# because w1 and w3 are shape-identical.
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}


class MixtralKeymapError(ValueError):
    """A checkpoint key did not map. Never downgrade to a warning — an
    unmapped tensor is a dropped weight, the failure mode with no symptom."""


def classify_key(key: str, *, num_hidden_layers: int):
    """Classify one Mixtral-convention checkpoint key.

    Returns one of::

        {"kind": "passthrough", "param": <module param name>}
        {"kind": "expert", "layer": int, "expert": int, "role": "gate"|"up"|"down"}

    Raises MixtralKeymapError for anything unrecognized.
    """
    if key in _GLOBAL:
        return {"kind": "passthrough", "param": key}
    m = _LAYER.match(key)
    if not m:
        raise MixtralKeymapError(f"unmapped Mixtral-convention key: {key!r}")
    layer, suffix = int(m.group(1)), m.group(2)
    if layer >= num_hidden_layers:
        raise MixtralKeymapError(
            f"key on layer {layer} but the model builds {num_hidden_layers}: {key!r}")
    em = _EXPERT.match(suffix)
    if em:
        return {"kind": "expert", "layer": layer, "expert": int(em.group(1)),
                "role": _PROJ_ROLE[em.group(2)]}
    if suffix == _ROUTER_SUFFIX:
        return {"kind": "passthrough",
                "param": f"model.layers.{layer}.{_ROUTER_TARGET}"}
    if suffix in _PASSTHROUGH_SUFFIXES:
        return {"kind": "passthrough", "param": key}
    raise MixtralKeymapError(f"unmapped Mixtral-convention key: {key!r}")


def fuse_experts(gate, up, down):
    """Per-expert lists -> the stacked tensors the fused module declares.

    ``gate[e]``/``up[e]`` are ``[inter, hidden]``; ``down[e]`` is
    ``[hidden, inter]``. Returns
    ``(gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter])``.

    Gate and up are concatenated as BLOCKS with the GATE FIRST — the forward
    chunks the linear's output and applies the activation to the first half.
    Interleaving, or ordering up-then-gate, computes a wrong activation with
    every shape agreeing. Missing experts raise rather than silently shrinking
    the stack.
    """
    import torch

    n = len(gate)
    if n == 0 or not (len(up) == len(down) == n):
        raise MixtralKeymapError(
            f"expert count mismatch: gate={len(gate)} up={len(up)} down={len(down)}")
    if any(t is None for t in (*gate, *up, *down)):
        raise MixtralKeymapError("missing expert tensors — refusing a partial stack")
    gate_up = torch.stack([torch.cat([gate[e], up[e]], dim=0) for e in range(n)])
    return gate_up, torch.stack(list(down))


def expected_param_names(num_hidden_layers: int) -> set[str]:
    """Every module parameter a full checkpoint populates — reverse-arm target."""
    names = set(_GLOBAL)
    for i in range(num_hidden_layers):
        for suffix in _PASSTHROUGH_SUFFIXES:
            names.add(f"model.layers.{i}.{suffix}")
        names.add(f"model.layers.{i}.{_ROUTER_TARGET}")
        names.add(f"model.layers.{i}.mlp.experts.gate_up_proj")
        names.add(f"model.layers.{i}.mlp.experts.down_proj")
    return names
