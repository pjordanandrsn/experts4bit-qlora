"""GLM-5 (Zhipu, ``glm_moe_dsa``) — checkpoint key map and expert fusion.

GLM-5 is DeepSeek-V3 lineage and reuses most of the ``deepseek_v4`` lane:
MLA attention (``q_a``/``q_b``, ``kv_a_proj_with_mqa``, ``kv_b_proj``), per-expert
MoE on disk, a shared expert beside the routed ones, and the first
``first_k_dense_replace`` layers dense. What is genuinely new is **DSA**
(sparse attention): every layer carries a lightning-indexer
(``self_attn.indexer.{wq_b,wk,k_norm,weights_proj}``) that selects
``index_topk`` keys. The indexer is small, dense, and per-layer — it maps
straight through; nothing about it needs a transform.

As with Glimmer, every mapping here was adjudicated against BOTH the real
released ``model.safetensors.index.json`` AND the instantiated transformers
tree (78 layers, on ``meta``) — not convention. Three things a plausible map
gets silently wrong:

* **The checkpoint has one MORE layer than the model.** Layers run 0..78 on
  disk (79) while ``num_hidden_layers`` is 78 (0..77). Layer 78 is the
  multi-token-prediction head — it carries the DeepSeek-V3 MTP signature
  (``eh_proj``, ``enorm``, ``hnorm``) plus a full 256-expert set, 791 keys.
  transformers does not build it. It is SKIPPED explicitly here (and is the
  natural donor for a future speculative-decoding lane); a map that let it
  through would either raise deep in placement or, worse, be silently absorbed.

* **Fused vs per-expert.** The module tree holds ONE stacked tensor per layer
  (``mlp.experts.gate_up_proj`` ``[E, 2*inter, hidden]``,
  ``mlp.experts.down_proj`` ``[E, hidden, inter]``) while the checkpoint ships
  ``mlp.experts.{e}.{gate,up,down}_proj.weight`` separately. Loading therefore
  FUSES: stack over e, and concatenate gate/up as BLOCKS
  (``cat([gate, up], dim=0)``) — because the forward is
  ``linear(x, gate_up_proj[e]).chunk(2, dim=-1)``, i.e. rows ``[0:inter]`` are
  the gate and ``[inter:2*inter]`` the up. Interleaving them (the GPT-OSS
  convention) would compute a wrong activation with every shape agreeing.

* **Only layers >= first_k_dense_replace are MoE.** Layers 0..2 have a plain
  dense ``mlp.{gate,up,down}_proj``; the same suffix means different things
  either side of that boundary, so the map is layer-index aware.

Shapes (verified against the built tree): hidden 6144, moe_intermediate 2048,
256 routed + 1 shared expert, 8 active. The shared expert stays per-projection
(it is a single MLP, not a stack) and maps through unchanged.
"""
from __future__ import annotations

import re

GLM5_MODEL_TYPES = {"glm_moe_dsa"}

# Non-layer tensors: identical spelling on both sides.
_GLOBAL = {
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
}

# Per-layer keys that map 1:1 (checkpoint spelling == module spelling).
_PASSTHROUGH_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    # DSA lightning indexer — the new surface, but a plain dense passthrough.
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
    # MoE router + shared expert (single MLP, not a stack). The router's
    # e_score_correction_bias is the DeepSeek-V3 aux-loss-free load-balancing
    # bias; transformers holds it as a BUFFER, not a Parameter (so a coverage
    # check that walks named_parameters() alone cannot see it — this map was
    # missing it until the reverse arm, which walks state_dict, caught it).
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    # Dense-layer MLP (layers < first_k_dense_replace only; validated below).
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}

# MTP head signature — present only on the extra trailing checkpoint layer.
_MTP_SUFFIX_MARKERS = ("eh_proj.weight", "enorm.weight", "hnorm.weight")

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class Glm5KeymapError(ValueError):
    """A checkpoint key did not map. Never downgrade to a warning — an
    unmapped tensor is a dropped weight, the failure mode with no symptom."""


def classify_key(key: str, *, num_hidden_layers: int, first_k_dense_replace: int):
    """Classify one checkpoint key.

    Returns a dict describing what to do with it:

      {"kind": "passthrough", "param": <module param name>}
      {"kind": "expert", "layer": int, "expert": int, "proj": "gate"|"up"|"down"}
      {"kind": "skip_mtp", "layer": int}      — the trailing MTP head

    Raises Glm5KeymapError for anything unrecognized.
    """
    if key in _GLOBAL:
        return {"kind": "passthrough", "param": key}
    m = _LAYER.match(key)
    if not m:
        raise Glm5KeymapError(f"unmapped GLM-5 key (not a layer): {key!r}")
    layer, suffix = int(m.group(1)), m.group(2)

    # Everything on a layer the model does not build is the MTP head.
    if layer >= num_hidden_layers:
        return {"kind": "skip_mtp", "layer": layer}
    if any(suffix.endswith(s) for s in _MTP_SUFFIX_MARKERS):
        # MTP markers on a BUILT layer would mean our layer math is wrong.
        raise Glm5KeymapError(
            f"MTP marker {suffix!r} on built layer {layer} — layer-count "
            f"assumption (num_hidden_layers={num_hidden_layers}) is wrong")

    em = _EXPERT.match(suffix)
    if em:
        if layer < first_k_dense_replace:
            raise Glm5KeymapError(
                f"expert tensor on dense layer {layer} "
                f"(first_k_dense_replace={first_k_dense_replace}): {key!r}")
        return {"kind": "expert", "layer": layer, "expert": int(em.group(1)),
                "proj": em.group(2).replace("_proj", "")}

    if suffix in _PASSTHROUGH_SUFFIXES:
        is_dense_mlp = suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight",
                                  "mlp.down_proj.weight"}
        if is_dense_mlp and layer >= first_k_dense_replace:
            raise Glm5KeymapError(
                f"dense-MLP tensor on MoE layer {layer}: {key!r}")
        moe_only = (suffix.startswith("mlp.gate.")
                    or suffix.startswith("mlp.shared_experts."))
        if moe_only and layer < first_k_dense_replace:
            raise Glm5KeymapError(f"MoE tensor on dense layer {layer}: {key!r}")
        return {"kind": "passthrough", "param": key}

    raise Glm5KeymapError(f"unmapped GLM-5 key: {key!r}")


def fuse_experts(gate, up, down):
    """Per-expert lists -> the two stacked tensors the module declares.

    ``gate``/``up``/``down`` are lists indexed by expert id:
      gate[e], up[e] : [inter, hidden]      down[e] : [hidden, inter]

    Returns (gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter]).
    Gate and up are concatenated as BLOCKS, not interleaved: the forward does
    ``linear(x, gate_up_proj[e]).chunk(2, -1)``, so rows [0:inter] must be the
    gate. Missing experts raise rather than silently shrinking the stack.
    """
    import torch

    n = len(gate)
    if not (len(up) == len(down) == n) or n == 0:
        raise Glm5KeymapError(
            f"expert count mismatch: gate={len(gate)} up={len(up)} down={len(down)}")
    if any(t is None for t in (*gate, *up, *down)):
        missing = [i for i, t in enumerate(gate) if t is None] or \
                  [i for i, t in enumerate(up) if t is None] or \
                  [i for i, t in enumerate(down) if t is None]
        raise Glm5KeymapError(f"missing expert tensors at indices {missing[:5]}")
    gate_up = torch.stack([torch.cat([gate[e], up[e]], dim=0) for e in range(n)])
    down_s = torch.stack(list(down))
    return gate_up, down_s


def expected_param_names(num_hidden_layers: int, first_k_dense_replace: int) -> set[str]:
    """Every module param a full checkpoint populates — for reverse-arm coverage."""
    names = set(_GLOBAL)
    for i in range(num_hidden_layers):
        moe = i >= first_k_dense_replace
        for suffix in _PASSTHROUGH_SUFFIXES:
            is_dense_mlp = suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight",
                                      "mlp.down_proj.weight"}
            moe_only = suffix.startswith("mlp.gate.") or suffix.startswith("mlp.shared_experts.")
            if is_dense_mlp and moe:
                continue
            if moe_only and not moe:
                continue
            names.add(f"model.layers.{i}.{suffix}")
        if moe:
            names.add(f"model.layers.{i}.mlp.experts.gate_up_proj")
            names.add(f"model.layers.{i}.mlp.experts.down_proj")
    return names
