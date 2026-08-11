"""A.X-K1 (SKT) key map — a DeepSeek-V3 MoE with one architectural delta and a
hybrid dense/MoE layer stack that the generic convention system cannot express
on its own.

Two things need per-layer knowledge the convention machinery does not have:

* **post_mlp_layernorm is layer-conditional.** transformers' AXK1MoE carries a
  ``post_mlp_layernorm`` on its output; the dense AXK1MLP (the first
  ``first_k_dense_replace`` layers) does not. The released checkpoint stores the
  norm at the DECODER-LAYER level (``layers.N.post_mlp_layernorm.weight``) for
  every layer it has one, and transformers renames it to
  ``layers.N.mlp.post_mlp_layernorm.weight`` at load — which only exists on MoE
  layers. On a dense layer that renamed key has no home, so transformers drops
  it as an unexpected key. A single global rename cannot do "rename on MoE, drop
  on dense"; this function does, keyed on ``first_k_dense_replace``.
* **e_score_correction_bias is an unshipped buffer.** The DeepSeek-V3-lineage
  router carries an ``mlp.gate.e_score_correction_bias`` buffer that A.X-K1's
  checkpoint does NOT ship (it is a zero-initialised correction). It must be
  excluded from the "every parameter must be supplied" check, not treated as a
  missing weight.

The EXPERT surface is plain native passthrough: skt/A.X-K1 ships its experts
PRE-FUSED (``mlp.experts.gate_up_proj`` [E, 2*inter, hidden] matching the tree
exactly, verified by shape — no transpose), so a never-per-expert convention
covers it.
"""
from __future__ import annotations

import re

from .moe_conventions import AXK1 as AXK1_CONVENTION

__all__ = ["AXK1_CONVENTION", "AXK1_IGNORE_PARAM_PATTERNS", "rewrite_axk1_keys"]

#: The router buffer the checkpoint never ships — excluded from the
#: everything-must-be-supplied check rather than reported as missing.
AXK1_IGNORE_PARAM_PATTERNS = (r"mlp\.gate\.e_score_correction_bias$",)

_POST_MLP = re.compile(r"(.*layers\.(\d+)\.)post_mlp_layernorm\.(.*)$")


def rewrite_axk1_keys(checkpoint_keys, first_k_dense_replace):
    """``(rewritten_keys, dropped_keys)``.

    post_mlp_layernorm at the layer level -> mlp.post_mlp_layernorm on MoE
    layers (``layer_idx >= first_k_dense_replace``); dropped on the dense layers
    below that, which have no such module — matching transformers' own
    unexpected-key handling. Every other key passes through unchanged.
    """
    out, dropped = [], []
    for k in checkpoint_keys:
        m = _POST_MLP.match(k)
        if m:
            prefix, layer, tail = m.group(1), int(m.group(2)), m.group(3)
            if layer < first_k_dense_replace:
                dropped.append(k)                       # dense layer: no home
                continue
            k = f"{prefix}mlp.post_mlp_layernorm.{tail}"
        out.append(k)
    return out, dropped
