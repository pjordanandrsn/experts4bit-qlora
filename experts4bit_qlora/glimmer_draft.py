"""Muse Glimmer DFlash drafter (`muse_glimmer_assistant`) — GGUF keymap.

The drafter is the speculative-decoding companion to Muse-Glimmer-30B: it
proposes a block of tokens that the target verifies in one pass. It is a
SEPARATE architecture, not a small Glimmer, and this module exists because
reusing :mod:`experts4bit_qlora.glimmer`'s map would be silently wrong in two
different ways.

Shape (adjudicated against the released `dflash-kquant.gguf` header AND the
instantiated `MuseGlimmerAssistantModel` on ``meta``): 5 layers, hidden 6656,
32 heads / 8 KV, head_dim 128, `block_size` 16. The GGUF holds 58 tensors and
the module declares exactly 58 parameters — they reconcile one-to-one.

What differs from the 30B, and why a shared map would corrupt weights:

* **`attn_q_norm` / `attn_k_norm` are REAL here.** In the 30B those GGUF
  tensors are uniform vectors equal to `config.qk_scale_factor` and 1.0,
  absorbed by a parameter-free norm, so the Glimmer map DROPS them after
  asserting that identity. The drafter has genuine learned
  ``self_attn.{q,k}_norm.weight`` parameters. Dropping them would discard two
  real weights per layer with nothing raising.

* **The norms are NOT centered.** ``MuseGlimmerAssistantRMSNorm.forward`` is
  ``self.weight * normed`` — plain. The 30B's four per-layer norms are
  `TextCentered` (`x*(1+w)`) with the `+1` baked into the GGUF bytes, so they
  need `gguf − 1.0`. Applying that subtraction here would shift every norm by
  one and quietly wreck the drafter's outputs.

* **No embeddings, no head.** The drafter shares the target's embedding space,
  so the GGUF ships no `token_embd`/`output`. It carries an encoder instead:
  `fc.weight` is ``[hidden, 5*hidden]``, projecting five concatenated
  hidden-state features down to one, plus its own `enc.output_norm`.

Only the map lives here. Wiring a proposal/verification loop is a separate
change; this is the piece that has to be exactly right first, because a
drafter loaded with silently-wrong weights degrades acceptance rate rather
than failing, which is the hardest kind of bug to notice.
"""
from __future__ import annotations

import re

GLIMMER_DRAFT_MODEL_TYPES = {"muse_glimmer_assistant"}

# Whole-model tensors: gguf name -> module parameter.
_GLOBAL = {
    "fc.weight": "encoder.fc.weight",
    "enc.output_norm.weight": "encoder.output_norm_enc.weight",
    "output_norm.weight": "norm.weight",
}

# Per-layer: gguf suffix under blk.N. -> parameter suffix under layers.N.
# Every one is a plain passthrough: no centering, no drops (see module docstring).
_PER_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",   # REAL here, dropped in the 30B
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


class GlimmerDraftKeymapError(ValueError):
    """A drafter GGUF tensor did not map. Never downgrade to a warning: a
    dropped drafter weight lowers the acceptance rate instead of failing,
    which is harder to notice than a crash."""


def map_draft_key(gguf_name: str) -> str:
    """Map one drafter GGUF tensor name to its module parameter name."""
    if gguf_name in _GLOBAL:
        return _GLOBAL[gguf_name]
    m = _BLK.match(gguf_name)
    if m:
        layer, suffix = m.group(1), m.group(2)
        if suffix in _PER_LAYER:
            return f"layers.{layer}.{_PER_LAYER[suffix]}"
    raise GlimmerDraftKeymapError(f"unmapped drafter GGUF tensor: {gguf_name!r}")


def expected_param_names(num_layers: int) -> set[str]:
    """Every parameter a full drafter GGUF populates — for reverse-arm coverage."""
    names = set(_GLOBAL.values())
    for i in range(num_layers):
        for suffix in _PER_LAYER.values():
            names.add(f"layers.{i}.{suffix}")
    return names
