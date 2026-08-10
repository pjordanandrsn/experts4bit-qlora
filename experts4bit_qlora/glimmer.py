"""Muse Glimmer (Meta) — GGUF k-quant text-tower keymap and load-time transforms.

Glimmer is DENSE (Gemma-3 lineage), not MoE, so it does not touch the expert
lanes at all. This module is the checkpoint-facing seam for the GGUF path: it
maps every llama.cpp tensor name in a released text-tower GGUF onto the
transformers ``MuseGlimmerForConditionalGeneration`` parameter tree, and names
the per-tensor value transform each one needs. The projection/embedding bytes
are decoded through grouped-nf4-gemm's k-quant lane (``kquant_ref``, computed
from the released bytes — never re-quantized); the norms need arithmetic, not
decoding, and getting that arithmetic right is the whole reason this module
exists rather than a dict literal.

Every mapping below was adjudicated against BOTH the real released headers
(meta-models / unsloth GGUFs, parsed) AND the instantiated transformers module
tree (classes + forward source) — not convention. The traps that a
name-only map would get silently wrong:

* **Untied head.** ``config.tie_word_embeddings is False``; the GGUF ships a
  distinct ``output.weight``. It maps to ``lm_head.weight`` and MUST NOT be
  aliased to ``embed_tokens`` (the multimodal-head-fixture blind spot: a loader
  that assumes tying silently ties an untied head and nothing raises).

* **Centered RMSNorm.** The four per-layer norms are
  ``MuseGlimmerTextCenteredRMSNorm``, whose forward is ``x_normed * (1.0 +
  weight)``. llama.cpp bakes the ``+1`` into the GGUF bytes (verified: real
  ``attn_norm`` centers at ~1.0, not ~0.0), so the transformers parameter =
  ``gguf_weight - 1.0``. The FINAL norm (``output_norm`` -> ``language_model.
  norm``) is a plain ``MuseGlimmerRMSNorm(with_scale=True)`` and is used AS-IS
  (verified: real ``output_norm`` centers at ~0.0). Applying ``-1`` to the
  wrong subset, or to all of them, is a silent accuracy bug.

* **QK-norm is a scalar in disguise.** transformers builds ``self_attn.qk_norm``
  as ``MuseGlimmerRMSNorm(with_scale=False)`` (no learnable weight) and does
  ``q = qk_norm(q) * config.qk_scale_factor``, ``k = qk_norm(k)``. The GGUF ships
  ``attn_q_norm``/``attn_k_norm`` as UNIFORM per-head-dim vectors — real values
  3.87 and 1.0 — i.e. the q-scale encoded as a vector and a no-op k-scale. Those
  equal ``config.qk_scale_factor`` (3.87) and 1.0 exactly, so both tensors are
  DROPPED, but only after an at-load assertion that they are uniform and match:
  a provider whose file diverges (a genuinely learned qk-norm) must fail loudly
  here, not lose weights silently.

Vision is a SEPARATE ``mmproj-*.gguf`` (50-layer ViT + adapter + projection) and
is out of scope for this text-tower map; the LLM serves text-only until the
vision keymap lands (follow-on). ``final_logit_softcapping = 20.0`` and the
sliding/full + NoPE attention schedule are transformers' concern at forward
time, not the loader's — they ride in the config this module never edits.
"""
from __future__ import annotations

import re

GLIMMER_MODEL_TYPES = {"muse_glimmer"}

# transformers text-tower prefixes (the multimodal config nests the LM here).
_LM = "model.language_model."

# Non-layer, whole-model tensors: gguf name -> (param, transform).
_GLOBAL = {
    "token_embd.weight": (f"{_LM}embed_tokens.weight", "dequant"),
    "output.weight": ("lm_head.weight", "dequant"),           # UNTIED — never alias to embed
    "output_norm.weight": (f"{_LM}norm.weight", "asis"),      # with_scale=True final norm
}

# Per-layer: gguf suffix (under blk.N.) -> (param suffix under layers.N., transform).
#   dequant : k-quant/f32 projection or embedding, decode via kquant_ref
#   sub1    : TextCenteredRMSNorm weight, transformers = gguf - 1.0
#   drop    : redundant with a config scalar; asserted uniform then discarded
_PER_LAYER = {
    "attn_norm.weight":            ("input_layernorm.weight", "sub1"),
    "post_attention_norm.weight":  ("post_attention_layernorm.weight", "sub1"),
    "ffn_norm.weight":             ("pre_feedforward_layernorm.weight", "sub1"),
    "post_ffw_norm.weight":        ("post_feedforward_layernorm.weight", "sub1"),
    "attn_q.weight":               ("self_attn.q_proj.weight", "dequant"),
    "attn_k.weight":               ("self_attn.k_proj.weight", "dequant"),
    "attn_v.weight":               ("self_attn.v_proj.weight", "dequant"),
    "attn_output.weight":          ("self_attn.o_proj.weight", "dequant"),
    "attn_gate.weight":            ("self_attn.gate_proj.weight", "dequant"),
    "ffn_gate.weight":             ("mlp.gate_proj.weight", "dequant"),
    "ffn_up.weight":               ("mlp.up_proj.weight", "dequant"),
    "ffn_down.weight":             ("mlp.down_proj.weight", "dequant"),
    "attn_q_norm.weight":          (None, "drop_q"),   # assert uniform == qk_scale_factor
    "attn_k_norm.weight":          (None, "drop_k"),   # assert uniform == 1.0
}

_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


class GlimmerKeymapError(ValueError):
    """A released GGUF tensor did not map, or a dropped tensor was not the
    scalar we proved it to be. Never downgrade to a warning — an unmapped
    tensor is a dropped weight, the failure mode with no symptom."""


def map_gguf_key(gguf_name: str) -> tuple[str | None, str]:
    """Map one GGUF text-tower tensor name to (transformers_param | None, transform).

    Returns ``(None, "drop_*")`` for the qk-norm scalars. Raises
    GlimmerKeymapError for any name not in the known text-tower surface, so a
    format change is caught at load, not absorbed.
    """
    if gguf_name in _GLOBAL:
        return _GLOBAL[gguf_name]
    m = _BLK.match(gguf_name)
    if m:
        layer, suffix = m.group(1), m.group(2)
        if suffix in _PER_LAYER:
            param_suffix, transform = _PER_LAYER[suffix]
            if param_suffix is None:
                return None, transform
            return f"{_LM}layers.{layer}.{param_suffix}", transform
    raise GlimmerKeymapError(f"unmapped Glimmer GGUF tensor: {gguf_name!r}")


def transform_weight(tensor, transform: str, *, qk_scale_factor: float,
                     name: str = ""):
    """Apply the load-time value transform to a decoded fp32 tensor.

    ``tensor`` is already dequantized (kquant_ref). Returns the tensor to place
    into the transformers parameter, or ``None`` for a validated drop. The drop
    paths assert the scalar identity that licensed the drop — divergence raises.
    """
    if transform == "dequant" or transform == "asis":
        return tensor
    if transform == "sub1":
        return tensor - 1.0
    if transform in ("drop_q", "drop_k"):
        want = qk_scale_factor if transform == "drop_q" else 1.0
        flat = tensor.flatten().float()
        # Uniform to fp16 storage precision (these ship F32 but originate as a
        # broadcast scalar). A learned qk-norm would fail this and must.
        if not (flat.max() - flat.min() <= 1e-3 and abs(flat[0].item() - want) <= 2e-2):
            raise GlimmerKeymapError(
                f"{name}: expected uniform {want} (absorbed into qk_scale_factor), "
                f"got range [{flat.min().item():.4f}, {flat.max().item():.4f}] — "
                f"this file has a learned qk-norm the transformers path drops")
        return None
    raise GlimmerKeymapError(f"unknown transform {transform!r} for {name!r}")


def expected_param_names(num_layers: int) -> set[str]:
    """The exact transformers text-tower parameter set a full GGUF populates —
    every mapped target, for reverse-arm coverage against the built model."""
    names = {p for p, _ in _GLOBAL.values()}
    for n in range(num_layers):
        for param_suffix, _ in _PER_LAYER.values():
            if param_suffix is not None:
                names.add(f"{_LM}layers.{n}.{param_suffix}")
    return names
