"""Keep MoE activations across a training step instead of recomputing them under gradient checkpointing (opt-in).

Hugging Face checkpoints each decoder layer whole, so its backward first re-runs the layer's forward: attention AND the MoE block.
In e4b's fused step the MoE half of that recompute is most of it -- on a 4-layer Qwen3-30B-A3B slice (RTX A2000, TC1's token rows)
the recomputed MoE forwards were 232 of 917 ms of device time and the step went 1.278 -> 0.975 s with only attention checkpointed,
every trainable gradient ``torch.equal`` (deterministic mode). The price is memory: the MoE block's saved activations live from its
forward to its backward. With grouped-nf4-gemm's ``NF4_QLORA_COMPACT_DELTA=1`` (the padded LoRA delta saves its input, not its padded
block) that is ~55 MB per layer per 380-token micro-batch at fp32 adapters, against 229 MB without it -- so set that too.

``E4B_MOE_KEEP_LAYERS`` = ``all`` or a count ``n`` (the LAST n decoder layers; at the end of the forward every kept layer's
activations coexist whichever n are kept, and the last ones are released first in backward). Unset or 0: unchanged.
``enable_fast_train`` applies it; ``disable_fast_train`` unwinds it. Only layers Hugging Face is checkpointing are touched: in each,
whole-layer checkpointing is switched off and ``self_attn`` alone is checkpointed with the layer's own checkpoint function (the
caller's ``gradient_checkpointing_kwargs``, e.g. ``use_reentrant=False``).
"""
from __future__ import annotations

import functools
import os

MOE_KEEP_STATS = {"layers": 0}


def moe_keep_layers_requested():
    """``None`` (off), ``"all"``, or a positive layer count, from ``E4B_MOE_KEEP_LAYERS``."""
    v = os.environ.get("E4B_MOE_KEEP_LAYERS", "").strip().lower()
    if v in ("", "0", "off", "none"):
        return None
    if v == "all":
        return "all"
    n = int(v)
    if n < 0:
        raise ValueError(f"E4B_MOE_KEEP_LAYERS must be 'all' or a count >= 0, got {v!r}")
    return n or None


def _decoder_layers(model):
    return [m for m in model.modules()
            if hasattr(m, "self_attn") and hasattr(m, "mlp") and hasattr(m, "gradient_checkpointing")]


def keep_moe_activations(model, n="all", verbose: bool = False) -> int:
    """Checkpoint only ``self_attn`` in the last ``n`` (or ``"all"``) checkpointed decoder layers of ``model``. Returns the number
    of layers changed (0 when Hugging Face gradient checkpointing is off: there is no recompute to avoid)."""
    layers = [m for m in _decoder_layers(model) if m.gradient_checkpointing and not hasattr(m, "_e4b_keep_attn_ref")]
    if n != "all":
        layers = layers[-int(n):] if int(n) > 0 else []
    for layer in layers:
        attn = layer.self_attn
        ckpt = getattr(layer, "_gradient_checkpointing_func", None)
        if ckpt is None:
            from torch.utils.checkpoint import checkpoint
            ckpt = functools.partial(checkpoint, use_reentrant=False)
        orig = attn.forward

        def forward(*args, _orig=orig, _ckpt=ckpt, _layer=layer, **kwargs):
            if _layer.training:                  # the condition Hugging Face's own layer checkpointing uses
                return _ckpt(functools.partial(_orig, **kwargs), *args)
            return _orig(*args, **kwargs)
        layer._e4b_keep_attn_ref = orig
        attn.forward = forward
        layer.gradient_checkpointing = False
    MOE_KEEP_STATS["layers"] += len(layers)
    if verbose:
        print(f"[e4b.moe_keep] MoE activations kept (attention-only checkpointing) in {len(layers)} decoder layer(s)")
    return len(layers)


def release_moe_activations(model) -> int:
    """Undo ``keep_moe_activations``: whole-layer checkpointing back on, ``self_attn`` unwrapped."""
    n = 0
    for layer in _decoder_layers(model):
        ref = getattr(layer, "_e4b_keep_attn_ref", None)
        if ref is not None:
            layer.self_attn.forward = ref
            del layer._e4b_keep_attn_ref
            layer.gradient_checkpointing = True
            n += 1
    MOE_KEEP_STATS["layers"] = max(0, MOE_KEEP_STATS["layers"] - n)
    return n
