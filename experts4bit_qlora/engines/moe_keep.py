"""Keep MoE activations across a training step instead of recomputing them under gradient checkpointing (opt-in).

Hugging Face checkpoints each decoder layer whole, so its backward first re-runs the layer's forward: attention AND the MoE block.
In e4b's fused step the MoE half of that recompute is most of it -- on a 4-layer Qwen3-30B-A3B slice (RTX A2000, TC1's token rows)
the recomputed MoE forwards were 232 of 917 ms of device time and the step went 1.278 -> 0.975 s with only attention checkpointed,
every trainable gradient ``torch.equal`` (deterministic mode). The price is memory: the MoE block's saved activations live from its
forward to its backward. With grouped-nf4-gemm's ``NF4_QLORA_COMPACT_DELTA=1`` (the padded LoRA delta saves its input, not its padded
block) that is ~55 MB per layer per 380-token micro-batch at fp32 adapters, against 229 MB without it -- so set that too.

``E4B_MOE_KEEP_LAYERS`` = ``all`` or a count ``n`` (the LAST n decoder layers; at the end of the forward every kept layer's
activations coexist whichever n are kept, and the last ones are released first in backward). Unset or 0: unchanged.
``enable_fast_train`` applies it; ``disable_fast_train`` unwinds it. Only checkpointed layers that hold an ``ExpertsLoRA`` are
touched: in each, whole-layer checkpointing is switched off and every OTHER weighted child (attention, a linear-attention or
state-space mixer, a dense MLP) is checkpointed on its own with the layer's own checkpoint function (the caller's
``gradient_checkpointing_kwargs``, e.g. ``use_reentrant=False``). Matched on structure -- the child holding the experts --
never on child names, so a family whose MoE block is ``block_sparse_moe`` / ``feed_forward`` / ``mixer`` gets it too. On a
Qwen3-MoE layer the result is exactly the earlier ``self_attn``-only form.
"""
from __future__ import annotations

import functools
import os

MOE_KEEP_STATS = {"layers": 0, "checkpointed_children": 0}


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


def _holds_experts(module) -> bool:
    """An expert stack lives somewhere below ``module``: an ``ExpertsLoRA`` (e4b's wrapper), or a Hugging Face fused stack --
    a module owning at least two rank-3 parameters with one shared leading ``num_experts`` dimension (``gate_up_proj`` and
    ``down_proj``). One rank-3 tensor is not enough: a depthwise conv (LFM2's short conv, a DeltaNet / Mamba ``conv1d``) owns
    exactly one."""
    from ..lora import ExpertsLoRA

    def _hf_stack(m):
        lead = [p.shape[0] for p in m.parameters(recurse=False) if p.dim() == 3]
        return len(lead) >= 2 and len(set(lead)) == 1 and lead[0] > 1
    return any(isinstance(m, ExpertsLoRA) or _hf_stack(m) for m in module.modules())


def _is_weighted(module) -> bool:
    """A child that does real work: it owns (directly or below) a parameter of rank >= 2. Norms (1-D weights) and
    parameter-free glue are left inside the layer's own forward -- recomputing them buys nothing."""
    return any(p.dim() >= 2 for p in module.parameters())


def _decoder_layers(model):
    """Hugging Face checkpointing units (anything carrying ``gradient_checkpointing``) that contain an ``ExpertsLoRA``.

    Matched on STRUCTURE, not on child names: the MoE block is the direct child that holds the experts (``mlp`` on
    Qwen3 / OLMoE / Mixtral / ERNIE, ``block_sparse_moe`` on Granite, ``feed_forward`` on LFM2 / Jamba, ``mixer`` on
    Nemotron-H, ``experts`` itself on Gemma-4). A layer without experts (a Mamba, Gated DeltaNet or dense layer of a hybrid)
    has no MoE activations to keep and is never touched."""
    return [m for m in model.modules()
            if hasattr(m, "gradient_checkpointing") and m is not model and _holds_experts(m)
            and not any(c is not m and hasattr(c, "gradient_checkpointing") and _holds_experts(c) for c in m.modules())]


def _experts_offloaded(layer) -> bool:
    """True if any ``ExpertsLoRA`` below ``layer`` has its frozen storage off the GPU (an offloaded or arena-backed home)."""
    from ..lora import ExpertsLoRA
    for m in layer.modules():
        if isinstance(m, ExpertsLoRA):
            w = getattr(m.base, "gate_up_proj", None)
            if w is None or not getattr(w, "is_cuda", False) or w.numel() == 0:
                return True
    return False


def _recompute_children(layer):
    """The direct children of ``layer`` to keep checkpointed: every weighted child that does not hold the experts (attention,
    a linear-attention / state-space / short-conv mixer, a parallel dense MLP, a router). On a Qwen3-MoE layer that is exactly
    ``self_attn`` -- the form this module had when it matched ``self_attn`` by name. Empty for a layer whose only weighted child
    is the MoE block (Nemotron-H's MoE layers): nothing is recomputed there."""
    return [(name, c) for name, c in layer.named_children() if _is_weighted(c) and not _holds_experts(c)]


def keep_moe_activations(model, n="all", verbose: bool = False) -> int:
    """Keep the MoE block's activations in the last ``n`` (or ``"all"``) checkpointed MoE-bearing layers of ``model``: the
    layer's whole-layer checkpointing is switched off and each of its other weighted children is checkpointed on its own
    (:func:`_recompute_children`). Returns the number of layers changed (0 when Hugging Face gradient checkpointing is off:
    there is no recompute to avoid)."""
    layers = [m for m in _decoder_layers(model) if m.gradient_checkpointing and not hasattr(m, "_e4b_keep_refs")]
    # Expert offload REQUIRES the layer's recompute: its backward re-dequantizes from whatever the recompute re-staged, and an
    # un-recomputed layer's experts are evicted by then (offload.py's invariant). A layer whose expert storage is not resident
    # on the GPU is therefore never kept; the count says how many were left on whole-layer checkpointing for that reason.
    offloaded = [m for m in layers if _experts_offloaded(m)]
    if offloaded:
        MOE_KEEP_STATS["skipped_offloaded"] = MOE_KEEP_STATS.get("skipped_offloaded", 0) + len(offloaded)
        layers = [m for m in layers if m not in offloaded]
    if n != "all":
        layers = layers[-int(n):] if int(n) > 0 else []
    n_children = 0
    for layer in layers:
        ckpt = getattr(layer, "_gradient_checkpointing_func", None)
        if ckpt is None:
            from torch.utils.checkpoint import checkpoint
            ckpt = functools.partial(checkpoint, use_reentrant=False)
        refs = []
        for _name, child in _recompute_children(layer):
            orig = child.forward

            def forward(*args, _orig=orig, _ckpt=ckpt, _layer=layer, **kwargs):
                if _layer.training:                  # the condition Hugging Face's own layer checkpointing uses
                    return _ckpt(functools.partial(_orig, **kwargs), *args)
                return _orig(*args, **kwargs)
            refs.append((child, orig))
            child.forward = forward
        n_children += len(refs)
        layer._e4b_keep_refs = refs
        layer.gradient_checkpointing = False
    MOE_KEEP_STATS["layers"] += len(layers)
    MOE_KEEP_STATS["checkpointed_children"] += n_children
    if verbose:
        print(f"[e4b.moe_keep] MoE activations kept in {len(layers)} layer(s); {n_children} non-MoE child(ren) still checkpointed"
              + (f"; {len(offloaded)} offloaded layer(s) left on whole-layer checkpointing" if offloaded else ""))
    return len(layers)


def release_moe_activations(model) -> int:
    """Undo ``keep_moe_activations``: whole-layer checkpointing back on, every wrapped child unwrapped."""
    n = 0
    for layer in _decoder_layers(model):
        refs = getattr(layer, "_e4b_keep_refs", None)
        if refs is not None:
            for child, orig in refs:
                child.forward = orig
            del layer._e4b_keep_refs
            layer.gradient_checkpointing = True
            n += 1
    MOE_KEEP_STATS["layers"] = max(0, MOE_KEEP_STATS["layers"] - n)
    return n
