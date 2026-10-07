"""Keep each checkpointed decoder layer's input in pinned host memory instead of on the GPU (opt-in).

Under gradient checkpointing every decoder layer keeps one tensor across the step: its input hidden states, so that backward can
re-run the layer. On Qwen3-30B-A3B's packed 4,096-token rows that is 47 tensors of 16.8 MB, 0.79 GB of e4b's training peak, and
the largest single group TC1 amendment 57's census found above Unsloth's (whose own checkpointing keeps no such group on the GPU;
``e4b.train.memory.packed-4k-train-census.5090.2026-10-07``).

How: each checkpointed layer runs PyTorch's REENTRANT checkpoint inside ``torch.autograd.graph.save_on_cpu(pin_memory=True)``.
The reentrant checkpoint runs the layer under ``no_grad`` and saves exactly its tensor inputs with ``save_for_backward``, so the
host-memory hook moves those inputs -- and nothing else -- to pinned host RAM in forward and brings them back for the recompute in
backward. (Hugging Face's default non-reentrant checkpoint holds the input by reference rather than as a saved tensor, so no
saved-tensor hook can reach it.) A reentrant checkpoint only produces gradients for what is inside the layer when its input
requires grad, so this also calls the model's ``enable_input_require_grads()``, as PEFT does for reentrant checkpointing; that
changes no value. The copies are synchronous: a device-to-host copy per layer in forward and host-to-device in backward. What
that costs a step is for a TC1 box to read.

``E4B_CKPT_OFFLOAD=1`` (unset / ``0``: unchanged). ``enable_fast_train`` applies it after its other switches; layers that
``E4B_MOE_KEEP_LAYERS`` took out of whole-layer checkpointing are left alone (they keep their activations by design). A model
without gradient checkpointing enabled is left unchanged, with a warning when the variable was set.
"""
from __future__ import annotations

import os

CKPT_OFFLOAD_STATS = {"layers": 0}


def checkpoint_offload_requested() -> bool:
    """``E4B_CKPT_OFFLOAD=1``."""
    return os.environ.get("E4B_CKPT_OFFLOAD", "").strip() == "1"


def offloaded_checkpoint(fn, *args, **kwargs):
    """A ``_gradient_checkpointing_func``: the reentrant checkpoint of ``fn`` with its saved inputs in pinned host memory."""
    import torch
    from torch.utils.checkpoint import checkpoint

    kwargs.pop("use_reentrant", None)
    with torch.autograd.graph.save_on_cpu(pin_memory=torch.cuda.is_available()):
        return checkpoint(fn, *args, use_reentrant=True, **kwargs)


def enable_checkpoint_offload(model, verbose: bool = False) -> int:
    """Route every checkpointed decoder layer of ``model`` through :func:`offloaded_checkpoint`; returns the layers changed.

    Only modules that are gradient-checkpointing layers with checkpointing ON are touched (``gradient_checkpointing`` True and a
    ``_gradient_checkpointing_func``). Their previous checkpoint function is kept for :func:`disable_checkpoint_offload`."""
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer as _Layer
    except ImportError:                                  # older transformers: any module that carries the checkpointing switch
        _Layer = object
    layers = [m for m in model.modules()
              if isinstance(m, _Layer) and m is not model
              and getattr(m, "gradient_checkpointing", False) is True and hasattr(m, "_gradient_checkpointing_func")]
    if not layers:
        import warnings
        warnings.warn("[e4b.ckpt_offload] E4B_CKPT_OFFLOAD=1, but no layer has gradient checkpointing enabled: nothing to offload. "
                      "Call model.gradient_checkpointing_enable() first.", RuntimeWarning, stacklevel=2)
        return 0
    n = 0
    for m in layers:
        if getattr(m, "_e4b_ckpt_offload_ref", None) is not None:
            continue
        m._e4b_ckpt_offload_ref = m._gradient_checkpointing_func
        m._gradient_checkpointing_func = offloaded_checkpoint
        n += 1
    if n and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    CKPT_OFFLOAD_STATS["layers"] += n
    if verbose:
        print(f"[e4b.ckpt_offload] {n} checkpointed layer(s): inputs kept in pinned host memory (reentrant checkpoint)")
    return n


def disable_checkpoint_offload(model) -> int:
    """Undo :func:`enable_checkpoint_offload`; returns the layers restored. The input-requires-grad hook stays (it changes no value)."""
    n = 0
    for m in model.modules():
        ref = getattr(m, "_e4b_ckpt_offload_ref", None)
        if ref is not None:
            m._gradient_checkpointing_func = ref
            del m._e4b_ckpt_offload_ref
            n += 1
    return n
