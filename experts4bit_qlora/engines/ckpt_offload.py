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

``E4B_CKPT_OFFLOAD=1`` (unset / ``0``: unchanged, Hugging Face's checkpointing). It is NOT a default yet: TC1 amendment 59's rule
requires a torch 2.8 read on a host-bound box first (TC1 amendment 63), and amendment 62 to say which half of the switch made the
field recipe's step faster. ``enable_fast_train`` applies it after its other switches; layers that ``E4B_MOE_KEEP_LAYERS`` took out
of whole-layer checkpointing are left alone (they keep their activations by design). A model without gradient checkpointing enabled
is left unchanged, with a warning when the variable was set. Ready for when the default flips (:data:`CKPT_OFFLOAD_DEFAULT`): a
default-path request leaves alone a model whose decoder layers carry ``enable_dense_offload``'s handles, because dense offload's
train-prefetch schedule is untested with this checkpoint (an explicit ``E4B_CKPT_OFFLOAD=1`` pairs them, and ``enable_dense_offload``
warns when it finds offloaded checkpoints), and it stays silent without checkpointing. The CLI trainer keeps its own checkpointing.

``E4B_CKPT_OFFLOAD=reentrant`` (a diagnostic, never a default) routes the same layers through the reentrant checkpoint WITHOUT the
host-memory hook: their inputs stay on the GPU. It separates the two halves of this switch -- the checkpoint flavour and the copies
-- for TC1 amendment 62, which asks which of them made the field recipe's step faster.

The evidence so far, Qwen3-30B-A3B on one RTX 5090 in torch 2.12, the matched and shipped arms:
- packed 4,096-token rows (TC1 amendment 58, ``e4b.train.ckpt-offload.packed-4k.5090.2026-10-07``): the training-phase peak 0.739 GB
  lower at 1.003 of the step, held-out +0.0002;
- TC1's field recipe (amendment 59, ``e4b.train.ckpt-offload.field.5090.2026-10-07``): 0.948 (matched) and 0.916 (shipped) of the
  step, the matched training-phase peak 0.171 GB lower, held-out within 0.003.
"""
from __future__ import annotations

import os

CKPT_OFFLOAD_STATS = {"layers": 0, "skipped": None}

#: What unset ``E4B_CKPT_OFFLOAD`` means. Off until TC1 amendment 59's rule is met (amendments 62 and 63 read); the default flip is
#: then this one line, in a library PR that cites the reads.
CKPT_OFFLOAD_DEFAULT = False

_ON, _OFF, _REENTRANT = ("1", "on", "true", "yes"), ("0", "off", "false", "no"), "reentrant"


def _setting() -> str:
    v = os.environ.get("E4B_CKPT_OFFLOAD", "").strip().lower()
    if v and v not in _ON and v not in _OFF and v != _REENTRANT:
        raise ValueError(f"E4B_CKPT_OFFLOAD must be 0, 1 or reentrant, got {v!r}")
    return v


def checkpoint_offload_requested() -> bool:
    """``E4B_CKPT_OFFLOAD``: ``1`` on, ``reentrant`` the reentrant checkpoint alone, ``0`` off, unset :data:`CKPT_OFFLOAD_DEFAULT` (off)."""
    v = _setting()
    return (v in _ON or v == _REENTRANT) if v else CKPT_OFFLOAD_DEFAULT


def checkpoint_offload_explicit() -> bool:
    """True when ``E4B_CKPT_OFFLOAD`` was set (``1`` or ``reentrant``, not the default): it then also pairs with dense offload and
    warns when it finds nothing to route."""
    v = _setting()
    return v in _ON or v == _REENTRANT


def checkpoint_offload_mode() -> str:
    """``"reentrant"`` under ``E4B_CKPT_OFFLOAD=reentrant``, else ``"offload"``."""
    return _REENTRANT if _setting() == _REENTRANT else "offload"


def offloaded_checkpoint(fn, *args, **kwargs):
    """A ``_gradient_checkpointing_func``: the reentrant checkpoint of ``fn`` with its saved inputs in pinned host memory."""
    import torch
    from torch.utils.checkpoint import checkpoint

    kwargs.pop("use_reentrant", None)
    with torch.autograd.graph.save_on_cpu(pin_memory=torch.cuda.is_available()):
        return checkpoint(fn, *args, use_reentrant=True, **kwargs)


def reentrant_checkpoint(fn, *args, **kwargs):
    """A ``_gradient_checkpointing_func``: the reentrant checkpoint of ``fn``, its inputs left on the device (the diagnostic half)."""
    from torch.utils.checkpoint import checkpoint

    kwargs.pop("use_reentrant", None)
    return checkpoint(fn, *args, use_reentrant=True, **kwargs)


def enable_checkpoint_offload(model, verbose: bool = False, explicit: bool = True, mode: str = "offload") -> int:
    """Route every checkpointed decoder layer of ``model`` through :func:`offloaded_checkpoint`; returns the layers changed.

    Only modules that are gradient-checkpointing layers with checkpointing ON are touched (``gradient_checkpointing`` True and a
    ``_gradient_checkpointing_func``). Their previous checkpoint function is kept for :func:`disable_checkpoint_offload`.
    ``explicit=False`` (the default path) skips a model whose layers carry ``enable_dense_offload``'s handles, recording why in
    ``CKPT_OFFLOAD_STATS["skipped"]``, and stays quiet when no layer is checkpointed."""
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer as _Layer
    except ImportError:                                  # older transformers: any module that carries the checkpointing switch
        _Layer = object
    layers = [m for m in model.modules()
              if isinstance(m, _Layer) and m is not model
              and getattr(m, "gradient_checkpointing", False) is True and hasattr(m, "_gradient_checkpointing_func")]
    if not explicit and any(getattr(m, "_dense_offload", None) is not None for m in model.modules()):
        CKPT_OFFLOAD_STATS["skipped"] = "dense offload is on (the pairing is untested; E4B_CKPT_OFFLOAD=1 pairs them)"
        if verbose:
            print(f"[e4b.ckpt_offload] default skipped: {CKPT_OFFLOAD_STATS['skipped']}")
        return 0
    if not layers and not explicit:
        return 0
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
        m._gradient_checkpointing_func = reentrant_checkpoint if mode == _REENTRANT else offloaded_checkpoint
        n += 1
    if n and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    CKPT_OFFLOAD_STATS["layers"] += n
    CKPT_OFFLOAD_STATS["mode"] = mode
    if verbose:
        where = "left on the device (E4B_CKPT_OFFLOAD=reentrant)" if mode == _REENTRANT else "kept in pinned host memory"
        print(f"[e4b.ckpt_offload] {n} checkpointed layer(s): reentrant checkpoint, inputs {where}")
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
