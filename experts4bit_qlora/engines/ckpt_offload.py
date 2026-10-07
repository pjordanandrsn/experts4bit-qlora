"""e4b's checkpoint for decoder layers: PyTorch's reentrant checkpoint by default, optionally with its inputs in pinned host memory.

``E4B_CKPT_OFFLOAD`` (under ``enable_fast_train``):
- unset (the default since TC1 amendment 64) or ``reentrant``: every checkpointed decoder layer runs PyTorch's REENTRANT checkpoint
  (:func:`reentrant_checkpoint`), its input left on the GPU;
- ``1``: the same checkpoint inside ``torch.autograd.graph.save_on_cpu(pin_memory=True)`` (:func:`offloaded_checkpoint`), so each
  layer's input lives in pinned host memory between forward and backward -- a memory lever, opt-in;
- ``0``: Hugging Face's own (non-reentrant) checkpoint, unchanged -- the way back.

Why the reentrant checkpoint is the default (Qwen3-30B-A3B on one RTX 5090, torch 2.12 and 2.8, two recipes; TC1 amendments 58, 59 and
62-64): it runs the layer's first forward under ``no_grad`` and saves only its tensor inputs, where Hugging Face's non-reentrant
checkpoint builds the layer's autograd graph in forward. Its step against Hugging Face's checkpoint:
- the field recipe, shipped arm: 0.900 in torch 2.12 (``e4b.train.ckpt-flavour.field.5090.2026-10-07``) and 0.838 in torch 2.8 on a
  host-bound box (``e4b.train.ckpt-flavour.default.5090.2026-10-07``);
- packed 4,096-token rows, matched arm: 0.980 (same register row).
It leaves the training peak where Hugging Face's checkpoint has it (identical on packed rows and at the field recipe), and held-out
stayed within 0.004 everywhere.

The offload (``1``) saves memory for a cost. Each layer's input stays off the GPU between forward and backward: on packed rows 47 x
16.8 MB, 0.74 GB of the training peak, for 1.023 of the reentrant checkpoint's step. At the field recipe it saves 0.17-0.22 GB for
1.02-1.04. The copies are synchronous: a device-to-host copy per layer in forward, host-to-device in backward.

What the reentrant checkpoint changes for a caller:
- it does not support ``torch.autograd.grad`` or ``backward(inputs=...)`` through the checkpointed layers;
- a layer gets gradients for its contents only when its input requires grad, so this calls the model's
  ``enable_input_require_grads()``, as PEFT does for reentrant checkpointing. That changes no value; ``disable_checkpoint_offload``
  leaves the hook in place.
``E4B_CKPT_OFFLOAD=0`` restores Hugging Face's checkpoint.

It applies to the layers checkpointed when it runs. Calling ``gradient_checkpointing_enable()`` afterwards puts Hugging Face's
checkpoint back on every layer, and Hugging Face's ``Trainer`` (TRL's too) does exactly that inside ``train()`` when its arguments say
``gradient_checkpointing=True``. To keep e4b's checkpoint there, enable checkpointing on the model before ``enable_fast_train`` and
leave the trainer's ``gradient_checkpointing`` off, as the guide's loop does; calling ``enable_fast_train`` (or this function) again
after a re-enable routes the layers again.

Where it is not applied: layers that ``E4B_MOE_KEEP_LAYERS`` took out of whole-layer checkpointing keep their activations by design. A
model without gradient checkpointing is left unchanged (silently by default, with a warning when the variable was set). By default, a
model whose decoder layers carry ``enable_dense_offload``'s handles is left alone too, because dense offload's train-prefetch schedule
is untested with this checkpoint. An explicit ``1`` or ``reentrant`` pairs them, and ``enable_dense_offload`` warns when it finds
these checkpoints. The CLI trainer (``python -m experts4bit_qlora.train``) keeps its own checkpointing.
"""
from __future__ import annotations

import os

CKPT_OFFLOAD_STATS = {"layers": 0, "skipped": None}

#: What unset ``E4B_CKPT_OFFLOAD`` means: ``"reentrant"`` since TC1 amendment 64 (``None`` would be Hugging Face's checkpoint).
CKPT_DEFAULT_MODE = "reentrant"

_ON, _OFF, _REENTRANT = ("1", "on", "true", "yes"), ("0", "off", "false", "no"), "reentrant"


def _setting() -> str:
    v = os.environ.get("E4B_CKPT_OFFLOAD", "").strip().lower()
    if v and v not in _ON and v not in _OFF and v != _REENTRANT:
        raise ValueError(f"E4B_CKPT_OFFLOAD must be 0, 1 or reentrant, got {v!r}")
    return v


def checkpoint_offload_requested() -> bool:
    """Whether ``enable_fast_train`` routes the checkpointed layers at all: ``1`` / ``reentrant`` yes, ``0`` no, unset when
    :data:`CKPT_DEFAULT_MODE` is set (the reentrant checkpoint, since TC1 amendment 64)."""
    v = _setting()
    return (v in _ON or v == _REENTRANT) if v else CKPT_DEFAULT_MODE is not None


def checkpoint_offload_explicit() -> bool:
    """True when ``E4B_CKPT_OFFLOAD`` was set (``1`` or ``reentrant``, not the default): it then also pairs with dense offload and
    warns when it finds nothing to route."""
    v = _setting()
    return v in _ON or v == _REENTRANT


def checkpoint_offload_mode() -> str:
    """``"offload"`` under ``E4B_CKPT_OFFLOAD=1``, ``"reentrant"`` under ``=reentrant`` or unset (:data:`CKPT_DEFAULT_MODE`)."""
    v = _setting()
    if v in _ON:
        return "offload"
    return _REENTRANT if v == _REENTRANT else (CKPT_DEFAULT_MODE or "offload")


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
    want = reentrant_checkpoint if mode == _REENTRANT else offloaded_checkpoint
    for m in layers:
        cur = m._gradient_checkpointing_func
        if cur is want:
            continue
        # A layer routed before whose checkpoint was replaced since -- Hugging Face's Trainer calls gradient_checkpointing_enable()
        # again inside train() -- keeps the replacement as the function disable restores; one already on the other mode keeps its ref.
        if cur not in (reentrant_checkpoint, offloaded_checkpoint) or getattr(m, "_e4b_ckpt_offload_ref", None) is None:
            m._e4b_ckpt_offload_ref = cur
        m._gradient_checkpointing_func = want
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
