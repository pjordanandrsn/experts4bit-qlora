"""Populate a Muse-Glimmer text tower from a released GGUF.

The keymap in :mod:`experts4bit_qlora.glimmer` says WHERE each released tensor
goes and WHAT arithmetic it needs; this module does the walk: read the GGUF
tensor table, decode each tensor's bytes through grouped-nf4-gemm's k-quant
lane (``kquant_ref`` — computed from the released bytes, never re-quantized),
apply the transform, and assign into a model built on ``meta``.

Two properties this module exists to guarantee, both enforced rather than
hoped for:

* **Nothing is silently skipped.** Every GGUF tensor must classify (the keymap
  raises otherwise) and every parameter the keymap promises must end up
  assigned. The load ends with a coverage reconciliation that raises on any
  unfilled parameter — the failure mode this whole lane is built to make
  impossible is a model that loads clean and computes nonsense.

* **The vision tower is out of scope, and says so.** A text-tower GGUF has no
  ``mmproj`` tensors; the vision half lives in a separate file. Loading only
  the text tower leaves the vision parameters unfilled, so callers get an
  explicit ``text_only=True`` acknowledgement rather than a mystery.

The gnf4 dependency is CAPABILITY-gated, not version-gated (the house rule for
cross-package opt-ins): if the installed grouped-nf4-gemm predates the k-quant
lane, this raises a clear install message instead of an AttributeError from
three frames down.
"""
from __future__ import annotations

import torch

from .glimmer import (
    GlimmerKeymapError,
    expected_param_names,
    map_gguf_key,
    transform_weight,
)


def _kquant_lane():
    """Import gnf4's k-quant lane, or explain exactly what to install."""
    try:
        from kquant_ref import dequantize_ggml  # noqa: PLC0415
        from gguf_reader import read_header, read_tensor_bytes  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the Glimmer GGUF path needs grouped-nf4-gemm's k-quant lane "
            "(kquant_ref + gguf_reader). Install/upgrade grouped-nf4-gemm to a "
            "release that carries them."
        ) from e
    return dequantize_ggml, read_header, read_tensor_bytes


def _assign(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Place `tensor` at dotted parameter `name`, replacing a meta parameter.

    Assigning through ``_parameters`` (rather than ``.data =``) is what makes
    this work for a model built on ``meta``: the meta parameter has no storage
    to write into, so it must be REPLACED.
    """
    parent, _, leaf = name.rpartition(".")
    mod = model.get_submodule(parent) if parent else model
    want = getattr(mod, leaf, None)
    if want is None:
        raise GlimmerKeymapError(f"no such parameter on the model: {name}")
    if tuple(want.shape) != tuple(tensor.shape):
        raise GlimmerKeymapError(
            f"{name}: checkpoint gives {tuple(tensor.shape)}, model declares "
            f"{tuple(want.shape)} — refusing to place a mis-shaped tensor")
    # The tensor arrives already at the caller's requested dtype; do NOT re-cast
    # to the meta parameter's declared dtype. Doing so silently ignored an
    # explicit dtype= (an fp32 load came back bf16-rounded) — the caller decides
    # serving precision, not the config the skeleton was built from.
    mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)


def _text_layer_count(model: torch.nn.Module) -> int:
    """The model's OWN text-tower depth. Derived, never taken on trust: a
    caller-supplied count that undershoots would let upper layers stay on meta
    while the coverage check passed (Bugbot #92)."""
    try:
        return len(model.get_submodule("model.language_model.layers"))
    except AttributeError as e:
        raise GlimmerKeymapError(
            "model has no model.language_model.layers — not a Glimmer tree") from e


def load_glimmer_text_tower(
    gguf_path: str,
    model: torch.nn.Module,
    *,
    qk_scale_factor: float,
    num_layers: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> dict:
    """Fill `model`'s text-tower parameters from the GGUF at `gguf_path`.

    `model` is typically built with ``AutoModelForImageTextToText.from_config``
    under ``torch.device("meta")``. Returns a report dict:
    ``{"assigned": n, "dropped": n, "text_only": True, "unfilled": [...]}``.
    With ``strict`` (the default) any unfilled promised parameter raises.
    """
    dequantize_ggml, read_header, read_tensor_bytes = _kquant_lane()
    depth = _text_layer_count(model)
    if num_layers is not None and num_layers != depth:
        raise GlimmerKeymapError(
            f"num_layers={num_layers} disagrees with the model's own depth "
            f"({depth}) — refusing to validate coverage against a wrong count")
    header = read_header(gguf_path)

    assigned, dropped = 0, 0
    for info in header.tensors:
        param_name, transform = map_gguf_key(info.name)
        raw = read_tensor_bytes(gguf_path, info)
        decoded = dequantize_ggml(info.ggml_type, raw, info.shape)
        out = transform_weight(decoded, transform,
                               qk_scale_factor=qk_scale_factor, name=info.name)
        if out is None:                      # validated drop (qk-norm scalars)
            dropped += 1
            continue
        _assign(model, param_name, out.to(dtype))
        assigned += 1

    # Coverage is checked against the MODEL, not against what we promised: every
    # text-tower parameter must be materialized. That is strictly stronger than
    # walking `promised` (which can only ever find holes it already knew about)
    # and it is derived from the tree, so no caller argument can weaken it.
    promised = expected_param_names(depth)
    unfilled = sorted(
        n for n, t in model.named_parameters()
        if t.is_meta and (n.startswith("model.language_model.") or n == "lm_head.weight"))
    missing_promised = sorted(promised - {n for n, _ in model.named_parameters()})
    if missing_promised:
        raise GlimmerKeymapError(
            f"keymap promises parameters the model lacks: {missing_promised[:3]}")
    if strict and unfilled:
        raise GlimmerKeymapError(
            f"{len(unfilled)} text-tower parameters still unfilled after load, "
            f"e.g. {unfilled[:3]} — the GGUF is incomplete or the keymap drifted")
    return {"assigned": assigned, "dropped": dropped,
            "text_only": True, "unfilled": unfilled}
