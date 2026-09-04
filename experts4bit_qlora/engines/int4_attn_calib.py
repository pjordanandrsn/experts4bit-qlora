# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calibrated int4 for the serving attention projections (and, opted in
via ``E4B_SERVE_LMHEAD_INT4_CALIB=1``, the output head)
(``E4B_SERVE_ATTN_INT4_CALIB=1``).

:mod:`int4_attn` packs each attention projection by rounding every weight
to its nearest grid point. That lane was refused on quality (+0.0558 ppl
against a 0.05 gate), and the fp8 lane that followed showed why the
obvious fix is not a fix: e4m3 carried 4.6x lower WEIGHT error than int4
and bought only ~12% less perplexity cost. Weight error is not what the
gate measures.

This module keeps the same grid, the same packed bytes and the same
kernel, and changes only WHICH grid point each weight lands on. A short
calibration pass records, for every attention projection, how the
model's own activations excite each input channel (``H = 2 X X^T``);
the kernel package's ``gptq_pack_int4_b32`` then chooses grid points
that minimise the projection's OUTPUT error under that weighting,
pushing each column's rounding residual into the columns that follow.
The comparison engine's checkpoint is quantised this way (GPTQ, group
128) and serves its attention in int4 at a quality its users accept;
ours never had the calibration, only the format.

Calibration runs FORWARD only, on the text the K8 bake already carries,
in 512-token chunks; each projection's Hessian is accumulated on the
GPU in fp32 (16 MB for a 2048-wide input, 64 MB for o_proj's 4096) and
freed as soon as that projection is packed. Engagement is the banner
``ATTNINT4 calibrated: N projections`` plus census presence of the int4
GEMV, exactly as the uncalibrated lane.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable

import torch
from torch import nn

__all__ = ["calibrate_attention_hessians", "enable_serve_attn_int4_calib"]


def _attention_linears(model) -> Dict[str, nn.Linear]:
    """Every ``nn.Linear`` that is a direct child of an ``*Attention``
    module -- the same structural rule :mod:`int4_attn` swaps on, so the
    calibrated lane patches exactly the set the uncalibrated one does."""
    out: Dict[str, nn.Linear] = {}
    for mname, mod in model.named_modules():
        if not type(mod).__name__.endswith("Attention"):
            continue
        for cname, child in mod.named_children():
            if type(child) is nn.Linear:
                out[f"{mname}.{cname}"] = child
    return out


_LM_HEAD_ENV = "E4B_SERVE_LMHEAD_INT4_CALIB"


def _output_head(model):
    """The model's output projection as ``(qualified_name, nn.Linear)``,
    or None. Found through ``get_output_embeddings`` (the transformers
    contract) with ``lm_head`` as the fallback, and located in the module
    tree by identity so the swap goes on the right parent."""
    head = None
    get = getattr(model, "get_output_embeddings", None)
    if callable(get):
        try:
            head = get()
        except Exception:
            head = None
    if head is None:
        head = getattr(model, "lm_head", None)
    if type(head) is not nn.Linear:
        return None
    for name, mod in model.named_modules():
        if mod is head:
            return name, head
    return None


def _int4_targets(model, include_attention: bool = True,
                  include_head: bool = False) -> Dict[str, nn.Linear]:
    """The projections one calibrated enable packs: the attention set
    (the structural rule above) and, opted in, the output head.

    The output head was measured +0.18 ppl UNCALIBRATED and refused
    (``int4_attn``); the calibrated packer is what turned attention from
    -0.006 to -0.042, so the head gets the same packer under its own
    flag and its own K8 arm -- never silently alongside attention. On a
    model with tied embeddings the head's int4 store is a second copy;
    the embedding keeps its bf16 table untouched (the swap replaces the
    Linear module, not the shared Parameter)."""
    lins = _attention_linears(model) if include_attention else {}
    if include_head:
        found = _output_head(model)
        if found is None:
            raise RuntimeError(
                f"{_LM_HEAD_ENV}=1: the model has no nn.Linear output head "
                "to pack (get_output_embeddings / lm_head)")
        name, head = found
        if head.bias is not None:
            raise RuntimeError(
                f"{_LM_HEAD_ENV}=1: the output head carries a bias; "
                "weight-only int4 refuses rather than dropping it")
        lins[name] = head
    return lins


@torch.no_grad()
def calibrate_attention_hessians(model, batches: Iterable[torch.Tensor],
                                 device=None, hessian_device="cpu",
                                 include_attention: bool = True,
                                 include_head: bool = False,
                                 ) -> Dict[str, torch.Tensor]:
    """Run ``batches`` (token-id tensors ``[B, T]``) through ``model`` and
    return ``{qualified_name: H}`` with ``H = 2 X X^T`` over every input
    row each attention projection saw. Hooks are removed on exit even if
    a batch raises.

    Each batch's Gram is computed on the model's device; the running
    Hessians LIVE on ``hessian_device`` (CPU by default). Keeping them on
    the card OOMed Mixtral-8x7B's calibration on a 32 GB GPU: 128
    projections x 64 MB beside a 23 GB model (receipts P24-GEN-B)."""
    from gptq_pack import HessianAccumulator

    lins = _int4_targets(model, include_attention, include_head)
    if not lins:
        raise RuntimeError("calibration found no attention projections")
    dev = device or next(model.parameters()).device
    accs = {n: HessianAccumulator(lin.in_features, device=hessian_device)
            for n, lin in lins.items()}
    handles = []
    for n, lin in lins.items():
        def _hook(mod, inputs, _n=n):
            accs[_n].add(inputs[0])            # Gram where the activations are
        handles.append(lin.register_forward_pre_hook(_hook))
    try:
        for ids in batches:
            model(ids.to(dev))
    finally:
        for h in handles:
            h.remove()
    return {n: a.H for n, a in accs.items()}


def enable_serve_attn_int4_calib(model, hessians: Dict[str, torch.Tensor],
                                 include_attention: bool = True,
                                 include_head: bool = False) -> int:
    """Swap every attention projection (and, opted in, the output head)
    for the int4 store, packed with calibration. A projection without a
    Hessian is NOT silently packed uncalibrated -- that would mix two
    quantisers under one banner and make the quality gate ambiguous; it
    raises instead."""
    from .int4_attn import Int4Linear, _kernels
    try:
        _kernels()
    except ImportError as e:
        raise RuntimeError(
            "E4B_SERVE_ATTN_INT4_CALIB=1 needs grouped-nf4-gemm with "
            f"int4_b32 and gptq_pack (missing: {e})") from e
    from gptq_pack import gptq_pack_int4_b32

    lins = _int4_targets(model, include_attention, include_head)
    missing = sorted(n for n in lins if n not in hessians)
    if missing:
        raise RuntimeError(
            f"calibrated enable: {len(missing)} attention projections have "
            f"no Hessian (first: {missing[0]}); refusing to pack them "
            "uncalibrated under the calibrated banner")
    n = 0
    for name, lin in lins.items():
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, child = model, name        # a top-level lm_head
        # Int4Linear packs on the CPU copy of the weight; the Cholesky of
        # a 4096x4096 fp32 Hessian there is seconds, and it keeps the GPU
        # free for the store buffers being built alongside
        H = hessians[name].to("cpu")

        def packer(w, _H=H):
            return gptq_pack_int4_b32(w, _H)
        setattr(parent, child, Int4Linear(lin, packer=packer))
        hessians[name] = None          # free the 16-64 MB as we go
        n += 1
    if n == 0:
        raise RuntimeError("E4B_SERVE_ATTN_INT4_CALIB=1 matched no "
                           "attention projections -- refusing a vacuous enable")
    return n


def enable_from_env(model, batches: Iterable[torch.Tensor]) -> int:
    """Harness convenience: calibrate then enable when the flags are set.
    ``E4B_SERVE_ATTN_INT4_CALIB=1`` packs the attention projections;
    ``E4B_SERVE_LMHEAD_INT4_CALIB=1`` packs the output head (alone, or
    beside attention). Both off is a no-op; the head flag alone is a
    head-only enable, never a silent ignore."""
    attn = os.environ.get("E4B_SERVE_ATTN_INT4_CALIB", "0") == "1"
    head = os.environ.get(_LM_HEAD_ENV, "0") == "1"
    if not attn and not head:
        return 0
    hess = calibrate_attention_hessians(model, batches,
                                        include_attention=attn,
                                        include_head=head)
    return enable_serve_attn_int4_calib(model, hess,
                                        include_attention=attn,
                                        include_head=head)
