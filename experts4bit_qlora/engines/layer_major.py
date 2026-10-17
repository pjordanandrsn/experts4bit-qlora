# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Layer-major gradient accumulation for expert-offloaded training (``LAYER_MAJOR=1``; opt-in, default off).

The stock accumulation loop runs each micro-batch's whole forward and backward before the next one. Under expert offload
(``OFFLOAD_EXPERTS=1``) every layer's experts then cross the host link twice per micro-batch, forward and recompute, so a
step of M micro-batches stages M x (2n - 1) layers. This step runs every micro-batch through layer L before layer L+1, with
the layer held resident across them (:func:`~experts4bit_qlora.engines.offload.hold_forward_residency`), then takes ONE
backward of the weighted sum of the micro-batch losses. Autograd's backward is then layer-major on its own (assumption A1,
``bench/q235/staged-bytes``, tested under both checkpoint kinds): each layer's M recomputes run back to back on one staging.
A step stages 2n - 1 layers in steady state (2n from cold): M times fewer bytes.

What is the same as the stock loop, and what is not:
- The forward of each micro-batch is the model's own, call for call: the embedding, the causal mask and the rotary
  embedding are built exactly as the model's forward builds them, each decoder layer is called with the same arguments
  (so its gradient checkpoint, reentrant or not, is the one the model installed), and the final norm, the head and the
  loss run through the model's OWN forward (``inputs_embeds`` = the last layer's output, with the decoder layers swapped
  out for that call), so a loss patch such as e4b's chunked loss applies unchanged. Losses are ``torch.equal`` to the
  stock forward's (``tests/test_layer_major.py``).
- The gradients are the same weighted sum in a different summation order: one backward of the sum, where the stock loop
  accumulates M backwards into ``.grad``. In bf16 that moves bits; the registered band is the parity test's, not equality.
- Memory: every micro-batch's checkpointed layer inputs and its loss's saved state are alive until the one backward
  (about +0.8 GB at Q235's Shape B, ``bench/q235/staged-bytes``). The per-layer working set is unchanged.

Supported: the classes in :data:`SUPPORTED` (v1: Qwen3-MoE). Anything else is refused by name, as is a model that
returns router logits (the auxiliary loss would need the layers' router outputs, which the head-only call cannot see).
"""
from __future__ import annotations

import contextlib

import torch

from .offload import hold_forward_residency, offload_handles

#: Causal-LM classes whose decoder forward this module reproduces (read in transformers' source, test-pinned).
SUPPORTED = ("Qwen3MoeForCausalLM",)


def _refuse(model) -> None:
    name = type(model).__name__
    if name not in SUPPORTED:
        raise NotImplementedError(
            f"layer_major_step: {name} is not supported (v1: {', '.join(SUPPORTED)}); use the stock accumulation loop"
        )
    if getattr(model.config, "output_router_logits", False):
        raise NotImplementedError(
            "layer_major_step: output_router_logits is on; the router auxiliary loss is not carried by the layer-major step"
        )


@contextlib.contextmanager
def _without_layers(inner):
    """For the head-only call: the decoder stack swapped for an empty one, restored on exit."""
    saved = inner.layers
    inner.layers = torch.nn.ModuleList()
    try:
        yield
    finally:
        inner.layers = saved


def layer_major_step(model, batches, weights, handles=None):
    """One accumulation step, layer-major. Returns the micro-batch losses (detached), after the backward.

    ``batches``: one dict per micro-batch with ``input_ids``, ``labels`` and optionally ``attention_mask`` (as the stock
    loop passes them to ``model(...)``). ``weights``: each micro-batch's weight in the step's loss (the stock loop's
    ``1 / GRAD_ACCUM``, or a token share). The backward is ``sum(w_k * loss_k)``. ``handles``: the offload handles to hold
    (default: :func:`offload_handles`); an empty list runs the same schedule with nothing held.
    """
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    _refuse(model)
    if len(batches) != len(weights):
        raise ValueError(f"layer_major_step: {len(batches)} micro-batches but {len(weights)} weights")
    inner = model.model
    cfg = inner.config
    mask_function = create_causal_mask if cfg.sliding_window is None else create_sliding_window_causal_mask
    handles = offload_handles(model) if handles is None else list(handles)

    # Per micro-batch, what Qwen3MoeModel.forward builds before its layer loop (no cache in training).
    states = []
    for b in batches:
        emb = inner.embed_tokens(b["input_ids"])
        pos = torch.arange(emb.shape[1], device=emb.device).unsqueeze(0)
        mask = mask_function(config=cfg, inputs_embeds=emb, attention_mask=b.get("attention_mask"),
                             past_key_values=None, position_ids=pos)
        states.append([emb, mask, pos, inner.rotary_emb(emb, position_ids=pos)])

    with hold_forward_residency(handles):
        for layer in inner.layers[: cfg.num_hidden_layers]:
            for s in states:
                s[0] = layer(s[0], attention_mask=s[1], position_ids=s[2], past_key_values=None, use_cache=False,
                             position_embeddings=s[3])

    losses = []
    with _without_layers(inner):
        for s, b in zip(states, batches):
            out = model(inputs_embeds=s[0], attention_mask=b.get("attention_mask"), labels=b["labels"], use_cache=False)
            losses.append(out.loss)
    sum(w * loss for w, loss in zip(weights, losses)).backward()
    return [loss.detach() for loss in losses]
