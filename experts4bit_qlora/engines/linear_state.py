# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-slot linear-attention state for the paged decode loop (hybrid families: Qwen3.5 / Qwen3.6 / Qwen3-Next).

:class:`~.paged_runner.PagedModelRunner` drives the model with ``use_cache=False`` and keys every sequence by its KV
slot; paged attention (:mod:`.paged_attention`) owns the attention layers' K/V. A hybrid model's linear-attention
layers (Gated DeltaNet) keep two more per-sequence tensors across calls -- a causal-conv window and a recurrent state
-- and, with no cache object, each call would start them from zero. This module keeps them in a per-slot pool.

For each forward, :func:`install` wraps every supported linear-attention module so that it receives transformers' OWN
``LinearAttentionLayer``, built from the pooled states of the rows the paged context binds (``ctx.slots``), and then
writes the updated states back. The state arithmetic (conv update, chunked and recurrent delta rules) stays
transformers'; only the bookkeeping is ours. Everything else -- attention, masks, the rest of the model -- is
untouched: the wrapper activates only while a paged context is bound and otherwise calls the module unchanged.

Invariants, each pinned by ``tests/test_linear_state.py``:
* a forward's rows either all carry state (decode) or none does (a prompt's first chunk); a mix is refused;
* a slot's state survives chunked prefill and batched decode in any row order, against transformers' DynamicCache;
* :meth:`LinearStatePool.reset` makes a recycled slot start from zero.

Decode graphs (:meth:`~.paged_runner.PagedModelRunner.enable_decode_graphs`) are refused for a model with linear
layers: the gather / scatter here is not yet captured.
"""
from __future__ import annotations

import functools

import torch

from .paged_attention import current_context

#: the layer type name transformers gives a linear-attention (and a Mamba-style) layer in ``config.layer_types``
LINEAR_LAYER_TYPE = "linear_attention"


def _linear_classes():
    """The linear-attention module classes this pool drives (those that read conv / recurrent state through
    ``cache_params``), importable in the installed transformers."""
    out = []
    for mod, name in (("qwen3_5_moe", "Qwen3_5MoeGatedDeltaNet"), ("qwen3_5", "Qwen3_5GatedDeltaNet"),
                      ("qwen3_next", "Qwen3NextGatedDeltaNet")):
        try:
            m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[name])
            out.append(getattr(m, name))
        except (ImportError, AttributeError):
            continue
    return tuple(out)


def layer_types(config) -> list[str]:
    """``config.layer_types``, or its ``text_config``'s for a composite (vision-language) config; empty when absent."""
    types = getattr(config, "layer_types", None)
    if not types:
        types = getattr(getattr(config, "text_config", None), "layer_types", None)
    return list(types or ())


def linear_layers(config) -> list[int]:
    """Indices of the linear-attention layers in :func:`layer_types` (empty for a non-hybrid model). transformers
    labels Mamba-style layers ``linear_attention`` too; :func:`install` refuses those it cannot drive."""
    return [i for i, t in enumerate(layer_types(config)) if t == LINEAR_LAYER_TYPE]


class LinearStatePool:
    """Per-slot conv / recurrent state for every linear-attention layer, allocated on first write."""

    def __init__(self, n_slots: int):
        self.n_slots = int(n_slots)
        self.conv: dict[int, torch.Tensor] = {}
        self.rec: dict[int, torch.Tensor] = {}
        self.kernel: dict[int, int] = {}
        self.has = [False] * self.n_slots          # a slot carries state once its first prompt chunk has run

    def view(self, layer: int, slots):
        """transformers' ``LinearAttentionLayer`` for ``slots``' rows of ``layer``: empty when none carries state."""
        from transformers.cache_utils import LinearAttentionLayer
        lal = LinearAttentionLayer()
        have = [self.has[s] for s in slots]
        if any(have):
            if not all(have):
                raise RuntimeError(f"linear layer {layer}: rows bound to slots {list(slots)} mix sequences with and "
                                   "without state -- a prompt's first chunk must run on its own")
            idx = torch.tensor(list(slots), device=self.conv[layer].device)
            conv = self.conv[layer].index_select(0, idx)
            rec = self.rec[layer].index_select(0, idx)
            lal.lazy_initialization(conv_states=conv, conv_kernel_size=self.kernel[layer])
            lal.conv_states[0].copy_(conv)
            lal.lazy_initialization(recurrent_states=rec)
            lal.recurrent_states[0].copy_(rec)
            lal.has_previous_state[0] = True
        return lal

    def store(self, layer: int, slots, lal) -> None:
        conv, rec = lal.conv_states[0], lal.recurrent_states[0]
        if conv is None or rec is None:
            raise RuntimeError(f"linear layer {layer} produced no conv / recurrent state for slots {list(slots)}")
        if layer not in self.conv:
            self.conv[layer] = torch.zeros((self.n_slots, *conv.shape[1:]), dtype=conv.dtype, device=conv.device)
            self.rec[layer] = torch.zeros((self.n_slots, *rec.shape[1:]), dtype=rec.dtype, device=rec.device)
            self.kernel[layer] = int(lal.conv_kernel_size[0])
        idx = torch.tensor(list(slots), device=conv.device)
        self.conv[layer].index_copy_(0, idx, conv)
        self.rec[layer].index_copy_(0, idx, rec)

    def mark(self, slots) -> None:
        """The rows bound to ``slots`` have run (a prompt chunk or a decode step): they now carry state."""
        for s in slots:
            self.has[s] = True

    def ensure_slots(self, n_slots: int) -> None:
        """Grow to at least ``n_slots`` slots, keeping every existing slot's state; never shrinks. A second runner on
        the same model with a larger batch shares this pool (``install`` returns it), and must not index past it."""
        n_slots = int(n_slots)
        if n_slots <= self.n_slots:
            return
        for d in (self.conv, self.rec):
            for layer, t in d.items():
                grown = torch.zeros((n_slots, *t.shape[1:]), dtype=t.dtype, device=t.device)
                grown[: self.n_slots].copy_(t)
                d[layer] = grown
        self.has += [False] * (n_slots - self.n_slots)
        self.n_slots = n_slots

    def reset(self, slot: int) -> None:
        """A recycled slot carries no history: its next forward starts every linear layer from zero."""
        if 0 <= slot < self.n_slots:
            self.has[slot] = False

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in (self.conv, self.rec) for t in d.values())


class _OneLayerCache:
    """The cache surface a Gated DeltaNet forward touches, for its own layer, backed by ``LinearAttentionLayer``."""

    def __init__(self, layer_idx: int, lal):
        self.layers = {layer_idx: lal}
        self._lal = lal

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return bool(self._lal.has_previous_state[0])

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kw):
        return self._lal.update_conv_state(conv_states, 0, **kw)

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx=0, **kw):
        return self._lal.update_recurrent_state(recurrent_states, 0, **kw)


def install(model, n_slots: int) -> LinearStatePool | None:
    """Wrap ``model``'s linear-attention modules to read and write a :class:`LinearStatePool` of ``n_slots`` slots
    while a paged context is bound. Returns the pool (also kept as ``model._e4b_linear_state``), or None when the
    model has no linear layers. Refuses a model whose ``layer_types`` names linear layers this module cannot drive."""
    want = linear_layers(getattr(model, "config", None))
    if not want:
        return None
    existing = getattr(model, "_e4b_linear_state", None)
    if existing is not None:
        existing.ensure_slots(n_slots)                 # a later runner may bind more slots than the first did
        return existing
    try:
        from transformers.cache_utils import LinearAttentionLayer  # noqa: F401  (the state carrier this pool fills)
    except ImportError as e:
        raise NotImplementedError("the installed transformers has no cache_utils.LinearAttentionLayer; the per-slot "
                                  "linear-attention state needs transformers >= 5.13") from e
    classes = _linear_classes()
    pool = LinearStatePool(n_slots)
    wrapped = []
    for mod in model.modules():
        if classes and isinstance(mod, classes):
            orig = mod.forward
            layer = int(mod.layer_idx)

            @functools.wraps(orig)
            def fwd(hidden_states, cache_params=None, attention_mask=None, _orig=orig, _layer=layer, **kw):
                ctx = current_context()
                if ctx is None or not ctx.slots:
                    return _orig(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kw)
                lal = pool.view(_layer, ctx.slots)
                out = _orig(hidden_states, cache_params=_OneLayerCache(_layer, lal), attention_mask=attention_mask, **kw)
                pool.store(_layer, ctx.slots, lal)
                return out

            mod.forward = fwd
            wrapped.append(layer)
    if sorted(wrapped) != sorted(want):
        raise NotImplementedError(
            f"config.layer_types names linear-attention layers {want}, but the per-slot state wrapper drives "
            f"{sorted(wrapped)} (supported modules: {[c.__name__ for c in classes]}); refusing a model whose linear "
            "layers would run without their state")
    model._e4b_linear_state = pool
    return pool
