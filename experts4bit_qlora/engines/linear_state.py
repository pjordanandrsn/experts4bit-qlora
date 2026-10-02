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

Decode graphs (:meth:`~.paged_runner.PagedModelRunner.enable_decode_graphs`). A captured graph cannot read
``ctx.slots``, a Python list baked at capture. On a bucketed decode step the wrapper therefore gathers and scatters
through the bound bucket's device selector (the KV's ``_g_sel``, rewritten before every replay; see
:func:`_bucket_selector`), and takes every row as carrying state. A graph cannot allocate, so the runner warms the
pool before capture, and the pool is then ``frozen``: :meth:`LinearStatePool.ensure_slots` refuses to move tensors a
graph holds.
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
        self.frozen = False                        # a captured decode graph holds these tensors' addresses

    def view(self, layer: int, slots, sel=None):
        """transformers' ``LinearAttentionLayer`` for ``slots``' rows of ``layer``: empty when none carries state.

        ``sel`` is a decode-graph bucket's device selector (the KV's ``_g_sel``): the rows are gathered through it, so a
        captured graph reads whichever slots the step loads into it, and every row is taken to carry state (each
        decoding sequence has run its prompt; a padding row reads its scratch slot). The layer must already be
        allocated (:meth:`allocated`): a graph cannot allocate the pool."""
        from transformers.cache_utils import LinearAttentionLayer
        lal = LinearAttentionLayer()
        if sel is not None:
            if layer not in self.conv:
                raise RuntimeError(f"linear layer {layer} has no pooled state yet; warm the pool before capturing a "
                                   "decode graph")
            self._fill(lal, layer, sel)
            return lal
        have = [self.has[s] for s in slots]
        if any(have):
            if not all(have):
                raise RuntimeError(f"linear layer {layer}: rows bound to slots {list(slots)} mix sequences with and "
                                   "without state -- a prompt's first chunk must run on its own")
            self._fill(lal, layer, torch.tensor(list(slots), device=self.conv[layer].device))
        return lal

    def _fill(self, lal, layer: int, idx) -> None:
        conv = self.conv[layer].index_select(0, idx)
        rec = self.rec[layer].index_select(0, idx)
        lal.lazy_initialization(conv_states=conv, conv_kernel_size=self.kernel[layer])
        lal.conv_states[0].copy_(conv)
        lal.lazy_initialization(recurrent_states=rec)
        lal.recurrent_states[0].copy_(rec)
        lal.has_previous_state[0] = True

    def allocated(self, layers) -> bool:
        """Every layer in ``layers`` has its pooled tensors (they are allocated on a layer's first store)."""
        return all(layer in self.conv for layer in layers)

    def store(self, layer: int, slots, lal, sel=None) -> None:
        """Write ``lal``'s updated state back to ``slots``' rows (through the device selector ``sel`` in a graph)."""
        conv, rec = lal.conv_states[0], lal.recurrent_states[0]
        if conv is None or rec is None:
            raise RuntimeError(f"linear layer {layer} produced no conv / recurrent state for slots {list(slots)}")
        if layer not in self.conv:
            if sel is not None:
                raise RuntimeError(f"linear layer {layer} has no pooled state yet; warm the pool before capturing a "
                                   "decode graph")
            self.conv[layer] = torch.zeros((self.n_slots, *conv.shape[1:]), dtype=conv.dtype, device=conv.device)
            self.rec[layer] = torch.zeros((self.n_slots, *rec.shape[1:]), dtype=rec.dtype, device=rec.device)
            self.kernel[layer] = int(lal.conv_kernel_size[0])
        idx = sel if sel is not None else torch.tensor(list(slots), device=conv.device)
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
        if self.frozen and self.conv:
            raise RuntimeError(f"the linear-state pool has {self.n_slots} slots and a captured decode graph holds its "
                               f"tensors' addresses; growing it to {n_slots} would leave the graph reading freed "
                               "memory -- size the first runner for the largest batch")
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


def _bucket_selector(ctx):
    """The KV's bound decode-graph bucket selector (``_g_sel``, the step's slot ids on device), when this forward is a
    bucketed decode step: the rows a captured graph addresses. None otherwise (prefill, the eager runner)."""
    if getattr(ctx, "mode", None) != "decode":
        return None
    sel = getattr(getattr(ctx, "kv", None), "_g_sel", None)
    if sel is None or sel.numel() != len(ctx.slots):
        return None
    return sel


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
                sel = _bucket_selector(ctx)
                # ``sel`` travels only on a bucketed graph step, so a caller that wraps view / store with their
                # original signatures (bench/p97's engagement counters) keeps working on every other path
                extra = {} if sel is None else {"sel": sel}
                lal = pool.view(_layer, ctx.slots, **extra)
                out = _orig(hidden_states, cache_params=_OneLayerCache(_layer, lal), attention_mask=attention_mask, **kw)
                pool.store(_layer, ctx.slots, lal, **extra)
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
