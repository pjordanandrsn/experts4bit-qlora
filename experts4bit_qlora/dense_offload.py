# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-layer residency for the NON-expert weights — the other half of a >VRAM MoE.

:mod:`experts4bit_qlora.nvme_experts` removed expert storage from the residency
budget entirely. For Kimi K3 that is 1.446 TB of 1.561 TB — 92.6% — but it leaves
**114.4 GB** of dense weights that must be resident regardless (measured across all
96 shard headers, 2026-07-30):

    attention        72.40 GB      shared experts   24.31 GB
    latent MoE       9.45 GB       embeddings        4.71 GB
    other + gates    3.52 GB       norms             0.01 GB

That will not fit a cheap card, and unlike experts it cannot be *tiered away*:
experts are top-16-of-896, so a token touches ~1.8% of them, while every token
touches all of the dense weights. Streaming them costs 107.3 GB/token against the
experts' 25.8 GB.

But it does not have to fit VRAM — only **pinned host RAM**, and a rented pod
exposes ~503 GB of that *independent of GPU count* (measured: 4x A40 returned the
same ceiling as 1x A5000). So the dense weights live pinned on the host and cross
PCIe per layer, with the next layer's copy overlapping this layer's compute.

**Bit-identity is preserved by construction, and that is the point.** A home is a
straight ``.to("cpu")`` copy of the loaded weight and staging is a same-dtype H2D
``copy_``. Nothing is quantized, cast, or re-derived — which is exactly what 4-bit
attention could not offer (measured on real K3 weights: NF4 leaves only 2.87% of
elements bit-identical, median relative error 10.4%, and the release deliberately
excludes attention from its own quant config).

Projected on the measured 19 GB/s host->device gather rate:

===================  ==============  ==========  =========
card(s)              dense streamed  s/token     $/hr spot
===================  ==============  ==========  =========
1x 4090 / A5000      95.2 GB         6.4         0.34
1x A40 / A6000       76.0 GB         5.4         0.44
2x A40 (pipeline)    37.6 GB         3.3         0.88
3x A40 (pipeline)    0 GB            1.4         1.32
===================  ==============  ==========  =========

At three cards every dense weight is resident and the remaining 1.4 s/token is the
expert floor. Use PIPELINE parallelism, not tensor: cheap pods have PCIe and no
NVLink, and PP passes only activations across a boundary.

One counterintuitive consequence worth knowing before tuning: dense traffic per
layer is FIXED so it amortizes perfectly over a batch, but expert traffic does not
— batch 256 routes to 99% of the expert set, so the tier stops helping. MoE weight
streaming favours SMALL batches (~4-16), the reverse of normal serving.
"""
from __future__ import annotations

import re

import torch

from .offload import _is_pinned, _placeholder, _prefetch_stream, _stats, _stats_enabled

# Default floor for what is worth moving. Below this a copy costs a launch and
# saves nothing: K3's norms/biases/A_log/dt_bias are 1-D and its conv1d kernels are
# [12288, 1, 4] f32 = 196 KB, so the whole "must stay resident" tail is 0.04 GB —
# 0.06% of the attention bucket. Excluding it is an efficiency call, not a
# correctness one; nothing here would be wrong if it were included.
MIN_BYTES = 1 << 20

_LAYER_RE = re.compile(r"(^|\.)layers\.\d+$")


def _is_expert_module(mod) -> bool:
    """Expert modules are somebody else's problem — `_ExpertOffload` owns the
    resident ones and `nvme_experts` leaves the tiered ones on `meta`."""
    if type(mod).__name__ in ("Experts4bit", "ExpertsNbit", "ExpertsLoRA",
                              "ExpertsMxfp4", "GptOssExperts4bit"):
        return True
    if getattr(mod, "_offload", None) is not None:
        return True
    return all(hasattr(mod, n) for n in
               ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"))


class _DenseOffload:
    """One decoder layer's dense weights, pinned on the host, streamed per forward.

    Deliberately a sibling of :class:`~experts4bit_qlora.offload._ExpertOffload`
    rather than a generalization of it: that class is built around four fixed
    expert tensor names and a routed-subset fast path, neither of which applies
    when every byte is needed every token.
    """

    # Keyed BY DEVICE. Pipeline parallelism puts layers on different GPUs, and a
    # single global slot would let a stage() on cuda:1 evict a layer on cuda:0 that
    # is still mid-pipeline. `_ExpertOffload` sidesteps this by refusing multi-device
    # outright (enable_inference_prefetch raises); this module is meant FOR the
    # multi-card case, so it tracks residency per device instead.
    _staged_now: dict = {}          # device -> set of handles
    _resident: dict = {}            # device -> handle

    @classmethod
    def _now(cls, dev) -> set:
        return cls._staged_now.setdefault(dev, set())

    def __init__(self, layer, device, *, pin: bool = True,
                 min_bytes: int = MIN_BYTES):
        self.layer = layer
        self.device = torch.device(device)
        self.pin = pin
        self.staged = False
        self.ready_event = None
        self._prefetch_next = None
        self._staged_dev = None
        # (module, attr, is_param, home) — home is a pinned CPU tensor holding the
        # EXACT loaded bytes.
        self.slots: list = []
        self._sd_keys: list = []
        self.bytes = 0
        for _name, mod in layer.named_modules():
            if _is_expert_module(mod):
                continue
            for store, is_param in ((mod._parameters, True), (mod._buffers, False)):
                for attr, t in list(store.items()):
                    if t is None or t.is_meta:
                        continue          # meta = served from the arena, not ours
                    nbytes = t.numel() * t.element_size()
                    if t.dim() < 2 or nbytes < min_bytes:
                        continue          # norms, biases, conv kernels: leave resident
                    home = t.detach().to("cpu")
                    if pin:
                        try:
                            home = home.pin_memory()
                        except (RuntimeError, AssertionError):
                            pass          # best-effort; pageable is correct, just sync
                    self.slots.append((mod, attr, is_param, home))
                    # key relative to the LAYER, for the state_dict hook below
                    self._sd_keys.append(f"{_name}.{attr}" if _name else attr)
                    self.bytes += nbytes
        self.pinned = all(_is_pinned(h) for _m, _a, _p, h in self.slots) if self.slots else True
        self._install_state_dict_hook()
        self.evict()                      # start evicted: the GPU copies just went away

    def _install_state_dict_hook(self) -> None:
        """Keep ``state_dict()`` correct while evicted.

        Between forwards these tensors are 0-element placeholders, so a naive
        ``state_dict()`` would silently serialize a model with **no attention
        weights** — a checkpoint that looks fine and is empty. Substitutes the
        pinned CPU homes by REFERENCE for any placeholder entry, so filtered saves
        (adapter-only, by key name) stay as cheap as before, and it is a no-op mid
        forward when the entries are the real device tensors.

        ``load_state_dict`` onto an evicted layer still fails loudly on the shape
        mismatch; that was never supported and is unchanged.
        """
        def hook(module, state_dict, prefix, local_metadata):
            for key, (_mod, _attr, _is_param, home) in zip(self._sd_keys, self.slots):
                full = prefix + key
                t = state_dict.get(full)
                if t is not None and t.numel() == 0:
                    state_dict[full] = home

        register = getattr(self.layer, "register_state_dict_post_hook", None)
        if register is None:      # older torch: same (mod, sd, prefix, meta) shape
            register = self.layer._register_state_dict_hook
        self._state_dict_hook_handle = register(hook)

    # ------------------------------------------------------------------ copy --
    def _copy_home_to_device(self, policy: str = "sync") -> None:
        dest = {}
        start = torch.cuda.Event(enable_timing=True) if _stats_enabled() else None
        if start is not None:
            start.record()
        for i, (_mod, _attr, _is_param, home) in enumerate(self.slots):
            d = torch.empty_like(home, device=self.device)
            d.copy_(home, non_blocking=_is_pinned(home))
            dest[i] = d
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            _stats().record_copy(start, end, self.bytes, len(self.slots), policy)
        self._staged_dev = dest
        self.staged = True

    def _bind(self) -> None:
        dest = self._staged_dev
        if dest is None:
            return
        # record_stream is not optional when the copy ran on the PREFETCH stream:
        # the caching allocator ties a block to the stream it was allocated on, so
        # without this it can hand the block to a later side-stream allocation
        # while the compute stream is still reading it — stale weights, no error.
        # `_ExpertOffload` does the same at its own bind sites.
        # `and self.slots`: no point querying a stream we will not mark anything on,
        # and it keeps _bind callable for a handle that selected no tensors.
        mark = self.device.type == "cuda" and bool(self.slots)
        cur = torch.cuda.current_stream(self.device) if mark else None
        for i, (mod, attr, is_param, _home) in enumerate(self.slots):
            t = dest[i]
            if mark:
                t.record_stream(cur)
            if is_param:
                mod._parameters[attr].data = t
            else:
                mod._buffers[attr] = t
        self._staged_dev = None

    def _consume_ready_event(self) -> None:
        evt = self.ready_event
        if evt is not None:
            if _stats_enabled():
                wait = torch.cuda.Event(enable_timing=True)
                wait.record()
                torch.cuda.current_stream().wait_event(evt)
                _stats().record_stall(wait, evt)
            else:
                torch.cuda.current_stream().wait_event(evt)
            self.ready_event = None
        self._bind()

    # ----------------------------------------------------------------- stage --
    def stage(self) -> None:
        """Single-slot synchronous staging — the conservative path, and the only
        one used when grad is enabled (training, checkpoint recompute).

        Two things here are load-bearing and were absent in the first version:

        * It sweeps **every** handle in ``_staged_now``, not just ``_resident``.
          A grad-enabled forward that follows an inference forward would otherwise
          inherit that forward's two residents and quietly exceed the single-slot
          bound this method is supposed to enforce.
        * It goes through :meth:`_consume_ready_event`, never ``_bind`` directly.
          A layer can arrive here already ``staged`` from a PREFETCH whose copy is
          still in flight on the side stream; binding without waiting on the event
          hands compute a partially-written weight.
        """
        cls = type(self)
        # Sweep BEFORE the already-bound early return. Otherwise a grad-enabled
        # stage() on a layer that inference left bound no-ops, and the sibling that
        # inference PREFETCHED stays resident — the single-slot bound this method
        # exists to restore is never restored. (Bugbot, PR #46.)
        now = cls._now(self.device)
        for h in list(now):
            if h is not self:
                h.evict()
        res = cls._resident.get(self.device)
        if res is not None and res is not self:
            res.evict()
        if self.staged and self._staged_dev is None and self.ready_event is None:
            cls._resident[self.device] = self
            now.add(self)
            return
        if not self.staged:
            self._copy_home_to_device("sync")
        self._consume_ready_event()
        cls._resident[self.device] = self
        now.add(self)

    def stage_for_inference(self) -> None:
        """Two-resident staging: make this layer usable now, then start the NEXT
        layer's copy on a side stream so it overlaps this layer's compute.

        Prefetch is PERFECT here, unlike for experts: the next layer's weight set
        is known without routing, so there is no speculation and no miss except a
        cold start."""
        cls = type(self)
        if not self.staged:
            if _stats_enabled():
                _stats().cold_misses += 1
            for h in list(cls._now(self.device)):
                if h is not self:
                    h.evict()
            res = cls._resident.get(self.device)
            if res is not None and res is not self:
                res.evict()
            self._copy_home_to_device("cold_miss")
            self._bind()
        else:
            self._consume_ready_event()
        cls._now(self.device).add(self)

        nxt = self._prefetch_next
        if nxt is not None and nxt is not self and not nxt.staged:
            stream = _prefetch_stream(nxt.device)
            with torch.cuda.stream(stream):
                nxt._copy_home_to_device("prefetch")
            evt = torch.cuda.Event(enable_timing=True) if _stats_enabled() else torch.cuda.Event()
            evt.record(stream)
            nxt.ready_event = evt
            cls._now(nxt.device).add(nxt)

    def evict(self) -> None:
        """Point every slot back at a shared 0-element placeholder, dropping the
        device copies so the allocator can reuse the memory. Idempotent."""
        cls = type(self)
        for mod, attr, is_param, home in self.slots:
            ph = _placeholder(self.device, home.dtype)
            if is_param:
                p = mod._parameters[attr]
                if p is not None:
                    p.data = ph
            else:
                mod._buffers[attr] = ph
        self._staged_dev = None
        self.staged = False
        self.ready_event = None
        cls._now(self.device).discard(self)
        if cls._resident.get(self.device) is self:
            cls._resident.pop(self.device, None)

    # ----------------------------------------------------------------- info --
    def home_bytes(self) -> int:
        return self.bytes

    def __repr__(self):
        return (f"<_DenseOffload {len(self.slots)} tensors "
                f"{self.bytes / 1e9:.2f} GB pinned={self.pinned} "
                f"staged={self.staged}>")


def decoder_layers(model):
    """``[(name, module)]`` for things named ``...layers.<i>``, in depth order.

    Matched by NAME rather than by class: the point is to be family-agnostic, and
    K3's block is `KimiDecoderLayer` under `trust_remote_code` — a name this
    package must not have to know."""
    out = [(n, m) for n, m in model.named_modules() if _LAYER_RE.search(n)]
    out.sort(key=lambda nm: int(nm[0].rsplit(".", 1)[1]))
    return out


def _layer_device(layer) -> "torch.device":
    """The device a layer's own weights live on — CUDA preferred.

    Resolved per layer so a pipelined model works: `device_map`-style sharding puts
    layers on different cards, and staging them all to one would put a layer's
    weights on a different device from its inputs.
    """
    for t in list(layer.parameters()) + list(layer.buffers()):
        if t is not None and not t.is_meta and t.device.type == "cuda":
            return t.device
    for t in list(layer.parameters()) + list(layer.buffers()):
        if t is not None and not t.is_meta:
            return t.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_dense_offload(model, device=None, *, pin: bool = True,
                         min_bytes: int = MIN_BYTES, prefetch: bool = True,
                         log=None) -> list:
    """Pin every decoder layer's dense weights on the host and stream them per layer.

    Returns the handles, also stashed on each layer as ``_dense_offload``. Pair with
    :func:`experts4bit_qlora.nvme_experts.enable_nvme_residency` (or the resident
    expert offload) — this function deliberately touches NOTHING inside an expert
    module, so the two compose without either knowing about the other.

    ``prefetch=True`` links each layer to the next so a ``no_grad`` forward keeps
    two layers resident and overlaps transfer with compute. The chain WRAPS, so
    after a full forward layer 0 is already warm for the next token — expect one
    layer staged at rest, not zero.

    Grad-enabled forwards take the single-slot synchronous path AND are not
    evicted afterwards, because backward still needs the weights. So a training
    step is correct but saves nothing; this module is for inference.
    """
    layers = decoder_layers(model)
    if not layers:
        raise ValueError(
            "no decoder layers found: expected modules named '...layers.<i>'. "
            "Pass a model whose blocks follow that convention, or offload by hand.")
    handles = []
    for _name, layer in layers:
        h = getattr(layer, "_dense_offload", None)
        if h is None:
            # device=None resolves PER LAYER. A pipelined model has its layers on
            # different GPUs, and one global device would stage every layer's
            # weights onto card 0 — wrong device for the layer's own inputs, not
            # merely wasteful. An explicit `device` still overrides, for the
            # single-card case and for tests.
            dev = device if device is not None else _layer_device(layer)
            h = _DenseOffload(layer, dev, pin=pin, min_bytes=min_bytes)
            layer._dense_offload = h

            def _pre(module, args, _h=h):
                # Prefetch only at inference, for the same reason the expert path
                # restricts it: a checkpoint recompute must re-stage identically,
                # and a no_grad forward of a module still in train() mode is the
                # *initial* reentrant-checkpoint forward, not inference.
                inference = not torch.is_grad_enabled() and not module.training
                if _h._prefetch_next is not None and inference:
                    _h.stage_for_inference()
                else:
                    _h.stage()

            layer.register_forward_pre_hook(_pre)
            layer.register_forward_hook(lambda m, a, o, _h=h: _h.evict())
            # Backward needs these weights again, and single-slot staging has
            # already evicted every layer but the last by the time the forward
            # returns — so without this, autograd fails on a 0-element placeholder
            # with a shape mismatch reported far from the cause. Re-staging here
            # keeps residency bounded (one layer, walking backwards) at the cost of
            # transferring each layer's weights twice per step, which is the same
            # trade activation checkpointing makes for activations.
            layer.register_full_backward_pre_hook(
                lambda m, grad_output, _h=h: _h.stage())
        handles.append(h)

    # Assigned UNCONDITIONALLY, so a later call with prefetch=False actually turns
    # prefetch off. Setting them only under `prefetch` left a second idempotent
    # call's links from the first call in place, and the hooks kept taking the
    # two-resident path while the caller had asked for the synchronous one.
    # Chains are PER DEVICE. A cross-device prefetch link would start a copy onto
    # the wrong card, and the side stream belongs to one device anyway.
    for h in handles:
        h._prefetch_next = None
    if prefetch:
        by_dev: dict = {}
        for h in handles:
            by_dev.setdefault(h.device, []).append(h)
        for chain in by_dev.values():
            if len(chain) > 1:
                for h, nxt in zip(chain, chain[1:] + chain[:1]):
                    h._prefetch_next = nxt

    total = sum(h.bytes for h in handles)
    unpinned = [i for i, h in enumerate(handles) if not h.pinned]
    if log is not None:
        log(f"  dense offload: {len(handles)} layers, "
            f"{sum(len(h.slots) for h in handles)} tensors, "
            f"{total / 1e9:.2f} GB pinned on the host "
            f"({total / len(handles) / 1e6:.0f} MB/layer)")
        if unpinned:
            log(f"  WARNING: {len(unpinned)} layer(s) could not pin their homes; "
                "H2D will be synchronous and prefetch buys nothing there")
    return handles


def dense_offload_report(handles) -> dict:
    """What is pinned, and what a token costs at a given link rate."""
    total = sum(h.bytes for h in handles)
    per_layer = total / len(handles) if handles else 0
    return {
        "layers": len(handles),
        "tensors": sum(len(h.slots) for h in handles),
        "host_bytes": total,
        "per_layer_bytes": int(per_layer),
        "all_pinned": all(h.pinned for h in handles),
        "staged_now": sum(len(v) for v in _DenseOffload._staged_now.values()),
        # s/token at the measured 19 GB/s host->device rate, weights only
        "seconds_per_token_at_19GBs": round(total / 19e9, 3),
    }
