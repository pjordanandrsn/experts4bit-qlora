# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-layer residency for the NON-expert weights — the other half of a >VRAM MoE.

:mod:`experts4bit_qlora.engines.nvme_experts` removed expert storage from the residency
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
import warnings

import torch

from ..formats.dense_disk import DiskHome
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

    Deliberately a sibling of :class:`~experts4bit_qlora.engines.offload._ExpertOffload`
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
                 min_bytes: int = MIN_BYTES, source=None, key_prefix: str = "",
                 verify: bool = False, skip_trainable: bool = False):
        """``source``: a :class:`~experts4bit_qlora.formats.dense_disk.DenseDiskSource`. When
        given, a tensor whose checkpoint key is present there gets a
        :class:`DiskHome` instead of a pinned host copy — same bytes, read on demand,
        so the host-RAM floor drops from the whole dense side of the model to one
        staging buffer. ``key_prefix`` + the layer-relative key must equal the
        checkpoint key. ``verify=True`` compares every disk home against the loaded
        tensor bit-for-bit at construction; cheap on a truncated model and the right
        gate to run once per checkpoint. ``skip_trainable=True`` keeps every trainable
        streamable parameter resident instead of streaming it. The default, ``False``, is
        the old selection. :func:`enable_dense_offload` passes the value it decided over the
        whole model: ``True`` iff some streamable parameter is frozen."""
        self.layer = layer
        self.device = torch.device(device)
        self.pin = pin
        self.staged = False
        self.ready_event = None
        self._prefetch_next = None
        self._staged_dev = None
        self._train = None          # DQ3 training schedule (enable_dense_offload's train_prefetch); None = off
        self._train_idx = None
        # (module, attr, is_param, home) — home is a pinned CPU tensor holding the
        # EXACT loaded bytes.
        self.slots: list = []
        self._sd_keys: list = []
        self.bytes = 0
        self.placed = 0        # small tensors moved onto self.device
        self.host_bytes = 0    # homes held in host RAM
        self.disk_bytes = 0    # homes served from a DenseDiskSource
        self.verified = 0      # disk homes checked bit-for-bit at construction
        self.kept_trainable = 0        # trainable params that would have streamed, kept resident (skip_trainable)
        self.kept_trainable_bytes = 0
        self.streamed_trainable = 0    # trainable params selected for streaming (skip_trainable off)
        self.streamed_trainable_bytes = 0
        for _name, mod in layer.named_modules():
            if _is_expert_module(mod):
                continue
            for store, is_param in ((mod._parameters, True), (mod._buffers, False)):
                for attr, t in list(store.items()):
                    if t is None or t.is_meta:
                        continue          # meta = served from the arena, not ours
                    nbytes = t.numel() * t.element_size()
                    keep_trainable = (skip_trainable and is_param and t.requires_grad
                                      and t.dim() >= 2 and nbytes >= min_bytes)
                    if keep_trainable:
                        # A TRAINABLE parameter beside frozen ones (a LoRA matrix: PEFT's lora_B for a 25600-wide
                        # projection is 1.6 MB, over MIN_BYTES) stays resident. Streamed, eviction swaps it for an
                        # empty placeholder and the optimizer steps a 0-element tensor against a full grad -- "The
                        # size of tensor a (0) must match the size of tensor b (16)" from AdamW (DQ3 rehearsal).
                        self.kept_trainable += 1
                        self.kept_trainable_bytes += nbytes
                    if t.dim() < 2 or nbytes < min_bytes or keep_trainable:
                        # norms, biases, conv kernels: too small to be worth moving
                        # per layer. But "leave resident" has to mean resident ON
                        # THIS DEVICE — if the caller staged the layer's weights to
                        # CPU (the sensible way to avoid a full-dense VRAM peak),
                        # leaving these behind puts a 1-D norm on CPU against CUDA
                        # activations and the layer dies on a device mismatch deep
                        # in the model's own code. Found on the first real K3
                        # integration, not by reasoning.
                        if t.device != self.device:
                            moved = t.to(self.device)
                            if is_param and t.requires_grad:
                                # IN PLACE for a trainable parameter: an optimizer built before this call holds THIS
                                # object, and a re-wrapped Parameter would leave it stepping the stale CPU copy --
                                # training that silently does nothing. Any grad already accumulated moves with it.
                                # (Optimizer STATE from steps taken before the move stays where it was: build the
                                # optimizer before offloading if you like, but step it after.)
                                t.data = moved
                                if t.grad is not None:
                                    t.grad = t.grad.to(self.device)
                            elif is_param:
                                store[attr] = torch.nn.Parameter(
                                    moved, requires_grad=t.requires_grad)
                            else:
                                store[attr] = moved
                            self.placed += 1
                        continue
                    key = f"{_name}.{attr}" if _name else attr
                    home = None
                    if source is not None:
                        full = key_prefix + key
                        loc = source.tensors.get(full)
                        if loc is not None:
                            if tuple(loc.shape) != tuple(t.shape) or loc.dtype != t.dtype:
                                raise ValueError(
                                    f"dense_disk: {full} is {tuple(loc.shape)}/"
                                    f"{loc.dtype} on disk but {tuple(t.shape)}/"
                                    f"{t.dtype} in the model — refusing to serve it")
                            home = DiskHome(source, full)
                            if verify and not torch.equal(home.view_now(),
                                                          t.detach().to("cpu")):
                                raise ValueError(
                                    f"dense_disk: {full} on disk differs from the "
                                    f"loaded tensor; the source is not this model's "
                                    f"checkpoint")
                            self.disk_bytes += nbytes
                            self.verified += int(verify)
                    if home is None:
                        home = t.detach().to("cpu")
                        if pin:
                            try:
                                home = home.pin_memory()
                            except (RuntimeError, AssertionError):
                                pass      # best-effort; pageable is correct, just sync
                        self.host_bytes += nbytes
                    self.slots.append((mod, attr, is_param, home))
                    if is_param and t.requires_grad:
                        self.streamed_trainable += 1
                        self.streamed_trainable_bytes += nbytes
                    # key relative to the LAYER, for the state_dict hook below
                    self._sd_keys.append(key)
                    self.bytes += nbytes
        # A DiskHome is not a tensor and is never "pinned" in this sense; report on
        # the host-resident homes only, so `all_pinned` keeps meaning what it says.
        # A layer with none has no pinning to report: None, not a vacuous True.
        _ram = [h for _m, _a, _p, h in self.slots if isinstance(h, torch.Tensor)]
        self.pinned = all(_is_pinned(h) for h in _ram) if _ram else None
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
                    # A disk home must be MATERIALIZED. Its view aliases the source's
                    # shared staging buffer, which the next fetch overwrites — saving
                    # the view would serialize whatever layer happened to be staged
                    # last, which is the same class of silent-empty-checkpoint bug
                    # this hook exists to prevent.
                    state_dict[full] = (home.materialize()
                                        if isinstance(home, DiskHome) else home)

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
            if isinstance(home, DiskHome):
                # Read into the source's shared staging buffer, then copy
                # SYNCHRONOUSLY. The next fetch overwrites those bytes, so an async
                # copy would race a later read into the same buffer — stale weights,
                # no error, exactly the failure mode `record_stream` exists to stop on
                # the other path. Synchronous is also what makes ONE staging buffer
                # sufficient under prefetch: by the time the next layer is read, this
                # layer's copy has already landed.
                #
                # The overlap given up is small: a layer is ~1.2 GB, which is a few
                # hundred ms of disk against ~60 ms of PCIe, so the read dominates.
                src = home.view_now()
                d = torch.empty(src.shape, dtype=src.dtype, device=self.device)
                d.copy_(src)
            else:
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
            # current_stream(SELF.DEVICE), not the thread's current device. Under
            # pipeline parallelism the thread's current device is whatever ran last,
            # so a bare current_stream() makes the WRONG stream wait and this layer's
            # compute proceeds against an in-flight copy. (Bugbot, PR #46.)
            compute = torch.cuda.current_stream(self.device)
            if _stats_enabled():
                wait = torch.cuda.Event(enable_timing=True)
                wait.record(compute)
                compute.wait_event(evt)
                _stats().record_stall(wait, evt)
            else:
                compute.wait_event(evt)
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
        if self.device.type != "cuda":
            self.stage()
            return
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


# ------------------------------------------------------------ train prefetch --
# DQ3 (experts4bit-qlora#1083, bench/dq3/DESIGN-dq3.md): the overlapped training path. On by default for CUDA chains
# since DQ3/DQ5 (#1188, #1218); with ``enable_dense_offload(..., train_prefetch=False)`` none of what follows is
# constructed or reached.
#
# Under non-reentrant gradient checkpointing a training step uses the layers in the order 0..L-1 (forward), then
# L-1..0 (backward: each layer's recompute, and its full-backward pre-hook, which may repeat the same index). Without
# checkpointing only the backward pre-hooks run in the backward, in the same descending order. Both are covered by one
# rule: keep the layer in use and ONE prefetch target resident, and copy the target on the prefetch stream while the
# layer in use computes.

def train_schedule(last, phase, i, n):
    """Pure. Given the previous use (``last``, ``phase``) and this use of layer ``i`` of ``n``, return
    ``(phase, target)``: ``phase`` is ``"fwd"``, ``"bwd"`` or ``"unscheduled"``, and ``target`` the layer to prefetch
    (or None).

    * Forward: target ``i + 1``; at the last layer, ``n - 2`` -- the backward starts there right after the loss.
    * Backward: target ``i - 1``; at layer 0, ``1`` -- the next step's forward starts with 0 (still resident), then 1.
    * A repeat of the same index continues the current phase, except at the last layer in the forward, where the repeat
      is the backward's recompute (the turnaround).
    * Anything else is out of order: ``"unscheduled"`` -- the caller stages synchronously and prefetches nothing.
    """
    if n <= 0 or not 0 <= i < n:
        return "unscheduled", None
    if last is None:
        phase = "fwd"
    elif phase == "fwd":
        if i == last + 1:
            phase = "fwd"
        elif i == last:
            phase = "bwd" if i == n - 1 else "fwd"
        elif i == last - 1 and last == n - 1:
            phase = "bwd"
        elif i == 0:
            phase = "fwd"
        else:
            return "unscheduled", None
    elif phase == "bwd":
        if i in (last, last - 1):
            phase = "bwd"
        elif i == last + 1 or i == 0:
            phase = "fwd"
        else:
            return "unscheduled", None
    else:                                     # after an unscheduled use: restart from here
        phase = "fwd"
    if phase == "fwd":
        target = i + 1 if i + 1 < n else (n - 2 if n > 1 else None)
    else:
        target = i - 1 if i > 0 else (1 if n > 1 else None)
    return phase, target


class _TrainPrefetch:
    """One device's chain of handles, the schedule's state, and the counters the DQ3 receipt reads."""

    def __init__(self, handles):
        self.handles = list(handles)
        self.last = None
        self.phase = None
        self.counts = {"uses": 0, "resident_hits": 0, "unscheduled": 0, "hwm_resident": 0}
        for ph in ("fwd", "bwd"):
            for k in ("prefetch_issued", "overlapped", "waited", "blocking"):
                self.counts[f"{ph}_{k}"] = 0

    def use(self, h) -> None:
        """Make ``h`` usable now, keep only it and the scheduled target resident, and start the target's copy."""
        cls = type(h)
        n = len(self.handles)
        i = h._train_idx
        phase, target = train_schedule(self.last, self.phase, i, n)
        self.counts["uses"] += 1
        tag = phase if phase in ("fwd", "bwd") else "fwd"
        if phase == "unscheduled":
            self.counts["unscheduled"] += 1
        # 1) this layer: consume an in-flight prefetch, keep a bound one, or copy synchronously
        if h.staged and h.ready_event is not None:
            done = bool(h.ready_event.query())
            self.counts[f"{tag}_{'overlapped' if done else 'waited'}"] += 1
            h._consume_ready_event()
        elif h.staged and h._staged_dev is None:
            self.counts["resident_hits"] += 1
        else:
            self.counts[f"{tag}_blocking"] += 1
            if not h.staged:
                h._copy_home_to_device("sync")
            h._consume_ready_event()
        cls._resident[h.device] = h
        cls._now(h.device).add(h)
        # 2) residency: this layer and the target, nothing else on this device
        keep = {h}
        nxt = self.handles[target] if (target is not None and phase != "unscheduled") else None
        if nxt is not None:
            keep.add(nxt)
        for other in list(cls._now(h.device)):
            if other not in keep:
                other.evict()
        # 3) the target's copy, on the prefetch stream, fenced at bind by record_stream (see _bind)
        # Only a CUDA target is copied ahead; on any other device the target is staged when it is used (synchronous,
        # still correct), which is also what lets the schedule and parity tests run on a CPU-only CI runner.
        if nxt is not None and nxt is not h and not nxt.staged and nxt.device.type == "cuda":
            stream = _prefetch_stream(nxt.device)
            with torch.cuda.stream(stream):
                nxt._copy_home_to_device("prefetch")
            evt = torch.cuda.Event()
            evt.record(stream)
            nxt.ready_event = evt
            cls._now(nxt.device).add(nxt)
            self.counts[f"{tag}_prefetch_issued"] += 1
        self.counts["hwm_resident"] = max(self.counts["hwm_resident"], len(cls._now(h.device)))
        self.last, self.phase = i, (phase if phase != "unscheduled" else "fwd")
        if phase == "unscheduled":
            self.last = None


# ------------------------------------------------------------- late-bound 4-bit backward --
# bitsandbytes 0.50's MatMul4Bit keeps the frozen packed weight for backward as a ctx ATTRIBUTE
# (``ctx.tensors = (None, B)``, B a view of ``Linear4bit.weight``) whenever the input needs grad. A ctx
# attribute bypasses save_for_backward, so checkpointing's saved-tensor hooks never see it: every layer's
# weight STORAGE stays referenced by the autograd graph from its forward to its backward, and eviction's
# ``p.data = placeholder`` frees nothing. Offloaded training therefore saved no VRAM (DQ3, experts4bit-qlora
# #1083: dq3-5090-3's streamed arm OOMed above the resident arm's footprint). The fix is the expert side's
# ``_FrozenLinearRecomputeBackward`` pattern: keep the MODULE, read the weight bound at backward time.
# The Function and the forward below mirror these four bitsandbytes 0.50.2 sources EXACTLY (the grad path only:
# Linear4bit.forward's CPU AVX-512 weight-packing branch is inference-only, never reached on the grad path, and is
# left to the stock forward that every no-grad call takes); pinned by sha256 of
# ``inspect.getsource`` (not by version: a patch release can change them under the same 0.50.x). Any mismatch keeps
# stock bnb, with a warning -- correct, just no VRAM saved in training.
_BNB_MIRRORED_SOURCES = {
    "MatMul4Bit.forward": "29f4e21a4fe99db3f6e0f94dbf6cbe8426220f9163c0501c86d33f362a54ccb6",
    "MatMul4Bit.backward": "053e8719319a526c61f288a37221ba5200de9eaa20dae7c7e45386315f284cd2",
    "matmul_4bit": "a219d40d62264de60a424e8f9f1bab7656136eb7264a575d39d57a7e2f25ed4a",
    "Linear4bit.forward": "d52cf63457717cc718ca0a449dad0967171e679be90d5d2adcafe2e8a141093d",
}


def _bnb_mirror_mismatches() -> list:
    """Names of the mirrored bitsandbytes sources whose sha256 differs from the pinned 0.50.2 bytes ([] = all match)."""
    import hashlib
    import inspect

    from bitsandbytes.autograd._functions import MatMul4Bit, matmul_4bit
    from bitsandbytes.nn.modules import Linear4bit
    objs = {"MatMul4Bit.forward": MatMul4Bit.forward, "MatMul4Bit.backward": MatMul4Bit.backward,
            "matmul_4bit": matmul_4bit, "Linear4bit.forward": Linear4bit.forward}
    out = []
    for name, want in _BNB_MIRRORED_SOURCES.items():
        try:
            got = hashlib.sha256(inspect.getsource(objs[name]).encode()).hexdigest()
        except (OSError, TypeError):
            got = None
        if got != want:
            out.append(name)
    return out


class _LateBoundMatMul4Bit(torch.autograd.Function):
    """``bnb.matmul_4bit`` against a frozen ``Linear4bit`` that reads the packed weight at BACKWARD time.

    Forward: ``bnb.matmul_4bit`` with grad disabled (a Function's forward always is), i.e. matmul_4bit's no-grad
    branch -- the same ``torch.ops.bitsandbytes.gemm_4bit`` call with the same arguments as MatMul4Bit.forward,
    nested double-quant included. Backward: MatMul4Bit.backward's expressions, on ``mod.weight`` as bound when the
    backward runs (dense offload re-stages the layer, byte-identical, before its backward). Nothing that holds the
    weight's storage is kept -- only the module. Bitwise identical to stock bnb by construction."""

    @staticmethod
    def forward(ctx, A, bias, mod, quant_state):
        import bitsandbytes as bnb
        ctx.mod, ctx.state = mod, quant_state
        ctx.dtype_bias = None if bias is None else bias.dtype
        return bnb.matmul_4bit(A, mod.weight, bias=bias, quant_state=quant_state)

    @staticmethod
    def backward(ctx, grad_output):
        import bitsandbytes.functional as BF
        req_grad_a, req_grad_bias = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_a = grad_bias = None
        if req_grad_bias:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)       # MatMul4Bit.backward's expression
        if req_grad_a:
            w = ctx.mod.weight
            if w.numel() == 0:
                raise RuntimeError(
                    "dense offload: a 4-bit projection's backward read an EVICTED weight (0-element placeholder). "
                    "The layer must be staged before its backward runs -- the backward pre-hook and the checkpoint "
                    "recompute do this; a hook removed or bypassed would land here.")
            grad_a = torch.matmul(grad_output, BF.dequantize_4bit(w.view(-1, 1), ctx.state).to(grad_output.dtype))
        return grad_a, grad_bias, None, None


def _late_bound_linear4bit_forward(self, x):
    """``bnb.nn.Linear4bit.forward`` (0.50.2, source-pinned) with the grad path routed through :class:`_LateBoundMatMul4Bit`.
    Anything without a grad to compute (inference, eval, no_grad, CPU packing) takes the stock class forward."""
    bias = self.bias
    if not (torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad))):
        return type(self).forward(self, x)
    from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module
    fix_4bit_weight_quant_state_from_module(self)
    quant_state = self.weight.quant_state
    if not self.compute_type_is_set:
        self.set_compute_type(x)
        self.compute_type_is_set = True
    inp_dtype = x.dtype
    if self.compute_dtype is not None:
        x = x.to(self.compute_dtype)
    if bias is not None:
        if bias.dtype != x.dtype:
            bias.data = bias.data.to(x.dtype)
        bias = bias.to(self.compute_dtype)
    return _LateBoundMatMul4Bit.apply(x, bias, self, quant_state).to(inp_dtype)


def _install_late_bound_backward(handles) -> int:
    """Route every offloaded ``bnb.nn.Linear4bit`` weight's grad-mode matmul through :class:`_LateBoundMatMul4Bit`.
    Idempotent; returns how many modules are routed. Without it, offloaded training saves no VRAM (see above)."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return 0
    mods = []
    for h in handles:
        for mod, attr, is_param, _home in h.slots:
            if is_param and attr == "weight" and isinstance(mod, bnb.nn.Linear4bit):
                mods.append(mod)
    if not mods:
        return 0
    bad = _bnb_mirror_mismatches()
    if bad:
        warnings.warn(
            f"dense offload: bitsandbytes {getattr(bnb, '__version__', '?')} differs from the 0.50.2 code the late-bound "
            f"4-bit backward mirrors ({', '.join(bad)}); keeping stock bnb, whose MatMul4Bit keeps each layer's weight "
            "alive until backward, so offloaded TRAINING saves no VRAM here (inference is unaffected)", stacklevel=3)
        return 0
    n = 0
    for mod in mods:
        if getattr(mod, "_dense_offload_late_bound", False):
            n += 1
            continue
        mod.forward = _late_bound_linear4bit_forward.__get__(mod, type(mod))
        mod._dense_offload_late_bound = True
        n += 1
    return n


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
                         source=None, key_prefix: str = "", verify: bool = False,
                         log=None, train_prefetch: bool | None = None) -> list:
    """Pin every decoder layer's dense weights on the host and stream them per layer.

    Returns the handles, also stashed on each layer as ``_dense_offload``. Pair with
    :func:`experts4bit_qlora.engines.nvme_experts.enable_nvme_residency` (or the resident
    expert offload) — this function deliberately touches NOTHING inside an expert
    module, so the two compose without either knowing about the other.

    ``prefetch=True`` links each layer to the next so a ``no_grad`` forward keeps
    two layers resident and overlaps transfer with compute. The chain WRAPS, so
    after a full forward layer 0 is already warm for the next token — expect one
    layer staged at rest, not zero.

    **Freeze the base first** (``model.requires_grad_(False)``, then add any adapters).
    Trainable parameters are selected by the model's shape. If any streamable
    parameter (2-D, ``>= min_bytes``) in the decoder layers is frozen, as with a
    QLoRA/PEFT model or a partial freeze, every trainable one stays resident:
    streamed, the optimizer would step an evicted placeholder. If none is frozen
    (an unfrozen model), the selection is unchanged and the trainable tensors
    stream. That is correct for inference, but an optimizer stepping them fails,
    so a warning names them. Either way, kept or streamed trainable tensors are
    reported by a warning.

    Grad-enabled forwards take the single-slot synchronous path; the backward
    pre-hook (or the checkpoint recompute) re-stages each layer before its backward.
    For bitsandbytes ``Linear4bit`` projections the grad-mode matmul goes through a
    late-bound backward (:class:`_LateBoundMatMul4Bit`) so an evicted weight is
    really freed between forward and backward: bnb 0.50.2's own MatMul4Bit keeps it
    on ``ctx`` until backward, which made offloaded training save nothing (bnb
    behaviour, worked around locally; not reported upstream).

    ``train_prefetch`` (lane DQ3, ``bench/dq3/``) replaces that for a model in ``train()`` mode: each training use of
    a layer keeps it and ONE scheduled neighbour resident and copies the neighbour on the prefetch stream while the
    layer computes -- forward i+1, backward (checkpoint recompute or backward pre-hook) i-1 (:func:`train_schedule`).
    Bitwise-identical results are the contract; the per-device counters are in :func:`dense_offload_report` under
    ``train_prefetch``. The default, ``None``, turns it on for every CUDA device chain and off elsewhere (on a non-CUDA
    device nothing is copied ahead, so it would only add bookkeeping). ``True`` forces it on every chain, ``False``
    keeps the synchronous single-slot path. On by default since DQ3 and DQ5: on an RTX 5090 at 2048 tokens it trained
    bitwise identically at 1.0023x the resident step time on PCIe gen 5 x16 and 1.0050x on gen 4 x16, against 1.18x
    and 1.51x synchronous (``bench/dq3/RESULTS-dq3.md``, ``bench/dq5/RESULTS-dq5.md``).

    Memory: DQ4 (``bench/dq4/RESULTS-dq4.md``) read the streamed arm's longest trainable sequence (chunked LM loss)
    at 2.00x the resident one under the default CUDA allocator, and 2.375x with
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``: under the default allocator the longest fitting sequence left
    6.22 GiB reserved but unallocated (5.69 GiB on DQ6's 24 GB RTX 4090, ``bench/dq6/RESULTS-dq6.md``). That setting
    is the caller's (it must be set before CUDA initialises); nothing here sets it.

    Use it when the dense (non-expert) side of the model is what does not fit: every decoder
    layer's dense weights are pinned on the host and streamed per layer. Returns the list of
    handles (assert it is non-empty). Composes with the expert residency engines. See
    ``docs/solutions/run-moe-larger-than-vram.md``.
    """
    layers = decoder_layers(model)
    if not layers:
        raise ValueError(
            "no decoder layers found: expected modules named '...layers.<i>'. "
            "Pass a model whose blocks follow that convention, or offload by hand.")
    # Decided over the whole call, not per layer: a fully trainable layer in an otherwise frozen model is a partial
    # fine-tune, and streaming its weights breaks the optimizer just the same.
    skip_trainable = any(
        not p.requires_grad and p.dim() >= 2 and p.numel() * p.element_size() >= min_bytes
        for _n, layer in layers for mod in layer.modules() if not _is_expert_module(mod)
        for p in mod._parameters.values() if p is not None and not p.is_meta)
    if any(getattr(layer, "_e4b_ckpt_offload_ref", None) is not None for _n, layer in layers):
        warnings.warn("[e4b.dense_offload] these decoder layers already run e4b's reentrant checkpoint "
                      "(E4B_CKPT_OFFLOAD under enable_fast_train). Dense offload's "
                      "train-prefetch schedule is untested with that reentrant checkpoint: set E4B_CKPT_OFFLOAD=0 before "
                      "enable_fast_train, or call engines.ckpt_offload.disable_checkpoint_offload(model) first.",
                      RuntimeWarning, stacklevel=2)
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
            # The checkpoint key is `key_prefix` + this layer's module path + the
            # layer-relative key. `decoder_layers` already gives the module path, and
            # the caller supplies whatever sits above the module it handed us (e.g.
            # "language_model.model." when called on `model.language_model.model`).
            h = _DenseOffload(layer, dev, pin=pin, min_bytes=min_bytes,
                              source=source,
                              key_prefix=f"{key_prefix}{_name}." if source else "",
                              verify=verify, skip_trainable=skip_trainable)
            layer._dense_offload = h

            def _pre(module, args, _h=h):
                # Prefetch only at inference, for the same reason the expert path
                # restricts it: a checkpoint recompute must re-stage identically,
                # and a no_grad forward of a module still in train() mode is the
                # *initial* reentrant-checkpoint forward, not inference.
                inference = not torch.is_grad_enabled() and not module.training
                if _h._train is not None and module.training:
                    _h._train.use(_h)          # DQ3 train_prefetch: scheduled, overlapped
                elif _h._prefetch_next is not None and inference:
                    _h.stage_for_inference()
                else:
                    _h.stage()

            def _post(module, args, output, _h=h):
                # Under DQ3's train_prefetch a training forward does NOT evict: residency is
                # bounded by the schedule (this layer + one target), and evicting here
                # would throw away the layer the backward's recompute starts with.
                if _h._train is not None and module.training:
                    return
                _h.evict()

            def _bwd_pre(module, grad_output, _h=h):
                if _h._train is not None and module.training:
                    _h._train.use(_h)
                else:
                    _h.stage()

            layer.register_forward_pre_hook(_pre)
            layer.register_forward_hook(_post)
            # Backward needs these weights again, and single-slot staging has
            # already evicted every layer but the last by the time the forward
            # returns — so without this, autograd fails on a 0-element placeholder
            # with a shape mismatch reported far from the cause. Re-staging here
            # keeps residency bounded (one layer, walking backwards) at the cost of
            # transferring each layer's weights twice per step, which is the same
            # trade activation checkpointing makes for activations.
            layer.register_full_backward_pre_hook(_bwd_pre)
        handles.append(h)

    late_bound = _install_late_bound_backward(handles)

    # Assigned UNCONDITIONALLY, so a later call with prefetch=False actually turns
    # prefetch off. Setting them only under `prefetch` left a second idempotent
    # call's links from the first call in place, and the hooks kept taking the
    # two-resident path while the caller had asked for the synchronous one.
    # Chains are PER DEVICE. A cross-device prefetch link would start a copy onto
    # the wrong card, and the side stream belongs to one device anyway.
    for h in handles:
        h._prefetch_next = None
        h._train = None
        h._train_idx = None
    if train_prefetch is not False:
        by_dev_t: dict = {}
        for h in handles:
            by_dev_t.setdefault(h.device, []).append(h)
        for dev, chain in by_dev_t.items():
            if train_prefetch is None and torch.device(dev).type != "cuda":
                continue                   # default: CUDA chains only (nothing is copied ahead elsewhere)
            sched = _TrainPrefetch(chain)
            for idx, h in enumerate(chain):
                h._train = sched
                h._train_idx = idx
    if prefetch:
        by_dev: dict = {}
        for h in handles:
            by_dev.setdefault(h.device, []).append(h)
        for chain in by_dev.values():
            if len(chain) > 1:
                for h, nxt in zip(chain, chain[1:] + chain[:1]):
                    h._prefetch_next = nxt

    kept = sum(h.kept_trainable for h in handles)
    streamed_tr = sum(h.streamed_trainable for h in handles)
    if kept:
        msg = (f"dense offload: kept {kept} trainable tensor(s) resident "
               f"({sum(h.kept_trainable_bytes for h in handles) / 1e9:.2f} GB) instead of streaming them -- an "
               "optimizer cannot step a streamed parameter. For inference, freeze the model first: "
               "model.requires_grad_(False)")
        warnings.warn(msg, stacklevel=2)
        if log is not None:
            log("  WARNING: " + msg)
    if streamed_tr:
        msg = (f"dense offload: streaming {streamed_tr} trainable tensor(s) "
               f"({sum(h.streamed_trainable_bytes for h in handles) / 1e9:.2f} GB) of an unfrozen model -- fine for "
               "inference, but an optimizer stepping them fails after eviction. Freeze for inference "
               "(model.requires_grad_(False)); for training, freeze the base")
        warnings.warn(msg, stacklevel=2)
        if log is not None:
            log("  WARNING: " + msg)
    total = sum(h.bytes for h in handles)
    host = sum(h.host_bytes for h in handles)
    disk = sum(h.disk_bytes for h in handles)
    # Only host-resident homes can fail to pin. Counting disk homes here reported
    # every layer as unpinned-capable and, worse, called disk bytes "pinned".
    unpinned = [i for i, h in enumerate(handles) if h.host_bytes and not h.pinned]
    if log is not None:
        where = []
        if host:
            where.append(f"{host / 1e9:.2f} GB pinned on the host")
        if disk:
            where.append(f"{disk / 1e9:.2f} GB served from disk (0 resident)")
        log(f"  dense offload: {len(handles)} layers, "
            f"{sum(len(h.slots) for h in handles)} tensors, "
            f"{' + '.join(where) or 'nothing managed'} "
            f"({total / len(handles) / 1e6:.0f} MB/layer)"
            + (f"; late-bound 4-bit backward on {late_bound} projections" if late_bound else ""))
        if unpinned:
            log(f"  WARNING: {len(unpinned)} layer(s) could not pin their homes; "
                "H2D will be synchronous and prefetch buys nothing there")
    return handles


def dense_offload_report(handles) -> dict:
    """What is pinned, and what a token costs at a given link rate.

    ``all_pinned`` is asked of the host-resident layers only: ``True`` or
    ``False`` when ``host_resident_layers > 0``, and ``None`` when there are none
    (every dense home served from disk, or no handles), since nothing is pinned
    or could have failed to pin.
    """
    total = sum(h.bytes for h in handles)
    per_layer = total / len(handles) if handles else 0
    host = sum(h.host_bytes for h in handles)
    disk = sum(h.disk_bytes for h in handles)
    host_resident = [h for h in handles if h.host_bytes]
    return {
        "layers": len(handles),
        "tensors": sum(len(h.slots) for h in handles),
        # `host_bytes` is what is actually RESIDENT on the host. Disk-served homes
        # are counted separately: reporting them as host bytes would hide the very
        # thing the disk path exists to remove.
        "host_bytes": host,
        "disk_bytes": disk,
        "verified_bit_exact": sum(h.verified for h in handles),
        "per_layer_bytes": int(per_layer),
        # Pinning is a property of HOST-resident homes. Asking it of a disk-served
        # handle is a category error, and `all(...)` over an empty set of them
        # answered True -- reporting "all_pinned" about a path that pins nothing.
        # With no host-resident layer the question has no answer: None.
        "host_resident_layers": len(host_resident),
        "all_pinned": (all(h.pinned for h in host_resident)
                       if host_resident else None),
        "staged_now": sum(len(v) for v in _DenseOffload._staged_now.values()),
        # s/token at the measured 19 GB/s host->device rate -- HOST bytes only. Disk
        # homes do not ride PCIe from pinned RAM; they are bounded by the device the
        # checkpoint sits on, which is one to two orders of magnitude slower. Costing
        # them at a PCIe rate said 5.7 s/token for a Kimi K3 run that measured 91.8.
        "seconds_per_token_at_19GBs": round(host / 19e9, 3),
        "disk_bytes_per_token": disk,
        # DQ3 counters, per device chain; None when train_prefetch is off (False, or the default on a non-CUDA device).
        # offloaded bnb Linear4bit projections whose grad-mode matmul is late-bound (0 = stock bnb, e.g. a source mismatch)
        "late_bound_4bit": sum(1 for h in handles for mod, _a, _p, _hm in h.slots
                               if getattr(mod, "_dense_offload_late_bound", False)),
        "train_prefetch": (None if not any(h._train is not None for h in handles) else
                           {str(dev): dict(sched.counts) for dev, sched in
                            {h.device: h._train for h in handles if h._train is not None}.items()}),
    }


def offload_plan(layers, *, pin: bool = True, train_prefetch: bool = True, min_bytes: int = MIN_BYTES,
                 skip_trainable: bool | None = None) -> dict:
    """What :func:`enable_dense_offload` would stream, pin and keep on the device for ``layers``, priced without building it.

    ``layers``: one entry per decoder layer, each a sequence of ``(nbytes, ndim, trainable)`` or ``(nbytes, ndim, trainable,
    is_param)`` for every parameter and buffer the layer holds outside expert modules -- the tensors a handle walks (``is_param``
    defaults to True; pass False for a buffer). The selection is the handle's: a tensor streams when it is at least 2-D and
    ``min_bytes``, unless it is a trainable parameter and ``skip_trainable`` holds. ``skip_trainable=None`` decides it as
    :func:`enable_dense_offload` does, over the whole model: True iff some streamable PARAMETER is frozen (a frozen buffer does
    not count, so pass ``is_param=False`` for buffers or a full fine-tune beside a large buffer is priced wrong).

    Returns bytes:

    * ``streamed``: the host homes, the tensors' exact bytes;
    * ``host_reserved``: what pinning them reserves -- each pinned request rounds up to a power of two (PyTorch's caching host
      allocator, grouped-nf4-gemm#71: ``recipe._pinned_cost``), 1.1355x on Qwen3-32B's NF4 layers as DQ3 measured; equal to
      ``streamed`` with ``pin=False``;
    * ``resident_slots``: device memory for the layers staged at once -- two under ``train_prefetch`` (the layer in use and
      its scheduled neighbour), one on the synchronous path -- times the largest layer's streamed bytes;
    * ``stays_on_device``: tensors never streamed (1-D, under ``min_bytes``, or trainable beside frozen ones);
    * ``link_per_microbatch``: host-to-device bytes per micro-batch, every streamed layer copied for its forward and again
      for its backward (an upper bound: DQ3 counted 62 + 62 copies over 64 layers, the boundary layers staying staged);
    * ``largest_layer``, and ``layers`` (the count).

    Quantization state that lives outside parameters and buffers (a bitsandbytes ``Params4bit``'s ``quant_state``) is not
    walked by a handle and stays on the device; price it with the weights it belongs to.
    """
    from ..recipe import _pinned_cost

    tensors = [[(e[0], e[1], e[2], e[3] if len(e) > 3 else True) for e in layer] for layer in layers]

    def streamable(n, d):
        return d >= 2 and n >= min_bytes

    if skip_trainable is None:
        skip_trainable = any(streamable(n, d) and not tr and param for layer in tensors for n, d, tr, param in layer)
    per_layer, reserved, stays = [], 0, 0
    for layer in tensors:
        s = 0
        for n, d, tr, _param in layer:
            if streamable(n, d) and not (skip_trainable and tr):
                s += int(n)
                reserved += _pinned_cost(n) if pin else int(n)
            else:
                stays += int(n)
        per_layer.append(s)
    largest = max(per_layer, default=0)
    return {"layers": len(per_layer), "streamed": sum(per_layer), "host_reserved": reserved,
            "resident_slots": (2 if train_prefetch else 1) * largest, "stays_on_device": stays,
            "link_per_microbatch": 2 * sum(per_layer), "largest_layer": largest}


def late_bound_4bit_refusal() -> str | None:
    """Why offloaded training of bitsandbytes ``Linear4bit`` layers would free no VRAM here, or None when it frees what
    :func:`offload_plan` prices.

    An evicted ``Linear4bit`` weight is released only through the late-bound backward (:class:`_LateBoundMatMul4Bit`), which
    mirrors four bitsandbytes 0.50.2 sources pinned by sha256 (``_BNB_MIRRORED_SOURCES``). With any of them different,
    :func:`enable_dense_offload` keeps stock bitsandbytes, whose autograd holds every layer's packed weight from its forward to
    its backward (DQ3: dq3-5090-3's streamed arm ran out of memory above the resident one)."""
    try:
        mismatched = _bnb_mirror_mismatches()
    except ImportError as e:
        return f"bitsandbytes is not importable ({e})"
    if not mismatched:
        return None
    from importlib.metadata import PackageNotFoundError, version

    try:
        have = version("bitsandbytes")
    except PackageNotFoundError:
        have = "?"
    return (f"bitsandbytes {have} differs from the 0.50.2 sources the late-bound backward mirrors "
            f"({', '.join(mismatched)}): offloaded Linear4bit weights stay referenced from forward to backward, so "
            "streaming them frees no VRAM in training")
