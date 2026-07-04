"""Optional CPU-offloading of the frozen 4-bit expert weights.

When enabled (``OFFLOAD_EXPERTS=1``), each :class:`Experts4bit` base keeps its four big tensors —
the packed 4-bit ``gate_up_proj`` / ``down_proj`` weights and their ``gate_up_absmax`` /
``down_absmax`` scales — in **pinned CPU RAM** instead of on the GPU. A forward *pre*-hook on the
enclosing :class:`~experts4bit_qlora.lora.ExpertsLoRA` copies that layer's experts onto the GPU just
before its experts forward, and a forward *post*-hook drops the GPU copy right after. Because
training uses gradient checkpointing (``use_reentrant=False``), each decoder layer's forward is
recomputed in the backward pass — the *same* pre-hook re-stages the experts for the recompute.
(PyTorch stops that recompute *early*, as soon as the saved tensors are regenerated, so the evict
post-hook does **not** fire on the recompute; eviction during backward is instead driven by a
single-resident-slot policy — staging a layer first evicts the previously-staged one.) So only
**one layer's** experts are GPU-resident at any instant, in forward and backward alike.

This lets a fused-MoE whose 4-bit experts exceed VRAM (Qwen3-30B-A3B ~15 GB, Gemma-4-26B-A4B
~13 GB) QLoRA-train on a 12 GB card, at the cost of one host->device expert transfer per layer per
pass (a memory-for-compute trade — see ``docs/METHODOLOGY.md`` §11).

Why this is correct (and why the hook goes on ``ExpertsLoRA``, not ``ExpertsNbit``):

* ``ExpertsLoRA.forward`` never calls ``base.forward()``; it reads ``base.gate_up_proj`` etc.
  directly and calls ``base._project(...)``. A pre-hook on the base would never fire in training,
  so the hook must sit on the module whose ``__call__`` actually runs — ``ExpertsLoRA``.
* ``ExpertsNbit._project`` runs its projection through :class:`_FrozenLinearRecomputeBackward`,
  which **re-dequantizes the packed weight in backward** (it saves only the tiny packed buffers,
  not the full dequantized expert — the training-memory win). So backward *does* read the packed
  weight, and eviction is safe only where the weight is guaranteed resident at backward time.

  **Invariant: offload requires gradient checkpointing (``use_reentrant=False``), which the trainer
  always enables.** Under checkpointing, each decoder layer's forward is recomputed in backward;
  the pre-hook re-stages that layer's experts for the recompute, and the recompute Function's
  backward (the re-dequant) runs *within that layer's backward segment* — while the layer is still
  the single GPU-resident one, before the next layer to recompute stages and evicts it. So the
  backward re-dequant always reads a staged weight. (Without checkpointing, the post-hook would
  evict a layer right after its initial forward and the plain backward's re-dequant would read the
  evicted placeholder — so **non-checkpointed offload training is unsupported**. It cannot corrupt
  gradients silently: the read hits a 0-element placeholder and
  :class:`~experts4bit_qlora._vendor.experts._FrozenLinearRecomputeBackward` raises a pointed
  error naming this invariant. The shipped trainer always enables checkpointing.) The
  initial-forward post-hook evict remains safe because the weight is re-staged on the recompute
  before it is next needed.

  The eviction hazard is purely a *backward* construct: under ``no_grad`` there is no autograd tape
  and nothing re-reads a packed weight after its projection ran, so the inference-only GEMV route
  (``ExpertsLoRA._use_infer_gemv``) is exempt and safe under offload.

The tiny NF4 ``code`` buffer and the trainable LoRA adapters stay GPU-resident throughout.

Inference (``no_grad``) prefetch — :func:`enable_inference_prefetch`:

At decode there is no backward and no gradient-checkpoint recompute, so the staging schedule is
fully deterministic: layers run in order, and a layer's *entire* expert stack moves together
(layer-granular staging means no expert-choice prediction is needed, unlike expert-granular
prefetch systems). That makes double-buffering trivial: when layer ``L``'s pre-hook runs, it kicks
off layer ``L+1``'s H2D copy on a dedicated side stream, so the transfer overlaps ``L``'s compute
instead of serializing in front of ``L+1``. The links are circular — the last MoE layer prefetches
the first, which is exactly the next token's first need at decode. Residency is bounded at **two**
layers (the computing one + the one in flight): the post-hook still evicts each layer right after
its forward. Training forwards keep the single-slot synchronous path — the pre-hook requires both
``no_grad`` *and* the module in eval mode before taking the prefetch policy (reentrant gradient
checkpointing runs the initial *training* forward under ``no_grad``, so grad mode alone would
misclassify it), and a training staging sweeps out any prefetch leftovers.

Stream-safety notes (why this is correct): a prefetched copy is *consumed* only after the compute
stream waits on the copy's recorded CUDA event, and the copied tensors are ``record_stream``-marked
for the compute stream so the caching allocator cannot recycle their memory under in-flight kernels
after the post-hook evict. Evicting a still-in-flight prefetch (e.g. a training step interrupting
generation) is safe because frees are stream-ordered on the tensor's *allocation* stream — the side
stream the copy runs on.
"""

from __future__ import annotations

import os

import torch


def _stats_enabled() -> bool:
    """Per-copy transfer instrumentation (``E4B_OFFLOAD_STATS=1``, default off). Zero overhead and
    zero extra CUDA events on the hot path when off."""
    return os.environ.get("E4B_OFFLOAD_STATS", "0") == "1"


def _arena_enabled() -> bool:
    """Copy-consolidation (``E4B_OFFLOAD_ARENA=1``, default off): pack a layer's four homes into one
    contiguous pinned arena per dtype so staging is 1-2 ``copy_``s instead of 4. Correctness is
    identical (same bytes land in the same per-tensor views); only the number of H2D copies changes."""
    return os.environ.get("E4B_OFFLOAD_ARENA", "0") == "1"


class _OffloadStats:
    """Accumulates CUDA-event-bracketed copy timings, prefetch stall/slack, and cold-miss counts for
    :func:`offload_stats_report`. Events are recorded on the stream that carries each copy and are
    reduced with a single ``synchronize()`` at report time — never in the hot path (a sync there
    would perturb the very overlap being measured)."""

    def __init__(self):
        self.copies = []  # (start_evt, end_evt, nbytes, ncopies, policy)
        self.stalls = []  # (wait_marker_evt, copy_end_evt) — reduced to stall(+)/slack(-) at report
        self.cold_misses = 0

    def record_copy(self, start, end, nbytes, ncopies, policy):
        self.copies.append((start, end, nbytes, ncopies, policy))

    def record_stall(self, wait_marker, copy_end):
        self.stalls.append((wait_marker, copy_end))


_STATS: "_OffloadStats | None" = None


def _stats() -> _OffloadStats:
    global _STATS
    if _STATS is None:
        _STATS = _OffloadStats()
    return _STATS


def reset_offload_stats() -> None:
    """Drop accumulated stats (and their retained CUDA events). Called at the start of a timed
    region so a warmup pass doesn't pollute the measured numbers."""
    global _STATS
    _STATS = None


# 0-element GPU placeholders that an evicted base's parameters/buffers point at, shared across all
# offloaded layers (reads never mutate them, so sharing is safe) and cached per device. Keeping the
# real "home" data OFF the module — only these placeholders are registered while evicted — means
# ``model.to(device)`` never drags the big expert tensors to the GPU. ``state_dict()`` substitutes
# the CPU homes for the placeholders via a post-hook (see ``_install_state_dict_hook``) so a
# full-model save stays *correct* — a naive placeholder state_dict would silently serialize a model
# with no expert weights — while adapter-only saves (key-filtered) remain exactly as cheap.
_PLACEHOLDERS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _placeholders(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ph = _PLACEHOLDERS.get(device)
    if ph is None:
        ph = (
            torch.empty(0, dtype=torch.uint8, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
        _PLACEHOLDERS[device] = ph
    return ph


def _is_pinned(t: torch.Tensor) -> bool:
    """Whether ``t`` is in pinned memory (so ``non_blocking`` H2D is truly async). Robust on hosts
    where ``is_pinned`` is unavailable/raises without CUDA."""
    try:
        return bool(t.is_pinned())
    except Exception:
        return False


# One side stream per device for prefetch H2D copies, so a transfer can overlap the compute
# stream's kernels instead of serializing in front of them. (With pageable homes — pin=False or a
# failed pin_memory — the copy still lands correctly via stream ordering; it just blocks the host
# thread, so the overlap benefit is lost.)
_PREFETCH_STREAMS: dict[torch.device, "torch.cuda.Stream"] = {}


def _prefetch_stream(device: torch.device) -> "torch.cuda.Stream":
    s = _PREFETCH_STREAMS.get(device)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _PREFETCH_STREAMS[device] = s
    return s


class _ExpertOffload:
    """Owns the pinned-CPU home copies of one :class:`Experts4bit` base's four big tensors and
    streams them to ``device`` for the duration of each forward / gradient-checkpoint recompute.

    The base's own ``gate_up_proj`` / ``down_proj`` parameters and ``gate_up_absmax`` /
    ``down_absmax`` buffers hold shared 0-element GPU placeholders while evicted, so nothing that
    walks the module tree (``.to()``, ``state_dict()``, checkpoint-save) touches the offloaded data.
    """

    _NAMES_PARAM = ("gate_up_proj", "down_proj")  # nn.Parameter (uint8, packed 4-bit, frozen)
    _NAMES_BUFFER = ("gate_up_absmax", "down_absmax")  # float32 quantization scales (buffers)

    # The single handle whose experts are currently GPU-staged. Under use_reentrant=False gradient
    # checkpointing the backward RECOMPUTE re-runs the layer forward but PyTorch stops it early (once
    # the saved tensors are regenerated), so the evict *post*-hook does NOT fire on the recompute.
    # Relying on the post-hook alone would leave every recomputed layer's experts staged through the
    # whole backward — accumulating to the full footprint offload exists to avoid. So staging a new
    # layer first evicts this previously-staged one: at most one layer is GPU-resident at any instant,
    # in forward AND backward, regardless of whether the post-hook fired.
    _resident = None

    # Handles staged by the *inference prefetch* policy (class-level, like _resident): at most the
    # computing layer + the prefetched next layer. A training stage() sweeps this set so leftovers
    # from a generate() (the circularly-prefetched first layer) never linger into a train step.
    _staged_now: set = set()

    def __init__(self, base, device, pin: bool = True):
        self.base = base
        self.device = torch.device(device)
        # Inference-prefetch state (see enable_inference_prefetch): the handle staged after this
        # one in forward order, and the CUDA event marking this handle's in-flight prefetch copy.
        self._prefetch_next = None
        self.ready_event = None
        # Capture the homes as SEPARATE (pinned) CPU tensors BEFORE any placeholder swap. Capturing
        # the live Parameter object instead would alias the very tensor we later overwrite with a
        # placeholder — losing the weights. The source is on the GPU at load time, so ``.to("cpu")``
        # is a real device->host copy that decouples the home from the module.
        #
        # Arena mode (E4B_OFFLOAD_ARENA=1) packs the four homes into one contiguous pinned arena per
        # dtype; ``self.home[n]`` is then a correctly-shaped CPU *view* into its arena (so the
        # state_dict hook and any per-tensor reader see the same thing as the non-arena path), and
        # staging copies whole arenas (1-2 copies) instead of four separate tensors.
        self.home, self._arena_cpu, self._arena_layout = self._build_homes(base, pin, _arena_enabled())
        self._staged_dev = None  # arena mode: the device-side arena tensors kept alive while staged
        # True iff every home landed in pinned memory (so staging's non_blocking H2D is real); False
        # when pin=False or pin_memory fell back to pageable. Surfaced by the loader's summary log.
        self.pinned = all(_is_pinned(t) for t in self.home.values())
        # Precomputed for stats: total H2D bytes per stage, and how many copies a stage issues
        # (arena: one per dtype; non-arena: one per tensor).
        self._stage_nbytes = sum(t.numel() * t.element_size() for t in self.home.values())
        self._stage_ncopies = len(self._arena_cpu) if self._arena_layout is not None else len(self._names())
        self.staged = False
        self.evict()  # start evicted: base holds placeholders, ~0 GPU footprint (frees the load copy)
        self._install_state_dict_hook()

    def _install_state_dict_hook(self) -> None:
        """Keep full-model ``state_dict()`` correct while evicted. Between forwards the base's
        registered tensors are 0-element placeholders, so a naive ``state_dict()`` would silently
        serialize a model with **no expert weights**. This hook substitutes the (pinned) CPU home
        copies for any placeholder entries — as references, not copies, so adapter-only saves (which
        filter by key name and never match ``base.*``) stay exactly as cheap as before. While
        *staged* (mid-forward) the entries are the real GPU tensors and the hook is a no-op.
        Note ``load_state_dict`` onto an evicted model still fails loudly on the placeholder shape
        mismatch — loading into an offloaded model was never supported and is unchanged here."""

        def hook(module, state_dict, prefix, local_metadata):
            for n in self._names():
                key = prefix + n
                t = state_dict.get(key)
                if t is not None and t.numel() == 0:
                    state_dict[key] = self.home[n]

        register = getattr(self.base, "register_state_dict_post_hook", None)
        if register is None:  # older torch: the private hook has the same (mod, sd, prefix, meta) shape
            register = self.base._register_state_dict_hook
        self._state_dict_hook_handle = register(hook)

    @classmethod
    def _names(cls):
        return cls._NAMES_PARAM + cls._NAMES_BUFFER

    @staticmethod
    def _to_home(t: torch.Tensor, pin: bool) -> torch.Tensor:
        cpu = t.to("cpu")
        if pin:
            try:
                return cpu.pin_memory()
            except (RuntimeError, AssertionError):
                pass  # pinning is best-effort; pageable fallback is correct, just no async H2D
        return cpu

    @classmethod
    def _build_homes(cls, base, pin: bool, arena: bool):
        """Return ``(home, arena_cpu, layout)``. Non-arena: ``home[n]`` is an independent (pinned)
        CPU tensor, ``arena_cpu``/``layout`` are ``None`` — byte-identical to the pre-arena path.
        Arena: one contiguous pinned CPU tensor per dtype (``arena_cpu[dtype]``), ``home[n]`` a
        shaped view into it, and ``layout[n] = (dtype, start, stop, shape)`` for device re-viewing."""
        names = cls._names()
        srcs = {n: getattr(base, n).detach().to("cpu") for n in names}
        if not arena:
            return {n: cls._to_home(srcs[n], pin) for n in names}, None, None

        by_dtype: dict = {}
        for n in names:  # preserve _names() order within each dtype group
            by_dtype.setdefault(srcs[n].dtype, []).append(n)
        home, arena_cpu, layout = {}, {}, {}
        for dt, ns in by_dtype.items():
            total = sum(srcs[n].numel() for n in ns)
            buf = cls._to_home(torch.empty(total, dtype=dt), pin)  # pin the arena, then view it
            off = 0
            for n in ns:
                k = srcs[n].numel()
                buf[off : off + k].copy_(srcs[n].reshape(-1))
                home[n] = buf[off : off + k].view(srcs[n].shape)
                layout[n] = (dt, off, off + k, tuple(srcs[n].shape))
                off += k
            arena_cpu[dt] = buf
        return home, arena_cpu, layout

    def _copy_home_to_device(self, policy: str = "sync") -> None:
        """Enqueue this layer's H2D copies on the *current* stream and mark the handle staged. The
        destination tensors are allocated on that stream, so a later free is stream-ordered against
        the copies automatically. Non-arena issues four copies (one per tensor); arena issues one
        per dtype and re-views the params/buffers into the device arena. ``policy`` tags the copy for
        :func:`offload_stats_report` (``sync`` / ``cold_miss`` / ``prefetch``)."""
        stats = _stats() if _stats_enabled() else None
        if stats is not None:
            stream = torch.cuda.current_stream(self.device)
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)

        b = self.base
        if self._arena_layout is not None:
            dev = {dt: a.to(self.device, non_blocking=True) for dt, a in self._arena_cpu.items()}
            for n in self._NAMES_PARAM:
                dt, s, e, shape = self._arena_layout[n]
                b._parameters[n].data = dev[dt][s:e].view(shape)
            for n in self._NAMES_BUFFER:
                dt, s, e, shape = self._arena_layout[n]
                b._buffers[n] = dev[dt][s:e].view(shape)
            self._staged_dev = dev  # keep the device arenas alive (the param views also reference them)
        else:
            for n in self._NAMES_PARAM:
                b._parameters[n].data = self.home[n].to(self.device, non_blocking=True)
            for n in self._NAMES_BUFFER:
                b._buffers[n] = self.home[n].to(self.device, non_blocking=True)

        if stats is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(stream)
            stats.record_copy(start, end, self._stage_nbytes, self._stage_ncopies, policy)
        self.staged = True

    def _consume_ready_event(self) -> None:
        """Order the compute stream after this handle's in-flight prefetch copy (no-op if none).
        Also ``record_stream``-marks the copied tensors for the compute stream: they were allocated
        on the side stream, and without the mark the caching allocator could hand their memory to a
        new allocation as soon as the post-hook evict drops them — under kernels still in flight."""
        if self.ready_event is None:
            return
        s = torch.cuda.current_stream(self.device)
        if _stats_enabled():
            # Stall = time the compute stream waits for the copy. Marker on the compute stream at
            # wait-issue vs the copy's end event: positive elapsed = real stall (overlap too small),
            # negative = slack (copy already done — overlap succeeded).
            marker = torch.cuda.Event(enable_timing=True)
            marker.record(s)
            _stats().record_stall(marker, self.ready_event)
        s.wait_event(self.ready_event)
        self.ready_event = None
        b = self.base
        for n in self._NAMES_PARAM:
            b._parameters[n].data.record_stream(s)
        for n in self._NAMES_BUFFER:
            b._buffers[n].record_stream(s)

    def stage(self) -> None:
        """Copy the four big tensors onto ``device`` (idempotent), first evicting the previously
        staged layer so at most one layer's experts are GPU-resident (holds through the backward
        recompute, where the evict post-hook does not fire). The H2D copy is enqueued on the current
        stream, so the immediately-following dequant kernels are correctly ordered after it."""
        if self.staged:
            self._consume_ready_event()  # a prefetched copy may still be in flight on the side stream
            return
        cls = type(self)
        for h in list(cls._staged_now):  # sweep prefetch leftovers (e.g. a train step right after generate)
            if h is not self:
                h.evict()
        if cls._resident is not None and cls._resident is not self:
            cls._resident.evict()  # single-slot: free the prior layer before staging this one
        self._copy_home_to_device("sync")
        cls._resident = self

    def stage_for_inference(self) -> None:
        """Two-resident staging for ``no_grad`` forwards with prefetch links installed: make this
        layer's experts usable on the compute stream, then start the *next* layer's H2D copy on the
        side stream so it overlaps this layer's compute. Residency stays bounded at two — the
        post-hook evicts each layer right after its forward, and a cold start sweeps out stale
        residents (a training step's slot, or leftovers from an interrupted generation) first."""
        cls = type(self)
        if not self.staged:
            # Cold start / miss: nothing prefetched this layer — stage synchronously, like stage().
            if _stats_enabled():
                _stats().cold_misses += 1
            for h in list(cls._staged_now):
                if h is not self:
                    h.evict()
            if cls._resident is not None and cls._resident is not self:
                cls._resident.evict()
            self._copy_home_to_device("cold_miss")
        else:
            self._consume_ready_event()
        cls._staged_now.add(self)

        nxt = self._prefetch_next
        if nxt is not None and nxt is not self and not nxt.staged:
            # Key the side stream by the DESTINATION device: the stream context redirects only
            # that device's current stream, and the ready event must record on the stream that
            # actually carries the copy. (enable_inference_prefetch enforces a single device, so
            # today nxt.device == self.device; this keeps the invariant local.)
            stream = _prefetch_stream(nxt.device)
            with torch.cuda.stream(stream):
                nxt._copy_home_to_device("prefetch")
            # Timing-enabled only under stats (so _consume_ready_event can measure stall vs this
            # event); a plain non-timing event otherwise, exactly as before.
            evt = torch.cuda.Event(enable_timing=True) if _stats_enabled() else torch.cuda.Event()
            evt.record(stream)
            nxt.ready_event = evt
            cls._staged_now.add(nxt)

    def evict(self) -> None:
        """Point the four big tensors back at shared 0-element placeholders (idempotent), dropping
        the GPU copies so the caching allocator can reuse the memory for the next layer. Safe even
        for a still-in-flight prefetch: the dropped tensors were allocated on the side stream, so
        their reuse is stream-ordered after the pending copy."""
        ph_u8, ph_f32 = _placeholders(self.device)
        b = self.base
        for n in self._NAMES_PARAM:
            b._parameters[n].data = ph_u8
        for n in self._NAMES_BUFFER:
            b._buffers[n] = ph_f32
        self._staged_dev = None  # drop our reference to the device arena(s); the param views are now placeholders
        self.staged = False
        self.ready_event = None
        cls = type(self)
        cls._staged_now.discard(self)
        if cls._resident is self:
            cls._resident = None


def enable_expert_offload(experts_lora, device, pin: bool = True) -> _ExpertOffload:
    """Offload one :class:`ExpertsLoRA`'s frozen 4-bit base to (pinned) CPU RAM and install the
    stream-in/evict hooks.

    Registers a forward pre-hook (stage the base onto ``device``) and a forward post-hook (evict it)
    on ``experts_lora`` — the module whose ``__call__`` runs on every forward *and* on the
    gradient-checkpoint recompute — and stashes the handle on ``experts_lora._offload`` so it stays
    alive with the module. Returns the handle.

    Idempotent: if ``experts_lora`` is already offloaded, the existing handle is returned unchanged
    (``device`` / ``pin`` are ignored). This is load-bearing, not a convenience — while evicted the
    base's registered tensors are 0-element placeholders, so a second handle would capture *those*
    as its CPU homes (losing the weights) and stack a second pair of stage/evict hooks.
    """
    existing = getattr(experts_lora, "_offload", None)
    if existing is not None:
        return existing
    base = getattr(experts_lora, "base", None)
    if base is None or not all(hasattr(base, n) for n in _ExpertOffload._names()):
        raise TypeError(
            "enable_expert_offload expects an ExpertsLoRA wrapping an Experts4bit base "
            f"(gate_up_proj/down_proj/gate_up_absmax/down_absmax); got {type(experts_lora).__name__}"
        )
    handle = _ExpertOffload(base, device, pin=pin)
    experts_lora._offload = handle

    def _stage_pre_hook(module, args):
        # Prefetch policy only at inference (no autograd tape, deterministic layer order). Any
        # grad-enabled forward — training, checkpoint recompute — takes the single-slot sync path,
        # and so does a no_grad forward of a module still in train() mode: reentrant gradient
        # checkpointing (use_reentrant=True) runs the *initial* training forward under
        # torch.no_grad(), which the grad-mode test alone would misread as inference.
        if handle._prefetch_next is not None and not torch.is_grad_enabled() and not module.training:
            handle.stage_for_inference()
        else:
            handle.stage()

    experts_lora.register_forward_pre_hook(_stage_pre_hook)
    experts_lora.register_forward_hook(lambda module, args, output: handle.evict())
    return handle


def enable_inference_prefetch(handles) -> list[_ExpertOffload]:
    """Enable deterministic next-layer expert prefetch across these offloaded layers at inference.

    ``handles`` must be the offload handles in **forward (layer) order** — as returned by the
    streaming loader or :func:`offload_model_experts`, both of which walk layers in order. Links
    them circularly (the last MoE layer prefetches the first — the next decode step's first need)
    and switches their pre-hooks to the two-resident policy for ``no_grad`` forwards **of modules
    in eval mode**; grad-enabled forwards — and no_grad forwards of a train()-mode module, e.g. a
    reentrant-checkpoint initial forward — are untouched (single-slot sync staging). Requires all
    offload handles on one CUDA device — the overlap comes from a per-device side stream.
    Idempotent; returns the handles.
    """
    handles = list(handles)
    if not handles:
        return handles
    if any(h.device.type != "cuda" for h in handles):
        raise RuntimeError("enable_inference_prefetch requires CUDA offload devices (got a non-CUDA handle)")
    if len({h.device for h in handles}) > 1:
        raise RuntimeError("enable_inference_prefetch requires all offload handles on a single CUDA device")
    if len(handles) == 1:
        return handles  # nothing to overlap with: a lone MoE layer keeps the sync staging path
    for h, nxt in zip(handles, handles[1:] + handles[:1]):
        h._prefetch_next = nxt
    return handles


def offload_stats_report(log=None) -> "dict | None":
    """Reduce the accumulated per-copy stats to a summary (``E4B_OFFLOAD_STATS=1``). Returns ``None``
    when stats are off or nothing was staged. Synchronizes once here (never in the hot path) to make
    the retained CUDA events readable, then reports, per policy: GB moved, copy count, mean per-copy
    ms, implied GB/s; plus total prefetch stall vs slack ms and the cold-miss count. ``implied GB/s``
    against the pinned ceiling (see :func:`report_offload_environment`) is the A-workstream number."""
    stats = _STATS
    if stats is None or not stats.copies:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    by_policy: dict = {}
    for start, end, nbytes, ncopies, policy in stats.copies:
        ms = start.elapsed_time(end)
        agg = by_policy.setdefault(policy, {"ms": 0.0, "bytes": 0, "stages": 0, "copies": 0})
        agg["ms"] += ms
        agg["bytes"] += nbytes
        agg["stages"] += 1
        agg["copies"] += ncopies

    stall_ms = slack_ms = 0.0
    for marker, copy_end in stats.stalls:
        d = marker.elapsed_time(copy_end)  # +: compute waited (stall); -: copy already done (slack)
        if d >= 0:
            stall_ms += d
        else:
            slack_ms += -d

    report = {"by_policy": {}, "stall_ms": stall_ms, "slack_ms": slack_ms, "cold_misses": stats.cold_misses}
    for policy, a in sorted(by_policy.items()):
        gb = a["bytes"] / 1e9
        gbps = gb / (a["ms"] / 1e3) if a["ms"] > 0 else 0.0
        report["by_policy"][policy] = {
            "gb": gb,
            "stages": a["stages"],
            "copies": a["copies"],
            "mean_stage_ms": a["ms"] / a["stages"],
            "gbps": gbps,
        }

    if log is not None:
        log("offload transfer stats (E4B_OFFLOAD_STATS):")
        for policy, r in report["by_policy"].items():
            log(
                f"  {policy:<10} {r['gb']:.2f} GB in {r['stages']} stages / {r['copies']} copies "
                f"| {r['mean_stage_ms']:.2f} ms/stage | {r['gbps']:.2f} GB/s"
            )
        log(f"  prefetch stall {stall_ms:.1f} ms | slack {slack_ms:.1f} ms | cold misses {stats.cold_misses}")
    return report


def report_offload_environment(device, log, ceiling_mb: int = 256, iters: int = 20) -> "dict | None":
    """One-shot PCIe link + H2D-ceiling report (``E4B_OFFLOAD_STATS=1``). Names the bus every
    per-layer ``GB/s`` should be read against, and warns if the link negotiated below its max width
    (a chipset x4/x8 slot produces exactly the reduced bandwidth the offload figures reflect).
    Fail-soft: any probe that raises is skipped, never breaks a load."""
    dev = torch.device(device)
    out: dict = {}

    try:  # PCIe link status via nvidia-smi (fail-soft: absent in many containers)
        import subprocess

        q = "name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            row = r.stdout.strip().splitlines()[0]
            out["pcie"] = row
            log(f"offload env: PCIe {row}")
            parts = [p.strip() for p in row.split(",")]
            if len(parts) == 5:
                gcur, gmax, wcur, wmax = parts[1:]
                if wcur.split("x")[-1] != wmax.split("x")[-1] or gcur != gmax:
                    log(
                        f"  WARNING: link negotiated below max (gen {gcur}/{gmax}, width {wcur}/{wmax}) "
                        "— reduced H2D bandwidth is expected; this bounds offload/prefetch throughput"
                    )
    except Exception as e:
        log(f"offload env: PCIe probe unavailable ({type(e).__name__})")

    if dev.type == "cuda" and torch.cuda.is_available():  # H2D ceiling micro-bench (pinned + pageable)
        try:
            n = ceiling_mb * 1024 * 1024
            dst = torch.empty(n, dtype=torch.uint8, device=dev)
            for kind, pin in (("pinned", True), ("pageable", False)):
                src = torch.empty(n, dtype=torch.uint8)
                if pin:
                    src = src.pin_memory()
                for _ in range(3):  # warm the path
                    dst.copy_(src, non_blocking=pin)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    dst.copy_(src, non_blocking=pin)
                e.record()
                torch.cuda.synchronize()
                gbps = iters * n / 1e9 / (s.elapsed_time(e) / 1e3)
                out[f"ceiling_{kind}_gbps"] = gbps
                log(f"offload env: {kind} H2D ceiling {gbps:.2f} GB/s ({ceiling_mb} MB x {iters})")
        except Exception as e:
            log(f"offload env: ceiling micro-bench unavailable ({type(e).__name__})")
    return out or None


def offload_model_experts(model, device=None, pin: bool = True) -> list[_ExpertOffload]:
    """Offload every :class:`ExpertsLoRA` in ``model`` to (pinned) CPU RAM.

    Convenience for the already-loaded / test path. The streaming loader does **not** use this — it
    offloads each layer inside its per-layer loop so the experts never all sit on the GPU at once
    (a post-load pass would require every layer GPU-resident first, defeating the purpose). ``device``
    defaults to the device of the first offloadable base found. Already-offloaded modules keep their
    existing handle (see :func:`enable_expert_offload`), so calling this on a model the loader
    offloaded is a safe no-op.
    """
    from .lora import ExpertsLoRA

    handles = []
    for module in model.modules():
        if isinstance(module, ExpertsLoRA):
            dev = device if device is not None else module.base.gate_up_proj.device
            handles.append(enable_expert_offload(module, dev, pin=pin))
    return handles
