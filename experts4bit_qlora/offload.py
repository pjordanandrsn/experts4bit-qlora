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

Choosing a staging policy for inference — read this before reaching for prefetch:

Two policies exist and they are **mutually exclusive**. Measured on Qwen3-235B-A22B
(E=128, top_k=8, 94 layers) over a 22.5 GB/s pinned link:

===========================  ==========  =========================================
policy                       s/token     what it does
===========================  ==========  =========================================
bulk (default)                    5.57    copies a layer's WHOLE expert stack
bulk + prefetch                   5.14    same bytes, overlapped with compute
``enable_routed_staging``         0.93    copies only the experts that routed
routed + ``enable_fast``          0.71    ...and fuses the per-expert loop
===========================  ==========  =========================================

**Routed staging is the recommended policy for streamed inference** and is worth
**5.95x**. The reason is that bulk staging moves ``E/top_k`` = **16x** the bytes the
routing actually needs, so overlapping that traffic (prefetch, worth 1.11x) is
optimizing the wrong term. Routed staging is **bit-identical** to bulk — verified
on the real 235B and at flagship per-layer shape — so it is a pure win where it
applies.

It is off by default because it changes what crosses the link, and because it
**cannot be combined with prefetch**: layer ``L+1``'s routing is decided by a router
reading layer ``L``'s output, so there is nothing to prefetch. That exclusion is
structural, not a tuning preference, and :func:`enable_routed_staging` raises
rather than silently degrading. Prefill and training fall back to bulk
automatically.

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
# offloaded layers (reads never mutate them, so sharing is safe) and cached per (device, dtype) —
# quantized schemes evict to uint8/float32, passthrough schemes to their storage dtype, so an
# evicted module's tensors keep honest dtypes. Keeping the real "home" data OFF the module — only
# these placeholders are registered while evicted — means ``model.to(device)`` never drags the big
# expert tensors to the GPU. ``state_dict()`` substitutes the CPU homes for the placeholders via a
# post-hook (see ``_install_state_dict_hook``) so a full-model save stays *correct* — a naive
# placeholder state_dict would silently serialize a model with no expert weights — while
# adapter-only saves (key-filtered) remain exactly as cheap.
_PLACEHOLDERS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ph = _PLACEHOLDERS.get((device, dtype))
    if ph is None:
        ph = torch.empty(0, dtype=dtype, device=device)
        _PLACEHOLDERS[(device, dtype)] = ph
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
    """Owns the pinned-CPU home copies of one :class:`ExpertsNbit` base's big tensors (the two
    packed projections plus, for quantized schemes, their two absmax scales — passthrough bf16/fp16
    bases have no absmax) and streams them to ``device`` for the duration of each forward /
    gradient-checkpoint recompute.

    The base's own ``gate_up_proj`` / ``down_proj`` parameters and ``gate_up_absmax`` /
    ``down_absmax`` buffers hold shared 0-element GPU placeholders while evicted, so nothing that
    walks the module tree (``.to()``, ``state_dict()``, checkpoint-save) touches the offloaded data.
    """

    _NAMES_PARAM = ("gate_up_proj", "down_proj")  # nn.Parameter (packed storage, frozen)
    _NAMES_BUFFER = ("gate_up_absmax", "down_absmax")  # float32 quantization scales (buffers);
    # registered as None on passthrough (bf16/fp16) bases — the instance tuples below skip those.

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
        # Routed-only staging (see enable_routed_staging): copy just the experts a
        # forward actually routes to instead of the layer's whole stack. Off by
        # default — it changes what crosses the link, so it is opt-in.
        self._routed_only = False
        self._routed_max = None      # above this many routed experts, bulk is cheaper
        # Which staging path last ran, and how many bytes it moved. The module's
        # own tensors cannot answer this after the fact — the post-hook evicts them
        # to 0-element placeholders as soon as the forward returns.
        self._last_stage_policy = None
        self._last_stage_nbytes = 0
        # Speculative prefetch state (see prefetch_speculative / enable_speculative_staging).
        self._spec_dev = None
        self._spec_ids = None
        self._spec_event = None
        self._spec_hits = self._spec_misses = self._spec_total = 0
        self._pool = None
        self._pool_layer = 0
        # Offload only the tensors this base actually carries: passthrough (bf16/fp16) schemes
        # register their absmax buffers as None — those stay None throughout (never swapped for a
        # placeholder), so `base.gate_up_absmax is None` remains a valid passthrough test while
        # offloaded, evicted, or staged.
        self._param_names = self._NAMES_PARAM
        self._buffer_names = tuple(n for n in self._NAMES_BUFFER if getattr(base, n, None) is not None)
        # Capture the homes as SEPARATE (pinned) CPU tensors BEFORE any placeholder swap. Capturing
        # the live Parameter object instead would alias the very tensor we later overwrite with a
        # placeholder — losing the weights. The source is on the GPU at load time, so ``.to("cpu")``
        # is a real device->host copy that decouples the home from the module.
        #
        # Arena mode (E4B_OFFLOAD_ARENA=1) packs the four homes into one contiguous pinned arena per
        # dtype; ``self.home[n]`` is then a correctly-shaped CPU *view* into its arena (so the
        # state_dict hook and any per-tensor reader see the same thing as the non-arena path), and
        # staging copies whole arenas (1-2 copies) instead of four separate tensors.
        self.home, self._arena_cpu, self._arena_layout = self._build_homes(
            base, self._param_names + self._buffer_names, pin, _arena_enabled()
        )
        self._staged_dev = None  # arena mode: the device-side arena tensors kept alive while staged
        # True iff every home landed in pinned memory (so staging's non_blocking H2D is real); False
        # when pin=False or pin_memory fell back to pageable. Surfaced by the loader's summary log.
        self.pinned = all(_is_pinned(t) for t in self.home.values())
        # Precomputed for stats: total H2D bytes per stage, and how many copies a stage issues
        # (arena: one per dtype; non-arena: one per tensor).
        self._stage_nbytes = sum(t.numel() * t.element_size() for t in self.home.values())
        self._stage_ncopies = len(self._arena_cpu) if self._arena_layout is not None else len(self.home)
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
            for n in self._param_names + self._buffer_names:
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
    def _build_homes(cls, base, names, pin: bool, arena: bool):
        """Return ``(home, arena_cpu, layout)`` for the given tensor ``names`` (the instance's
        param+buffer tuples — passthrough bases carry no absmax). Non-arena: ``home[n]`` is an
        independent (pinned) CPU tensor, ``arena_cpu``/``layout`` are ``None`` — byte-identical to
        the pre-arena path. Arena: one contiguous pinned CPU tensor per dtype (``arena_cpu[dtype]``),
        ``home[n]`` a shaped view into it, and ``layout[n] = (dtype, start, stop, shape)`` for
        device re-viewing."""
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
        the copies automatically. Non-arena issues one copy per home tensor; arena issues one per
        dtype and re-views the params/buffers into the device arena. ``policy`` tags the copy for
        :func:`offload_stats_report` (``sync`` / ``cold_miss`` / ``prefetch``)."""
        stats = _stats() if _stats_enabled() else None
        if stats is not None:
            stream = torch.cuda.current_stream(self.device)
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)

        b = self.base
        if self._arena_layout is not None:
            dev = {dt: a.to(self.device, non_blocking=True) for dt, a in self._arena_cpu.items()}
            for n in self._param_names:
                dt, s, e, shape = self._arena_layout[n]
                b._parameters[n].data = dev[dt][s:e].view(shape)
            for n in self._buffer_names:
                dt, s, e, shape = self._arena_layout[n]
                b._buffers[n] = dev[dt][s:e].view(shape)
            self._staged_dev = dev  # keep the device arenas alive (the param views also reference them)
        else:
            for n in self._param_names:
                b._parameters[n].data = self.home[n].to(self.device, non_blocking=True)
            for n in self._buffer_names:
                b._buffers[n] = self.home[n].to(self.device, non_blocking=True)

        if stats is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(stream)
            stats.record_copy(start, end, self._stage_nbytes, self._stage_ncopies, policy)
        self._last_stage_policy = policy
        self._last_stage_nbytes = self._stage_nbytes
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
        for n in self._param_names:
            b._parameters[n].data.record_stream(s)
        for n in self._buffer_names:
            b._buffers[n].record_stream(s)

    def stage(self) -> None:
        """Copy the home tensors onto ``device`` (idempotent), first evicting the previously
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
        self._release_speculation()
        self._copy_home_to_device("sync")
        cls._resident = self

    def _alloc_dest(self):
        """Full-shaped device tensors, only some rows of which will be written."""
        return {n: torch.empty(self.home[n].shape, dtype=self.home[n].dtype,
                               device=self.device)
                for n in self._param_names + self._buffer_names}

    def _copy_rows_into(self, dest, ids) -> int:
        """Copy rows ``ids`` into ``dest`` — from the device pool when cached,
        from the pinned home otherwise. Returns bytes that crossed the LINK
        (pool hits cost device-to-device bandwidth, which is ~50x cheaper and is
        not what the transfer-bound step is bound by)."""
        E = self.base.num_experts
        names = self._param_names + self._buffer_names
        pool = getattr(self, "_pool", None)
        views = {n: (self.home[n].view(E, -1), dest[n].view(E, -1)) for n in names}
        nbytes = 0
        for e in ids:
            e = int(e)
            slot = pool.get((self._pool_layer, e)) if pool is not None else None
            if slot is not None:
                for n in names:                       # device -> device
                    views[n][1][e].copy_(pool.buf[n][slot], non_blocking=True)
                continue
            for n in names:                           # host -> device, over the link
                hv, dv = views[n]
                dv[e].copy_(hv[e], non_blocking=True)
                nbytes += hv.shape[1] * self.home[n].element_size()
            if pool is not None:
                pool.put((self._pool_layer, e), {n: views[n][1][e] for n in names})
        return nbytes

    def _bind(self, dest) -> None:
        b = self.base
        for n in self._param_names:
            b._parameters[n].data = dest[n]
        for n in self._buffer_names:
            b._buffers[n] = dest[n]
        self._staged_dev = None
        self.staged = True

    def prefetch_speculative(self, pred_ids) -> None:
        """Stage PREDICTED experts on the side stream, before the true set exists.

        Correctness does not depend on the prediction: the destination keeps the
        full ``[E, ...]`` shape, and :meth:`stage_routed` copies whatever the
        router actually asked for and did not find here. A wrong guess costs
        bandwidth, never an answer.
        """
        if self.staged or self._spec_dev is not None:
            return
        dest = self._alloc_dest()
        stream = _prefetch_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._copy_rows_into(dest, pred_ids)
            for d in dest.values():
                d.record_stream(torch.cuda.current_stream(self.device))
        self._spec_event = torch.cuda.Event()
        self._spec_event.record(stream)
        self._spec_ids = set(int(e) for e in pred_ids)
        self._spec_dev = dest

    def _release_speculation(self) -> None:
        """Drop an unconsumed speculative buffer. Any staging path that does not
        go through the speculative branch must call this, or the evict guard
        pins the buffer for the rest of the run."""
        self._spec_dev = None
        self._spec_ids = None
        self._spec_event = None

    def _copy_routed_to_device(self, ids) -> None:
        """Copy ONLY the rows in ``ids`` into a full-shaped device tensor.

        The destination keeps the home's full ``[E, ...]`` shape even though only a
        few rows are filled, so **every consumer indexes by the original expert id
        and nothing downstream changes** — ``ExpertsLoRA.forward`` walks
        ``expert_hit``, and the grouped kernel is handed explicit ``expert_ids``;
        both touch exactly the routed rows. The unrouted rows hold uninitialized
        memory and are never read.

        Delegates the row copies to :meth:`_copy_rows_into` so the expert cache
        applies here too. It previously duplicated that loop inline, which made
        the pool dead code on this path — the routed path, i.e. the only one that
        matters — while the speculative path used it. Caught by
        ``test_cache_is_bit_identical_and_actually_hits``.
        """
        stats = _stats() if _stats_enabled() else None
        if stats is not None:
            stream = torch.cuda.current_stream(self.device)
            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record(stream)

        dest = self._alloc_dest()
        nbytes = self._copy_rows_into(dest, ids)
        self._bind(dest)

        if stats is not None:
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record(stream)
            stats.record_copy(start_ev, end_ev, nbytes,
                              len(ids) * (len(self._param_names) + len(self._buffer_names)),
                              "routed")
        self._last_stage_policy = "routed"
        self._last_stage_nbytes = nbytes

    def stage_routed(self, ids) -> None:
        """Single-slot stage of just the routed experts.

        Deliberately synchronous and **incompatible with next-layer prefetch**:
        which experts a layer needs is decided by the router from the *previous*
        layer's output, so it is not knowable early enough to prefetch. Routed-only
        trades that overlap (measured worth 1.11x) for 16x fewer bytes.
        """
        if self.staged:
            self._consume_ready_event()
            return
        cls = type(self)
        for h in list(cls._staged_now):
            if h is not self:
                h.evict()
        if cls._resident is not None and cls._resident is not self:
            cls._resident.evict()
        if self._spec_dev is not None:
            # A speculative prefetch is in flight or complete. Order the compute
            # stream after it, then fill only what the router asked for and the
            # guess missed. Every true row is present afterwards either way.
            torch.cuda.current_stream(self.device).wait_event(self._spec_event)
            missing = [e for e in ids if int(e) not in self._spec_ids]
            self._spec_hits += len(ids) - len(missing)
            self._spec_total += len(ids)
            if missing:
                self._spec_misses += len(missing)
                self._copy_rows_into(self._spec_dev, missing)
            self._bind(self._spec_dev)
            self._last_stage_policy = "speculative"
            self._spec_dev = None
            self._spec_ids = None
        else:
            self._release_speculation()
            self._copy_routed_to_device(ids)
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
        """Point the big tensors back at shared 0-element placeholders (idempotent), dropping the
        GPU copies so the caching allocator can reuse the memory for the next layer. Placeholders
        keep each tensor's home dtype (uint8 packed / float32 absmax / bf16-fp16 passthrough). Safe
        even for a still-in-flight prefetch: the dropped tensors were allocated on the side stream,
        so their reuse is stream-ordered after the pending copy."""
        if self._spec_dev is not None and not self.staged:
            # A speculative prefetch for THIS handle is in flight for a layer that
            # has not run yet. Evicting would silently discard it and the guess
            # would be paid for twice. The buffer is small (one layer) and bounded
            # by the lookahead distance.
            return
        b = self.base
        for n in self._param_names:
            b._parameters[n].data = _placeholder(self.device, self.home[n].dtype)
        for n in self._buffer_names:
            b._buffers[n] = _placeholder(self.device, self.home[n].dtype)
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
    if base is None and all(hasattr(experts_lora, n) for n in _ExpertOffload._names()):
        # Bare experts module with no LoRA wrapper (e.g. GptOssExperts4bit on the
        # inference/probe/kernel path): it holds the packed tensors directly, so it is
        # its own offload base and the stage/evict hooks register on it.
        base = experts_lora
    if base is None or not all(hasattr(base, n) for n in _ExpertOffload._names()):
        raise TypeError(
            "enable_expert_offload expects an ExpertsLoRA wrapping an Experts4bit base, or a bare "
            "experts module holding the packed tensors "
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
        inference = not torch.is_grad_enabled() and not module.training
        # Routed-only staging. args[1] is this layer's top_k_index — the router
        # already ran in the parent block, so the routing IS known here. Gated to
        # inference for the same reason prefetch is: a checkpoint recompute must
        # re-stage identically, and the conservative bulk path always does.
        if handle._routed_only and inference and len(args) > 1 and torch.is_tensor(args[1]):
            ids = torch.unique(args[1]).tolist()
            # Above the crossover essentially every expert routes (prefill), where
            # one bulk copy beats E strided ones. Fall through to the normal path.
            if len(ids) <= handle._routed_max:
                handle.stage_routed(ids)
                return
        if handle._prefetch_next is not None and inference:
            handle.stage_for_inference()
        else:
            handle.stage()

    experts_lora.register_forward_pre_hook(_stage_pre_hook)
    experts_lora.register_forward_hook(lambda module, args, output: handle.evict())
    return handle


class _ExpertPool:
    """A device-resident LRU cache of individual experts, shared across layers.

    Finding #32 measured **0.4513** overlap between the experts a layer routes to
    at token t and at token t-1. Re-staging those from pinned host costs the full
    link rate; keeping them on the device and copying **device-to-device** costs
    ~1 TB/s instead of ~22 GB/s, i.e. essentially nothing.

    The cache cannot be the per-layer destination itself: that keeps its full
    ``[E, ...]`` shape (1.36 GB/layer on a 235B), so holding one per layer would
    need ~128 GB. This is a flat pool of individual expert rows keyed by
    ``(layer, expert)``, sized independently of the layer count, from which the
    per-layer destination is filled.

    Bit-identity is preserved by construction: a pooled row is a byte copy of the
    same pinned home row, so a hit and a miss write identical bytes.
    """

    def __init__(self, slots, row_bytes, device):
        self.slots = slots
        self.device = device
        self.buf = {}
        self.row_bytes = row_bytes
        self._map = {}                     # (layer, expert) -> slot
        self._lru = []                     # slot order, oldest first
        self._free = list(range(slots))
        self.hits = self.misses = 0

    def _ensure(self, name, nelem, dtype):
        if name not in self.buf:
            self.buf[name] = torch.empty(self.slots, nelem, dtype=dtype, device=self.device)
        return self.buf[name]

    def get(self, key):
        slot = self._map.get(key)
        if slot is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            self._lru.remove(slot)
        except ValueError:
            pass
        self._lru.append(slot)
        return slot

    def put(self, key, rows):
        """rows: {name: 1-D device tensor}. Evicts LRU when full."""
        if key in self._map:
            return
        if self._free:
            slot = self._free.pop()
        else:
            slot = self._lru.pop(0)
            for k, v in list(self._map.items()):
                if v == slot:
                    del self._map[k]
                    break
        for name, row in rows.items():
            self._ensure(name, row.numel(), row.dtype)[slot].copy_(row, non_blocking=True)
        self._map[key] = slot
        self._lru.append(slot)

    def stats(self):
        tot = self.hits + self.misses
        return self.hits, self.misses, tot, (self.hits / tot if tot else float("nan"))


def enable_expert_cache(handles, slots=None, top_k=8):
    """Keep recently-used experts on the device so they are not re-streamed.

    ``slots`` defaults to ``len(handles) * top_k`` — enough to hold one full
    previous-token working set for every layer, which is what #32's 0.4513 reuse
    figure was measured against. Smaller pools trade reuse for VRAM.
    """
    if not handles:
        return None
    h0 = handles[0]

    # The pool allocates one row shape and reuses it for every layer, so a model
    # whose layers differ in expert geometry — different `num_experts`, different
    # projection widths, or a different set of offloaded tensors — would have
    # later layers writing into rows sized for the first. That corrupts silently,
    # which is the one failure mode this cache must not have. Check, do not
    # assume: `SUPPORTED_ARCHITECTURES` spans six model types and nothing
    # guarantees uniformity across them.
    def _shape_key(h):
        return (h.base.num_experts,
                tuple(sorted((n, h.home[n].numel() // h.base.num_experts,
                              str(h.home[n].dtype))
                             for n in h._param_names + h._buffer_names)))

    k0 = _shape_key(h0)
    odd = [i for i, h in enumerate(handles) if _shape_key(h) != k0]
    if odd:
        raise RuntimeError(
            f"enable_expert_cache requires every offloaded layer to have the same "
            f"per-expert geometry; layers {odd[:5]}"
            f"{'...' if len(odd) > 5 else ''} of {len(handles)} differ from layer 0. "
            f"A shared pool would size their rows from layer 0 and corrupt them. "
            f"Run without the cache, or pool per distinct geometry."
        )

    row_bytes = sum(h0.home[n].numel() // h0.base.num_experts * h0.home[n].element_size()
                    for n in h0._param_names + h0._buffer_names)

    # ONE POOL PER LAYER, not one shared pool.
    #
    # A single pool sized at `layers * top_k` is exactly one token's working set,
    # and a decode step touches every one of those rows in layer order. Global
    # LRU then evicts layer 0's entries to make room for layer 93's, one token
    # before layer 0 needs them again — a perfect thrash. Measured on the 235B:
    # **6 hits out of 34,719** (0.02%), while costing 7.98 GB and 1.11x the time.
    #
    # Partitioning per layer removes the interference entirely: layer L's rows
    # only ever compete with layer L's, so a `top_k`-sized partition holds
    # exactly the previous token's working set for that layer — which is what
    # finding #32's 0.4513 reuse figure was measured over.
    per_layer = slots if slots else top_k
    pools = []
    for i, h in enumerate(handles):
        pool = _ExpertPool(per_layer, row_bytes, h0.device)
        h._pool = pool
        h._pool_layer = i
        pools.append(pool)
    return _PoolGroup(pools, row_bytes)


class _PoolGroup:
    """The per-layer pools, with one aggregate view for reporting."""

    def __init__(self, pools, row_bytes):
        self.pools = pools
        self.row_bytes = row_bytes
        self.slots = sum(p.slots for p in pools)

    def stats(self):
        h = sum(p.hits for p in self.pools)
        m = sum(p.misses for p in self.pools)
        return h, m, h + m, (h / (h + m) if (h + m) else float("nan"))


def enable_speculative_staging(model, distance: int = 2, k: int = 8,
                               max_fraction: float = 0.5):
    """Prefetch each MoE layer's experts using its routing PREDICTED `distance`
    layers early, correcting misses synchronously.

    Routed staging is synchronous because layer L+1's routing is produced by a
    router reading layer L's output. But that routing is *predictable*: applying
    layer i's own router to layer (i-d)'s output recovers, of the true top-8,
    0.909 at d=1 and **0.847 at d=2** on Qwen3-30B-A3B (finding #32) — because
    routing is set by what the experts wrote to the residual, and one or two
    intervening layers change it only slightly.

    So this predicts, prefetches on the side stream during the intervening
    layers' compute, and stages whatever the router actually asked for and the
    guess missed. **Correctness does not depend on the prediction**: the
    destination keeps its full ``[E, ...]`` shape and every truly-routed row is
    present before the forward runs, so the result is bit-identical to routed
    staging no matter how bad the guess is. A wrong guess costs bandwidth only.

    `k` is deliberately the model's own top_k rather than a superset. #32's
    arithmetic: staging K experts costs ``K/top_k x`` routed staging's bytes, and
    a streamed decode step is transfer-bound, so K=16 measured *slower* than not
    speculating at all. The win comes from hiding compute, not from covering more
    of the distribution.

    Inference only, and mutually exclusive with `enable_inference_prefetch` for
    the same reason routed staging is.
    """
    layers = model.model.layers
    handles, hooked = [], 0
    for i, ly in enumerate(layers):
        h = getattr(getattr(ly, "mlp", None), "_offload", None)
        if h is None:
            for m in ly.modules():
                h = getattr(m, "_offload", None)
                if h is not None:
                    break
        if h is None:
            continue
        handles.append(h)
        h._routed_only = True
        h._routed_max = max(1, int(getattr(h.base, "num_experts", k * 2) * max_fraction))
        src = i - distance
        if src < 0:
            continue                      # first `distance` layers stay synchronous

        def make(target_layer, handle):
            def hook(mod, args, out):
                if torch.is_grad_enabled() or mod.training:
                    return
                o = out[0] if isinstance(out, tuple) else out
                if o.shape[1] > 1:
                    # Prefill: the target layer will route nearly every expert and
                    # take the bulk fallback, so a top-k guess is both useless and
                    # unconsumed. Prefetching here allocates a buffer nothing
                    # claims.
                    return
                hs = o[:, -1:, :]
                tl = layers[target_layer]
                gate = tl.mlp.gate
                x = tl.post_attention_layernorm(hs)
                logits = torch.nn.functional.linear(x.to(gate.weight.dtype), gate.weight)
                pred = logits.float().view(-1, logits.shape[-1]).topk(k, -1).indices
                handle.prefetch_speculative(pred.view(-1).unique().tolist())
            return hook

        layers[src].register_forward_hook(make(i, h))
        hooked += 1
    return handles, hooked


def speculative_stats(handles):
    """(hits, misses, total, hit_rate) over every speculatively-staged layer."""
    hi = sum(h._spec_hits for h in handles)
    mi = sum(h._spec_misses for h in handles)
    to = sum(h._spec_total for h in handles)
    return hi, mi, to, (hi / to if to else float("nan"))


def enable_routed_staging(handles, max_fraction: float = 0.5) -> list[_ExpertOffload]:
    """Stage only the experts a forward routes to, instead of the whole layer stack.

    The offload pre-hook copies a layer's entire expert tensor — all ``E`` of them —
    while a decode step routes to ``top_k``. On Qwen3-235B (E=128, top_k=8) that is
    **16x the bytes the routing needs**, which measured as a 5.14 s/token step where
    routed-only bytes imply 0.37 s at the same link. This makes the copy follow the
    router.

    **Mutually exclusive with inference prefetch, and not by preference.** Which
    experts layer L+1 needs is decided by a router reading layer L's output, so it
    cannot be known in time to prefetch. Routed-only gives up that overlap (measured
    1.11x) to move 16x less. Calling this on prefetch-linked handles raises.

    ``max_fraction`` is the crossover: when a forward routes to more than this share
    of the experts — prefill, where nearly all of them light up — the bulk copy is
    cheaper than ``E`` strided ones and the pre-hook falls back to it automatically.

    Inference only (``no_grad`` **and** ``eval``). Training and checkpoint recomputes
    keep the bulk single-slot path, so gradient semantics are untouched.
    """
    handles = list(handles)
    linked = [h for h in handles if h._prefetch_next is not None]
    if linked:
        raise RuntimeError(
            f"routed-only staging is incompatible with inference prefetch "
            f"({len(linked)} handles are prefetch-linked). The next layer's routing "
            f"is not known until this layer's output exists, so there is nothing to "
            f"prefetch. Load with prefetch=False, or call disable_inference_prefetch "
            f"first."
        )
    for h in handles:
        n_exp = getattr(h.base, "num_experts", None)
        if not n_exp:
            continue
        h._routed_only = True
        h._routed_max = max(1, int(n_exp * max_fraction))
    return handles


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

    Raises:
        RuntimeError: If ``model`` contains no :class:`ExpertsLoRA` modules. Offload was requested
            and nothing would be offloaded — every expert would silently stay GPU-resident, which
            is the exact silent-skip failure this package exists to prevent.
    """
    from .lora import ExpertsLoRA

    handles = []
    for module in model.modules():
        if isinstance(module, ExpertsLoRA):
            dev = device if device is not None else module.base.gate_up_proj.device
            handles.append(enable_expert_offload(module, dev, pin=pin))
    if not handles:
        raise RuntimeError(
            "offload_model_experts found no ExpertsLoRA modules — nothing would be offloaded and "
            "every expert would stay GPU-resident. Wrap the fused expert modules in ExpertsLoRA "
            "first, or load via load_moe_4bit_streaming(..., offload=True)."
        )
    return handles
