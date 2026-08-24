# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hybrid three-tier executor — Phase 3 of the hybrid CPU/GPU tier.

One engine, three disjoint flows, enforced by construction:

  VRAM bus   hot experts resident on the GPU (arena-sourced, meta-module
             friendly — inherited unchanged from ``_NvmeResidency``).
  DRAM bus   warm experts computed IN PLACE by grouped-nf4-gemm's native
             CPU kernels on host-resident packed NF4 bytes. Weight bytes
             never cross PCIe; only activation-sized traffic does. The
             stacks are deliberately pageable (never pinned): pinning is
             for H2D streaming, which this tier exists to not do.
  PCIe       activations, plus cold experts streamed NVMe→GPU through the
             ColdTier (inherited ``_TieredStack`` machinery). A cold miss
             never touches the DRAM stacks; a warm expert never touches
             the tier.

Placement comes from an ``engines.placement`` manifest (measured
calibration + routing profile in, expert→tier map out). Composition with
the parent classes is a single seam: ``_cold_contrib`` splits the
non-VRAM rows by the DRAM mask and delegates the NVMe remainder to the
parent's unchanged streaming path.

Numerics: the CPU tier dequantizes LUT×absmax in pure fp32 (the native
kernels' locked tree), while the GPU paths round dequant through the
module's compute dtype. Same placement ⇒ bit-identical runs; moving an
expert between tiers changes its rounding path within the documented
cross-placement tolerance — which is why the placement manifest is part
of run identity.

Known limitation (receipted on NPS4 bare metal): the DRAM stacks are
allocated by torch on whatever NUMA node the loader thread runs on. On a
multi-node box run the process under ``numactl --interleave=all`` until
executor-native node-local allocation lands; the enable path warns when
it detects multiple nodes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import torch

from ..util import log
from .hot_residency import enable_hot_residency, target_modules
from .nvme_experts import NF4_SEGMENTS, _NvmeResidency
from .placement import load_manifest

_PF_POOL = None                 # single-worker executor, owned by enable

_MARKER = "_e4b_hybrid"


def _parse_cold_dest(dest):
    """Normalize the cold execution destination: ``"gpu"``, ``"cpu"``, or a
    positive rows-per-unique-expert threshold.

    Module-level and validated at ENABLE time rather than at the first cold
    step, so a typo fails when the run is configured instead of thousands of
    tokens later — and so the rule is testable without a GPU.
    """
    if dest in ("gpu", "cpu", "deadline"):
        return dest
    try:
        val = float(dest)
    except (TypeError, ValueError):
        raise ValueError(
            f"cold_dest must be 'gpu', 'cpu', 'deadline', or a "
            f"rows-per-unique threshold; got {dest!r}") from None
    if val <= 0 or val != val:
        raise ValueError(f"cold_dest threshold must be > 0, got {dest!r}")
    return val


def build_cold_view(tier, index, *, direct=True):
    """The single cold CPU view for a tier, attached as ``_e4b_cold_view``.

    Built once per tier because the tier's slot space is global; see the
    note in ``_HybridTier.__init__``.

    ``direct=True`` asks for the preadv-scatter landing — segments DMA
    straight into the kernel-shaped stacks instead of arena row ->
    ``segment_into`` -> stack (measured ~43% faster on the fill path, gnf4
    ``RESULTS-direct-scatter.md``). It is not always legal, so the outcome
    is RECORDED on the view as ``e4b_path`` rather than silently chosen: a
    quiet fallback here is a quiet 43% regression, which is the same reason
    ``ArenaExpertSource.fetch_raw`` records ``last_fetch_path``.

    Two things can force the copy path, both stated in the reason string:
    an arena whose segment lengths, gaps or row padding are not
    align-aligned cannot scatter at all; and an arena storing absmax
    narrower than fp32 needs a widening cast, which a DMA has nowhere to
    perform.
    """
    from cold_cpu_view import ColdCpuView
    sufs = [NF4_SEGMENTS[k] for k in ("c_gu_p", "c_gu_a", "c_dn_p", "c_dn_a")]
    geo = {g["suffix"]: g for g in index["segments"]}
    narrow = [s for s in (NF4_SEGMENTS["c_gu_a"], NF4_SEGMENTS["c_dn_a"])
              if geo.get(s, {}).get("dtype") != "F32"]
    casts = {s: torch.float32 for s in narrow}
    reason = None
    if direct and narrow:
        reason = (f"absmax stored as {geo[narrow[0]]['dtype']}, which needs a "
                  f"widening cast a DMA cannot perform")
    view = None
    if direct and not narrow:
        try:
            view = ColdCpuView(tier, index, sufs, direct=True)
            tier.attach_landing(view.landing)
            view.e4b_path = "direct-scatter"
        except Exception as exc:                     # noqa: BLE001
            # Broad on purpose. An older gnf4 whose ColdCpuView has no
            # `direct` keyword raises TypeError, and a narrow except would
            # kill enable_hybrid_tier at its DEFAULT setting rather than
            # taking the copy path that has always worked (Bugbot, e4b#176).
            # The reason is kept WHOLE: splitting on "." truncated messages
            # that name an arena suffix like `nf4.gate_up_blocks`, which is
            # exactly the message a reader needs.
            view, reason = None, f"{type(exc).__name__}: {exc}"
    if view is None:
        view = ColdCpuView(tier, index, sufs, casts=casts or None)
        view.e4b_path = "copy"
        view.e4b_fallback_reason = reason
    tier._e4b_cold_view = view
    return view


def _refuse_direct_dtype_mismatch(view, index) -> None:
    """Refuse an external landing whose stacks are not the arena's own dtype.

    Only meaningful once a direct landing is attached: ``_TieredStack`` falls
    back to ``segment_tensor`` for any segment the view declines, and
    ``segment_tensor`` reads ``tier.row()``, which an external-landing tier
    refuses by name. The fallback and the landing are supposed to be mutually
    exclusive -- direct cannot cast, so a direct view's stacks always carry
    the stored dtype -- but that is an argument, and this is the check. It
    runs at setup so a violation names its cause instead of surfacing as a
    crash on the first cold row.
    """
    if getattr(view, "e4b_path", None) != "direct-scatter":
        return
    from nvme_residency import segment_geometry
    for suf in view.segments:
        native, *_rest = segment_geometry(index, suf)
        got = view.stack(suf).dtype
        if got != native:
            raise RuntimeError(
                f"direct landing attached but segment {suf!r} is "
                f"materialized as {got} against the arena's {native}. "
                f"_TieredStack would fall back to segment_tensor, which "
                f"reads tier.row(), which this tier refuses. Refusing at "
                f"setup rather than failing on the first cold row.")


def _split_oversize_groups(sizes, eids, max_rows=8):
    """Split same-expert groups into ``max_rows`` chunks.

    SUPERSEDED as a dispatch step (Phase 8): the native kernel now chunks
    a group across its 8-row register blocking INTERNALLY, keeping the
    weight row L1-hot, where splitting into separate groups re-read those
    weights from DRAM per chunk — the exact amortization G8 measures.
    Kept because :mod:`hybrid_train`'s backward still batches through the
    older contract, and because it is the reference the kernel's
    equivalence test is written against.
    """
    out_sizes, out_eids = [], []
    for s, e in zip(sizes, eids):
        while s > max_rows:
            out_sizes.append(max_rows)
            out_eids.append(e)
            s -= max_rows
        out_sizes.append(s)
        out_eids.append(e)
    return out_sizes, out_eids


class _HybridTier(_NvmeResidency):
    """VRAM-hot + DRAM-computed-warm + NVMe-streamed-cold, one module."""

    def __init__(self, mod, hot_ids, device):
        super().__init__(mod, hot_ids, device)          # VRAM + tier stacks
        from nvme_residency import segment_tensor

        dram_ids = torch.as_tensor(
            getattr(mod, "_e4b_hybrid_dram_ids"), dtype=torch.long).unique()
        E = mod.num_experts
        # Controller mode (C2): the DRAM stacks cover EVERY expert so a
        # demotion is a pure mask flip and a promotion sources its H2D from
        # d_* locally -- runtime swaps never touch the arena or any
        # membership structure. The manifest still defines the HOT set; the
        # overlap check is meaningless here because overlap is the point.
        self.swappable = bool(getattr(mod, "_e4b_hybrid_swappable", False))
        if self.swappable:
            dram_ids = torch.arange(E, dtype=torch.long)
        elif dram_ids.numel():
            if int(dram_ids.min()) < 0 or int(dram_ids.max()) >= E:
                raise ValueError(f"dram ids outside [0, {E})")
            overlap = set(dram_ids.tolist()) & set(self.hot_ids.tolist())
            if overlap:
                raise ValueError(f"experts placed in BOTH vram and dram: "
                                 f"{sorted(overlap)[:8]}")
        self.dram_ids = dram_ids
        self._threads = int(getattr(mod, "_e4b_hybrid_threads", 0))

        # Serving tier for residency/runtime; SETUP tier for the one-shot
        # bulk reads that build the DRAM stacks below (see _build_hot).
        tier = mod._e4b_cold_tier
        setup = getattr(mod, "_e4b_setup_tier", None) or tier
        index = tier.reader.index
        layer = int(getattr(mod, "_e4b_arena_layer", 0))
        ids = [int(e) for e in dram_ids.tolist()]
        chunk = max(1, min(len(ids) or 1, setup.hot_rows))
        for attr, suffix in (("d_gu_p", NF4_SEGMENTS["c_gu_p"]),
                             ("d_gu_a", NF4_SEGMENTS["c_gu_a"]),
                             ("d_dn_p", NF4_SEGMENTS["c_dn_p"]),
                             ("d_dn_a", NF4_SEGMENTS["c_dn_a"])):
            if not ids:
                geo = next(g for g in index["segments"] if g["suffix"] == suffix)
                shp = (0,) + tuple(geo["shape_per_expert"])
                dt = torch.float32 if geo["dtype"] == "F32" else torch.uint8
                setattr(self, attr, torch.empty(shp, dtype=dt))
                continue
            parts = [segment_tensor(setup, index, layer, ids[i:i + chunk], suffix)
                     for i in range(0, len(ids), chunk)]
            stacked = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            # pageable host memory ON PURPOSE — the DRAM-bus law
            setattr(self, attr, stacked.contiguous().cpu())
        if self.d_gu_a.dtype != torch.float32:
            self.d_gu_a = self.d_gu_a.float()
        if self.d_dn_a.dtype != torch.float32:
            self.d_dn_a = self.d_dn_a.float()

        # Companion to is_dram: lets the deadline rule size the GPU's
        # committed work with a mask lookup rather than a set search on the
        # hot path.
        self.is_vram = torch.zeros(E, dtype=torch.bool, device=self.device)
        if self.hot_ids.numel():
            self.is_vram[self.hot_ids.to(self.device)] = True
        self.is_dram = torch.zeros(E, dtype=torch.bool, device=self.device)
        if dram_ids.numel():
            self.is_dram[dram_ids.to(self.device)] = True
        g2d = torch.full((E,), -1, dtype=torch.long)
        g2d[dram_ids] = torch.arange(dram_ids.numel())
        self.g2d_cpu = g2d
        if self.gptoss:
            ci = dram_ids
            self.d_gu_b = (mod.gate_up_bias.detach().index_select(0, ci)
                           .to("cpu", torch.float32).contiguous())
            self.d_dn_b = (mod.down_bias.detach().index_select(0, ci)
                           .to("cpu", torch.float32).contiguous())

        # Phase 4: speculative prefetch (route L+1 from L's hidden). Wired
        # by enable when prefetch=True and a router chain is discoverable.
        self.nvme_set = frozenset(range(E)) - set(
            int(e) for e in self.hot_ids.tolist()) - set(
            int(e) for e in dram_ids.tolist())
        self.pf = None                 # {"np_w","np_b","next","k"} for L+1
        self.pf_enabled = False
        self.pf_submitted = 0
        self.pf_rows = 0
        self.pf_pin = None             # pinned hidden staging, 2-slot ring
        self.pf_ev = None
        self.pf_slot = 0

    def swap_expert(self, promote: int, demote: int):
        """Retarget the demoted expert's VRAM slot to the promoted expert.

        Controller mode only. The hot stacks keep their size; the slot's
        rows are overwritten from the all-expert DRAM stacks (one H2D per
        segment), then the id algebra flips: g2h, is_hot, is_vram, and the
        hot_ids introspection tensor. The demoted expert stays servable
        from d_* (which covers every expert in this mode), so no other
        structure changes. Called BETWEEN steps -- never concurrently with
        a forward.
        """
        assert self.swappable, "swap_expert requires controller mode"
        slot = int(self.g2h[demote].item())
        di = int(self.g2d_cpu[promote].item())
        assert slot >= 0, f"demote {demote} is not VRAM-resident"
        assert int(self.g2h[promote].item()) < 0, f"promote {promote} already hot"
        assert di >= 0
        self.h_gu_p[slot].copy_(self.d_gu_p[di], non_blocking=False)
        self.h_gu_a[slot].copy_(self.d_gu_a[di], non_blocking=False)
        self.h_dn_p[slot].copy_(self.d_dn_p[di], non_blocking=False)
        self.h_dn_a[slot].copy_(self.d_dn_a[di], non_blocking=False)
        if self.gptoss:
            self.h_gu_b[slot].copy_(
                self.d_gu_b[di].to(self.h_gu_b.dtype, copy=False))
            self.h_dn_b[slot].copy_(
                self.d_dn_b[di].to(self.h_dn_b.dtype, copy=False))
        self.g2h[promote] = slot
        self.g2h[demote] = -1
        self.is_hot[promote] = True
        self.is_hot[demote] = False
        self.is_vram[promote] = True
        self.is_vram[demote] = False
        hm = self.hot_ids == demote
        self.hot_ids = torch.where(
            hm, torch.tensor(promote, dtype=self.hot_ids.dtype), self.hot_ids)

        # Phase 8 amortization instrument. OFF by default and structurally
        # free when off (invariant 9): the counting block is guarded and
        # does its own unique() work only when armed, so a serving run pays
        # nothing. Bytes are per-expert weight bytes on THIS module's
        # geometry, read off the stacks rather than assumed.
        self.amort = None              # dict when armed, else None
        self._exp_bytes = None
        self._gpu_only = False         # Phase 9 mixed mode, off by default
        # Per-STEP offload switch (Phase 8 follow-up): when this step's
        # DRAM rows-per-unique-expert exceeds the threshold, take the GPU
        # path for the DRAM experts instead of computing in place. The
        # rows-curve diagnostic measured the CPU tier's cost as
        # a + b*rows per expert — compute-BOUND past a few rows — while
        # the GPU path's cost is one H2D per unique expert, flat in rows.
        # Concentrated routing (G8's own amortization finding) puts
        # serving decode past the crossover at B=8, so without this
        # switch the "warm" tier is the slowest bus exactly when batching
        # works. None = off (invariant 9: no behavior change unless set).
        self.offload_rows = getattr(mod, "_e4b_hybrid_offload_rows", None)
        self.fused_ffn = getattr(mod, "_e4b_hybrid_fused_ffn", False)
        self.offload_steps = 0         # steps that took the GPU path
        # Thin-layer routing (coresweep receipts): concentrated
        # placement leaves early layers with a handful of DRAM
        # experts, and every such call pays the pool's per-call floor
        # for a few MB of work. A layer whose TOTAL DRAM population
        # is <= the threshold can never make a fat call, so the
        # decision is STATIC — a runtime uniqueness test would cost a
        # data-dependent device sync per call, the exact stall class
        # removed from the KV path.
        thin = getattr(mod, "_e4b_hybrid_thin_uniq", None)
        n_dram = int(self.is_dram.sum())
        self.dram_thin = thin is not None and 0 < n_dram <= int(thin)
        self.thin_steps = 0

        # Stage 3 (tribrid): the CPU destination for cold experts. The tier
        # reads a cold row from NVMe exactly as before; the view re-lays it
        # out into the contiguous per-segment stacks the native kernels
        # require, keyed by the tier's own SLOT. Residency stays the tier's
        # decision — this adds a destination, not a second cache.
        self.layer = layer
        self._cold_dest = _parse_cold_dest(
            getattr(mod, "_e4b_hybrid_cold_dest", "gpu"))
        self.view = None
        self.cold_cpu_steps = 0
        self.cold_cpu_rows = 0
        self.cold_gpu_rows = 0
        # Workstream 4. `costs` is None unless cold_dest="deadline"; the
        # rule then reads BOTH engines' committed work for this layer and
        # picks the destination that reaches the join first. Decisions are
        # recorded with their counterfactual so the model can be SCORED
        # rather than trusted (the prereg's falsifiability hook).
        self.costs = getattr(mod, "_e4b_hybrid_costs", None)
        self.deadline_log = []
        self.deadline_flips = 0
        if self._cold_dest != "gpu":
            if self.gptoss:
                # gpt-oss per-expert biases do NOT ride the arena (the
                # loader refuses arena serving for exactly this reason), so
                # a cold row has no bias to add. Dropping it would produce
                # plausible numbers that are quietly wrong — refuse instead.
                raise ValueError(
                    "cold_dest != 'gpu' is unavailable for gpt-oss: per-expert "
                    "biases do not ride the arena, so a cold row computed on "
                    "the CPU would silently omit them. Bake biases into the "
                    "arena first (HANDOFF open work #3).")
            # ONE view per TIER, not per module. The tier is shared across
            # every MoE module and its slot space is global (keyed
            # (layer, expert)), so a per-module view sized tier.hot_rows
            # allocated hot_rows*row_bytes PER LAYER while only ever using
            # the slots holding its own layer's experts — 16x the host RAM
            # it needed on a 16-layer model. The view is built once, in
            # enable_hybrid_tier, and every module reads the same one.
            self.view = getattr(tier, "_e4b_cold_view", None)
            if self.view is None:
                raise RuntimeError(
                    "cold_dest != 'gpu' but no cold view is attached to the "
                    "tier. enable_hybrid_tier builds it; constructing "
                    "_HybridTier directly needs build_cold_view() first.")

    def expert_bytes(self) -> int:
        """Weight bytes one expert occupies (gate/up + down, payload plus
        absmax), from the resident stacks — never a spec-sheet number."""
        if self._exp_bytes is None:
            n = 0
            for attr in ("d_gu_p", "d_gu_a", "d_dn_p", "d_dn_a"):
                s = getattr(self, attr, None)
                if s is not None and s.shape[0]:
                    n += s[0].numel() * s.element_size()
            if n == 0:                       # no DRAM experts on this module
                for attr in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a"):
                    s = getattr(self, attr, None)
                    if s is not None and s.shape[0]:
                        n += s[0].numel() * s.element_size()
            self._exp_bytes = int(n)
        return self._exp_bytes

    def arm_amortization(self, on: bool = True):
        """Start (or clear) per-step unique-expert accounting."""
        self.amort = {"steps": 0, "acts": 0,
                      "uniq_vram": 0, "uniq_dram": 0, "uniq_nvme": 0,
                      "acts_vram": 0, "acts_dram": 0, "acts_nvme": 0,
                      "dram_groups": 0, "expert_bytes": self.expert_bytes(),
                      # Stage 3: the cold CPU destination, counted apart from
                      # the DRAM bus. Same bus, different bytes-origin — and
                      # conflating them would hide exactly the cold-path cost
                      # the tribrid prereg is written to measure.
                      "cold_cpu_groups": 0, "cold_cpu_ns": 0,
                      # per-bus wall time, measured with a PER-OP PROBE:
                      # the CPU bus is synchronous host work so its wall
                      # is exact, and the GPU bus is bracketed by its own
                      # CUDA events. Attribution by subtracting one arm
                      # from another failed three times in this program
                      # (Phase 1's wake-time hunt); it is not used here.
                      "dram_ns": 0, "gpu_ns": 0,
                      # per-expert routing histogram: the empirical p_e the
                      # general amortization law needs. The gate's closed
                      # form assumes these are all k/E; whether they are is
                      # a measurement, not an axiom.
                      "hist": torch.zeros(int(self.mod.num_experts),
                                          dtype=torch.long,
                                          device=self.device),
                      # per-expert count of STEPS in which the expert was
                      # touched at all. hist/asks answers "how often is e
                      # drawn"; this answers "in what fraction of steps
                      # does e appear" -- the step-level touch frequency
                      # the co-routing model consumes directly, with no
                      # independence assumption between a step's tokens.
                      "touch": torch.zeros(int(self.mod.num_experts),
                                           dtype=torch.long,
                                           device=self.device),
                      # steps in which THIS layer had any DRAM-tier work at
                      # all: the per-call fixed cost bills once per such
                      # step, so the regime-split conversion model needs
                      # this count separately from the unique totals.
                      "dram_steps": 0,
                      # optional per-step series of touched expert ids
                      # (all tiers -- routing is placement-independent).
                      # GPU tensors are appended sync-free and moved to
                      # host only when a consumer drains the list, so a
                      # timing run without a series consumer pays nothing.
                      "series": [],
                      } if on else None
        return self.amort

    def _count_amortization(self, top_k_index):
        """Unique experts touched per tier for THIS step, plus activation
        counts. The pair is the whole measurement: their ratio is the
        amortization the batch actually bought, against B*k."""
        a = self.amort
        flat = top_k_index.reshape(-1).to(self.device)
        a["steps"] += 1
        a["acts"] += int(flat.numel())
        uniq = torch.unique(flat)
        a["touch"][uniq] += 1
        a["series"].append(uniq)
        hot_u = self.is_hot[uniq]
        dram_u = self.is_dram[uniq]
        a["uniq_vram"] += int(hot_u.sum())
        n_dram_u = int((dram_u & ~hot_u).sum())
        a["uniq_dram"] += n_dram_u
        if n_dram_u:
            a["dram_steps"] += 1
        a["uniq_nvme"] += int((~hot_u & ~dram_u).sum())
        hot_a = self.is_hot[flat]
        dram_a = self.is_dram[flat]
        a["acts_vram"] += int(hot_a.sum())
        a["acts_dram"] += int((dram_a & ~hot_a).sum())
        a["acts_nvme"] += int((~hot_a & ~dram_a).sum())
        a["hist"] += torch.bincount(flat, minlength=a["hist"].numel())

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.amort is not None:
            self._count_amortization(top_k_index)
        out = super().forward(hidden_states, top_k_index, top_k_weights)
        if (self.pf_enabled and self.pf is not None
                and hidden_states.shape[0] <= 8):
            self._submit_prefetch(hidden_states)
        return out

    def _submit_prefetch(self, hidden):
        """Predict layer L+1's routing from THIS hidden and warm the
        predicted NVMe-resident experts' tier slots on a background worker.

        Correctness is trivial by construction: the tier is lock-guarded
        with publish-after-fill slots, and the demand path is unchanged —
        a mispredict costs one wasted disk read, never a wrong byte.
        Deviation from the directive's letter, documented: v1 warms the
        PINNED tier slots (killing the disk read on the critical path,
        which is the stall) rather than a VRAM pool; the remaining
        pinned→GPU hop is the measured 25–56 GB/s fast path."""
        pf = self.pf
        nxt = pf["next"]
        if not nxt.nvme_set:
            return                          # free when unused, structurally
        pool = _PF_POOL
        if pool is None or pool._work_queue.qsize() > 1:
            return                          # never queue into lag
        # The hidden leaves the GPU on the MAIN thread as an async pinned
        # copy gated by an event; the worker host-waits the event and never
        # issues a CUDA call. v1 did h.float().cpu() INSIDE the worker — a
        # cross-thread stream synchronize per layer that cost 4x end-to-end
        # (measured 6.21 -> 1.57 tok/s at 235B): the same sync class the
        # CPU router exists to kill, reintroduced from a thread.
        # 4-slot ring vs a live-task bound of 3 (1 running + <=1 queued by
        # the qsize guard above + this submit): a slot can never be
        # overwritten while a worker still reads it. The old 2-slot ring
        # could — qsize() does not count the RUNNING task, so a late copy_
        # raced the worker's float().numpy() read via CUDA DMA and fed the
        # predictor torn bytes (wasted fetches, never wrong outputs).
        t_rows = hidden.shape[0]
        if self.pf_pin is None or self.pf_pin[0].shape[0] < t_rows:
            self.pf_pin = [torch.empty(max(8, t_rows), hidden.shape[-1],
                                       dtype=hidden.dtype, pin_memory=True)
                           for _ in range(4)]
            self.pf_ev = [torch.cuda.Event() for _ in range(4)]
        s = self.pf_slot
        self.pf_slot = (s + 1) % 4
        pin = self.pf_pin[s]
        ev = self.pf_ev[s]
        pin[:t_rows].copy_(hidden.detach(), non_blocking=True)
        ev.record(torch.cuda.current_stream(hidden.device))

        def task():
            import numpy as np
            ev.synchronize()               # host wait only; no CUDA work
            hc = pin[:t_rows].float().numpy()
            # BLAS-free on purpose: `hc @ np_w.T` dispatches to OpenBLAS,
            # which spawns a full spinning thread team PER FIRE on a big
            # host. ~5.8K fires oversubscribed every core against the gnf4
            # pool and torch's OMP workers, and every parallel region on
            # the MAIN thread (each torch.stack, every layer) inherited the
            # barrier cost — measured as the whole 6.6x arm-C slowdown at
            # 235B while disk and tier sat idle. Ufunc broadcast+reduce is
            # single-threaded and this product is 0.5 MFLOP; it needs no
            # team.
            logits = (pf["np_w"][None, :, :] * hc[:, None, :]).sum(axis=2)
            if pf["np_b"] is not None:
                logits = logits + pf["np_b"]
            k = pf["k"]
            ids = set()
            for row in logits:
                ids.update(np.argpartition(-row, k)[:k].tolist())
            want = sorted(ids & nxt.nvme_set)
            # never let speculation own more than a quarter of the slots
            want = want[: max(1, nxt._tier.hot_rows // 4)]
            if want:
                # speculative=True is the tier's own contract: fills overlap
                # demand I/O instead of convoying it (a lock-wrapped demand
                # ensure here measured 6.6x SLOWER end-to-end at 235B), the
                # demand window is unevictable, and no-victim means skip, not
                # error. Speculation must never kill serving, so any read
                # failure is swallowed — the demand path will surface it.
                try:
                    nxt._tier.ensure(int(nxt.mod._e4b_arena_layer), want,
                                     speculative=True)
                except Exception:                 # noqa: BLE001
                    return
                self.pf_rows += len(want)
            self.pf_submitted += 1

        try:
            pool.submit(task)
        except RuntimeError:
            pass                            # executor shut down mid-flight

    # ------------------------------------------------------- mixed mode --
    def prefill_gpu_only(self, on: bool = True):
        """Phase 9 mixed mode: route DRAM experts to the GPU for this
        step instead of computing them on the CPU.

        Prefill is compute-bound — G8 measured the DRAM tier leaving the
        bandwidth-bound regime near ~8 tokens per expert, and a prefill
        chunk is far past it — so a chunk's expert weights should cross
        PCIe ONCE and amortize over its many tokens. Decode is the
        opposite and stays on the hybrid tier.

        The bytes are the SAME bytes: this streams from ``d_*``, the one
        host copy the CPU tier already computes on, so no expert is
        duplicated to serve the second path. Routing through the
        inherited cold path instead would look equivalent and quietly
        re-read from the ARENA FILE — those stacks are ``_TieredStack``
        views over disk, not host memory — turning a DRAM hit into an
        NVMe read for bytes already resident.

        NUMERICS, measured rather than assumed: a per-op probe (same
        inputs, one layer, both buses) agrees to 1e-4 absolute on
        outputs of norm ~1.2 — bf16 rounding scale, where a wrong
        permutation or misaligned stack would land O(1). Over a whole
        forward those differences accumulate and CAN flip an argmax,
        because switching a DRAM expert from the CPU tier's fp32 dequant
        to the GPU's compute-dtype dequant is precisely the
        cross-placement rounding change this module documents at the
        top: same placement is bit-identical, a moved expert is not.
        Mixed mode therefore belongs in run identity alongside the
        manifest, and a bit-exactness check against the CPU tier is the
        wrong test to write.
        """
        self._gpu_only = bool(on)

    def _dram_on_gpu(self, x, flat, row_token, row_slot, dr, top_k_weights,
                     out, dev):
        """DRAM experts, computed on the GPU from the host stacks."""
        from .hot_residency import _fused_over_stack

        glob = flat.index_select(0, dr).cpu()
        local_full = self.g2d_cpu.index_select(0, glob)
        routed, compact = torch.unique(local_full, return_inverse=True)
        # ONE H2D per unique expert per chunk — the amortization that
        # makes prefill worth offloading at all (G8's law: unique reads,
        # not activations)
        # absmax crosses in the COMPUTE dtype: the DRAM stacks keep fp32
        # scales for the CPU kernels' locked tree, while the fused GPU
        # kernel is written against the hot stack's compute-dtype scales
        # (h_gu_a is bf16). Casting on transfer also halves these bytes,
        # which measured 2.9x -> 6.7x on a 21-token prefill.
        cd = x.dtype
        gu_p = self.d_gu_p.index_select(0, routed).to(dev, non_blocking=True)
        gu_a = self.d_gu_a.index_select(0, routed).to(dev, cd,
                                                      non_blocking=True)
        dn_p = self.d_dn_p.index_select(0, routed).to(dev, non_blocking=True)
        dn_a = self.d_dn_a.index_select(0, routed).to(dev, cd,
                                                      non_blocking=True)
        xr = x.index_select(0, row_token.index_select(0, dr))
        gptoss = None
        if self.gptoss:
            r_dev = routed.to(dev)
            gptoss = (self.d_gu_b.index_select(0, r_dev).to(dev),
                      self.d_dn_b.index_select(0, r_dev).to(dev),
                      self.alpha, self.limit)
        dn = _fused_over_stack(xr, compact.to(dev), gu_p, gu_a, dn_p, dn_a,
                               self.shapes, self.has_gate, self.act_fn,
                               gptoss=gptoss, clamp_limit=self.clamp_limit)
        w = top_k_weights[row_token.index_select(0, dr),
                          row_slot.index_select(0, dr)].to(torch.float32)
        out.index_put_((row_token.index_select(0, dr),
                        row_slot.index_select(0, dr)),
                       dn.to(torch.float32) * w[:, None])

    # ------------------------------------------------------------------ #
    def _cold_contrib(self, x, flat, row_token, row_slot, cr, top_k_weights,
                      out, dev):
        dmask = self.is_dram[flat.index_select(0, cr)]
        nr = cr[~dmask]
        dr = cr[dmask]
        if nr.numel():
            # Stage 3: cold rows now have TWO destinations. The choice is
            # still static — a deadline estimate is workstream 4, and
            # guessing one here would be a threshold wearing a scheduler's
            # name. `cold_dest` defaults to "gpu", which is the parent's
            # path verbatim (invariant 9: no behaviour change unless set).
            if self._cold_to_cpu(nr, flat):
                self.cold_cpu_steps += 1
                self.cold_cpu_rows += int(nr.numel())
                self._cold_on_cpu(x, flat, row_token, row_slot, nr,
                                  top_k_weights, out, dev)
            else:
                self.cold_gpu_rows += int(nr.numel())
                super()._cold_contrib(x, flat, row_token, row_slot, nr,
                                      top_k_weights, out, dev)
        if dr.numel():
            gpu_route = getattr(self, "_gpu_only", False)
            if not gpu_route and getattr(self, "dram_thin", False):
                gpu_route = True
                self.thin_steps += 1
            thresh = getattr(self, "offload_rows", None)
            if not gpu_route and thresh is not None:
                uniq = int(torch.unique(flat.index_select(0, dr)).numel())
                if dr.numel() / max(1, uniq) >= thresh:
                    gpu_route = True
                    self.offload_steps += 1
            if gpu_route:
                self._dram_on_gpu(x, flat, row_token, row_slot, dr,
                                  top_k_weights, out, dev)
            else:                            # DRAM bus: compute in place
                self._dram_contrib(x, flat, row_token, row_slot, dr,
                                   top_k_weights, out, dev)

    def _dram_contrib(self, x, flat, row_token, row_slot, dr, top_k_weights,
                      out, dev):
        if self.amort is not None:
            t0 = time.perf_counter_ns()
            try:
                return self._dram_contrib_inner(
                    x, flat, row_token, row_slot, dr, top_k_weights, out, dev)
            finally:
                self.amort["dram_ns"] += time.perf_counter_ns() - t0
        return self._dram_contrib_inner(x, flat, row_token, row_slot, dr,
                                        top_k_weights, out, dev)

    def _dram_contrib_inner(self, x, flat, row_token, row_slot, dr,
                            top_k_weights, out, dev):
        glob = flat.index_select(0, dr).cpu()
        local = self.g2d_cpu.index_select(0, glob)
        self._cpu_over_stacks(
            x, row_token, row_slot, dr, local, top_k_weights, out, dev,
            (self.d_gu_p, self.d_gu_a, self.d_dn_p, self.d_dn_a),
            (self.d_gu_b, self.d_dn_b) if self.gptoss else None,
            amort_key="dram_groups")

    # ------------------------------------------------- the cold CPU path --
    def _cold_to_cpu(self, nr, flat) -> bool:
        """Destination for this step's cold rows: True = CPU, False = GPU.

        Deliberately the same SHAPE of rule the DRAM tier already uses
        (``offload_rows``), read the other way round. The DRAM tier asks
        "are there enough rows per unique expert that the GPU's flat-in-rows
        cost wins?"; a cold row has to be read from NVMe either way, so the
        question is which engine turns those bytes into a contribution
        first. Below the threshold the CPU's per-call floor dominates and
        the GPU wins; at and above it the CPU's in-place compute wins
        because the bytes never have to cross PCIe a second time.

        This is a threshold, not a scheduler, and the PREREG says so: gate 2
        is what decides whether a deadline estimate beats it, and it cannot
        be answered by picking a better constant here.

        A step-by-step destination is therefore also a step-by-step ROUNDING
        path (see ``enable_hybrid_tier``'s numerics note). Under a threshold
        the same expert can take either destination depending on the step's
        routing, so a threshold arm is not bit-reproducible against either
        fixed destination — only against itself on the same routing trace.
        """
        if self._cold_dest == "gpu":
            return False
        if self.view is None:                # no CPU destination available
            return False
        if self._cold_dest == "cpu":
            return True
        if self._cold_dest == "deadline":
            return self._cold_to_cpu_deadline(nr, flat)
        # "auto": rows per unique cold expert, the DRAM tier's own statistic
        uniq = int(torch.unique(flat.index_select(0, nr)).numel())
        return (nr.numel() / max(1, uniq)) >= self._cold_dest

    def _group(self, idx, flat):
        """(rows, unique experts) for a row selection. One device->host sync
        per call, which is why the deadline rule is opt-in."""
        if idx.numel() == 0:
            return 0, 0
        e = flat.index_select(0, idx)
        return int(idx.numel()), int(torch.unique(e).numel())

    def _cold_to_cpu_deadline(self, nr, flat):
        """Destination by predicted time-to-contribution, including what each
        engine is ALREADY committed to for this layer.

        The committed work is what makes this a deadline estimate rather than
        a threshold. Both sides are this step's, and both are knowable here:

          CPU  the DRAM rows this layer will compute in place. They have not
               started -- the cold group is dispatched first -- but for a
               `max(cpu_side, gpu_side)` join it is the SIDE TOTAL that
               matters, not the order within a side, so counting them is
               correct either way. Excluded when they are being offloaded to
               the GPU, because then they are not the CPU's work at all.
          GPU  the resident VRAM rows, already submitted by the parent's
               forward and still in flight.

        **Cost, stated rather than hidden**: `_group` forces a device->host
        sync per call, and this rule makes up to three. That is the stall
        class the CPU router exists to remove, which is exactly why this is
        opt-in and not the default. A sync-free version needs the counts to
        ride the router's existing host-side copy; that is the follow-up,
        and until it lands the deadline arm is a measurement tool rather
        than a serving default.
        """
        import cold_deadline

        c = self.costs
        if c is None:                        # no measured constants: refuse
            raise RuntimeError(
                "cold_dest='deadline' needs measured cost constants; pass "
                "costs= to enable_hybrid_tier. Guessing them would put a "
                "spec-sheet number into a scheduling decision.")
        rows, uniq = self._group(nr, flat)
        if rows == 0:
            return False

        # Which engine the DRAM rows land on has to be decided the SAME way
        # _cold_contrib decides it, including the offload_rows test -- and
        # when they go to the GPU their cost belongs to the GPU side, not
        # nowhere. Dropping it made the estimator see a busy CPU beside an
        # idle GPU that was in fact doing the DRAM work, so cold rows could
        # be sent to the engine already carrying it (Bugbot, e4b#179).
        d_idx = torch.nonzero(self.is_dram[flat], as_tuple=True)[0]
        d_rows, d_uniq = self._group(d_idx, flat)
        dram_on_gpu = (getattr(self, "_gpu_only", False)
                       or getattr(self, "dram_thin", False))
        thresh = getattr(self, "offload_rows", None)
        if not dram_on_gpu and thresh is not None and d_rows:
            dram_on_gpu = (d_rows / max(1, d_uniq)) >= thresh
        cpu_committed = (0.0 if dram_on_gpu
                         else cold_deadline.cpu_us(d_rows, d_uniq, c))

        v_idx = torch.nonzero(self.is_vram[flat], as_tuple=True)[0]
        v_rows, v_uniq = self._group(v_idx, flat)
        gpu_committed = cold_deadline.gpu_us(v_rows, v_uniq, c)
        if dram_on_gpu:
            gpu_committed += cold_deadline.gpu_us(d_rows, d_uniq, c)

        d = cold_deadline.choose(rows, uniq, c, cpu_backlog_us=cpu_committed,
                                 gpu_backlog_us=gpu_committed)
        rec = d.record()
        rec.update(rows=rows, uniq=uniq, cpu_committed_us=cpu_committed,
                   gpu_committed_us=gpu_committed)
        self.deadline_log.append(rec)
        if d.flipped_by_backlog:
            self.deadline_flips += 1
        return d.dest == "cpu"

    def _cold_on_cpu(self, x, flat, row_token, row_slot, nr, top_k_weights,
                     out, dev):
        """Cold experts, computed on the CPU from tier-resident packed bytes.

        The bytes are read from NVMe ONCE, into the cold tier's own rows, and
        the view re-lays them out for the kernel — no second read, and no
        duplicate of an expert to serve the second destination (the trap
        ``prefill_gpu_only`` documents for the DRAM tier, in the other
        direction). ``ColdCpuView.ensure`` returns TIER SLOTS, which index
        its stacks directly, so slot IS the kernel's expert id here.
        """
        t0 = time.perf_counter_ns() if self.amort is not None else None
        glob = flat.index_select(0, nr).cpu()
        slots = self.view.ensure(self.layer, glob.tolist())
        local = torch.as_tensor(slots, dtype=torch.long)
        v = self.view
        self._cpu_over_stacks(
            x, row_token, row_slot, nr, local, top_k_weights, out, dev,
            (v.stack(NF4_SEGMENTS["c_gu_p"]), v.stack(NF4_SEGMENTS["c_gu_a"]),
             v.stack(NF4_SEGMENTS["c_dn_p"]), v.stack(NF4_SEGMENTS["c_dn_a"])),
            None,                            # gpt-oss is refused at enable
            amort_key="cold_cpu_groups")
        if t0 is not None:
            self.amort["cold_cpu_ns"] += time.perf_counter_ns() - t0

    def _cpu_over_stacks(self, x, row_token, row_slot, sel, local,
                         top_k_weights, out, dev, stacks, biases,
                         *, amort_key=None):
        """The CPU bus, over whichever host stacks hold these rows' bytes.

        Extracted from ``_dram_contrib_inner`` so the DRAM tier and the cold
        CPU destination run ONE implementation rather than two that drift:
        the activation chain, the gpt-oss bias/clamp handling, the fused-FFN
        eligibility rules and the scatter back are identical work, and only
        the stacks and the per-row index into them differ. ``local`` indexes
        ``stacks``, whatever those are — the DRAM tier's local expert index,
        or the cold tier's SLOT index.
        """
        import cpu_grouped

        gu_p, gu_a, dn_p, dn_a = stacks
        rows = row_token.index_select(0, sel)
        xr = x.index_select(0, rows).to("cpu", torch.float32)

        order = torch.argsort(local)
        sl = local.index_select(0, order)
        xs = xr.index_select(0, order).contiguous()
        uniq, counts = torch.unique_consecutive(sl, return_counts=True)
        # NO caller-side split (Phase 8): one group per unique expert, so
        # its weights are read once and the kernel's internal chunking
        # amortizes them over every routed row. Splitting here made the
        # DRAM bus re-read a whole expert every 8 rows.
        sizes, eids = counts.tolist(), uniq.tolist()
        if self.amort is not None and amort_key is not None:
            # post-split group count, not the unique count: each split
            # chunk re-reads its expert's weights, so THIS is the number
            # the DRAM bus actually pays. The gap between it and
            # uniq_dram is the split tax, and it must be visible.
            self.amort[amort_key] += len(sizes)

        # Fused expert-FFN path (one kernel call, one pool wake, no
        # intermediate through python) — OFF BY DEFAULT: measured on a
        # serving-class TR 7975WX (bench/hybrid-g9/remeasure/), the
        # coarse (group, row-chunk) partition loses 13-37% to worker
        # imbalance against the two-call path's fine column tiles, and
        # degrades with batch. Opt in (fused_ffn=True) for floor-heavy
        # small-pool hosts (+24% on the AVX2 dev box) or singleton
        # decode. Eligibility: plain gated silu only — gpt-oss and
        # clamped variants always use the two-call path. Numerics note:
        # the fused silu is gnf4's LOCKED polynomial, not torch's sleef.
        if (getattr(self, "fused_ffn", False) and self.has_gate
                and not self.gptoss and self.clamp_limit is None
                and _act_is_plain_silu(self.act_fn)
                and hasattr(cpu_grouped, "gemm_nf4_ffn_grouped_cpu")):
            dn = cpu_grouped.gemm_nf4_ffn_grouped_cpu(
                xs, gu_p, gu_a, dn_p, dn_a,
                sizes, eids, threads=self._threads)
            dn_all = torch.empty_like(dn)
            dn_all.index_copy_(0, order, dn)
            w = top_k_weights[rows, row_slot.index_select(0, sel)].to(
                torch.float32)
            out.index_put_((rows, row_slot.index_select(0, sel)),
                           dn_all.to(dev) * w[:, None])
            return

        gu = cpu_grouped.gemv_nf4_grouped_cpu(
            xs, gu_p, gu_a, sizes, eids, threads=self._threads)
        if biases is not None:
            gu = gu + biases[0].index_select(0, sl)
            gate, up = gu.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            h = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
        elif self.has_gate:
            gate, up = gu.chunk(2, dim=-1)
            if self.clamp_limit is not None:
                gate = gate.clamp(max=self.clamp_limit)
                up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            h = self.act_fn(gate) * up
        else:
            h = self.act_fn(gu)
        dn = cpu_grouped.gemv_nf4_grouped_cpu(
            h.contiguous(), dn_p, dn_a, sizes, eids,
            threads=self._threads)
        if biases is not None:
            dn = dn + biases[1].index_select(0, sl)

        dn_all = torch.empty_like(dn)
        dn_all.index_copy_(0, order, dn)
        w = top_k_weights[rows, row_slot.index_select(0, sel)].to(torch.float32)
        out.index_put_((rows, row_slot.index_select(0, sel)),
                       dn_all.to(dev) * w[:, None])


# ---------------------------------------------------------------------- #

def _act_is_plain_silu(act_fn) -> bool:
    """True only for the unmodified silu the fused kernel implements —
    an nn.SiLU instance or torch.nn.functional.silu itself. Anything
    else (approximate gelu, custom callables) keeps the two-call path."""
    return (isinstance(act_fn, torch.nn.SiLU)
            or act_fn is torch.nn.functional.silu)


def hybrid_available() -> bool:
    """CUDA for the hot/cold buses + the native CPU kernels for the warm."""
    if not torch.cuda.is_available():
        return False
    try:
        import cpu_grouped
        return cpu_grouped.cpu_kernels_available()
    except ImportError:
        return False


def enable_hybrid_tier(model, arena_path: str, manifest, *,
                       hot_rows: int, device: str = "cuda", qd: int = 4,
                       threads: int = 0, pool: bool = True,
                       offload_rows: float | None = None,
                       fused_ffn: bool = False,
                       swappable: bool = False,
                       offload_thin_uniq: int | None = None,
                       cold_dest: str | float = "gpu",
                       protected_rows: int | None = None,
                       cold_direct: bool = True,
                       gpu_stacks_via_view: bool = True,
                       costs=None,
                       prefetch: bool = False,
                       layers: Sequence[int] | None = None,
                       verbose: bool = False) -> int:
    """Patch every MoE module per the placement manifest. Returns the patch
    count (0 = not engaged — record it, never infer from timings).

    ``manifest`` is a path or dict from ``engines.placement``. ``layers``
    maps module order to arena/manifest layer ids when they differ.

    ``cold_dest`` picks where a cold (NVMe-placed) expert EXECUTES, which
    is a separate question from where it lives:

      ``"gpu"``   default, and the pre-Stage-3 path exactly — cold rows
                  stream NVMe→pinned→H2D and run on the GPU.
      ``"cpu"``   cold rows are computed in place from the tier's own
                  packed bytes. One NVMe read either way; no PCIe crossing.
      *number*    rows-per-unique-cold-expert threshold, at or above which
                  the CPU path is taken — the DRAM tier's ``offload_rows``
                  statistic, read the other way round.

    ``protected_rows`` (<= ``hot_rows``, default ``hot_rows``) is the cold
    tier's capacity-ownership budget. Rows beyond it are RECLAIMABLE: still
    mapped and still readable, but first in line to be overwritten, so a
    request for one before its slot is reused is a **resurrection** costing
    no disk read. At the default nothing is ever reclaimable and the tier
    behaves exactly as it did pre-Stage-3 — which is also why
    ``resurrections`` reads 0 until this is set, and why R1-R10 cannot be
    measured without it.

    ``gpu_stacks_via_view`` (default True) lets a GPU-destined cold stack
    take its rows from the cold view instead of rebuilding them on every
    call. The view already knows which slots are unchanged, so a repeat costs
    no host copy — the staging gate 1 attributes ~98% of cold cost to. Pass
    False for the pre-change engine when A/B-ing.

    ``cold_direct`` (default True) asks the cold CPU view for the
    preadv-scatter landing: segments DMA straight into the kernel-shaped
    stacks instead of arena row -> ``segment_into`` -> stack. Measured ~43%
    faster on the fill path. It is not always legal — an arena that cannot
    scatter, or one storing absmax narrower than fp32, falls back to the
    copy path — so the outcome is RECORDED (``view.e4b_path``) and logged
    under ``verbose`` rather than silently chosen. Pass False to force the
    copy path for an A/B.

    It is a threshold, not a scheduler. Whether a deadline estimate beats
    it is gate 2 of the tribrid prereg and is not settled by this knob.
    Unavailable for gpt-oss (per-expert biases do not ride the arena) and
    needs a gnf4 carrying ``ColdCpuView``; both refuse by name.

    **NUMERICS — this knob is part of run identity.** It picks an execution
    destination, so the module's cross-placement rounding law (top of this
    file) applies to it exactly as it applies to a tier move: the CPU
    destination runs the native kernels' locked fp32 tree, the GPU
    destination rounds each grouped GEMM's output and the SwiGLU epilogue
    through the compute dtype.

    The trap that follows, because it reads as a cold-path defect (#171):
    ``placement.force_cold_mass`` moves experts out of the DRAM tier by
    default (``source="dram"``), and a DRAM expert EXECUTES ON THE CPU. So
    ``cold_dest="cpu"`` reproduces the pre-move arithmetic exactly while
    ``cold_dest="gpu"`` does not — an arm comparison across the two is
    reading the CPU/GPU rounding difference, not the cold path. Compare
    each destination against a MATCHED reference: ``"gpu"`` against the
    same experts placed in ``vram``, ``"cpu"`` against them in ``dram``.

    Measured, A2000 + torch 2.8.0+cu128, one layer at OLMoE-1B-7B geometry
    (H=2048, I=1024, 8 experts moved), ``bench/hybrid-g9/issue171/``::

        control_dram vs cold_cpu       0.000e+00   bitwise
        control_vram vs cold_gpu       0.000e+00   bitwise
        control_dram vs cold_gpu       4.622e-03
        control_dram vs control_vram   4.622e-03   <- no NVMe in this pair

    The last row is the whole finding: DRAM against VRAM, with no cold path
    anywhere in it, reproduces the cold arm's divergence to the digit. Both
    destinations are exact against their matched control, and both of those
    equalities are pinned in ``tests/test_hybrid_cold_dest.py``."""
    try:
        from nvme_residency import ColdTier
    except ImportError as exc:                        # pragma: no cover
        raise ImportError("hybrid tier needs grouped-nf4-gemm's N-series "
                          "modules on the import path") from exc
    import cpu_grouped
    if not cpu_grouped.cpu_kernels_available():
        raise RuntimeError("hybrid tier needs the gnf4_native CPU kernels "
                           "(no C compiler / build failed) — the DRAM bus "
                           "cannot engage")

    m = manifest if isinstance(manifest, dict) else load_manifest(manifest)
    per_layer: dict[int, dict[str, list[int]]] = {}
    for tier_name, pairs in m["tiers"].items():
        for layer, e in pairs:
            per_layer.setdefault(int(layer), {"vram": [], "dram": [],
                                              "nvme": []})[tier_name].append(int(e))

    if prefetch:
        import inspect
        if "speculative" not in inspect.signature(ColdTier.ensure).parameters:
            raise RuntimeError(
                "hybrid prefetch needs grouped-nf4-gemm with speculative "
                "ensures (ColdTier.ensure(..., speculative=)) — this gnf4 "
                "would serialize demand fetches behind prefetch disk time "
                "and can evict rows the serving thread is reading")

    tier = ColdTier(arena_path, hot_rows=hot_rows, pinned=True, qd=qd,
                    protected_rows=protected_rows)
    # A SEPARATE tier for the one-shot bulk reads that materialize the
    # resident VRAM/DRAM stacks. Two reasons, one of which is fatal:
    #   * an external-landing serving tier cannot serve row() at all, so
    #     reading setup bytes through it makes the direct cold landing
    #     impossible to attach (and attach_landing refuses a used tier);
    #   * those reads filled serving slots with experts that then live
    #     permanently in VRAM/DRAM, evicting rows the cold path wants.
    # Small on purpose -- it only needs to hold one chunk at a time -- and
    # closed before serving starts.
    setup_tier = ColdTier(arena_path, hot_rows=max(8, min(hot_rows, 64)),
                          pinned=False, qd=qd, index=tier.reader.index,
                          reader=None)

    # The cold CPU view is built ONCE per tier (its slot space is global) and
    # before any ensure(), because attach_landing refuses a tier that has
    # already filled rows into its own buffer.
    _dest = _parse_cold_dest(cold_dest)
    # A pure-GPU destination used to skip the view entirely, which made
    # `gpu_stacks_via_view` unreachable -- the stack had nothing to read. It
    # now builds one when that switch is on, in COPY mode: a non-direct view
    # owns its own buffers and never calls attach_landing, so the tier keeps
    # filling its own rows and `row()` still answers. Only the EXTERNAL
    # landing is incompatible with the GPU path, not the view itself.
    #
    # It is gated on the switch rather than hoisted unconditionally because
    # the view costs hot_rows * row_bytes of DRAM (~1.3 GB at this arena's
    # geometry with hot_rows=384). Mixed destinations already pay it; pure
    # GPU should not start paying it for a feature that is off.
    if _dest == "gpu" and gpu_stacks_via_view:
        # The direct landing is now legal here, which it was not before the
        # GPU stack learned to read the view. The old restriction was that
        # the GPU cold path reads tier.row() and an external-landing tier
        # refuses it -- but with the view serving those rows, nothing on this
        # path calls row() at all:
        #   * _build_hot reads setup bytes through _e4b_setup_tier, never the
        #     serving tier (that is what the setup tier is for);
        #   * _TieredStack takes the view for every segment it holds, and
        #     build_cold_view materializes ALL FOUR NF4 segments;
        #   * the view's dtype guard can only send it back to segment_tensor
        #     when the view widened a segment -- and direct REFUSES to cast,
        #     so a direct view's stacks are always the arena's own dtype.
        # That last clause is the load-bearing one: the fallback and the
        # external landing are mutually exclusive by construction, not by
        # timing. Verified below rather than asserted in prose.
        v = build_cold_view(tier, tier.reader.index, direct=cold_direct)
        v.e4b_serve_gpu_stacks = True
        _refuse_direct_dtype_mismatch(v, tier.reader.index)
        if verbose:
            log(f"  hybrid: cold view for GPU stacks = {v.e4b_path}")
    elif _dest != "gpu":
        # The direct landing and the GPU cold path are MUTUALLY EXCLUSIVE on
        # one tier: the GPU path reads raw rows through tier.row(), and an
        # external-landing tier never fills its own buffer, so row() refuses
        # by name. With cold_dest="cpu" no cold row ever takes the GPU path
        # and direct is safe; with "deadline", a threshold, or "auto" either
        # destination can be chosen per step, so the tier must stay able to
        # serve rows.
        #
        # Downgraded rather than refused, because the fallback is correct and
        # the cost is bounded (the copy path, ~12.5% slower on the fill path
        # at high cold mass, gnf4#122) -- but RECORDED, because a silent
        # downgrade is a silent regression.
        # A MIXED destination (deadline, a threshold, "auto") can send a row
        # either way, so it used to be barred from the direct landing for the
        # same reason pure GPU was. With the GPU stack reading the view, both
        # halves of a mixed step go through the view and neither calls row(),
        # so the bar lifts here too -- and this is where it is worth the most:
        # a deadline scheduler's CPU-routed rows were paying the copy path
        # while a pure-CPU run got the fast one, which biased the very
        # comparison the scheduler exists to make (gnf4#143).
        _direct = cold_direct and (_dest == "cpu" or gpu_stacks_via_view)
        v = build_cold_view(tier, tier.reader.index, direct=_direct)
        # Whether a GPU-destined cold stack reads this view's rows or rebuilds
        # them per call. False reproduces the pre-change engine exactly, which
        # is what makes the two A/B-able (gnf4#133's rule for DevRowCache).
        v.e4b_serve_gpu_stacks = bool(gpu_stacks_via_view)
        # A mixed destination can attach the landing too now, so it needs the
        # same check -- guarding one branch and trusting the other is how a
        # first-cold-row crash gets back in (Bugbot, e4b#185).
        _refuse_direct_dtype_mismatch(v, tier.reader.index)
        if cold_direct and not _direct:
            v.e4b_fallback_reason = (
                f"cold_dest={cold_dest!r} can route a cold row to the GPU, "
                f"which reads tier.row(); an external landing would make "
                f"that refuse. Set gpu_stacks_via_view=True so the GPU stack "
                f"reads the view instead, or use cold_dest='cpu'.")
        if verbose:
            why = getattr(v, "e4b_fallback_reason", None)
            log(f"  hybrid: cold CPU landing = {v.e4b_path}"
                + (f" ({why})" if why else ""))
    mods = target_modules(model)
    lay = list(layers) if layers is not None else list(range(len(mods)))
    if len(lay) < len(mods):
        raise ValueError(
            f"layers has {len(lay)} entries for {len(mods)} MoE module(s) — "
            f"refusing a partial map, which would stamp modules with the "
            f"wrong arena layer and silently serve another layer's weights")
    hot_sets = []
    for i, mod in enumerate(mods):
        li = lay[i]
        place = per_layer.get(li, {"vram": [], "dram": [], "nvme": []})
        mod._e4b_cold_tier = tier
        mod._e4b_arena_layer = li
        mod._e4b_hybrid_dram_ids = place["dram"]
        mod._e4b_hybrid_swappable = swappable
        mod._e4b_hybrid_threads = threads
        mod._e4b_hybrid_offload_rows = offload_rows
        mod._e4b_hybrid_fused_ffn = fused_ffn
        mod._e4b_hybrid_thin_uniq = offload_thin_uniq
        mod._e4b_hybrid_cold_dest = cold_dest
        mod._e4b_hybrid_costs = costs
        mod._e4b_setup_tier = setup_tier
        hot_sets.append(place["vram"])
    try:
        n_nodes = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
        if n_nodes > 1:
            log(f"  hybrid: {n_nodes} NUMA nodes — run under "
                f"`numactl --interleave=all` until node-local allocation "
                f"lands (see the Phase-2 metal receipt)")
    except OSError:
        pass

    try:
        n = enable_hot_residency(model, hot_sets, device=device,
                                 verbose=verbose, state_cls=_HybridTier,
                                 reach_wrapped=True)
    except Exception:
        # no half-enabled state left behind: every stamp this call made
        # comes off before the error propagates (Bugbot)
        for mod in mods:
            for attr in ("_e4b_cold_tier", "_e4b_arena_layer",
                         "_e4b_hybrid_dram_ids", "_e4b_hybrid_threads",
                         "_e4b_hybrid_offload_rows",
                         "_e4b_hybrid_fused_ffn",
                         "_e4b_hybrid_thin_uniq",
                         "_e4b_hybrid_cold_dest", "_e4b_hybrid_costs"):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        raise
    finally:
        # The setup tier's job ends with construction, on BOTH paths. Drop
        # it from every module and close it before serving, so nothing can
        # accidentally resolve residency against it later.
        for mod in mods:
            if hasattr(mod, "_e4b_setup_tier"):
                delattr(mod, "_e4b_setup_tier")
        try:
            setup_tier.close()
        except Exception:                          # noqa: BLE001
            pass
    # pool starts only after enable succeeds, and ownership is recorded so
    # disable never tears down a pool the caller started themselves (Bugbot)
    if pool:
        got = cpu_grouped.pool_start(threads)
        if verbose:
            log(f"  hybrid: CPU pool started ({got} pinned workers)")
    for mod in mods:
        setattr(mod, _MARKER, True)
        mod._e4b_hybrid_owns_pool = bool(pool)
    if prefetch:
        _wire_prefetch(model, mods, verbose=verbose)
    log(f"  hybrid tier active on {n} module(s): "
        f"vram/dram/nvme masses "
        f"{m['masses']['vram_frac']:.2f}/{m['masses']['dram_frac']:.2f}/"
        f"{m['masses']['nvme_frac']:.2f}")
    return n


def _wire_prefetch(model, mods, *, verbose=False) -> int:
    """Build the L->L+1 router chain for speculative prefetch. Router
    modules are discovered with the same per-arch table the CPU router
    uses; a count mismatch disables prefetch loudly rather than guessing
    which router belongs to which expert stack."""
    global _PF_POOL
    import concurrent.futures

    import numpy as np

    from .cpu_router import _ROUTER_KINDS

    routers = [m for _, m in model.named_modules()
               if type(m).__name__ in _ROUTER_KINDS]
    if len(routers) != len(mods):
        log(f"  hybrid prefetch DISABLED: {len(routers)} router modules "
            f"for {len(mods)} expert stacks — cannot align the chain")
        return 0
    states = [getattr(m, "_hot_residency") for m in mods]
    wired = 0
    for i in range(len(states) - 1):
        r = routers[i + 1]
        st = states[i]
        st.pf = {
            "np_w": np.ascontiguousarray(
                r.weight.detach().to("cpu", torch.float32).numpy()),
            "np_b": (None if getattr(r, "bias", None) is None else
                     np.ascontiguousarray(
                         r.bias.detach().to("cpu", torch.float32).numpy())),
            "next": states[i + 1],
            "k": int(r.top_k),
        }
        st.pf_enabled = True
        wired += 1
    if _PF_POOL is None:
        _PF_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="e4b-prefetch")
    if verbose:
        log(f"  hybrid prefetch wired on {wired} layer transitions")
    return wired


def set_prefetch(model, on: bool) -> int:
    """Cheap A/B toggle: flips the flag on already-wired states without
    rebuilding any stacks (a 235B re-enable costs minutes; this is free)."""
    n = 0
    for _, mod in model.named_modules():
        st = getattr(mod, "_hot_residency", None)
        if st is not None and getattr(st, "pf", None) is not None:
            st.pf_enabled = bool(on)
            n += 1
    return n


def prefetch_stats(model) -> dict:
    subs = rows = 0
    for _, mod in model.named_modules():
        st = getattr(mod, "_hot_residency", None)
        if st is not None:
            subs += getattr(st, "pf_submitted", 0)
            rows += getattr(st, "pf_rows", 0)
    return {"prefetch_submitted": subs, "prefetch_rows": rows}


def cold_stats(model) -> dict:
    """Where this run's cold rows were executed, and what the tier paid.

    The destination split is the falsifiability hook the tribrid prereg
    asks for: a scheduler that never flips is a placement rule, and this is
    the counter that shows it. Tier counters are summed from the cold tier
    itself (one per model), so `resurrections` and `physical_reads` here are
    the whole run's, not one layer's.
    """
    cpu_rows = gpu_rows = steps = 0
    tier = None
    for _, mod in model.named_modules():
        st = getattr(mod, "_hot_residency", None)
        if st is not None:
            cpu_rows += getattr(st, "cold_cpu_rows", 0)
            gpu_rows += getattr(st, "cold_gpu_rows", 0)
            steps += getattr(st, "cold_cpu_steps", 0)
        t = getattr(mod, "_e4b_cold_tier", None)
        if t is not None:
            tier = t
    total = cpu_rows + gpu_rows
    view = getattr(tier, "_e4b_cold_view", None) if tier is not None else None
    flips = deadline = 0
    for _, mod in model.named_modules():
        st = getattr(mod, "_hot_residency", None)
        if st is not None:
            flips += getattr(st, "deadline_flips", 0)
            deadline += len(getattr(st, "deadline_log", ()) or ())
    out = {"deadline_decisions": deadline, "deadline_flips": flips,
           "cold_landing": getattr(view, "e4b_path", None),
           "cold_landing_fallback": getattr(view, "e4b_fallback_reason", None),
           "cold_rows_cpu": cpu_rows, "cold_rows_gpu": gpu_rows,
           "cold_rows": total, "cold_cpu_steps": steps,
           "cold_cpu_frac": (cpu_rows / total) if total else 0.0}
    if tier is not None:
        ts = tier.stats()
        for k in ("resurrections", "logical_evictions", "evictions",
                  "reuse_before_overwrite", "resurrection_bytes_saved",
                  # The DENOMINATOR of reuse_before_overwrite, and the second
                  # half of its numerator. Without these a receipt carries the
                  # headline ratio but nothing to audit it against: a caller
                  # defaulting the missing keys to 0 reads
                  # "reclaimable_overwritten: 0", which says the ratio should
                  # be 1.0 and silently contradicts the reported value.
                  # Resolved evictions are also strictly fewer than
                  # logical_evictions (rows still sitting reclaimable have
                  # resolved neither way), so logical_evictions cannot stand
                  # in for the denominator either.
                  "spec_resurrections", "reclaimable_overwritten",
                  "protected_rows", "reclaimable_rows",
                  "disk_reads", "disk_bytes"):
            if k in ts:
                out[k] = ts[k]
    return out


def disable_hybrid_tier(model) -> int:
    """Restore forwards and remove EVERY stamped attribute (the
    enable_nvme_residency teardown gap is the counterexample)."""
    from .hot_residency import disable_hot_residency
    import cpu_grouped

    # Quiesce the prefetch worker BEFORE stripping module attributes: an
    # in-flight task reads mod._e4b_arena_layer, and deleting attrs first
    # turns those tasks into AttributeErrors parked on unretrieved futures
    # during teardown. Disable new fires, then drain what is queued.
    global _PF_POOL
    set_prefetch(model, False)
    if _PF_POOL is not None:
        _PF_POOL.submit(lambda: None).result()    # barrier: queue drained

    owns_pool = False
    n = disable_hot_residency(model)
    for _, mod in model.named_modules():
        owns_pool = owns_pool or getattr(mod, "_e4b_hybrid_owns_pool", False)
        for attr in (_MARKER, "_e4b_cold_tier", "_e4b_arena_layer",
                     "_e4b_hybrid_dram_ids", "_e4b_hybrid_threads",
                     "_e4b_hybrid_owns_pool", "_e4b_hybrid_offload_rows",
                     "_e4b_hybrid_cold_dest", "_e4b_hybrid_costs",
                     "_e4b_hybrid_fused_ffn", "_e4b_hybrid_thin_uniq"):
            if hasattr(mod, attr):
                delattr(mod, attr)
    if owns_pool:
        try:
            cpu_grouped.pool_stop()
        except (ImportError, RuntimeError):
            pass
    if _PF_POOL is not None:
        _PF_POOL.shutdown(wait=True)
        _PF_POOL = None
    return n
