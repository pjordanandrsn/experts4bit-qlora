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
        if dram_ids.numel():
            if int(dram_ids.min()) < 0 or int(dram_ids.max()) >= E:
                raise ValueError(f"dram ids outside [0, {E})")
            overlap = set(dram_ids.tolist()) & set(self.hot_ids.tolist())
            if overlap:
                raise ValueError(f"experts placed in BOTH vram and dram: "
                                 f"{sorted(overlap)[:8]}")
        self.dram_ids = dram_ids
        self._threads = int(getattr(mod, "_e4b_hybrid_threads", 0))

        tier = mod._e4b_cold_tier
        index = tier.reader.index
        layer = int(getattr(mod, "_e4b_arena_layer", 0))
        ids = [int(e) for e in dram_ids.tolist()]
        chunk = max(1, min(len(ids) or 1, tier.hot_rows))
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
            parts = [segment_tensor(tier, index, layer, ids[i:i + chunk], suffix)
                     for i in range(0, len(ids), chunk)]
            stacked = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            # pageable host memory ON PURPOSE — the DRAM-bus law
            setattr(self, attr, stacked.contiguous().cpu())
        if self.d_gu_a.dtype != torch.float32:
            self.d_gu_a = self.d_gu_a.float()
        if self.d_dn_a.dtype != torch.float32:
            self.d_dn_a = self.d_dn_a.float()

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
        hot_u = self.is_hot[uniq]
        dram_u = self.is_dram[uniq]
        a["uniq_vram"] += int(hot_u.sum())
        a["uniq_dram"] += int((dram_u & ~hot_u).sum())
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
        if nr.numel():                       # NVMe→GPU, parent's path verbatim
            super()._cold_contrib(x, flat, row_token, row_slot, nr,
                                  top_k_weights, out, dev)
        if dr.numel():
            gpu_route = getattr(self, "_gpu_only", False)
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
        import cpu_grouped

        glob = flat.index_select(0, dr).cpu()
        local = self.g2d_cpu.index_select(0, glob)
        rows = row_token.index_select(0, dr)
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
        if self.amort is not None:
            # post-split group count, not the unique count: each split
            # chunk re-reads its expert's weights, so THIS is the number
            # the DRAM bus actually pays. The gap between it and
            # uniq_dram is the split tax, and it must be visible.
            self.amort["dram_groups"] += len(sizes)

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
                xs, self.d_gu_p, self.d_gu_a, self.d_dn_p, self.d_dn_a,
                sizes, eids, threads=self._threads)
            dn_all = torch.empty_like(dn)
            dn_all.index_copy_(0, order, dn)
            w = top_k_weights[rows, row_slot.index_select(0, dr)].to(
                torch.float32)
            out.index_put_((rows, row_slot.index_select(0, dr)),
                           dn_all.to(dev) * w[:, None])
            return

        gu = cpu_grouped.gemv_nf4_grouped_cpu(
            xs, self.d_gu_p, self.d_gu_a, sizes, eids, threads=self._threads)
        if self.gptoss:
            gu = gu + self.d_gu_b.index_select(0, sl)
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
            h.contiguous(), self.d_dn_p, self.d_dn_a, sizes, eids,
            threads=self._threads)
        if self.gptoss:
            dn = dn + self.d_dn_b.index_select(0, sl)

        dn_all = torch.empty_like(dn)
        dn_all.index_copy_(0, order, dn)
        w = top_k_weights[rows, row_slot.index_select(0, dr)].to(torch.float32)
        out.index_put_((rows, row_slot.index_select(0, dr)),
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
                       prefetch: bool = False,
                       layers: Sequence[int] | None = None,
                       verbose: bool = False) -> int:
    """Patch every MoE module per the placement manifest. Returns the patch
    count (0 = not engaged — record it, never infer from timings).

    ``manifest`` is a path or dict from ``engines.placement``. ``layers``
    maps module order to arena/manifest layer ids when they differ."""
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

    tier = ColdTier(arena_path, hot_rows=hot_rows, pinned=True, qd=qd)
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
        mod._e4b_hybrid_threads = threads
        mod._e4b_hybrid_offload_rows = offload_rows
        mod._e4b_hybrid_fused_ffn = fused_ffn
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
                         "_e4b_hybrid_fused_ffn"):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        raise
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
                     "_e4b_hybrid_fused_ffn"):
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
