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
    """The native kernel's decode contract caps a group at 8 rows; a prefill
    group larger than that is split into same-expert chunks (pure row
    batching — outputs are per-row, so chunking cannot change them)."""
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

    def forward(self, hidden_states, top_k_index, top_k_weights):
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
        if pool is None or pool._work_queue.qsize() > 2:
            return                          # never queue into lag
        h = hidden.detach()

        def task():
            import numpy as np
            hc = h.float().cpu().numpy()
            logits = hc @ pf["np_w"].T
            if pf["np_b"] is not None:
                logits = logits + pf["np_b"]
            k = pf["k"]
            ids = set()
            for row in logits:
                ids.update(np.argpartition(-row, k)[:k].tolist())
            want = sorted(ids & nxt.nvme_set)
            if want:
                nxt._tier.ensure(int(nxt.mod._e4b_arena_layer), want)
                self.pf_rows += len(want)
            self.pf_submitted += 1

        try:
            pool.submit(task)
        except RuntimeError:
            pass                            # executor shut down mid-flight

    # ------------------------------------------------------------------ #
    def _cold_contrib(self, x, flat, row_token, row_slot, cr, top_k_weights,
                      out, dev):
        dmask = self.is_dram[flat.index_select(0, cr)]
        nr = cr[~dmask]
        dr = cr[dmask]
        if nr.numel():                       # NVMe→GPU, parent's path verbatim
            super()._cold_contrib(x, flat, row_token, row_slot, nr,
                                  top_k_weights, out, dev)
        if dr.numel():                       # DRAM bus: compute in place
            self._dram_contrib(x, flat, row_token, row_slot, dr,
                               top_k_weights, out, dev)

    def _dram_contrib(self, x, flat, row_token, row_slot, dr, top_k_weights,
                      out, dev):
        import cpu_grouped

        glob = flat.index_select(0, dr).cpu()
        local = self.g2d_cpu.index_select(0, glob)
        rows = row_token.index_select(0, dr)
        xr = x.index_select(0, rows).to("cpu", torch.float32)

        order = torch.argsort(local)
        sl = local.index_select(0, order)
        xs = xr.index_select(0, order).contiguous()
        uniq, counts = torch.unique_consecutive(sl, return_counts=True)
        sizes, eids = _split_oversize_groups(counts.tolist(), uniq.tolist())

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
        out.index_add_(0, rows, dn_all.to(dev) * w[:, None])


# ---------------------------------------------------------------------- #

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
                                 verbose=verbose, state_cls=_HybridTier)
    except Exception:
        # no half-enabled state left behind: every stamp this call made
        # comes off before the error propagates (Bugbot)
        for mod in mods:
            for attr in ("_e4b_cold_tier", "_e4b_arena_layer",
                         "_e4b_hybrid_dram_ids", "_e4b_hybrid_threads"):
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

    owns_pool = False
    n = disable_hot_residency(model)
    for _, mod in model.named_modules():
        owns_pool = owns_pool or getattr(mod, "_e4b_hybrid_owns_pool", False)
        for attr in (_MARKER, "_e4b_cold_tier", "_e4b_arena_layer",
                     "_e4b_hybrid_dram_ids", "_e4b_hybrid_threads",
                     "_e4b_hybrid_owns_pool"):
            if hasattr(mod, attr):
                delattr(mod, attr)
    if owns_pool:
        try:
            cpu_grouped.pool_stop()
        except (ImportError, RuntimeError):
            pass
    global _PF_POOL
    if _PF_POOL is not None:
        _PF_POOL.shutdown(wait=True)
        _PF_POOL = None
    return n
