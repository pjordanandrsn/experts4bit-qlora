# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P66 Level L -- the residency launch census over one MoE token (bench/p66/P66-PREREG.md).

A "token" here is one decode step's worth of expert work: every MoE layer of a real checkpoint's expert
bytes, in order, each called exactly as the model calls it (``forward(hidden [1, H], top_k_index [1, k],
top_k_weights [1, k])``). Attention, norms, the router and the LM head are left out on purpose: residency
does not touch them, so they cancel in every delta this lane reports, and leaving them out is what lets the
routing -- and therefore the cold fraction -- be SET rather than hoped for (#108's method).

Per arm (a path at a hot fraction) and per window (a routing condition) it records, for the same tokens:

- ``eager``: a torch.profiler window. CUDA API calls are counted only inside ``e4b::p66.token`` regions
  (the instrument's closing sync is outside by construction) and attributed to the engine's own phase
  regions (``e4b::p66.fetch`` etc.), wrapped around the engine's methods for the profiled passes only.
  Device kernels (count + self device time) come from the same window's device view, and the window is
  also written as a key-averages table that ``bench/hybrid-g9/f1/step_budget.py`` parses unchanged.
- ``graph``: the whole token captured as ONE CUDA graph and replayed, profiled the same way -- where
  capture works. Where the engine refuses capture, the refusal text is the receipt.
- a ``torch.cuda.set_sync_debug_mode("warn")`` pass, warnings aggregated by call site;
- an unprofiled timing pass (eager, and graph where captured);
- the realized routing: cold lanes, cold rows actually copied (the pipelined have-skip is simulated from
  the ids fed, then checked against the engine's own counters in a separate pass), and cold_deadline's
  predicted transfer time for exactly those rows.

Paths (``--path``):

- ``ref``       all-resident step: ``enable_hybrid_tier`` with every expert in VRAM and the certified
                all-resident collapse (``collapse_resident=True``), i.e. step_decomp's R0 arm.
- ``pipe``      ``enable_pipelined_residency`` at each ``--hot-fracs`` (one process, one pinning).
- ``hyb``       ``enable_hybrid_tier`` with the hot fraction in VRAM and the rest on NVMe (``--dram-frac``
                moves part of the cold set to the CPU tier instead).
- ``mref``      MXFP4: ``Mxfp4PipelinedGptOss`` with every expert hot (all-resident, capturable).
- ``mpin``      MXFP4: ``Mxfp4PipelinedGptOss`` at each hot fraction (pinned all-E host arena, UVA).
- ``mnvme``     MXFP4: ``Mxfp4NvmeResidency`` at each hot fraction (the NVMe tier; refuses capture).

Rows go to ``<out>/rows_<arm>.json``; ``p66_reduce.py`` reads them.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

# Triton bound at MODULE scope (the probe's kernel annotations are strings under `from __future__ import
# annotations`, which triton resolves against the function's __globals__; a local import raised NameError on
# triton 3.2 -- experts4bit_qlora/engines/pipelined.py:65-73). Optional: the CPU tests import this file.
try:
    import triton
    import triton.language as tl
except ImportError:                                         # pragma: no cover
    triton = tl = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p66_reduce as red  # noqa: E402  (staged beside this file; pure python)

SCHEMA = "p66-census/1"
R_TOKEN, R_LAYER, R_INPUT = "e4b::p66.token", "e4b::p66.layer", "e4b::p66.input"
R_FETCH, R_RESOLVE, R_INVALIDATE, R_COLD = ("e4b::p66.fetch", "e4b::p66.resolve",
                                            "e4b::p66.invalidate", "e4b::p66.cold")
PHASES = (R_LAYER, R_INPUT, R_FETCH, R_RESOLVE, R_INVALIDATE, R_COLD)


def say(*a):
    print("[p66]", *a, flush=True)


# =========================================================================== routing
def hot_sets_for(L: int, E: int, n_hot: int, seed: int) -> list:
    """Per-layer hot ids: the first n_hot of a seeded permutation. Same seed -> same sets on every path,
    so a hot fraction means the same experts whichever engine holds them."""
    out = []
    for li in range(L):
        g = torch.Generator().manual_seed(seed * 1000 + li)
        out.append(sorted(torch.randperm(E, generator=g)[:n_hot].tolist()))
    return out


def make_routing(T: int, L: int, E: int, k: int, *, hot_sets=None, cold_lanes=None, seed: int = 0):
    """``[T, L, k]`` expert ids (int64) and bf16 weights, on the CPU.

    ``cold_lanes is None``: k distinct experts uniform over E (the realized cold fraction then follows the
    hot set). Otherwise exactly ``cold_lanes`` of the k come from the layer's cold set and the rest from its
    hot set, lane order shuffled so hot and cold lanes land on every slot position (the pipelined have-skip
    is per slot)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty(T, L, k, dtype=torch.long)
    for li in range(L):
        hot = hot_sets[li] if hot_sets is not None else None
        cold = [e for e in range(E) if hot is None or e not in set(hot)]
        for t in range(T):
            if cold_lanes is None:
                ids[t, li] = torch.randperm(E, generator=g)[:k]
                continue
            c = int(cold_lanes)
            if c > len(cold) or (k - c) > len(hot):
                raise ValueError(f"layer {li}: {c} cold lanes from {len(cold)} cold, {k - c} hot from {len(hot)} hot")
            pc = torch.tensor(cold)[torch.randperm(len(cold), generator=g)[:c]]
            ph = torch.tensor(hot, dtype=torch.long)[torch.randperm(len(hot), generator=g)[:k - c]]
            lane = torch.cat([pc, ph])
            ids[t, li] = lane[torch.randperm(k, generator=g)]
    w = torch.softmax(torch.randn(T, L, k, generator=g), dim=-1).to(torch.bfloat16)
    return ids, w


# =========================================================================== profiler extraction
def _is_cpu(ev) -> bool:
    dt = getattr(ev, "device_type", None)
    return dt is None or str(dt).endswith("CPU")


def region_counts(events) -> dict:
    """Walk the profiler's CPU event tree under every ``e4b::p66.token`` region.

    Returns API counts for the token scope, per phase INCLUSIVE (an event under fetch->resolve counts for
    both), per phase NEAREST, the aten ops seen, the device kernels linked to CPU events by phase, and the
    number of token regions found (the divisor is checked against it, never assumed)."""
    api = collections.Counter()
    incl = collections.defaultdict(collections.Counter)
    near = collections.defaultdict(collections.Counter)
    ops = collections.Counter()
    kern_near = collections.defaultdict(collections.Counter)
    n_tok = 0

    def walk(ev, stack):
        for c in ev.cpu_children:
            nm = c.name
            st = stack + [nm] if nm in PHASES else stack
            ph = st[-1] if st else R_TOKEN
            if red.api_class(nm) is not None:
                api[nm] += 1
                near[ph][nm] += 1
                for p in set(st) or {R_TOKEN}:
                    incl[p][nm] += 1
            elif nm.startswith("aten::"):
                ops[nm] += 1
            for kk in (getattr(c, "kernels", None) or []):
                kern_near[ph][kk.name] += 1
            walk(c, st)

    for ev in events:
        if ev.name == R_TOKEN and _is_cpu(ev) and (ev.cpu_parent is None or ev.cpu_parent.name != R_TOKEN):
            n_tok += 1
            walk(ev, [])
    return {"api": dict(api), "api_by_phase": {k: dict(v) for k, v in incl.items()},
            "api_by_phase_nearest": {k: dict(v) for k, v in near.items()},
            "ops": dict(ops), "kernels_by_phase": {k: dict(v) for k, v in kern_near.items()},
            "token_regions": n_tok}


def api_global(events) -> dict:
    """Every CUDA API event in the window, region or not. The region-scoped count must equal this minus
    the instrument's own closing sync; a gap means API events failed to nest under the token regions
    (kineto links them to the enclosing op by thread), and the region counts would be under-counts."""
    c = collections.Counter()
    for ev in events:
        if _is_cpu(ev) and red.api_class(ev.name) is not None:
            c[ev.name] += 1
    return dict(c)


def device_rows(prof) -> dict:
    """Device-view rows (kernels, memcpy, memset) of the whole window: count + self device time (us).
    User-annotation spans (``e4b::`` / ``ProfilerStep``) are regions, not work, and are dropped."""
    out = {}
    for e in prof.key_averages():
        if str(getattr(e, "device_type", "")).endswith("CPU"):
            continue
        name = e.key
        if name.startswith(("e4b::", "ProfilerStep")):
            continue
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0.0)
        if e.count <= 0:
            continue
        r = out.setdefault(name, {"count": 0, "self_device_us": 0.0})
        r["count"] += int(e.count)
        r["self_device_us"] += float(t)
    return out


def write_budget_table(prof, path: Path, tokens: int):
    """The window as step_decomp's kernels.txt, so step_budget.py reads it unchanged."""
    try:
        tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=400)
    except Exception:                                       # noqa: BLE001 - torch renamed the sort key
        tbl = prof.key_averages().table(sort_by="self_device_time_total", row_limit=400)
    path.write_text(f"profiled decode steps: {tokens} (active window: {tokens}/{tokens})\n" + tbl)


# =========================================================================== phase regions
@contextlib.contextmanager
def phase_regions(active: bool):
    """Wrap the engines' own methods in record_function regions -- for the profiled passes only, so the
    timing passes run the engine exactly as shipped. Every wrapper is removed on exit."""
    if not active:
        yield
        return
    from torch.autograd.profiler import record_function
    patches = []

    def wrap(cls, meth, region):
        orig = cls.__dict__.get(meth)
        if orig is None:
            return

        def w(self, *a, **kw):
            with record_function(region):
                return orig(self, *a, **kw)
        setattr(cls, meth, w)
        patches.append((cls, meth, orig))

    try:
        from experts4bit_qlora.engines.pipelined import _PipelinedResidency
        wrap(_PipelinedResidency, "_fetch", R_FETCH)
    except ImportError:
        pass
    try:
        from experts4bit_qlora.engines.hybrid import _HybridTier
        wrap(_HybridTier, "_cold_contrib", R_COLD)
    except ImportError:
        pass
    try:
        from mxfp4_pipelined import Mxfp4PipelinedGptOss
        wrap(Mxfp4PipelinedGptOss, "_fetch", R_FETCH)
        from mxfp4_residency import Mxfp4NvmeResidency
        wrap(Mxfp4NvmeResidency, "_resolve_src", R_RESOLVE)
        wrap(Mxfp4NvmeResidency, "_invalidate", R_INVALIDATE)
    except ImportError:
        pass
    try:
        yield
    finally:
        for cls, meth, orig in reversed(patches):
            setattr(cls, meth, orig)


# =========================================================================== the token
class Driver:
    """Feeds tokens to an arm and appends every EXECUTED token's routing to the arm's feed log, in order.

    The log is per ARM, not per window: the engine's slot state (the pipelined ``have``) carries across
    windows, so the have-skip simulator must replay everything the engine actually saw since it was built."""

    def __init__(self, arm, ids_cpu, ids_dev, w_dev, xs, log):
        self.arm, self.ids_cpu, self.ids, self.w, self.xs = arm, ids_cpu, ids_dev, w_dev, xs
        self.L = len(xs)
        self.log = log                # [L][k] id lists, one entry per executed token, in execution order
        self.idx_st = torch.zeros_like(ids_dev[0]).unsqueeze(1)    # [L, 1, k]
        self.w_st = torch.zeros_like(w_dev[0]).unsqueeze(1)
        self.graph = None
        self.graph_outs = None
        self.graph_nodes = None
        self.graph_dot_head = None

    def token(self, t: int, regions: bool = False):
        from torch.autograd.profiler import record_function
        tc = record_function(R_TOKEN) if regions else contextlib.nullcontext()
        with tc, torch.no_grad():
            for li in range(self.L):
                lc = record_function(R_LAYER) if regions else contextlib.nullcontext()
                with lc:
                    self.arm.fwd(li, self.xs[li], self.ids[t, li].unsqueeze(0), self.w[t, li].unsqueeze(0))
        self.log.append(self.ids_cpu[t].tolist())

    def _stage(self, t: int, regions: bool):
        from torch.autograd.profiler import record_function
        ic = record_function(R_INPUT) if regions else contextlib.nullcontext()
        with ic:
            self.idx_st.copy_(self.ids[t].unsqueeze(1))
            self.w_st.copy_(self.w[t].unsqueeze(1))

    def capture(self, t_warm: int):
        """Warm on a side stream, then capture the whole L-layer token into one graph. The warm calls
        EXECUTE (they are logged); the captured call does not."""
        self._stage(t_warm, False)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                for li in range(self.L):
                    self.arm.fwd(li, self.xs[li], self.idx_st[li], self.w_st[li])
                self.log.append(self.ids_cpu[t_warm].tolist())
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # keep_graph=True keeps the captured cudaGraph_t alive past capture, so its node list can be READ: the exact
        # count of what one replay runs. The profiler's device view of a replay can lose a record (A2000 rehearsal
        # round 5), so captured counts come from the graph. Two readers: the CUDA runtime (cudaGraphGetNodes +
        # cudaGraphNodeGetType) and torch's own dot dump (cudaGraphDebugDotPrint), which must agree (G4).
        # Without keep_graph, torch 2.8's capture_end drops the graph and debug_dump() silently writes nothing
        # (the rehearsal's first probe); the dump also destroys the kept graph, so it comes after instantiate().
        g = torch.cuda.CUDAGraph(keep_graph=True)
        g.enable_debug_mode()
        with torch.cuda.graph(g), torch.no_grad():
            self.graph_outs = [self.arm.fwd(li, self.xs[li], self.idx_st[li], self.w_st[li])
                               for li in range(self.L)]
        nodes = runtime_graph_nodes(g)
        g.instantiate()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dot = Path(td) / "token.dot"
            g.debug_dump(str(dot))
            text = dot.read_text() if dot.exists() else ""
        parsed = red.parse_graph_dot(text) if text else None
        if nodes is None:
            nodes = parsed                       # the runtime reader was unavailable; the dot is then the count
        elif parsed is not None:
            nodes["dot"] = parsed
        if nodes is not None:
            nodes["dot_bytes"] = len(text)
        self.graph = g
        self.graph_nodes = nodes
        self.graph_dot_head = text[:4000]

    def graph_check(self, t: int) -> dict:
        """Replay against eager on the SAME token. A graph that baked a stale pointer replays happily while
        returning garbage (engines/capture.py), so a captured count is only read when this holds."""
        with torch.no_grad():
            eager = [self.arm.fwd(li, self.xs[li], self.ids[t, li].unsqueeze(0), self.w[t, li].unsqueeze(0)).clone()
                     for li in range(self.L)]
        self.log.append(self.ids_cpu[t].tolist())
        self.replay(t)
        torch.cuda.synchronize()
        d = max(float((a.float() - b.float()).abs().max()) for a, b in zip(eager, self.graph_outs))
        ref = max(float(a.float().abs().max()) for a in eager)
        return {"token": t, "bitwise": all(torch.equal(a, b) for a, b in zip(eager, self.graph_outs)),
                "max_abs": d, "b_rel": d / ref if ref > 0 else None}

    def replay(self, t: int, regions: bool = False):
        from torch.autograd.profiler import record_function
        tc = record_function(R_TOKEN) if regions else contextlib.nullcontext()
        with tc:
            self._stage(t, regions)
            self.graph.replay()
        self.log.append(self.ids_cpu[t].tolist())


_CUDA_GRAPH_NODE_TYPES = {0: "KERNEL", 1: "MEMCPY", 2: "MEMSET", 3: "HOST", 4: "GRAPH", 5: "EMPTY", 6: "WAIT_EVENT",
                          7: "EVENT_RECORD", 8: "EXT_SEMAS_SIGNAL", 9: "EXT_SEMAS_WAIT", 10: "MEM_ALLOC",
                          11: "MEM_FREE", 13: "CONDITIONAL"}


def _libcudart():
    import ctypes
    import glob
    cands = ["libcudart.so.12"] + sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*")) + sorted(
        glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*")))
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


def runtime_graph_nodes(g) -> dict | None:
    """Node counts by type of a kept (not yet dumped) CUDA graph, read through the CUDA runtime. None when the
    runtime or the handle is unavailable -- the caller then counts from the dot dump alone."""
    import ctypes
    rt = _libcudart()
    if rt is None or not hasattr(g, "raw_cuda_graph"):
        return None
    h = ctypes.c_void_p(g.raw_cuda_graph())
    n = ctypes.c_size_t(0)
    if rt.cudaGraphGetNodes(h, None, ctypes.byref(n)) != 0:
        return None
    arr = (ctypes.c_void_p * n.value)()
    if rt.cudaGraphGetNodes(h, arr, ctypes.byref(n)) != 0:
        return None
    counts: dict[str, int] = {}
    for i in range(n.value):
        t = ctypes.c_int(-1)
        rt.cudaGraphNodeGetType(ctypes.c_void_p(arr[i]), ctypes.byref(t))
        k = _CUDA_GRAPH_NODE_TYPES.get(t.value, f"TYPE_{t.value}")
        counts[k] = counts.get(k, 0) + 1
    work = sum(counts.get(t, 0) for t in red.WORK_NODE_TYPES)
    return {"by_type": counts, "work": work, "nodes": n.value, "source": "cudaGraphGetNodes"}


def profile_window(drv: Driver, toks, graph: bool, table_path: Path | None):
    from torch.profiler import ProfilerActivity, profile
    run = drv.replay if graph else drv.token
    torch.cuda.synchronize()
    with phase_regions(not graph):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            for t in toks:
                run(t, regions=True)
            torch.cuda.synchronize()              # outside every token region: never counted
    rc = region_counts(prof.events())
    if rc["token_regions"] != len(toks):
        raise RuntimeError(f"found {rc['token_regions']} token regions for {len(toks)} tokens -- the event "
                           "tree does not nest as assumed, so no count here would mean what it says")
    rec = {"tokens": len(toks), **rc, "api_global": api_global(prof.events()), "kernels": device_rows(prof)}
    if table_path is not None:
        write_budget_table(prof, table_path, len(toks))
        rec["table"] = table_path.name
    return rec


def sync_debug_window(drv: Driver, toks):
    """torch's own sync detector over the same kind of tokens, by call site (innermost frame in the engines)."""
    import traceback as tb
    sites = collections.Counter()
    n = [0]
    outside = collections.Counter()
    active = [False]
    keep = ("experts4bit_qlora", "mxfp4_", "nvme_", "nf4_grouped", "cold_cpu_view", "cpu_grouped", "host_gather")

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return                    # not the sync detector (a deprecation etc.): not a sync
        if not active[0]:
            # raised while the instrument toggles the mode (torch/cuda/__init__.py's set_sync_debug_mode,
            # seen once per process on the A2000 rehearsal) -- recorded, never counted as the engine's
            outside[f"{Path(filename).name}:{lineno}"] += 1
            return
        n[0] += 1
        fr = [f for f in tb.extract_stack()[:-1] if any(k in f.filename for k in keep)]
        site = f"{Path(fr[-1].filename).name}:{fr[-1].lineno} {fr[-1].name}" if fr else f"{Path(filename).name}:{lineno}"
        sites[site] += 1

    torch.cuda.synchronize()
    old = warnings.showwarning
    warnings.showwarning = show
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            torch.cuda.set_sync_debug_mode("warn")
            try:
                active[0] = True
                for t in toks:
                    drv.token(t)
            finally:
                active[0] = False
                torch.cuda.set_sync_debug_mode("default")
    finally:
        warnings.showwarning = old
    torch.cuda.synchronize()
    return {"tokens": len(toks), "warnings": n[0], "per_token": n[0] / len(toks),
            "by_site": dict(sites.most_common()), "outside_tokens": dict(outside)}


def time_window(drv: Driver, toks, graph: bool):
    run = drv.replay if graph else drv.token
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    e0.record()
    for t in toks:
        run(t)
    t_sub = time.perf_counter()
    e1.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n = len(toks)
    return {"tokens": n, "wall_ms_per_token": (t1 - t0) * 1e3 / n,
            "submit_ms_per_token": (t_sub - t0) * 1e3 / n,
            "gpu_ms_per_token": e0.elapsed_time(e1) / n}


# =========================================================================== arms: NF4 (e4b engines)
def _nf4_meta_modules(index: dict, L: int):
    from experts4bit_qlora.engines.nvme_experts import build_meta_experts
    E = int(index["n_experts_per_layer"])
    return [build_meta_experts(index, E, compute_dtype=torch.bfloat16) for _ in range(L)]


def materialize_from_arena(mods, arena_path: str, layer_ids):
    """step_decomp's ``_materialize_from_arena`` (PREREG-b1 R1 mechanics), restated for this lane: fill the
    meta modules' expert tensors from the SAME arena file on the host -- identical packed bytes, per-segment
    lengths asserted. The pipelined engine sources its pinned arena from module tensors."""
    import numpy as np
    import torch.nn as nn
    from nvme_arena import _seg_len, _seg_off, load_index, row_offset
    idx = load_index(arena_path)
    seg_map = (("gate_up_proj", "nf4.gate_up_blocks", True), ("gate_up_absmax", "nf4.gate_up_absmax", False),
               ("down_proj", "nf4.down_blocks", True), ("down_absmax", "nf4.down_absmax", False))
    mm = np.memmap(arena_path, dtype=np.uint8, mode="r")
    for mi, base in enumerate(mods):
        E, li = base.num_experts, layer_ids[mi]
        for attr, suffix, is_param in seg_map:
            meta = getattr(base, attr)
            t = torch.empty(meta.shape, dtype=meta.dtype)
            flat = t.view(torch.uint8).reshape(E, -1) if t.dtype != torch.uint8 else t.reshape(E, -1)
            off, ln = _seg_off(idx, suffix), _seg_len(idx, suffix)
            assert flat.shape[1] == ln, (attr, tuple(meta.shape), flat.shape[1], ln)
            for e in range(E):
                lo = row_offset(idx, li, e) + off
                flat[e] = torch.from_numpy(np.ascontiguousarray(mm[lo:lo + ln]))
            setattr(base, attr, nn.Parameter(t, requires_grad=False) if is_param else t)
    del mm


def _manifest(L, E, layer_ids, vram, dram):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for i, li in enumerate(layer_ids):
        v, d = set(vram[i]), set(dram[i]) if dram else set()
        for e in range(E):
            tiers["vram" if e in v else ("dram" if e in d else "nvme")].append([li, e])
    n = L * E
    return {"tiers": tiers, "masses": {"vram_frac": len(tiers["vram"]) / n, "dram_frac": len(tiers["dram"]) / n,
                                       "nvme_frac": len(tiers["nvme"]) / n}}


class HybridArm:
    """``ref`` (all-VRAM, collapse) and ``hyb`` (hot in VRAM, cold on NVMe / DRAM)."""
    path = "hybrid"
    residency_phase = R_COLD

    def __init__(self, arena, index, layer_ids, vram_sets, dram_sets, hot_rows, threads):
        import torch.nn as nn
        from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
        L, E = len(layer_ids), int(index["n_experts_per_layer"])
        self.mods = _nf4_meta_modules(index, L)
        self.model = nn.ModuleList(self.mods)
        man = _manifest(L, E, layer_ids, vram_sets, dram_sets)
        self.masses = man["masses"]
        n = enable_hybrid_tier(self.model, arena, man, hot_rows=hot_rows, threads=threads,
                               pool=bool(dram_sets and any(dram_sets)), collapse_resident=True,
                               layers=layer_ids)
        assert n == L, f"hybrid patched {n}/{L}"
        self.states = [m._hot_residency for m in self.mods]
        self.tier = self.mods[0]._e4b_cold_tier
        self.capturable = all(st._all_hot() for st in self.states)

    def fwd(self, li, x, idx, w):
        return self.mods[li](x, idx, w)

    def tier_stats(self):
        return self.tier.stats()

    def close(self):
        from experts4bit_qlora.engines.hybrid import disable_hybrid_tier
        disable_hybrid_tier(self.model)
        with contextlib.suppress(Exception):
            self.tier.close()


class PipeArm:
    path = "pipelined"
    residency_phase = R_FETCH
    capturable = True

    def __init__(self, mods, hot_sets, k):
        import torch.nn as nn
        from experts4bit_qlora.engines.pipelined import enable_pipelined_residency
        self.mods = mods
        self.model = nn.ModuleList(mods)
        n = enable_pipelined_residency(self.model, [torch.tensor(h, dtype=torch.long) for h in hot_sets],
                                       device="cuda", k_slots=k)
        assert n == len(mods), f"pipelined patched {n}/{len(mods)}"
        self.states = [m._pipelined for m in mods]
        for st in self.states:
            st.count_traffic = False
        self.row_bytes = self.states[0].row_bytes

    def fwd(self, li, x, idx, w):
        return self.mods[li](x, idx, w)

    def counters(self):
        return sum(int(st.cold_pcie_bytes.item()) for st in self.states) // self.row_bytes

    def close(self):
        from experts4bit_qlora.engines.pipelined import disable_pipelined_residency
        disable_pipelined_residency(self.model)   # drops mod._pipelined; the pinned arena cache survives
        self.states = []


# =========================================================================== arms: MXFP4 (gnf4 engines)
def gptoss_layer_tensors(snapshot: str, li: int):
    """One layer's MXFP4 expert tensors + biases from the checkpoint, as the engines take them."""
    from safetensors import safe_open
    wm = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.layers.{li}.mlp.experts."
    out = {}
    for nm in ("gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
               "down_proj_blocks", "down_proj_scales", "down_proj_bias"):
        with safe_open(str(Path(snapshot) / wm[pre + nm]), framework="pt") as f:
            out[nm] = f.get_tensor(pre + nm)
    E, n1 = out["gate_up_proj_blocks"].shape[:2]
    n2 = out["down_proj_blocks"].shape[1]
    return {"gu_b": out["gate_up_proj_blocks"].reshape(E, n1, -1).contiguous(),
            "gu_s": out["gate_up_proj_scales"].reshape(E, n1, -1).contiguous(),
            "dn_b": out["down_proj_blocks"].reshape(E, n2, -1).contiguous(),
            "dn_s": out["down_proj_scales"].reshape(E, n2, -1).contiguous(),
            "gu_bias": out["gate_up_proj_bias"].to(torch.bfloat16),
            "dn_bias": out["down_proj_bias"].to(torch.bfloat16)}


class MxArm:
    """``mref``/``mpin`` (Mxfp4PipelinedGptOss, pinned all-E arena) and ``mnvme`` (Mxfp4NvmeResidency)."""
    residency_phase = R_FETCH

    def __init__(self, kind, snapshot, arena, index, layer_ids, hot_sets, k, hot_rows, tensors_cache):
        self.kind = kind
        self.engines = []
        self.path = "mxfp4-nvme" if kind == "mnvme" else "mxfp4-pinned"
        if kind == "mnvme":
            from mxfp4_residency import Mxfp4NvmeResidency
            for i, li in enumerate(layer_ids):
                tb = tensors_cache(li)
                prev = self.engines[0] if self.engines else None
                self.engines.append(Mxfp4NvmeResidency(
                    arena, li, hot_ids=hot_sets[i], k_slots=k, hot_rows=hot_rows,
                    gate_up_bias=tb["gu_bias"], down_bias=tb["dn_bias"], device="cuda", index=index,
                    tier=prev.tier if prev else None, store=prev.store if prev else None))
            self.tier = self.engines[0].tier
            self.capturable = False
        else:
            from mxfp4_pipelined import Mxfp4PipelinedGptOss
            for i, li in enumerate(layer_ids):
                tb = tensors_cache(li)
                prev = self.engines[0] if self.engines else None
                self.engines.append(Mxfp4PipelinedGptOss(
                    tb["gu_b"], tb["gu_s"], tb["dn_b"], tb["dn_s"], tb["gu_bias"], tb["dn_bias"],
                    hot_sets[i], k, device="cuda", store=prev.store if prev else None))
            self.tier = None
            self.capturable = True
        self.row_bytes = self.engines[0].row_bytes

    def fwd(self, li, x, idx, w):
        return self.engines[li].forward(x, idx, w)

    def traffic(self):
        c = h = 0
        for e in self.engines:
            c += int(e.cold_pcie_bytes.item())
            h += int(e.hot_d2d_bytes.item())
        return {"cold_rows": c // self.row_bytes, "hot_d2d_rows": h // self.row_bytes}

    def tier_stats(self):
        return self.tier.stats() if self.tier is not None else None

    def close(self):
        if self.tier is not None:
            with contextlib.suppress(Exception):
                self.tier.close()
        self.engines = []


# =========================================================================== box record
def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                                  # noqa: BLE001
        return f"<{type(e).__name__}>"


def host_record() -> dict:
    """Who measured: the host is part of the reading (#108: host cost varied ~5x across boxes)."""
    rec = {"platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
           "triton": getattr(triton, "__version__", None),
           "cuda": torch.version.cuda, "cpu_model": _sh("lscpu | sed -n 's/^Model name:[ ]*//p' | head -1"),
           "nproc": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "cgroup_cpu_max": _sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"),
           "cgroup_memory_max": _sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"),
           "meminfo": _sh("grep -E 'MemTotal|MemAvailable' /proc/meminfo | tr -s ' '"),
           "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
           if Path("/sys/devices/system/node").exists() else None,
           "gpu": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,"
                      "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,clocks.max.sm "
                      "--format=csv,noheader"),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}
    return rec


def host_probes() -> dict:
    """Unit host costs on THIS box: what one launch and one sync cost the CPU, plus the link and VRAM
    bandwidths the transfer model uses when no calibration blob is given."""
    dev = torch.device("cuda")
    x = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    for _ in range(200):
        x.add_(1)
    torch.cuda.synchronize()
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    aten_us = (time.perf_counter() - t0) * 1e6 / n
    torch.cuda.synchronize()
    tri_us = None
    try:
        if triton is None:
            raise ImportError("triton")

        @triton.jit
        def _nop(p, BLOCK: tl.constexpr):
            o = tl.arange(0, BLOCK)
            tl.store(p + o, tl.load(p + o) + 1)
        y = torch.zeros(16, device=dev, dtype=torch.int32)
        for _ in range(200):
            _nop[(1,)](y, BLOCK=16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            _nop[(1,)](y, BLOCK=16)
        tri_us = (time.perf_counter() - t0) * 1e6 / n
        torch.cuda.synchronize()
    except Exception as e:                                  # noqa: BLE001
        tri_us = f"<{type(e).__name__}: {e}>"[:200]
    m = 2000
    t0 = time.perf_counter()
    for _ in range(m):
        torch.cuda.synchronize()
    sync_us = (time.perf_counter() - t0) * 1e6 / m
    t0 = time.perf_counter()
    for _ in range(m):
        x.item()
    item_us = (time.perf_counter() - t0) * 1e6 / m
    # link + VRAM (medians; the calibration blob is preferred when present -- this is the fallback)
    h = torch.empty(64 << 20, dtype=torch.uint8).pin_memory()
    d = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    ts = []
    for _ in range(12):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    h2d = (64 << 20) / statistics.median(ts[2:]) / 1e9
    a = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    ts = []
    for _ in range(8):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    d2d = 2 * (1 << 30) / statistics.median(ts[2:]) / 1e9
    del h, d, a, b
    torch.cuda.empty_cache()
    return {"aten_launch_host_us": aten_us, "triton_launch_host_us": tri_us, "sync_idle_us": sync_us,
            "item_roundtrip_us": item_us, "h2d_pinned_64mb_gbs": h2d, "d2d_copy_rw_gbs": d2d}


def costs_for(calib_path: str | None, probes: dict) -> dict:
    """cold_deadline's constants: from the calibration blob by the model's own field names when given
    (``Costs.from_blob``), else from this run's probes (recorded as such)."""
    if calib_path:
        blob = json.loads(Path(calib_path).read_text())
        dev = blob["gpu_bench"]["devices"]
        return {"source": f"calib:{Path(calib_path).name}",
                "b_link_gbs": max(d["b_link"]["h2d_64mb"]["gbs"] for d in dev),
                "b_vram_gbs": max(d["b_vram_triad_gbs"] for d in dev),
                "b_dram_gbs": blob["cpu_bench"]["triad_best"]["gbs"]}
    return {"source": "probe", "b_link_gbs": probes["h2d_pinned_64mb_gbs"],
            "b_vram_gbs": probes["d2d_copy_rw_gbs"], "b_dram_gbs": None}


def predicted_us(rows_per_layer_token, row_bytes: int, costs: dict) -> float:
    """Sum over layers of cold_deadline.gpu_us for each layer-token's cold rows, mean over tokens. Uses the
    installed package's own function; the reducer's restatement is only for CPU tests."""
    try:
        from cold_deadline import Costs, gpu_us
        c = Costs(cpu_us_fixed=0.0, cpu_us_per_row=0.0, b_dram_gbs=costs.get("b_dram_gbs") or 1.0,
                  b_vram_gbs=costs["b_vram_gbs"], b_link_gbs=costs["b_link_gbs"], bytes_per_expert=row_bytes)
        f = lambda r: gpu_us(r, r, c)                       # noqa: E731  one row per distinct expert (T=1)
    except ImportError:
        f = lambda r: red.gpu_us(r, row_bytes, costs["b_link_gbs"], costs["b_vram_gbs"])  # noqa: E731
    tot = [sum(f(int(r)) for r in tok) for tok in rows_per_layer_token]
    return sum(tot) / len(tot) if tot else 0.0


# =========================================================================== one arm, all windows
PASSES_PER_WINDOW = 6          # profiled, sync-debug, timing, graph-profiled, graph-timing, traffic check


def tokens_needed(warm: int, n: int) -> int:
    """Every pass takes FRESH tokens, so no pass replays ids another pass already made resident
    (+1 the capture's warm token, +1 the graph-vs-eager check)."""
    return warm + PASSES_PER_WINDOW * n + 2


def run_arm(a, arm, name, family, L, E, k, hot_frac, n_hot, hot_sets, ids_by_window, xs, costs, row_bytes,
            out: Path, reference: bool):
    rec = {"schema": SCHEMA, "arm": name, "family": family, "path": arm.path, "hot_frac": hot_frac,
           "n_hot": n_hot, "layers": L, "experts": E, "k": k, "row_bytes": row_bytes, "reference": reference,
           "residency_phase": getattr(arm, "residency_phase", None), "capturable": bool(arm.capturable),
           "windows": []}
    hot = [set(h) for h in hot_sets]
    log = []                                    # every token this arm's engines executed, in order

    def simulated(positions):
        """Per layer-token cold rows the pipelined engine copied, at log positions (from the whole log)."""
        per_layer = [red.simulate_pipelined_traffic([log[j][li] for j in range(len(log))], hot[li], k)
                     for li in range(L)]
        return [[per_layer[li][j] for li in range(L)] for j in positions]

    for wname, (ids_cpu, w_cpu) in ids_by_window.items():
        say(f"{name}: window {wname}")
        drv = Driver(arm, ids_cpu, ids_cpu.cuda(), w_cpu.cuda(), xs, log)
        T, W, N = ids_cpu.shape[0], a.warm, a.tokens
        if T < tokens_needed(W, N):
            raise ValueError(f"routing has {T} tokens, the window schedule needs {tokens_needed(W, N)}")
        seq = iter(range(T))

        def take(n):
            return [next(seq) for _ in range(n)]

        wrec = {"name": wname, "modes": {}}
        tier0 = arm.tier_stats() if hasattr(arm, "tier_stats") else None
        for t in take(W):
            drv.token(t)
        mx0 = arm.traffic() if hasattr(arm, "traffic") else None
        prof_toks = take(N)
        p0 = len(log)
        wrec["modes"]["eager"] = profile_window(drv, prof_toks, False, out / f"table_{name}_{wname}_eager.txt")
        prof_pos = list(range(p0, len(log)))
        mx1 = arm.traffic() if hasattr(arm, "traffic") else None
        wrec["sync_debug"] = sync_debug_window(drv, take(N))
        timing = {"eager": time_window(drv, take(N), False)}
        # ---- realized routing over the profiled eager tokens
        lanes = [log[j] for j in prof_pos]
        cold_lanes = sum(1 for tok in lanes for li, ln in enumerate(tok) for e in ln if e not in hot[li])
        routing = {"cold_fraction": cold_lanes / (len(prof_toks) * L * k),
                   "cold_lanes_per_token": cold_lanes / len(prof_toks), "lanes_per_token": L * k}
        if arm.path == "pipelined":
            rows_lt = simulated(prof_pos)      # the have-skip: a cold lane already in its slot copies nothing
        else:
            # hybrid streams every routed cold expert each token; the MXFP4 engines share ONE slot store
            # across layers and poison it on every handover, so every cold lane is gathered
            rows_lt = [[sum(1 for e in ln if e not in hot[li]) for li, ln in enumerate(tok)] for tok in lanes]
        if mx0 is not None:
            routing["counters_cold_rows_per_token"] = (mx1["cold_rows"] - mx0["cold_rows"]) / len(prof_toks)
            routing["counters_hot_d2d_rows_per_token"] = (mx1["hot_d2d_rows"] - mx0["hot_d2d_rows"]) / len(prof_toks)
        routing["cold_rows_per_token"] = sum(map(sum, rows_lt)) / len(rows_lt)
        routing["cold_rows_per_layer_token"] = rows_lt
        wrec["predicted_transfer_us_per_token"] = predicted_us(rows_lt, row_bytes, costs)
        wrec["routing"] = routing
        # ---- graph, where the engine captures
        if arm.capturable and not a.no_graph:
            try:
                drv.capture(take(1)[0])
                wrec["graph_nodes"] = drv.graph_nodes
                if not (out / f"graph_{name}.dot.head.txt").exists():   # one sample of the dump per arm
                    (out / f"graph_{name}.dot.head.txt").write_text(drv.graph_dot_head)
                wrec["modes"]["graph"] = profile_window(drv, take(N), True, out / f"table_{name}_{wname}_graph.txt")
                timing["graph"] = time_window(drv, take(N), True)
                wrec["graph_check"] = drv.graph_check(take(1)[0])
            except Exception as e:                          # noqa: BLE001 - the refusal IS the receipt
                wrec["modes"]["graph"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            finally:
                drv.graph = drv.graph_outs = None
                torch.cuda.synchronize()
        elif not arm.capturable:
            wrec["modes"]["graph"] = {"refused": getattr(arm, "capture_refusal", "engine is not capturable")}
        wrec["timing"] = timing
        if tier0 is not None:
            tier1 = arm.tier_stats()
            wrec["tier"] = {kk: (tier1[kk] - tier0[kk]) for kk in ("requests", "hits", "misses", "disk_reads",
                                                                    "disk_bytes", "demand_fill_ns", "evictions")
                            if isinstance(tier1.get(kk), (int, float)) and isinstance(tier0.get(kk), (int, float))}
        # ---- the pipelined simulator, checked against the engine's own counters. A separate pass: the
        # counters add launches of their own, so they never run inside a counted window.
        if arm.path == "pipelined":
            chk = take(N)
            c0 = arm.counters()
            for st in arm.states:
                st.count_traffic = True
            q0 = len(log)
            for t in chk:
                drv.token(t)
            torch.cuda.synchronize()
            for st in arm.states:
                st.count_traffic = False
            c1 = arm.counters()
            sim = sum(map(sum, simulated(range(q0, len(log)))))
            wrec["traffic_check"] = {"tokens": len(chk), "simulated_cold_rows": sim, "counted_cold_rows": c1 - c0,
                                     "match": sim == (c1 - c0)}
            say(f"{name}/{wname}: have-skip simulator {sim} vs engine counters {c1 - c0} cold rows")
        rec["windows"].append(wrec)
        del drv
        torch.cuda.empty_cache()
    (out / f"rows_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    say(f"{name}: wrote rows_{name}.json")
    return rec


def capture_probe(arm, xs, k) -> dict:
    """Try to capture one token on an engine that does not claim capture, and NAME what stops it.

    First under ``set_sync_debug_mode("error")`` OUTSIDE capture: the first synchronizing op raises with a
    Python traceback, so the innermost engine frame is the op that makes capture impossible (a failed capture
    alone reports only "operation failed due to a previous error during capture", which names nothing). Then
    the capture itself, for its own message. Run LAST in its process: a capture that dies on a sync can leave
    the context unusable, and nothing after it is measured."""
    import traceback as tb
    ids = torch.zeros(len(xs), 1, k, dtype=torch.long, device="cuda")
    for li in range(len(xs)):
        ids[li, 0] = torch.arange(k, device="cuda")
    w = torch.full((len(xs), 1, k), 1.0 / k, dtype=torch.bfloat16, device="cuda")
    keep = ("experts4bit_qlora", "mxfp4_", "nvme_", "nf4_grouped", "cold_cpu_view", "cpu_grouped")
    out = {}
    with torch.no_grad():
        for li in range(len(xs)):
            arm.fwd(li, xs[li], ids[li], w[li])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.no_grad():
            arm.fwd(0, xs[0], ids[0], w[0])
        out["first_sync"] = None
    except Exception as e:                                  # noqa: BLE001
        fr = [f for f in tb.extract_tb(e.__traceback__) if any(kk in f.filename for kk in keep)]
        out["first_sync"] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}",
                             "site": (f"{Path(fr[-1].filename).name}:{fr[-1].lineno} {fr[-1].name}: "
                                      f"{(fr[-1].line or '').strip()[:160]}") if fr else None}
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            for li in range(len(xs)):
                arm.fwd(li, xs[li], ids[li], w[li])
        out["capture"] = "CAPTURED (unexpected for this engine -- recorded, not relied on)"
    except Exception as e:                                  # noqa: BLE001
        out["capture"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
    return out


# =========================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, help="label for the rows, e.g. qwen3-30b-a3b / gpt-oss-20b")
    ap.add_argument("--path", required=True, choices=["ref", "pipe", "hyb", "mref", "mpin", "mnvme"])
    ap.add_argument("--arena", required=True, help="NF4 arena (ref/pipe/hyb) or MXFP4 arena (mnvme; its index "
                    "also fixes the layer list for mref/mpin)")
    ap.add_argument("--snapshot", required=True, help="the checkpoint dir: config.json (k, hidden) and, for the "
                    "MXFP4 paths, the expert tensors + biases")
    ap.add_argument("--layers", default="all", help="'all' or a-b: which arena layers make the token")
    ap.add_argument("--hot-fracs", default="1.0", help="comma list; each is one arm in this process")
    ap.add_argument("--controlled", default="", help="comma list of cold-lane counts for extra windows at "
                    "hot fractions strictly between 0 and 1")
    ap.add_argument("--dram-frac", type=float, default=0.0, help="hyb: share of the COLD set on the CPU tier")
    ap.add_argument("--hot-rows", type=int, default=64, help="pinned tier rows (hybrid / mnvme)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=8, help="tokens per pass (profiled, sync-debug, timing)")
    ap.add_argument("--seed", type=int, default=66)
    ap.add_argument("--calib", default=None, help="gnf4 calibration blob for cold_deadline's constants")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--capture-probe", action="store_true", help="after the arms, try capture on a "
                    "non-capturable engine and record the refusal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    assert torch.cuda.is_available(), "the census needs CUDA"

    from nvme_arena import load_index
    index = load_index(a.arena)
    layer_ids = sorted({int(r[0]) for r in index["rows"]})
    if a.layers != "all":
        lo, hi = (int(v) for v in a.layers.split("-"))
        layer_ids = [li for li in layer_ids if lo <= li <= hi]
    L = len(layer_ids)
    E = int(index["n_experts_per_layer"])
    cfg = json.loads((Path(a.snapshot) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    k = int(cfg["num_experts_per_tok"])
    H = int(cfg["hidden_size"])

    box_p = out / "box.json"
    box = json.loads(box_p.read_text()) if box_p.exists() else {}
    if "probes" not in box:
        box = {"schema": SCHEMA, "host": host_record(), "probes": host_probes()}
    costs = costs_for(a.calib, box["probes"])
    box["costs"] = costs
    box_p.write_text(json.dumps(box, indent=1, default=str))
    say(f"host {box['host']['cpu_model']} | gpu {box['host']['gpu']} | costs {costs}")

    fracs = [float(v) for v in a.hot_fracs.split(",")] if a.path not in ("ref", "mref") else [1.0]
    controlled = [int(v) for v in a.controlled.split(",") if v.strip()]
    g = torch.Generator().manual_seed(a.seed + 7)
    xs = [(torch.randn(1, H, generator=g) * 0.1).to(torch.bfloat16).cuda() for _ in range(L)]
    T = tokens_needed(a.warm, a.tokens)
    cache = {}

    def tensors(li):
        if li not in cache:
            cache[li] = gptoss_layer_tensors(a.snapshot, li)
        return cache[li]

    base_mods = None
    arm = None
    for hf in fracs:
        if arm is not None:                    # free the previous arm BEFORE building the next one
            arm.close()
            arm = None
            gc.collect()
            torch.cuda.empty_cache()
        n_hot = int(round(hf * E))
        hot_sets = hot_sets_for(L, E, n_hot, a.seed) if a.path not in ("ref", "mref") else [list(range(E))] * L
        windows = {"uniform": make_routing(T, L, E, k, seed=a.seed + 101)}
        if 0 < n_hot < E:
            for c in controlled:
                if c <= E - n_hot and (k - c) <= n_hot:
                    windows[f"ctrl{c}"] = make_routing(T, L, E, k, hot_sets=hot_sets, cold_lanes=c,
                                                       seed=a.seed + 200 + c)
        name = {"ref": "ref", "mref": "mref"}.get(a.path, f"{a.path}-{hf:.2f}")
        say(f"arm {name}: L={L} E={E} k={k} n_hot={n_hot} windows={list(windows)}")
        t0 = time.time()
        if a.path == "ref":
            arm = HybridArm(a.arena, index, layer_ids, hot_sets, None, a.hot_rows, a.threads)
            row_bytes = int(index["row_bytes"])
        elif a.path == "hyb":
            cold = [[e for e in range(E) if e not in set(h)] for h in hot_sets]
            dram = [c[:int(round(a.dram_frac * len(c)))] for c in cold] if a.dram_frac > 0 else None
            arm = HybridArm(a.arena, index, layer_ids, hot_sets, dram, a.hot_rows, a.threads)
            arm.capture_refusal = ("the v0 dispatch under the hybrid tier synchronizes (nonzero / tolist / "
                                   "tier reads), which capture forbids; only the all-VRAM collapse captures")
            row_bytes = int(index["row_bytes"])
        elif a.path == "pipe":
            if base_mods is None:
                base_mods = _nf4_meta_modules(index, L)
                materialize_from_arena(base_mods, a.arena, layer_ids)
            arm = PipeArm(base_mods, hot_sets, k)
            row_bytes = arm.row_bytes
        else:
            arm = MxArm(a.path, a.snapshot, a.arena, index, layer_ids, hot_sets, k, a.hot_rows, tensors)
            if a.path == "mnvme":
                arm.capture_refusal = ("Mxfp4NvmeResidency._resolve_src reads the wanted ids to the host to "
                                       "issue disk reads and raises inside capture by design")
            row_bytes = arm.row_bytes
        setup_s = time.time() - t0
        rec = run_arm(a, arm, name, a.family, L, E, k, hf, n_hot, hot_sets, windows, xs, costs, row_bytes, out,
                      reference=a.path in ("ref", "mref"))
        rec["setup_s"] = setup_s
        if hasattr(arm, "masses"):
            rec["masses"] = arm.masses
        (out / f"rows_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    if a.capture_probe and arm is not None and not arm.capturable:
        res = capture_probe(arm, xs, k)
        say(f"capture probe on {a.path}: {res}")
        (out / f"capture_probe_{a.path}.json").write_text(json.dumps(res, indent=1) + "\n")
    if arm is not None:
        arm.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(32)
