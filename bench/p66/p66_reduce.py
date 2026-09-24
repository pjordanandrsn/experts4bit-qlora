# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P66 reducer: the census's raw profiler aggregates -> launches, syncs and kernels per token, what
residency adds against the all-resident step, whether that moves with cold fraction, and cold_deadline's
transfer prediction scored against the measured transfer. Applies bench/p66/P66-PREREG.md literally.

Pure Python on purpose: no torch, no GPU, no gnf4. ``tests/test_p66_reduce.py`` drives every counting
rule and every verdict branch with synthetic event lists, so the reading on the box is the same code
the CI checked.

Three ways a count lies, all guarded here (the step_budget lessons, applied to API events):

1. **The instrument's own calls.** The census synchronizes once to close each profiled window. Every
   count here is scoped to events inside a ``p66::token`` region, so that sync is outside the scope by
   construction instead of subtracted by hand (the pipelined ladder's sync audit once counted its own
   timing syncs as the engine's).
2. **Host launches are not device kernels.** Under graph replay the host issues one
   ``cudaGraphLaunch`` per token while the device still runs every kernel. Both are reported and never
   summed: ``launches`` are host API calls, ``kernels`` are device executions.
3. **A blocking copy is a sync without the word "Synchronize".** ``cudaMemcpy`` (no ``Async``) blocks
   the host; torch's own blocking copies issue ``cudaMemcpyAsync`` + ``cudaStreamSynchronize``, which
   the explicit list already counts. Both kinds are kept apart so neither is double-counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- API-name classes
# CUDA runtime/driver calls as kineto names them. Triton launches through the DRIVER (cuLaunchKernel /
# cuLaunchKernelEx); torch's own kernels through the runtime (cudaLaunchKernel). A list that holds only the
# runtime name under-counts every Triton kernel -- which is the whole residency engine.
LAUNCH_APIS = frozenset({
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice", "cuLaunchKernel", "cuLaunchKernelEx",
    "cudaLaunchKernel_ptsz", "cudaLaunchKernelExC_ptsz", "cuLaunchKernel_ptsz",
})
SYNC_APIS = frozenset({
    "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
    "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize",
})
# Blocking copies/sets: the host waits, so each is also a sync (counted in `blocking`, not in `syncs_explicit`).
BLOCKING_COPY_APIS = frozenset({
    "cudaMemcpy", "cudaMemcpy2D", "cudaMemset", "cuMemcpyHtoD", "cuMemcpyHtoD_v2", "cuMemcpyDtoH",
    "cuMemcpyDtoH_v2", "cuMemcpy", "cuMemcpyDtoD", "cuMemcpyDtoD_v2",
})
ASYNC_COPY_APIS = frozenset({
    "cudaMemcpyAsync", "cudaMemcpy2DAsync", "cudaMemcpyToSymbolAsync", "cudaMemcpyFromSymbolAsync",
    "cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2",
    "cuMemcpyAsync", "cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2", "cudaMemcpyAsync_ptsz",
})
MEMSET_APIS = frozenset({"cudaMemsetAsync", "cuMemsetD8Async", "cuMemsetD32Async", "cuMemsetD8_v2",
                         "cuMemsetD32_v2", "cudaMemsetAsync_ptsz"})
GRAPH_APIS = frozenset({"cudaGraphLaunch", "cuGraphLaunch", "cudaGraphLaunch_ptsz"})

# Device-side names. The gather is the one kernel that reads over the link (UVA); a Memcpy HtoD row is
# a copy engine moving host bytes. Together they are the "transfer" the cost model predicts.
GATHER_KERNEL_TOKENS = ("_gather_rows_addr", "_gather_rows_perm")
H2D_ROW_PREFIX = "Memcpy HtoD"


# CUDA graph node types as cudaGraphDebugDotPrint names them. Work nodes are what an eager token would submit.
WORK_NODE_TYPES = ("KERNEL", "MEMCPY", "MEMSET")
GRAPH_NODE_TYPES = WORK_NODE_TYPES + ("HOST", "GRAPH", "EMPTY", "WAIT_EVENT", "EVENT_RECORD", "EXT_SEMAS_SIGNAL",
                                      "EXT_SEMAS_WAIT", "MEM_ALLOC", "MEM_FREE", "BATCH_MEM_OP", "CONDITIONAL")


def parse_graph_dot(text: str) -> dict:
    """Node counts by type from a ``torch.cuda.CUDAGraph.debug_dump`` (cudaGraphDebugDotPrint) DOT file.

    The format, as the A2000 rehearsal's probe printed it (torch 2.8.0, CUDA 12.8)::

        "graph_1_node_0"[style="bold" shape="record" label="{KERNEL
        | {ID | 0 (topoId: 3) | _ZN2at6native29vectorized_elementwise_kernel...}
        ...}"];
        "graph_1_node_2"[style="solid" shape="record" label="{
        MEMCPY
        | {{ID | node handle} | ...}"];
        "graph_1_node_0" -> "graph_1_node_1" [headlabel=0];

    Only node STATEMENTS are counted (a quoted id at line start followed directly by ``[``); edges and cluster
    headers are not. A node's type is the type keyword that appears FIRST in its attribute block: the printer
    writes it at the head of the label, and kernel names are mangled lower-case C++ symbols. A node with no type
    keyword counts as ``UNKNOWN``, which fails G4 rather than hiding."""
    import re
    counts: dict[str, int] = {}
    kw = re.compile(r"(?<![A-Za-z_])(" + "|".join(GRAPH_NODE_TYPES) + r")(?![A-Za-z_])")
    for m in re.finditer(r'^\s*"([^"]+)"\s*\[(.*?)\];?[ \t]*$', text, flags=re.M | re.S):
        hit = kw.search(m.group(2))
        kind = hit.group(1) if hit else "UNKNOWN"
        counts[kind] = counts.get(kind, 0) + 1
    work = sum(counts.get(t, 0) for t in WORK_NODE_TYPES)
    return {"by_type": counts, "work": work, "nodes": sum(counts.values())}


def api_class(name: str) -> str | None:
    """Which census bucket a CPU-side CUDA API event belongs to, or None when it is not one we count."""
    if name in LAUNCH_APIS:
        return "launch"
    if name in SYNC_APIS:
        return "sync"
    if name in BLOCKING_COPY_APIS:
        return "blocking"
    if name in ASYNC_COPY_APIS:
        return "memcpy"
    if name in MEMSET_APIS:
        return "memset"
    if name in GRAPH_APIS:
        return "graph"
    return None


def kernel_class(name: str) -> str:
    """Device rows: the gather, H2D copies, other copies/sets, and everything the GPU computes."""
    if any(t in name for t in GATHER_KERNEL_TOKENS):
        return "gather"
    if name.startswith(H2D_ROW_PREFIX):
        return "h2d"
    if name.startswith("Memcpy") or name.startswith("Memset"):
        return "copy"
    return "compute"


# --------------------------------------------------------------------------- per-token counts
def per_token(mode: dict) -> dict:
    """One profiled window -> per-token rates. ``mode`` is the census's raw record:

    ``tokens``   tokens inside the window (the divisor; never the profiler's step count)
    ``api``      {api_name: count} for events INSIDE p66::token regions
    ``api_by_phase`` {phase: {api_name: count}} -- the same events under each enclosing ``e4b::p66.*``
                 phase region (INCLUSIVE: an event under fetch->resolve counts in both)
    ``kernels``  {kernel_name: {"count": n, "self_device_us": t}} over the window (device view)
    """
    n = int(mode["tokens"])
    if n <= 0:
        raise ValueError(f"non-positive token count {n}")
    cls: dict[str, float] = {"launch": 0, "sync": 0, "blocking": 0, "memcpy": 0, "memset": 0, "graph": 0}
    for name, c in mode.get("api", {}).items():
        k = api_class(name)
        if k is not None:
            cls[k] += c
    phases = {}
    for ph, apis in mode.get("api_by_phase", {}).items():
        pc = {"launch": 0, "sync": 0, "blocking": 0, "memcpy": 0, "memset": 0, "graph": 0}
        for name, c in apis.items():
            k = api_class(name)
            if k is not None:
                pc[k] += c
        phases[ph] = {k: v / n for k, v in pc.items()}
    kc = {"gather": [0, 0.0], "h2d": [0, 0.0], "copy": [0, 0.0], "compute": [0, 0.0]}
    for name, r in mode.get("kernels", {}).items():
        b = kc[kernel_class(name)]
        b[0] += r["count"]
        b[1] += r["self_device_us"]
    dev_n = sum(v[0] for v in kc.values())
    dev_us = sum(v[1] for v in kc.values())
    return {
        "tokens": n,
        # host side: what the CPU issued
        "launches": cls["launch"] / n,
        "syncs": (cls["sync"] + cls["blocking"]) / n,
        "syncs_explicit": cls["sync"] / n,
        "blocking_copies": cls["blocking"] / n,
        "async_copies": cls["memcpy"] / n,
        "memsets": cls["memset"] / n,
        "graph_launches": cls["graph"] / n,
        # every host->device submission, whatever its kind: the number a host launch floor multiplies
        "submissions": (cls["launch"] + cls["memcpy"] + cls["blocking"] + cls["memset"] + cls["graph"]) / n,
        # device side: what the GPU executed
        "kernels": dev_n / n,
        "device_us": dev_us / n,
        "kernels_by_class": {k: v[0] / n for k, v in kc.items()},
        "device_us_by_class": {k: v[1] / n for k, v in kc.items()},
        "transfer_us": (kc["gather"][1] + kc["h2d"][1]) / n,
        "phases": phases,
    }


def per_name_delta(arm: dict, ref: dict, key: str) -> dict:
    """Per-name difference in per-token calls, arm minus reference, over ``key`` ('api' or 'kernels').
    Names absent on one side count as zero there. Zero rows are dropped."""
    na, nr = int(arm["tokens"]), int(ref["tokens"])

    def rate(m, name, n):
        v = m.get(key, {}).get(name, 0)
        return (v["count"] if isinstance(v, dict) else v) / n

    names = set(arm.get(key, {})) | set(ref.get(key, {}))
    out = {}
    for name in names:
        d = rate(arm, name, na) - rate(ref, name, nr)
        if abs(d) > 1e-9:
            out[name] = d
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def split_device_delta(arm: dict, ref: dict) -> dict:
    """Split the arm-minus-reference device time per token into the part owed to kernels whose CALL COUNT
    differs (what residency adds or removes: its launches' own device cost) and the part owed to kernels both
    run equally often (the same kernel running slower or faster on the arm's data layout -- the RFC's
    "hot-bank geometry" suspect). The two never mix: a kernel is in one bucket by its count, not its time."""
    na, nr = int(arm["tokens"]), int(ref["tokens"])
    ka, kr = arm.get("kernels", {}), ref.get("kernels", {})
    changed = shared = 0.0
    shared_rows = {}
    for name in set(ka) | set(kr):
        ca = ka.get(name, {}).get("count", 0) / na
        cr = kr.get(name, {}).get("count", 0) / nr
        dt = ka.get(name, {}).get("self_device_us", 0.0) / na - kr.get(name, {}).get("self_device_us", 0.0) / nr
        if abs(ca - cr) > 1e-9:
            changed += dt
        else:
            shared += dt
            if abs(dt) > 1e-9:
                shared_rows[name] = dt
    return {"count_changed_us": changed, "count_equal_us": shared,
            "count_equal_by_kernel": dict(sorted(shared_rows.items(), key=lambda kv: -abs(kv[1])))}


def attribute(arm_mode: dict, ref_mode: dict, layers: int, residency_phase: str | None = None) -> dict:
    """What the residency path adds per token (and per layer) against the all-resident step, same mode.

    ``by_phase``: the arm's own per-phase counts. The engine's fetch region is residency by definition
    (table lookup, gather, have update, tier resolution); everything else in the layer is compute and glue
    that the all-resident step also runs, so the delta outside fetch is the engine's DIFFERENT compute
    path, reported separately rather than folded into "residency"."""
    a, r = per_token(arm_mode), per_token(ref_mode)
    d = {k: a[k] - r[k] for k in ("launches", "syncs", "syncs_explicit", "blocking_copies", "async_copies",
                                  "memsets", "graph_launches", "submissions", "kernels", "device_us",
                                  "transfer_us")}
    fetch = a["phases"].get(residency_phase, {}) if residency_phase else {}
    return {
        "delta_per_token": d,
        "delta_per_layer": {k: v / layers for k, v in d.items()},
        "fetch_per_token": fetch,
        "fetch_per_layer": {k: v / layers for k, v in fetch.items()},
        "by_api": per_name_delta(arm_mode, ref_mode, "api"),
        "by_kernel": per_name_delta(arm_mode, ref_mode, "kernels"),
        "device_split": split_device_delta(arm_mode, ref_mode),
        "arm": a, "ref": r,
    }


# --------------------------------------------------------------------------- does it move with cold fraction
def moves_with_cold(points: list) -> dict:
    """``points``: [(realized_cold_fraction, per_token_value), ...] for ONE engine configuration's windows.

    A structurally fixed count is the same integer in every window, so its spread is exactly zero -- no
    tolerance is needed or used for counts. The least-squares slope (per unit cold fraction, i.e. per
    100 %) is reported beside it so a moving count carries its size, not only its sign."""
    if len(points) < 2:
        return {"n": len(points), "spread": 0.0, "slope": None, "fixed": None}
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    spread = max(ys) - min(ys)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx > 0 else None
    return {"n": len(points), "spread": spread, "slope": slope, "fixed": spread <= 1e-9,
            "xs": xs, "ys": ys}


# --------------------------------------------------------------------------- the pipelined have-skip
def simulate_pipelined_traffic(fetches: list, hot: set, k: int, prime_expert: int = 0) -> list:
    """Cold rows the pipelined engine copies per fetch, from the ids it was fed -- exactly its rule.

    ``_PipelinedResidency._fetch``: a hot lane is read in place (its ``have`` is left alone); a cold lane
    copies iff the slot's ``have`` identity differs from the wanted expert's, then ``have`` becomes it.
    ``_prime`` fills every slot with expert 0's identity. Returns one count per fetch.

    Kept here, pure, so the census can derive per-token cold traffic without the engine's counters --
    which add launches of their own and would contaminate the very window they measure."""
    have = [prime_expert] * k
    out = []
    for want in fetches:
        if len(want) != k:
            raise ValueError(f"fetch of {len(want)} ids for k={k}")
        c = 0
        for i, e in enumerate(want):
            if e in hot:
                continue
            if have[i] != e:
                c += 1
                have[i] = e
        out.append(c)
    return out


# --------------------------------------------------------------------------- cold_deadline's transfer model
def gpu_us(uniq: int, bytes_per_expert: int, b_link_gbs: float, b_vram_gbs: float) -> float:
    """grouped-nf4-gemm ``kernel/cold_deadline.py::gpu_us`` (bytes over the link plus bytes over VRAM).
    Re-stated so the reducer runs without the package; ``test_gpu_us_matches_cold_deadline`` holds the two
    equal when the package is importable."""
    if uniq <= 0:
        return 0.0
    gb = uniq * bytes_per_expert / 1e9
    return (gb / max(b_link_gbs, 1e-9) + gb / max(b_vram_gbs, 1e-9)) * 1e6


def transfer_score(measured_us: float, predicted_us: float) -> dict:
    """measured / predicted, with the zero-traffic case named instead of divided."""
    if predicted_us <= 0:
        return {"ratio": None, "measured_us": measured_us, "predicted_us": predicted_us,
                "note": "no predicted traffic"}
    return {"ratio": measured_us / predicted_us, "measured_us": measured_us, "predicted_us": predicted_us}


# --------------------------------------------------------------------------- loading
def load_rows(root: Path) -> tuple[dict, list]:
    box = json.loads((root / "box.json").read_text())
    arms = [json.loads(p.read_text()) for p in sorted(root.glob("rows_*.json"))]
    if not arms:
        raise SystemExit(f"no rows_*.json under {root}")
    return box, arms


def _arms_by_family(arms: list) -> dict:
    fam = {}
    for a in arms:
        fam.setdefault(a["family"], []).append(a)
    return fam


def _ref_of(arms: list) -> dict | None:
    refs = [a for a in arms if a.get("reference")]
    return refs[0] if refs else None


def _window(arm: dict, name: str) -> dict | None:
    for w in arm.get("windows", []):
        if w["name"] == name:
            return w
    return None


def _mode(w: dict | None, mode: str) -> dict | None:
    if w is None:
        return None
    m = w.get("modes", {}).get(mode)
    return m if (m and "refused" not in m and "error" not in m) else None


# --------------------------------------------------------------------------- the read
def census(box: dict, arms: list) -> dict:
    """Every arm x window x mode -> per-token rows, deltas against the family's reference, and the
    cold-fraction test per engine configuration. No verdicts here; ``verdicts`` reads this."""
    out = {"families": {}}
    for fam, fa in _arms_by_family(arms).items():
        ref = _ref_of(fa)
        famrec = {"reference": ref["arm"] if ref else None, "arms": {}}
        ref_w = _window(ref, "uniform") if ref else None
        for a in fa:
            L = int(a["layers"])
            arec = {"path": a["path"], "hot_frac": a.get("hot_frac"), "n_hot": a.get("n_hot"),
                    "reference": bool(a.get("reference")), "windows": {}, "cold_test": {}}
            for w in a.get("windows", []):
                wrec = {"cold_fraction": w.get("routing", {}).get("cold_fraction"),
                        "cold_rows_per_token": w.get("routing", {}).get("cold_rows_per_token"),
                        "modes": {}}
                for mname, m in w.get("modes", {}).items():
                    if "refused" in m or "error" in m:
                        wrec["modes"][mname] = {"refused": m.get("refused") or m.get("error")}
                        continue
                    row = {"per_token": per_token(m)}
                    rm = _mode(ref_w, mname)
                    if rm is not None and not a.get("reference"):
                        row["vs_reference"] = attribute(m, rm, L, a.get("residency_phase"))
                    wrec["modes"][mname] = row
                pred = w.get("predicted_transfer_us_per_token")
                em = _mode(w, "eager")
                if pred is not None and em is not None:
                    wrec["transfer"] = transfer_score(per_token(em)["transfer_us"], pred)
                wrec["timing"] = w.get("timing", {})
                wrec["sync_debug"] = w.get("sync_debug", {})
                arec["windows"][w["name"]] = wrec
            for mname in ("eager", "graph"):
                for q in ("launches", "syncs", "kernels", "submissions"):
                    pts = []
                    for w in a.get("windows", []):
                        m = _mode(w, mname)
                        cf = w.get("routing", {}).get("cold_fraction")
                        if m is not None and cf is not None:
                            pts.append((cf, per_token(m)[q]))
                    if pts:
                        arec["cold_test"].setdefault(mname, {})[q] = moves_with_cold(pts)
            famrec["arms"][a["arm"]] = arec
        out["families"][fam] = famrec
    out["box"] = {k: box.get(k) for k in ("gpu", "host", "probes", "costs")}
    return out


# --------------------------------------------------------------------------- gates (P0)
INSTRUMENT_SYNCS_PER_WINDOW = 2      # the census's closing synchronize + the profiler's own stop


def nesting_gate(mode: dict) -> dict:
    """Region-scoped API counts must equal the window's global counts for every class but sync, and the only
    syncs outside the token regions are the instrument's own (at most two cudaDeviceSynchronize). A gap means
    CUDA API events failed to nest under the regions, so the region counts would be under-counts."""
    reg, glob = mode.get("api", {}), mode.get("api_global")
    if glob is None:
        return {"ok": False, "why": "no api_global in the record"}
    gaps = {}
    for name in set(reg) | set(glob):
        d = glob.get(name, 0) - reg.get(name, 0)
        if d:
            gaps[name] = d
    bad = {k: v for k, v in gaps.items()
           if not (k == "cudaDeviceSynchronize" and 0 < v <= INSTRUMENT_SYNCS_PER_WINDOW)}
    return {"ok": not bad, "outside_regions": gaps, "unexplained": bad}


# Device rows an engine runs ONLY when not capturing, by design. The MXFP4 decode engine guards against a padded
# routing index with ``bool((want >= self.E).any())`` in ``_forward_decode`` and skips it under capture: one compare,
# one bool reduction, one D2H copy per layer (seen by name on the A2000 rehearsal).
EAGER_ONLY_PER_LAYER = {"mxfp4-pinned": 3, "mxfp4-nvme": 3}


def graph_parity_nodes(eager: dict, nodes: dict, layers: int = 0, eager_only_per_layer: int = 0) -> dict:
    """G4 on the graph's OWN node list (exact): the captured token's work nodes (kernels + copies + sets) equal the
    eager token's host submissions, less the engine's registered eager-only guard. The input-staging copies run
    OUTSIDE the graph (before each replay), so they are not nodes and are not added."""
    pe = per_token(eager)
    he = pe["launches"] + pe["async_copies"] + pe["blocking_copies"] + pe["memsets"]
    work = nodes.get("work")
    guard = eager_only_per_layer * layers
    other = {k: v for k, v in nodes.get("by_type", {}).items() if k not in WORK_NODE_TYPES}
    # two independent readers of the same graph (the CUDA runtime's node list, and the printed DOT) must agree
    dot = nodes.get("dot")
    agree = dot is None or dot.get("by_type") == nodes.get("by_type")
    return {"ok": work is not None and abs(work - (he - guard)) < 1e-9 and not other and agree,
            "eager_submissions": he, "graph_work_nodes": work, "eager_only_guard": guard, "non_work_nodes": other,
            "runtime_and_dot_agree": agree}


def graph_parity(eager: dict, graph: dict, layers: int = 0, eager_only_per_layer: int = 0,
                 staging_copies: int = 2) -> dict:
    """Capture changes what the HOST submits, never what the device runs: graph device rows per token must be the
    eager HOST submissions (exact, one device row each -- G6), less the engine's registered eager-only guard, plus
    the two input-staging copies the replay adds. Any other difference is work the graph lost or gained. The eager
    side is read from the host because the eager device view can lose a record (G6)."""
    pe = per_token(eager)
    he = pe["launches"] + pe["async_copies"] + pe["blocking_copies"] + pe["memsets"]
    kg = per_token(graph)["kernels"]
    guard = eager_only_per_layer * layers
    return {"ok": abs((kg - staging_copies) - (he - guard)) < 1e-9, "eager_submissions": he, "graph": kg,
            "eager_only_guard": guard}


# The profiler can LOSE a device record while the host record is intact. The A2000 rehearsals lost them only on the
# paths that synchronize, eager (MXFP4 NVMe one per window; hybrid up to 16 of 3,072 = 0.52 % in round 5), and in
# captured windows the input-staging copy (1 of ~1,290 rows). The host counts were exact every time, so NO count
# is graded on the device view: eager counts come from host API calls, captured counts from the graph's own node
# list (``graph_nodes``). The device view is used for device TIME only (P3, P7), and a lost record biases a time
# by at most its own share, so the gate bounds the loss where that bias stays small against P7's band.
MAX_DROPPED_RECORD_RATE = 0.02


def record_completeness(mode: dict) -> dict:
    """Eager only: every host submission (kernel launch, async or blocking copy, memset) yields exactly ONE device
    row. A shortfall is lost profiler records: recorded, and failing only above ``MAX_DROPPED_RECORD_RATE`` of the
    submissions. MORE device rows than submissions always fails -- that is mis-scoped regions, not loss."""
    n = int(mode["tokens"])
    host = sum(c for name, c in mode.get("api", {}).items()
               if api_class(name) in ("launch", "memcpy", "blocking", "memset"))
    dev = sum(r["count"] for r in mode.get("kernels", {}).values())
    missing = host - dev
    ceiling = max(1, int(MAX_DROPPED_RECORD_RATE * host))
    return {"ok": 0 <= missing <= ceiling, "host_submissions": host, "device_rows": dev,
            "dropped_records": missing, "loss_rate": (missing / host) if host else 0.0, "ceiling": ceiling,
            "tokens": n}


def gates(arms: list, budgets: dict | None = None) -> dict:
    """P0: every instrument gate, over every window of every arm. Nothing is read unless all hold."""
    out = {"nesting": [], "simulator": [], "graph_check": [], "graph_parity": [], "budget_coverage": [],
           "record_completeness": []}
    for a in arms:
        for w in a.get("windows", []):
            tag = f"{a['arm']}/{w['name']}"
            e = _mode(w, "eager")
            if e is not None:
                g = nesting_gate(e)
                out["nesting"].append((tag, g["ok"], g["unexplained"]))
                rc = record_completeness(e)
                out["record_completeness"].append((tag, rc["ok"], rc))
            if a["path"] == "pipelined":
                tc = w.get("traffic_check")
                out["simulator"].append((tag, bool(tc and tc.get("match")), tc))
            gm = _mode(w, "graph")
            if gm is not None:
                gc = w.get("graph_check") or {}
                ok = gc.get("b_rel") is not None and gc["b_rel"] <= GRAPH_B_REL_MAX
                out["graph_check"].append((tag, ok, gc))
                if e is not None:
                    guard = EAGER_ONLY_PER_LAYER.get(a["path"], 0)
                    if w.get("graph_nodes"):
                        gp = graph_parity_nodes(e, w["graph_nodes"], int(a["layers"]), guard)
                    else:                       # receipts from before the node count (rehearsal rounds 1-5)
                        gp = graph_parity(e, gm, int(a["layers"]), guard)
                    out["graph_parity"].append((tag, gp["ok"], gp))
            elif "graph" in w.get("modes", {}) and "error" in w["modes"]["graph"] and a.get("capturable", True):
                # an engine that CLAIMS capture and failed to capture is an instrument failure, not a refusal
                out["graph_check"].append((tag, False, w["modes"]["graph"]))
    for tag, cov in (budgets or {}).items():
        out["budget_coverage"].append((tag, cov is not None and cov >= 0.90, cov))
    ok = all(v[1] for rows in out.values() for v in rows)
    return {"ok": ok, **out}


# --------------------------------------------------------------------------- the registered predictions
# P66-PREREG.md "Predictions". Per-layer counts are code-structural (the engines' Python enqueue sequence),
# set from the A2000 rehearsal's reading of the code and registered for the 5090 before it runs.
REG = {
    "pipe_fetch_per_layer": {True: {"launch": 8, "memcpy": 2, "sync": 0}, False: {"launch": 5, "memcpy": 2, "sync": 0}},
    "pipe_delta_per_layer": {True: {"launches": 10, "async_copies": 2, "syncs": 0},
                             False: {"launches": 7, "async_copies": 2, "syncs": 0}},
    "mnvme_syncs_per_layer": 4, "mnvme_fetch_syncs_per_layer": 3,
    "mnvme_delta_per_layer": {"launches": -1, "async_copies": 3, "syncs": 3},
    "mpin_delta_per_layer": {"launches": 0, "async_copies": 0, "syncs": 0},
    "hyb_syncs_per_layer": {"ctrl0": 4, "mixed": 10, "cold_only": 8},
    "transfer_hold": (0.80, 1.25), "transfer_refute": 1.50,
    "tax_graph_ms": (0.50, 1.50),        # the ADDED kernels' device time, all hot, captured, Qwen3 on the 5090
    "rfc_share": (0.10, 0.25),
}
GRAPH_B_REL_MAX = 1.5e-2             # tests/test_pipelined_graphs.py's replay tolerance
RFC_COLD = 22 / 128                  # vLLM #57794's 88 of 512 offloaded, as a fraction of Qwen3's 128


def _eq(x, y):
    return x is not None and abs(x - y) < 1e-9


def _arms_of(arms, path, family=None):
    return [a for a in arms if a["path"] == path and (family is None or a["family"] == family)]


def _per_layer_delta(arm, ref, mode="eager", window="uniform"):
    m, r = _mode(_window(arm, window), mode), _mode(_window(ref, "uniform"), mode)
    if m is None or r is None:
        return None
    return attribute(m, r, int(arm["layers"]), arm.get("residency_phase"))


def registered_box(box: dict) -> bool:
    """The time bands (P3, P7) are registered for the RTX 5090 only; counts are structural and graded anywhere."""
    return "5090" in str((box.get("host") or {}).get("gpu", ""))


def verdicts(box: dict, arms: list, budgets: dict | None = None, level_m: dict | None = None) -> dict:
    v = {"P0": gates(arms, budgets)}
    on_box = registered_box(box)
    fams = _arms_by_family(arms)

    def fixed_within(arm, mode):
        """Every per-token count identical across the arm's windows (spread exactly 0).

        Eager is graded on the HOST counts, which are exact; the device view can lose a record (G6 bounds that
        and records it). Captured windows have only the device view (the host submits one graph launch), so
        they are graded on device rows, which G4 already ties to the eager submissions."""
        res = {}
        if mode == "graph" and any(w.get("graph_nodes") for w in arm.get("windows", [])):
            # the captured graph's own node counts: exact, one capture = one token
            for q in ("work", "kernel", "memcpy", "memset"):
                pts = [(w.get("routing", {}).get("cold_fraction"),
                        (w["graph_nodes"].get(q) if q == "work" else w["graph_nodes"]["by_type"].get(q.upper(), 0)))
                       for w in arm.get("windows", []) if w.get("graph_nodes") and _mode(w, "graph") is not None]
                res[f"graph_nodes.{q}"] = moves_with_cold(pts)
            return res
        for q in (("launches", "async_copies", "syncs", "memsets") if mode == "eager" else ("kernels",)):
            pts = [(w.get("routing", {}).get("cold_fraction"), per_token(m)[q])
                   for w in arm.get("windows", []) if (m := _mode(w, mode)) is not None]
            res[q] = moves_with_cold(pts)
        return res

    # ---- P1: pipelined is fixed across windows (eager and graph), and across n_hot > 0 arms
    p1 = {"arms": {}, "across_hot": {}}
    ok1 = True
    for fam, fa in fams.items():
        pipes = _arms_of(fa, "pipelined")
        for a in pipes:
            for mode in ("eager", "graph"):
                r = fixed_within(a, mode)
                n = sum(1 for w in a.get("windows", []) if _mode(w, mode) is not None)
                good = all(x["fixed"] in (True, None) for x in r.values())
                p1["arms"][f"{a['arm']}/{mode}"] = {"windows": n, "fixed": good,
                                                    "spreads": {k: x["spread"] for k, x in r.items()}}
                ok1 &= good
        hot = [a for a in pipes if a.get("n_hot")]
        vals = {a["arm"]: per_token(_mode(_window(a, "uniform"), "eager"))["launches"] for a in hot
                if _mode(_window(a, "uniform"), "eager") is not None}
        same = len(set(round(x, 6) for x in vals.values())) <= 1
        p1["across_hot"][fam] = {"launches": vals, "same": same}
        ok1 &= same
    v["P1"] = {"status": "HOLDS" if ok1 and p1["arms"] else ("UNREAD" if not p1["arms"] else "REFUTED"), **p1}

    # ---- P2: the pipelined count against the all-resident step, per layer, and the fetch region
    p2, ok2 = {}, True
    for fam, fa in fams.items():
        ref = next((a for a in fa if a.get("reference") and a["path"] == "hybrid"), None)
        for a in _arms_of(fa, "pipelined"):
            at = _per_layer_delta(a, ref) if ref else None
            if at is None:
                continue
            key = bool(a.get("n_hot"))
            want_d, want_f = REG["pipe_delta_per_layer"][key], REG["pipe_fetch_per_layer"][key]
            got_d = {k: at["delta_per_layer"][k] for k in want_d}
            got_f = {k: at["fetch_per_layer"].get(k) for k in want_f}
            good = all(_eq(got_d[k], want_d[k]) for k in want_d) and all(_eq(got_f[k], want_f[k]) for k in want_f)
            p2[a["arm"]] = {"delta_per_layer": got_d, "want": want_d, "fetch_per_layer": got_f,
                            "want_fetch": want_f, "holds": good,
                            "delta_per_token": {k: at["delta_per_token"][k] for k in want_d}}
            ok2 &= good
    v["P2"] = {"status": "HOLDS" if p2 and ok2 else ("UNREAD" if not p2 else "REFUTED"), "arms": p2}

    # ---- P3: the fixed residency tax (all hot, zero cold traffic) in captured device time
    p3 = {}
    for fam, fa in fams.items():
        ref = next((a for a in fa if a.get("reference") and a["path"] == "hybrid"), None)
        allhot = next((a for a in _arms_of(fa, "pipelined") if a.get("hot_frac") == 1.0), None)
        if ref is None or allhot is None:
            continue
        rows = {}
        for mode in ("graph", "eager"):
            at = _per_layer_delta(allhot, ref, mode)
            if at is not None:
                sp = at["device_split"]
                rows[mode] = {"device_ms_per_token": at["delta_per_token"]["device_us"] / 1e3,
                              "added_kernels_ms_per_token": sp["count_changed_us"] / 1e3,
                              "shared_kernels_ms_per_token": sp["count_equal_us"] / 1e3,
                              "shared_by_kernel_us": dict(list(sp["count_equal_by_kernel"].items())[:6]),
                              "submissions_per_token": at["delta_per_token"]["submissions"]}
        tg, te = (_window(allhot, "uniform") or {}).get("timing", {}), (_window(ref, "uniform") or {}).get("timing", {})
        for mode in ("eager", "graph"):
            if mode in tg and mode in te:
                rows.setdefault(mode, {})["wall_ms_per_token"] = tg[mode]["wall_ms_per_token"] - te[mode]["wall_ms_per_token"]
        p3[fam] = rows
    lo, hi = REG["tax_graph_ms"]
    g = [r["graph"]["added_kernels_ms_per_token"] for r in p3.values() if "graph" in r]
    v["P3"] = {"status": ("UNREAD" if not g else "RECORDED (not the registered box)" if not on_box
                          else "HOLDS" if all(lo <= x <= hi for x in g) else "REFUTED"),
               "band_ms": (lo, hi), "families": p3,
               "note": "a band on the 5090 reading only; any other box is recorded, not graded"}

    # ---- P4 / P5: the MXFP4 engines against their all-resident sibling
    p4, p5, ok4, ok5 = {}, {}, True, True
    for fam, fa in fams.items():
        mref = next((a for a in fa if a.get("reference") and a["path"] == "mxfp4-pinned"), None)
        for a in fa:
            if mref is None or a is mref or a["path"] not in ("mxfp4-nvme", "mxfp4-pinned"):
                continue
            fx = fixed_within(a, "eager")
            fixed = all(x["fixed"] in (True, None) for x in fx.values())
            at = _per_layer_delta(a, mref)
            if at is None:
                continue
            L = int(a["layers"])
            if a["path"] == "mxfp4-nvme":
                want = REG["mnvme_delta_per_layer"]
                got = {k: at["delta_per_layer"][k] for k in want}
                syn = at["arm"]["syncs"] / L
                fs = at["fetch_per_layer"].get("sync")
                good = (fixed and all(_eq(got[k], want[k]) for k in want)
                        and _eq(syn, REG["mnvme_syncs_per_layer"]) and _eq(fs, REG["mnvme_fetch_syncs_per_layer"]))
                p4[a["arm"]] = {"fixed": fixed, "delta_per_layer": got, "want": want, "syncs_per_layer": syn,
                                "fetch_syncs_per_layer": fs, "holds": good}
                ok4 &= good
            else:
                want = REG["mpin_delta_per_layer"]
                got = {k: at["delta_per_layer"][k] for k in want}
                good = fixed and all(_eq(got[k], want[k]) for k in want)
                p5[a["arm"]] = {"fixed": fixed, "delta_per_layer": got, "want": want, "holds": good}
                ok5 &= good
    v["P4"] = {"status": "HOLDS" if p4 and ok4 else ("UNREAD" if not p4 else "REFUTED"), "arms": p4}
    v["P5"] = {"status": "HOLDS" if p5 and ok5 else ("UNREAD" if not p5 else "REFUTED"), "arms": p5}

    # ---- P6: the hybrid tier's v0 dispatch MOVES with cold fraction, by layer composition
    p6, ok6 = {}, True
    for fam, fa in fams.items():
        for a in _arms_of(fa, "hybrid"):
            if a.get("reference"):
                continue
            L, k = int(a["layers"]), int(a["k"])
            ctrl = {w["name"]: per_token(m) for w in a.get("windows", []) if w["name"].startswith("ctrl")
                    and (m := _mode(w, "eager")) is not None}
            if not ctrl:
                continue
            spread = max(c["launches"] for c in ctrl.values()) - min(c["launches"] for c in ctrl.values())
            want = REG["hyb_syncs_per_layer"]
            per = {}
            for nm, c in ctrl.items():
                cl = int(nm[4:])
                comp = "ctrl0" if cl == 0 else ("cold_only" if cl == k else "mixed")
                per[nm] = {"composition": comp, "syncs_per_layer": c["syncs"] / L,
                           "launches_per_layer": c["launches"] / L, "want_syncs": want[comp]}
            good = spread > 0 and all(_eq(x["syncs_per_layer"], x["want_syncs"]) for x in per.values())
            p6[a["arm"]] = {"launch_spread_per_token": spread, "windows": per, "holds": good}
            ok6 &= good
    v["P6"] = {"status": "HOLDS" if p6 and ok6 else ("UNREAD" if not p6 else "REFUTED"), "arms": p6}

    # ---- P7: cold_deadline's transfer prediction against the measured link-crossing device time
    lo, hi = REG["transfer_hold"]
    p7, rat = {}, []
    for a in arms:
        if a["path"] not in ("pipelined", "hybrid", "mxfp4-nvme", "mxfp4-pinned"):
            continue
        for w in a.get("windows", []):
            e, pred = _mode(w, "eager"), w.get("predicted_transfer_us_per_token")
            if e is None or not pred or pred <= 0:
                continue
            meas = per_token(e)["transfer_us"]
            hot_us = 0.0
            if a["path"].startswith("mxfp4"):
                # the MXFP4 engines ALSO re-copy hot rows device-to-device into their slots; that is outside
                # the model by construction (gpu_us charges only cold bytes), so it is taken off the measure
                # using the all-hot sibling's own gather time per hot row
                hot_us = w.get("routing", {}).get("counters_hot_d2d_rows_per_token", 0.0) * \
                    _hot_row_us(arms, a["family"])
            r = (meas - hot_us) / pred
            p7[f"{a['arm']}/{w['name']}"] = {"path": a["path"], "measured_us": meas, "hot_d2d_us": hot_us,
                                            "predicted_us": pred, "ratio": r}
            if a["path"] == "pipelined" and w["name"] == "uniform":
                rat.append(r)
    st = "UNREAD" if not rat else "RECORDED (not the registered box)" if not on_box else (
        "HOLDS" if all(lo <= x <= hi for x in rat)
        else "REFUTED" if any(x > REG["transfer_refute"] or x < lo for x in rat) else "BETWEEN BANDS")
    v["P7"] = {"status": st, "band": (lo, hi), "refute_above": REG["transfer_refute"], "windows": p7,
               "graded": "pipelined, uniform windows"}

    # ---- P8 (informational): the RFC question -- how much of residency's cost at the RFC's cold fraction is
    # the fixed tax a two-point comparison cannot separate from per-cold-expert cost
    p8 = {}
    for fam, fa in fams.items():
        ref = next((a for a in fa if a.get("reference") and a["path"] == "hybrid"), None)
        pipes = {a["hot_frac"]: a for a in _arms_of(fa, "pipelined")}
        if ref is None or 1.0 not in pipes:
            continue
        for hf, a in sorted(pipes.items()):
            if hf == 1.0:
                continue
            for mode in ("graph", "eager"):
                ta = (_window(a, "uniform") or {}).get("timing", {}).get(mode)
                t1 = (_window(pipes[1.0], "uniform") or {}).get("timing", {}).get(mode)
                tr = (_window(ref, "uniform") or {}).get("timing", {}).get(mode)
                if not (ta and t1 and tr):
                    continue
                total = ta["wall_ms_per_token"] - tr["wall_ms_per_token"]
                tax = t1["wall_ms_per_token"] - tr["wall_ms_per_token"]
                p8[f"{fam}/{a['arm']}/{mode}"] = {"cold_fraction": 1 - hf, "residency_ms": total,
                                                  "fixed_tax_ms": tax,
                                                  "tax_share": (tax / total) if total > 0 else None}
    v["P8"] = {"status": "INFORMATIONAL", "rows": p8, "bands": REG["rfc_share"]}
    v["P9"] = {"status": "INFORMATIONAL", "cross_level": cross_level(level_m, arms), "level_m": level_m or {}}
    v["decision"] = decide(v)
    return v


def _hot_row_us(arms, family) -> float:
    """Device us per hot row the MXFP4 engine re-copies D2D, from the family's all-hot sibling (zero cold)."""
    ref = next((a for a in arms if a["family"] == family and a.get("reference") and a["path"] == "mxfp4-pinned"), None)
    w = _window(ref, "uniform") if ref else None
    e = _mode(w, "eager")
    if e is None:
        return 0.0
    rows = w.get("routing", {}).get("counters_hot_d2d_rows_per_token") or 0.0
    return per_token(e)["transfer_us"] / rows if rows else 0.0


def decide(v: dict) -> str:
    """P66-PREREG.md "Decision rule", literally."""
    if not v["P0"]["ok"]:
        return "¬P0: an instrument gate failed -- nothing is read; the failing gate is filed and the run repeated once"
    out = []
    if v["P1"]["status"] == "HOLDS" and v["P2"]["status"] == "HOLDS":
        out.append("P1 ∧ P2: the pipelined engine's residency cost is a FIXED per-layer count (as registered), "
                   "independent of cold fraction -> register the measured claim; the cost model gets a per-step "
                   "fixed term (P3's graph device time, and submissions x the host's unit cost when eager), "
                   "never a per-cold-expert launch term (grouped-nf4-gemm follow-up on kernel/cold_deadline.py)")
    elif v["P1"]["status"] == "REFUTED":
        out.append("¬P1: the pipelined enqueue sequence moved with routing -- a defect against its design law "
                   "(pipelined.py:38-43); filed with the moving kernel names before anything is claimed")
    elif v["P2"]["status"] == "REFUTED":
        out.append("P1 ∧ ¬P2: fixed but not the registered count -- the measured per-layer count is registered "
                   "instead, with the difference attributed by name")
    if v["P4"]["status"] == "HOLDS":
        out.append("P4: the MXFP4 NVMe engine's +3 syncs/layer are structural; a pinned-staging follow-up (two of "
                   "the three are pageable H2D copies) is licensed only if their host cost >= 10 % of that token")
    if v["P6"]["status"] == "HOLDS":
        out.append("P6: the hybrid tier's cost is layer-COMPOSITION-dependent (hot-only / mixed / cold-only); "
                   "cold_dest='deadline' prices only bytes, so its GPU side omits the mixed-layer dispatch term")
    t = v["P7"]["status"]
    if t == "HOLDS":
        out.append("P7: bytes/link + bytes/VRAM predicts the pipelined gather within band -> the variable part of "
                   "residency is the transfer the model already has; any RFC-style gap in this engine is the "
                   "fixed tax (P3/P8) or host launch cost, not transfer")
    elif t in ("REFUTED", "BETWEEN BANDS"):
        out.append(f"P7 {t}: the transfer itself departs from bytes/link -- the model needs a measured "
                   "efficiency factor (UVA-read efficiency, #105 candidate 4) before any RFC comparison")
    # P8, the RFC reading: only at the RFC-matched cold fraction, captured (the shipped serving shape)
    lo, hi = REG["rfc_share"]
    rfc = [r for key, r in v["P8"]["rows"].items() if key.endswith("/graph")
           and r.get("tax_share") is not None and abs(r["cold_fraction"] - RFC_COLD) < 1e-6]
    if rfc:
        sh = rfc[0]["tax_share"]
        if sh >= hi:
            out.append(f"P8: at the RFC's 17.2 % cold, {sh:.0%} of residency's cost is the fixed tax -- a two-point "
                       "comparison (all-resident vs offloaded) would book it as cold-expert cost; consistent with "
                       "the RFC's launch suspect acting as a FIXED term")
        elif sh <= lo:
            out.append(f"P8: at the RFC's 17.2 % cold the fixed tax is {sh:.0%} of residency's cost -- in this "
                       "engine launches cannot explain an RFC-size (2-4x) gap; the gap would be link-side")
        else:
            out.append(f"P8: at the RFC's 17.2 % cold the fixed tax is {sh:.0%} of residency's cost -- partial; "
                       "stated as measured, no attribution claimed for the RFC")
    return " | ".join(out) if out else "no registered branch fired"


# --------------------------------------------------------------------------- Level M (context)
LEVEL_M_APIS = ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaStreamSynchronize")


def level_m(root: Path, step_budget_path: Path | None) -> dict:
    """step_decomp's own outputs per arm: device kernel calls per step (its kernels.txt through step_budget, the
    SV2 route) and host API counts per step (its sync-attr JSON over the same active window).

    The kernels.txt is ROW-LIMITED (step_decomp prints 80 rows): its time coverage reaches ~99.9 % while its
    CALL counts miss the smallest rows, which is exactly where residency's added kernels live (the A2000
    rehearsal: a Delta of 148.7 calls/step against the true 192). So counts are compared through the sync-attr
    API counts, which are complete; the table is kept for device time."""
    sb = None
    if step_budget_path is not None and Path(step_budget_path).is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_budget", step_budget_path)
        sb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sb)
    out = {}
    for kt in sorted(Path(root).glob("*_kernels.txt")):
        arm = kt.name[:-len("_kernels.txt")]
        rec = {}
        if sb is not None:
            try:
                steps, rows, tot = sb.parse(str(kt))
                b = sb.budget(steps, rows, tot)
                rec.update({"steps": steps, "coverage": b["coverage"],
                            "kernel_calls_per_step": sum(r["calls_per_step"] for r in rows),
                            "device_ms_per_step": b["total_device_us_per_step"] / 1e3})
            except SystemExit as e:
                rec["budget_refused"] = str(e)
        rj = kt.with_name(arm + ".json")
        if rj.is_file():
            try:
                rec["manifest_counts"] = json.loads(rj.read_text()).get("manifest_counts")
            except (ValueError, OSError):
                pass
        sj = kt.with_name(arm + "_sync.json")
        if sj.is_file():
            d = json.loads(sj.read_text())
            n = max(1, int(d.get("active_steps") or 0))
            oc = d.get("op_counts", {})
            rec["active_steps"] = d.get("active_steps")
            rec["per_step"] = {k: oc.get(k, 0) / n for k in ("cudaLaunchKernel", "cudaMemcpyAsync",
                                                              "cudaStreamSynchronize", "cudaDeviceSynchronize",
                                                              "aten::item", "aten::nonzero")}
        out[arm] = rec
    ref = out.get("M_ref")
    if ref:
        for arm, rec in out.items():
            if arm == "M_ref":
                continue
            if "kernel_calls_per_step" in rec and "kernel_calls_per_step" in ref:
                rec["delta_kernel_calls_per_step_rowlimited"] = rec["kernel_calls_per_step"] - ref["kernel_calls_per_step"]
            if "per_step" in rec and "per_step" in ref:
                rec["delta_api_per_step"] = {k: rec["per_step"][k] - ref["per_step"][k] for k in LEVEL_M_APIS}
    return out


def cross_level(lm: dict, arms: list) -> dict:
    """P9: the served step's per-step API deltas against the MoE token's per-token deltas, by name, for the
    pipelined arms both levels ran (``M_pipe-<h>`` against Level L's ``pipe-<h>``, uniform, eager). Residency
    touches only the MoE layers, so the two must agree exactly if Level L models the step's residency cost."""
    out = {}
    for arm, rec in (lm or {}).items():
        if not arm.startswith("M_pipe-") or "delta_api_per_step" not in rec:
            continue
        la = next((a for a in arms if a["arm"] == arm[2:] and a["path"] == "pipelined"), None)
        ref = next((a for a in arms if a.get("reference") and a["path"] == "hybrid"), None)
        if la is None or ref is None:
            continue
        at = _per_layer_delta(la, ref)
        if at is None:
            continue
        L = {k: at["by_api"].get(k, 0.0) for k in LEVEL_M_APIS}
        M = rec["delta_api_per_step"]
        out[arm] = {"level_m_per_step": M, "level_l_per_token": L,
                    "agree": all(abs(M[k] - L[k]) < 1e-6 for k in LEVEL_M_APIS)}
    return out


# --------------------------------------------------------------------------- the generated read
def render_md(rep: dict, v: dict) -> str:
    L = []
    box = rep.get("box") or {}
    host = box.get("host") or {}
    L.append("## Generated read (p66_reduce.py, unedited)\n")
    L.append(f"Box: {host.get('gpu')} ({host.get('sm_count')} SMs) on {host.get('cpu_model')}; "
             f"torch {host.get('torch')}, triton {host.get('triton')}. Host unit costs: "
             f"{json.dumps(box.get('probes'), default=str)}. Transfer constants: {json.dumps(box.get('costs'))}.\n")
    for fam, fr in rep["families"].items():
        L.append(f"### {fam} (reference: `{fr['reference']}`)\n")
        L.append("| arm | window | cold frac | mode | launches/tok | copies/tok | syncs/tok | kernels/tok | "
                 "device ms/tok | Δlaunch vs ref | Δsync vs ref | wall ms/tok |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm, ar in fr["arms"].items():
            for wn, wr in ar["windows"].items():
                for mn, mr in wr["modes"].items():
                    cf = wr.get("cold_fraction")
                    if "refused" in mr:
                        L.append(f"| {arm} | {wn} | {cf:.3f} | {mn} | refused: {str(mr['refused'])[:60]} "
                                 "| | | | | | | |")
                        continue
                    p = mr["per_token"]
                    d = mr.get("vs_reference", {}).get("delta_per_token", {})
                    t = (wr.get("timing") or {}).get(mn, {})
                    L.append(f"| {arm} | {wn} | {cf:.3f} | {mn} | {p['launches']:.1f} | {p['async_copies']:.1f} | "
                             f"{p['syncs']:.1f} | {p['kernels']:.1f} | {p['device_us'] / 1e3:.3f} | "
                             f"{d.get('launches', 0):+.1f} | {d.get('syncs', 0):+.1f} | "
                             f"{t.get('wall_ms_per_token', float('nan')):.3f} |")
        L.append("")
    L.append("### Verdicts (pre-registered)\n")
    g = v["P0"]
    drops = sum(x[2]["dropped_records"] for x in g["record_completeness"] if x[2]["dropped_records"] > 0)
    L.append(f"- **P0 (gates): {'ALL HOLD' if g['ok'] else 'FAILED'}** -- nesting "
             f"{sum(x[1] for x in g['nesting'])}/{len(g['nesting'])}, simulator {sum(x[1] for x in g['simulator'])}/"
             f"{len(g['simulator'])}, graph check {sum(x[1] for x in g['graph_check'])}/{len(g['graph_check'])}, "
             f"graph parity {sum(x[1] for x in g['graph_parity'])}/{len(g['graph_parity'])}, budget coverage "
             f"{sum(x[1] for x in g['budget_coverage'])}/{len(g['budget_coverage'])}, record completeness "
             f"{sum(x[1] for x in g['record_completeness'])}/{len(g['record_completeness'])} "
             f"({drops} dropped device record(s) in total)")
    for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        L.append(f"- **{k}: {v[k]['status']}** -- `{json.dumps({kk: vv for kk, vv in v[k].items() if kk != 'status'}, default=str)[:600]}`")
    for k in ("P8", "P9"):
        L.append(f"- **{k} (informational)** -- `{json.dumps(v[k], default=str)[:900]}`")
    L.append(f"\n**Decision rule:** {v['decision']}\n")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="directory holding box.json and rows_*.json")
    ap.add_argument("--json", default=None, help="write the full read here")
    ap.add_argument("--md", default=None, help="write the generated markdown read here")
    ap.add_argument("--level-m", default=None, help="step_decomp outputs (Level M context)")
    ap.add_argument("--step-budget", default=None, help="bench/hybrid-g9/f1/step_budget.py (the budget parser)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    box, arms = load_rows(root)
    rep = census(box, arms)
    budgets = {}
    if a.step_budget:
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_budget", a.step_budget)
        sb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sb)
        for t in sorted(root.glob("table_*_eager.txt")):
            try:
                budgets[t.name] = sb.budget(*sb.parse(str(t)))["coverage"]
            except SystemExit:
                budgets[t.name] = None
    lm = level_m(Path(a.level_m), Path(a.step_budget) if a.step_budget else None) if a.level_m else None
    v = verdicts(box, arms, budgets, lm)
    rep["verdicts"] = v
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    md = render_md(rep, v)
    if a.md:
        Path(a.md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
