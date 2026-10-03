#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_census.py -- lane SC1b (#846): one engine's steady decode step, read from two Nsight Systems sqlite exports, and
the decomposition of a gap between two engines (bench/sc1b/SC1b-PREREG.md).

Per step (medians with interquartile ranges over the kept steps):
  graph mode (`--cuda-graph-trace=graph`): P period (delimiter to delimiter), S graph span, D_out every non-graph device
      interval in the period (any stream, as a union with S, minus S), idle_out = P - |S u non-graph| (device idle outside
      graphs: host/CPU work, sync and launch latency outside the graph).
  node mode (`--cuda-graph-trace=node`): U_in the union of one replay's in-graph kernels, the per-class kernel/copy time
      (bench/sc1b/kernel_classes.json; classes by name, grid, in/out graph, MoE segment by node order, inherit-next/prev),
      kernels per step.
  mixed: I_in = S - U_in (in-graph idle: launch/dependency gaps); O = sum(class) + I_in + idle_out - P (overlap, plus
      median non-additivity and the out-of-graph node/graph-mode mismatch), so P = sum(class) - O + I_in + idle_out exactly.

Step delimiters: `graph` = one replay of the steady (modal graphExecId) decode graph; `d2h` = llama.cpp's logits
device-to-host copy (bytes = n_out x vocab x 4), with exactly one steady replay per step. In both modes the first and
last step are trimmed; a step longer than 1.5x the median, or with more out-of-graph kernels than the modal count, or (d2h)
without exactly one replay, is dropped and counted; the central `keep` steps are read.

Gates (per arm): G-inflate |P/unprofiled - 1| <= 3 % (else PROFILER_INFLATED: idle_out not nameable); G-node median U_in <=
median S x 1.02 (else NODE_TRACE_INFLATED: classes and I_in unread); G-map residual <= 2 % of node-mode device time (else
CLASS_MAP_INCOMPLETE). A gap reads only when both arms pass G-node and G-map.

  sc1b_census.py arm --graph G.sqlite --node N.sqlite --engine E --batch B --classes kernel_classes.json
                     [--delim graph|d2h --d2h-bytes N] [--unprofiled-ms X] --out arm.json
  sc1b_census.py gap --e4b E4B.json --comp COMP.json [--sc1-ratio R] --out gap.json
  sc1b_census.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

CLASSES = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual")
KEEP = 62
LONG_PERIOD = 1.5
G_INFLATE = 0.05          # vs pass 1's MEDIAN-of-3 slope (round 2 M9: a single draw's min-of-3 moves > 3 %)
G_NODE = 1.02
G_MAP = 0.02
G_CLOCK = 0.03
MIN_STEPS = 56            # round 2 M8: fewer kept steps is VOID, never a 3-step median
MOE_LAYERS = 48           # Qwen3-30B-A3B: every steady replay must open and close 48 MoE segments (round 2 M6)
NAMED_SHARE = 0.5
DIAG_BAD = ("error", "dropped", "lost", "overflow", "buffer full", "failed", "insufficient")


# ------------------------------------------------------------------------------------------------------------- sqlite --
def _has(con, table):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _d2h_kinds(con):
    """The copyKind ids that mean device-to-host (ENUM_CUDA_MEMCPY_OPER when exported; CUPTI's DTOH = 2 otherwise)."""
    if _has(con, "ENUM_CUDA_MEMCPY_OPER"):
        cols = [r[1] for r in con.execute("PRAGMA table_info(ENUM_CUDA_MEMCPY_OPER)")]
        for label in ("label", "name"):
            if label in cols:
                ids = {i for i, v in con.execute(f"SELECT id, {label} FROM ENUM_CUDA_MEMCPY_OPER")
                       if "to-host" in str(v).lower().replace(" ", "-") or "DTOH" in str(v).upper()}
                if ids:
                    return ids
    return {2}


def load(path):
    """Every device interval and graph record in an export, as plain dicts (ns timestamps)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    s = dict(con.execute("SELECT id, value FROM StringIds")) if _has(con, "StringIds") else {}
    d2h = _d2h_kinds(con)
    kern, copies, launches, graphs = [], [], [], []
    if _has(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for st, en, stream, corr, node, short, dem, gx, gy, gz in con.execute(
                "SELECT start, end, streamId, correlationId, graphNodeId, shortName, demangledName, gridX, gridY, gridZ "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            kern.append({"start": st, "end": en, "stream": stream, "corr": corr, "node": node, "name": s.get(short, "?"),
                         "demangled": s.get(dem, "?"), "grid": (gx, gy, gz), "kind": "kernel"})
    if _has(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for st, en, stream, corr, node, nbytes, kind in con.execute(
                "SELECT start, end, streamId, correlationId, graphNodeId, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            copies.append({"start": st, "end": en, "stream": stream, "corr": corr, "node": node, "bytes": nbytes,
                           "d2h": kind in d2h, "name": "memcpy", "demangled": "memcpy", "grid": (0, 0, 0), "kind": "memcpy"})
    if _has(con, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for st, en, stream, corr, node, nbytes in con.execute(
                "SELECT start, end, streamId, correlationId, graphNodeId, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            copies.append({"start": st, "end": en, "stream": stream, "corr": corr, "node": node, "bytes": nbytes,
                           "d2h": False, "name": "memset", "demangled": "memset", "grid": (0, 0, 0), "kind": "memset"})
    if _has(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for st, en, corr, name in con.execute("SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            if s.get(name, "").startswith("cudaGraphLaunch"):
                launches.append({"start": st, "end": en, "corr": corr})
    diags = []
    if _has(con, "DIAGNOSTIC_EVENT"):
        diags = [{"severity": sev, "text": txt} for sev, txt in con.execute("SELECT severity, text FROM DIAGNOSTIC_EVENT")]
    if _has(con, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        for st, en, corr, gid, gexec in con.execute(
                "SELECT start, end, correlationId, graphId, graphExecId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
            graphs.append({"start": st, "end": en, "corr": corr, "graph": gid, "exec": gexec})
    con.close()
    for xs in (kern, copies, launches, graphs):
        xs.sort(key=lambda r: r["start"])
    return {"kernels": kern, "copies": copies, "launches": launches, "graphs": graphs, "diagnostics": diags}


def _nongraph(r):
    """Eager (out-of-graph) work: graphNodeId NULL -- or 0, should an export write 0 for "none" (round 2 M7)."""
    return r["node"] in (None, 0)


# ------------------------------------------------------------------------------------------------------------ helpers --
def union_ns(intervals, lo, hi):
    """Length of the union of [start, end) intervals clipped to [lo, hi)."""
    iv = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    tot, cur_a, cur_b = 0, None, None
    for a, b in iv:
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot


def _modal(values):
    c = Counter(values)
    return c.most_common(1)[0][0] if c else None


def _stat(xs):
    """Median and interquartile range, in ms (inputs in ns)."""
    xs = sorted(xs)
    if not xs:
        return {"median_ms": None, "iqr_ms": None}
    if len(xs) < 4:
        q1, q3 = xs[0], xs[-1]
    else:
        q = statistics.quantiles(xs, n=4, method="inclusive")
        q1, q3 = q[0], q[2]
    return {"median_ms": round(statistics.median(xs) / 1e6, 6), "iqr_ms": round((q3 - q1) / 1e6, 6)}


def _central(rows, keep):
    if len(rows) <= keep:
        return rows
    off = (len(rows) - keep) // 2
    return rows[off:off + keep]


def _filter(rows, keep, out_key, positions=None):
    """Select the registered positions (when the capture holds the whole run) or trim the capture's edges (when an engine
    hook bracketed it); drop long periods and steps with extra out-of-graph kernels; keep the central `keep`."""
    dropped = Counter()
    if positions:
        start, count = positions
        rows = rows[start:start + count]
    else:
        rows = rows[1:-1] if len(rows) > 2 else []
    if not rows:
        return [], dropped
    med = statistics.median(r["P"] for r in rows)
    modal_out = _modal(r[out_key] for r in rows)
    kept = []
    for r in rows:
        if r["P"] > LONG_PERIOD * med:
            dropped["period_over_1.5x_median"] += 1
        elif r[out_key] > modal_out:
            dropped["extra_out_of_graph_kernels"] += 1
        else:
            kept.append(r)
    return _central(kept, keep), dropped


# --------------------------------------------------------------------------------------------------------- the classes --
def load_classes(path, engine):
    with open(path) as f:
        cmap = json.load(f)
    spec = cmap.get(engine)
    if not spec:
        raise SystemExit(f"no class map for engine {engine!r} in {path}")
    for r in spec.get("rules", []):
        if r.get("class") not in CLASSES:
            raise SystemExit(f"{path}: {engine}: unknown class {r.get('class')!r}")
    return spec


def _hit(item, subs):
    subs = subs if isinstance(subs, list) else [subs]
    return any(x in item["name"] or x in item["demangled"] for x in subs)


def _where_ok(item, w, in_graph):
    gx, gy, gz = item["grid"]
    dims = {"gridX": gx, "gridY": gy, "gridZ": gz}
    for k, v in (w or {}).items():
        if k == "in_graph":
            if bool(v) != in_graph:
                return False
            continue
        dim, op = k.rsplit("_", 1)
        x = dims[dim]
        if (op == "gt" and not x > v) or (op == "le" and not x <= v) or (op == "eq" and not x == v):
            return False
    return True


def classify_seq(items, spec, in_graph, stats=None):
    """Classes for one ordered sequence (one replay's in-graph records in start-time order, or one step's out-of-graph
    items). `stats`, when given, receives the MoE segment opens / closes and whether they alternated."""
    base = []
    for it in items:
        if it["kind"] in ("memcpy", "memset"):
            base.append("memcpy")
            continue
        cls = None
        for r in spec.get("rules", []):
            if _hit(it, r["match"]) and _where_ok(it, r.get("where"), in_graph):
                cls = r["class"]
                break
        base.append(cls)
    seg = spec.get("segment") or {}
    inh_next, inh_prev = spec.get("inherit_next", []), spec.get("inherit_prev", [])
    pending = {}
    out = list(base)
    in_seg = seen_expert = False
    for i, it in enumerate(items):
        if it["kind"] in ("memcpy", "memset"):
            continue
        if inh_next and _hit(it, inh_next):
            pending[i] = "next"
            continue
        if inh_prev and _hit(it, inh_prev):
            pending[i] = "prev"
            continue
        if seg and _hit(it, seg["start"]) and not (in_seg and _hit(it, seg["end"])):
            if in_seg and not seen_expert:
                out[i] = base[i] or "moe_route"                        # a second routing anchor of the same layer
                continue
            if in_seg and stats is not None:
                stats["nested"] = stats.get("nested", 0) + 1          # a new layer's router before this layer's end anchor
            if stats is not None:
                stats["starts"] = stats.get("starts", 0) + 1
            in_seg, seen_expert = True, False
            out[i] = base[i] or "moe_route"
            continue
        if in_seg:
            if _hit(it, seg["end"]):
                out[i] = base[i] or "moe_route"
                in_seg = False
                if stats is not None:
                    stats["ends"] = stats.get("ends", 0) + 1
            elif _hit(it, spec.get("expert_names", [])):
                out[i] = "moe_expert"
                seen_expert = True
            else:
                out[i] = "moe_route"
    if in_seg and stats is not None:
        stats["open_at_end"] = 1
    mm = spec.get("matmul_names", [])
    for i, way in sorted(pending.items()):
        rng = range(i + 1, len(items)) if way == "next" else range(i - 1, -1, -1)
        out[i] = next((out[j] for j in rng if j not in pending and _hit(items[j], mm) and out[j]), None)
    return [c or "residual" for c in out]


# --------------------------------------------------------------------------------------------------------- graph mode --
def graph_mode(ex, delim="graph", d2h_bytes=None, keep=KEEP, positions=None, min_steps=MIN_STEPS):
    g = ex["graphs"]
    if not g:
        return {"status": "void", "why": "no CUPTI_ACTIVITY_KIND_GRAPH_TRACE rows (was the export taken with --cuda-graph-trace=graph?)"}
    steady_exec = _modal(r["exec"] for r in g)
    steady = [r for r in g if r["exec"] == steady_exec]
    nongraph = [r for r in ex["kernels"] + ex["copies"] if _nongraph(r)]
    if delim == "graph":
        bounds = [(a["start"], b["start"], [a]) for a, b in zip(steady, steady[1:])]
    elif delim == "d2h":
        ds = [c for c in ex["copies"] if c["d2h"] and _nongraph(c) and (d2h_bytes is None or c["bytes"] == d2h_bytes)]
        if len(ds) < 3:
            return {"status": "void", "why": f"{len(ds)} logits D2H delimiters (bytes {d2h_bytes})"}
        bounds = [(a["start"], b["start"], [r for r in steady if a["start"] <= r["start"] < b["start"]]) for a, b in zip(ds, ds[1:])]
    else:
        raise SystemExit(f"unknown delimiter {delim!r}")
    rows = []
    for lo, hi, reps in bounds:
        if len(reps) != 1:
            rows.append({"P": hi - lo, "wrong": len(reps)})
            continue
        rep = reps[0]
        s_lo, s_hi = max(rep["start"], lo), min(rep["end"], hi)
        sp = max(0, s_hi - s_lo)
        u = union_ns([(r["start"], r["end"]) for r in nongraph] + [(s_lo, s_hi)], lo, hi)
        rows.append({"P": hi - lo, "S": sp, "D_out": u - sp, "idle_out": (hi - lo) - u,
                     "out_kernels": sum(1 for r in nongraph if r["kind"] == "kernel" and lo <= r["start"] < hi)})
    selected = bool(positions)
    if selected:                                             # registered positions: no edge trim afterwards
        rows = rows[positions[0]:positions[0] + positions[1]]
    wrong = sum(1 for r in rows if "wrong" in r)
    rows = [r for r in rows if "wrong" not in r]
    kept, dropped = _filter(rows, keep, "out_kernels", (0, len(rows)) if selected else None) if rows else ([], Counter())
    if wrong:
        dropped["not_exactly_one_replay"] = wrong
    if len(kept) < min_steps:
        return {"status": "void", "why": f"{len(kept)} steady steps kept < {min_steps} (dropped {dict(dropped)})", "dropped": dict(dropped)}
    return {"status": "ok", "delim": delim, "steps_kept": len(kept), "dropped": dict(dropped), "steady_graph_exec": steady_exec,
            "other_graph_replays": len(g) - len(steady), **{k: _stat([r[k] for r in kept]) for k in ("P", "S", "D_out", "idle_out")}}


# ---------------------------------------------------------------------------------------------------------- node mode --
def node_mode(ex, spec, keep=KEEP, positions=None, min_steps=MIN_STEPS, moe_layers=MOE_LAYERS):
    by_corr = defaultdict(list)
    for r in ex["kernels"] + ex["copies"]:
        if not _nongraph(r):
            by_corr[r["corr"]].append(r)
    launches = [x for x in ex["launches"] if by_corr.get(x["corr"])]
    if not launches:
        return {"status": "void", "why": "no graph node records (was the export taken with --cuda-graph-trace=node?)"}
    modal_n = _modal(len(by_corr[x["corr"]]) for x in launches)
    steady = [x for x in launches if len(by_corr[x["corr"]]) == modal_n]
    outside = [r for r in ex["kernels"] + ex["copies"] if _nongraph(r)]
    rows, names = [], defaultdict(Counter)
    for a, b in zip(steady, steady[1:]):
        ins = sorted(by_corr[a["corr"]], key=lambda r: r["start"])          # "node order" = start-time order in the replay
        lo, hi = ins[0]["start"], min(r["start"] for r in by_corr[b["corr"]])
        outs = [r for r in outside if lo <= r["start"] < hi]
        u_in = union_ns([(r["start"], r["end"]) for r in ins], lo, hi)        # kernels, copies and memsets (round 2 M5)
        sp = max(r["end"] for r in ins) - lo
        row = {"P": hi - lo, "U_in": u_in, "span": sp, "I_node": sp - u_in,
               "k_in": sum(r["kind"] == "kernel" for r in ins), "out_kernels": sum(r["kind"] == "kernel" for r in outs)}
        st = {}
        for side, seq, flag in (("in", ins, True), ("out", outs, False)):
            for r, cls in zip(seq, classify_seq(seq, spec, flag, st if flag else None)):
                row[f"{side}:{cls}"] = row.get(f"{side}:{cls}", 0) + (r["end"] - r["start"])
                names[r["name"]][cls] += 1
        if spec.get("segment"):
            row["seg_broken"] = int(st.get("starts", 0) != moe_layers or st.get("ends", 0) != moe_layers
                                    or bool(st.get("nested")) or bool(st.get("open_at_end")))
            row["seg_starts"], row["seg_ends"] = st.get("starts", 0), st.get("ends", 0)
        rows.append(row)
    kept, dropped = _filter(rows, keep, "out_kernels", positions)
    if len(kept) < min_steps:
        return {"status": "void", "why": f"{len(kept)} steady launches kept < {min_steps} (dropped {dict(dropped)})", "dropped": dict(dropped)}
    cls = {c: _stat([r.get(f"in:{c}", 0) + r.get(f"out:{c}", 0) for r in kept]) for c in CLASSES}
    cls_in = {c: _stat([r.get(f"in:{c}", 0) for r in kept])["median_ms"] for c in CLASSES}
    dev = sum(sum(r.get(f"{s_}:{c}", 0) for s_ in ("in", "out") for c in CLASSES) for r in kept)
    resid = sum(r.get("in:residual", 0) + r.get("out:residual", 0) for r in kept)
    seg = None
    if spec.get("segment"):
        seg = {"replays_broken": sum(r["seg_broken"] for r in kept), "starts_modal": _modal(r["seg_starts"] for r in kept),
               "ends_modal": _modal(r["seg_ends"] for r in kept), "moe_layers": moe_layers}
    return {"status": "ok", "steps_kept": len(kept), "dropped": dict(dropped), "kernels_per_graph_modal": modal_n,
            "other_launches": len(launches) - len(steady), "U_in": _stat([r["U_in"] for r in kept]),
            "span": _stat([r["span"] for r in kept]), "I_in_node": _stat([r["I_node"] for r in kept]), "segments": seg,
            "kernels_in_graph": statistics.median(r["k_in"] for r in kept),
            "kernels_out_graph": statistics.median(r["out_kernels"] for r in kept),
            "class": cls, "class_in_graph_median_ms": cls_in, "residual_fraction": round(resid / dev, 4) if dev else 0.0,
            "name_map": {n: dict(c) for n, c in sorted(names.items())},
            "top_residual_kernels": [n for n, c in sorted(names.items(), key=lambda kv: -kv[1].get("residual", 0)) if c.get("residual")][:8]}


# ---------------------------------------------------------------------------------------------------------------- arm --
BLOCKING = ("NODE_TRACE_INFLATED", "CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS")


def _diag_bad(ex):
    return [d["text"][:200] for d in ex.get("diagnostics", []) if any(k in d["text"].lower() for k in DIAG_BAD)]


def arm(graph_ex, node_ex, engine, batch, spec, delim="graph", d2h_bytes=None, unprofiled_ms=None, positions=None, clocks=None,
        min_steps=MIN_STEPS, moe_layers=MOE_LAYERS):
    gm = graph_mode(graph_ex, delim, d2h_bytes, positions=positions, min_steps=min_steps)
    nm = node_mode(node_ex, spec, positions=positions, min_steps=min_steps, moe_layers=moe_layers)
    rec = {"engine": engine, "batch": batch, "graph": gm, "node": nm, "unprofiled_ms": unprofiled_ms, "positions": positions,
           "clocks_mhz": clocks, "gates": {}, "labels": []}
    bad = _diag_bad(graph_ex) + _diag_bad(node_ex)
    rec["nsys_diagnostics"] = [d["text"][:200] for d in graph_ex.get("diagnostics", []) + node_ex.get("diagnostics", [])][:20]
    if gm["status"] != "ok" or nm["status"] != "ok":
        rec["status"] = "void"
        return rec
    p, s, u = gm["P"]["median_ms"], gm["S"]["median_ms"], nm["U_in"]["median_ms"]
    g = rec["gates"]
    g["inflate"] = None if unprofiled_ms is None else round(p / unprofiled_ms - 1, 4)
    if g["inflate"] is None:
        rec["labels"].append("NO_UNPROFILED_TIMING")
    elif abs(g["inflate"]) > G_INFLATE:
        rec["labels"].append("PROFILER_INFLATED")
    g["node_union_over_span"] = round(u / s, 4) if s else None
    g["node_span_over_span"] = round(nm["span"]["median_ms"] / s, 4) if s else None
    if not s or u > s * G_NODE or nm["span"]["median_ms"] > s * G_NODE:
        rec["labels"].append("NODE_TRACE_INFLATED")
    g["map_residual_fraction"] = nm["residual_fraction"]
    if nm["residual_fraction"] > G_MAP:
        rec["labels"].append("CLASS_MAP_INCOMPLETE")
    if nm["segments"] and nm["segments"]["replays_broken"]:
        rec["labels"].append("CLASS_MAP_SEGMENT_BROKEN")
    if bad:
        rec["labels"].append("NSYS_DIAGNOSTIC_ERRORS")
        rec["nsys_diagnostic_errors"] = bad[:10]
    if clocks:
        vals = [v for v in clocks.values() if v]
        g["clock_spread"] = round(max(vals) / min(vals) - 1, 4) if vals else None
        if g["clock_spread"] is not None and g["clock_spread"] > G_CLOCK:
            rec["labels"].append("CLOCK_MISMATCH")
    terms = {c: nm["class"][c]["median_ms"] for c in CLASSES}
    terms["I_in"] = round(s - u, 6)
    terms["idle_out"] = gm["idle_out"]["median_ms"]
    terms["O"] = round(sum(nm["class"][c]["median_ms"] for c in CLASSES) + terms["I_in"] + terms["idle_out"] - p, 6)
    noise = {c: nm["class"][c]["iqr_ms"] for c in CLASSES}
    noise.update(I_in=round(gm["S"]["iqr_ms"] + nm["U_in"]["iqr_ms"], 6), idle_out=gm["idle_out"]["iqr_ms"], O=None)
    rec.update(P_ms=p, terms=terms, noise_iqr=noise, I_in_node_ms=nm["I_in_node"]["median_ms"],
               status="ok" if not rec["labels"] else "labelled")
    return rec


def gap(e, c, sc1_ratio=None):
    """dP = sum(dclass) - dO + dI_in + didle_out, e4b minus the comparator (positive = e4b slower). The named cause is the
    largest-|d| nameable term with the sign of dP, >= 50 % of |dP| and above twice its noise (the sum of the two arms'
    per-step interquartile ranges: a spread, not a confidence interval)."""
    why = [f"{x['engine']}: {lab}" for x in (e, c) for lab in x.get("labels", []) if lab in BLOCKING]
    if e.get("status") == "void" or c.get("status") == "void" or why:
        return {"status": "unread", "why": why or ["an arm's capture is VOID"], "engine": e.get("engine"), "comparator": c.get("engine")}
    dp = round(e["P_ms"] - c["P_ms"], 6)
    d = {k: round(e["terms"][k] - c["terms"][k], 6) for k in e["terms"]}
    noise = {k: (None if e["noise_iqr"].get(k) is None else round(e["noise_iqr"][k] + c["noise_iqr"][k], 6)) for k in e["terms"]}
    inflated = any(lab in ("PROFILER_INFLATED", "NO_UNPROFILED_TIMING") for x in (e, c) for lab in x["labels"])
    nameable = [k for k in d if k != "O" and not (k == "idle_out" and inflated)]
    ratio = round(e["unprofiled_ms"] / c["unprofiled_ms"], 4) if e.get("unprofiled_ms") and c.get("unprofiled_ms") else None
    rec = {"engine": e["engine"], "comparator": c["engine"], "batch": e["batch"], "delta_P_ms": dp, "delta": d, "noise_iqr": noise,
           "profiled_P_ms": [e["P_ms"], c["P_ms"]], "unprofiled_ms": [e.get("unprofiled_ms"), c.get("unprofiled_ms")],
           "census_ratio_unprofiled": ratio, "sc1_ratio": sc1_ratio, "idle_out_nameable": not inflated, "delta_O_ms": d["O"],
           "labels": sorted({lab for x in (e, c) for lab in x.get("labels", [])})}
    if dp and abs(d["O"]) >= NAMED_SHARE * abs(dp):
        rec.update(status="ok", reading="UNEXPLAINED", named_cause=None)
        return rec
    order = sorted(nameable, key=lambda k: -abs(d[k]))
    named = next((k for k in order if dp and (d[k] > 0) == (dp > 0) and abs(d[k]) >= NAMED_SHARE * abs(dp)
                  and (noise[k] is None or abs(d[k]) > 2 * noise[k])), None)
    rec.update(status="ok", reading="named" if named else "spread", named_cause=named, top_three=[[k, d[k]] for k in order[:3]])
    return rec


# ---------------------------------------------------------------------------------------------------------- self-test --
def _mkdb(path, kernels=(), copies=(), launches=(), graphs=()):
    con = sqlite3.connect(path)
    con.executescript("""
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER PRIMARY KEY, name TEXT, label TEXT);
INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'CUDA_MEMCPY_KIND_HTOD', 'Host-to-Device'), (2, 'CUDA_MEMCPY_KIND_DTOH', 'Device-to-Host');
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, streamId INTEGER,
  correlationId INTEGER, globalPid INTEGER, demangledName INTEGER, shortName INTEGER, gridX INTEGER, gridY INTEGER,
  gridZ INTEGER, graphNodeId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, streamId INTEGER,
  correlationId INTEGER, globalPid INTEGER, bytes INTEGER, copyKind INTEGER, graphNodeId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
  correlationId INTEGER, nameId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
  streamId INTEGER, correlationId INTEGER, globalPid INTEGER, graphId INTEGER, graphExecId INTEGER);
""")
    ids = {}

    def sid(v):
        if v not in ids:
            ids[v] = len(ids) + 1
            con.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[v], v))
        return ids[v]
    for st, en, corr, node, name, grid in kernels:
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,0,1,7,?,1,?,?,?,?,?,?)", (st, en, corr, sid(name), sid(name), *grid, node))
    for st, en, corr, node, nbytes, kind in copies:
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,0,1,8,?,1,?,?,?)", (st, en, corr, nbytes, kind, node))
    for st, en, corr in launches:
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,0,1,?,?)", (st, en, corr, sid("cudaGraphLaunch_v10000")))
    for st, en, corr, gexec in graphs:
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?,?,0,1,7,?,1,1,?)", (st, en, corr, gexec))
    con.commit()
    con.close()


def _it(name, grid=(1, 1, 1)):
    return {"name": name, "demangled": name, "grid": grid, "kind": "kernel"}


def self_test():
    d = tempfile.mkdtemp()
    us = 1000
    checks = 0
    # graph mode, graph delimiter: replays every 1000 us (exec 5), span 600, an out-of-graph kernel 100 us after each graph
    # and a 50 us copy -> idle_out 250, D_out 150. One stray exec-9 replay; one long (re-warm) period, dropped.
    gk, gc, gg = [], [], []
    t = 0
    for i in range(10):
        t += 1000 * us if i != 6 else 2500 * us
        gg.append((t, t + 600 * us, 100 + i, 5))
        gk.append((t + 650 * us, t + 750 * us, 200 + i, None, "argmax_k", (1, 1, 1)))
        gc.append((t + 800 * us, t + 850 * us, 300 + i, None, 64, 2))
    gg.append((t + 300 * us, t + 310 * us, 999, 9))
    p = os.path.join(d, "g.sqlite")
    _mkdb(p, kernels=gk, copies=gc, graphs=gg)
    r = graph_mode(load(p), min_steps=3)
    assert r["status"] == "ok" and r["P"]["median_ms"] == 1.0 and r["S"]["median_ms"] == 0.6, r
    assert r["idle_out"]["median_ms"] == 0.25 and r["D_out"]["median_ms"] == 0.15, r
    assert r["dropped"].get("period_over_1.5x_median") == 1 and r["steps_kept"] == 6 and r["other_graph_replays"] == 1, r
    assert graph_mode(load(p))["status"] == "void"                    # the registered floor: 6 kept steps < 56 is VOID
    checks += 4
    # graphNodeId 0 means eager, exactly like NULL (round 2 M7)
    p0 = os.path.join(d, "g0.sqlite")
    _mkdb(p0, kernels=[(a, b, c, 0, n, gr) for a, b, c, _x, n, gr in gk], copies=[(a, b, c, 0, nb, k) for a, b, c, _x, nb, k in gc], graphs=gg)
    assert graph_mode(load(p0), min_steps=3)["D_out"]["median_ms"] == 0.15
    checks += 1
    # d2h delimiter (llama.cpp): the whole run is captured; registered positions are selected by counting logits copies;
    # an eager re-warm step (no replay) inside them is dropped and counted
    lk, lc, lg = [], [], []
    t = 0
    for i in range(30):
        t += 1000 * us
        lc.append((t, t + 40 * us, 400 + i, None, 4 * 151936, 2))
        lc.append((t + 50 * us, t + 60 * us, 500 + i, None, 64, 1))
        if i != 17:
            lg.append((t + 100 * us, t + 700 * us, 600 + i, 3))
        else:
            lk.append((t + 100 * us, t + 900 * us, 700, None, "eager_kernel", (1, 1, 1)))
    p = os.path.join(d, "l.sqlite")
    _mkdb(p, kernels=lk, copies=lc, graphs=lg)
    r = graph_mode(load(p), delim="d2h", d2h_bytes=4 * 151936, positions=(10, 12), min_steps=3)
    assert r["status"] == "ok" and r["dropped"].get("not_exactly_one_replay") == 1 and r["steps_kept"] == 11, r
    assert r["P"]["median_ms"] == 1.0 and r["S"]["median_ms"] == 0.6 and r["idle_out"]["median_ms"] == 0.35, r
    checks += 2
    # classification: name, grid, in/out graph, MoE segment by node order, inherit-next / inherit-prev, residual
    spec = {"rules": [{"class": "attn", "match": "attn"}, {"class": "norm_elem", "match": "rmsnorm"},
                      {"class": "dense_gemm", "match": "gemv", "where": {"gridY_eq": 1}},
                      {"class": "dense_gemm", "match": "nvjet", "where": {"gridX_le": 4}},
                      {"class": "dense_gemm", "match": "nvjet", "where": {"in_graph": False}},
                      {"class": "sample", "match": "argmax"}],
            "segment": {"start": ["router_epi"], "end": ["combine"]}, "expert_names": ["gemv", "swiglu"],
            "matmul_names": ["gemv", "nvjet"], "inherit_next": ["quant_x"], "inherit_prev": ["reduce_partials"]}
    seq = [_it("rmsnorm"), _it("quant_x"), _it("gemv", (8, 1, 1)), _it("reduce_partials"), _it("attn_decode"), _it("nvjet", (2, 1, 1)),
           _it("router_epi"), _it("quant_x"), _it("gemv", (8, 8, 1)), _it("swiglu"), _it("index_select"), _it("gemv", (8, 8, 1)),
           _it("combine"), _it("mystery")]
    st = {}
    got = classify_seq(seq, spec, True, st)
    want = ["norm_elem", "dense_gemm", "dense_gemm", "dense_gemm", "attn", "dense_gemm", "moe_route", "moe_expert", "moe_expert",
            "moe_expert", "moe_route", "moe_expert", "moe_route", "residual"]
    assert got == want and st == {"starts": 1, "ends": 1}, (list(zip([s_["name"] for s_ in seq], got, want)), st)
    assert classify_seq([_it("nvjet", (900, 1, 1))], spec, False) == ["dense_gemm"]     # the lm_head outside the graph
    assert classify_seq([_it("nvjet", (900, 1, 1))], spec, True) == ["residual"]        # inside: only the gridX rule
    # an unclosed segment (the end anchor missing) is detected when the next layer's router follows expert kernels
    st2 = {}
    classify_seq([_it("router_epi"), _it("gemv", (8, 8, 1)), _it("rmsnorm"), _it("attn_decode"), _it("router_epi"),
                  _it("gemv", (8, 8, 1)), _it("combine")], spec, True, st2)
    assert st2.get("nested") == 1 and st2["starts"] == 2 and st2["ends"] == 1, st2
    checks += 4
    # node mode: per replay rmsnorm 100, an in-graph D2D copy 50, dense gemv 200, attn 150 (a 50 us gap); outside: argmax 40,
    # copy 20. U_in counts the in-graph copy (round 2 M5), so I_in is the 50 us gap only.
    nk, nc, nl = [], [], []
    for i in range(10):
        t, c = 1_000_000 * (i + 1), 1000 + i
        nl.append((t - 20, t - 10, c))
        nk += [(t, t + 100 * us, c, 1, "rmsnorm", (1, 1, 1)), (t + 150 * us, t + 350 * us, c, 2, "gemv", (8, 1, 1)),
               (t + 400 * us, t + 550 * us, c, 3, "attn_decode", (4, 1, 1)),
               (t + 650 * us, t + 690 * us, 2000 + i, None, "argmax", (1, 1, 1))]
        nc += [(t + 100 * us, t + 150 * us, c, 4, 4096, 8), (t + 700 * us, t + 720 * us, 3000 + i, None, 64, 2)]
    pn = os.path.join(d, "n.sqlite")
    _mkdb(pn, kernels=nk, copies=nc, launches=nl)
    nspec = dict(spec, segment={})
    nm = node_mode(load(pn), nspec, min_steps=3)
    assert nm["status"] == "ok" and nm["U_in"]["median_ms"] == 0.5 and nm["kernels_in_graph"] == 3, nm
    assert nm["span"]["median_ms"] == 0.55 and nm["I_in_node"]["median_ms"] == 0.05, nm
    assert nm["class"]["dense_gemm"]["median_ms"] == 0.2 and nm["class"]["sample"]["median_ms"] == 0.04, nm
    assert nm["class"]["memcpy"]["median_ms"] == 0.07 and nm["residual_fraction"] == 0.0, nm
    checks += 4
    # arm: the same engine's graph mode -> P 1.0, S 0.6 (I_in = 0.6 - 0.5 = 0.1); the identity holds exactly
    gp = os.path.join(d, "g2.sqlite")
    gg2 = [(1_000_000 * (i + 1), 1_000_000 * (i + 1) + 600 * us, 4000 + i, 5) for i in range(10)]
    gk2 = [(1_000_000 * (i + 1) + 650 * us, 1_000_000 * (i + 1) + 690 * us, 5000 + i, None, "argmax", (1, 1, 1)) for i in range(10)]
    gc2 = [(1_000_000 * (i + 1) + 700 * us, 1_000_000 * (i + 1) + 720 * us, 6000 + i, None, 64, 2) for i in range(10)]
    _mkdb(gp, kernels=gk2, copies=gc2, graphs=gg2)
    a = arm(load(gp), load(pn), "e4b", 1, nspec, unprofiled_ms=0.98, min_steps=3)
    assert a["status"] == "ok" and a["terms"]["I_in"] == 0.1 and a["terms"]["idle_out"] == 0.34, (a["labels"], a["terms"])
    assert abs(a["P_ms"] - (sum(a["terms"][c] for c in CLASSES) - a["terms"]["O"] + a["terms"]["I_in"] + a["terms"]["idle_out"])) < 1e-9
    assert "PROFILER_INFLATED" in arm(load(gp), load(pn), "e4b", 1, nspec, unprofiled_ms=0.9, min_steps=3)["labels"]
    a3 = arm(load(gp), load(pn), "e4b", 1, nspec, unprofiled_ms=1.0, min_steps=3, clocks={"pass1": 2800, "graph": 2810, "node": 2950})
    assert "CLOCK_MISMATCH" in a3["labels"], a3["labels"]
    # node-mode inflation: replays stretched past the graph span -> NODE_TRACE_INFLATED (the span gate, round 2 M10)
    pn2 = os.path.join(d, "n2.sqlite")
    nk2 = [(a_, b_ + 300 * us if name == "attn_decode" else b_, c, nd, name, gr) for a_, b_, c, nd, name, gr in nk]
    _mkdb(pn2, kernels=nk2, copies=nc, launches=nl)
    assert "NODE_TRACE_INFLATED" in arm(load(gp), load(pn2), "e4b", 1, nspec, unprofiled_ms=1.0, min_steps=3)["labels"]
    checks += 5
    # nsys diagnostics with an error label the arm and block its gaps (R17)
    pd = os.path.join(d, "diag.sqlite")
    _mkdb(pd, kernels=gk2, copies=gc2, graphs=gg2)
    con = sqlite3.connect(pd)
    con.execute("CREATE TABLE DIAGNOSTIC_EVENT (timestamp INTEGER, timestampType INTEGER, source INTEGER, severity INTEGER, text TEXT, globalPid INTEGER)")
    con.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (1, 0, 0, 2, 'CUPTI: activity buffer full, records dropped', 1)")
    con.commit()
    con.close()
    ad = arm(load(pd), load(pn), "e4b", 1, nspec, unprofiled_ms=1.0, min_steps=3)
    assert "NSYS_DIAGNOSTIC_ERRORS" in ad["labels"] and gap(ad, a)["status"] == "unread", ad["labels"]
    checks += 1

    # gap readings: named (I_in), spread, UNEXPLAINED, unread, idle_out not nameable when inflated, and the first QUALIFYING
    # term is named even when a larger term has the opposite sign (round 2 m2)
    def mk(engine, P, terms, labels=()):
        tt = dict({c: 0.0 for c in CLASSES}, I_in=0.0, idle_out=0.0)
        tt.update(terms)
        tt["O"] = round(sum(tt[c] for c in CLASSES) + tt["I_in"] + tt["idle_out"] - P, 6)
        return {"engine": engine, "batch": 1, "status": "ok", "labels": list(labels), "P_ms": P, "terms": tt,
                "noise_iqr": dict({c: 0.001 for c in CLASSES}, I_in=0.002, idle_out=0.002, O=None), "unprofiled_ms": P}
    lc_ = mk("llamacpp", 3.0, {"moe_expert": 1.1, "attn": 0.7, "dense_gemm": 0.5, "I_in": 0.3, "idle_out": 0.4})
    g1 = gap(mk("e4b", 4.5, {"moe_expert": 1.2, "attn": 0.8, "dense_gemm": 0.6, "I_in": 1.4, "idle_out": 0.5}), lc_)
    assert g1["reading"] == "named" and g1["named_cause"] == "I_in" and g1["delta_P_ms"] == 1.5, g1
    g2 = gap(mk("e4b", 4.5, {"moe_expert": 1.6, "attn": 1.0, "dense_gemm": 0.9, "I_in": 0.6, "idle_out": 0.4}), lc_)
    assert g2["reading"] == "spread" and g2["named_cause"] is None and len(g2["top_three"]) == 3, g2
    g3 = gap(mk("e4b", 4.5, {"moe_expert": 1.1, "attn": 0.7, "dense_gemm": 0.5, "I_in": 0.3, "idle_out": 0.4}), lc_)
    assert g3["reading"] == "UNEXPLAINED", g3
    assert gap(mk("e4b", 4.5, {}, labels=["CLASS_MAP_SEGMENT_BROKEN"]), lc_)["status"] == "unread"
    b16e = mk("e4b", 9.6, {"moe_expert": 5.0, "attn": 2.0, "dense_gemm": 1.5, "I_in": 0.6, "idle_out": 0.5})
    b16l = mk("llamacpp", 15.4, {"moe_expert": 6.0, "attn": 2.2, "dense_gemm": 1.6, "I_in": 0.4, "idle_out": 5.2})
    g5 = gap(b16e, b16l)
    assert g5["named_cause"] == "idle_out" and g5["delta_P_ms"] == -5.8, g5
    g6 = gap(b16e, dict(b16l, labels=["PROFILER_INFLATED"]))
    assert g6["named_cause"] != "idle_out" and not g6["idle_out_nameable"], g6
    e7 = mk("e4b", 6.5, {"moe_expert": 0.3, "attn": 3.5, "dense_gemm": 0.5, "I_in": 2.2})
    l7 = mk("llamacpp", 5.0, {"moe_expert": 3.6, "attn": 0.7, "dense_gemm": 0.5, "I_in": 0.2})
    g7 = gap(e7, l7)                    # dmoe -3.3 is the largest |d| but opposite-signed; dattn +2.8 (>= 0.75) is the named cause
    assert g7["named_cause"] == "attn" and g7["delta_O_ms"] == 0.0, g7
    checks += 7
    # a wrong-mode export is VOID with its reason, never a zero
    assert graph_mode(load(pn), min_steps=3)["status"] == "void" and node_mode(load(gp), spec, min_steps=3)["status"] == "void"
    checks += 1
    print(f"self-test OK ({checks} checks)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("arm", "gap"))
    ap.add_argument("--graph")
    ap.add_argument("--node")
    ap.add_argument("--engine")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--classes")
    ap.add_argument("--delim", default="graph", choices=("graph", "d2h"))
    ap.add_argument("--d2h-bytes", type=int)
    ap.add_argument("--unprofiled-ms", type=float, help="pass 1's decode_ms_per_step_median (SC1's arm, this box)")
    ap.add_argument("--positions", help="START:COUNT of the steady steps to read, when the capture holds the whole run (llama.cpp)")
    ap.add_argument("--clocks", help="pass1=MHz,graph=MHz,node=MHz: median SM clock per pass (the sampler's clocks.sm)")
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS)
    ap.add_argument("--moe-layers", type=int, default=MOE_LAYERS)
    ap.add_argument("--e4b")
    ap.add_argument("--comp")
    ap.add_argument("--sc1-ratio", type=float)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "arm":
        pos = tuple(int(x) for x in a.positions.split(":")) if a.positions else None
        clk = {k: float(v) for k, v in (kv.split("=") for kv in a.clocks.split(",") if "=" in kv and kv.split("=")[1])} if a.clocks else None
        rec = arm(load(a.graph), load(a.node), a.engine, a.batch, load_classes(a.classes, a.engine), a.delim, a.d2h_bytes, a.unprofiled_ms,
                  pos, clk, a.min_steps, a.moe_layers)
    elif a.cmd == "gap":
        with open(a.e4b) as f, open(a.comp) as g:
            rec = gap(json.load(f), json.load(g), a.sc1_ratio)
    else:
        ap.error("arm | gap | --self-test")
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("SC1B " + json.dumps({k: rec.get(k) for k in ("engine", "comparator", "batch", "status", "reading", "named_cause", "labels")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
