#!/usr/bin/env python3
"""pername_table.py -- the per-name table of one node-mode census export (bench/pername/PREREG-pername.md; e4b#1313).

SC1b's class census bills each kernel its whole duration. Under programmatic dependent launch a kernel starts while its
predecessor still runs and waits there, so the class sums double-count (the census's ``overlap_in`` term). This table
walks each kept replay once, in start order, with a running frontier (the latest end seen so far), and splits every
in-graph item into three parts that add up, exactly and in integer nanoseconds, to the replay's span:

    gap_before   idle on the critical path before the item starts:  max(0, start - frontier)
    excl         the item's own advance of the frontier:              max(0, end - max(start, frontier))
    billed       the rest of its duration, run under its predecessor: (end - start) - excl

``gap_before + excl`` is what removing the item would take off the step (an upper bound: whatever replaces it costs
something). The replays are the ones SC1b's ``node_mode`` keeps (the same slicing, the same ``_filter``), so the table
and the class census read the same steps.

Keys are the kernel name and its grid; the two expert GEMVs share a name, so ``_gemv_nf4_bw`` also carries its role
(``gate_up`` just before ``_swiglu_rows``, ``down`` just after).

    python pername_table.py --node census_e4b_b1_node.sqlite --classes kernel_classes_pername.json --out pername_b1.json
    python pername_table.py --self-test
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc1b_census as sc  # noqa: E402  (staged at P136's pinned bytes: load, node slicing helpers, classify_seq, _filter)

SWIGLU, GEMV = "_swiglu_rows", "_gemv_nf4_bw"
PARTS = ("n", "dur", "excl", "billed", "gap_before")


def _replays(ex):
    """(in-graph items in start order, out-of-graph items, lo, hi) per consecutive pair of steady launches: SC1b
    ``node_mode``'s slicing, line for line."""
    by_corr = defaultdict(list)
    for r in ex["kernels"] + ex["copies"]:
        if not sc._nongraph(r):
            by_corr[r["corr"]].append(r)
    launches = [x for x in ex["launches"] if by_corr.get(x["corr"])]
    if not launches:
        return None, None
    modal_n = sc._modal(len(by_corr[x["corr"]]) for x in launches)
    steady = [x for x in launches if len(by_corr[x["corr"]]) == modal_n]
    outside = [r for r in ex["kernels"] + ex["copies"] if sc._nongraph(r)]
    out = []
    for a, b in zip(steady, steady[1:]):
        ins = sorted(by_corr[a["corr"]], key=lambda r: r["start"])
        lo, hi = ins[0]["start"], min(r["start"] for r in by_corr[b["corr"]])
        outs = [r for r in outside if lo <= r["start"] < hi]
        out.append((ins, outs, lo, hi))
    return out, modal_n


def key_of(items, i):
    it = items[i]
    k = f"{it['name']}|{'x'.join(str(g) for g in it['grid'])}"
    if it["name"] == GEMV:
        prev_k = next((items[j]["name"] for j in range(i - 1, -1, -1) if items[j]["kind"] == "kernel"), None)
        next_k = next((items[j]["name"] for j in range(i + 1, len(items)) if items[j]["kind"] == "kernel"), None)
        role = "gate_up" if next_k == SWIGLU else "down" if prev_k == SWIGLU else "other"
        k += f"|{role}"
    return k


def split_replay(ins, lo):
    """Per-item (gap_before, excl, billed) in ns, and the replay's span; sum(gap_before) + sum(excl) == span."""
    frontier, parts = lo, []
    for r in ins:
        gap = max(0, r["start"] - frontier)
        excl = max(0, r["end"] - max(r["start"], frontier))
        parts.append((gap, excl, (r["end"] - r["start"]) - excl))
        frontier = max(frontier, r["end"])
    return parts, frontier - lo


def table(ex, spec, keep=sc.KEEP, min_steps=sc.MIN_STEPS) -> dict:
    reps, modal_n = _replays(ex)
    if not reps:
        return {"status": "void", "why": "no graph node records (was the export taken with --cuda-graph-trace=node?)"}
    rows = []
    for ins, outs, lo, hi in reps:
        parts, span = split_replay(ins, lo)
        classes = sc.classify_seq(ins, spec, True)
        per = defaultdict(lambda: dict.fromkeys(PARTS, 0))
        cls_of = defaultdict(Counter)
        for i, ((gap, excl, billed), cls) in enumerate(zip(parts, classes)):
            k = key_of(ins, i)
            p = per[k]
            p["n"] += 1
            p["dur"] += ins[i]["end"] - ins[i]["start"]
            p["excl"] += excl
            p["billed"] += billed
            p["gap_before"] += gap
            cls_of[k][cls] += 1
        closure = sum(g for g, _e, _b in parts) + sum(e for _g, e, _b in parts) - span
        rows.append({"P": hi - lo, "span": span, "tail": hi - lo - span, "closure": closure,
                     "out_kernels": sum(r["kind"] == "kernel" for r in outs), "per": per, "cls": cls_of})
    kept, dropped = sc._filter(rows, keep, "out_kernels")
    if len(kept) < min_steps:
        return {"status": "void", "why": f"{len(kept)} steady launches kept < {min_steps} (dropped {dict(dropped)})"}
    keys = sorted({k for r in kept for k in r["per"]})
    ms = lambda xs: round(statistics.median(xs) / 1e6, 6)  # noqa: E731
    out_rows = []
    for k in keys:
        cls = Counter()
        for r in kept:
            cls.update(r["cls"].get(k, {}))
        vals = {p: [r["per"][k][p] if k in r["per"] else 0 for r in kept] for p in PARTS}
        name, grid, *role = k.split("|")
        row = {"key": k, "name": name, "grid": grid, "role": role[0] if role else None,
               "class": cls.most_common(1)[0][0], "classes": dict(cls), "n_per_step": statistics.median(vals["n"])}
        row.update({f"{p}_ms": ms(vals[p]) for p in PARTS if p != "n"})
        rem = [g + e for g, e in zip(vals["gap_before"], vals["excl"])]
        row["removable_ms"] = ms(rem)
        # REPORTING (the review of #1756): the per-step spread, and the share of kept steps where this line alone is
        # >= 3 % of that step's own period -- so a median near a bar shows as marginal
        row["removable_iqr_ms"] = sc._stat(rem)["iqr_ms"]
        row["steps_ge_3pct"] = round(sum(x >= 0.03 * r["P"] for x, r in zip(rem, kept)) / len(kept), 4)
        out_rows.append(row)
    out_rows.sort(key=lambda r: -r["removable_ms"])
    class_rem = defaultdict(list)
    for r in kept:
        acc = Counter()
        for k, p in r["per"].items():
            acc[r["cls"][k].most_common(1)[0][0]] += p["gap_before"] + p["excl"]
        for c in sc.CLASSES:
            class_rem[c].append(acc.get(c, 0))
    return {"status": "ok", "steps_kept": len(kept), "dropped": dict(dropped), "kernels_per_graph_modal": modal_n,
            "P_ms": ms([r["P"] for r in kept]), "span_ms": ms([r["span"] for r in kept]),
            "tail_ms": ms([r["tail"] for r in kept]), "closure_ns_max": max(abs(r["closure"]) for r in kept),
            "class_removable_ms": {c: ms(v) for c, v in class_rem.items()}, "rows": out_rows}


# ---------------------------------------------------------------------------------------------------------- self-test --
SPEC = {"rules": [{"class": "moe_expert", "match": [GEMV, SWIGLU]}, {"class": "norm_elem", "match": ["_rmsnorm_rows"]},
                  {"class": "moe_route", "match": ["_router_epilogue"]}]}
US = 1000


def _fake_db(path, steps=60, pdl_overlap_us=2, swiglu_us=3):
    """`steps` + 2 replays of: rmsnorm, router, gate_up GEMV, swiglu, down GEMV, rmsnorm (one stream). The swiglu starts
    `pdl_overlap_us` before gate_up ends (PDL: launched early, waiting), so its duration over-bills it."""
    kernels, launches, t, corr = [], [], 0, 0
    for _s in range(steps + 2):
        corr += 1
        launches.append((t, t + 5 * US, corr))
        t0 = t + 10 * US
        seq = [("_rmsnorm_rows", (16, 1, 1), 0, 2), ("_router_epilogue", (1, 1, 1), 1, 3),
               (GEMV, (96, 8, 1), 1, 12), (SWIGLU, (8, 2, 1), -pdl_overlap_us, swiglu_us + pdl_overlap_us),
               (GEMV, (128, 8, 1), 1, 7), ("_rmsnorm_rows", (16, 1, 1), 1, 2)]
        cur = t0
        for node, (name, grid, gap, dur) in enumerate(seq, start=1):
            st = cur + gap * US
            kernels.append((st, st + dur * US, corr, node, name, grid))
            cur = st + dur * US
        t = cur + 20 * US
    sc._mkdb(path, kernels=kernels, launches=launches)


def self_test() -> int:
    cases = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = os.path.join(d, "n.sqlite")
        _fake_db(db)
        t = table(sc.load(db), SPEC, keep=sc.KEEP, min_steps=56)
        by = {r["key"]: r for r in t["rows"]}
        cases.append(("ok over 59 kept steps (62 launches, 61 replays, edges trimmed)", t["status"] == "ok" and t["steps_kept"] == 59))
        cases.append(("gap + excl closes every replay exactly", t["closure_ns_max"] == 0))
        sw = by.get(f"{SWIGLU}|8x2x1")
        # swiglu runs 5 us, 2 of them under gate_up: it advances the frontier by 3 us and is billed 2
        cases.append(("swiglu's excl is its own advance, not its duration", sw and (sw["dur_ms"], sw["excl_ms"], sw["billed_ms"])
                      == (0.005, 0.003, 0.002)))
        cases.append(("swiglu's removable time is excl + gap", sw and sw["removable_ms"] == 0.003 and sw["gap_before_ms"] == 0.0))
        # every replay is identical: no spread; 3 us of a 63 us period is 4.8 %, every step over 3 %
        cases.append(("the spread is reported beside the median", sw and sw["removable_iqr_ms"] == 0.0 and sw["steps_ge_3pct"] == 1.0))
        cases.append(("the GEMVs carry their roles", {f"{GEMV}|96x8x1|gate_up", f"{GEMV}|128x8x1|down"} <= set(by)))
        dn = by.get(f"{GEMV}|128x8x1|down")
        cases.append(("down's 1 us launch gap is on the critical path", dn and dn["gap_before_ms"] == 0.001 and dn["excl_ms"] == 0.007))
        cases.append(("the span is the replay's gap + excl time (4 + 29 us)", t["span_ms"] == 0.033))
        cases.append(("class sums of removable time (13 + 3 + 8 us)", t["class_removable_ms"]["moe_expert"] == 0.024))
        short = os.path.join(d, "short.sqlite")
        _fake_db(short, steps=40)
        cases.append(("too few steady steps is void", table(sc.load(short), SPEC)["status"] == "void"))
    bad = [n for n, ok in cases if not ok]
    print(f"pername_table self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--node")
    p.add_argument("--classes")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    t = table(sc.load(a.node), sc.load_classes(a.classes, "e4b"))
    with open(a.out, "w") as f:
        json.dump(t, f, indent=1)
    top = [(r["key"], r["removable_ms"]) for r in t.get("rows", [])[:8]]
    print("PERNAME_TABLE " + json.dumps({"status": t["status"], "steps_kept": t.get("steps_kept"), "P_ms": t.get("P_ms"),
                                         "closure_ns_max": t.get("closure_ns_max"), "top": top, "why": t.get("why")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
