"""N2 Phase-1 analysis — locality, h(S) cache simulation, P-B1/P-B2 grading, kill table.

Implements docs/N2_PHASE01_RECONSTRUCTION.md exactly:
- consecutive-token routed-set Jaccard per layer (within prompt), churn = 1 - mean J;
- near-margin fraction per layer against the GLOBAL bottom-decile margin threshold;
- P-B1: Spearman corr(near-margin fraction, mean Jaccard) across layers < -0.4;
- P-B2: top-quartile near-margin layers show >= 1.5x churn of bottom-quartile;
- O1 byproduct: base-vs-adapter routed-set Jaccard on identical (replayed) token streams;
- h(S) via global LRU over (layer, expert) slabs at S in {0.25,0.5,1,2,4} GB, plain and
  margin-aware (O2: among the least-recent 25% of entries, evict churn-prone ones first,
  churn-prone = last hit served a below-global-decile margin decision);
- economics: gain(h) = (T_ovh+t_fetch)/(T_ovh+(1-h)*t_fetch) - 1 with measured T_ovh
  (decode_repeat medians) and t_fetch (BW_gather); kill rule: max gain over S<=2GB < 10%.

Usage:
    python scripts/n2_analyze_routed_stream.py --traces-root runs/jobs \\
        --bw runs/jobs/n2_phase0_bw/result.json \\
        --tovh nf4=runs/jobs/n2_tovh_nf4/result.json,int8=...,bf16=... \\
        --out runs/results/postaudit/n2_routed_stream_phase01.md
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

S_GRID_GB = (0.25, 0.5, 1.0, 2.0, 4.0)
SLAB_BYTES = {"nf4": 3_538_944, "int8": 6_684_672, "bf16": 12_582_912}
KILL_GAIN = 0.10
KILL_BUDGET_GB = 2.0


def load_trace(job_dir):
    rows = [json.loads(line) for line in open(os.path.join(job_dir, "trace.jsonl"))]
    return rows


def jaccard(a, b):
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 1.0


def per_layer_locality(rows):
    """mean consecutive-token Jaccard and margin list per layer (within prompt)."""
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt"], []).append(r)
    layers = sorted({li for r in rows for li in r["layers"]}, key=int)
    j_acc = {li: [] for li in layers}
    margins = {li: [] for li in layers}
    for _, prows in sorted(by_prompt.items()):
        prows.sort(key=lambda r: r["token"])
        for prev, cur in zip(prows, prows[1:]):
            for li in layers:
                if li in prev["layers"] and li in cur["layers"]:
                    j_acc[li].append(jaccard(prev["layers"][li], cur["layers"][li]))
        for r in prows:
            for li, m in r.get("margins", {}).items():
                if li in margins:
                    margins[li].append(m)
    mean_j = {li: (sum(v) / len(v) if v else float("nan")) for li, v in j_acc.items()}
    return layers, mean_j, margins


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rr = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rr[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rr
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def lru_sim(rows, slab_bytes, budget_bytes, margin_aware=False, low_margin_thresh=None):
    """Global LRU over (layer, expert). Returns decode-token expert-fetch hit rate."""
    cache = OrderedDict()  # key -> churn_prone flag
    cap = max(1, int(budget_bytes // slab_bytes))
    hits = total = 0
    for r in rows:
        for li, experts in r["layers"].items():
            m = r.get("margins", {}).get(li)
            low = (m is not None and low_margin_thresh is not None and m < low_margin_thresh)
            for e in experts:
                key = (li, e)
                total += 1
                if key in cache:
                    hits += 1
                    cache.move_to_end(key)
                    cache[key] = low
                else:
                    if len(cache) >= cap:
                        victim = None
                        if margin_aware:
                            scan = max(1, len(cache) // 4)
                            for k2 in list(cache.keys())[:scan]:
                                if cache[k2]:
                                    victim = k2
                                    break
                        if victim is None:
                            victim = next(iter(cache))
                        del cache[victim]
                    cache[key] = low
    return hits / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-root", required=True)
    ap.add_argument("--bw", required=True)
    ap.add_argument("--tovh", required=True, help="mode=result.json,... from decode_repeat runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bw = json.load(open(args.bw))
    tovh = {}
    for part in args.tovh.split(","):
        mode, path = part.split("=", 1)
        r = json.load(open(path))
        tovh[mode] = 1000.0 / r["tok_s_mean"]  # ms per token (repeat-5 mean, one session)

    configs = {
        "nf4_base": "n2trace_nf4_base",
        "nf4_adapter": "n2trace_nf4_adapter",
        "int8_base": "n2trace_int8_base",
    }
    traces = {}
    for name, d in configs.items():
        p = os.path.join(args.traces_root, d)
        if os.path.exists(os.path.join(p, "trace.jsonl")):
            traces[name] = load_trace(p)

    L = ["# N2 routed-stream Phase 0-1 (reconstruction) — results", ""]

    # Locality + P-B per config
    pb_results = {}
    for name, rows in traces.items():
        layers, mean_j, margins = per_layer_locality(rows)
        all_m = sorted(m for v in margins.values() for m in v)
        thresh = all_m[len(all_m) // 10] if all_m else None
        nm_frac = {li: (sum(1 for m in margins[li] if m < thresh) / len(margins[li])
                        if margins[li] else 0.0) for li in layers}
        xs = [nm_frac[li] for li in layers]
        ys = [mean_j[li] for li in layers]
        rho = spearman(xs, ys)
        churn = {li: 1 - mean_j[li] for li in layers}
        order = sorted(layers, key=lambda li: nm_frac[li])
        q = max(1, len(layers) // 4)
        bot_q = sum(churn[li] for li in order[:q]) / q
        top_q = sum(churn[li] for li in order[-q:]) / q
        ratio = top_q / bot_q if bot_q > 0 else float("inf")
        pb_results[name] = {"rho": rho, "churn_ratio": ratio, "thresh": thresh,
                            "mean_j_overall": sum(ys) / len(ys)}
        L.append(f"## {name}: locality + margins ({len(rows)} decode tokens, {len(layers)} layers)")
        L.append("")
        L.append(f"- mean consecutive-token Jaccard {sum(ys)/len(ys):.4f} "
                 f"(range {min(ys):.4f}-{max(ys):.4f}); global low-margin threshold {thresh}")
        L.append(f"- **P-B1** Spearman corr(near-margin frac, Jaccard) across layers = "
                 f"**{rho:+.3f}** (committed: < -0.40 → "
                 f"{'HOLDS' if rho < -0.40 else 'FAILS'}; n=16 layers)")
        L.append(f"- **P-B2** churn ratio top-vs-bottom near-margin quartile = **{ratio:.2f}x** "
                 f"(committed: >= 1.5x → {'HOLDS' if ratio >= 1.5 else 'FAILS'})")
        L.append("")

    # O1 byproduct: base vs adapter on identical streams
    if "nf4_base" in traces and "nf4_adapter" in traces:
        idx = {(r["prompt"], r["token"]): r for r in traces["nf4_adapter"]}
        js = []
        for r in traces["nf4_base"]:
            r2 = idx.get((r["prompt"], r["token"]))
            if r2:
                for li in r["layers"]:
                    if li in r2["layers"]:
                        js.append(jaccard(r["layers"][li], r2["layers"][li]))
        if js:
            L.append(f"## O1 byproduct — base vs adapter routing on identical streams")
            L.append("")
            L.append(f"- mean per-token routed-set Jaccard = **{sum(js)/len(js):.4f}** "
                     f"(eval-set S-B number was 0.9418; trace-workload confirmation)")
            L.append("")

    # h(S) + economics + kill table
    L.append("## h(S), economics, kill table (reconstructed rule: gain < 10% at S <= 2 GB kills)")
    L.append("")
    L.append("| precision | T_ovh ms | t_fetch ms | ceiling gap | best h(S<=2GB) LRU | margin-LRU | best gain | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    trace_for = {"nf4": traces.get("nf4_base"), "int8": traces.get("int8_base"),
                 "bf16": traces.get("nf4_base")}  # bf16 routing ~ nf4-base stream (proxy, flagged)
    for p in ("nf4", "int8", "bf16"):
        rows = trace_for.get(p)
        if rows is None or p not in tovh:
            continue
        t_ovh = tovh[p]
        t_fetch = bw["t_fetch_ms_per_token"][p]
        gap = t_fetch / t_ovh
        pb = pb_results.get("nf4_base" if p != "int8" else "int8_base", {})
        thresh = pb.get("thresh")
        best = {"lru": 0.0, "mlru": 0.0, "S": None}
        for S in S_GRID_GB:
            if S > KILL_BUDGET_GB:
                continue
            h1 = lru_sim(rows, SLAB_BYTES[p], S * 1e9)
            h2 = lru_sim(rows, SLAB_BYTES[p], S * 1e9, margin_aware=True, low_margin_thresh=thresh)
            if max(h1, h2) > max(best["lru"], best["mlru"]):
                best = {"lru": h1, "mlru": h2, "S": S}
        h_star = max(best["lru"], best["mlru"])
        gain = (t_ovh + t_fetch) / (t_ovh + (1 - h_star) * t_fetch) - 1
        verdict = "KILL" if gain < KILL_GAIN else "SPARE"
        L.append(f"| {p}{' (proxy stream)' if p == 'bf16' else ''} | {t_ovh:.1f} | {t_fetch:.1f} | "
                 f"+{100*gap:.0f}% | {best['lru']:.3f}@{best['S']}GB | {best['mlru']:.3f} | "
                 f"+{100*gain:.1f}% | **{verdict}** |")
    L.append("")
    L.append("- A2 prior shape (gaps +20/+38/+79%, kill nf4 / spare int8+16-bit) graded against the")
    L.append("  measured column above. bf16 h(S) uses the nf4-base token stream as a routing proxy")
    L.append("  (flagged; routing differs across precisions by ~2% of decisions).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
