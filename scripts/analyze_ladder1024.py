"""Pre-committed analysis for the n=1024 ∅-ladder re-pin (docs/NULL_LADDER_1024_AMENDMENT.md §3-§4).

Implements exactly the preregistered inference plan — nothing more is claimed:

- PRIMARY: paired per-example nf4 − int8 (resident), two-sided, |t| >= 3.
- CO-PRIMARY: Wilcoxon signed-rank on the same contrast (normal approximation with tie/zero
  handling — zeros dropped, average ranks).
- SECONDARY (exploratory): all 15 resident pairs, Bonferroni-flagged at per-test |t| >= 3.0.
- Tail report: share of Σ|d_i| in the top 5% / 10% of |d_i|; 11-bucket d_i histogram
  (primary pair).
- Mechanism probe: per-example routing disagreement (1 − mean-layer Jaccard of routed sets)
  between mode pairs; Spearman corr(|d_i|, disagreement); mean Jaccard per pair for the
  committed Jaccard(int8,bf16) ≫ Jaccard(nf4,bf16) prediction.
- Integrity: every job must self-attest the committed eval-set hash; every job must be on one
  GPU architecture; D2/determinism bitwise checks re-run at n=1024.
- Prints the §4 branch table with each branch's measured outcome.

Controller-only. Usage:
    python scripts/analyze_ladder1024.py --jobs-root runs/jobs --out runs/results/null_ladder_1024.md
"""

import argparse
import glob
import json
import math
import os
import sys

MODES = ("fp4", "nf4", "int8", "fp8", "bf16", "fp16")
EVAL_SET_SHA256 = "3e836c1a01ab5cce90b7034477f174f5058f4cd4c1690dcc25b01741dc1a851f"
TRIO = ("int8", "bf16", "fp16")
CLUSTER_4BIT = ("fp4", "nf4")


def load(jobs_root):
    out = {}
    for d in sorted(glob.glob(os.path.join(jobs_root, "null1024_olmoe_*"))):
        rp, rows_p = os.path.join(d, "result.json"), os.path.join(d, "result_rows.jsonl")
        if not (os.path.exists(rp) and os.path.exists(rows_p)):
            continue
        res = json.load(open(rp))
        rows = [json.loads(l) for l in open(rows_p)]
        losses = {r["example_index"]: r["loss"] for r in rows if not r.get("is_nan")}
        routed = {}
        sp = os.path.join(d, "routed_sets.jsonl")
        if os.path.exists(sp):
            for l in open(sp):
                r = json.loads(l)
                routed[r["example_index"]] = {k: {e for e, _ in v} for k, v in r["routed"].items()}
        out[os.path.basename(d)] = {"result": res, "losses": losses, "routed": routed}
    return out


def key(mode, placement, rep=None):
    return f"null1024_olmoe_{mode}_{placement}" + (f"_rep{rep}" if rep else "")


def paired(a, b):
    ks = sorted(set(a) & set(b))
    ds = [a[k] - b[k] for k in ks]
    n = len(ds)
    mean = sum(ds) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in ds) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": mean / se if se > 0 else (0.0 if mean == 0 else float("inf")), "ds": ds, "ks": ks}


def wilcoxon_z(ds):
    """Signed-rank normal approximation: zeros dropped, average ranks, tie-corrected variance."""
    nz = [d for d in ds if d != 0]
    n = len(nz)
    if n < 10:
        return None, n
    ranked = sorted((abs(d), i) for i, d in enumerate(nz))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k2 in range(i, j + 1):
            ranks[ranked[k2][1]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    # tie correction
    from collections import Counter

    ties = Counter(abs(d) for d in nz)
    var -= sum(c ** 3 - c for c in ties.values() if c > 1) / 48
    z = (w_plus - mu) / math.sqrt(var) if var > 0 else float("inf")
    return z, n


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k2 in range(i, j + 1):
                r[order[k2]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def routing_disagreement(routed_a, routed_b, ks):
    """Per-example 1 − mean-layer Jaccard between the two modes' routed sets."""
    out = {}
    for k in ks:
        ra, rb = routed_a.get(k), routed_b.get(k)
        if not ra or not rb:
            continue
        js = []
        for layer in ra:
            if layer in rb:
                a, b = ra[layer], rb[layer]
                u = len(a | b)
                if u:
                    js.append(len(a & b) / u)
        if js:
            out[k] = 1.0 - sum(js) / len(js)
    return out


def fmt_p(p):
    return f"{p['mean']:+.5f} ± {p['se']:.5f} (sd {p['sd']:.4f}, n={p['n']}, |t|={abs(p['t']):.2f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = load(args.jobs_root)
    if not jobs:
        print("no null1024 jobs found", file=sys.stderr)
        return 1

    L = ["# n=1024 ∅-ladder re-pin — preregistered analysis", ""]

    # Integrity block
    hashes = {j: d["result"].get("eval_set_sha256") for j, d in jobs.items()}
    bad_hash = {j: h for j, h in hashes.items() if h != EVAL_SET_SHA256}
    gpus = sorted({d["result"].get("gpu_name") for d in jobs.values()})
    L.append("## Integrity")
    L.append("")
    L.append(f"- jobs: {len(jobs)} | GPU(s): {gpus}")
    L.append(f"- eval-set hash: {'ALL MATCH committed ' + EVAL_SET_SHA256[:16] + '…' if not bad_hash else 'MISMATCH: ' + json.dumps(bad_hash)}")
    if bad_hash or len(gpus) > 1:
        L.append("- **STOP: integrity violation — analysis below is void.**")
    res = {m: jobs.get(key(m, "resident")) for m in MODES}

    # D2 at n=1024 + determinism
    L.append("")
    L.append("## D2 + determinism at n=1024 (bitwise)")
    L.append("")
    for m in MODES:
        jr, jo = jobs.get(key(m, "resident")), jobs.get(key(m, "offload"))
        if jr and jo:
            ident = sum(1 for k in jr["losses"] if jr["losses"][k] == jo["losses"].get(k))
            L.append(f"- {m}: {ident}/{len(jr['losses'])} bitwise-identical resident vs offload")
    r1, r2 = jobs.get(key("int8", "resident")), jobs.get(key("int8", "resident", rep=2))
    if r1 and r2:
        ident = sum(1 for k in r1["losses"] if r1["losses"][k] == r2["losses"].get(k))
        L.append(f"- determinism repeat (int8-resident): {ident}/{len(r1['losses'])} bitwise")

    # Ladder
    L.append("")
    L.append("## Ladder (resident, mean ± SE)")
    L.append("")
    for m in MODES:
        if res[m]:
            vs = list(res[m]["losses"].values())
            n = len(vs)
            mu = sum(vs) / n
            sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / (n - 1))
            L.append(f"- {m}: {mu:.5f} ± {sd / math.sqrt(n):.5f}")

    # PRIMARY
    L.append("")
    L.append("## PRIMARY — nf4 − int8 (paired, two-sided, |t| ≥ 3)")
    L.append("")
    primary = None
    if res["nf4"] and res["int8"]:
        primary = paired(res["nf4"]["losses"], res["int8"]["losses"])
        L.append(f"- G_int8 = {fmt_p(primary)}")
        L.append(f"- **PRIMARY {'CLEARS' if abs(primary['t']) >= 3 else 'DOES NOT CLEAR'} |t| ≥ 3**")
        z, nnz = wilcoxon_z(primary["ds"])
        if z is not None:
            L.append(f"- CO-PRIMARY Wilcoxon signed-rank: z = {z:+.2f} (n_nonzero = {nnz}) — "
                     f"{'consistent' if (abs(z) >= 3) == (abs(primary['t']) >= 3) else 'DISAGREES with t'}")

    # SECONDARY
    L.append("")
    L.append("## SECONDARY (exploratory) — 15 resident pairs, Bonferroni |t| ≥ 3.0")
    L.append("")
    L.append("| pair | mean d ± SE | sd | |t| | Bonferroni |")
    L.append("|---|---|---|---|---|")
    pair_stats = {}
    for i, m1 in enumerate(MODES):
        for m2 in MODES[i + 1:]:
            if res[m1] and res[m2]:
                p = paired(res[m1]["losses"], res[m2]["losses"])
                pair_stats[(m1, m2)] = p
                L.append(f"| {m1} − {m2} | {p['mean']:+.5f} ± {p['se']:.5f} | {p['sd']:.4f} | "
                         f"{abs(p['t']):.2f} | {'**survives**' if abs(p['t']) >= 3.0 else '—'} |")

    # G_total + coverage rule
    g_total = pair_stats.get(("nf4", "bf16"))
    L.append("")
    if g_total:
        L.append(f"- G_total (nf4 − bf16) = {fmt_p(g_total)}")
        if primary and abs(primary["t"]) >= 3 and abs(g_total["t"]) >= 3:
            cov = 100 * primary["mean"] / g_total["mean"]
            L.append(f"- coverage = {cov:.0f}% (both clear 3σ — ratio licensed)")
        else:
            L.append("- coverage: NOT REPORTED (silly-ratio rule: both G_int8 and G_total must clear 3σ)")

    # Tail report (primary pair)
    if primary:
        ads = sorted((abs(d) for d in primary["ds"]), reverse=True)
        tot = sum(ads) or 1.0
        top5 = sum(ads[: max(1, len(ads) // 20)]) / tot
        top10 = sum(ads[: max(1, len(ads) // 10)]) / tot
        L.append("")
        L.append("## Tail report (primary pair)")
        L.append("")
        L.append(f"- top 5% of |d_i| carries {100 * top5:.0f}% of Σ|d_i|; top 10% carries {100 * top10:.0f}%")
        lo, hi = min(primary["ds"]), max(primary["ds"])
        width = (hi - lo) / 11 or 1.0
        buckets = [0] * 11
        for d in primary["ds"]:
            buckets[min(10, int((d - lo) / width))] += 1
        L.append(f"- d_i histogram [{lo:+.4f} … {hi:+.4f}], 11 buckets: {buckets}")

    # Mechanism probe
    L.append("")
    L.append("## Mechanism probe — routing disagreement")
    L.append("")
    probe_pairs = [("int8", "bf16"), ("nf4", "bf16"), ("nf4", "int8"), ("fp4", "bf16"),
                   ("int8", "fp16"), ("bf16", "fp16"), ("fp4", "nf4")]
    corr_cross_pos, corr_trio_small = [], []
    jacc = {}
    for m1, m2 in probe_pairs:
        if not (res.get(m1) and res.get(m2)):
            continue
        p = pair_stats.get((m1, m2)) or pair_stats.get((m2, m1))
        if p is None:
            p = paired(res[m1]["losses"], res[m2]["losses"])
        dis = routing_disagreement(res[m1]["routed"], res[m2]["routed"], p["ks"])
        ks = [k for k in p["ks"] if k in dis]
        if len(ks) < 10:
            L.append(f"- {m1}/{m2}: routed sets unavailable")
            continue
        ad = [abs(res[m1]["losses"][k] - res[m2]["losses"][k]) for k in ks]
        dd = [dis[k] for k in ks]
        rho = spearman(ad, dd)
        mean_j = 1.0 - sum(dd) / len(dd)
        jacc[(m1, m2)] = mean_j
        cross = (m1 in CLUSTER_4BIT) != (m2 in CLUSTER_4BIT)
        both_trio = m1 in TRIO and m2 in TRIO
        L.append(f"- {m1}/{m2}: mean Jaccard {mean_j:.4f} | Spearman corr(|d_i|, disagreement) = {rho:+.3f}"
                 f"{' [cross-cluster]' if cross else ''}{' [within trio]' if both_trio else ''}")
        if cross:
            corr_cross_pos.append(rho)
        if both_trio:
            corr_trio_small.append(rho)
    if ("int8", "bf16") in jacc and ("nf4", "bf16") in jacc:
        L.append(f"- committed prediction Jaccard(int8,bf16) ≫ Jaccard(nf4,bf16): "
                 f"{jacc[('int8','bf16')]:.4f} vs {jacc[('nf4','bf16')]:.4f} — "
                 f"{'HOLDS' if jacc[('int8','bf16')] > jacc[('nf4','bf16')] else 'FAILS'}")

    # Branch table
    L.append("")
    L.append("## §4 branch table — outcomes")
    L.append("")
    if primary:
        b1 = abs(primary["t"]) >= 3
        L.append(f"- G_int8 ≥ 3σ: **{'YES — precision gap real; program resumes' if b1 else 'NO — bounded null: |G_int8| < ' + format(3 * primary['se'], '.4f') + ' (3σ); calibrated flat axis ships'}**")
    trio_pairs = [pair_stats.get(p) for p in (("int8", "bf16"), ("int8", "fp16"), ("bf16", "fp16"))]
    if all(trio_pairs):
        trio_flat = all(abs(p["mean"]) <= 3 * p["se"] for p in trio_pairs)
        trio_res = max(3 * p["se"] for p in trio_pairs)
        L.append(f"- trio {{int8,bf16,fp16}} within ±3σ (resolution ±{trio_res:.4f}): "
                 f"**{'YES — flat above int8 ships' if trio_flat else 'NO — structure inside the trio'}**")
    fp8_reps = [pair_stats.get(("fp4", "fp8")), pair_stats.get(("nf4", "fp8"))]
    if all(fp8_reps):
        fp8_clears = all(p["mean"] > 0 and abs(p["t"]) >= 3 for p in fp8_reps)
        L.append(f"- fp8 advantage over {{nf4, fp4}} at 3σ: **{'YES — promoted to finding' if fp8_clears else 'NO — stays scramble'}**")
    if primary:
        L.append(f"- top-10% carries >50% of Σ|d_i|: **{'YES — subpopulation phenomenon' if top10 > 0.5 else 'NO'}** ({100 * top10:.0f}%)")
    if corr_cross_pos:
        pos = sum(1 for r in corr_cross_pos if r > 0.1)
        L.append(f"- cross-cluster corr(|d|, routing disagreement) > 0: {pos}/{len(corr_cross_pos)} pairs with ρ > 0.1; "
                 f"within-trio ρ: {['%+.3f' % r for r in corr_trio_small]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
