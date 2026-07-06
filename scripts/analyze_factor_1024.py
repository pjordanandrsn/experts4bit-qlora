"""Exploratory factor/telemetry analysis at n=1024 — P-A1..P-A6 (lanes addendum 1).

SEPARATE from scripts/analyze_ladder1024.py by design: that script is the preregistered
CONFIRMATORY analysis (amendment §3, untouched); this one is the exploratory lanes layer,
committed before any n=1024 join is computed.

Implements, per docs/SPECULATIVE_LANES_ADDENDUM_1.md §3:
- P-A1: PC1 of the deviation matrix (vs bf16, resident) explains >= 50% of variance.
- P-A2: Spearman corr(PC1 score_i, total flip count_i across {nf4, fp4, fp8}) > 0.4.
- P-A3: rho(int8, fp16) >= 0.5 replicates AND within-trio |d_i| uncorrelated with flips.
- P-A4: fp8 mean per-example flip count within +/-25% of nf4's (sign condition reported
  alongside the amendment's primary, not judged here).
- P-A5: pilot top-decile |nf4-bf16| examples exceed the n=1024 top-decile CUTOFF at >= 3x
  the 10% base rate. (Interpretation note: the pilot and confirmation sets are disjoint, so
  "predicts membership" is operationalized as cross-set threshold transfer — the pilot's
  fragile examples' |d| values landing above the confirmation set's top-decile cutoff.
  Stated here because the addendum text admits more than one reading.)
- P-A6: resolved by Z3 (shared compute path) — reported as context, nothing computed.

Flip count_i(p) := sum over layers of the symmetric difference |routed_p XOR routed_bf16|
for example i. Pearson AND Spearman are reported everywhere (Z1's shared-outlier lesson).

Usage:
    python scripts/analyze_factor_1024.py --jobs-root runs/jobs \\
        --pilot-probe runs/results/postaudit/probe_set_candidates_n64.json \\
        --out runs/results/factor_structure_1024.md
"""

import argparse
import json
import math
import os
import sys

MODES = ("fp4", "nf4", "int8", "fp8", "fp16")
SCATTERED = ("fp4", "nf4", "fp8")
TRIO = ("int8", "fp16")


def load_job(jobs_root, mode, placement="resident"):
    d = os.path.join(jobs_root, f"null1024_olmoe_{mode}_{placement}")
    losses, routed = {}, {}
    if not os.path.exists(os.path.join(d, "result.json")):
        return None
    for line in open(os.path.join(d, "result_rows.jsonl")):
        r = json.loads(line)
        if not r.get("is_nan"):
            losses[r["example_index"]] = r["loss"]
    sp = os.path.join(d, "routed_sets.jsonl")
    if os.path.exists(sp):
        for line in open(sp):
            r = json.loads(line)
            routed[r["example_index"]] = {k: {e for e, _ in v} for k, v in r["routed"].items()}
    return {"losses": losses, "routed": routed}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den > 0 else 0.0


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def eig_sym(M, iters=500):
    """Top eigenpairs of a small symmetric matrix by deflated power iteration (no numpy)."""
    n = len(M)
    A = [row[:] for row in M]
    pairs = []
    for _ in range(n):
        v = [1.0 / math.sqrt(n)] * n
        lam = 0.0
        for _ in range(iters):
            w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            nrm = math.sqrt(sum(x * x for x in w))
            if nrm == 0:
                break
            v = [x / nrm for x in w]
            lam = sum(v[i] * sum(A[i][j] * v[j] for j in range(n)) for i in range(n))
        pairs.append((lam, v))
        for i in range(n):
            for j in range(n):
                A[i][j] -= lam * v[i] * v[j]
    return pairs


def flip_count(routed_a, routed_b, k):
    ra, rb = routed_a.get(k), routed_b.get(k)
    if not ra or not rb:
        return None
    c = 0
    for layer in ra:
        if layer in rb:
            c += len(ra[layer] ^ rb[layer])
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--pilot-probe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bf16 = load_job(args.jobs_root, "bf16")
    jobs = {m: load_job(args.jobs_root, m) for m in MODES}
    if bf16 is None or any(v is None for v in jobs.values()):
        print("missing jobs; run after the ladder drains", file=sys.stderr)
        return 1
    ks = sorted(set(bf16["losses"]) & set.intersection(*(set(j["losses"]) for j in jobs.values())))
    D = {m: [jobs[m]["losses"][k] - bf16["losses"][k] for k in ks] for m in MODES}
    L = ["# n=1024 factor/telemetry analysis — P-A1..P-A6 (exploratory, lanes addendum 1)", ""]
    L.append(f"- n = {len(ks)} examples; modes vs bf16, resident")

    # Correlation matrices
    L.append("")
    L.append("## Deviation correlations (Pearson / Spearman)")
    L.append("")
    L.append("| pair | Pearson | Spearman |")
    L.append("|---|---|---|")
    P = [[0.0] * len(MODES) for _ in MODES]
    for i, m1 in enumerate(MODES):
        for j, m2 in enumerate(MODES):
            P[i][j] = pearson(D[m1], D[m2]) if i != j else 1.0
    for i, m1 in enumerate(MODES):
        for m2 in MODES[i + 1:]:
            L.append(f"| {m1}–{m2} | {pearson(D[m1], D[m2]):+.3f} | {spearman(D[m1], D[m2]):+.3f} |")

    pairs = eig_sym(P)
    ev = [p[0] for p in pairs]
    pc1_load = pairs[0][1]
    pc1_share = ev[0] / len(MODES)
    L.append("")
    L.append(f"- eigenvalues: {[round(e, 3) for e in ev]} — PC1 share {100 * pc1_share:.0f}%")
    L.append(f"- PC1 loadings: { {m: round(ld, 3) for m, ld in zip(MODES, pc1_load)} }")
    L.append(f"- **P-A1 (PC1 ≥ 50%): {'HOLDS' if pc1_share >= 0.5 else 'FAILS'}**")

    # PC1 scores (standardized deviations)
    Z = {}
    for m in MODES:
        mu = sum(D[m]) / len(ks)
        sd = math.sqrt(sum((d - mu) ** 2 for d in D[m]) / (len(ks) - 1))
        Z[m] = [(d - mu) / sd for d in D[m]]
    pc1_scores = [sum(pc1_load[i] * Z[m][t] for i, m in enumerate(MODES)) for t in range(len(ks))]

    # Flip counts
    have_routing = all(jobs[m]["routed"] for m in MODES) and bf16["routed"]
    if have_routing:
        fc = {m: [flip_count(jobs[m]["routed"], bf16["routed"], k) or 0 for k in ks] for m in MODES}
        total_scattered = [sum(fc[m][t] for m in SCATTERED) for t in range(len(ks))]
        rho_a2 = spearman(pc1_scores, total_scattered)
        L.append("")
        L.append("## Telemetry joins")
        L.append("")
        L.append("- mean flip count vs bf16: " + ", ".join(f"{m} {sum(fc[m])/len(ks):.1f}" for m in MODES))
        L.append(f"- **P-A2 (Spearman(PC1 score, scattered flips) > 0.4): ρ = {rho_a2:+.3f} — "
                 f"{'HOLDS' if rho_a2 > 0.4 else 'FAILS'}** (sign convention: |ρ| judged, loadings sign-free: "
                 f"|ρ| = {abs(rho_a2):.3f})")
        rho_i16 = pearson(D["int8"], D["fp16"])
        trio_ok = True
        trio_lines = []
        for m in TRIO:
            r = spearman([abs(d) for d in D[m]], fc[m])
            trio_lines.append(f"{m}: ρ(|d|, flips) = {r:+.3f}")
            if abs(r) > 0.2:
                trio_ok = False
        L.append(f"- **P-A3: ρ(int8, fp16) = {rho_i16:+.3f} ({'≥0.5 ✓' if rho_i16 >= 0.5 else '<0.5 ✗'}); "
                 f"within-trio {' | '.join(trio_lines)} — "
                 f"{'HOLDS' if rho_i16 >= 0.5 and trio_ok else 'FAILS/PARTIAL'}**")
        m_nf4, m_fp8 = sum(fc["nf4"]) / len(ks), sum(fc["fp8"]) / len(ks)
        within = abs(m_fp8 - m_nf4) <= 0.25 * m_nf4 if m_nf4 > 0 else False
        L.append(f"- **P-A4 (fp8 flips within ±25% of nf4): nf4 {m_nf4:.1f}, fp8 {m_fp8:.1f} — "
                 f"{'HOLDS' if within else 'FAILS'}** (sign condition judged with the amendment's primary)")
    else:
        L.append("- routing telemetry incomplete; P-A2/P-A3/P-A4 not computable")

    # P-A5: cross-set threshold transfer
    pilot = json.load(open(args.pilot_probe))["candidates"]
    ad = sorted((abs(d) for d in D["nf4"]), reverse=True)
    cutoff = ad[max(0, len(ad) // 10 - 1)]
    top_pilot = [c for c in pilot if c["rank"] <= max(1, len(pilot) // 10)]
    n_above = sum(1 for c in top_pilot if c["abs_dev_nf4_vs_bf16"] >= cutoff)
    frac = n_above / len(top_pilot) if top_pilot else 0.0
    L.append("")
    L.append("## P-A5 — probe-set cross-validation (threshold-transfer operationalization)")
    L.append("")
    L.append(f"- n=1024 top-decile cutoff on |nf4−bf16|: {cutoff:.4f}")
    L.append(f"- pilot top-decile examples above that cutoff: {n_above}/{len(top_pilot)} = {100*frac:.0f}% "
             f"(base rate 10%) — **{'HOLDS (≥3×)' if frac >= 0.3 else 'FAILS'}**")
    L.append("- interpretation note: disjoint sets, so 'membership prediction' is operationalized as "
             "cross-set cutoff transfer; stated in the script header before the join was computed.")

    L.append("")
    L.append("## P-A6 — resolved by Z3 (no computation)")
    L.append("")
    L.append("- Compute paths are SHARED by construction (OFFLOAD_MEMORY_FACTS.md Z3): ρ(int8, fp16) "
             "cannot be a kernel-path artifact; it stands as example-level smooth-sensitivity if it "
             "replicates (see P-A3).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
