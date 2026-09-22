#!/usr/bin/env python3
"""P61 read: p61_rows.json -> the tables + the pre-registered verdicts (P61-PREREG.md, applied literally).

  fit  per projection over the grid, t(R, D) = a + b*R + c*D per call (least squares), on graph time and on the
       GEMV kernel's own time (profiler);
  P0   the shared-store served arm on P60's recorded routing within +-5 % of P60's served 6.479 ms/step (gate);
  P1   b_graph x the recorded rows removed per step predicts the per-layer-store served - dedup gap within +-25 %
       (refuted beyond +-50 %);
  P2   b1, the GEMV kernel's cost per extra row with its one expert L2-resident (the D = 1 column), over the L2
       re-read floor (expert bytes / L2-resident copy bandwidth): <= 1.3 on both projections -> L2-BOUND,
       >= 2.0 on both -> NOT-BYTES, else MIXED;
  P3   c_gemv over the DRAM floor per distinct expert (informational);
  P4   per-layer-store served minus shared-store served (informational: what P60's shared store gifted itself).
Pure python (no numpy), so the unit test runs anywhere. Prints markdown; quotes nothing the rows do not carry.
"""
from __future__ import annotations

import json
import sys

P60_SERVED = 6.479                     # ms/step, P60's served arm on its (shared) store -- the P0 anchor
P0_BAND, P1_HOLD, P1_REFUTE = 0.05, 0.25, 0.50
P2_L2, P2_NOT = 1.3, 2.0


def _solve3(A, y):
    """Gaussian elimination with partial pivoting for a 3x3 system."""
    M = [row[:] + [v] for row, v in zip(A, y)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(3):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * z for x, z in zip(M[r], M[i])]
    return [M[i][3] / M[i][i] for i in range(3)]


def fit_abc(cells, key):
    """Least squares t = a + b R + c D over cells; returns (a, b, c, r2, max |relative residual|)."""
    X = [(1.0, float(c["R"]), float(c["D"])) for c in cells]
    y = [float(c[key]) for c in cells]
    XtX = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
    Xty = [sum(x[i] * v for x, v in zip(X, y)) for i in range(3)]
    a, b, c = _solve3(XtX, Xty)
    pred = [a + b * x[1] + c * x[2] for x in X]
    mean = sum(y) / len(y)
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred))
    ss_tot = sum((v - mean) ** 2 for v in y) or 1e-30
    rel = max(abs(v - p) / v for v, p in zip(y, pred) if v)
    return a, b, c, 1 - ss_res / ss_tot, rel


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def verdicts(d):
    bw, bpe, grid, rec = d["bandwidth"], d["bytes_per_expert"], d["grid"], d["recorded"]
    out = {"proj": {}}
    for proj in bpe:
        cells = [c for c in grid if c["proj"] == proj]
        g = fit_abc(cells, "graph_us_per_call")
        k = fit_abc(cells, "gemv_us_per_call")
        d1 = sorted((c for c in cells if c["D"] == 1), key=lambda c: c["R"])
        b1 = slope([c["R"] for c in d1], [c["gemv_us_per_call"] for c in d1])
        l2_floor = bpe[proj] / (bw["l2_copy_gbs"] * 1e9) * 1e6                 # us to re-read one slice from L2
        dram_floor = bpe[proj] / (bw["dram_copy_gbs"] * 1e9) * 1e6             # us to stream one slice from DRAM
        out["proj"][proj] = {"fit_graph": g, "fit_gemv": k, "b1": b1, "l2_floor_us": l2_floor,
                             "p2_ratio": b1 / l2_floor, "dram_floor_us": dram_floor, "p3_ratio": k[2] / dram_floor}
    shared, perl = rec["shared"], rec["per_layer"]
    out["p0_rel"] = shared["served"]["step_ms_median"] / P60_SERVED - 1
    out["p0"] = abs(out["p0_rel"]) <= P0_BAND
    removed = d["distinct_per_call"]["rows_removed_per_step_mean"]
    out["p1_pred_ms"] = sum(p["fit_graph"][1] for p in out["proj"].values()) * removed / 1000.0
    out["p1_meas_ms"] = perl["served"]["step_ms_median"] - perl["dedup"]["step_ms_median"]
    out["p1_rel"] = out["p1_pred_ms"] / out["p1_meas_ms"] - 1 if out["p1_meas_ms"] else float("inf")
    out["p1"] = "HELD" if abs(out["p1_rel"]) <= P1_HOLD else ("REFUTED" if abs(out["p1_rel"]) > P1_REFUTE else "NOT HELD")
    ratios = [p["p2_ratio"] for p in out["proj"].values()]
    out["p2"] = "L2-BOUND" if all(r <= P2_L2 for r in ratios) else ("NOT-BYTES" if all(r >= P2_NOT for r in ratios) else "MIXED")
    out["p4_ms"] = perl["served"]["step_ms_median"] - shared["served"]["step_ms_median"]
    if not out["p0"]:
        out["decision"] = "¬P0 → the instrument does not reproduce P60; nothing else is read."
    elif out["p1"] == "REFUTED":
        out["decision"] = ("P0 ∧ ¬P1 → the grid's linear model does not transfer to the recorded routing (skew or an "
                           "R x D interaction); P2 is informational only and no lever lane follows from this read.")
    else:
        out["decision"] = {
            "L2-BOUND": "P0 ∧ P1 ∧ L2-BOUND → the per-row cost is the weight re-read from L2: sharing loads is still the "
                        "lever, and K18 lost on its own overhead. Next: a low-overhead load-sharing design, "
                        "pre-registered with b as its bar.",
            "NOT-BYTES": "P0 ∧ P1 ∧ NOT-BYTES → the per-row cost is not the weight read: sharing loads cannot buy it. "
                         "Next: cut the per-row arithmetic (a tensor-core int8 path for experts with >= 2 rows), "
                         "pre-registered with b as its bar.",
            "MIXED": "P0 ∧ P1 ∧ MIXED → both contribute; the next lane is chosen by the larger measured share and "
                     "pre-registered separately.",
        }[out["p2"]]
        if out["p1"] == "NOT HELD":
            out["decision"] += " (P1 between bands: the model transfers only roughly; its b is quoted with that caveat.)"
    return out


def render(d) -> str:
    v = verdicts(d)
    bw = d["bandwidth"]
    L = [f"device {d['device']} ({d['sm_count']} SMs), torch {d['torch']}, L2 {d['l2_bytes'] >> 20} MiB; "
         f"{d['layers']} layers x {d['steps']} recorded steps; eids sha256 {d['eids_sha256'][:12]}...; "
         f"copy bandwidth DRAM {bw['dram_copy_gbs']:.0f} GB/s, L2-resident {bw['l2_copy_gbs']:.0f} GB/s "
         f"({bw['l2_buf_bytes'] >> 20} MiB x2); plans (sk by R) {d['plans']}", ""]
    L += ["| proj | R | D | rows/expert | graph us/call | gemv us/call | reduce us/call |", "|---|---|---|---|---|---|---|"]
    for c in d["grid"]:
        L.append(f"| {c['proj']} | {c['R']} | {c['D']} | {c['R'] // c['D']} | {c['graph_us_per_call']:.2f} | "
                 f"{c['gemv_us_per_call']:.2f} | {c['reduce_us_per_call']:.2f} |")
    L += ["", "| proj | fit on | a us | b us/row | c us/expert | R^2 | max rel resid |", "|---|---|---|---|---|---|---|"]
    for proj, p in v["proj"].items():
        for lab, f in (("graph", p["fit_graph"]), ("gemv kernel", p["fit_gemv"])):
            L.append(f"| {proj} | {lab} | {f[0]:.2f} | {f[1]:.4f} | {f[2]:.4f} | {f[3]:.4f} | {f[4] * 100:.1f} % |")
    L += ["", "| proj | b1 (gemv, D=1) us/row | L2 re-read floor us | b1 / floor | c_gemv us/expert | DRAM floor us | c / floor |",
          "|---|---|---|---|---|---|---|"]
    for proj, p in v["proj"].items():
        L.append(f"| {proj} | {p['b1']:.4f} | {p['l2_floor_us']:.4f} | {p['p2_ratio']:.2f} | {p['fit_gemv'][2]:.4f} | "
                 f"{p['dram_floor_us']:.4f} | {p['p3_ratio']:.2f} |")
    rec = d["recorded"]
    L += ["", "| recorded routing | served ms/step | dedup ms/step | served - dedup | served gemv kernel ms/step |",
          "|---|---|---|---|---|"]
    for store in ("shared", "per_layer"):
        r = rec[store]
        k = r["served"]["gemv_kernel_ms_median"]
        ks = "—" if k is None else f"{k:.3f}"
        L.append(f"| {store} store | {r['served']['step_ms_median']:.3f} | {r['dedup']['step_ms_median']:.3f} | "
                 f"{r['served']['step_ms_median'] - r['dedup']['step_ms_median']:+.3f} | {ks} |")
    dp = d["distinct_per_call"]
    L += ["", f"Recorded distinct experts per call: mean {dp['mean']:.2f} ({dp['min']}–{dp['max']}); rows removed per "
              f"step by dedup: {dp['rows_removed_per_step_mean']:.1f}.", ""]
    L.append(f"- **P0 (instrument: shared-store served within ±5 % of P60's {P60_SERVED}):** "
             f"{'HOLDS' if v['p0'] else 'FAILS'} — {rec['shared']['served']['step_ms_median']:.3f} ms/step ({v['p0_rel'] * 100:+.1f} %)")
    L.append(f"- **P1 (b_graph × rows removed predicts the per-layer served − dedup gap within ±25 %):** {v['p1']} — "
             f"predicted {v['p1_pred_ms']:.3f} vs measured {v['p1_meas_ms']:.3f} ms/step ({v['p1_rel'] * 100:+.1f} %)")
    L.append(f"- **P2 (b1 / L2 re-read floor; ≤ {P2_L2} both → L2-BOUND, ≥ {P2_NOT} both → NOT-BYTES):** {v['p2']} — "
             + ", ".join(f"{proj} {p['p2_ratio']:.2f}" for proj, p in v["proj"].items()))
    L.append("- **P3 (informational, c_gemv / DRAM floor per expert):** "
             + ", ".join(f"{proj} {p['p3_ratio']:.2f}" for proj, p in v["proj"].items()))
    L.append(f"- **P4 (informational, per-layer served − shared served):** {v['p4_ms']:+.3f} ms/step")
    L.append("")
    L.append(f"**Decision rule:** {v['decision']}")
    return "\n".join(L)


if __name__ == "__main__":
    print(render(json.load(open(sys.argv[1]))))
