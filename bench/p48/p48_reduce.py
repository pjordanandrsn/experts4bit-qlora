#!/usr/bin/env python3
"""bench/p48/p48_reduce.py -- the P48 reducer (P48-PREREG.md): the per-layer one-NF4-layer profile of Gemma-4-26B-A4B-it read
from `gemma4layer_kl.json`, plus the QNAP weight probe (`g4_expert_probe.jsonl`) when present. Rules only; a missing row is NOT_READ.

    python bench/p48/p48_reduce.py <run_dir> [--probe g4_expert_probe.jsonl] [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os

FULL_LOADER_NF4 = 1.0837        # P47 `loader_nf4` (all 30 layers), the sum's referent
ADDITIVITY_BAND = (0.5, 1.5)    # P1: sum of the 30 single-layer rows within this factor of FULL_LOADER_NF4
TOP5_SHARE_MIN = 0.50           # P2: the five largest single-layer rows carry >= 50 % of the sum (concentrated)
FIRST_HALF_SHARE_MIN = 0.80     # P2b: layers 0-14 carry >= 80 % of the sum (P47's lo/hi split, per layer)
MODEL_BOUND_MAX = 0.02          # P3: the SMALLEST single-layer row <= 0.02 nats (e4b's Gemma-4 modelling with 29 bf16 layers)
PROBE_RHO_MAX = 0.5             # P4: |Spearman rho| between per-layer KL and the probe's NF4 relative error < 0.5 (weights don't predict it)


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def scorer_admitted(rec: dict) -> tuple:
    used = (rec.get("scorer") or {}).get("used") or (rec.get("scorer") or {}).get("name")
    ctl = ((rec.get("reference_pass") or {}).get("self_consistency") or (rec.get("controls") or {}).get("reference_decode_vs_prefill") or {})
    kl = ctl.get("kl_mean")
    if used == "prefill":
        return True, used, kl
    if used == "decode":
        return bool(ctl.get("passes", ctl.get("passes_1e-2"))), used, kl
    return False, used, kl


def layer_rows(rec: dict) -> dict:
    out = {}
    for r in rec.get("rows", []):
        arm = r["arm"]
        if arm.startswith("L") and arm[1:].isdigit():
            out[int(arm[1:])] = r
    return out


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def probe_by_layer(path):
    """The QNAP weight probe: per layer, the mean over both projections of the median per-expert NF4 relative error,
    and the max amax/rms (outlier-ness)."""
    if not path or not os.path.exists(path):
        return None
    by = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d = by.setdefault(r["layer"], {"nf4_rel_err_med": [], "amax_over_rms_max": [], "rms": []})
            d["nf4_rel_err_med"].append(r["nf4_rel_err_med"])
            d["amax_over_rms_max"].append(r["expert_amax_over_rms_max"])
            d["rms"].append(r["rms"])
    return {k: {"nf4_rel_err_med": sum(v["nf4_rel_err_med"]) / len(v["nf4_rel_err_med"]),
                "amax_over_rms_max": max(v["amax_over_rms_max"]), "n_proj": len(v["rms"])} for k, v in by.items()}


def reduce(run_dir: str, probe_path: str | None = None, n_layers: int = 30) -> dict:
    rec = _load(os.path.join(run_dir, "gemma4layer_kl.json"))
    if rec is None:
        return {"verdict": "NOT_READ: no gemma4layer_kl.json"}
    ok, used, ckl = scorer_admitted(rec)
    rows = layer_rows(rec)
    out = {"scorer_control": {"admitted": ok, "scorer": used, "reference_self_kl": ckl},
           "not_measured": rec.get("not_measured", {}), "n_rows": len(rows), "n_layers": n_layers,
           "per_layer": {i: {"kl_mean": rows[i]["kl_mean"], "top1": rows[i].get("top1_agreement"),
                             "n_quantized": rows[i].get("engagement", {}).get("n_quantized")} for i in sorted(rows)}}
    if not ok:
        out["verdict"] = f"NOT_READ: the {used} scorer is refused for this family (reference self-KL {ckl})"
        return out
    complete = len(rows) == n_layers
    kls = {i: rows[i]["kl_mean"] for i in rows}
    total = sum(kls.values())
    out["sum_single_layer"] = total
    out["complete"] = complete
    # P1 additivity (needs every layer)
    if complete and total > 0:
        ratio = total / FULL_LOADER_NF4
        out["P1_additivity"] = {"sum": total, "full_loader_nf4": FULL_LOADER_NF4, "ratio": ratio, "band": ADDITIVITY_BAND,
                                "verdict": "HOLDS" if ADDITIVITY_BAND[0] <= ratio <= ADDITIVITY_BAND[1] else "REFUTED"}
    else:
        out["P1_additivity"] = {"verdict": "NOT_READ", "reason": f"{len(rows)}/{n_layers} rows"}
    # P2 concentration (needs every layer for the shares)
    if complete and total > 0:
        top = sorted(kls.items(), key=lambda kv: -kv[1])[:5]
        share5 = sum(v for _, v in top) / total
        first_half = sum(v for i, v in kls.items() if i < n_layers // 2) / total
        out["P2_concentration"] = {"top5": [{"layer": i, "kl_mean": v, "share": v / total} for i, v in top], "top5_share": share5,
                                   "first_half_share": first_half, "top5_min": TOP5_SHARE_MIN, "first_half_min": FIRST_HALF_SHARE_MIN,
                                   "verdict": "HOLDS" if share5 >= TOP5_SHARE_MIN else "REFUTED",
                                   "P2b_first_half": "HOLDS" if first_half >= FIRST_HALF_SHARE_MIN else "REFUTED"}
    else:
        out["P2_concentration"] = {"verdict": "NOT_READ", "reason": f"{len(rows)}/{n_layers} rows"}
    # P3 modelling bound (any rows: the min over measured rows is an upper bound only when complete; report both)
    if rows:
        i_min = min(kls, key=kls.get)
        out["P3_model_bound"] = {"min_layer": i_min, "min_kl": kls[i_min], "max": MODEL_BOUND_MAX,
                                 "verdict": ("HOLDS" if kls[i_min] <= MODEL_BOUND_MAX else "REFUTED") if complete else
                                            ("HOLDS (partial: bound from measured rows only)" if kls[i_min] <= MODEL_BOUND_MAX else "NOT_READ (partial rows, none under the bound yet)")}
    else:
        out["P3_model_bound"] = {"verdict": "NOT_READ"}
    # P4 the weight probe does not predict it
    pb = probe_by_layer(probe_path)
    if pb and rows:
        common = sorted(i for i in rows if i in pb)
        rho_err = spearman([kls[i] for i in common], [pb[i]["nf4_rel_err_med"] for i in common])
        rho_out = spearman([kls[i] for i in common], [pb[i]["amax_over_rms_max"] for i in common])
        out["P4_probe"] = {"n": len(common), "spearman_kl_vs_nf4_rel_err": rho_err, "spearman_kl_vs_amax_over_rms": rho_out, "max": PROBE_RHO_MAX,
                           "probe_nf4_rel_err_range": [min(pb[i]["nf4_rel_err_med"] for i in common), max(pb[i]["nf4_rel_err_med"] for i in common)],
                           "verdict": ("NOT_READ" if rho_err is None or not complete else ("HOLDS" if abs(rho_err) < PROBE_RHO_MAX else "REFUTED"))}
    else:
        out["P4_probe"] = {"verdict": "NOT_READ", "reason": "no probe file or no rows"}
    return out


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P48 — verdicts read from the rows", ""]
    if "per_layer" not in v:
        return "\n".join(L + [v.get("verdict", "NOT_READ")])
    sc = v["scorer_control"]
    L += [f"Scorer: `{sc['scorer']}` (admitted: {sc['admitted']}; reference self-KL {_f(sc['reference_self_kl'])}); rows {v['n_rows']}/{v['n_layers']}", ""]
    L += ["| layer (NF4 experts in THIS layer only) | KL nats/token | top-1 | quantised stacks |", "|---|---|---|---|"]
    for i, r in v["per_layer"].items():
        L.append(f"| {i} | {_f(r['kl_mean'])} | {_f(r['top1'])} | {r['n_quantized']} |")
    for k in v.get("not_measured", {}):
        L.append(f"| `{k}` | NOT MEASURED | — | — |")
    L += ["", f"Sum of single-layer rows: {_f(v.get('sum_single_layer'))} (P47 full loader_nf4 {FULL_LOADER_NF4})", ""]
    for key in ("P1_additivity", "P2_concentration", "P3_model_bound", "P4_probe"):
        p = v.get(key, {})
        extra = ""
        if key == "P2_concentration" and "top5" in p:
            extra = " — top-5: " + ", ".join(f"L{t['layer']} {t['kl_mean']:.3f} ({t['share']:.0%})" for t in p["top5"]) + f"; first half {p['first_half_share']:.0%} (P2b {p['P2b_first_half']})"
        if key == "P3_model_bound" and "min_kl" in p:
            extra = f" — min at L{p['min_layer']}: {p['min_kl']:.4f}"
        if key == "P4_probe" and p.get("n"):
            extra = f" — Spearman(KL, NF4 rel err) = {_f(p['spearman_kl_vs_nf4_rel_err'], 3)}, (KL, amax/rms) = {_f(p['spearman_kl_vs_amax_over_rms'], 3)}; probe rel-err range {p['probe_nf4_rel_err_range'][0]:.4f}–{p['probe_nf4_rel_err_range'][1]:.4f}"
        L.append(f"- **{key}: {p.get('verdict', 'NOT_READ')}**{extra}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    v = reduce(a.run_dir, a.probe)
    md = render_md(v)
    print(md)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump(v, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
