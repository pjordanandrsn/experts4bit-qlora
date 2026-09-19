#!/usr/bin/env python3
"""bench/p44/p44_reduce.py -- read the P44 verdicts from the rows, with the registered rules and nothing else.

    python bench/p44/p44_reduce.py <run-dir> [--md RESULTS-p44.md] [--json verdicts.json]

`<run-dir>` holds whatever the two lanes fetched: `olmoe_ppl_<arm>_<src>.json` (step_decomp's K8 receipts),
`census_<family>.json` (expert_residuals.py), `<family>_kl.json` (kl_serve.py). A missing file is a missing row,
never a zero and never a pass.

Rules (P44-PREREG.md, verbatim in code):
  * K8 -- `experts4bit_qlora.k8_gate.verdict`, budget 0.05 ppl, one pair per scoring text: `int4all` two-sided
    (uncalibrated), `calibexp_all` one-sided with the calibrated clause and `calibration_domain="wikitext"` (the
    pack calibrates on wikitext-train; c4val1 is the outside text).
  * KL reading rule -- a candidate LICENSES iff on EVERY stratum KL(cand) <= 1.10 x KL(nf4 control) AND pooled
    KL(cand) - KL(nf4) <= 0.005 nats/token. The licence label is `licensed_by: kl-vs-bf16`.
  * P4 (Gemma-4 `r1epi` control) -- |KL(r1epi) - KL(nf4)| < 1e-4 nats on every stratum.
  * P3 (census) -- the recipe's error per expert (GPTQ where the recipe would pack GPTQ, RTN otherwise; RTN
    throughout for a family whose timed arm is RTN), summed over roles; the fraction of experts that carries 50 %
    of the summed error; P3 holds iff that fraction <= 0.10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experts4bit_qlora.k8_gate import BUDGET, Arm, verdict  # noqa: E402

K8_ARMS = {"int4all": {"calibrated": False, "calibration_domain": None},
           "calibexp_all": {"calibrated": True, "calibration_domain": "wikitext"}}
K8_TEXTS = ("wikitext", "c4val1")
KL_BAND = 1.10
KL_POOLED_DELTA = 0.005
P4_EPS = 1e-4
P3_TAIL_FRACTION = 0.10
CENSUS_METHOD = {"granite": "recipe", "mixtral": "rtn"}     # the family's timed arm: calibexp_r12epi / all (RTN)


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ------------------------------------------------------------------------------------------- K8 (P44-a)
def k8_arm(rec: dict, src: str) -> Arm:
    return Arm(ppl=float(rec["ppl"]), text_sha=rec["text_sha"], steps=int(rec["steps"]),
               ppl_source=rec.get("ppl_source", src))


def k8_verdicts(run_dir: str, family: str = "olmoe") -> dict:
    out = {}
    for arm, rule in K8_ARMS.items():
        pairs, missing = [], []
        for src in K8_TEXTS:
            base = _load(os.path.join(run_dir, f"{family}_ppl_nf4_{src}.json"))
            cand = _load(os.path.join(run_dir, f"{family}_ppl_{arm}_{src}.json"))
            if base is None or cand is None or "ppl" not in (base or {}) or "ppl" not in (cand or {}):
                missing.append(src)
                continue
            pairs.append((k8_arm(base, src), k8_arm(cand, src)))
        if missing or len(pairs) < len(K8_TEXTS):
            out[arm] = {"verdict": "NOT_READ", "reason": f"missing K8 rows for {missing or 'a text'} -- the two-text "
                                                          "rule needs both texts", "pairs": len(pairs)}
            continue
        passed, lines = verdict(pairs, calibrated=rule["calibrated"], budget=BUDGET,
                                calibration_domain=rule["calibration_domain"])
        out[arm] = {"verdict": "PASS" if passed else "FAIL", "lines": lines, "rule": rule, "budget": BUDGET,
                    "deltas": {c.ppl_source: round(c.ppl - b.ppl, 5) for b, c in pairs},
                    "steps": pairs[0][0].steps, "licenses": passed}
    return out


# ------------------------------------------------------------------------------------------- census (P3)
def expert_errors(census: dict, method: str) -> dict:
    """(layer, expert) -> summed activation-weighted squared error over roles under `method`
    ('recipe' = gptq where recipe_method says gptq and a gptq row exists, rtn otherwise; 'rtn' = rtn throughout).
    Experts without a Hessian (never routed) have no activation error and are EXCLUDED, counted separately."""
    errs, skipped = {}, 0
    for r in census["rows"]:
        pick = r["rtn"]
        if method == "recipe" and r.get("recipe_method") == "gptq" and r.get("gptq"):
            pick = r["gptq"]
        if pick is None or pick.get("sq_err_act") is None:
            skipped += 1
            continue
        key = (r["layer"], r["expert"])
        errs[key] = errs.get(key, 0.0) + float(pick["sq_err_act"])
    return {"errors": errs, "roles_without_hessian": skipped}


def tail_fraction(errs: dict, share: float = 0.5) -> dict:
    """The smallest fraction of experts whose summed error reaches `share` of the total."""
    if not errs:
        return {"n_experts": 0, "fraction_for_share": None}
    vals = sorted(errs.values(), reverse=True)
    total = sum(vals)
    if total <= 0:
        return {"n_experts": len(vals), "fraction_for_share": None}
    acc, k = 0.0, 0
    for v in vals:
        acc += v
        k += 1
        if acc >= share * total:
            break
    top10 = int(max(1, round(0.10 * len(vals))))
    return {"n_experts": len(vals), "experts_for_share": k, "fraction_for_share": k / len(vals),
            "share": share, "top10pct_share": sum(vals[:top10]) / total, "max_share": vals[0] / total}


def census_verdict(run_dir: str, family: str) -> dict:
    c = _load(os.path.join(run_dir, f"census_{family}.json"))
    if c is None:
        return {"verdict": "NOT_READ", "reason": "no census file"}
    method = CENSUS_METHOD[family]
    ee = expert_errors(c, method)
    tf = tail_fraction(ee["errors"])
    if tf["fraction_for_share"] is None:
        return {"verdict": "NOT_READ", "reason": "no activation-weighted rows", "method": method, **tf}
    holds = tf["fraction_for_share"] <= P3_TAIL_FRACTION
    return {"verdict": "P3_HOLDS" if holds else "P3_REFUTED", "method": method, "threshold": P3_TAIL_FRACTION,
            "rows": len(c["rows"]), "layers": len(c.get("layers_censused", [])), **tf, **{k: v for k, v in ee.items() if k != "errors"}}


# ------------------------------------------------------------------------------------------- KL (P44-b)
def kl_rows(rec: dict) -> dict:
    return {r["arm"]: r for r in rec.get("rows", [])}


def reading_rule(control: dict, cand: dict) -> dict:
    strata = sorted(set(control["per_stratum"]) | set(cand["per_stratum"]))
    per = {}
    ok_band = True
    for s in strata:
        a = control["per_stratum"].get(s, {}).get("kl_mean")
        b = cand["per_stratum"].get(s, {}).get("kl_mean")
        if a is None or b is None:
            per[s] = {"control": a, "candidate": b, "within_band": False, "reason": "stratum missing on one side"}
            ok_band = False
            continue
        ratio = (b / a) if a > 0 else (float("inf") if b > 0 else 1.0)
        within = b <= KL_BAND * a or (a == 0 and b == 0)
        per[s] = {"control": a, "candidate": b, "ratio": ratio, "within_band": within}
        ok_band = ok_band and within
    pooled_delta = cand["kl_mean"] - control["kl_mean"]
    ok_pooled = pooled_delta <= KL_POOLED_DELTA
    return {"licenses": bool(ok_band and ok_pooled), "per_stratum": per, "pooled_control": control["kl_mean"],
            "pooled_candidate": cand["kl_mean"], "pooled_delta": pooled_delta, "band": KL_BAND,
            "pooled_delta_max": KL_POOLED_DELTA, "licensed_by": "kl-vs-bf16" if (ok_band and ok_pooled) else None}


def p4_control(control: dict, r1epi: dict) -> dict:
    per = {}
    ok = True
    for s in sorted(set(control["per_stratum"]) | set(r1epi["per_stratum"])):
        a = control["per_stratum"].get(s, {}).get("kl_mean")
        b = r1epi["per_stratum"].get(s, {}).get("kl_mean")
        d = None if a is None or b is None else abs(b - a)
        per[s] = d
        ok = ok and d is not None and d < P4_EPS
    return {"holds": ok, "eps": P4_EPS, "abs_delta_per_stratum": per}


def kl_verdicts(run_dir: str) -> dict:
    out = {}
    g = _load(os.path.join(run_dir, "gemma4_kl.json"))
    if g is not None:
        rows = kl_rows(g)
        fam = {"not_measured": g.get("not_measured", {}), "scorer": g.get("scorer", {}).get("name")}
        if "nf4" in rows:
            for cand in ("int4_r1epi", "calattn_r1epi"):
                fam[cand] = reading_rule(rows["nf4"], rows[cand]) if cand in rows else {"verdict": "NOT_READ"}
            fam["P4_r1epi_equals_nf4"] = p4_control(rows["nf4"], rows["r1epi"]) if "r1epi" in rows else {"verdict": "NOT_READ"}
        else:
            fam["verdict"] = "NOT_READ: no nf4 control row"
        out["gemma4"] = fam
    gp = _load(os.path.join(run_dir, "gptoss_kl.json"))
    if gp is not None:
        rows = kl_rows(gp)
        fam = {"not_measured": gp.get("not_measured", {}), "scorer": gp.get("scorer", {}).get("name")}
        if "nf4_r12" in rows and "store_r12" in rows:
            fam["store_r12"] = reading_rule(rows["nf4_r12"], rows["store_r12"])
            fam["P6_store_at_floor"] = {"kl_mean": rows["store_r12"]["kl_mean"], "floor": 1e-3,
                                        "holds": rows["store_r12"]["kl_mean"] < 1e-3}
        else:
            fam["verdict"] = "NOT_READ: control or candidate row missing"
        out["gptoss"] = fam
    return out


# ------------------------------------------------------------------------------------------- render
def _fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P44 — verdicts read from the rows", ""]
    L += ["## P44-a — OLMoE two-text K8 (budget 0.05 ppl)", "", "| arm | rule | wikitext Δ | c4val1 Δ | verdict |", "|---|---|---|---|---|"]
    for arm, r in v.get("k8", {}).items():
        d = r.get("deltas", {})
        rule = "two-sided |Δ| ≤ 0.05" if arm == "int4all" else "one-sided Δ ≤ +0.05, calibrated on wikitext-train"
        L.append(f"| `{arm}` | {rule} | {_fmt(d.get('wikitext'), 5)} | {_fmt(d.get('c4val1'), 5)} | **{r['verdict']}**"
                 + (f" — {r['reason']}" if r.get("reason") else "") + " |")
    L += ["", "## P44-a — per-expert residual census (P3: ≤ 10 % of experts carry ≥ 50 % of the summed error)", "",
          "| family | method | experts | experts for 50 % | fraction | top-10 % share | verdict |", "|---|---|---|---|---|---|---|"]
    for fam, r in v.get("census", {}).items():
        L.append(f"| {fam} | {r.get('method', '—')} | {_fmt(r.get('n_experts'))} | {_fmt(r.get('experts_for_share'))} | "
                 f"{_fmt(r.get('fraction_for_share'))} | {_fmt(r.get('top10pct_share'))} | **{r['verdict']}**"
                 + (f" — {r['reason']}" if r.get("reason") else "") + " |")
    L += ["", "## P44-b — KL from the family's reference (reading rule: every stratum ≤ 1.10× NF4 and pooled Δ ≤ 0.005 nats/token)", ""]
    for fam, r in v.get("kl", {}).items():
        L += [f"### {fam} (scorer: {r.get('scorer')})", ""]
        for arm, rr in r.items():
            if not isinstance(rr, dict) or "licenses" not in rr:
                continue
            L += [f"- `{arm}`: pooled control {rr['pooled_control']:.4e}, candidate {rr['pooled_candidate']:.4e}, "
                  f"Δ {rr['pooled_delta']:+.4e} → **{'LICENSES (kl-vs-bf16)' if rr['licenses'] else 'does not license'}**"]
            for s, ps in rr["per_stratum"].items():
                L.append(f"    - {s}: {_fmt(ps.get('control'), 6)} → {_fmt(ps.get('candidate'), 6)} "
                         f"(×{_fmt(ps.get('ratio'), 3)}) {'ok' if ps.get('within_band') else 'OVER'}")
        if "P4_r1epi_equals_nf4" in r:
            p4 = r["P4_r1epi_equals_nf4"]
            L.append(f"- P4 (`r1epi` = `nf4` to < 1e-4 per stratum): **{'holds' if p4.get('holds') else 'FAILS'}** "
                     f"{p4.get('abs_delta_per_stratum', p4)}")
        if "P6_store_at_floor" in r:
            p6 = r["P6_store_at_floor"]
            L.append(f"- P6 (`store_r12` at the instrument floor < 1e-3): **{'holds' if p6['holds'] else 'FAILS'}** (KL {p6['kl_mean']:.3e})")
        if r.get("not_measured"):
            L.append(f"- not measured: {sorted(r['not_measured'])}")
        L.append("")
    return "\n".join(L) + "\n"


def reduce(run_dir: str) -> dict:
    return {"k8": k8_verdicts(run_dir), "census": {f: census_verdict(run_dir, f) for f in CENSUS_METHOD},
            "kl": kl_verdicts(run_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    v = reduce(a.run_dir)
    md = render_md(v)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
