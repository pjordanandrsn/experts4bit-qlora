#!/usr/bin/env python3
"""bench/p52/p52_reduce.py -- the P52 reducer: the Gemma-4 gate, read on held-out prompts against a bar
registered before the measurement. Rules only. G2 refuted means NOTHING is read.

    python bench/p52/p52_reduce.py <run_dir> [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

BAR_KL = 0.10          # registered in P51 amendment 3, from configurations e4b already ships
BAR_TOP1 = 0.93
ANCHOR_COMMITTED = 0.0469      # bf16_20 on kl_prompts.py
GPTOSS_COMMITTED = 0.0222      # nf4_r12 on kl_prompts.py -- the bar's provenance
TOL = 0.15                     # G1/G2: within +-15 % of the committed-set value
EXPECT_SET = "kl_prompts_heldout"


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


def _rows(rec):
    return {r["arm"]: {"kl_mean": r["kl_mean"], "top1": r.get("top1_agreement"),
                       "per_stratum": {k: v.get("kl_mean") for k, v in (r.get("per_stratum") or {}).items()},
                       "expert_gb": (r.get("engagement") or {}).get("expert_bytes_total_gb")}
            for r in rec.get("rows", [])}


def _within(got, want, tol=TOL):
    return abs(got - want) <= tol * want


def reduce(run_dir: str) -> dict:
    g = _load(os.path.join(run_dir, "gemma4mix_kl.json"))
    o = _load(os.path.join(run_dir, "gptoss_kl.json"))
    out = {"bar": {"kl_max": BAR_KL, "top1_min": BAR_TOP1,
                   "registered": "P51 amendment 3, on main before this lane; derived from shipped configurations"}}
    if g is None:
        return {**out, "verdict": "NOT_READ: no gemma4mix_kl.json"}
    ok, used, ckl = scorer_admitted(g)
    out["scorer_control"] = {"admitted": ok, "scorer": used, "reference_self_kl": ckl}
    pset = (g.get("prompt_set") or {})
    out["prompt_set"] = {"name": pset.get("name"), "sha256": (pset.get("sha256") or "")[:16],
                         "n": pset.get("n_scored"), "strata": pset.get("strata")}
    if pset.get("name") != EXPECT_SET:
        return {**out, "verdict": f"NOT_READ: this lane scores the held-out set; the receipt names {pset.get('name')!r}"}
    if not ok:
        return {**out, "verdict": f"NOT_READ: the {used} scorer is refused (reference self-KL {ckl})"}
    gr = _rows(g)
    out["rows"] = gr
    out["not_measured"] = g.get("not_measured", {})

    # G2 first: the bar's provenance must replicate on THIS set, or nothing is read.
    if o is None:
        out["G2_bar_provenance"] = {"verdict": "NOT_READ", "reason": "no gptoss_kl.json"}
        return {**out, "verdict": "NOT_READ: the bar's provenance point was not measured on this set (G2's registered consequence)"}
    ook, oused, ockl = scorer_admitted(o)
    orows = _rows(o)
    ref = orows.get("nf4_r12")
    if not ook or ref is None:
        out["G2_bar_provenance"] = {"verdict": "NOT_READ", "scorer": oused, "admitted": ook}
        return {**out, "verdict": "NOT_READ: the bar's provenance point is unreadable on this set (G2)"}
    g2 = _within(ref["kl_mean"], GPTOSS_COMMITTED) and (ref["top1"] or 0) >= BAR_TOP1
    out["G2_bar_provenance"] = {"kl": ref["kl_mean"], "committed": GPTOSS_COMMITTED, "top1": ref["top1"],
                                "tol": TOL, "verdict": "HOLDS" if g2 else "REFUTED",
                                "reads": ("the bar transfers to this set" if g2 else
                                          "the bar's derivation does not hold on this set -- the gate is NOT read")}
    if not g2:
        return {**out, "verdict": "NOT_READ: G2 refuted -- the bar cannot be applied to a set on which its own provenance fails"}

    a = gr.get("bf16_20")
    out["G1_set_comparable"] = ({"kl": a["kl_mean"], "committed": ANCHOR_COMMITTED, "tol": TOL,
                                 "verdict": "HOLDS" if _within(a["kl_mean"], ANCHOR_COMMITTED) else "REFUTED",
                                 "reads": ("the held-out set reads the anchor where the committed set does"
                                           if _within(a["kl_mean"], ANCHOR_COMMITTED) else
                                           "the sets are not interchangeable; every number here is read on the held-out set alone")}
                                if a else {"verdict": "NOT_READ"})
    s = gr.get("graded_10_10")
    if s:
        passes = s["kl_mean"] <= BAR_KL and (s["top1"] or 0) >= BAR_TOP1
        out["G3_gate"] = {"kl": s["kl_mean"], "top1": s["top1"], "kl_max": BAR_KL, "top1_min": BAR_TOP1,
                          "verdict": "PASSES" if passes else "FAILS",
                          "reads": ("the graded map clears the bar and may ship as the Gemma-4 default"
                                    if passes else
                                    "the graded map does not clear the bar: it stays a documented option, and the gate is now RUN and not passed")}
    else:
        out["G3_gate"] = {"verdict": "NOT_READ"}
    u = gr.get("bf16_13")
    if s and u:
        out["G4_matched_bytes"] = {"graded": s["kl_mean"], "uniform": u["kl_mean"],
                                   "ratio": u["kl_mean"] / s["kl_mean"],
                                   "verdict": "HOLDS" if s["kl_mean"] < u["kl_mean"] else "REFUTED",
                                   "reads": ("P51's matched-bytes result replicates on held-out prompts"
                                             if s["kl_mean"] < u["kl_mean"] else
                                             "P51's matched-bytes result does NOT replicate -- the shape recommendation re-opens")}
    else:
        out["G4_matched_bytes"] = {"verdict": "NOT_READ"}
    out["decision"] = ("SHIP the graded map as the Gemma-4 default" if out["G3_gate"].get("verdict") == "PASSES"
                       else "DOCUMENTED OPTION only -- the gate was run and not passed"
                       if out["G3_gate"].get("verdict") == "FAILS" else "NOT_READ")
    return out


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P52 — the Gemma-4 gate, read on held-out prompts", ""]
    b = v["bar"]
    L += [f"Bar (registered before this lane): **KL ≤ {b['kl_max']} and top-1 ≥ {b['top1_min']}**.", ""]
    ps = v.get("prompt_set") or {}
    if ps:
        L += [f"Prompt set: `{ps.get('name')}` sha `{ps.get('sha256')}`, {ps.get('n')} prompts, strata {ps.get('strata')}", ""]
    if "rows" in v:
        L += ["| arm | KL nats/token | top-1 | expert store |", "|---|---|---|---|"]
        for arm, r in v["rows"].items():
            L.append(f"| `{arm}` | {_f(r['kl_mean'])} | {_f(r['top1'], 3)} | {_f(r['expert_gb'], 2)} |")
        L += [""]
    for key in ("G2_bar_provenance", "G1_set_comparable", "G3_gate", "G4_matched_bytes"):
        p = v.get(key)
        if not p:
            continue
        extra = f" — {p['reads']}" if "reads" in p else ""
        L.append(f"- **{key}: {p.get('verdict', 'NOT_READ')}**{extra}")
    if "decision" in v:
        L += ["", f"**Decision: {v['decision']}**"]
    elif v.get("verdict"):
        L += ["", v["verdict"]]
    return "\n".join(L) + "\n"


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
        open(a.md, "w").write(md)
    if a.json:
        json.dump(v, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
