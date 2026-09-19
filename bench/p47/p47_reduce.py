#!/usr/bin/env python3
"""bench/p47/p47_reduce.py -- the P47 reducer: the registered reading rules (P47-PREREG.md) applied to the one
receipt `gemma4diag_kl.json`, nothing else. A missing or refused row is NOT_READ, never a verdict.

    python bench/p47/p47_reduce.py <run_dir> [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

ANCHOR_BAND = (0.95, 1.25)     # P1: served_nf4 reproduces P44-b run 5's 1.077 (runs 2/3/5 agreed to 4 digits under their scorers)
STACK_EPS = 0.05               # P2: |loader_nf4 - served_nf4| < 0.05 -> the gap is the quantised MODEL, not the serving stack
STACK_ALT = 0.15               # P2 alternative: loader_nf4 < 0.15 while served_nf4 is in the anchor band -> the STACK
MODEL_FAITHFUL = 0.05          # P3: loader_bf16experts < 0.05 -> e4b's Gemma-4 modelling is faithful; NF4 of the experts is the cost
MODEL_DEFECT = 0.50            # P3 alternative: >= 0.50 -> a modelling defect independent of quantisation
DEPTH_DOMINANT = 0.70          # P4: one half carries > 70 % of (lo + hi) -> the cost is localised in depth
ADDITIVITY = 0.30              # P4: |lo + hi - loader_nf4| / loader_nf4 <= 0.30 -> the halves add (else interaction, not localisation)
SELF_KL_MAX = 1e-2             # the P44 amendment-5 control: decode admitted only under this


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def scorer_admitted(rec: dict) -> tuple:
    used = (rec.get("scorer") or {}).get("used") or (rec.get("scorer") or {}).get("name")
    ctl = ((rec.get("reference_pass") or {}).get("self_consistency")
           or (rec.get("controls") or {}).get("reference_decode_vs_prefill") or {})
    kl = ctl.get("kl_mean")
    if used == "prefill":
        return True, used, kl
    if used == "decode":
        return bool(ctl.get("passes", ctl.get("passes_1e-2"))), used, kl
    return False, used, kl


def rows_by_arm(rec: dict) -> dict:
    return {r["arm"]: r for r in rec.get("rows", [])}


def p1_anchor(rows: dict) -> dict:
    r = rows.get("served_nf4")
    if r is None:
        return {"verdict": "NOT_READ", "reason": "no served_nf4 row"}
    kl = r["kl_mean"]
    return {"kl_mean": kl, "band": ANCHOR_BAND, "holds": ANCHOR_BAND[0] <= kl <= ANCHOR_BAND[1],
            "verdict": "HOLDS" if ANCHOR_BAND[0] <= kl <= ANCHOR_BAND[1] else "REFUTED"}


def p2_stack_vs_model(rows: dict) -> dict:
    s, ld = rows.get("served_nf4"), rows.get("loader_nf4")
    if s is None or ld is None:
        return {"verdict": "NOT_READ", "reason": "served_nf4 or loader_nf4 row missing"}
    d = ld["kl_mean"] - s["kl_mean"]
    out = {"served": s["kl_mean"], "loader": ld["kl_mean"], "delta_loader_minus_served": d, "eps": STACK_EPS, "alt": STACK_ALT}
    if abs(d) < STACK_EPS:
        out["verdict"], out["reads"] = "HOLDS", "the quantised MODEL carries the gap; the serving stack (arena bake + placement + hybrid tier) adds nothing"
    elif ld["kl_mean"] < STACK_ALT and s["kl_mean"] >= ANCHOR_BAND[0]:
        out["verdict"], out["reads"] = "REFUTED", "the serving STACK carries the gap; the training-loader model is close to bf16"
    else:
        out["verdict"], out["reads"] = "MIXED", "both sides carry part of the gap -- neither registered reading; report the split"
    return out


def p3_modelling(rows: dict) -> dict:
    b = rows.get("loader_bf16experts")
    if b is None:
        return {"verdict": "NOT_READ", "reason": "no loader_bf16experts row"}
    kl = b["kl_mean"]
    out = {"kl_mean": kl, "faithful_max": MODEL_FAITHFUL, "defect_min": MODEL_DEFECT}
    if kl < MODEL_FAITHFUL:
        out["verdict"], out["reads"] = "HOLDS", "e4b's Gemma-4 modelling with bf16 experts matches the checkpoint; NF4 of THIS family's experts is the cost"
    elif kl >= MODEL_DEFECT:
        out["verdict"], out["reads"] = "REFUTED", "the gap survives with bf16 experts: a modelling defect independent of quantisation"
    else:
        out["verdict"], out["reads"] = "PARTIAL", "bf16 experts remove most but not all of the gap -- a modelling component below the defect line; report both"
    return out


def p4_depth(rows: dict) -> dict:
    lo, hi, full = rows.get("loader_nf4_lo"), rows.get("loader_nf4_hi"), rows.get("loader_nf4")
    if lo is None or hi is None:
        return {"verdict": "NOT_READ", "reason": "a half row is missing"}
    a, b = lo["kl_mean"], hi["kl_mean"]
    tot = a + b
    out = {"lo": a, "hi": b, "share_lo": (a / tot) if tot > 0 else None, "share_hi": (b / tot) if tot > 0 else None,
           "dominant_min": DEPTH_DOMINANT}
    if full is not None and full["kl_mean"] > 0:
        out["additivity"] = abs(tot - full["kl_mean"]) / full["kl_mean"]
        out["adds"] = out["additivity"] <= ADDITIVITY
    else:
        out["additivity"], out["adds"] = None, None
    if tot <= 0:
        out["verdict"] = "NOT_READ"
        return out
    dom = max(out["share_lo"], out["share_hi"])
    if dom > DEPTH_DOMINANT:
        out["verdict"] = "HOLDS"
        out["reads"] = f"localised: the {'first' if out['share_lo'] > out['share_hi'] else 'second'} half carries {dom:.0%} of lo + hi"
    else:
        out["verdict"], out["reads"] = "REFUTED", "spread: neither half carries more than 70 % -- the cost is per-layer, not a depth region"
    if out["adds"] is False:
        out["reads"] += "; the halves do NOT add (interaction) -- the shares are reported, not read as localisation"
    return out


def reduce(run_dir: str) -> dict:
    rec = _load(os.path.join(run_dir, "gemma4diag_kl.json"))
    if rec is None:
        return {"verdict": "NOT_READ: no gemma4diag_kl.json"}
    ok, used, ckl = scorer_admitted(rec)
    out = {"scorer_control": {"admitted": ok, "scorer": used, "reference_self_kl": ckl},
           "not_measured": rec.get("not_measured", {}),
           "rows": {r["arm"]: {"kl_mean": r["kl_mean"], "top1": r.get("top1_agreement"), "builder": r.get("builder"),
                               "per_stratum": {k: v.get("kl_mean") for k, v in (r.get("per_stratum") or {}).items()},
                               "engagement": {k: r.get("engagement", {}).get(k) for k in ("builder", "n_quantized", "n_unquantized", "expected_quantized", "expected_unquantized", "int4_expert_layers")}}
                    for r in rec.get("rows", [])}}
    if not ok:
        out["verdict"] = f"NOT_READ: the {used} scorer is refused for this family (reference self-KL {ckl}; control (i) must pass)"
        return out
    rows = rows_by_arm(rec)
    out["P1_anchor"] = p1_anchor(rows)
    out["P2_stack_vs_model"] = p2_stack_vs_model(rows)
    out["P3_modelling"] = p3_modelling(rows)
    out["P4_depth"] = p4_depth(rows)
    ctl = (rec.get("reference_pass") or {}).get("self_consistency") or {}
    out["P5_reference_self_kl"] = {"kl_mean": ctl.get("kl_mean"), "replicates_0.279": (ctl.get("kl_mean") is not None and 0.2 <= ctl["kl_mean"] <= 0.36),
                                   "verdict": ("HOLDS" if ctl.get("kl_mean") is not None and 0.2 <= ctl["kl_mean"] <= 0.36 else ("NOT_READ" if ctl.get("kl_mean") is None else "REFUTED"))}
    return out


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P47 — verdicts read from the rows", ""]
    if "rows" not in v:
        return "\n".join(L + [v.get("verdict", "NOT_READ")])
    sc = v["scorer_control"]
    L += [f"Scorer: `{sc['scorer']}` (admitted: {sc['admitted']}; reference decode-vs-prefill self-KL {_f(sc['reference_self_kl'])})", ""]
    L += ["| arm | builder | KL nats/token | top-1 | quantised / unquantised stacks |", "|---|---|---|---|---|"]
    for arm, r in v["rows"].items():
        e = r["engagement"]
        L.append(f"| `{arm}` | {r['builder']} | {_f(r['kl_mean'])} | {_f(r['top1'])} | {e.get('n_quantized')} / {e.get('n_unquantized')} |")
    for k in v.get("not_measured", {}):
        L.append(f"| `{k}` | — | NOT MEASURED | — | — |")
    L += [""]
    for key in ("P1_anchor", "P2_stack_vs_model", "P3_modelling", "P4_depth", "P5_reference_self_kl"):
        p = v.get(key, {})
        L.append(f"- **{key}: {p.get('verdict', 'NOT_READ')}** — {p.get('reads', p.get('reason', ''))}")
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
