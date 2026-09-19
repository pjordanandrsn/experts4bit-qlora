#!/usr/bin/env python3
"""bench/p50/p50_reduce.py -- the P50 reducer (P50-PREREG.md): the bf16-early-layers curve for Gemma-4, quality and bytes
from the same rows. Rules only; a missing row is NOT_READ.

    python bench/p50/p50_reduce.py <run_dir> [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

K15_ANCHOR = (0.11, 0.16)     # P1: P47's loader_nf4_hi = 0.1334
FLOOR = 0.05                  # P2/P3: the other families' NF4 floor
COST_MIN = 2.5                # P4: expert bytes at the passing k, over the all-NF4 store
TOP1_MIN = 0.95               # P5
NF4_ALL_LAYERS = 1.0837       # P47 loader_nf4, for the curve's k=0 end


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


def nf4_all_bytes(rows: dict) -> float | None:
    """The all-NF4 expert store in GB, derived from any row's census: (30 - k) quantised stacks cost q GB, so one costs
    q / (30 - k) and thirty cost 30x that. Averaged over the rows that carry the census."""
    per = []
    for k, r in rows.items():
        e = r.get("expert_bytes") or {}
        nq = r.get("n_quantized")
        if e.get("quantized") and nq:
            per.append(e["quantized"] / nq / 2 ** 30)
    return 30 * sum(per) / len(per) if per else None


def reduce(run_dir: str, n_layers: int = 30) -> dict:
    rec = _load(os.path.join(run_dir, "gemma4keep_kl.json"))
    if rec is None:
        return {"verdict": "NOT_READ: no gemma4keep_kl.json"}
    ok, used, ckl = scorer_admitted(rec)
    out = {"scorer_control": {"admitted": ok, "scorer": used, "reference_self_kl": ckl}, "not_measured": rec.get("not_measured", {})}
    rows = {}
    for r in rec.get("rows", []):
        arm = r["arm"]
        if not (arm.startswith("K") and arm[1:].isdigit()):
            continue
        e = r.get("engagement") or {}
        rows[int(arm[1:])] = {"kl_mean": r["kl_mean"], "top1": r.get("top1_agreement"), "expert_bytes": e.get("expert_bytes"),
                              "expert_gb": e.get("expert_bytes_total_gb"), "n_quantized": e.get("n_quantized"), "n_unquantized": e.get("n_unquantized")}
    out["rows"] = rows
    if not ok:
        out["verdict"] = f"NOT_READ: the {used} scorer is refused (reference self-KL {ckl})"
        return out
    ks = sorted(rows)
    out["P1_anchor"] = ({"k": 15, "kl": rows[15]["kl_mean"], "band": K15_ANCHOR,
                         "verdict": "HOLDS" if K15_ANCHOR[0] <= rows[15]["kl_mean"] <= K15_ANCHOR[1] else "REFUTED"}
                        if 15 in rows else {"verdict": "NOT_READ"})
    if len(ks) >= 2:
        mono = all(rows[a]["kl_mean"] > rows[b]["kl_mean"] for a, b in zip(ks, ks[1:]))
        out["P2_curve"] = {"monotone": mono, "k15_above_floor": (rows[15]["kl_mean"] > FLOOR) if 15 in rows else None, "floor": FLOOR,
                           "verdict": ("HOLDS" if mono and 15 in rows and rows[15]["kl_mean"] > FLOOR else
                                       "REFUTED" if 15 in rows and rows[15]["kl_mean"] <= FLOOR else "NOT_READ")}
    else:
        out["P2_curve"] = {"verdict": "NOT_READ"}
    passing = [k for k in ks if rows[k]["kl_mean"] <= FLOOR]
    kmin = min(passing) if passing else None
    out["passing_k"] = kmin
    if 20 in rows:
        out["P3_k20"] = {"k20": rows[20]["kl_mean"], "floor": FLOOR,
                         "verdict": "HOLDS" if rows[20]["kl_mean"] <= FLOOR else
                                    ("ALTERNATIVE (only k=24 reaches the floor)" if 24 in rows and rows[24]["kl_mean"] <= FLOOR else
                                     "REFUTED (no k reaches the floor)" if 24 in rows else "NOT_READ")}
    else:
        out["P3_k20"] = {"verdict": "NOT_READ"}
    base = nf4_all_bytes(rows)
    if kmin is not None and base and rows[kmin].get("expert_gb"):
        ratio = rows[kmin]["expert_gb"] / base
        out["P4_cost"] = {"passing_k": kmin, "expert_gb": rows[kmin]["expert_gb"], "all_nf4_gb": round(base, 3), "ratio": ratio,
                          "min_ratio": COST_MIN, "verdict": "HOLDS" if ratio >= COST_MIN else "REFUTED",
                          "reads": ("the remedy costs the 4-bit advantage: at the quality floor this is a mostly-bf16 model"
                                    if ratio >= COST_MIN else "the remedy is cheap: the default is easy to justify")}
    else:
        out["P4_cost"] = {"verdict": "NOT_READ", "passing_k": kmin, "all_nf4_gb": round(base, 3) if base else None}
    if kmin is not None and rows[kmin].get("top1") is not None:
        out["P5_top1"] = {"k": kmin, "top1": rows[kmin]["top1"], "min": TOP1_MIN,
                          "verdict": "HOLDS" if rows[kmin]["top1"] >= TOP1_MIN else "REFUTED"}
    else:
        out["P5_top1"] = {"verdict": "NOT_READ"}
    return out


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P50 — verdicts read from the rows", ""]
    if "rows" not in v:
        return "\n".join(L + [v.get("verdict", "NOT_READ")])
    sc = v["scorer_control"]
    L += [f"Scorer: `{sc['scorer']}` (admitted: {sc['admitted']}; reference self-KL {_f(sc['reference_self_kl'])})", "",
          "| k (bf16 layers 0..k−1) | NF4 layers | KL nats/token | top-1 | expert store GB |", "|---|---|---|---|---|",
          f"| 0 (all NF4, P47) | 30 | {NF4_ALL_LAYERS:.4f} | 0.638 | — |"]
    for k in sorted(v["rows"]):
        r = v["rows"][k]
        L.append(f"| {k} | {r['n_quantized']} | {_f(r['kl_mean'])} | {_f(r['top1'], 3)} | {_f(r['expert_gb'], 2)} |")
    for k in v.get("not_measured", {}):
        L.append(f"| `{k}` | — | NOT MEASURED | — | — |")
    L += ["", f"Smallest k at or under the {FLOOR} floor: **{v.get('passing_k', '—')}**", ""]
    for key in ("P1_anchor", "P2_curve", "P3_k20", "P4_cost", "P5_top1"):
        p = v.get(key, {})
        extra = f" — {p['reads']}" if "reads" in p else ""
        if key == "P4_cost" and p.get("ratio"):
            extra = f" — {p['expert_gb']:.1f} GB vs {p['all_nf4_gb']:.1f} GB all-NF4 = {p['ratio']:.2f}x." + extra
        L.append(f"- **{key}: {p.get('verdict', 'NOT_READ')}**{extra}")
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
