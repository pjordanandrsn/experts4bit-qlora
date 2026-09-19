#!/usr/bin/env python3
"""bench/p51/p51_reduce.py -- the P51 reducer (P51-PREREG.md): the graded store map for Gemma-4, quality and
bytes from the same rows. Rules only; a missing row is NOT_READ, and a refuted anchor reads nothing else.

    python bench/p51/p51_reduce.py <run_dir> [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

ANCHOR = (0.040, 0.055)      # M1: P50's keep-20 = 0.0469
INT8_HEAD_FAILS = 0.50       # M2
INT8_HEAD_REOPENS = 0.30
GRADED_SHIPS = 0.10          # M3
GRADED_FAILS = 0.30
BYTES_MAX = 0.85             # M4: graded store <= 0.85x the anchor's
CRUSH_BYTES = 0.05           # M5: crushing the tail moves the store by < 5% ...
CRUSH_KL = 1.3               # ... and the KL by < 1.3x


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


def reduce(run_dir: str) -> dict:
    rec = _load(os.path.join(run_dir, "gemma4mix_kl.json"))
    if rec is None:
        return {"verdict": "NOT_READ: no gemma4mix_kl.json"}
    ok, used, ckl = scorer_admitted(rec)
    out = {"scorer_control": {"admitted": ok, "scorer": used, "reference_self_kl": ckl}, "not_measured": rec.get("not_measured", {})}
    rows = {}
    for r in rec.get("rows", []):
        e = r.get("engagement") or {}
        rows[r["arm"]] = {"kl_mean": r["kl_mean"], "top1": r.get("top1_agreement"), "builder": r.get("builder"),
                          "expert_gb": e.get("expert_bytes_total_gb"), "stacks_by_store": e.get("stacks_by_store")}
    out["rows"] = rows
    if not ok:
        out["verdict"] = f"NOT_READ: the {used} scorer is refused (reference self-KL {ckl})"
        return out
    a = rows.get("bf16_20")
    out["M1_anchor"] = ({"kl": a["kl_mean"], "band": ANCHOR, "expert_gb": a["expert_gb"],
                         "verdict": "HOLDS" if ANCHOR[0] <= a["kl_mean"] <= ANCHOR[1] else "REFUTED"}
                        if a else {"verdict": "NOT_READ"})
    if not a or out["M1_anchor"]["verdict"] != "HOLDS":
        out["verdict"] = ("NOT_READ: the anchor is refused, so the map does not rebuild P50's configuration and "
                          "no other row in this lane is read (M1's registered consequence)")
        return out
    i8 = rows.get("int8_20")
    out["M2_uniform_int8"] = ({"kl": i8["kl_mean"], "fails_min": INT8_HEAD_FAILS, "reopens_max": INT8_HEAD_REOPENS,
                               "verdict": ("HOLDS" if i8["kl_mean"] >= INT8_HEAD_FAILS else
                                           "REFUTED" if i8["kl_mean"] < INT8_HEAD_REOPENS else "INCONCLUSIVE"),
                               "reads": ("a uniform int8 head fails, as P49's layer-0 row implies"
                                         if i8["kl_mean"] >= INT8_HEAD_FAILS else
                                         "a uniform int8 head is better than P49's single-layer row implies -- the head question reopens"
                                         if i8["kl_mean"] < INT8_HEAD_REOPENS else
                                         "between the thresholds: reported, not read either way")}
                              if i8 else {"verdict": "NOT_READ"})
    g = rows.get("graded_10_10")
    if g:
        out["M3_graded"] = {"kl": g["kl_mean"], "ships_max": GRADED_SHIPS, "fails_min": GRADED_FAILS,
                            "vs_anchor": g["kl_mean"] / a["kl_mean"],
                            "verdict": ("HOLDS" if g["kl_mean"] <= GRADED_SHIPS else
                                        "REFUTED" if g["kl_mean"] >= GRADED_FAILS else "INCONCLUSIVE")}
        out["M4_bytes"] = ({"graded_gb": g["expert_gb"], "anchor_gb": a["expert_gb"], "ratio": g["expert_gb"] / a["expert_gb"],
                            "max": BYTES_MAX, "saved_gb": round(a["expert_gb"] - g["expert_gb"], 2),
                            "verdict": "HOLDS" if g["expert_gb"] / a["expert_gb"] <= BYTES_MAX else "REFUTED"}
                           if g.get("expert_gb") and a.get("expert_gb") else {"verdict": "NOT_READ"})
    else:
        out["M3_graded"] = out["M4_bytes"] = {"verdict": "NOT_READ"}
    c = rows.get("graded_10_10_crush")
    if c and g and g.get("expert_gb"):
        db = abs(c["expert_gb"] - g["expert_gb"]) / g["expert_gb"]
        dk = c["kl_mean"] / g["kl_mean"]
        out["M5_crush"] = {"crush_gb": c["expert_gb"], "graded_gb": g["expert_gb"], "bytes_delta": db, "kl_ratio": dk,
                           "bytes_max": CRUSH_BYTES, "kl_max": CRUSH_KL,
                           "verdict": "HOLDS" if db < CRUSH_BYTES and dk < CRUSH_KL else "REFUTED",
                           "reads": ("crushing the tail is the cheap half: the head is where the memory lives"
                                     if db < CRUSH_BYTES else "crushing the tail pays -- the tail store belongs in the default")}
    else:
        out["M5_crush"] = {"verdict": "NOT_READ"}
    m3 = out["M3_graded"].get("verdict")
    m4 = out["M4_bytes"].get("verdict")
    out["decision"] = ("SHIP the graded map (M1, M3, M4 hold)" if m3 == "HOLDS" and m4 == "HOLDS" else
                       "SHIP P50's uniform bf16 head; grading does not pay" if m3 == "REFUTED" else
                       "REPORT only: grading is between the registered thresholds" if m3 == "INCONCLUSIVE" else
                       "NOT_READ")
    return out


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P51 — verdicts read from the rows", ""]
    if "rows" not in v:
        return "\n".join(L + [v.get("verdict", "NOT_READ")])
    sc = v["scorer_control"]
    L += [f"Scorer: `{sc['scorer']}` (admitted: {sc['admitted']}; reference self-KL {_f(sc['reference_self_kl'])})", "",
          "| arm | tiers | KL nats/token | top-1 | expert store GB |", "|---|---|---|---|---|"]
    for arm, r in v["rows"].items():
        tiers = (r["builder"] or "").replace("loader_tiers_", "")
        L.append(f"| `{arm}` | `{tiers}` | {_f(r['kl_mean'])} | {_f(r['top1'], 3)} | {_f(r['expert_gb'], 2)} |")
    for k in v.get("not_measured", {}):
        L.append(f"| `{k}` | — | NOT MEASURED | — | — |")
    L += [""]
    if "M1_anchor" in v and v["M1_anchor"].get("verdict") != "HOLDS":
        return "\n".join(L + [f"- **M1_anchor: {v['M1_anchor'].get('verdict')}**", "", v.get("verdict", "")]) + "\n"
    for key in ("M1_anchor", "M2_uniform_int8", "M3_graded", "M4_bytes", "M5_crush"):
        p = v.get(key, {})
        extra = f" — {p['reads']}" if "reads" in p else ""
        if key == "M4_bytes" and p.get("ratio"):
            extra = f" — {p['graded_gb']:.1f} GB vs {p['anchor_gb']:.1f} GB = {p['ratio']:.2f}× (saves {p['saved_gb']} GB)" + extra
        if key == "M3_graded" and p.get("vs_anchor"):
            extra = f" — {p['vs_anchor']:.1f}× the anchor's KL" + extra
        L.append(f"- **{key}: {p.get('verdict', 'NOT_READ')}**{extra}")
    L += ["", f"**Decision: {v.get('decision', 'NOT_READ')}**"]
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
