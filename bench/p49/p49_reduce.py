#!/usr/bin/env python3
"""bench/p49/p49_reduce.py -- the P49 reducer (P49-PREREG.md): expert-format rows on layer 0 (KL vs the bf16 checkpoint, from
`gemma4fmt_kl.json`) and the activation probe (`gemma4fmt_act.json`). Rules only; a missing row is NOT_READ.

    python bench/p49/p49_reduce.py <run_dir> [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os

L0_NF4_ANCHOR = (0.75, 1.05)   # P0: L00_nf4 reproduces P48's layer-0 row (0.892)
INT8_SURVIVES = 0.02           # P1: L00_int8 <= 0.02 nats   (alt >= 0.2: 8-bit does not rescue layer 0 either)
INT8_FAILS = 0.20
FP8_SURVIVES = 0.05            # P2: L00_fp8 <= 0.05
FP4_FLOOR = 0.50               # P3: L00_fp4 >= 0.50 (no better than NF4's class)
B32_FLOOR_RATIO = 0.55         # P4: L00_nf4b32 >= 0.55 x L00_nf4 (block size does not rescue)
AMP_RHO_MIN = 0.8              # P5: Spearman(KL, rel_err_norm2) over the NF4 depth arms >= 0.8 ...
AMP_RATIO_MIN = 5.0            # ... and rel_err_norm2(L0) / rel_err_norm2(L27) >= 5 while rel_err_raw stays within 2x across depths
RAW_SPREAD_MAX = 2.0


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


def _v(cond_hold, cond_refute):
    return "HOLDS" if cond_hold else ("REFUTED" if cond_refute else "INCONCLUSIVE")


def reduce(run_dir: str) -> dict:
    kl = _load(os.path.join(run_dir, "gemma4fmt_kl.json"))
    act = _load(os.path.join(run_dir, "gemma4fmt_act.json"))
    out = {"kl_present": kl is not None, "act_present": act is not None}
    rows = {}
    if kl is not None:
        ok, used, ckl = scorer_admitted(kl)
        out["scorer_control"] = {"admitted": ok, "scorer": used, "reference_self_kl": ckl}
        out["not_measured"] = kl.get("not_measured", {})
        if ok:
            rows = {r["arm"]: r for r in kl.get("rows", [])}
        else:
            out["kl_verdict"] = f"NOT_READ: the {used} scorer is refused (reference self-KL {ckl})"
    out["rows"] = {a: {"kl_mean": r["kl_mean"], "top1": r.get("top1_agreement"), "builder": r.get("builder"),
                       "quant_type": (r.get("engagement") or {}).get("quant_type"), "blocksize": (r.get("engagement") or {}).get("blocksize"),
                       "quantized_types": (r.get("engagement") or {}).get("quantized_types")} for a, r in rows.items()}
    g = lambda a: rows[a]["kl_mean"] if a in rows else None  # noqa: E731
    nf4 = g("L00_nf4")
    out["P0_anchor"] = ({"kl": nf4, "band": L0_NF4_ANCHOR, "verdict": "HOLDS" if L0_NF4_ANCHOR[0] <= nf4 <= L0_NF4_ANCHOR[1] else "REFUTED"}
                        if nf4 is not None else {"verdict": "NOT_READ"})
    i8 = g("L00_int8")
    out["P1_int8"] = ({"kl": i8, "survives_max": INT8_SURVIVES, "fails_min": INT8_FAILS, "verdict": _v(i8 <= INT8_SURVIVES, i8 >= INT8_FAILS),
                       "reads": ("int8 experts SURVIVE layer 0: an 8-bit store is the remedy candidate" if i8 <= INT8_SURVIVES else
                                 "even 8-bit does not rescue layer 0: the early layers stay high-precision" if i8 >= INT8_FAILS else
                                 "int8 costs a measurable but sub-0.2 gap: report, gate before use")}
                      if i8 is not None else {"verdict": "NOT_READ"})
    f8 = g("L00_fp8")
    out["P2_fp8"] = ({"kl": f8, "max": FP8_SURVIVES, "verdict": "HOLDS" if f8 <= FP8_SURVIVES else "REFUTED"} if f8 is not None else {"verdict": "NOT_READ"})
    f4 = g("L00_fp4")
    out["P3_fp4"] = ({"kl": f4, "floor": FP4_FLOOR, "verdict": "HOLDS" if f4 >= FP4_FLOOR else "REFUTED"} if f4 is not None else {"verdict": "NOT_READ"})
    b32 = g("L00_nf4b32")
    out["P4_block32"] = ({"kl": b32, "ratio_to_nf4": (b32 / nf4) if nf4 else None, "floor_ratio": B32_FLOOR_RATIO,
                          "verdict": ("HOLDS" if nf4 and b32 >= B32_FLOOR_RATIO * nf4 else "REFUTED") if nf4 else "NOT_READ"}
                         if b32 is not None else {"verdict": "NOT_READ"})
    # P5: the activation probe over the NF4 depth arms
    if act is not None and act.get("arms"):
        depth = sorted([a for a in act["arms"] if a["quant_type"] == "nf4" and a["blocksize"] == 64], key=lambda a: a["layer"])
        kls = [g(a["arm"]) for a in depth]
        pairs = [(a, k) for a, k in zip(depth, kls) if k is not None]
        prof = {a["layer"]: {"rel_err_raw": a["rel_err_raw"], "rel_err_norm2": a["rel_err_norm2"], "amplification": a["amplification_norm2_over_raw"],
                             "cos_raw": a["cos_raw_mean"], "kl": k} for a, k in zip(depth, kls)}
        ref = {int(i): s for i, s in (act.get("reference") or {}).get("layers", {}).items()}
        p5 = {"n_depths": len(depth), "profile": prof, "not_measured": act.get("not_measured", {}),
              "reference_contrib": {i: {"contrib_norm2_over_resid": ref[i]["contrib_norm2_over_resid"], "rms_raw": ref[i]["rms_raw"],
                                        "norm_gain": ref[i]["norm_gain_norm2_over_raw"], "dense_over_moe": ref[i]["dense_over_moe"]}
                                    for i in sorted(ref) if i in prof}}
        if len(pairs) >= 3:
            rho_n2 = spearman([k for _, k in pairs], [a["rel_err_norm2"] for a, _ in pairs])
            rho_raw = spearman([k for _, k in pairs], [a["rel_err_raw"] for a, _ in pairs])
            raws = [a["rel_err_raw"] for a, _ in pairs]
            n2s = {a["layer"]: a["rel_err_norm2"] for a, _ in pairs}
            lo, hi = min(n2s), max(n2s)
            ratio = n2s[lo] / max(n2s[hi], 1e-30)
            spread = max(raws) / max(min(raws), 1e-30)
            p5.update({"spearman_kl_vs_rel_norm2": rho_n2, "spearman_kl_vs_rel_raw": rho_raw, "norm2_ratio_first_over_last": ratio,
                       "raw_spread_max_over_min": spread, "rho_min": AMP_RHO_MIN, "ratio_min": AMP_RATIO_MIN, "raw_spread_max": RAW_SPREAD_MAX})
            amp_holds = rho_n2 is not None and rho_n2 >= AMP_RHO_MIN and ratio >= AMP_RATIO_MIN and spread <= RAW_SPREAD_MAX
            raw_story = spread > RAW_SPREAD_MAX and rho_raw is not None and rho_raw >= AMP_RHO_MIN
            p5["verdict"] = "HOLDS" if amp_holds else ("ALTERNATIVE (raw output error itself tracks the KL: cancellation inside the expert sum, not the post-norm)" if raw_story else "REFUTED")
        else:
            p5["verdict"] = "NOT_READ"
        out["P5_amplification"] = p5
    else:
        out["P5_amplification"] = {"verdict": "NOT_READ", "reason": "no activation-probe receipt"}
    return out


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    L = ["# P49 — verdicts read from the rows", ""]
    if v.get("scorer_control"):
        sc = v["scorer_control"]
        L += [f"Scorer: `{sc['scorer']}` (admitted: {sc['admitted']}; reference self-KL {_f(sc['reference_self_kl'])})", ""]
    L += ["| arm | store | KL nats/token | top-1 |", "|---|---|---|---|"]
    for a, r in v.get("rows", {}).items():
        L.append(f"| `{a}` | {r['quant_type']} b{r['blocksize']} | {_f(r['kl_mean'])} | {_f(r['top1'], 3)} |")
    for k in v.get("not_measured", {}):
        L.append(f"| `{k}` | — | NOT MEASURED | — |")
    L += [""]
    for key in ("P0_anchor", "P1_int8", "P2_fp8", "P3_fp4", "P4_block32", "P5_amplification"):
        p = v.get(key, {})
        extra = ""
        if key == "P1_int8" and "reads" in p:
            extra = " — " + p["reads"]
        if key == "P5_amplification" and "profile" in p:
            extra = " — " + "; ".join(f"L{i}: raw {q['rel_err_raw']:.3f} / norm2 {q['rel_err_norm2']:.3f} (amp {q['amplification']:.2f}, KL {_f(q['kl'], 3)})" for i, q in sorted(p["profile"].items()))
            if "spearman_kl_vs_rel_norm2" in p:
                extra += f"; Spearman(KL, norm2) {_f(p['spearman_kl_vs_rel_norm2'], 3)}, (KL, raw) {_f(p['spearman_kl_vs_rel_raw'], 3)}; norm2 first/last {p['norm2_ratio_first_over_last']:.1f}, raw spread {p['raw_spread_max_over_min']:.2f}"
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
