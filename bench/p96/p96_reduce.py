#!/usr/bin/env python3
"""Lane P96's reducer (bench/p96/PREREG-p96.md; e4b#564). K25's default, asked again with P95's windowed K8 gate.

P94 read K25 (t) against the served NF4 M-tile (m) on one window per text: QUALITY_FAIL. P95 measured that one window
cannot resolve 0.05 on these families and named the gate this lane applies, per family (Granite r12epi, OLMoE nf4):
  m  the served NF4 M-tile kernel        t  K25, the select tree through TF32
on fresh windows (c4val1 9-16, wikitext 9-12; P94 used 0, P95 0-8).

Reads, per family F: F_k8_{m,t}_{c4val1,wikitext}_w{k}.json, and logs/census_F_{m,t}_b1_d1.txt (engagement: m runs
``_gemm_nf4_grouped`` 2 x layers per step and no ``_gemv_nf4*``; t runs ``_gemm_nf4_grouped_smallm`` 2 x layers per
step and neither of the others).

  VOID          an engagement census fails; fewer complete windows (both arms present) than P95's windows_needed
                (c4val1 5, wikitext 2); a window's arms scored different text; two windows scored the same text; t is
                bit-equal to m in every complete window of a text
  LICENSED      |mean(t - m)| <= 0.05 ppl over the complete windows on every text in both families -- with P93's speed
                (B=16 x0.594 / x0.598), E4B_NF4_GROUPED_SMALLM defaults to auto (rows above T == 1)
  QUALITY_FAIL  otherwise
Reported, not gated: each text's per-window t - m, mean, SD and SE, and the SD against P95's sigma for that text.

    python p96_reduce.py --dir <dir> --out verdict.json
    python p96_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

FAMILIES = {"granite": 32, "olmoe": 16}
WINDOWS = {"c4val1": (9, 10, 11, 12, 13, 14, 15, 16), "wikitext": (9, 10, 11, 12)}
MIN_WINDOWS = {"c4val1": 5, "wikitext": 2}         # P95's windows_needed (bench/p95/RESULTS-p95.md)
ARMS = ("m", "t")
SERVED = "_gemm_nf4_grouped"
K25 = "_gemm_nf4_grouped_smallm"
GEMV_PREFIX = "_gemv_nf4"
GATE = 0.05
#: P95's per-window sigma by family and text (reported beside this lane's SD, not gated)
P95_SIGMA = {("granite", "c4val1"): 0.0535, ("granite", "wikitext"): 0.0258,
             ("olmoe", "c4val1"): 0.0546, ("olmoe", "wikitext"): 0.0303}


def _p42():
    here = Path(__file__).resolve().parent
    for cand in (here / "p42_reduce.py", here.parent / "p42" / "p42_reduce.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("p42_reduce", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("p42_reduce.py not found beside this reducer or in bench/p42")


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def load(d: Path) -> dict:
    p42 = _p42()
    fams = {}
    for fam in FAMILIES:
        f = {"k8": {}, "census": {}}
        for src, ks in WINDOWS.items():
            for k in ks:
                for arm in ARMS:
                    r = _json(d / f"{fam}_k8_{arm}_{src}_w{k}.json")
                    if isinstance(r, dict) and "ppl" in r and "mean_nll" in r and "text_sha" in r:
                        f["k8"][(arm, src, k)] = r
        for arm in ARMS:
            c = d / "logs" / f"census_{fam}_{arm}_b1_d1.txt"
            if c.is_file():
                replays, rows = p42.parse_census(c.read_text())
                if replays:
                    calls: dict = {}
                    for row in rows:
                        calls[row["name"].strip()] = calls.get(row["name"].strip(), 0) + row["calls"]
                    f["census"][arm] = {n: k / replays for n, k in calls.items()}
        fams[fam] = f
    return {"families": fams}


def _engagement(fam: str, cen: dict) -> list:
    L, out = FAMILIES[fam], []
    for arm, want in (("m", SERVED), ("t", K25)):
        c = cen.get(arm)
        if c is None:
            out.append(f"missing: census {arm} B=1")
            continue
        others = [n for n in (SERVED, K25) if n != want]
        gemv = sum(v for n, v in c.items() if n.startswith(GEMV_PREFIX))
        if c.get(want, 0) != 2 * L or any(c.get(n, 0) for n in others) or gemv:
            out.append(f"engagement {arm} B=1: {want} {c.get(want, 0):g}/step (want {2 * L}), "
                       + ", ".join(f"{n} {c.get(n, 0):g}" for n in others) + f", NF4 GEMV {gemv:g} (want 0)")
    return out


def reduce(data: dict) -> dict:
    out: dict = {"lane": "P96", "families": [], "verdict": None, "reasons": []}
    void, fails = [], []
    for fam, f in data["families"].items():
        r: dict = {"family": fam, "texts": {}, "reasons": _engagement(fam, f["census"])}
        k8 = f["k8"]
        shas: dict = {}
        for src, ks in WINDOWS.items():
            complete = [k for k in ks if all((a, src, k) in k8 for a in ARMS)]
            for k in complete:
                s = {k8[(a, src, k)]["text_sha"] for a in ARMS}
                if len(s) != 1:
                    r["reasons"].append(f"{src} window {k}: the arms scored different text")
                    continue
                sha = s.pop()
                if sha in shas:
                    r["reasons"].append(f"{src} window {k} scored the same text as {shas[sha]}: the windows did not move")
                shas[sha] = f"{src} window {k}"
            if len(complete) < MIN_WINDOWS[src]:
                r["reasons"].append(f"{src}: {len(complete)} complete windows (want >= {MIN_WINDOWS[src]})")
            if complete and all(k8[("t", src, k)]["mean_nll"] == k8[("m", src, k)]["mean_nll"] for k in complete):
                r["reasons"].append(f"{src}: t is bit-equal to m in every complete window (K25 did not engage)")
            d = [float(k8[("t", src, k)]["ppl"]) - float(k8[("m", src, k)]["ppl"]) for k in complete]
            t: dict = {"windows": complete, "t-m": d}
            if d:
                t["mean"] = sum(d) / len(d)
                t["sd"] = statistics.stdev(d) if len(d) > 1 else float("nan")
                t["se"] = t["sd"] / math.sqrt(len(d)) if len(d) > 1 else float("nan")
                t["p95_sigma"] = P95_SIGMA[(fam, src)]
                t["pass"] = abs(t["mean"]) <= GATE
            r["texts"][src] = t
        if r["reasons"]:
            void.append(f"{fam} ({'; '.join(r['reasons'])})")
        else:
            fails += [f"{fam} {src} (mean t - m {t['mean']:+.4f})" for src, t in r["texts"].items() if not t["pass"]]
        out["families"].append(r)
    if void:
        out["verdict"] = "VOID"
        out["reasons"].append("VOID: " + " | ".join(void))
    elif fails:
        out["verdict"] = "QUALITY_FAIL"
        out["reasons"].append(f"|mean t - m| > {GATE} in: " + ", ".join(fails))
    else:
        out["verdict"] = "LICENSED"
        out["reasons"].append(f"|mean t - m| <= {GATE} over the windows on every text in both families; with P93's speed, "
                              "the T > 1 default moves to auto")
    return out


def _synthetic(shift=None, noise=0.03, drop=None, served_calls=None, k25_calls=None, same_sha=False, t_equal_m=False,
               mixed_sha=False, windows=None):
    """Per-window t - m = shift[(fam, src)] + noise * a fixed zero-mean pattern."""
    pattern = [1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25]
    shift = shift or {}
    fams = {}
    for fam, L in FAMILIES.items():
        f = {"k8": {}, "census": {
            "m": {SERVED: served_calls if served_calls is not None else 2 * L, "void cutlass::...": 9},
            "t": {K25: k25_calls if k25_calls is not None else 2 * L, "void cutlass::...": 9}}}
        for src, ks in WINDOWS.items():
            for i, k in enumerate((windows or {}).get(src, ks)):
                m = 10.0 + 0.1 * k
                tv = m if t_equal_m else m + shift.get((fam, src), 0.0) + noise * pattern[i % len(pattern)]
                sha = f"sha-{src}" if same_sha else f"sha-{src}-{k}"
                for a, v in (("m", m), ("t", tv)):
                    f["k8"][(a, src, k)] = {"ppl": v, "mean_nll": v / 7,
                                            "text_sha": (sha + a) if (mixed_sha and i == 1) else sha}
        fams[fam] = f
    if drop:
        fams[drop[0]]["k8"].pop(drop[1], None)
    return {"families": fams}


def self_test() -> None:
    v = reduce(_synthetic())
    assert v["verdict"] == "LICENSED", v                                              # zero-mean noise: inside the gate
    assert abs(v["families"][0]["texts"]["c4val1"]["mean"]) < 1e-9, v
    assert reduce(_synthetic(shift={("olmoe", "c4val1"): 0.06}))["verdict"] == "QUALITY_FAIL"
    assert reduce(_synthetic(shift={("granite", "wikitext"): -0.06}))["verdict"] == "QUALITY_FAIL"   # two-sided
    assert reduce(_synthetic(shift={("olmoe", "c4val1"): 0.049}))["verdict"] == "LICENSED"  # inside (0.05 inexact)
    assert reduce(_synthetic(windows={"c4val1": (9, 10, 11, 12)}))["verdict"] == "VOID"      # 4 < 5 windows
    assert reduce(_synthetic(windows={"c4val1": (9, 10, 11, 12, 13)}))["verdict"] == "LICENSED"  # 5 is enough
    assert reduce(_synthetic(drop=("granite", ("t", "wikitext", 10)), windows={"wikitext": (9, 10)}))["verdict"] == "VOID"
    assert reduce(_synthetic(served_calls=0))["verdict"] == "VOID"                    # m did not engage
    assert reduce(_synthetic(k25_calls=0))["verdict"] == "VOID"                       # t did not engage
    assert reduce(_synthetic(t_equal_m=True))["verdict"] == "VOID"                    # K25 did not change anything
    assert reduce(_synthetic(same_sha=True))["verdict"] == "VOID"                     # the windows did not move
    assert reduce(_synthetic(mixed_sha=True))["verdict"] == "VOID"                    # arms scored different text
    print("p96_reduce self-test OK (12 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P96_VERDICT {v['verdict']} | " + " | ".join(v["reasons"]))
    for r in v["families"]:
        for src, t in r["texts"].items():
            if "mean" not in t:
                continue
            print(f"  {r['family']} {src}: n={len(t['windows'])} mean t-m {t['mean']:+.4f} sd {t['sd']:.4f} se {t['se']:.4f} "
                  f"(P95 sigma {t['p95_sigma']:.4f}) {'pass' if t['pass'] else 'FAIL'} | "
                  + " ".join(f"{x:+.3f}" for x in t["t-m"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
