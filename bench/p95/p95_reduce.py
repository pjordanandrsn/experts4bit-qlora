#!/usr/bin/env python3
"""Lane P95's reducer (bench/p95/PREREG-p95.md; e4b#564). K8's spread across arithmetics of equal per-GEMM error.

P94 read K8 at T == 1 under three arithmetics on ONE window of each text, and on c4val1 the pairs differed by 0.013 to
0.168 ppl. P95 scores the same three arithmetics on disjoint windows of each text, per family (Granite r12epi, OLMoE nf4):
  g  the scalar fp32 decode GEMV        m  the served NF4 M-tile kernel        t  K25, the select tree through TF32
Window 0 is P94's window (the reproduction control); the fresh windows are c4val1 1-8 and wikitext 1-4.

Reads, per family F: F_k8_{g,m,t}_{c4val1,wikitext}_w{k}.json, and logs/census_F_m_b1_d1.txt (engagement: the m arm
runs ``_gemm_nf4_grouped`` 2 x layers per step at T == 1 and no ``_gemv_nf4*``).

Per family and text, over the COMPLETE fresh windows (all three arms present), the per-window deltas of the pairs m - g
and t - m (the two equal-error pairs; grouped-nf4-gemm K27 read their per-GEMM rms error equal to four digits) give a
sample SD each; the instrument's per-window spread is sigma = max(SD(m - g), SD(t - m)).
  VOID            the m arm's engagement fails; a window-0 arm is missing; fewer than MIN_FRESH complete fresh windows
                  (c4val1 6, wikitext 3); a window's arms scored different text; two windows scored the same text (the
                  windows did not move); m or t is bit-equal to g in every complete fresh window of a text
  RESOLVED        sigma <= 0.025 on every text in both families: a single-window 0.05 gate falsely fails an unbiased
                  equal-error change at most ~5 % of the time (2 sigma), so the single-window K8 gate stands
  UNDER_RESOLVED  sigma > 0.025 on some family and text; for each, the windows a windowed-mean gate needs for the same
                  ~5 % bound: W = ceil((sigma / 0.025) ** 2)
Reported, not gated: per pair the mean, SD and SE over the fresh windows, whether t - m's mean is a consistent shift
(|mean| > t_0.975(n - 1) * SE), and whether window 0 reproduces P94's twelve values exactly. The lane licenses nothing.

    python p95_reduce.py --dir <dir> --out verdict.json
    python p95_reduce.py --self-test
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
FRESH = {"c4val1": (1, 2, 3, 4, 5, 6, 7, 8), "wikitext": (1, 2, 3, 4)}
MIN_FRESH = {"c4val1": 6, "wikitext": 3}
ARMS = ("g", "m", "t")
PAIRS = (("m", "g"), ("t", "m"), ("t", "g"))
SERVED = "_gemm_nf4_grouped"
GEMV_PREFIX = "_gemv_nf4"
SIGMA_MAX = 0.025                         # half the 0.05 gate: a 2-sigma bound on a single window
GATE = 0.05
#: P94's window-0 K8 ppl (bench/p94/RESULTS-p94.md), the reproduction control -- reported, not gated
P94 = {("granite", "wikitext"): {"g": 5.36059, "m": 5.35515, "t": 5.33507},
       ("granite", "c4val1"): {"g": 11.57477, "m": 11.49663, "t": 11.59831},
       ("olmoe", "wikitext"): {"g": 6.91177, "m": 6.92827, "t": 6.92315},
       ("olmoe", "c4val1"): {"g": 18.90600, "m": 18.89285, "t": 19.06063}}
#: two-sided 95 % Student-t critical values by degrees of freedom
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


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
        f = {"k8": {}, "census_m_b1": None}
        for src, fresh in FRESH.items():
            for k in (0, *fresh):
                for arm in ARMS:
                    r = _json(d / f"{fam}_k8_{arm}_{src}_w{k}.json")
                    if isinstance(r, dict) and "ppl" in r and "mean_nll" in r and "text_sha" in r:
                        f["k8"][(arm, src, k)] = r
        c = d / "logs" / f"census_{fam}_m_b1_d1.txt"
        if c.is_file():
            replays, rows = p42.parse_census(c.read_text())
            if replays:
                calls: dict = {}
                for row in rows:
                    calls[row["name"].strip()] = calls.get(row["name"].strip(), 0) + row["calls"]
                f["census_m_b1"] = {n: k / replays for n, k in calls.items()}
        fams[fam] = f
    return {"families": fams}


def _stats(xs: list) -> dict:
    n = len(xs)
    mean = sum(xs) / n
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "se": se, "values": xs}


def reduce(data: dict) -> dict:
    out: dict = {"lane": "P95", "families": [], "verdict": None, "reasons": [], "under_resolved": []}
    void = []
    for fam, f in data["families"].items():
        L = FAMILIES[fam]
        r: dict = {"family": fam, "texts": {}, "reasons": []}
        cen = f["census_m_b1"]
        if cen is None:
            r["reasons"].append("missing: census m B=1")
        else:
            gemv = sum(v for n, v in cen.items() if n.startswith(GEMV_PREFIX))
            if cen.get(SERVED, 0) != 2 * L or gemv != 0:
                r["reasons"].append(f"engagement m B=1: {SERVED} {cen.get(SERVED, 0):g}/step (want {2 * L}), "
                                    f"NF4 GEMV {gemv:g} (want 0)")
        k8 = f["k8"]
        shas: dict = {}
        for src, fresh in FRESH.items():
            t: dict = {}
            w0 = [a for a in ARMS if (a, src, 0) not in k8]
            if w0:
                r["reasons"].append(f"{src}: window 0 missing {', '.join(w0)}")
            complete = [k for k in (0, *fresh) if all((a, src, k) in k8 for a in ARMS)]
            for k in complete:
                s = {k8[(a, src, k)]["text_sha"] for a in ARMS}
                if len(s) != 1:
                    r["reasons"].append(f"{src} window {k}: the arms scored different text")
                    continue
                sha = s.pop()
                if sha in shas:
                    r["reasons"].append(f"{src} window {k} scored the same text as {shas[sha]}: the windows did not move")
                shas[sha] = f"{src} window {k}"
            fresh_ok = [k for k in complete if k != 0]
            if len(fresh_ok) < MIN_FRESH[src]:
                r["reasons"].append(f"{src}: {len(fresh_ok)} complete fresh windows (want >= {MIN_FRESH[src]})")
            for arm in ("m", "t"):
                if fresh_ok and all(k8[(arm, src, k)]["mean_nll"] == k8[("g", src, k)]["mean_nll"] for k in fresh_ok):
                    r["reasons"].append(f"{src}: {arm} is bit-equal to g in every complete fresh window (the arm did not "
                                        "engage)")
            t["windows"] = fresh_ok
            if 0 in complete:
                got = {a: round(float(k8[(a, src, 0)]["ppl"]), 5) for a in ARMS}
                t["window0"] = got
                t["reproduces_p94"] = got == P94[(fam, src)]
            if len(fresh_ok) >= 2:
                for a, b in PAIRS:
                    t[f"{a}-{b}"] = _stats([float(k8[(a, src, k)]["ppl"]) - float(k8[(b, src, k)]["ppl"]) for k in fresh_ok])
                sigma = max(t["m-g"]["sd"], t["t-m"]["sd"])
                t["sigma"] = sigma
                t["resolved"] = sigma <= SIGMA_MAX
                t["windows_needed"] = 1 if t["resolved"] else math.ceil((sigma / SIGMA_MAX) ** 2)
                tm = t["t-m"]
                t["t-m_consistent_shift"] = abs(tm["mean"]) > T975[tm["n"] - 1] * tm["se"]
            r["texts"][src] = t
        if r["reasons"]:
            void.append(f"{fam} ({'; '.join(r['reasons'])})")
        out["families"].append(r)
    if void:
        out["verdict"] = "VOID"
        out["reasons"].append("VOID: " + " | ".join(void))
        return out
    for r in out["families"]:
        for src, t in r["texts"].items():
            if not t["resolved"]:
                out["under_resolved"].append({"family": r["family"], "text": src, "sigma": t["sigma"],
                                              "windows_needed": t["windows_needed"]})
    if out["under_resolved"]:
        out["verdict"] = "UNDER_RESOLVED"
        out["reasons"].append("per-window spread above 0.025 in: " + ", ".join(
            f"{u['family']} {u['text']} (sigma {u['sigma']:.4f}, a windowed gate needs W = {u['windows_needed']})"
            for u in out["under_resolved"]))
    else:
        out["verdict"] = "RESOLVED"
        out["reasons"].append("per-window spread <= 0.025 on every text in both families: the single-window 0.05 K8 gate "
                              "stands")
    return out


def _synthetic(noise=None, w0=True, fresh=None, drop=None, served_calls=None, same_sha=False, m_equal_g=False,
               shift=0.0, mixed_sha=False):
    """Deterministic per-window deltas: noise[(fam, src)] scales a fixed zero-mean pattern; shift adds to t - m."""
    import itertools
    pattern = [1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25]          # mean 0, SD ~1.0
    noise = noise or {}
    fams = {}
    for fam, L in FAMILIES.items():
        f = {"k8": {}, "census_m_b1": {SERVED: served_calls if served_calls is not None else 2 * L, "void cutlass::...": 9}}
        for (src, ks), base in itertools.product(FRESH.items(), (10.0,)):
            sc = noise.get((fam, src), 0.004)
            for k in (0, *(fresh or {}).get(src, ks)):
                if k == 0 and not w0:
                    continue
                if k == 0:
                    vals = dict(P94[(fam, src)])
                else:
                    p = pattern[(k - 1) % len(pattern)]
                    g = base + 0.01 * k
                    m = g if m_equal_g else g + sc * p
                    vals = {"g": g, "m": m, "t": m + shift + sc * pattern[(k + 2) % len(pattern)]}
                sha = f"sha-{src}" if same_sha else f"sha-{src}-{k}"
                for a in ARMS:
                    f["k8"][(a, src, k)] = {"ppl": vals[a], "mean_nll": vals[a] / 7,
                                            "text_sha": (sha + a) if (mixed_sha and k == 2) else sha}
        fams[fam] = f
    if drop:
        fams[drop[0]]["k8"].pop(drop[1], None)
    return {"families": fams}


def self_test() -> None:
    v = reduce(_synthetic())
    assert v["verdict"] == "RESOLVED", v                                              # small spread everywhere
    t = v["families"][0]["texts"]["c4val1"]
    assert t["reproduces_p94"] and t["windows"] == [1, 2, 3, 4, 5, 6, 7, 8] and not t["t-m_consistent_shift"], t
    v = reduce(_synthetic(noise={("olmoe", "c4val1"): 0.08}))
    assert v["verdict"] == "UNDER_RESOLVED", v                                        # one family-text too wide
    u = v["under_resolved"]
    assert len(u) == 1 and u[0]["family"] == "olmoe" and u[0]["text"] == "c4val1", u
    assert u[0]["windows_needed"] == math.ceil((u[0]["sigma"] / SIGMA_MAX) ** 2) > 1, u
    v = reduce(_synthetic(shift=0.2))                                                  # a consistent K25 shift: reported
    assert v["verdict"] == "RESOLVED" and v["families"][1]["texts"]["c4val1"]["t-m_consistent_shift"], v
    assert reduce(_synthetic(w0=False))["verdict"] == "VOID"                            # the control is missing
    assert reduce(_synthetic(fresh={"c4val1": (1, 2, 3, 4, 5)}))["verdict"] == "VOID"   # 5 < 6 fresh windows
    v = reduce(_synthetic(fresh={"c4val1": (1, 2, 3, 4, 5, 6)}))                        # 6 complete: enough
    assert v["verdict"] == "RESOLVED", v
    assert reduce(_synthetic(drop=("granite", ("t", "wikitext", 3)), fresh={"wikitext": (1, 2, 3)}))["verdict"] == "VOID"
    assert reduce(_synthetic(same_sha=True))["verdict"] == "VOID"                       # the windows did not move
    assert reduce(_synthetic(mixed_sha=True))["verdict"] == "VOID"                      # arms scored different text
    assert reduce(_synthetic(served_calls=0))["verdict"] == "VOID"                      # engagement
    assert reduce(_synthetic(m_equal_g=True))["verdict"] == "VOID"                      # m did not change the arithmetic
    syn = _synthetic()
    syn["families"]["granite"]["k8"][("t", "c4val1", 0)]["ppl"] = 11.6
    v = reduce(syn)
    assert v["verdict"] == "RESOLVED" and not v["families"][0]["texts"]["c4val1"]["reproduces_p94"], v  # reported only
    print("p95_reduce self-test OK (12 cases)")


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
    print(f"P95_VERDICT {v['verdict']} | " + " | ".join(v["reasons"]))
    for r in v["families"]:
        for src, t in r["texts"].items():
            if "sigma" not in t:
                continue
            print(f"  {r['family']} {src}: n={len(t['windows'])} sigma {t['sigma']:.4f} "
                  f"({'resolved' if t['resolved'] else 'UNDER, W=' + str(t['windows_needed'])}) | "
                  + " ".join(f"{p} mean {t[p]['mean']:+.4f} sd {t[p]['sd']:.4f}" for p in ("m-g", "t-m", "t-g"))
                  + f" | t-m shift {t['t-m_consistent_shift']} | w0 reproduces P94 {t.get('reproduces_p94')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
