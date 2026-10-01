#!/usr/bin/env python3
"""Lane P94's reducer (bench/p94/PREREG-p94.md; e4b#564). K8 against the arithmetic a T > 1 route replaces.

Lanes P92 and P93 read K25 at T == 1 against the scalar decode GEMV, but the batched rows K25 serves under ``auto`` run
the served NF4 M-tile kernel (TF32) today. P94 reads K8 at T == 1 under three arithmetics, per family and text:
  g  the scalar fp32 decode GEMV        (E4B_NF4_GROUPED_SMALLM=0, E4B_NF4_T1_DEVICE_GROUPING=0) -- P93's OFF
  m  the served NF4 M-tile kernel        (E4B_NF4_GROUPED_SMALLM=0, E4B_NF4_T1_DEVICE_GROUPING=1) -- B=16's arithmetic
  t  K25, the select tree through TF32   (E4B_NF4_GROUPED_SMALLM=1)                                -- P93's ON

Reads, per family F in (granite, olmoe): F_k8_{g,m,t}_{wikitext,c4val1}.json, and logs/census_F_m_b1_d1.txt (engagement:
the m arm must run ``_gemm_nf4_grouped`` 2 x layers per step at T == 1 and no ``_gemv_nf4*``).

  VOID          an arm is missing; m's engagement fails; m or t is bit-equal to g on both texts of a family
  LICENSED      ``experts4bit_qlora.k8_gate.verdict`` (uncalibrated, |delta| <= 0.05 ppl per text) passes with base = m and
                candidate = t, in BOTH families -- with P93's speed (both families B=16 <= 0.95), the T > 1 default moves
  QUALITY_FAIL  that gate fails in either family
Reported, not gated: m - g and t - g per text (whether the GEMV-to-tile difference alone exceeds 0.05: BASELINE_SHIFT).

    python p94_reduce.py --dir <dir> --out verdict.json
    python p94_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

FAMILIES = {"granite": 32, "olmoe": 16}
TEXTS = ("wikitext", "c4val1")
ARMS = ("g", "m", "t")
SERVED = "_gemm_nf4_grouped"
GEMV_PREFIX = "_gemv_nf4"
GATE = 0.05


def _p42():
    here = Path(__file__).resolve().parent
    for cand in (here / "p42_reduce.py", here.parent / "p42" / "p42_reduce.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("p42_reduce", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("p42_reduce.py not found beside this reducer or in bench/p42")


def _k8_gate():
    try:
        from experts4bit_qlora import k8_gate
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from experts4bit_qlora import k8_gate
    return k8_gate


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
        for arm in ARMS:
            for src in TEXTS:
                r = _json(d / f"{fam}_k8_{arm}_{src}.json")
                f["k8"][(arm, src)] = r if isinstance(r, dict) and "ppl" in r else None
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


def _arm(k8_gate, r, src):
    return k8_gate.Arm(ppl=float(r["ppl"]), text_sha=r["text_sha"], steps=int(r["steps"]), ppl_source=r.get("ppl_source", src))


def reduce(data: dict, k8_gate=None) -> dict:
    k8_gate = k8_gate or _k8_gate()
    out: dict = {"lane": "P94", "families": [], "verdict": None, "reasons": []}
    passes = {}
    for fam, f in data["families"].items():
        L = FAMILIES[fam]
        r: dict = {"family": fam, "reasons": []}
        missing = [f"k8 {a} {s}" for (a, s), v in f["k8"].items() if v is None]
        if f["census_m_b1"] is None:
            missing.append("census m B=1")
        if missing:
            r["verdict"] = "VOID"
            r["reasons"].append("missing: " + ", ".join(missing))
            out["families"].append(r)
            continue
        cen = f["census_m_b1"]
        gemv = sum(v for n, v in cen.items() if n.startswith(GEMV_PREFIX))
        if cen.get(SERVED, 0) != 2 * L or gemv != 0:
            r["reasons"].append(f"engagement m B=1: {SERVED} {cen.get(SERVED, 0):g}/step (want {2 * L}), NF4 GEMV {gemv:g} (want 0)")
        k8 = f["k8"]
        for arm in ("m", "t"):
            if all(k8[(arm, s)]["mean_nll"] == k8[("g", s)]["mean_nll"] for s in TEXTS):
                r["reasons"].append(f"K8 {arm} is bit-equal to g on both texts: the arm did not change the arithmetic")
        r["ppl"] = {s: {a: float(k8[(a, s)]["ppl"]) for a in ARMS} for s in TEXTS}
        r["delta"] = {s: {"m-g": r["ppl"][s]["m"] - r["ppl"][s]["g"], "t-g": r["ppl"][s]["t"] - r["ppl"][s]["g"],
                          "t-m": r["ppl"][s]["t"] - r["ppl"][s]["m"]} for s in TEXTS}
        r["baseline_shift"] = any(abs(r["delta"][s]["m-g"]) > GATE for s in TEXTS)
        if r["reasons"]:
            r["verdict"] = "VOID"
            out["families"].append(r)
            continue
        try:
            ok, lines = k8_gate.verdict([(_arm(k8_gate, k8[("m", s)], s), _arm(k8_gate, k8[("t", s)], s)) for s in TEXTS],
                                        calibrated=False)
        except ValueError as e:
            r["verdict"] = "VOID"
            r["reasons"].append(f"K8 arms do not compare: {e}")
            out["families"].append(r)
            continue
        r["gate_lines"] = lines
        r["verdict"] = "PASS" if ok else "FAIL"
        passes[fam] = ok
        out["families"].append(r)
    if any(r["verdict"] == "VOID" for r in out["families"]):
        out["verdict"] = "VOID"
        out["reasons"].append("a family is VOID: " + ", ".join(f"{r['family']} ({'; '.join(r['reasons'])})"
                                                                for r in out["families"] if r["verdict"] == "VOID"))
    elif all(passes.values()):
        out["verdict"] = "LICENSED"
        out["reasons"].append(f"K8 of K25 (t) against the served M-tile arithmetic (m) within {GATE} ppl on every text in both "
                              "families; with P93's speed, the T > 1 default moves")
    else:
        out["verdict"] = "QUALITY_FAIL"
        out["reasons"].append("K8 of t against m outside the gate in: " + ", ".join(f for f, ok in passes.items() if not ok))
    return out


def _synthetic(tm=0.004, mg=0.15, drop=None, served_calls=None, m_equal_g=False):
    fams = {}
    for fam, L in FAMILIES.items():
        f = {"k8": {}, "census_m_b1": {SERVED: served_calls if served_calls is not None else 2 * L, "void cutlass::...": 64}}
        for src, base in (("wikitext", 7.0), ("c4val1", 19.0)):
            g = base
            m = g if m_equal_g else g + mg
            t = m + tm
            for arm, v in (("g", g), ("m", m), ("t", t)):
                f["k8"][(arm, src)] = {"ppl": v, "mean_nll": v / 10, "text_sha": f"sha-{src}", "steps": 2048, "ppl_source": src}
        fams[fam] = f
    if drop:
        fams[drop[0]]["k8"][drop[1]] = None
    return {"families": fams}


def self_test() -> None:
    gate = _k8_gate()
    v = reduce(_synthetic(), gate)
    assert v["verdict"] == "LICENSED", v                                             # t ~ m, m far from g: licensed
    assert all(r["baseline_shift"] for r in v["families"]), v                       # the GEMV->tile shift is reported
    assert reduce(_synthetic(tm=0.049), gate)["verdict"] == "LICENSED"              # inside (0.05 is not exact in binary)
    assert reduce(_synthetic(tm=0.06), gate)["verdict"] == "QUALITY_FAIL"
    assert reduce(_synthetic(tm=-0.06), gate)["verdict"] == "QUALITY_FAIL"          # two-sided
    v = reduce(_synthetic(mg=0.01), gate)                                           # a small m - g: no BASELINE_SHIFT
    assert v["verdict"] == "LICENSED" and not v["families"][0]["baseline_shift"], v
    assert reduce(_synthetic(m_equal_g=True), gate)["verdict"] == "VOID"            # m did not engage
    assert reduce(_synthetic(served_calls=0), gate)["verdict"] == "VOID"            # census engagement
    assert reduce(_synthetic(drop=("olmoe", ("t", "c4val1"))), gate)["verdict"] == "VOID"
    bad = _synthetic()
    bad["families"]["granite"]["k8"][("t", "wikitext")]["text_sha"] = "other"
    assert reduce(bad, gate)["verdict"] == "VOID"                                   # different text
    print("p94_reduce self-test OK (10 cases)")


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
    print(f"P94_VERDICT {v['verdict']} | " + " | ".join(v["reasons"]))
    for r in v["families"]:
        print(f"  {r['family']}: {r['verdict']}" + (" | " + "; ".join(r["reasons"]) if r["reasons"] else ""))
        for s, d in (r.get("delta") or {}).items():
            p = r["ppl"][s]
            print(f"    {s}: g {p['g']:.5f} m {p['m']:.5f} t {p['t']:.5f} | m-g {d['m-g']:+.5f} t-g {d['t-g']:+.5f} t-m {d['t-m']:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
