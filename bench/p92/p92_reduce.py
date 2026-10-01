#!/usr/bin/env python3
"""Lane P92's reducer (bench/p92/PREREG-p92.md; e4b#564). Does K25 make the NF4 families' B=16 decode faster on an RTX
5090 without moving their K8?

Reads, from the run directory, for each family F in (granite, olmoe):
- e4b_F_{off,on}_b16_d{1,2}.json and e4b_F_{off,on}_b1_d1.json (`step_ms_clean`, the graph-replay window)
- logs/census_F_{off,on}_b16_d1.txt and logs/census_F_{off,on}_b1_d1.txt (P42's replay census table, parsed by
  bench/p42/p42_reduce.py's `parse_census`)
- F_k8_{off,on}_{wikitext,c4val1}.json (K8: `ppl`, `mean_nll`, `text_sha`, `steps`, `ppl_source`)

Per family, in this order:
  VOID          an arm is missing; a B=16 arm's two draws are more than 3 % apart; engagement fails -- at B=16 and at
                B=1 the ON census must run `_gemm_nf4_grouped_smallm` exactly 2 x layers times per step with no
                `_gemm_nf4_grouped` and no `_gemv_nf4*`, and the OFF census must run no `_gemm_nf4_grouped_smallm`; or the
                K8 ON arm is bit-equal to OFF on both texts (the eager loop did not read the kernel)
  QUALITY_FAIL  `experts4bit_qlora.k8_gate.verdict`, uncalibrated (|delta ppl| <= 0.05 on every text), fails
  LICENSED      B=16 ON/OFF (the mean of each arm's two draws) <= 0.95
  NOT_FASTER    otherwise
B=1 ON/OFF is reported beside the verdict and scopes a default (T > 1 only when it is above 1.0).

    python p92_reduce.py --dir <dir> --out verdict.json
    python p92_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

FAMILIES = {"granite": 32, "olmoe": 16}           # MoE layers: two expert GEMMs per layer per step
K25 = "_gemm_nf4_grouped_smallm"
SERVED = "_gemm_nf4_grouped"
GEMV_PREFIX = "_gemv_nf4"
DRAW_SPREAD = 0.03
LICENSE_RATIO = 0.95
TEXTS = ("wikitext", "c4val1")


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
    except ImportError:                                # from the repo checkout, without an install
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from experts4bit_qlora import k8_gate
    return k8_gate


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _census(p42, path: Path):
    """{name: (ms_per_step, calls_per_step)} or None."""
    if not path.is_file():
        return None
    replays, rows = p42.parse_census(path.read_text())
    if not replays:
        return None
    agg: dict = {}
    for row in rows:
        ms, calls = agg.get(row["name"].strip(), (0.0, 0))
        agg[row["name"].strip()] = (ms + row["self_ms"], calls + row["calls"])
    return {n: (ms / replays, calls / replays) for n, (ms, calls) in agg.items()}


def load(d: Path) -> dict:
    p42 = _p42()
    fams = {}
    for fam in FAMILIES:
        f = {"step": {}, "census": {}, "k8": {}}
        for arm in ("off", "on"):
            for B, draws in ((16, (1, 2)), (1, (1,))):
                for D in draws:
                    r = _json(d / f"e4b_{fam}_{arm}_b{B}_d{D}.json")
                    f["step"][(arm, B, D)] = float(r["step_ms_clean"]) if isinstance(r, dict) and "step_ms_clean" in r else None
                f["census"][(arm, B)] = _census(p42, d / "logs" / f"census_{fam}_{arm}_b{B}_d1.txt")
            for src in TEXTS:
                r = _json(d / f"{fam}_k8_{arm}_{src}.json")
                f["k8"][(arm, src)] = r if isinstance(r, dict) and "ppl" in r else None
        fams[fam] = f
    return {"families": fams}


def _calls(c, name):
    return c.get(name, (0.0, 0))[1]


def _gemv_calls(c):
    return sum(v[1] for n, v in c.items() if n.startswith(GEMV_PREFIX))


def reduce_family(fam: str, f: dict, k8_gate) -> dict:
    L = FAMILIES[fam]
    v: dict = {"family": fam, "verdict": None, "reasons": []}
    missing = [f"step {a} B={b} d{dr}" for (a, b, dr), s in f["step"].items() if s is None]
    missing += [f"census {a} B={b}" for (a, b), c in f["census"].items() if c is None]
    missing += [f"k8 {a} {s}" for (a, s), r in f["k8"].items() if r is None]
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append("missing: " + ", ".join(missing))
        return v
    st = f["step"]
    spread = {arm: abs(st[(arm, 16, 1)] / st[(arm, 16, 2)] - 1.0) for arm in ("off", "on")}
    off16 = (st[("off", 16, 1)] + st[("off", 16, 2)]) / 2
    on16 = (st[("on", 16, 1)] + st[("on", 16, 2)]) / 2
    v.update({"b16_off_ms": off16, "b16_on_ms": on16, "b16_ratio": on16 / off16,
              "b1_off_ms": st[("off", 1, 1)], "b1_on_ms": st[("on", 1, 1)],
              "b1_ratio": st[("on", 1, 1)] / st[("off", 1, 1)], "draw_spread": spread})
    cen = f["census"]
    v["census"] = {f"{a}_b{b}": {"k25_ms": c.get(K25, (0.0, 0))[0], "k25_calls": _calls(c, K25),
                                  "served_ms": c.get(SERVED, (0.0, 0))[0], "served_calls": _calls(c, SERVED),
                                  "gemv_calls": _gemv_calls(c), "kernel_ms": sum(x[0] for x in c.values())}
                   for (a, b), c in cen.items()}
    for arm, s in spread.items():
        if s > DRAW_SPREAD:
            v["reasons"].append(f"B=16 {arm} draws {s:.1%} apart (> {DRAW_SPREAD:.0%})")
    for B in (16, 1):
        on, off = cen[("on", B)], cen[("off", B)]
        if _calls(on, K25) != 2 * L or _calls(on, SERVED) != 0 or _gemv_calls(on) != 0:
            v["reasons"].append(f"engagement B={B} ON: {K25} {_calls(on, K25):g}/step (want {2 * L}), {SERVED} "
                                f"{_calls(on, SERVED):g}, NF4 GEMV {_gemv_calls(on):g} (want 0, 0)")
        if _calls(off, K25) != 0:
            v["reasons"].append(f"engagement B={B} OFF: {K25} ran {_calls(off, K25):g}/step")
    k8 = f["k8"]
    if all(k8[("on", s)]["mean_nll"] == k8[("off", s)]["mean_nll"] for s in TEXTS):
        v["reasons"].append("K8 ON is bit-equal to OFF on both texts: the eager loop did not read K25")
    if v["reasons"]:
        v["verdict"] = "VOID"
        return v
    pairs = [(k8_gate.Arm(ppl=float(k8[("off", s)]["ppl"]), text_sha=k8[("off", s)]["text_sha"],
                          steps=int(k8[("off", s)]["steps"]), ppl_source=k8[("off", s)].get("ppl_source", s)),
              k8_gate.Arm(ppl=float(k8[("on", s)]["ppl"]), text_sha=k8[("on", s)]["text_sha"],
                          steps=int(k8[("on", s)]["steps"]), ppl_source=k8[("on", s)].get("ppl_source", s)))
             for s in TEXTS]
    try:
        ok, lines = k8_gate.verdict(pairs, calibrated=False)
    except ValueError as e:                           # different text or step counts: the arms do not compare
        v["verdict"] = "VOID"
        v["reasons"].append(f"K8 arms do not compare: {e}")
        return v
    v["k8"] = {s: {"off": float(k8[("off", s)]["ppl"]), "on": float(k8[("on", s)]["ppl"]),
                   "delta": float(k8[("on", s)]["ppl"]) - float(k8[("off", s)]["ppl"])} for s in TEXTS}
    v["k8_lines"] = lines
    if not ok:
        v["verdict"] = "QUALITY_FAIL"
        v["reasons"].append("K8 (uncalibrated, |delta| <= 0.05 ppl on every text) failed")
    elif v["b16_ratio"] <= LICENSE_RATIO:
        v["verdict"] = "LICENSED"
        v["reasons"].append(f"B=16 ON/OFF {v['b16_ratio']:.3f} <= {LICENSE_RATIO}; K8 passed")
    else:
        v["verdict"] = "NOT_FASTER"
        v["reasons"].append(f"B=16 ON/OFF {v['b16_ratio']:.3f} > {LICENSE_RATIO}; K8 passed")
    return v


def reduce(data: dict, k8_gate=None) -> dict:
    k8_gate = k8_gate or _k8_gate()
    fams = [reduce_family(fam, f, k8_gate) for fam, f in data["families"].items()]
    return {"lane": "P92", "families": fams,
            "verdict": " ".join(f"{r['family']}={r['verdict']}" for r in fams)}


# --------------------------------------------------------------------------------------------- self-test --
def _synthetic(ratio=0.75, b1=1.05, spread=0.0, dk8=0.004, drop=None, engaged=True, k8_bit_equal=False):
    fams = {}
    for fam, L in FAMILIES.items():
        f = {"step": {}, "census": {}, "k8": {}}
        for arm in ("off", "on"):
            base16 = 10.0 * (ratio if arm == "on" else 1.0)
            f["step"][(arm, 16, 1)] = base16
            f["step"][(arm, 16, 2)] = base16 * (1 + spread)
            f["step"][(arm, 1, 1)] = 3.0 * (b1 if arm == "on" else 1.0)
            for B in (16, 1):
                if arm == "on" and engaged:
                    c = {K25: (4.0, 2 * L), "void cutlass::Kernel2<...>": (1.0, 64)}
                elif arm == "on":
                    c = {SERVED: (6.0, 2 * L), "void cutlass::Kernel2<...>": (1.0, 64)}
                else:
                    c = {SERVED if B == 16 else "_gemv_nf4_dotpad": (6.0, 2 * L), "void cutlass::Kernel2<...>": (1.0, 64)}
                f["census"][(arm, B)] = c
            for src, base in (("wikitext", 9.0), ("c4val1", 14.0)):
                d = 0.0 if (arm == "off" or k8_bit_equal) else dk8
                f["k8"][(arm, src)] = {"ppl": base + d, "mean_nll": 2.0 + d, "text_sha": f"sha-{src}", "steps": 2048,
                                       "ppl_source": src}
        fams[fam] = f
    if drop:
        fam, kind, key = drop
        fams[fam][kind][key] = None
    return {"families": fams}


def self_test() -> None:
    gate = _k8_gate()
    got = reduce(_synthetic(), gate)
    assert got["verdict"] == "granite=LICENSED olmoe=LICENSED", got
    g = got["families"][0]
    assert abs(g["b16_ratio"] - 0.75) < 1e-12 and abs(g["b1_ratio"] - 1.05) < 1e-12, g             # B=1 reported
    assert reduce(_synthetic(ratio=0.95), gate)["families"][0]["verdict"] == "LICENSED"            # the boundary
    assert reduce(_synthetic(ratio=0.97), gate)["families"][0]["verdict"] == "NOT_FASTER"
    assert reduce(_synthetic(dk8=0.06), gate)["families"][1]["verdict"] == "QUALITY_FAIL"          # |delta| > 0.05
    assert reduce(_synthetic(dk8=-0.06), gate)["families"][1]["verdict"] == "QUALITY_FAIL"         # two-sided
    assert reduce(_synthetic(spread=0.04), gate)["families"][0]["verdict"] == "VOID"               # draws 4 % apart
    assert reduce(_synthetic(spread=0.02), gate)["families"][0]["verdict"] == "LICENSED"           # 2 %: inside
    assert reduce(_synthetic(engaged=False), gate)["families"][0]["verdict"] == "VOID"             # K25 never ran
    assert reduce(_synthetic(k8_bit_equal=True), gate)["families"][0]["verdict"] == "VOID"         # K8 not engaged
    v = reduce(_synthetic(drop=("olmoe", "k8", ("on", "c4val1"))), gate)
    assert v["verdict"] == "granite=LICENSED olmoe=VOID", v                                        # per family
    assert reduce(_synthetic(drop=("granite", "census", ("off", 1))), gate)["families"][0]["verdict"] == "VOID"
    bad = _synthetic()
    bad["families"]["granite"]["census"][("on", 16)][K25] = (4.0, 2 * 32 - 2)                      # a layer missed
    assert reduce(bad, gate)["families"][0]["verdict"] == "VOID"
    bad = _synthetic()
    bad["families"]["olmoe"]["k8"][("on", "wikitext")]["text_sha"] = "other"                       # different text
    assert reduce(bad, gate)["families"][1]["verdict"] == "VOID"
    print("p92_reduce self-test OK (14 cases)")


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
    print(f"P92_VERDICT {v['verdict']}")
    for r in v["families"]:
        print(f"  {r['family']}: {r['verdict']} | " + " | ".join(r["reasons"]))
        if "b16_ratio" in r:
            print(f"    B=16 OFF {r['b16_off_ms']:.3f} ON {r['b16_on_ms']:.3f} ms (x{r['b16_ratio']:.3f}); "
                  f"B=1 OFF {r['b1_off_ms']:.3f} ON {r['b1_on_ms']:.3f} (x{r['b1_ratio']:.3f})")
        for s, k in (r.get("k8") or {}).items():
            print(f"    K8 {s}: OFF {k['off']:.5f} ON {k['on']:.5f} (delta {k['delta']:+.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
