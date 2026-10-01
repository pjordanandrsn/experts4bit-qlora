#!/usr/bin/env python3
"""Lane P91's reducer (bench/p91/PREREG-p91.md; e4b#564). Where do the NF4 families' decode steps go?

Reads, from the run directory, for each family F in (granite, olmoe) and B in (16, 1):
- e4b_F_bB.json (`step_ms_clean`, the graph-replay window)
- logs/census_F_bB.txt (P42's replay census table, parsed by bench/p42/p42_reduce.py's `parse_census`)

  READ      every arm is present and every census reconciles with its step (kernel ms per step within 10 % of
            step_ms_clean, P86's tolerance)
  NOT_READ  an arm is missing, or a census does not reconcile -- no share is quoted from this run

Reported for every arm: the step, the census total, the NF4 expert kernels' ms per step and share of kernel time
(`_gemm_nf4_grouped` at B=16, the `_gemv_nf4*` decode GEMVs at B=1), and the top kernels. The registered decision rule:
if the NF4 grouped GEMM is at least 40 % of B=16 kernel time in either family, the next lane is the NF4 grouped small-M
kernel; otherwise it is the family's largest non-NF4 kernel family.

    python p91_reduce.py --dir <dir> --out verdict.json
    python p91_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

FAMILIES = ("granite", "olmoe")
BATCHES = (16, 1)
RECONCILE = 0.10
NF4_GROUPED = "_gemm_nf4_grouped"
NF4_GEMV_PREFIX = "_gemv_nf4"
DECISION_SHARE = 0.40
TOP = 8


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
    arms = {}
    for fam in FAMILIES:
        for B in BATCHES:
            r = _json(d / f"e4b_{fam}_b{B}.json")
            step = float(r["step_ms_clean"]) if isinstance(r, dict) and "step_ms_clean" in r else None
            c = d / "logs" / f"census_{fam}_b{B}.txt"
            per = None
            if c.is_file():
                replays, rows = p42.parse_census(c.read_text())
                if replays:
                    agg = {}
                    for row in rows:
                        agg[row["name"].strip()] = agg.get(row["name"].strip(), 0.0) + row["self_ms"]
                    per = {n: ms / replays for n, ms in agg.items()}
            arms[(fam, B)] = {"step": step, "per_step_ms": per}
    return {"arms": arms}


def _arm_report(fam, B, a):
    per = a["per_step_ms"]
    total = sum(per.values())
    if B == 16:
        nf4 = per.get(NF4_GROUPED, 0.0)
    else:
        nf4 = sum(v for n, v in per.items() if n.startswith(NF4_GEMV_PREFIX))
    top = sorted(per.items(), key=lambda kv: -kv[1])[:TOP]
    return {"family": fam, "batch": B, "step_ms": a["step"], "kernel_ms": total, "reconcile": total / a["step"] - 1.0,
            "nf4_expert_ms": nf4, "nf4_expert_share": nf4 / total if total else 0.0,
            "top": [{"kernel": n[:80], "ms": v, "share": v / total} for n, v in top]}


def reduce(data: dict) -> dict:
    v: dict = {"lane": "P91", "verdict": None, "reasons": [], "arms": []}
    for fam in FAMILIES:
        for B in BATCHES:
            a = data["arms"].get((fam, B))
            if not a or a.get("step") is None or not a.get("per_step_ms"):
                v["verdict"] = "NOT_READ"
                v["reasons"].append(f"{fam} B={B}: the step or its census is missing")
                continue
            rep = _arm_report(fam, B, a)
            v["arms"].append(rep)
            if abs(rep["reconcile"]) > RECONCILE:
                v["verdict"] = "NOT_READ"
                v["reasons"].append(f"{fam} B={B}: census {rep['kernel_ms']:.3f} ms vs step {rep['step_ms']:.3f} ms "
                                    f"({rep['reconcile']:+.1%}), outside {RECONCILE:.0%}")
    if v["verdict"] is None:
        v["verdict"] = "READ"
        b16 = {r["family"]: r for r in v["arms"] if r["batch"] == 16}
        share = {f: b16[f]["nf4_expert_share"] for f in b16}
        if max(share.values()) >= DECISION_SHARE:
            v["next_lane"] = "nf4-grouped-small-m-kernel"
            v["reasons"].append("NF4 grouped share at B=16: " + ", ".join(f"{f} {s:.1%}" for f, s in share.items())
                                + f" (>= {DECISION_SHARE:.0%} in at least one family)")
        else:
            best = max(v["arms"], key=lambda r: max((t["ms"] for t in r["top"] if t["kernel"] != NF4_GROUPED), default=0))
            name = next(t["kernel"] for t in best["top"] if t["kernel"] != NF4_GROUPED)
            v["next_lane"] = f"largest-non-nf4: {name}"
            v["reasons"].append("NF4 grouped share at B=16: " + ", ".join(f"{f} {s:.1%}" for f, s in share.items())
                                + f" (< {DECISION_SHARE:.0%} in both)")
    return v


def _synthetic(nf4_16=(5.0, 6.0), other_16=(4.0, 4.0), nf4_1=(1.2, 1.5), other_1=(2.0, 2.2), steps=None, drop=None):
    arms = {}
    for i, fam in enumerate(FAMILIES):
        p16 = {NF4_GROUPED: nf4_16[i], "void cutlass::Kernel2<...>": other_16[i]}
        p1 = {"_gemv_nf4_dotpad": nf4_1[i], "void cutlass::Kernel2<...>": other_1[i]}
        s16 = (steps or {}).get((fam, 16), nf4_16[i] + other_16[i])
        s1 = (steps or {}).get((fam, 1), nf4_1[i] + other_1[i])
        arms[(fam, 16)] = {"step": s16, "per_step_ms": p16}
        arms[(fam, 1)] = {"step": s1, "per_step_ms": p1}
    if drop:
        arms[drop] = {"step": None, "per_step_ms": None}
    return {"arms": arms}


def self_test() -> None:
    got = reduce(_synthetic())
    assert got["verdict"] == "READ" and got["next_lane"] == "nf4-grouped-small-m-kernel", got     # 5/9 = 55.6 %
    g16 = next(r for r in got["arms"] if r["family"] == "granite" and r["batch"] == 16)
    assert abs(g16["nf4_expert_share"] - 5 / 9) < 1e-9 and g16["top"][0]["kernel"] == NF4_GROUPED, g16
    g1 = next(r for r in got["arms"] if r["family"] == "granite" and r["batch"] == 1)
    assert abs(g1["nf4_expert_ms"] - 1.2) < 1e-9, g1                                              # B=1 counts the GEMV
    low = reduce(_synthetic(nf4_16=(2.0, 3.0), other_16=(6.0, 6.0)))
    assert low["verdict"] == "READ" and low["next_lane"].startswith("largest-non-nf4: void cutlass"), low
    assert reduce(_synthetic(nf4_16=(3.6, 1.0), other_16=(5.4, 6.0)))["next_lane"] == "nf4-grouped-small-m-kernel"
    assert reduce(_synthetic(steps={("olmoe", 16): 8.0}))["verdict"] == "NOT_READ"                  # 10/8: +25 %
    assert reduce(_synthetic(steps={("granite", 1): 3.2 * 1.09}))["verdict"] == "READ"              # -8 %: inside
    assert reduce(_synthetic(drop=("olmoe", 1)))["verdict"] == "NOT_READ"
    print("p91_reduce self-test OK (8 cases)")


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
    print(f"P91_VERDICT {v['verdict']} next_lane={v.get('next_lane')} | " + " | ".join(v["reasons"]))
    for r in v["arms"]:
        print(f"  {r['family']} B={r['batch']}: step {r['step_ms']:.3f} ms, census {r['kernel_ms']:.3f} ({r['reconcile']:+.1%}); "
              f"NF4 experts {r['nf4_expert_ms']:.3f} ms ({r['nf4_expert_share']:.1%})")
        for t in r["top"][:5]:
            print(f"      {t['ms']:7.3f} ms {t['share']:6.1%}  {t['kernel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
