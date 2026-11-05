"""PGX reducer: the registered rule, applied to one box's (or a split's merged) records. Fixed before any data.

    python bench/pgx/pgx_reduce.py --work DIR [DIR...] --out verdict_pgx.json

Each DIR holds ``<arm>/<budget>/d<draw>/lifetime.json`` plus that lifetime's pass records (``P1.json`` .. ``P4.json``,
``pgx_drive.py``'s output). ``lifetime.json`` is the box's provenance for one server lifetime:

    {"arm", "budget", "draw", "k_registered", "k_measured", "counters_ok", "pager_giveups",
     "vram_tier_bytes", "nvme_experts", "profile": {"kind", "sha256"}, "dram_tier_s_per_token": {"P2": ...}}

Exit 0 when the verdict is READ, 27 otherwise (P238's codes). Stdlib only; self-tested by tests/test_pgx_reduce.py.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ARMS = ("S4", "S8", "V8", "E-hyb")
BASELINES = ("S4", "S8", "V8")
BUDGETS = (4, 8, 12)
DRAWS = (1, 2, 3)
# control lifetimes: (arm, budget key, passes)
CONTROLS = (("E-hyb-u", 8, ("P1", "P2", "P3", "P4")), ("E-allvram-g", "all", ("P1", "P2", "P3")),
            ("E-allvram-e", "all", ("P1", "P2", "P3")))
STABLE = 1.03                      # max/min over the three draws (P235's self-pair rule, applied to three)
SHARE = 0.5                        # an attribution branch is named at >= 50 % of the gap
TOKEN_RULED = {"S8", "V8", "E-hyb", "E-hyb-u", "E-allvram-g", "E-allvram-e"}   # P2 vs P1 identity rules; S4 reported


def registered_passes(budget) -> tuple:
    return ("P1", "P2", "P3", "P4") if budget == 8 else ("P1", "P2", "P3")


def schedule() -> list:
    """Every registered (arm, budget, draw, pass): the matched arms, then the controls."""
    out = [(a, b, d, p) for a in ARMS for b in BUDGETS for d in DRAWS for p in registered_passes(b)]
    out += [(a, b, d, p) for a, b, ps in CONTROLS for d in DRAWS for p in ps]
    return out


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def lifetime_faults(lt: dict, reg: dict) -> list:
    """Faults that VOID every pass of a lifetime. ``reg`` is the registration: {"profile_sha", "budget_bytes"}."""
    f = []
    arm = lt.get("arm")
    if lt.get("k_registered") is not None and lt.get("k_measured") != lt.get("k_registered"):
        f.append(f"K measured {lt.get('k_measured')} != registered {lt.get('k_registered')}")
    if not lt.get("counters_ok"):
        f.append("counter dump missing")
    if arm == "V8" and (lt.get("pager_giveups") or 0) > 0:
        f.append(f"pager give-ups {lt['pager_giveups']}")
    if arm in ("E-hyb", "E-hyb-u"):
        if lt.get("nvme_experts") != 0:
            f.append(f"NVMe-tier experts {lt.get('nvme_experts')} != 0")
        cap = reg["budget_bytes"].get(str(lt.get("budget")))
        if cap is None or lt.get("vram_tier_bytes") is None or lt["vram_tier_bytes"] > cap:
            f.append(f"VRAM-tier bytes {lt.get('vram_tier_bytes')} over the budget {cap}")
        prof = lt.get("profile") or {}
        want = ("measured", reg["profile_sha"]) if arm == "E-hyb" else ("uniform-assumed", None)
        if (prof.get("kind"), prof.get("sha256")) != want:
            f.append(f"placement profile {prof.get('kind')}/{str(prof.get('sha256'))[:12]} != {want[0]}")
    return f


def pass_faults(rec, want_sha: str) -> list:
    if rec is None:
        return ["record missing"]
    f = []
    if rec.get("faults"):
        f.append(f"{rec['faults']} request fault(s)")
    if rec.get("prompts_sha") != want_sha:
        f.append("prompts_sha differs from the registered set")
    return f


def reduce(dirs, reg: dict) -> dict:
    """reg: {"profile_sha", "budget_bytes": {"4": int, ...}, "sets": {"warm": sha, "shift": sha}}."""
    rows, lifetimes = {}, {}
    for d in dirs:
        for arm in sorted(os.listdir(d)):
            for bud in sorted(os.listdir(os.path.join(d, arm))):
                for dr in sorted(os.listdir(os.path.join(d, arm, bud))):
                    base = os.path.join(d, arm, bud, dr)
                    key = (arm, int(bud) if bud.isdigit() else bud, int(dr.lstrip("d")))
                    if key in lifetimes:
                        lifetimes[key] = {"duplicate": True}
                        continue
                    lifetimes[key] = {"lt": _load(os.path.join(base, "lifetime.json")), "dir": base}
    findings = []
    for (arm, bud, dr, p) in schedule():
        L = lifetimes.get((arm, bud, dr))
        faults, lt_dram = [], None
        if L is None or L.get("duplicate") or L.get("lt") is None:
            faults.append("lifetime missing or duplicated")
            rec = None
        else:
            faults += lifetime_faults(L["lt"], reg)
            rec = _load(os.path.join(L["dir"], f"{p}.json"))
            faults += pass_faults(rec, reg["sets"]["shift" if p == "P4" else "warm"])
            lt_dram = (L["lt"].get("dram_tier_s_per_token") or {}).get(p)
        rows[(arm, bud, dr, p)] = {"faults": faults, "rec": rec, "lt_dram": lt_dram}
    # token invariance within a lifetime: P2 must reproduce P1's greedy text, request by request
    for (arm, bud, dr) in {(a, b, d) for a, b, d, _ in rows}:
        p1, p2 = rows.get((arm, bud, dr, "P1")), rows.get((arm, bud, dr, "P2"))
        if not (p1 and p2 and p1["rec"] and p2["rec"]):
            continue
        same = [a["text_sha"] == b["text_sha"] for a, b in zip(p1["rec"]["requests"], p2["rec"]["requests"])]
        if all(same):
            continue
        findings.append({"kind": "token_invariance", "arm": arm, "budget": bud, "draw": dr,
                         "differing_requests": same.count(False), "rules": arm in TOKEN_RULED})
        if arm in TOKEN_RULED:
            for p in ("P1", "P2", "P3", "P4"):
                if (arm, bud, dr, p) in rows:
                    rows[(arm, bud, dr, p)]["faults"].append("P2 differs from P1 (token invariance)")
    # cross-budget identity for the Triton arms: reported only
    for arm in ("S8", "V8"):
        for dr in DRAWS:
            shas = {b: [r["text_sha"] for r in rows[(arm, b, dr, "P2")]["rec"]["requests"]]
                    for b in BUDGETS if rows[(arm, b, dr, "P2")]["rec"]}
            if len({tuple(v) for v in shas.values()}) > 1:
                findings.append({"kind": "cross_budget_tokens", "arm": arm, "draw": dr, "rules": False})
    # cells: (arm, budget, pass) over the three draws
    cells = {}
    for (arm, bud, dr, p), r in rows.items():
        c = cells.setdefault((arm, bud, p), {"draws": {}, "void": []})
        if r["faults"]:
            c["void"].append({"draw": dr, "faults": r["faults"]})
        else:
            c["draws"][dr] = {k: r["rec"][k] for k in ("tpot_p50", "tpot_p99", "ttft_p50", "ttft_p99", "out_tok_s")}
    for c in cells.values():
        if c["void"]:
            c["state"] = "VOID"
            continue
        tp = [v["tpot_p50"] for v in c["draws"].values()]
        ts = [v["out_tok_s"] for v in c["draws"].values()]
        spread = max(max(tp) / min(tp), max(ts) / min(ts)) if min(tp) > 0 and min(ts) > 0 else float("inf")
        c["spread"] = spread
        c["state"] = "READ" if spread <= STABLE else "UNSTABLE"
        c["median"] = {k: statistics.median(v[k] for v in c["draws"].values()) for k in
                       ("tpot_p50", "tpot_p99", "ttft_p50", "ttft_p99", "out_tok_s")}
    readings = []
    for bud in BUDGETS:
        for p in ("P2", "P3"):
            e = cells[("E-hyb", bud, p)]
            row = {"budget": bud, "pass": p, "E-hyb": e["state"], "ratios": {}}
            for b in BASELINES:
                c = cells[(b, bud, p)]
                if e["state"] == "READ" and c["state"] == "READ":
                    row["ratios"][b] = {"tpot": e["median"]["tpot_p50"] / c["median"]["tpot_p50"],
                                        "tok_s": e["median"]["out_tok_s"] / c["median"]["out_tok_s"]}
            readings.append(row)
    attribution = [attribute(cells, rows, bud, p) for bud in BUDGETS for p in ("P2", "P3")]
    void = sorted({k[:3] for k, r in rows.items() if r["faults"]}, key=str)
    verdict = "READ" if not any(c["state"] == "VOID" for c in cells.values()) else "VOID"
    return {"verdict": verdict, "cells": {"|".join(map(str, k)): {kk: vv for kk, vv in v.items()}
                                          for k, v in sorted(cells.items(), key=lambda kv: str(kv[0]))},
            "readings": readings, "attribution": attribution, "findings": findings,
            "void_lifetimes": [list(v) for v in void]}


def attribute(cells, rows, bud, p) -> dict:
    """The registered attribution of an E-hyb loss at (budget, pass): eager tax, DRAM-tier host time, placement."""
    out = {"budget": bud, "pass": p, "branches": [], "shares": {}}
    e = cells[("E-hyb", bud, p)]
    base = [cells[(b, bud, p)] for b in BASELINES if cells[(b, bud, p)]["state"] == "READ"]
    if e["state"] != "READ" or not base:
        out["state"] = "NOT_READ"
        return out
    best = min(c["median"]["tpot_p50"] for c in base)
    gap = e["median"]["tpot_p50"] - best
    if gap <= 0:
        out["state"] = "NO_LOSS"
        return out
    out["state"] = "LOSS"
    g, ea = cells[("E-allvram-g", "all", p)], cells[("E-allvram-e", "all", p)]
    if g["state"] == "READ" and ea["state"] == "READ":
        out["shares"]["eager_tax"] = (ea["median"]["tpot_p50"] - g["median"]["tpot_p50"]) / gap
    host = [rows[("E-hyb", bud, d, p)]["lt_dram"] for d in DRAWS
            if rows[("E-hyb", bud, d, p)]["lt_dram"] is not None]
    if host:
        out["shares"]["dram_host"] = statistics.median(host) / gap
    if bud == 8:
        u = cells[("E-hyb-u", 8, p)]
        if u["state"] == "READ" and u["median"]["tpot_p50"] - best > 0:
            out["shares"]["placement"] = (u["median"]["tpot_p50"] - e["median"]["tpot_p50"]) / (
                u["median"]["tpot_p50"] - best)
    names = {"eager_tax": "capture-legal hybrid tier", "dram_host": "lean prefill / GPU paging",
             "placement": "a better profile (placement quality)"}
    out["branches"] = [names[k] for k, v in out["shares"].items() if v >= SHARE]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", nargs="+", required=True)
    ap.add_argument("--registration", required=True, help="JSON: profile_sha, budget_bytes, sets")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.registration, encoding="utf-8") as f:
        reg = json.load(f)
    v = reduce(a.work, reg)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1, default=str)
    print(f"verdict {v['verdict']}; {len(v['void_lifetimes'])} void lifetime(s); {len(v['findings'])} finding(s)")
    return 0 if v["verdict"] == "READ" else 27


if __name__ == "__main__":
    sys.exit(main())
