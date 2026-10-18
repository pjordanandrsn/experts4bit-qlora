#!/usr/bin/env python3
"""pername_reduce.py -- the per-name census's rule (bench/pername/PREREG-pername.md; e4b#1313).

VOID   anything P136's reducer VOIDs (imported at its pinned bytes, with Qwen3-30B-A3B registered as a W1-only family:
       the source pins, the self-pair, the 32-skip / 64-step window, SC1b's blocking labels, the class terms closing the
       step), or the per-name table missing, not ok, not closing every replay exactly, or without the expert path's three
       lines at 48 launches per step (gate_up GEMV, _swiglu_rows, down GEMV).
READ   otherwise, with these lines, every share over the node replay's own period P:
  swiglu   removable = gap_before + excl of _swiglu_rows. P2 is the swiglu-into-down fusion read iff it is >= 3 % of P.
  gemv     each expert GEMV's removable time against K33's isolated time for the same 48 launches (gnf4
           kernel/RESULTS-k33-nf4-decode-gemv-bw.md, bw_prmt32: gate_up 0.6068 ms, down 0.3540 ms): the in-situ excess.
  ranked   every line by addressable time = removable - its floor (K33 for the two expert GEMVs, 0 otherwise);
           dense_gemm is listed and never ranked (closed for B=1 on e4b#1313, 2026-10-18).
  p2       the swiglu fusion when its line clears 3 %; otherwise the top ranked line, PROPOSED to the maintainer with
           this table (a retarget is a new registration, never automatic).

    python pername_reduce.py --dir /root/pername --out verdict_pername.json --e4b-sha <sha>
    python pername_reduce.py --self-test
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p136_reduce as pr  # noqa: E402  (staged at P136's pinned bytes)

FAMILY, MODEL, REV, LAYERS = "qwen3", "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", 48
pr.FAMILIES[FAMILY] = (MODEL, REV, (1,))
pr.MOE_LAYERS[FAMILY] = LAYERS
SWIGLU, GEMV = "_swiglu_rows", "_gemv_nf4_bw"
K33_MS = {"gate_up": 0.6068, "down": 0.3540}          # isolated, 48 launches each, bw_prmt32 on an RTX 5090 (K33)
P2_BAR = 0.03
CLOSED = {"dense_gemm"}


def _load(root, name):
    try:
        return json.loads((Path(root) / name).read_text())
    except (OSError, ValueError):
        return None


def table_faults(tab) -> list:
    if not tab:
        return ["per-name table missing"]
    if tab.get("status") != "ok":
        return [f"per-name table {tab.get('status')}: {tab.get('why')}"]
    faults = []
    if tab.get("closure_ns_max") != 0:
        faults.append(f"per-name table does not close a replay (max {tab.get('closure_ns_max')} ns)")
    if (tab.get("steps_kept") or 0) < 56:
        faults.append(f"per-name table kept {tab.get('steps_kept')} steps < 56")
    n = {}
    for r in tab.get("rows") or []:
        line = "swiglu" if r["name"] == SWIGLU else r.get("role") if r["name"] == GEMV else None
        if line:
            n[line] = n.get(line, 0) + r["n_per_step"]
    for line in ("gate_up", "swiglu", "down"):
        if n.get(line) != LAYERS:
            faults.append(f"expert path: {line} at {n.get(line)} launches per step, not {LAYERS}")
    if n.get("other"):
        faults.append(f"an expert GEMV with no swiglu beside it ({n['other']} per step)")
    return faults


def decide(tab) -> dict:
    p = tab["P_ms"]
    lines = []
    for r in tab["rows"]:
        floor = K33_MS.get(r.get("role"), 0.0) if r["name"] == GEMV else 0.0
        addr = max(0.0, r["removable_ms"] - floor)
        lines.append({"key": r["key"], "class": r["class"], "n_per_step": r["n_per_step"], "dur_ms": r["dur_ms"],
                      "billed_ms": r["billed_ms"], "removable_ms": r["removable_ms"], "floor_ms": floor,
                      "addressable_ms": round(addr, 6), "share": round(r["removable_ms"] / p, 5),
                      "addressable_share": round(addr / p, 5), "closed": r["class"] in CLOSED})
    sw = round(sum(x["removable_ms"] for x in lines if x["key"].startswith(SWIGLU + "|")), 6)
    gemv = {x["key"].rsplit("|", 1)[1]: {"removable_ms": x["removable_ms"], "k33_ms": x["floor_ms"],
                                         "in_situ_excess_ms": round(x["removable_ms"] - x["floor_ms"], 6),
                                         "in_situ_excess_share": round((x["removable_ms"] - x["floor_ms"]) / p, 5)}
            for x in lines if x["key"].startswith(GEMV + "|")}
    ranked = sorted((x for x in lines if not x["closed"]), key=lambda x: -x["addressable_ms"])
    if sw / p >= P2_BAR:
        p2 = {"target": "swiglu-into-down fusion (P0)", "why": f"swiglu removable {sw:.4f} ms = {sw / p:.2%} of P >= 3 %"}
    else:
        top = ranked[0] if ranked else None
        p2 = {"target": f"PROPOSE {top['key']}" if top else None,
              "why": f"swiglu removable {sw:.4f} ms = {sw / p:.2%} of P < 3 %; top addressable line "
                     f"{top['key'] if top else None} ({top['addressable_ms'] if top else 0:.4f} ms)"}
    return {"P_ms": p, "swiglu": {"removable_ms": sw, "share": round(sw / p, 5), "clears_3pct": sw / p >= P2_BAR},
            "gemv": gemv, "p2": p2, "ranked": ranked[:16],
            "closed": [x for x in lines if x["closed"]][:4], "class_removable_ms": tab.get("class_removable_ms")}


def assess(root, e4b_sha) -> dict:
    base = pr.assess(Path(root), FAMILY, e4b_sha)
    tab = _load(root, "pername_b1.json")
    faults = list(base["why"]) + table_faults(tab)
    rec = {"lane": "pername", "family": FAMILY, "model": MODEL, "revision": REV, "e4b_sha": e4b_sha,
           "gnf4_sha": pr.GNF4_SHA, "verdict": "READ" if not faults else "VOID", "why": faults,
           "census": base.get("rows")}
    if not faults:
        rec["decision"] = decide(tab)
    return rec


# ---------------------------------------------------------------------------------------------------------- self-test --
def _row(key, cls, removable, n=48, dur=None, billed=0.0):
    name, grid, *role = key.split("|")
    return {"key": key, "name": name, "grid": grid, "role": role[0] if role else None, "class": cls, "n_per_step": n,
            "dur_ms": dur if dur is not None else removable, "excl_ms": removable, "billed_ms": billed,
            "gap_before_ms": 0.0, "removable_ms": removable}


def _tab(sw=0.13, sw_dur=None, gate_up=0.70, down=0.40, attn=0.20, dense=1.80):
    return {"status": "ok", "steps_kept": 61, "closure_ns_max": 0, "P_ms": 4.0, "class_removable_ms": {},
            "rows": [_row("kernel|1x1x1", "dense_gemm", dense, n=145), _row(f"{GEMV}|96x8x1|gate_up", "moe_expert", gate_up),
                     _row(f"{GEMV}|128x8x1|down", "moe_expert", down), _row(f"{SWIGLU}|8x2x1", "moe_expert", sw, dur=sw_dur),
                     _row("_fp8_append_bt1_side|4x1x1", "attn", attn, n=96)]}


def self_test() -> int:
    cases = []
    with tempfile.TemporaryDirectory() as d:
        r = assess(d, "a" * 40)
        cases.append(("missing receipts VOID", r["verdict"] == "VOID" and "per-name table missing" in r["why"]))
    d1 = decide(_tab(sw=0.13))
    cases.append(("swiglu at 3.25 % targets the fusion", d1["p2"]["target"].startswith("swiglu") and d1["swiglu"]["clears_3pct"]))
    d2 = decide(_tab(sw=0.11))
    # gate_up removes 0.70 ms but K33 isolates it at 0.6068: addressable 0.093 < attn's 0.20; dense (1.80) is closed
    cases.append(("under 3 %, the top ADDRESSABLE open line is proposed", d2["p2"]["target"] == "PROPOSE _fp8_append_bt1_side|4x1x1"))
    cases.append(("the expert GEMVs' in-situ excess is over K33", abs(d2["gemv"]["gate_up"]["in_situ_excess_ms"] - 0.0932) < 1e-9
                  and abs(d2["gemv"]["down"]["in_situ_excess_ms"] - 0.046) < 1e-9))
    cases.append(("dense_gemm is never ranked", all(not x["closed"] for x in d2["ranked"]) and [x["key"] for x in d2["closed"]] == ["kernel|1x1x1"]))
    # a swiglu billed 5 % by its duration but removing 2 %: the rule reads removable time, so no fusion read
    d3 = decide(_tab(sw=0.08, sw_dur=0.20))
    cases.append(("duration is not removable time", not d3["swiglu"]["clears_3pct"]))
    ok = _tab()
    cases.append(("a complete table has no faults", table_faults(ok) == []))
    bad = _tab()
    bad["closure_ns_max"] = 3
    cases.append(("a replay that does not close is a fault", any("close" in f for f in table_faults(bad))))
    miss = _tab()
    miss["rows"] = [x for x in miss["rows"] if x["name"] != SWIGLU]
    cases.append(("a missing swiglu line is a fault", any("swiglu at None" in f for f in table_faults(miss))))
    lone = _tab()
    lone["rows"].append(_row(f"{GEMV}|64x8x1|other", "moe_expert", 0.1))
    cases.append(("a GEMV with no swiglu beside it is a fault", any("no swiglu" in f for f in table_faults(lone))))
    cases.append(("Qwen3 registered with P136's reducer, W1 only", pr.FAMILIES[FAMILY] == (MODEL, REV, (1,))
                  and pr.MOE_LAYERS[FAMILY] == 48))
    bad_ = [n for n, ok_ in cases if not ok_]
    print(f"pername_reduce self-test {'OK' if not bad_ else 'FAILED ' + str(bad_)} ({len(cases)} cases)")
    return 0 if not bad_ else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    rec = assess(a.dir, a.e4b_sha)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    dec = rec.get("decision") or {}
    print("PERNAME " + rec["verdict"] + " " + json.dumps({"why": rec["why"], "swiglu": dec.get("swiglu"),
                                                          "p2": dec.get("p2")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
