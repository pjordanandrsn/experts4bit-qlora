#!/usr/bin/env python3
"""p145_reduce.py -- P145's rule (bench/p145/PREREG-p145.md): the per-name census (bench/pername/pername_reduce.py) by
counted substitutions, on Mixtral-8x7B at W1 and W16 with its expert GEMVs on K33 by default (grouped-nf4-gemm #556).

VOID   anything P136's reducer VOIDs (imported at its pinned bytes; Mixtral is already its (1, 16) family, and only its
       grouped-nf4-gemm pin is replaced by this lane's), or either batch's per-name table missing, not ok, not closing
       every replay exactly or under 56 kept steps, or, at W1, the expert path's three lines not at 32 launches per step
       (gate_up GEMV, _swiglu_rows, down GEMV on the bandwidth route).
READ   otherwise, with these lines, every share over that batch's node replay period P:
  gemv     (W1) each expert GEMV's removable time against P141's isolated time for the same 32 launches (phase 1 on an
           RTX 5090, run p141-5090-2, the selected plans: gate_up 3.0862 ms, down 1.9502 ms): the in-situ excess.
  ranked   every line by addressable time = removable - its floor (P141's isolated time for the two expert GEMVs, 0
           otherwise). Nothing is closed for Mixtral: dense_gemm (its bf16 attention projections and head) is ranked.
  next     the top-ranked W1 line, PROPOSED to the maintainer with this table (a lever is a new registration, never
           automatic). A dense_gemm line on top names a quality premise first, never a kernel (the maintainer's rule,
           2026-10-18). W16's ranking is reported, never proposed.

    python p145_reduce.py --dir /root/p145 --out verdict_p145.json --e4b-sha <sha>
    python p145_reduce.py --self-test
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p136_reduce as pr  # noqa: E402  (staged at P136's pinned bytes)

FAMILY, LAYERS = "mixtral", 32
GNF4_PIN = "eea5df6baefbe38b795621286fe70eac2980aa44"          # grouped-nf4-gemm v0.47.0 (#556: Mixtral's shapes and plans in auto)
pr.GNF4_SHA = GNF4_PIN
MODEL, REV, BATCHES = pr.FAMILIES[FAMILY]
SWIGLU, GEMV = "_swiglu_rows", "_gemv_nf4_bw"
P141_MS = {"gate_up": 3.0862, "down": 1.9502}     # isolated, 32 launches each, bw_prmt32 at the selected plans, an RTX
                                                  # 5090 (P141 phase 1, select.json; synthetic stores, the served forms)
MIN_STEPS = 56


def _load(root, name):
    try:
        return json.loads((Path(root) / name).read_text())
    except (OSError, ValueError):
        return None


def table_faults(tab, batch) -> list:
    if not tab:
        return [f"W{batch}: per-name table missing"]
    if tab.get("status") != "ok":
        return [f"W{batch}: per-name table {tab.get('status')}: {tab.get('why')}"]
    faults = []
    if tab.get("closure_ns_max") != 0:
        faults.append(f"W{batch}: per-name table does not close a replay (max {tab.get('closure_ns_max')} ns)")
    if (tab.get("steps_kept") or 0) < MIN_STEPS:
        faults.append(f"W{batch}: per-name table kept {tab.get('steps_kept')} steps < {MIN_STEPS}")
    if batch != 1:
        return faults
    n = {}
    for r in tab.get("rows") or []:
        line = "swiglu" if r["name"] == SWIGLU else r.get("role") if r["name"] == GEMV else None
        if line:
            n[line] = n.get(line, 0) + r["n_per_step"]
    for line in ("gate_up", "swiglu", "down"):
        if n.get(line) != LAYERS:
            faults.append(f"W1 expert path: {line} at {n.get(line)} launches per step, not {LAYERS}")
    if n.get("other"):
        faults.append(f"W1: an expert GEMV with no swiglu beside it ({n['other']} per step)")
    return faults


def rank(tab, floors) -> dict:
    p = tab["P_ms"]
    lines = []
    for r in tab["rows"]:
        floor = floors.get(r.get("role"), 0.0) if r["name"] == GEMV else 0.0
        addr = max(0.0, r["removable_ms"] - floor)
        lines.append({"key": r["key"], "class": r["class"], "n_per_step": r["n_per_step"], "dur_ms": r["dur_ms"],
                      "billed_ms": r["billed_ms"], "removable_ms": r["removable_ms"], "floor_ms": floor,
                      "addressable_ms": round(addr, 6), "share": round(r["removable_ms"] / p, 5),
                      "addressable_share": round(addr / p, 5)})
    ranked = sorted(lines, key=lambda x: -x["addressable_ms"])
    by_class = {}
    for x in lines:
        by_class[x["class"]] = round(by_class.get(x["class"], 0.0) + x["removable_ms"], 6)
    return {"P_ms": p, "ranked": ranked[:20], "class_removable_ms": by_class, "lines": lines}


def decide(tab1, tab16) -> dict:
    w1 = rank(tab1, P141_MS)
    gemv = {x["key"].rsplit("|", 1)[1]: {"removable_ms": x["removable_ms"], "p141_isolated_ms": x["floor_ms"],
                                         "in_situ_excess_ms": round(x["removable_ms"] - x["floor_ms"], 6),
                                         "in_situ_excess_share": round((x["removable_ms"] - x["floor_ms"]) / w1["P_ms"], 5)}
            for x in w1["lines"] if x["key"].startswith(GEMV + "|")}
    sw = round(sum(x["removable_ms"] for x in w1["lines"] if x["key"].startswith(SWIGLU + "|")), 6)
    top = w1["ranked"][0] if w1["ranked"] else None
    if top is None:
        nxt = {"target": None, "why": "no line"}
    elif top["class"] == "dense_gemm":
        nxt = {"target": f"PROPOSE a quality premise for {top['key']}",
               "why": f"top addressable W1 line is dense_gemm ({top['addressable_ms']:.4f} ms, "
                      f"{top['addressable_share']:.2%} of P): a quality premise comes before any kernel"}
    else:
        nxt = {"target": f"PROPOSE {top['key']}",
               "why": f"top addressable W1 line ({top['addressable_ms']:.4f} ms, {top['addressable_share']:.2%} of P)"}
    w16 = rank(tab16, {})
    for w in (w1, w16):
        w.pop("lines")
    return {"W1": w1, "W16": w16, "gemv": gemv, "swiglu_removable_ms": sw, "next": nxt}


def assess(root, e4b_sha) -> dict:
    base = pr.assess(Path(root), FAMILY, e4b_sha)
    tabs = {b: _load(root, f"pername_b{b}.json") for b in BATCHES}
    faults = list(base["why"]) + [f for b in BATCHES for f in table_faults(tabs[b], b)]
    rec = {"lane": "P145", "family": FAMILY, "model": MODEL, "revision": REV, "e4b_sha": e4b_sha,
           "gnf4_sha": GNF4_PIN, "verdict": "READ" if not faults else "VOID", "why": faults, "census": base.get("rows")}
    if not faults:
        rec["decision"] = decide(tabs[1], tabs[16])
    return rec


# ---------------------------------------------------------------------------------------------------------- self-test --
def _row(key, cls, removable, n=32, dur=None, billed=0.0):
    name, grid, *role = key.split("|")
    return {"key": key, "name": name, "grid": grid, "role": role[0] if role else None, "class": cls, "n_per_step": n,
            "dur_ms": dur if dur is not None else removable, "excl_ms": removable, "billed_ms": billed,
            "gap_before_ms": 0.0, "removable_ms": removable}


def _tab(gate_up=3.30, down=2.10, dense=1.90, attn=0.40, sw=0.05, n_gu=32):
    return {"status": "ok", "steps_kept": 61, "closure_ns_max": 0, "P_ms": 8.1, "class_removable_ms": {},
            "rows": [_row("nvjet_bf16|64x1x1", "dense_gemm", dense, n=97),
                     _row(f"{GEMV}|1792x1x1|gate_up", "moe_expert", gate_up, n=n_gu),
                     _row(f"{GEMV}|256x1x2|down", "moe_expert", down), _row(f"{SWIGLU}|8x2x1", "moe_expert", sw),
                     _row("_fp8_paged_decode_split_f8dot|128x1x1", "attn", attn)]}


def self_test() -> int:
    cases = []
    with tempfile.TemporaryDirectory() as d:
        r = assess(d, "a" * 40)
        cases.append(("missing receipts VOID", r["verdict"] == "VOID" and any("table missing" in w for w in r["why"])))
    dd = decide(_tab(), _tab())
    # dense removes 1.90 ms; gate_up 3.30 - 3.0862 = 0.214; attn 0.40: dense is on top -> a quality premise first
    cases.append(("dense on top names a quality premise", dd["next"]["target"].startswith("PROPOSE a quality premise")
                  and "nvjet_bf16" in dd["next"]["target"]))
    cases.append(("the expert GEMVs' excess is over P141's isolated times",
                  abs(dd["gemv"]["gate_up"]["in_situ_excess_ms"] - 0.2138) < 1e-9
                  and abs(dd["gemv"]["down"]["in_situ_excess_ms"] - 0.1498) < 1e-9))
    da = decide(_tab(dense=0.10, attn=0.60), _tab())
    cases.append(("an open line on top is proposed", da["next"]["target"] == "PROPOSE _fp8_paged_decode_split_f8dot|128x1x1"))
    dg = decide(_tab(dense=0.10, attn=0.05, gate_up=4.20), _tab())
    cases.append(("a GEMV ranks by its addressable time, not its removable", dg["next"]["target"]
                  == f"PROPOSE {GEMV}|1792x1x1|gate_up" and abs(dg["W1"]["ranked"][0]["addressable_ms"] - 1.1138) < 1e-9))
    cases.append(("dense_gemm is ranked for Mixtral", any(x["class"] == "dense_gemm" for x in dd["W1"]["ranked"])))
    cases.append(("W16 has no floor", all(x["floor_ms"] == 0.0 for x in dd["W16"]["ranked"])))
    cases.append(("a complete W1 table has no faults", table_faults(_tab(), 1) == []))
    cases.append(("W16 skips the expert-path count", table_faults(_tab(n_gu=7), 16) == []))
    cases.append(("a W1 expert path off 32 is a fault", any("gate_up at 31" in f for f in table_faults(_tab(n_gu=31), 1))))
    bad = _tab()
    bad["closure_ns_max"] = 3
    cases.append(("a replay that does not close is a fault", any("close" in f for f in table_faults(bad, 16))))
    few = _tab()
    few["steps_kept"] = 40
    cases.append(("too few kept steps is a fault", any("< 56" in f for f in table_faults(few, 1))))
    cases.append(("Mixtral at (1, 16) in P136's reducer, 32 layers", pr.FAMILIES[FAMILY][2] == (1, 16)
                  and pr.MOE_LAYERS[FAMILY] == LAYERS == 32 and BATCHES == (1, 16)))
    cases.append(("the grouped-nf4-gemm pin is this lane's", pr.GNF4_SHA == GNF4_PIN))
    bad_ = [n for n, ok_ in cases if not ok_]
    print(f"p145_reduce self-test {'OK' if not bad_ else 'FAILED ' + str(bad_)} ({len(cases)} cases)")
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
    print("P145 " + rec["verdict"] + " " + json.dumps({"why": rec["why"], "next": dec.get("next"), "gemv": dec.get("gemv")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
