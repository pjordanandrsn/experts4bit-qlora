#!/usr/bin/env python3
"""p152_reduce.py -- P152's rule (bench/p152/PREREG-p152.md): P145's per-name census reducer by counted substitutions, on
Qwen3-30B-A3B at W1 and W16 (W16 the subject), with the maintainer's three terms.

VOID   anything P136's reducer VOIDs (imported at its pinned bytes; Qwen3 registered with it at (1, 16), its
       grouped-nf4-gemm pin replaced by this lane's), or either batch's per-name table missing, not ok, not closing every
       replay exactly or under 56 kept steps, or, at W1, the expert path's three lines not at 48 launches per step
       (gate_up GEMV, _swiglu_rows, down GEMV on the bandwidth route), or an arm's unprofiled step missing at either batch.
READ   otherwise, with these lines:
  denominators  each batch's UNPROFILED step: the mean of the S1a / S1b arms' decode_ms_per_step at that batch (the
           maintainer's rule, 2026-10-19). Every share is over it; the profiled node period P is context only.
  W16      the subject: every line by removable (exclusive) time with its share of the unprofiled W16 step; the top five
           NAMED as candidates. Nothing is closed at W16; the head (one launch a step) is marked as stream (A)'s (P150).
  W1       the re-read: the same, with dense_gemm closed (stream (A): D1 / D2, P151 / P150) and the two expert GEMVs
           against K33's isolated times (gate_up 0.6068, down 0.3540 ms for 48 launches) as the per-name census reads them.
  expert_floor  (REPORTED) the expert path's byte floor at COPY_GBPS: W1 from 8 experts x 48 layers; W16 from the routing
           census (route_b16.json: the unique experts per layer and step, on real traffic), each expert's bytes at the
           arena's 0.5625 B a weight. Routing missing or faulty: the W16 floor reads UNREAD, the census stands.
  gpu      (REPORTED) gpu_phases.csv: clocks, power, temperature and clock event reasons at each phase's start and end.
A READ names candidates and proposes nothing automatically: any lever is its own registration.

    python p152_reduce.py --dir /root/p152 --out verdict_p152.json --e4b-sha <sha>
    python p152_reduce.py --self-test
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p136_reduce as pr  # noqa: E402  (staged at P136's pinned bytes)

FAMILY, MODEL, REV, LAYERS = "qwen3", "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", 48
BATCHES = (1, 16)
pr.FAMILIES[FAMILY] = (MODEL, REV, BATCHES)
pr.MOE_LAYERS[FAMILY] = LAYERS
GNF4_PIN = "38b308887688ea5bda7dd449bd3cc3dc1426b995"          # grouped-nf4-gemm v0.47.1 (e4b CI's pin)
pr.GNF4_SHA = GNF4_PIN
SWIGLU, GEMV = "_swiglu_rows", "_gemv_nf4_bw"
K33_MS = {"gate_up": 0.6068, "down": 0.3540}     # isolated, 48 launches each (the per-name census's K33 floor)
MIN_STEPS = 56
TOP = 5
W1_CLOSED = {"dense_gemm"}                       # stream (A): D1 (q/k/v, o) and D2 (the head), P151 / P150
COPY_GBPS = 1498.9                               # P141 phase 1's measured copy bandwidth on an RTX 5090 (select.json)
TOP_K, N_EXPERTS = 8, 128
#: one expert's NF4 bytes: gate_up (1536 x 2048) + down (2048 x 768) weights at 0.5625 B each (packed nibbles + an fp32
#: absmax per 64, bench/p39/k8_bake.py's arena)
EXPERT_BYTES = (1536 * 2048 + 2048 * 768) * 9 // 16


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


def unprofiled(root) -> tuple:
    """Each batch's unprofiled step (ms): the mean of the two arms' decode_ms_per_step. Returns (ms by batch, faults)."""
    out, faults = {}, []
    for b in BATCHES:
        v = []
        for tag in ("S1a", "S1b"):
            arm = _load(root, f"arm_{tag}.json") or {}
            ms = ((arm.get("workloads") or {}).get(f"W{b}") or {}).get("decode_ms_per_step")
            if isinstance(ms, (int, float)) and ms > 0:
                v.append(ms)
        if len(v) == 2:
            out[b] = round(sum(v) / 2, 4)
        else:
            faults.append(f"W{b}: an arm's unprofiled step is missing")
    return out, faults


def rank(tab, batch, u_ms) -> dict:
    lines = []
    for r in tab["rows"]:
        closed = batch == 1 and r["class"] in W1_CLOSED
        floor = K33_MS.get(r.get("role"), 0.0) if batch == 1 and r["name"] == GEMV else 0.0
        addr = max(0.0, r["removable_ms"] - floor)
        owner = "stream (A): P150's int8 head" if r["class"] == "dense_gemm" and r["n_per_step"] == 1 else None
        lines.append({"key": r["key"], "class": r["class"], "n_per_step": r["n_per_step"], "dur_ms": r["dur_ms"],
                      "removable_ms": r["removable_ms"], "floor_ms": floor, "addressable_ms": round(addr, 6),
                      "share_unprofiled": round(r["removable_ms"] / u_ms, 5),
                      "addressable_share_unprofiled": round(addr / u_ms, 5),
                      "share_profiled_P_context": round(r["removable_ms"] / tab["P_ms"], 5),
                      "closed": closed, "owner": owner})
    ranked = sorted((x for x in lines if not x["closed"]), key=lambda x: -x["addressable_ms"])
    by_class = {}
    for x in lines:
        by_class[x["class"]] = round(by_class.get(x["class"], 0.0) + x["removable_ms"], 6)
    return {"unprofiled_ms": u_ms, "P_ms_profiled_context": tab["P_ms"], "span_ms": tab.get("span_ms"),
            "ranked": ranked[:20], "candidates": [x for x in ranked if not x["owner"]][:TOP],
            "closed": [x for x in lines if x["closed"]], "class_removable_ms": by_class}


def expert_floor(route, w16_moe_ms, w1_moe_ms) -> dict:
    w1_bytes = TOP_K * LAYERS * EXPERT_BYTES
    out = {"copy_gbps": COPY_GBPS, "expert_bytes": EXPERT_BYTES, "reporting_only": True,
           "W1": {"bytes_per_step": w1_bytes, "floor_ms": round(w1_bytes / (COPY_GBPS * 1e6), 6),
                  "moe_expert_removable_ms": w1_moe_ms}}
    if not route or route.get("faults") or not route.get("unique_per_step"):
        why = "missing" if not route else ("; ".join((route.get("faults") or [])[:3]) or "no steps")
        out["W16"] = {"status": "UNREAD", "why": why}
        return out
    totals = [sum(st) for st in route["unique_per_step"]]
    mean_unique = sum(totals) / len(totals)
    floor = mean_unique * EXPERT_BYTES / (COPY_GBPS * 1e6)
    out["W16"] = {"status": "READ", "unique_experts_per_step_mean": round(mean_unique, 2),
                  "unique_per_layer_step": route["summary"]["unique_per_layer_step"],
                  "uniform_expectation_per_layer": route["summary"]["uniform_expectation_per_layer"],
                  "hottest_expert_share_of_a_layer": route["summary"]["hottest_expert_share_of_a_layer"],
                  "bytes_per_step": round(mean_unique * EXPERT_BYTES), "floor_ms": round(floor, 6),
                  "moe_expert_removable_ms": w16_moe_ms,
                  "headroom_ms": round(w16_moe_ms - floor, 6) if w16_moe_ms is not None else None}
    return out


def gpu_phases(root) -> list:
    p = Path(root) / "gpu_phases.csv"
    try:
        with p.open() as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def assess(root, e4b_sha) -> dict:
    base = pr.assess(Path(root), FAMILY, e4b_sha)
    tabs = {b: _load(root, f"pername_b{b}.json") for b in BATCHES}
    u, ufaults = unprofiled(root)
    faults = list(base["why"]) + [f for b in BATCHES for f in table_faults(tabs[b], b)] + ufaults
    rec = {"lane": "P152", "family": FAMILY, "model": MODEL, "revision": REV, "e4b_sha": e4b_sha, "gnf4_sha": GNF4_PIN,
           "verdict": "READ" if not faults else "VOID", "why": faults, "census": base.get("rows"),
           "unprofiled_ms": u, "gpu_phases": gpu_phases(root)}
    if not faults:
        w16, w1 = rank(tabs[16], 16, u[16]), rank(tabs[1], 1, u[1])
        rec["W16"], rec["W1"] = w16, w1
        rec["expert_floor"] = expert_floor(_load(root, "route_b16.json"), w16["class_removable_ms"].get("moe_expert"),
                                           w1["class_removable_ms"].get("moe_expert"))
        rec["named"] = [f"{x['key']} ({x['removable_ms']:.4f} ms, {x['share_unprofiled']:.2%} of the unprofiled W16 step)"
                        for x in w16["candidates"]]
    return rec


# ---------------------------------------------------------------------------------------------------------- self-test --
def _row(key, cls, removable, n=48):
    name, grid, *role = key.split("|")
    return {"key": key, "name": name, "grid": grid, "role": role[0] if role else None, "class": cls, "n_per_step": n,
            "dur_ms": removable, "excl_ms": removable, "billed_ms": 0.0, "gap_before_ms": 0.0, "removable_ms": removable}


def _tab(p=4.38, gate_up=0.587, down=0.362, n_gu=48, grouped=None):
    rows = [_row("kernel|1280x1x1", "dense_gemm", 0.719), _row("kernel|37984x1x1", "dense_gemm", 0.377, n=1),
            _row(f"{GEMV}|8x96x1|gate_up", "moe_expert", gate_up, n=n_gu), _row(f"{GEMV}|8x128x1|down", "moe_expert", down),
            _row(f"{SWIGLU}|8x1x1", "moe_expert", 0.031), _row("_fp8_paged_decode_split_f8dot|128x1x1", "attn", 0.556),
            _row("_router_epilogue|1x1x1", "moe_route", 0.131)]
    if grouped:
        rows = [_row("kernel|1280x1x1", "dense_gemm", 1.1), _row("kernel|37984x1x1", "dense_gemm", 0.9, n=1),
                _row("_gemm_nf4_grouped_smallm|64x8x1", "moe_expert", grouped),
                _row("_fp8_paged_decode_split_f8dot|128x1x1", "attn", 0.9)]
    return {"status": "ok", "steps_kept": 61, "closure_ns_max": 0, "P_ms": p, "span_ms": p - 0.4, "rows": rows}


def _route(uniq=80, steps=64, faults=None):
    per = [[uniq] * LAYERS for _ in range(steps)]
    return {"unique_per_step": per, "faults": faults or [],
            "summary": {"unique_per_layer_step": {"mean": float(uniq), "min": uniq, "max": uniq},
                        "uniform_expectation_per_layer": 82.4, "hottest_expert_share_of_a_layer": 0.05}}


def self_test() -> int:
    cases = []
    with tempfile.TemporaryDirectory() as d:
        r = assess(d, "a" * 40)
        cases.append(("missing receipts VOID", r["verdict"] == "VOID" and any("table missing" in w for w in r["why"])))
        cases.append(("missing arms VOID the denominators", any("unprofiled step is missing" in w for w in r["why"])))
    w16 = rank(_tab(p=17.0, grouped=11.5), 16, 16.2)
    cases.append(("W16 ranks by removable over the UNPROFILED step", w16["ranked"][0]["key"].startswith("_gemm_nf4_grouped")
                  and abs(w16["ranked"][0]["share_unprofiled"] - round(11.5 / 16.2, 5)) < 1e-12))
    cases.append(("the profiled P is context only", w16["ranked"][0]["share_profiled_P_context"] == round(11.5 / 17.0, 5)
                  and w16["unprofiled_ms"] == 16.2))
    cases.append(("the head is stream (A)'s and never a candidate", all("37984" not in x["key"] for x in w16["candidates"])
                  and any(x["owner"] for x in w16["ranked"])))
    cases.append(("nothing closed at W16", w16["closed"] == []))
    w1 = rank(_tab(), 1, 3.913)
    cases.append(("dense_gemm closed at W1 (stream (A))", all(x["class"] != "dense_gemm" for x in w1["ranked"])
                  and len(w1["closed"]) == 2))
    gu = next(x for x in w1["ranked"] if x["key"].endswith("gate_up"))
    cases.append(("a W1 GEMV ranks on removable minus K33's isolated time", gu["floor_ms"] == 0.6068 and gu["addressable_ms"] == 0.0))
    ef = expert_floor(_route(uniq=80), 11.5, 0.98)
    cases.append(("the W16 floor is the real unique experts x the expert bytes",
                  abs(ef["W16"]["floor_ms"] - 80 * LAYERS * EXPERT_BYTES / (COPY_GBPS * 1e6)) < 1e-6
                  and ef["W16"]["headroom_ms"] == round(11.5 - ef["W16"]["floor_ms"], 6)))
    cases.append(("the W1 floor is 8 experts x 48 layers", abs(ef["W1"]["floor_ms"] - 0.680) < 0.001))
    cases.append(("the expert bytes are 0.5625 B a weight", EXPERT_BYTES == 2654208))
    cases.append(("routing missing reads the W16 floor UNREAD", expert_floor(None, 11.5, 0.98)["W16"]["status"] == "UNREAD"))
    cases.append(("routing faults read the W16 floor UNREAD", expert_floor(_route(faults=["x"]), 11.5, 0.98)["W16"]["status"] == "UNREAD"))
    cases.append(("a complete W1 table has no faults", table_faults(_tab(), 1) == []))
    cases.append(("a W1 expert path off 48 is a fault", any("gate_up at 47" in f for f in table_faults(_tab(n_gu=47), 1))))
    cases.append(("W16 skips the expert-path count", table_faults(_tab(grouped=11.0), 16) == []))
    bad = _tab()
    bad["closure_ns_max"] = 3
    cases.append(("a replay that does not close is a fault", any("close" in f for f in table_faults(bad, 16))))
    few = _tab()
    few["steps_kept"] = 40
    cases.append(("too few kept steps is a fault", any("< 56" in f for f in table_faults(few, 1))))
    cases.append(("Qwen3 at (1, 16) in P136's reducer, 48 layers", pr.FAMILIES[FAMILY] == (MODEL, REV, (1, 16))
                  and pr.MOE_LAYERS[FAMILY] == 48))
    cases.append(("the grouped-nf4-gemm pin is this lane's", pr.GNF4_SHA == GNF4_PIN))
    bad_ = [n for n, ok_ in cases if not ok_]
    print(f"p152_reduce self-test {'OK' if not bad_ else 'FAILED ' + str(bad_)} ({len(cases)} cases)")
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
    print("P152 " + rec["verdict"] + " " + json.dumps({"why": rec["why"], "named": rec.get("named"),
                                                       "expert_floor_W16": (rec.get("expert_floor") or {}).get("W16")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
