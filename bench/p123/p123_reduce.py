#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""p123_reduce.py -- lane P123's rule (#1313; bench/p123/PREREG-p123.md): where the shipped default's decode step goes.

Inputs in --dir: the unprofiled speed arms ``arm_S1a.json`` / ``arm_S1b.json`` (p123_box.py), the census drivers'
records ``census_drv_b1.json`` / ``census_drv_b16.json`` (p123_census.py), the census arms ``census_b1.json`` /
``census_b16.json`` (bench/sc1b/sc1b_census.py ``arm`` with the v1 NF4 class map, unchanged), and on the proof
``router_census.json`` (Amendment 1: p123_box.py --mode routers).

First rung that applies:
  VOID   a record missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; on the proof, the router
         epilogue licensed on fewer than every router under E4B_FUSE_ROUTER_EPI=auto alone (Amendment 1: the
         fam-prove-1 case, #1398's regression check); not the shipped default
         (a fusion knob not resolved by the default, the fusion census or the decode-GEMV route off the registration);
         a speed slope void; a census arm void, or labelled CLASS_MAP_INCOMPLETE, CLASS_MAP_SEGMENT_BROKEN,
         NSYS_DIAGNOSTIC_ERRORS or CLOCK_MISMATCH; a census window not exactly the registered steps.
  NOISY  the speed self-pair S1b/S1a outside [0.97, 1.03] on W1 or W16 (the census's unprofiled reference is noise).
  READ   otherwise. There is no default consequence: a census prices the next lever, it moves nothing.

Read per batch (B = 1 and 16): P (the step, graph mode), each class's time and its share of P, the RESIDUAL row --
P minus the eight named classes, which holds the unmapped kernels, the in-graph idle, the out-of-graph idle and the
overlap terms, so the breakdown cannot hide a hole -- the GPU busy fraction (P - I_in - idle_out) / P (UNREAD when the
profiler inflated the step past SC1b's 5 % gate), kernels per step, peak memory, and each registered prediction graded
HELD or MISSED. Ratios are within this box only.

  p123_reduce.py --dir D --out verdict_p123.json --e4b-sha SHA [--proof]
  p123_reduce.py --self-test
"""
import argparse
import json
import os
import sys

GNF4_SHA = "6ee2e10408161a9d3c874975c9191a7f2957e6f4"
QWEN, GRAN = "Qwen/Qwen3-30B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct"
REVS = {QWEN: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", GRAN: "a02780686e08a03fe0d2679a293b5c74a90efa89"}
KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
# the shipped default at the registration commit (#1361): Qwen3-MoE's four knobs resolve to auto (default-allowlisted),
# Granite-MoE's to 0 (default-off; its proof runs the unfused default and grouped-nf4-gemm's scalar GEMV)
DEFAULT = {QWEN: {"source": "default-allowlisted",
                  "census": {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48],
                             "fuse_router_epilogue_n": 48}},
           GRAN: {"source": "default-off",
                  "census": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0],
                             "fuse_router_epilogue_n": 0}}}
MOE_LAYERS = {QWEN: 48, GRAN: 32}
BATCHES = (1, 16)
STEPS = 64
SELF_PAIR = (0.97, 1.03)
CLASSES = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual")
NAMED = CLASSES[:-1]                         # the census's own "residual" class is the unmapped kernels
BLOCKING = ("CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS", "CLOCK_MISMATCH")
BW = ("bw_tree", "bw_prmt32", "bw_splitk")

# Predictions, written before any data (PREREG-p123.md): shares of the graph-mode step P on Qwen3-30B-A3B, per batch.
PREDICTED = {
    1: {"dense_gemm": (0.33, 0.50), "moe_expert": (0.15, 0.28), "attn": (0.06, 0.15), "moe_route": (0.02, 0.09),
        "norm_elem": (0.04, 0.12), "RESIDUAL": (0.02, 0.15), "busy_fraction": (0.85, 1.00),
        "kernels_per_step": (900, 1600)},
    16: {"dense_gemm": (0.12, 0.30), "moe_expert": (0.40, 0.65), "attn": (0.04, 0.15), "moe_route": (0.02, 0.10),
         "norm_elem": (0.02, 0.10), "RESIDUAL": (0.01, 0.12), "busy_fraction": (0.90, 1.00),
         "kernels_per_step": (900, 1700)},
}


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def speed_faults(t, r, e4b_sha, model, proof) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha or r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{t}: e4b {r.get('e4b_sha')} / gnf4 {r.get('gnf4_sha')}, registered {e4b_sha} / {GNF4_SHA}")
    if r.get("model") != model or r.get("revision") != REVS[model]:
        out.append(f"{t}: model {r.get('model')}@{r.get('revision')}, registered {model}@{REVS[model]}")
    want = DEFAULT[model]
    src = r.get("fusion_sources") or {}
    if set(src) != set(KNOBS) or any(v != want["source"] for v in src.values()):
        out.append(f"{t}: fusion sources {src}, the shipped default resolves every knob {want['source']}")
    if {k: (r.get("fusions") or {}).get(k) for k in CENSUS_KEYS} != want["census"]:
        out.append(f"{t}: fusion census {r.get('fusions')}, the default's is {want['census']}")
    if any(v for v in (r.get("knobs") or {}).values()):
        out.append(f"{t}: a knob was set {r.get('knobs')}")
    d = r.get("dispatch_build") or {}
    if model == QWEN and (d.get("bw_prmt32", 0) <= 0 or d.get("dotpad", 0) or d.get("dotpad_splitk", 0)):
        out.append(f"{t}: dispatch at the build {d}; the default is bw_prmt32 and no dot-pad")
    if model == GRAN and (any(d.get(k, 0) for k in BW) or d.get("scalar", 0) + d.get("scalar_splitk", 0) <= 0):
        out.append(f"{t}: dispatch at the build {d}; Granite's shapes take the scalar GEMV")
    for w in ("W1", "W16"):
        s = (r.get("workloads") or {}).get(w) or {}
        if s.get("status") != "ok" or not s.get("decode_ms_per_step"):
            out.append(f"{t}: {w} slope void ({s.get('status')})")
    return out


def grade(value, band):
    if value is None:
        return "UNREAD"
    lo, hi = band
    return "HELD" if lo <= value <= hi else "MISSED"


def read_batch(c, drv, unprof_ms, b, proof):
    """One batch's census, read: classes, shares, the RESIDUAL row, busy fraction, predictions graded."""
    p = c["P_ms"]
    t = c["terms"]
    named = sum(t[k] for k in NAMED)
    residual_row = p - named
    inflated = "PROFILER_INFLATED" in c.get("labels", []) or "NO_UNPROFILED_TIMING" in c.get("labels", [])
    busy = None if inflated else (p - t["I_in"] - t["idle_out"]) / p
    kps = (c.get("node") or {}).get("kernels_per_graph_modal")
    row = {"P_ms": round(p, 4), "unprofiled_ms": unprof_ms, "inflate": c.get("gates", {}).get("inflate"),
           "labels": c.get("labels", []), "classes_ms": {k: round(t[k], 4) for k in NAMED},
           "shares": {k: round(t[k] / p, 4) for k in NAMED},
           "RESIDUAL": {"ms": round(residual_row, 4), "share": round(residual_row / p, 4),
                        "unmapped_kernels_ms": t["residual"], "I_in_ms": t["I_in"], "idle_out_ms": t["idle_out"],
                        "overlap_in_ms": t["overlap_in"], "O_ms": t["O"]},
           "busy_fraction": None if busy is None else round(busy, 4),
           "busy_fraction_status": "UNREAD (the profiler inflated the step)" if inflated else "read",
           "kernels_per_step": kps, "peak_reserved_bytes": drv.get("max_memory_reserved"),
           "window_positions": drv.get("window_decode_positions")}
    if not proof:
        pred = PREDICTED[b]
        g = {k: grade(row["shares"][k], pred[k]) for k in ("dense_gemm", "moe_expert", "attn", "moe_route", "norm_elem")}
        g["RESIDUAL"] = grade(row["RESIDUAL"]["share"], pred["RESIDUAL"])
        g["busy_fraction"] = grade(row["busy_fraction"], pred["busy_fraction"])
        g["kernels_per_step"] = grade(kps, pred["kernels_per_step"])
        row["predictions"] = g
    row["largest_class"] = max(NAMED, key=lambda k: t[k])
    return row


def router_faults(r, e4b_sha, model) -> list:
    """Amendment 1, the proof: every router licensed by the epilogue's probe under E4B_FUSE_ROUTER_EPI=auto alone."""
    out = []
    if r.get("e4b_sha") != e4b_sha or r.get("model") != model:
        out.append(f"router_census: e4b {r.get('e4b_sha')} model {r.get('model')}")
    want = {k: (None if k != "E4B_FUSE_ROUTER_EPI" else "auto") for k in KNOBS}
    if {k: (r.get("knobs") or {}).get(k) for k in KNOBS} != want:
        out.append(f"router_census: knobs {r.get('knobs')}, Amendment 1 sets E4B_FUSE_ROUTER_EPI=auto alone")
    if (r.get("fusion_sources") or {}).get("E4B_FUSE_ROUTER_EPI") != "explicit":
        out.append(f"router_census: the router knob resolved {(r.get('fusion_sources') or {}).get('E4B_FUSE_ROUTER_EPI')}")
    n = (r.get("fusions") or {}).get("fuse_router_epilogue_n")
    if n != MOE_LAYERS[model]:
        out.append(f"router_census: the router epilogue licensed {n} of {MOE_LAYERS[model]} routers "
                   f"(report {r.get('router_report')})")
    return out


def reduce(arms, drvs, census, e4b_sha, proof=False, routers=None):
    model = GRAN if proof else QWEN
    out = {"lane": "P123", "proof": proof, "model": model, "verdict": None, "reasons": []}
    missing = [f"arm_{t}" for t in ("S1a", "S1b") if not arms.get(t) or arms[t].get("status") != "ok"]
    if proof and (not routers or routers.get("status") != "ok"):
        missing.append("router_census")
    missing += [f"census_drv_b{b}" for b in BATCHES if not drvs.get(b)]
    missing += [f"census_b{b}" for b in BATCHES if not census.get(b)]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = []
    if proof:
        void += router_faults(routers, e4b_sha, model)
        out["routers"] = {"licensed": (routers.get("fusions") or {}).get("fuse_router_epilogue_n"),
                          "of": MOE_LAYERS[model], "report": routers.get("router_report")}
    for t, r in arms.items():
        void += speed_faults(t, r, e4b_sha, model, proof)
    for b in BATCHES:
        d, c = drvs[b], census[b]
        if d.get("steps_bracketed") != STEPS or d.get("batch") != b:
            void.append(f"census_drv_b{b}: {d.get('steps_bracketed')} steps at batch {d.get('batch')}, registered {STEPS} at {b}")
        if d.get("e4b_sha") != e4b_sha or d.get("model") != model:
            void.append(f"census_drv_b{b}: e4b {d.get('e4b_sha')} model {d.get('model')}")
        if c.get("status") not in ("ok", "labelled") or c.get("batch") != b:
            void.append(f"census_b{b}: status {c.get('status')} at batch {c.get('batch')}")
            continue
        void += [f"census_b{b}: {lab}" for lab in c.get("labels", []) if lab in BLOCKING]
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    ms = {w: [arms[t]["workloads"][w]["decode_ms_per_step"] for t in ("S1a", "S1b")] for w in ("W1", "W16")}
    pairs = {w: round(ms[w][1] / ms[w][0], 4) for w in ms}
    out["speed"] = {"decode_ms_per_step": ms, "self_pair_S1b_over_S1a": pairs,
                    "peak_allocated_bytes": max(arms[t]["mem_after_runs"]["max_memory_allocated"] for t in arms)}
    noisy = [f"{w} self-pair {v}" for w, v in pairs.items() if not SELF_PAIR[0] <= v <= SELF_PAIR[1]]
    unprof = {1: sum(ms["W1"]) / 2, 16: sum(ms["W16"]) / 2}
    out["batches"] = {str(b): read_batch(census[b], drvs[b], round(unprof[b], 4), b, proof) for b in BATCHES}
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
        return out
    b1 = out["batches"]["1"]
    out.update(verdict="READ", reasons=[
        f"B=1: P {b1['P_ms']} ms, largest class {b1['largest_class']} ({b1['shares'][b1['largest_class']]:.0%}), "
        f"RESIDUAL {b1['RESIDUAL']['share']:.1%}, busy {b1['busy_fraction']}"])
    return out


# ------------------------------------------------------------------ self-test
E = "a" * 40


def fake_arm(tag, model=QWEN, w1=4.6, w16=17.0, sources=None, census=None, dispatch=None, knobs=None):
    """An arm as p123_box.py writes it (keys checked against the box's record by tests/test_p123_staged_pin.py)."""
    want = DEFAULT[model]
    d = dispatch or ({"bw_prmt32": 288, "dotpad": 0} if model == QWEN else {"scalar": 192, "bw_prmt32": 0})
    return {"mode": "speed", "tag": tag, "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": model, "revision": REVS[model],
            "torch": "2.8.0", "load_s": 30.0, "knobs": knobs or {k: None for k in KNOBS},
            "fusions": census or dict(want["census"]), "fusion_modes": {k: "auto" for k in KNOBS},
            "fusion_sources": sources or {k: want["source"] for k in KNOBS}, "model_type": "qwen3_moe",
            "graph_status": {str(b): "graph" for b in (1, 2, 4, 8, 16)}, "max_seqs": 16, "buckets": [1, 2, 4, 8, 16],
            "dispatch_build": d, "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3,
            "mem_after_load": {"max_memory_allocated": 23 << 30}, "status": "ok",
            "workloads": {"W1": {"batch": 1, "decode_ms_per_step": w1, "status": "ok"},
                          "W16": {"batch": 16, "decode_ms_per_step": w16, "status": "ok"}},
            "graph_stats": {}, "dispatch_total": d, "mem_after_runs": {"max_memory_allocated": 24 << 30},
            "nvidia_smi": "x"}


def fake_drv(b, model=QWEN, steps=STEPS):
    return {"engine": "e4b", "batch": b, "mode": "census", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": model,
            "revision": REVS[model], "steps_bracketed": steps, "profiled_ms_per_step": 5.0,
            "window_decode_positions": [40, 140], "max_memory_reserved": 25 << 30}


def fake_routers(model=GRAN, n=None, knobs=None, source="explicit"):
    """A router census as p123_box.py --mode routers writes it."""
    return {"mode": "routers", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": model, "revision": REVS[model],
            "torch": "2.8.0", "load_s": 5.0,
            "knobs": knobs or {k: ("auto" if k == "E4B_FUSE_ROUTER_EPI" else None) for k in KNOBS},
            "fusions": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0],
                        "fuse_router_epilogue_n": MOE_LAYERS[model] if n is None else n},
            "fusion_modes": {k: ("auto" if k == "E4B_FUSE_ROUTER_EPI" else "0") for k in KNOBS},
            "fusion_sources": {k: (source if k == "E4B_FUSE_ROUTER_EPI" else "default-off") for k in KNOBS},
            "model_type": "granitemoe", "router_report": {
                "mode": "auto", "patched": MOE_LAYERS[model] if n is None else n,
                "failed_probe": 0 if n is None else MOE_LAYERS[model] - n, "no_kernel_mode": 0, "fp32_upstream": 0},
            "moe_layers": MOE_LAYERS[model], "status": "ok"}


def fake_census(b, p=4.6, shares=None, labels=None, i_in=0.15, idle_out=0.2, kps=1200, status=None):
    """A census arm as sc1b_census.arm writes it (P_ms, terms, labels, gates, node)."""
    sh = shares or {"dense_gemm": 0.40, "moe_expert": 0.22, "attn": 0.10, "moe_route": 0.05, "norm_elem": 0.08,
                    "sample": 0.01, "input_prep": 0.01, "memcpy": 0.01}
    t = {k: round(sh.get(k, 0.0) * p, 6) for k in NAMED}
    t["residual"] = 0.01
    t["I_in"], t["idle_out"], t["overlap_in"] = i_in, idle_out, 0.0
    t["O"] = round(sum(t[k] for k in CLASSES) + t["overlap_in"] + t["I_in"] + t["idle_out"] - p, 6)
    labs = labels or []
    return {"engine": "e4b", "batch": b, "graph": {"status": "ok"}, "node": {"status": "ok", "kernels_per_graph_modal": kps},
            "unprofiled_ms": p, "gates": {"inflate": 0.01, "map_residual_fraction": 0.002}, "labels": labs,
            "P_ms": p, "terms": t, "status": status or ("labelled" if labs else "ok")}


def self_test() -> int:
    def run(arms=None, drvs=None, census=None, proof=False, routers=None):
        model = GRAN if proof else QWEN
        arms = arms or {"S1a": fake_arm("S1a", model), "S1b": fake_arm("S1b", model)}
        drvs = drvs or {b: fake_drv(b, model) for b in BATCHES}
        census = census or {1: fake_census(1), 16: fake_census(16, p=17.0, shares={
            "dense_gemm": 0.2, "moe_expert": 0.5, "attn": 0.08, "moe_route": 0.05, "norm_elem": 0.05})}
        return reduce(arms, drvs, census, E, proof=proof, routers=routers or (fake_routers() if proof else None))
    v = lambda **kw: run(**kw)["verdict"]  # noqa: E731
    r = run()
    b1 = r["batches"]["1"]
    cases = [
        ("read", r["verdict"] == "READ"),
        ("the RESIDUAL row is P minus the named classes", abs(b1["RESIDUAL"]["ms"] - (b1["P_ms"] - sum(
            b1["classes_ms"].values()))) < 1e-3),
        ("busy fraction", b1["busy_fraction"] == round((4.6 - 0.15 - 0.2) / 4.6, 4)),
        ("predictions graded", b1["predictions"]["dense_gemm"] == "HELD" and set(b1["predictions"]) >= {
            "RESIDUAL", "busy_fraction", "kernels_per_step"}),
        ("a missed prediction is graded MISSED, not VOID", v(census={1: fake_census(1, shares={
            "dense_gemm": 0.7, "moe_expert": 0.1}), 16: fake_census(16, p=17.0)}) == "READ" and run(census={
                1: fake_census(1, shares={"dense_gemm": 0.7}), 16: fake_census(16, p=17.0)})["batches"]["1"][
                    "predictions"]["dense_gemm"] == "MISSED"),
        ("inflated: busy UNREAD, still READ", (lambda x: x["verdict"] == "READ" and x["batches"]["1"][
            "busy_fraction"] is None)(run(census={1: fake_census(1, labels=["PROFILER_INFLATED"]),
                                                  16: fake_census(16, p=17.0)}))),
        ("noisy self-pair", v(arms={"S1a": fake_arm("S1a"), "S1b": fake_arm("S1b", w1=4.9)}) == "NOISY"),
        ("class map incomplete", v(census={1: fake_census(1, labels=["CLASS_MAP_INCOMPLETE"]),
                                           16: fake_census(16, p=17.0)}) == "VOID"),
        ("segment broken", v(census={1: fake_census(1), 16: fake_census(16, p=17.0,
                                                                         labels=["CLASS_MAP_SEGMENT_BROKEN"])}) == "VOID"),
        ("census void", v(census={1: fake_census(1, status="void"), 16: fake_census(16, p=17.0)}) == "VOID"),
        ("not the shipped default: a knob set", v(arms={"S1a": fake_arm("S1a", knobs={KNOBS[0]: "auto"}),
                                                        "S1b": fake_arm("S1b")}) == "VOID"),
        ("not the shipped default: resolved explicitly", v(arms={"S1a": fake_arm("S1a", sources={
            k: "explicit" for k in KNOBS}), "S1b": fake_arm("S1b")}) == "VOID"),
        ("fusion census off", v(arms={"S1a": fake_arm("S1a", census={**DEFAULT[QWEN]["census"], "fuse_qkv_n": 0}),
                                      "S1b": fake_arm("S1b")}) == "VOID"),
        ("dot-pad at the build", v(arms={"S1a": fake_arm("S1a", dispatch={"bw_prmt32": 200, "dotpad": 88}),
                                         "S1b": fake_arm("S1b")}) == "VOID"),
        ("window short", v(drvs={1: fake_drv(1, steps=63), 16: fake_drv(16)}) == "VOID"),
        ("missing census", reduce({"S1a": fake_arm("S1a"), "S1b": fake_arm("S1b")}, {b: fake_drv(b) for b in BATCHES},
                                  {1: fake_census(1)}, E)["verdict"] == "VOID"),
        ("proof on Granite's unfused default", run(proof=True, arms={"S1a": fake_arm("S1a", GRAN),
                                                                    "S1b": fake_arm("S1b", GRAN)})["verdict"] == "READ"),
        ("Amendment 1: a proof whose routers all license", run(proof=True, arms={
            "S1a": fake_arm("S1a", GRAN), "S1b": fake_arm("S1b", GRAN)})["routers"]["licensed"] == 32),
        ("Amendment 1: 27 of 32 routers VOIDs the proof (fam-prove-1)", run(proof=True, arms={
            "S1a": fake_arm("S1a", GRAN), "S1b": fake_arm("S1b", GRAN)}, routers=fake_routers(n=27))["verdict"] == "VOID"),
        ("Amendment 1: a proof without the router census VOIDs", reduce(
            {"S1a": fake_arm("S1a", GRAN), "S1b": fake_arm("S1b", GRAN)}, {b: fake_drv(b, GRAN) for b in BATCHES},
            {1: fake_census(1), 16: fake_census(16, p=17.0)}, E, proof=True)["verdict"] == "VOID"),
        ("Amendment 1: another knob set VOIDs", run(proof=True, arms={
            "S1a": fake_arm("S1a", GRAN), "S1b": fake_arm("S1b", GRAN)}, routers=fake_routers(knobs={
                **{k: None for k in KNOBS}, "E4B_FUSE_ROUTER_EPI": "auto", "E4B_FUSE_T1_GLUE": "auto"}))["verdict"] == "VOID"),
        ("the reading's records on a proof", run(proof=True, arms={"S1a": fake_arm("S1a"), "S1b": fake_arm("S1b")},
                                                  drvs={b: fake_drv(b) for b in BATCHES})["verdict"] == "VOID"),
        ("the default's census is #1361's", DEFAULT[QWEN]["census"]["fuse_t1_glue_n"] == 4 * MOE_LAYERS[QWEN] + 1),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p123_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--proof", action="store_true")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    arms = {t: r for t in ("S1a", "S1b") if (r := _load(a.dir, f"arm_{t}.json"))}
    drvs = {b: r for b in BATCHES if (r := _load(a.dir, f"census_drv_b{b}.json"))}
    census = {b: r for b in BATCHES if (r := _load(a.dir, f"census_b{b}.json"))}
    v = reduce(arms, drvs, census, a.e4b_sha, proof=a.proof, routers=_load(a.dir, "router_census.json"))
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P123_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
