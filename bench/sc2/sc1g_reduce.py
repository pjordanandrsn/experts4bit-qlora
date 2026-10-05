#!/usr/bin/env python3
"""sc1g_reduce.py -- lane SC1g (#846): the registered rule (bench/sc2/SC1g-PREREG.md, amendment A1).

Windows: conv1, conv2 are GRADED (ultrachat_200k conversations in gpt-oss's chat template); wikitext is a DESCRIPTIVE control
(the chat-framed wikitext the proof showed is out of distribution). Every row is SC1's quality row (sc1_reduce.quality_row:
the window's text_sha, scored range [513, 2560], 2,048 steps, the scorer's own verdict). An e4b row is VALID only if its ROUTE
RECORD passes (sc1g_k8.py writes routes/<arm>.<pid>.json at exit; atexit does not run on a signal, so a missing, empty or null
record FAILS the row):
  T == 1 rows (served, pdl0, nofold, chunk1) on the MXFP4 store: `mxfp4_gemv|le256` at least steps x 24 layers, no
                 `mxfp4_*|gt256`, no route outside {mxfp4_gemv|*, nf4_mtile_host|*} (the 512-token prompt runs the kept NF4)
  prefill-shaped at least one `nf4_mtile_host|*` and no `mxfp4_*|gt256` (KEEP_NF4 observed)
  e4b_nf4 rows   a non-empty record with no `mxfp4_*` route at all
The floor (registered): per graded window, |NLL(e4b_serve prefill, chunk 64) - NLL(e4b_serve prefill, chunk 128)|; F = the max
over conv1 and conv2. A delta is "within the floor" iff |delta| <= F on BOTH graded windows. G1-G5 are read on conv1/conv2 only;
G6 is the kernel check on the 5090 (sc1g_gemv_check.py: the kernel against the bf16-rounded exact reference, with a mutation arm
that must disagree). The control and the diagnostics are reported, never graded.

  sc1g_reduce.py --dir SC1G_DIR --out verdict_sc1g.json [--prove]
  sc1g_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRCS = ("conv1", "conv2")          # graded
CTRL = ("wikitext",)               # descriptive
SHAPES = ("prefill", "served")
LAYERS = 24                        # gpt-oss-20b's MoE layers
SAME = 1e-9                        # two NLLs this close are the same arithmetic (the q8 switch did not engage)
# arm -> shape -> receipt stem (format with the window); e4b's quoted prefill row is chunk 128
ARMS = {
    "e4b_serve": {"served": "e4b_serve_served_{}", "prefill": "e4b_serve_prefill128_{}"},
    "e4b_serve_p64": {"prefill": "e4b_serve_prefill64_{}"},
    "e4b_nf4": {"served": "e4b_nf4_served_{}", "prefill": "e4b_nf4_prefill128_{}"},
    "vllm": {"served": "nll_vllm_served_{}", "prefill": "nll_vllm_prefill_{}"},
    "sglang_native": {"served": "nll_sglang_native_served_{}", "prefill": "nll_sglang_native_prefill_{}"},
    "sglang_marlin": {"served": "nll_sglang_marlin_served_{}", "prefill": "nll_sglang_marlin_prefill_{}"},
    "llamacpp": {"served": "nll_llamacpp_decode_{}", "prefill": "nll_llamacpp_prefill_{}"},
    "llamacpp_q8": {"served": "nll_llamacpp_q8_decode_{}", "prefill": "nll_llamacpp_q8_prefill_{}"},
}
# the diagnostics (graded windows only; descriptive)
DIAG = {"e4b_serve_pdl0": "e4b_serve_pdl0_{}", "e4b_serve_nofold": "e4b_serve_nofold_{}",
        "e4b_serve_chunk1": "e4b_serve_chunk1_{}", "e4b_nf4_chunk1": "e4b_nf4_chunk1_{}"}
MEANING = {"pdl0": "moving vs served => an ordering (PDL) bug in the served T == 1 path",
           "nofold": "moving vs served => a fold (E4B_FUSE_*) bug in the served T == 1 path",
           "chunk1_vs_served": "the T == 1 expert route held, attention switched from paged fp8-KV decode to transformers' eager "
                               "attention with a bf16 cache: the share of the served-vs-prefill gap the paged path carries",
           "chunk1_vs_prefill": "attention held (transformers' eager), the expert route switched from T == 1 to the prefill M-tile"}
ARITH = {"e4b_serve": "served: MXFP4 GEMV on int8 activations (W4A8), paged fp8 decode attention; prefill: kept-NF4 host M-tile, transformers' eager attention",
         "e4b_serve_p64": "the floor's second order (chunk 64)",
         "e4b_nf4": "NF4 experts everywhere (the control)",
         "vllm": "Marlin W4A16; TRITON_ATTN",
         "sglang_native": "flashinfer_mxfp4 (cutlass_sm120, W4A8 with MXFP8 activations); triton attention",
         "sglang_marlin": "Marlin W4A16; triton attention",
         "llamacpp": "published GGUF (attention Q8_0); MMVQ W4A8 decode, MMQ W4A4 prefill",
         "llamacpp_q8": "published GGUF (attention Q8_0); GGML_CUDA_MMQ_PREC=q8: W4A8 prefill"}
# the proof's subset (conv1): A1's new paths plus one scoring per comparator engine
PROVE = ("e4b_serve_served_conv1", "e4b_serve_prefill64_conv1", "e4b_serve_chunk1_conv1", "nll_vllm_prefill_conv1",
         "nll_sglang_native_prefill_conv1", "nll_llamacpp_q8_prefill_conv1")


def _sc1():
    for p in (os.path.join(HERE, "sc1_reduce.py"), os.path.join(HERE, "..", "sc1", "sc1_reduce.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("sc1_reduce", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["sc1_reduce"] = m          # its dataclasses resolve their module through sys.modules
            spec.loader.exec_module(m)
            return m
    raise SystemExit("sc1_reduce.py not found beside this file or in ../sc1")


def _load(d, stem):
    p = os.path.join(d, stem + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def _wsha(d, src):
    p = os.path.join(d, f"k8_window_{src}.json")
    return json.load(open(p)).get("text_sha") if os.path.exists(p) else None


def route_gate(d, stem, steps=2048):
    """-> (ok, why, merged route_seen). Reads every routes/<stem>.<pid>.json."""
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "routes", stem + ".*.json")))]
    if not recs:
        return False, "no route record (atexit did not run, or the process never loaded e4b)", None
    seen = {}
    for r in recs:
        if r.get("route_seen") is None:
            return False, "route record is null (the e4b build predates e4b#1129's counters)", None
        for k, v in r["route_seen"].items():
            seen[k] = seen.get(k, 0) + int(v)
    if not seen:
        return False, "route record is empty", seen
    if stem.startswith("e4b_nf4_"):
        bad = [k for k in seen if k.startswith("mxfp4_")]
        return (not bad), (f"MXFP4 route(s) {bad} on the NF4 control" if bad else ""), seen
    gt = [k for k in seen if k.startswith("mxfp4_") and k.endswith("|gt256")]
    if gt:
        return False, f"{gt}: rows above 256 on the MXFP4 store (the NF4 stacks were not kept)", seen
    if any(f"_{x}_" in stem for x in ("served", "pdl0", "nofold", "chunk1")):
        other = [k for k in seen if not (k.startswith("mxfp4_gemv|") or k.startswith("nf4_mtile_host|"))]
        n = seen.get("mxfp4_gemv|le256", 0)
        if other:
            return False, f"route(s) {other} outside the T == 1 GEMV + the prompt's NF4", seen
        if n < steps * LAYERS:
            return False, f"mxfp4_gemv|le256 {n} < {steps} steps x {LAYERS} layers", seen
        return True, "", seen
    if not any(k.startswith("nf4_mtile_host|") for k in seen):
        return False, "no nf4_mtile_host route on the prefill shape", seen
    return True, "", seen


def row(d, stem, src, R):
    q = R.quality_row(_load(d, stem), _wsha(d, src))
    q["stem"] = stem
    if stem.startswith("e4b_") and q["verdict"] == "VALID":
        ok, why, seen = route_gate(d, stem)
        q["route_seen"] = seen
        if not ok:
            q.update(verdict="VOID", why=f"route gate: {why}")
    if stem.startswith("nll_llamacpp"):
        q["mmq_prec_env"] = (_load(d, stem) or {}).get("mmq_prec_env")
    return q


def gemv(d):
    """The kernel check receipts in the box's dir: -> ({name: summary}, G6)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "gemv_*.json"))):
        r = json.load(open(p))
        out[os.path.basename(p)[:-5]] = {"verdict": r.get("verdict"), "kernel_vs_exact_worst": r.get("kernel_vs_exact_worst"),
                                        "metric": r.get("metric"), "device": r.get("device"), "mutate": r.get("mutate"),
                                        "summary": r.get("summary")}
    real = {k: v for k, v in out.items() if not v["mutate"]}
    mut = [v for v in out.values() if v["mutate"]]
    if not real or not mut:
        g6 = {"verdict": "UNREAD", "why": "no kernel check receipt" if not real else "no mutation arm"}
    elif any(v["verdict"] != "KERNEL_DISAGREES" for v in mut):
        g6 = {"verdict": "UNREAD", "why": "the mutation arm AGREED: the check could not see a wrong expert (inert)"}
    else:
        bad = [k for k, v in real.items() if v["verdict"] != "KERNEL_AGREES"]
        g6 = {"verdict": "REFUTED" if bad else "HOLDS", "disagreeing": bad,
              "worst": max((v["kernel_vs_exact_worst"] or 0.0) for v in real.values())}
    return out, g6


def reduce(d: str) -> dict:
    R = _sc1()
    wins = SRCS + CTRL
    rows = {a: {sh: {s: row(d, stem.format(s), s, R) for s in wins} for sh, stem in shp.items()} for a, shp in ARMS.items()}
    diag = {a: {s: row(d, stem.format(s), s, R) for s in SRCS} for a, stem in DIAG.items()}

    def nll(arm, shape, src):
        x = rows.get(arm, {}).get(shape, {}).get(src)
        return x["mean_nll"] if x and x["verdict"] == "VALID" else None

    def dnll(arm, src):
        x = diag[arm][src]
        return x["mean_nll"] if x["verdict"] == "VALID" else None

    def floor(srcs):
        f = {}
        for s in srcs:
            a, b = nll("e4b_serve_p64", "prefill", s), nll("e4b_serve", "prefill", s)
            f[s] = None if None in (a, b) else abs(a - b)
        return f, (None if None in f.values() else max(f.values()))
    fl, F = floor(SRCS)
    flc, Fc = floor(CTRL)

    def delta(a1, a2, shape, srcs=SRCS):
        out = {}
        for s in srcs:
            x, y = nll(a1, shape, s), nll(a2, shape, s)
            out[s] = None if None in (x, y) else round(x - y, 6)
        return out

    def within(dl):
        if F is None or None in dl.values():
            return None
        return all(abs(v) <= F for v in dl.values())

    def sub(a, b):
        return None if None in (a, b) else round(a - b, 6)

    p = {}
    d1 = delta("e4b_serve", "vllm", "served")
    w = within(d1)
    p["G1"] = {"verdict": "UNREAD" if w is None else ("HOLDS" if w else "REFUTED"), "delta": d1, "floor": F}
    pre = {s: nll("e4b_serve", "prefill", s) for s in SRCS}
    srv = {s: nll("e4b_serve", "served", s) for s in SRCS}
    if None in pre.values() or None in srv.values():
        p["G2"] = {"verdict": "UNREAD"}
    else:
        gap = {s: round(pre[s] - srv[s], 6) for s in SRCS}
        p["G2"] = {"verdict": "HOLDS" if all(v >= 0 for v in gap.values()) else "REFUTED", "prefill_minus_served": gap}
    d3 = delta("llamacpp", "llamacpp_q8", "prefill")
    if None not in d3.values() and all(abs(v) < SAME for v in d3.values()):
        p["G3"] = {"verdict": "UNREAD", "why": "NOT_ENGAGED: default and q8 prefill NLL identical (the switch did not take)", "delta": d3}
    else:
        w = within(d3)
        p["G3"] = {"verdict": "UNREAD" if w is None else ("HOLDS" if not w else "REFUTED"), "delta": d3, "floor": F}
    d4 = {sh: delta("vllm", "sglang_marlin", sh) for sh in SHAPES}
    w4 = [within(d4[sh]) for sh in SHAPES]
    p["G4"] = {"verdict": "UNREAD" if None in w4 else ("HOLDS" if all(w4) else "REFUTED"), "matched_arm_disagreement": d4, "floor": F}
    st = {f"{a}/{sh}/{s}": rows[a][sh][s]["verdict"] for a, shp in ARMS.items() for sh in shp for s in SRCS}
    p["G5"] = {"verdict": "HOLDS" if all(v == "VALID" for v in st.values()) else "REFUTED",
               "not_valid": {k: v for k, v in st.items() if v != "VALID"}}
    kern, p["G6"] = gemv(d)

    # ---- descriptive: the diagnostics, the served-vs-prefill gap per arm, the control
    dg = {}
    for s in SRCS:
        srv_m, pre_m = nll("e4b_serve", "served", s), nll("e4b_serve", "prefill", s)
        srv_n, pre_n = nll("e4b_nf4", "served", s), nll("e4b_nf4", "prefill", s)
        c1m, c1n = dnll("e4b_serve_chunk1", s), dnll("e4b_nf4_chunk1", s)
        frac = {}
        for k, (sv, c1, pr) in {"mxfp4": (srv_m, c1m, pre_m), "nf4": (srv_n, c1n, pre_n)}.items():
            frac[k] = None if None in (sv, c1, pr) or abs(sv - pr) < 1e-12 else round((sv - c1) / (sv - pr), 4)
        dg[s] = {"pdl0_minus_served": sub(dnll("e4b_serve_pdl0", s), srv_m),
                 "nofold_minus_served": sub(dnll("e4b_serve_nofold", s), srv_m),
                 "mxfp4": {"served_minus_prefill": sub(srv_m, pre_m), "served_minus_chunk1": sub(srv_m, c1m),
                           "chunk1_minus_prefill": sub(c1m, pre_m)},
                 "nf4": {"served_minus_prefill": sub(srv_n, pre_n), "served_minus_chunk1": sub(srv_n, c1n),
                         "chunk1_minus_prefill": sub(c1n, pre_n)},
                 "share_of_served_gap_closed_by_chunk1": frac}
    gaps = {a: {s: sub(nll(a, "served", s), nll(a, "prefill", s)) for s in SRCS + CTRL}
            for a, shp in ARMS.items() if "served" in shp and "prefill" in shp}
    arms_cmp = ("e4b_serve", "e4b_nf4", "sglang_native", "sglang_marlin", "llamacpp", "llamacpp_q8")
    rep = {a: {sh: delta(a, "vllm", sh) for sh in SHAPES} for a in arms_cmp}
    ctl = {a: {sh: delta(a, "vllm", sh, CTRL) for sh in SHAPES} for a in arms_cmp}
    roles = {}
    for s in SRCS:
        p_ = os.path.join(d, f"k8_window_{s}.json")
        roles[s] = json.load(open(p_)).get("scored_target_roles") if os.path.exists(p_) else None
    return {"rows": rows, "diagnostic_rows": diag, "arith": ARITH, "floor": {"per_window": fl, "F": F},
            "predictions": p, "delta_vs_vllm": rep,
            "within_floor_vs_vllm": {a: {sh: within(v) for sh, v in by.items()} for a, by in rep.items()},
            "diagnostics": {"meaning": MEANING, "per_window": dg}, "served_minus_prefill": gaps, "kernel_check": kern,
            "scored_target_roles": roles,
            "control_wikitext": {"floor": {"per_window": flc, "F": Fc}, "delta_vs_vllm": ctl,
                                 "note": "descriptive: out of distribution for gpt-oss (the proof read ppl ~560)"}}


def prove(d: str) -> dict:
    R = _sc1()
    out = {stem: row(d, stem, "conv1", R) for stem in PROVE}
    bad = {k: f"{v['verdict']}: {v['why']}" for k, v in out.items() if v["verdict"] != "VALID"}
    kern, g6 = gemv(d)
    if g6["verdict"] != "HOLDS":
        bad["kernel_check"] = f"G6 {g6['verdict']}: {g6.get('why') or g6.get('disagreeing')}"
    if not any(k.startswith("gemv_capture_conv1") for k in kern):
        bad["kernel_check_capture"] = "no kernel check on captured conv1 activations"
    return {"rows": out, "kernel_check": kern, "G6": g6, "bad": bad, "proved": not bad}


# ------------------------------------------------------------------ self-test --

def _fixture(d, nll, routes, sha="s" * 64, steps=2048, gemv_v=("KERNEL_AGREES", "KERNEL_DISAGREES")):
    for s in SRCS + CTRL:
        json.dump({"ids": [0] * 2561, "text_sha": sha, "source": s, "prompt_len": 512, "steps": 2048,
                   "scored_target_roles": {"assistant": 0.95}}, open(os.path.join(d, f"k8_window_{s}.json"), "w"))
    os.makedirs(os.path.join(d, "routes"), exist_ok=True)
    stems = [(a, sh, stem.format(s)) for a, shp in ARMS.items() for sh, stem in shp.items() for s in SRCS + CTRL]
    stems += [(a, "diag", stem.format(s)) for a, stem in DIAG.items() for s in SRCS]
    for a, sh, st in stems:
        s = st.rsplit("_", 1)[1]
        v = nll(a, sh, s)
        if v is None:
            continue
        if st.startswith("nll_llamacpp"):
            rec = {"mean_nll": v, "text_sha256": sha, "targets": {"first_index": 513, "last_index": 2560}, "steps": steps}
        else:
            rec = {"mean_nll": v, "text_sha": sha, "prompt_len": 512, "steps": steps}
        json.dump(rec, open(os.path.join(d, st + ".json"), "w"))
        if st.startswith("e4b_"):
            r = routes(st)
            if r is not False:
                json.dump({"pid": 1, "route_seen": r, "attn_seen": None}, open(os.path.join(d, "routes", st + ".1.json"), "w"))
    for name, v, mut in (("gemv_capture_conv1_5090", gemv_v[0], False), ("gemv_mutate_5090", gemv_v[1], True)):
        json.dump({"verdict": v, "kernel_vs_exact_worst": 1e-6 if v == "KERNEL_AGREES" else 0.9, "metric": "kernel_vs_exact_bf16",
                   "device": {"cc": [12, 0]}, "mutate": mut, "summary": {}}, open(os.path.join(d, name + ".json"), "w"))


def _good_routes(st):
    if st.startswith("e4b_nf4_"):
        return {"nf4_singleton|le256": 49152, "nf4_mtile_host|gt256": 24}
    if any(f"_{x}_" in st for x in ("served", "pdl0", "nofold", "chunk1")):
        return {"mxfp4_gemv|le256": 2048 * 24 + 16 * 24, "nf4_mtile_host|gt256": 24}
    return {"nf4_mtile_host|gt256": 400}


def _base(a, sh, s):
    b = {"conv1": 1.2, "conv2": 1.4, "wikitext": 6.3}[s]
    off = {"e4b_serve": 0.004 if sh == "served" else 0.012, "e4b_serve_p64": 0.012 + (0.006 if s == "conv1" else -0.004),
           "e4b_nf4": 0.02, "vllm": 0.0, "sglang_native": 0.003, "sglang_marlin": 0.002,
           "llamacpp": 0.002 if sh == "served" else 0.04, "llamacpp_q8": 0.002 if sh == "served" else 0.005,
           "e4b_serve_pdl0": 0.004, "e4b_serve_nofold": 0.004, "e4b_serve_chunk1": 0.008, "e4b_nf4_chunk1": 0.02}[a]
    return b + off


def self_test() -> int:
    import tempfile
    cases = []
    with tempfile.TemporaryDirectory() as d:          # every prediction holds; F = 0.006 (conv1's chunk pair)
        _fixture(d, _base, _good_routes)
        v = reduce(d)
        cases.append(("all hold", all(v["predictions"][g]["verdict"] == "HOLDS" for g in ("G1", "G2", "G3", "G4", "G5", "G6"))
                      and abs(v["floor"]["F"] - 0.006) < 1e-9 and v["diagnostics"]["per_window"]["conv1"]["pdl0_minus_served"] == 0.0))
    with tempfile.TemporaryDirectory() as d:          # an e4b arm killed by alarm: no route record -> VOID, never a pass
        _fixture(d, _base, lambda st: False if st == "e4b_serve_served_conv2" else _good_routes(st))
        v = reduce(d)
        cases.append(("missing record fails", v["rows"]["e4b_serve"]["served"]["conv2"]["verdict"] == "VOID"
                      and v["predictions"]["G1"]["verdict"] == "UNREAD" and v["predictions"]["G5"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # KEEP_NF4 not engaged: prefill rows on the MXFP4 store above 256 rows
        _fixture(d, _base, lambda st: {"mxfp4_gemv|gt256": 400} if "prefill" in st and st.startswith("e4b_serve") else _good_routes(st))
        v = reduce(d)
        cases.append(("mxfp4 gt256 fails", v["rows"]["e4b_serve"]["prefill"]["conv1"]["verdict"] == "VOID" and v["floor"]["F"] is None))
    with tempfile.TemporaryDirectory() as d:          # a pre-#1129 build: null record
        _fixture(d, _base, lambda st: None if st == "e4b_nf4_served_conv1" else _good_routes(st))
        cases.append(("null record fails", reduce(d)["rows"]["e4b_nf4"]["served"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # q8 did not engage: identical prefill -> G3 UNREAD
        _fixture(d, lambda a, sh, s: _base("llamacpp", sh, s) if a == "llamacpp_q8" else _base(a, sh, s), _good_routes)
        cases.append(("q8 not engaged", reduce(d)["predictions"]["G3"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:          # e4b served 0.03 off vLLM on the graded windows: outside a 0.006 floor
        _fixture(d, lambda a, sh, s: _base(a, sh, s) + (0.026 if (a, sh) == ("e4b_serve", "served") else 0), _good_routes)
        cases.append(("G1 refuted", reduce(d)["predictions"]["G1"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # the control is never graded: a large wikitext gap leaves G1 holding
        _fixture(d, lambda a, sh, s: _base(a, sh, s) + (0.5 if (a, sh, s) == ("e4b_serve", "served", "wikitext") else 0), _good_routes)
        v = reduce(d)
        cases.append(("control ungraded", v["predictions"]["G1"]["verdict"] == "HOLDS"
                      and v["control_wikitext"]["delta_vs_vllm"]["e4b_serve"]["served"]["wikitext"] > 0.4))
    with tempfile.TemporaryDirectory() as d:          # a row on other text: sha mismatch VOIDs it
        _fixture(d, _base, _good_routes)
        json.dump({"mean_nll": 2.0, "text_sha": "x" * 64, "prompt_len": 512, "steps": 2048, "scored_index_range": [513, 2560]},
                  open(os.path.join(d, "nll_vllm_served_conv1.json"), "w"))
        cases.append(("sha VOID", reduce(d)["rows"]["vllm"]["served"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # the proof's subset alone proves
        _fixture(d, _base, _good_routes)
        for p in glob.glob(os.path.join(d, "*_conv2.json")) + glob.glob(os.path.join(d, "*_wikitext.json")):
            os.remove(p)
        cases.append(("proof subset", prove(d)["proved"]))
    with tempfile.TemporaryDirectory() as d:          # a served record short of the steps x layers GEMV count
        _fixture(d, _base, lambda st: {"mxfp4_gemv|le256": 100, "nf4_mtile_host|gt256": 24} if st == "e4b_serve_served_conv1" else _good_routes(st))
        cases.append(("gemv count", reduce(d)["rows"]["e4b_serve"]["served"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # the mutation agreed: the check is inert -> G6 UNREAD, the proof fails
        _fixture(d, _base, _good_routes, gemv_v=("KERNEL_AGREES", "KERNEL_AGREES"))
        cases.append(("inert check", reduce(d)["predictions"]["G6"]["verdict"] == "UNREAD" and not prove(d)["proved"]))
    with tempfile.TemporaryDirectory() as d:          # the kernel disagrees with its exact reference on the 5090
        _fixture(d, _base, _good_routes, gemv_v=("KERNEL_DISAGREES", "KERNEL_DISAGREES"))
        cases.append(("kernel disagrees", reduce(d)["predictions"]["G6"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # chunk 1 closes the served gap: share ~1 (the paged path carries it)
        _fixture(d, lambda a, sh, s: _base(a, sh, s) + (0.25 if (a, sh) == ("e4b_serve", "served") else 0), _good_routes)
        share = reduce(d)["diagnostics"]["per_window"]["conv1"]["share_of_served_gap_closed_by_chunk1"]["mxfp4"]
        cases.append(("chunk1 share", share is not None and share > 0.9))
    bad = [n for n, ok in cases if not ok]
    print(f"sc1g_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--prove", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("--dir is required")
    v = prove(a.dir) if a.prove else reduce(a.dir)
    if a.out:
        json.dump(v, open(a.out, "w"), indent=1, sort_keys=True)
    if a.prove:
        for k, why in v["bad"].items():
            print(f"SC1G_PROVE_BAD {k}: {why}")
        print(f"SC1G_PROVE {'OK' if v['proved'] else 'FAILED'} ({len(v['rows'])} rows; G6 {v['G6']['verdict']})")
        return 0 if v["proved"] else 1
    for g, x in v["predictions"].items():
        print(f"SC1G_{g} {x['verdict']} {json.dumps({k: w for k, w in x.items() if k not in ('verdict', 'not_valid')})[:300]}")
    print(f"SC1G_FLOOR {json.dumps(v['floor'])}")
    print(f"SC1G_DIAG {json.dumps(v['diagnostics']['per_window'])[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
