#!/usr/bin/env python3
"""p63_reduce.py -- lane P63's reducer: the arm receipts against the pre-registration, literally (bench/p63/P63-PREREG.md).

    python p63_reduce.py <out-dir> [--md RESULTS-p63-generated.md] [--json p63_rep.json]

``<out-dir>`` holds one ``<stack>/p63_arm.json`` per stack (nf4, int4, int4nf). Every prediction below is the
registered one; a reading the receipts do not carry is UNREAD, never inferred. Nothing here is tuned to a result: the
tables are the pre-registration's, and ``tests/test_p63_reduce.py`` fires every verdict branch on synthetic receipts.

Gates first (G0): a stack whose control or verify repeat is not bit-identical, or whose module replay at n = 1 does not
reproduce the recorded T = 1 output, is an INSTRUMENT FAULT -- nothing from that stack is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

#: P1 -- kernel census route classes (layer-0 gate_up, the control's own routing), at every T in the census.
#: (stack family, path at T=1, path at T tokens) -> predicted class. "nf4.decode" resolves to the decode path the
#: box's dispatch actually ran (nf4.dotpad on >= 160-SM parts at the Qwen3 census shapes, else nf4.gemv_scalar).
P1_CENSUS = {
    ("int4", "int4.gemv", "int4.gemv"): "EXACT",
    ("int4", "int4.gemv", "int4.deq_bf16"): "PRECISION",
    ("int4", "int4.gemv", "int4.grouped_gemm"): "REORDER",
    ("nf4", "nf4.decode", "nf4.decode"): "EXACT",
    ("nf4", "nf4.decode", "nf4.mtile"): "PRECISION",
    ("nf4", "nf4.gemv_scalar[dotpad=0]", "nf4.gemv_scalar[dotpad=0]"): "REORDER",
}

#: P2 -- module replay: is every row of an n-row call the module's T = 1 output? (stack, sub-arm, module kind) ->
#: {n: "EXACT" | "NOT"}; a kind absent from a sub-arm's table carries no prediction there (reported, not decided).
_EXP_NOT = {16: "NOT", 17: "NOT", 160: "NOT"}
_EXP_DEV = {16: "EXACT", 17: "EXACT", 160: "NOT"}
_EXP_ALL = {16: "EXACT", 17: "EXACT", 160: "EXACT"}
P2_REPLAY = {
    ("int4", "hf.default", "experts"): _EXP_NOT,
    ("int4", "hf.device", "experts"): _EXP_DEV,
    ("int4", "hf.singleton", "experts"): _EXP_ALL,
    ("int4", "hf.combine0", "experts"): _EXP_NOT,
    ("int4", "paged.default", "experts"): _EXP_NOT,
    ("int4", "paged.device", "experts"): _EXP_DEV,
    ("int4", "hf.default", "attn"): _EXP_NOT,
    ("int4", "hf.default", "lm_head"): _EXP_NOT,
    ("int4", "hf.default", "router"): _EXP_NOT,
    ("int4", "hf.default", "input_layernorm"): {16: "EXACT", 17: "EXACT", 160: "NOT"},
    ("int4nf", "hf.default", "experts"): _EXP_NOT,
    ("int4nf", "hf.device", "experts"): _EXP_DEV,
    ("int4nf", "paged.device", "experts"): _EXP_DEV,
    ("int4nf", "hf.default", "attn"): _EXP_NOT,
    ("int4nf", "hf.default", "lm_head"): _EXP_NOT,
    ("nf4", "hf.default", "experts"): _EXP_NOT,
    ("nf4", "hf.singleton", "experts"): _EXP_ALL,
    ("nf4", "hf.device", "experts"): _EXP_NOT,
    ("nf4", "hf.combine0", "experts"): _EXP_NOT,
    ("nf4", "hf.singleton.dotpad0", "experts"): _EXP_NOT,
    ("nf4", "paged.default", "experts"): _EXP_NOT,
    ("nf4", "hf.default", "attn"): _EXP_NOT,
    ("nf4", "hf.default", "lm_head"): _EXP_NOT,
    ("nf4", "hf.default", "router"): _EXP_NOT,
}

#: P3 -- combine census: combine_rows' row for a token does not depend on T (one program per token, slot order).
P3_COMBINE = {"combine_rows": "EXACT"}           # the torch chain carries no prediction (torch picks its layout)

#: P4 -- T = 1 controls across sub-arms of one stack: "EXACT" = logits bit-equal at every position (the grouping
#: flags act only at T > 1); "NOT" = a different T = 1 arithmetic (combine0: B393's reorder-class difference;
#: dotpad0: the scalar GEMV; paged: fp8 KV). The combine0 row also carries B393's end-to-end size (P5).
P4_CROSS = {
    "hf.singleton": "EXACT", "hf.device": "EXACT", "hf.combine0": "NOT", "hf.singleton.dotpad0": "NOT",
    "paged.default": "NOT", "paged.device": "NOT",
}
#: P5 -- the E4B_FUSE_COMBINE=0 control vs the default control (B393's hand-off): reorder-class size bands.
P5_HOLD = {"kl_mean_max": 0.01, "flip_frac_max": 0.03}
P5_REFUTE = {"kl_mean_min": 0.03}

#: P7 -- router output dtype by row count (found by the NAS rehearsal, registered before the lane's data): with
#: E4B_FUSE_ROUTER_EPI=1 the fused router returns fp32 routing weights up to 64 rows (router_epilogue.py:35) and the
#: upstream router bf16 above. stack -> {n: differs?}. On the stacks without the fold no dtype ever differs.
P7_ROUTER_DTYPE = {"int4": {16: False, 17: False, 160: True}, "int4nf": {16: False, 17: False, 160: False},
                   "nf4": {16: False, 17: False, 160: False}}

#: P6 -- end to end, every sub-arm and mode: NOT exact, the first difference at layer 0; the size is read against
#: the band and the shipped KL-from-checkpoint bar (the outer limit a same-model arithmetic change may reach).
P6_BAND = {"kl_mean_max": 0.02, "top1_min": 0.95}
SHIPPED_BAR = {"kl_mean_max": 0.10, "top1_min": 0.93}


def _kind(name: str) -> str:
    if name == "lm_head":
        return "lm_head"
    part = name.split(".", 1)[1]
    return "attn" if part.startswith("attn.") else part


def _load(out_dir: str) -> dict:
    arms = {}
    for stack in ("nf4", "int4", "int4nf"):
        p = os.path.join(out_dir, stack, "p63_arm.json")
        if os.path.exists(p):
            arms[stack] = json.load(open(p))
        elif os.path.exists(p + ".gz"):              # committed rehearsal receipts are gzipped
            import gzip
            with gzip.open(p + ".gz", "rt") as f:
                arms[stack] = json.load(f)
    return arms


def gate(arm: dict) -> dict:
    """G0 for one stack. Returns {"ok": bool, "why": [...]}."""
    why = []
    for name, sa in arm.get("subarms", {}).items():
        if "error" in sa:
            continue
        for k, v in sa.get("determinism", {}).items():
            if v is not True:
                why.append(f"{name}: {k} is {v}")
        for mod, r in sa.get("replay", {}).items():
            if "replay_faithful_at_1" in r and not r["replay_faithful_at_1"]:
                why.append(f"{name}: replay of {mod} at n=1 does not reproduce its T=1 output")
    return {"ok": not why, "why": why}


def _census_decode_path(arm: dict) -> str | None:
    for r in arm.get("kernel_census", {}).get("records", []):
        if r["path"] in ("nf4.dotpad", "nf4.gemv_scalar") and r["T"] == 1:
            return r["path"]
    return None


def p1(arms: dict) -> list:
    rows = []
    for (fam, pa, pb), want in P1_CENSUS.items():
        for stack, arm in arms.items():
            if not stack.startswith(fam):
                continue
            kc = arm.get("kernel_census")
            dec = _census_decode_path(arm)
            a = dec if pa == "nf4.decode" else pa
            b = dec if pb == "nf4.decode" else pb
            if not kc or "pairs" not in kc or (fam == "nf4" and dec is None):
                rows.append({"stack": stack, "pair": f"{pa} -> {pb}", "want": want, "got": None, "verdict": "UNREAD"})
                continue
            got = {p["b"].split("@T=")[1]: p["class"] for p in kc["pairs"]
                   if p["a"] == f"{a}@T=1" and p["b"].startswith(f"{b}@T=")}
            if not got:
                rows.append({"stack": stack, "pair": f"{a} -> {b}", "want": want, "got": None, "verdict": "UNREAD"})
                continue
            bad = {T: c for T, c in got.items() if c != want}
            defect = any(c == "DEFECT?" for c in got.values())
            rows.append({"stack": stack, "pair": f"{a} -> {b}", "want": want, "got": got,
                         "verdict": "DEFECT?" if defect else ("HELD" if not bad else "REFUTED")})
    return rows


def p2(arms: dict) -> list:
    rows = []
    for (stack, sub, kind), want in P2_REPLAY.items():
        sa = arms.get(stack, {}).get("subarms", {}).get(sub)
        if sa is None or "error" in sa or "replay" not in sa:
            rows.append({"stack": stack, "subarm": sub, "kind": kind, "want": want, "got": None, "verdict": "UNREAD"})
            continue
        got = {}
        for mod, r in sa["replay"].items():
            if _kind(mod) != kind:
                continue
            for n, e in r.get("by_n", {}).items():
                n = int(n)
                if n not in want or "rows" not in e:
                    continue
                exact = e["rows_bit_equal"] == e["rows"]
                got.setdefault(n, []).append(exact)
        if not got:
            rows.append({"stack": stack, "subarm": sub, "kind": kind, "want": want, "got": None, "verdict": "UNREAD"})
            continue
        # "EXACT" at n needs every module of the kind exact; "NOT" is held if ANY module of the kind is not
        obs = {n: ("EXACT" if all(v) else "NOT") for n, v in got.items()}
        bad = {n: o for n, o in obs.items() if o != want[n]}
        rows.append({"stack": stack, "subarm": sub, "kind": kind, "want": {str(k): v for k, v in want.items()},
                     "got": {str(k): v for k, v in obs.items()},
                     "modules": {str(n): f"{sum(v)}/{len(v)} exact" for n, v in got.items()},
                     "verdict": "HELD" if not bad else "REFUTED"})
    return rows


def p3(arms: dict) -> list:
    rows = []
    for stack, arm in arms.items():
        cc = arm.get("combine_census")
        if not cc:
            continue
        for path in ("combine_rows", "chain"):
            recs = [r for r in cc["records"] if r["path"] == path and r["T"] > 1]
            if not recs:
                continue
            exact = all(r["rows_bit_equal"] == r["rows"] for r in recs)
            ratio = max(r["bound_ratio"] for r in cc["records"] if r["path"] == path)
            want = P3_COMBINE.get(path)
            got = "EXACT" if exact else "NOT"
            v = ("DEFECT?" if ratio > 1 else ("INFO" if want is None else ("HELD" if got == want else "REFUTED")))
            rows.append({"stack": stack, "path": path, "want": want, "got": got, "max_bound_ratio": ratio,
                         "vs_chain": [r.get("bit_equal_to_chain") for r in recs if "bit_equal_to_chain" in r],
                         "verdict": v})
    return rows


def p4_p5(arms: dict) -> tuple:
    r4, r5 = [], []
    for stack, arm in arms.items():
        for key, c in arm.get("cross_controls", {}).items():
            other = key.split(" vs ")[1].split(" (")[0]
            want = P4_CROSS.get(other)
            got = "EXACT" if c["logits_bit_equal_positions"] == c["positions"] else "NOT"
            r4.append({"stack": stack, "pair": key, "want": want, "got": got,
                       "argmax_flips": c["argmax_flips"], "kl_mean": c["kl_mean"],
                       "first_layer_not_bit_equal": c["first_layer_not_bit_equal"],
                       "verdict": "INFO" if want is None else ("HELD" if got == want else "REFUTED")})
            if other == "hf.combine0":
                frac = c["argmax_flips"] / max(1, c["positions"])
                if c["kl_mean"] > P5_REFUTE["kl_mean_min"]:
                    v = "REFUTED"
                elif c["kl_mean"] <= P5_HOLD["kl_mean_max"] and frac <= P5_HOLD["flip_frac_max"]:
                    v = "HELD"
                else:
                    v = "BETWEEN"
                r5.append({"stack": stack, "kl_mean": c["kl_mean"], "kl_max": c["kl_max"], "flip_frac": frac,
                           "first_layer_not_bit_equal": c["first_layer_not_bit_equal"], "verdict": v})
    return r4, r5


def p6(arms: dict) -> list:
    rows = []
    for stack, arm in arms.items():
        for sub, sa in arm.get("subarms", {}).items():
            if "error" in sa:
                rows.append({"stack": stack, "subarm": sub, "mode": "*", "verdict": "UNREAD", "error": sa["error"][:200]})
                continue
            for mode, m in sa.get("modes", {}).items():
                top1 = 1.0 - m["argmax_flips"] / max(1, m["positions"])
                exact = m["exact_positions"] == m["positions"]
                layer0 = m["min_first_diff_layer"] == 0
                if m["kl_mean"] > SHIPPED_BAR["kl_mean_max"] or top1 < SHIPPED_BAR["top1_min"]:
                    size = "OVER-BAR"
                elif m["kl_mean"] <= P6_BAND["kl_mean_max"] and top1 >= P6_BAND["top1_min"]:
                    size = "IN-BAND"
                else:
                    size = "BETWEEN"
                rows.append({"stack": stack, "subarm": sub, "mode": mode, "positions": m["positions"],
                             "exact_positions": m["exact_positions"], "min_first_diff_layer": m["min_first_diff_layer"],
                             "first_diff_site_hist": m["first_diff_site_hist"], "argmax_flips": m["argmax_flips"],
                             "top1": round(top1, 4), "kl_mean": m["kl_mean"], "kl_max": m["kl_max"], "size": size,
                             "verdict": "EXACT (prediction REFUTED)" if exact else
                                        ("HELD" if layer0 else "HELD-not-layer0")})
    return rows


def _ratio(rec: dict):
    """The DEFECT-line ratio of one record: the fp32-reduction rerun for a cuBLAS path, else the default-flag ratio."""
    r = rec.get("bound_ratio_fp32_reduction", rec.get("bound_ratio"))
    return None if r is None else float(r)


def defects(arms: dict) -> dict:
    """Every accuracy record against fp64 -- kernel census, module replay, combine census -- whose DEFECT-line ratio
    exceeds 1, plus the cuBLAS records whose default-flag ratio exceeds 1 while the fp32-reduction rerun does not (the
    reduced-precision-reduction attribution, reported, not a defect)."""
    found, attributed, n = [], [], 0
    for stack, arm in arms.items():
        recs = [("census", r) for r in arm.get("kernel_census", {}).get("records", [])]
        recs += [("combine", r) for r in arm.get("combine_census", {}).get("records", [])]
        for sub, sa in arm.get("subarms", {}).items():
            for mod, r in sa.get("replay", {}).items():
                for nn, e in r.get("by_n", {}).items():
                    if "bound_ratio" in e:
                        recs.append((f"replay {sub} {mod} n={nn}", e))
        for where, r in recs:
            rr = _ratio(r)
            if rr is None:
                continue
            n += 1
            tag = {"stack": stack, "where": where, "path": r.get("path"), "T": r.get("T"),
                   "ratio": rr, "ratio_default_flags": r.get("bound_ratio")}
            if rr > 1.0:
                found.append(tag)
            elif (r.get("bound_ratio") or 0) > 1.0:
                attributed.append(tag)
    return {"records": n, "defects": found, "reduced_precision_reduction_only": attributed,
            "verdict": "DEFECT?" if found else "NONE"}


def p7(arms: dict) -> list:
    rows = []
    for stack, want in P7_ROUTER_DTYPE.items():
        arm = arms.get(stack)
        sa = (arm or {}).get("subarms", {}).get("hf.default")
        if not sa or "replay" not in sa:
            rows.append({"stack": stack, "want": want, "got": None, "verdict": "UNREAD"})
            continue
        got = {}
        for mod, r in sa["replay"].items():
            if _kind(mod) != "router":
                continue
            for nn, e in r.get("by_n", {}).items():
                if int(nn) in want and "rows" in e:
                    got[int(nn)] = got.get(int(nn), False) or bool(e.get("dtype_differs"))
        if not got:
            rows.append({"stack": stack, "want": want, "got": None, "verdict": "UNREAD"})
            continue
        bad = {k: v for k, v in got.items() if v != want[k]}
        rows.append({"stack": stack, "want": {str(k): v for k, v in want.items()},
                     "got": {str(k): v for k, v in got.items()}, "verdict": "HELD" if not bad else "REFUTED"})
    return rows


def reduce(out_dir: str) -> dict:
    arms = _load(out_dir)
    gates = {s: gate(a) for s, a in arms.items()}
    readable = {s: a for s, a in arms.items() if gates[s]["ok"]}
    r4, r5 = p4_p5(readable)
    return {"stacks_present": sorted(arms), "gates": gates,
            "devices": {s: a.get("device") for s, a in arms.items()},
            "census": {s: a.get("census") for s, a in arms.items()},
            "refused": {s: a.get("refused") for s, a in arms.items()},
            "P1": p1(readable), "P2": p2(readable), "P3": p3(readable), "P4": r4, "P5": r5, "P6": p6(readable),
            "P7": p7(readable), "D": defects(readable)}


def to_md(rep: dict) -> str:
    L = ["# P63 — generated results (p63_reduce.py; read against P63-PREREG.md)", ""]
    L.append(f"Stacks: {', '.join(rep['stacks_present']) or 'none'}; devices: {rep['devices']}")
    L.append("")
    L.append("## G0 — instrument gates")
    for s, g in rep["gates"].items():
        L.append(f"- **{s}**: {'OK' if g['ok'] else 'INSTRUMENT FAULT — nothing read'}" +
                 ("" if g["ok"] else ": " + "; ".join(g["why"][:6])))
    for s, r in rep["refused"].items():
        for name, why in (r or {}).items():
            L.append(f"- {s}/{name}: {why}")
    L.append("")
    L.append("## P1 — kernel census (layer-0 gate_up)")
    L.append("| stack | route pair | predicted | observed (by T) | verdict |")
    L.append("|---|---|---|---|---|")
    for r in rep["P1"]:
        L.append(f"| {r['stack']} | {r['pair']} | {r['want']} | {r['got']} | {r['verdict']} |")
    L.append("")
    L.append("## P2 — module replay (same bytes in)")
    L.append("| stack | sub-arm | module | predicted | observed | modules exact | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rep["P2"]:
        L.append(f"| {r['stack']} | {r['subarm']} | {r['kind']} | {r['want']} | {r.get('got')} | "
                 f"{r.get('modules', '')} | {r['verdict']} |")
    L.append("")
    L.append("## P3 — combine census")
    for r in rep["P3"]:
        L.append(f"- {r['stack']} {r['path']}: {r['got']} (predicted {r['want']}), max bound ratio "
                 f"{r['max_bound_ratio']:.3f}, vs chain {r['vs_chain']} — **{r['verdict']}**")
    L.append("")
    L.append("## P4 / P5 — T = 1 controls across sub-arms")
    for r in rep["P4"]:
        L.append(f"- {r['stack']} {r['pair']}: {r['got']} (predicted {r['want']}); flips {r['argmax_flips']}, "
                 f"KL mean {r['kl_mean']:.2e}, first non-equal layer {r['first_layer_not_bit_equal']} — **{r['verdict']}**")
    for r in rep["P5"]:
        L.append(f"- **P5** {r['stack']} combine0 vs default: KL mean {r['kl_mean']:.2e} (max {r['kl_max']:.2e}), "
                 f"flip fraction {r['flip_frac']:.3f} — **{r['verdict']}**")
    L.append("")
    L.append("## P7 — router output dtype by row count")
    for r in rep["P7"]:
        L.append(f"- {r['stack']}: dtype differs by n {r.get('got')} (predicted {r['want']}) — **{r['verdict']}**")
    d = rep["D"]
    L.append("")
    L.append(f"## D — accuracy against fp64 (the DEFECT line): {d['verdict']} over {d['records']} records")
    for x in d["defects"]:
        L.append(f"- DEFECT? {x['stack']} {x['where']} {x['path']} T={x['T']}: ratio {x['ratio']:.3f}")
    if d["reduced_precision_reduction_only"]:
        L.append(f"- {len(d['reduced_precision_reduction_only'])} cuBLAS record(s) exceed the bound only under torch's "
                 "default bf16 reduced-precision reduction (inside it with fp32 split-K reduction); max default-flag "
                 f"ratio {max(x['ratio_default_flags'] for x in d['reduced_precision_reduction_only']):.3f}")
    L.append("")
    L.append("## P6 — end to end (control vs verify / prefill)")
    L.append("| stack | sub-arm | mode | exact / positions | first diff layer | first-diff sites | flips | top-1 | "
             "KL mean | KL max | size | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rep["P6"]:
        if r.get("mode") == "*":
            L.append(f"| {r['stack']} | {r['subarm']} | * | | | | | | | | | UNREAD ({r.get('error', '')}) |")
            continue
        L.append(f"| {r['stack']} | {r['subarm']} | {r['mode']} | {r['exact_positions']}/{r['positions']} | "
                 f"{r['min_first_diff_layer']} | {r['first_diff_site_hist']} | {r['argmax_flips']} | {r['top1']} | "
                 f"{r['kl_mean']:.2e} | {r['kl_max']:.2e} | {r['size']} | {r['verdict']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rep = reduce(a.out_dir)
    md = to_md(rep)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    print(md)
    return 0 if rep["stacks_present"] and all(g["ok"] for g in rep["gates"].values()) else 14


if __name__ == "__main__":
    sys.exit(main())
