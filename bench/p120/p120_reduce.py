#!/usr/bin/env python3
"""Lane P120's reducer (bench/p120/PREREG-p120.md): checks the box ran as registered, then applies the registered rule
to ``E4B_INT4_WIDE_TILES`` at 64 decode rows. Every bound is a ratio inside the box.

In order, the first that fires is the verdict:
- **NO_READING**: no box record.
- **VOID**: the e4b or grouped-nf4-gemm commit or the model revision is not the registered one; the reading's stack is
  not SC2e's; a profile arm did not profile 8 padded eager steps of bucket 64 (``capture=False``) under its setting;
  a served arm is missing, was not captured under its setting (every bucket ``graph``), or did not replay bucket 64
  exactly ``warm + steps`` times with no eager step and no other bucket; KV bookkeeping is not bulk; a step or token
  count is short; **not engaged** (with ``1`` the profile shows a radix sort, or not one ``_tile_table_r1`` launch for
  each chained build ``0`` showed); **nondeterministic** (``OFF_a`` and ``OFF_b``, or ``ON_a`` and ``ON_b``, emit
  different tokens); **the mutant survived** (its tokens equal ``OFF_a``'s: the token gate could not fail).
- **PREMISE_ABSENT** (the reading only): with ``0`` the profile shows fewer chained builds than MoE layers per step, or
  the chained builder's frozen kernels (``BUILDER``) are under ``PREMISE_SHARE`` of the eager step's device time.
- **TOKENS_DIFFER**: ``ON_a`` emits a token ``OFF_a`` does not, at any decode step and row.
- **NOISY**: the two OFF arms' medians, or the two ON arms', differ by more than ``NOISE``.
- **DEFAULT_ON**: ``ON_a / OFF_a`` and ``ON_b / OFF_b`` (median served step) are both at most ``BAR``.
- **SLOWER**: both above 1. **NO_GAIN**: otherwise.

The predictions (Q1-Q5) are evaluated beside the verdict and never change it. Floats are summed with ``math.fsum`` and
medians taken by sorting, so the output is byte-identical on any Python.

    p120_reduce.py --dir RUN --out verdict.json [--e4b-sha SHA]
    p120_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

GNF4_SHA = "3ce2ecd8ae20725a6b13ccc6d06c6a85ba5ce3ac"     # grouped-nf4-gemm 0.43.0 + #515 (the cumsum rank), its merge commit
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
READING_MODEL = "Qwen/Qwen3-30B-A3B"
B64 = (1, 2, 4, 8, 16, 32, 64)
SERVED = ("OFF_a", "ON_a", "ON_b", "OFF_b")
WARM, STEPS, PROFILED = 5, 256, 8
BUILDER = ("radixSortKVInPlace", "_cuda_scatter_gather_internal_kernel<true", "DeviceScan", "searchsorted",
           "arange_cuda_out")
TABLE = "_tile_table_r1"
PREMISE_SHARE = 0.05      # the chained builder's frozen kernels, as a share of the 64-row eager step (P119: 0.089)
NOISE = 0.015             # same-setting medians must agree within this
BAR = 0.98                # ON / OFF in both ABBA pairs at most this flips the default
# (name, statistic, lo, hi): written before the data; evaluated beside the verdict
PREDICTIONS = (("Q1", "builder_share_off", 0.06, 0.12),
               ("Q2", "table_on_over_builder_off", 0.5, 1.2),
               ("Q3", "eager_on_over_off", 0.90, 0.98),
               ("Q4", "served_ratios", 0.88, 0.97),
               ("Q5", "same_setting_disagreement", 0.0, 0.005))


def _fsum(xs) -> float:
    return math.fsum(float(x) for x in xs)


def median(xs) -> float:
    s = sorted(float(x) for x in xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else math.fsum(s[n // 2 - 1:n // 2 + 1]) / 2


def _sum_ms(table: dict, keys) -> float:
    return _fsum(ms for k, _c, ms in table.get("kernels") or () if any(x in k for x in keys))


def _sum_calls(table: dict, keys) -> float:
    return _fsum(c for k, c, _ms in table.get("kernels") or () if any(x in k for x in keys))


def faults(box: dict, e4b_sha: str) -> list:
    out = []
    if box.get("e4b_sha") != e4b_sha:
        out.append(f"e4b {box.get('e4b_sha')} != {e4b_sha}")
    if box.get("gnf4_sha") != GNF4_SHA:
        out.append(f"grouped-nf4-gemm {box.get('gnf4_sha')} != {GNF4_SHA}")
    if REVS.get(box.get("model")) != box.get("revision"):
        out.append(f"model {box.get('model')}@{box.get('revision')} is not a registered revision")
    cb = box.get("census_build") or {}
    if box.get("model") == READING_MODEL:
        if not (cb.get("int4_expert_layers") and cb.get("int4_expert_layers") == cb.get("moe_layers")):
            out.append(f"int4 experts on {cb.get('int4_expert_layers')} of {cb.get('moe_layers')} MoE layers")
        if not cb.get("int4_attn_projections"):
            out.append("int4 attention not engaged")
        for k in ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n"):
            if not cb.get(k):
                out.append(f"{k} is {cb.get(k)}: SC2e's stack folds it")
    rows = int(box.get("rows") or 0)
    if not 32 < rows <= 64:
        out.append(f"rows {rows}: the registered step is one bucket-64 piece (33 to 64 rows)")
        return out
    prof = box.get("profile") or {}
    for label, wide in (("off", False), ("on", True)):
        p = prof.get(label)
        if p is None:
            out.append(f"profile {label} missing")
            continue
        if p.get("wide") is not wide:
            out.append(f"profile {label} ran with wide={p.get('wide')}")
        st = (p.get("profiled") or {}).get("64") or {}
        want = (PROFILED, 0, rows * PROFILED, (64 - rows) * PROFILED)
        if (st.get("eager_steps"), st.get("replays"), st.get("rows"), st.get("pad_rows")) != want:
            out.append(f"profile {label}: bucket 64 profiled {st}, registered {want} (eager, replays, rows, pad)")
        other = [b for b, g in (p.get("profiled") or {}).items() if b != "64" and (g.get("eager_steps") or g.get("replays"))]
        if other:
            out.append(f"profile {label}: buckets {other} ran")
        if any(v != "eager: capture=False" for v in (p.get("graph_status") or {"?": None}).values()):
            out.append(f"profile {label}: graph status {p.get('graph_status')}")
        if not p.get("device_ms"):
            out.append(f"profile {label}: the profiler saw no device kernel")
    srv = box.get("served") or {}
    for label in SERVED + ("MUTANT",):
        s = srv.get(label)
        if s is None:
            out.append(f"served {label} missing")
            continue
        wide = label != "OFF_a" and label != "OFF_b"
        if s.get("wide") is not wide or s.get("mutant") is not (label == "MUTANT"):
            out.append(f"served {label} ran with wide={s.get('wide')} mutant={s.get('mutant')}")
        if s.get("bulk_kv") is not True:
            out.append(f"served {label}: KV bookkeeping not bulk (the server's default)")
        gs = s.get("graph_status") or {}
        if sorted(gs, key=int) != [str(b) for b in B64] or any(v != "graph" for v in gs.values()):
            out.append(f"served {label}: graph status {gs}")
        n = WARM + STEPS
        st = (s.get("graph_stats") or {}).get("64") or {}
        want = (n, 0, rows * n, (64 - rows) * n)
        if (st.get("replays"), st.get("eager_steps"), st.get("rows"), st.get("pad_rows")) != want:
            out.append(f"served {label}: bucket 64 ran {st}, registered {want} (replays, eager, rows, pad)")
        other = [b for b, g in (s.get("graph_stats") or {}).items() if b != "64" and (g.get("eager_steps") or g.get("replays"))]
        if other:
            out.append(f"served {label}: buckets {other} ran")
        if len(s.get("step_ms") or ()) != STEPS or not all(float(x) > 0 for x in s.get("step_ms") or ()):
            out.append(f"served {label}: {len(s.get('step_ms') or ())} timed steps, registered {STEPS}")
        toks = s.get("tokens") or []
        if len(toks) != n or any(len(t) != rows for t in toks):
            out.append(f"served {label}: tokens for {len(toks)} steps, registered {n} x {rows} rows")
    return out


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        for j, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return [i, j]
    return None


def _differing(a, b) -> int:
    return sum(1 for x, y in zip(a, b) for u, v in zip(x, y) if u != v)


def tabulate(box: dict) -> dict:
    off, on = box["profile"]["off"], box["profile"]["on"]
    srv = box["served"]
    t = {"profile": {}, "served": {}}
    for label, p in (("off", off), ("on", on)):
        t["profile"][label] = {"device_ms": round(float(p["device_ms"]), 4),
                               "builder_ms": round(_sum_ms(p, BUILDER), 4),
                               "table_ms": round(_sum_ms(p, (TABLE,)), 4),
                               "table_calls": round(_sum_calls(p, (TABLE,)), 3),
                               "radix_sort_calls": round(_sum_calls(p, ("radixSortKVInPlace",)), 3)}
    po, pn = t["profile"]["off"], t["profile"]["on"]
    t["builder_share_off"] = round(po["builder_ms"] / po["device_ms"], 4)
    t["table_on_over_builder_off"] = round(pn["table_ms"] / po["builder_ms"], 4) if po["builder_ms"] else None
    t["eager_on_over_off"] = round(pn["device_ms"] / po["device_ms"], 4)
    t["eager_off_minus_on_ms"] = round(po["device_ms"] - pn["device_ms"], 4)
    for label in SERVED + ("MUTANT",):
        xs = srv[label]["step_ms"]
        t["served"][label] = {"median_ms": round(median(xs), 4), "min_ms": round(min(xs), 4), "max_ms": round(max(xs), 4)}
    m = {k: median(srv[k]["step_ms"]) for k in SERVED}
    t["ratio_a"] = round(m["ON_a"] / m["OFF_a"], 4)
    t["ratio_b"] = round(m["ON_b"] / m["OFF_b"], 4)
    t["served_ratios"] = [t["ratio_a"], t["ratio_b"]]
    t["off_disagreement"] = round(abs(m["OFF_b"] / m["OFF_a"] - 1), 4)
    t["on_disagreement"] = round(abs(m["ON_b"] / m["ON_a"] - 1), 4)
    t["same_setting_disagreement"] = max(t["off_disagreement"], t["on_disagreement"])
    t["off_minus_on_ms"] = round(math.fsum([m["OFF_a"], m["OFF_b"], -m["ON_a"], -m["ON_b"]]) / 2, 4)
    tok = {k: srv[k]["tokens"] for k in SERVED + ("MUTANT",)}
    t["tokens"] = {"on_vs_off_first_diff": _first_diff(tok["ON_a"], tok["OFF_a"]),
                   "on_vs_off_differing": _differing(tok["ON_a"], tok["OFF_a"]),
                   "mutant_vs_off_differing": _differing(tok["MUTANT"], tok["OFF_a"]),
                   "mutant_vs_off_first_diff": _first_diff(tok["MUTANT"], tok["OFF_a"]),
                   "compared": sum(len(x) for x in tok["OFF_a"])}
    return t


def predictions(t: dict) -> dict:
    out = {}
    for name, stat, lo, hi in PREDICTIONS:
        v = t.get(stat)
        vs = v if isinstance(v, list) else [v]
        held = None if any(x is None for x in vs) else all(lo <= x <= hi for x in vs)
        out[name] = {"statistic": stat, "value": v, "band": [lo, hi],
                     "result": "UNREAD" if held is None else ("HELD" if held else "MISSED")}
    return out


def reduce_obj(box: dict, e4b_sha: str) -> dict:
    void = faults(box, e4b_sha)
    if void:
        return {"lane": "P120", "verdict": "VOID", "reasons": void}
    t = tabulate(box)
    po, pn = t["profile"]["off"], t["profile"]["on"]
    moe = int((box.get("census_build") or {}).get("moe_layers") or 0)
    if pn["radix_sort_calls"] or not po["radix_sort_calls"] or pn["table_calls"] != po["radix_sort_calls"]:
        void.append(f"not engaged: 0 shows {po['radix_sort_calls']} chained builds per step, 1 shows "
                    f"{pn['table_calls']} one-launch tables and {pn['radix_sort_calls']} radix sorts")
    srv = box["served"]
    if srv["OFF_a"]["tokens"] != srv["OFF_b"]["tokens"] or srv["ON_a"]["tokens"] != srv["ON_b"]["tokens"]:
        void.append("nondeterministic: a setting's two arms emitted different tokens")
    if srv["MUTANT"]["tokens"] == srv["OFF_a"]["tokens"]:
        void.append("the mutant survived: its tokens equal OFF's, so the token gate could not fail")
    base = {"lane": "P120", "model": box["model"], "tables": t}
    if void:
        return {**base, "verdict": "VOID", "reasons": void}
    base["predictions"] = predictions(t)
    reasons = []
    if box["model"] == READING_MODEL and (po["radix_sort_calls"] < moe or t["builder_share_off"] < PREMISE_SHARE):
        reasons.append(f"premise: {po['radix_sort_calls']} chained builds per step for {moe} MoE layers, builder "
                       f"share {t['builder_share_off']} (needs >= {PREMISE_SHARE})")
        return {**base, "verdict": "PREMISE_ABSENT", "reasons": reasons}
    if srv["ON_a"]["tokens"] != srv["OFF_a"]["tokens"]:
        return {**base, "verdict": "TOKENS_DIFFER",
                "reasons": [f"ON differs from OFF at [step, row] {t['tokens']['on_vs_off_first_diff']}"]}
    if t["same_setting_disagreement"] > NOISE:
        return {**base, "verdict": "NOISY", "reasons": [f"OFF pair {t['off_disagreement']}, ON pair "
                                                       f"{t['on_disagreement']} (bound {NOISE})"]}
    ra, rb = t["ratio_a"], t["ratio_b"]
    if ra <= BAR and rb <= BAR:
        return {**base, "verdict": "DEFAULT_ON", "reasons": [f"ON/OFF {ra}, {rb} <= {BAR}"]}
    if ra > 1 and rb > 1:
        return {**base, "verdict": "SLOWER", "reasons": [f"ON/OFF {ra}, {rb} > 1"]}
    return {**base, "verdict": "NO_GAIN", "reasons": [f"ON/OFF {ra}, {rb}: not both <= {BAR}"]}


def reduce(run: Path, e4b_sha: str) -> dict:
    p = run / "box.json"
    if not p.is_file():
        return {"lane": "P120", "verdict": "NO_READING", "reasons": ["no box record"]}
    return reduce_obj(json.loads(p.read_text()), e4b_sha)


# ------------------------------------------------------------------------------------------------ self-test --

def _kernels(off: bool, device_ms: float):
    if off:   # P119's d64 builder kernels, one chained build per MoE layer
        ks = [["void at::native::radixSortKVInPlace<-2, -1, 32, 32, long, long, unsigned int>", 48.0, 0.95],
              ["void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_"
               "kernel<true, long, long>::op", 48.0, 0.11],
              ["void at_cuda_detail::cub::DeviceScanKernel<x>", 96.0, 0.12],
              ["void at_cuda_detail::cub::DeviceScanInitKernel<x>", 96.0, 0.07],
              ["void at::native::(anonymous namespace)::searchsorted_cuda_kernel<long, long>", 48.0, 0.08],
              ["void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(x)", 96.0, 0.07]]
    else:
        ks = [["_tile_table_r1", 48.0, 0.9]]
    rest = device_ms - _fsum(k[2] for k in ks)
    return ks + [["_gemm_int4_b32_grouped_kernel", 96.0, round(rest, 4)]]


def _box(model=READING_MODEL, e4b="a" * 40, rows=64, ms=(18.0, 16.9, 16.95, 18.05), tokens=None):
    g = (lambda i, r: (i * 131 + r * 7) % 1000) if tokens is None else tokens
    n = WARM + STEPS
    toks = [[g(i, r) for r in range(rows)] for i in range(n)]

    def served(label, med):
        steps = [round(med + 0.01 * ((i % 5) - 2), 4) for i in range(STEPS)]
        return {"wide": label not in ("OFF_a", "OFF_b"), "mutant": label == "MUTANT", "rows": rows, "steps": STEPS,
                "warm": WARM, "bulk_kv": True, "graph_status": {str(b): "graph" for b in B64},
                "graph_stats": {str(b): ({"replays": n, "eager_steps": 0, "rows": rows * n, "pad_rows": (64 - rows) * n}
                                         if b == 64 else {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0})
                                for b in B64},
                "step_ms": steps, "median_ms": med, "tokens": copy.deepcopy(toks)}

    def prof(off, dms):
        st = {str(b): ({"replays": 0, "eager_steps": PROFILED, "rows": rows * PROFILED, "pad_rows": (64 - rows) * PROFILED}
                       if b == 64 else {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}) for b in B64}
        return {"wide": not off, "device_ms": dms, "profiled": st, "graph_status": {str(b): "eager: capture=False" for b in B64},
                "layers": 48, "kernels": _kernels(off, dms), "classes": {}}

    box = {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "rows": rows,
           "census_build": {"moe_layers": 48, "int4_expert_layers": 48, "int4_attn_projections": 96, "fuse_qkv_n": 48,
                            "fuse_t1_glue_n": 193, "fuse_router_epilogue_n": 48},
           "profile": {"off": prof(True, 15.6), "on": prof(False, 14.8)},
           "served": {k: served(k, m) for k, m in zip(SERVED, ms)}}
    mut = served("MUTANT", 17.0)
    mut["tokens"] = [[(x + 1) % 1000 for x in row] for row in mut["tokens"]]
    box["served"]["MUTANT"] = mut
    return box


def self_test() -> int:
    cases = []

    def run(box, e4b="a" * 40):
        return reduce_obj(box, e4b)

    ok = run(_box())
    t = ok.get("tables") or {}
    cases.append(("a registered box flips the default", ok["verdict"] == "DEFAULT_ON"))
    cases.append(("ratios from medians", t.get("served_ratios") == [0.9389, 0.9391]))
    cases.append(("builder share", t.get("builder_share_off") == 0.0897))
    cases.append(("table over builder", t.get("table_on_over_builder_off") == 0.6429))
    cases.append(("predictions all held", all(v["result"] == "HELD" for v in ok.get("predictions", {}).values())))
    cases.append(("median of an even count", median([1, 4, 2, 3]) == 2.5 and median([3, 1, 2]) == 2))
    cases.append(("wrong e4b", run(_box(), e4b="b" * 40)["verdict"] == "VOID"))
    b = _box()
    b["gnf4_sha"] = "f" * 40
    cases.append(("wrong grouped-nf4-gemm", run(b)["verdict"] == "VOID"))
    b = _box()
    b["census_build"]["fuse_qkv_n"] = 0
    cases.append(("a fold missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["profile"]["on"]["profiled"]["64"]["eager_steps"] = 7
    cases.append(("a profiled step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["profile"]["on"]["wide"] = False
    cases.append(("a profile arm under the wrong setting", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["ON_b"]["graph_status"]["64"] = "eager: RuntimeError: capture"
    cases.append(("bucket 64 not captured", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["OFF_a"]["graph_stats"]["64"]["eager_steps"] = 1
    cases.append(("an eager step in a served arm", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["ON_a"]["graph_stats"]["32"]["replays"] = 3
    cases.append(("another bucket replayed", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["OFF_b"]["step_ms"] = b["served"]["OFF_b"]["step_ms"][:-1]
    cases.append(("a timed step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["ON_a"]["bulk_kv"] = False
    cases.append(("per-layer KV bookkeeping", run(b)["verdict"] == "VOID"))
    b = _box()
    del b["served"]["MUTANT"]
    cases.append(("the mutant arm missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["profile"]["on"]["kernels"] = _kernels(True, 15.6)
    r = run(b)
    cases.append(("not engaged: ON still sorts", r["verdict"] == "VOID" and "not engaged" in r["reasons"][0]))
    b = _box()
    b["profile"]["on"]["kernels"][0][1] = 24.0
    cases.append(("not engaged: half the tables", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["OFF_b"]["tokens"][100][3] += 1
    r = run(b)
    cases.append(("nondeterministic", r["verdict"] == "VOID" and "nondeterministic" in " ".join(r["reasons"])))
    b = _box()
    b["served"]["MUTANT"]["tokens"] = copy.deepcopy(b["served"]["OFF_a"]["tokens"])
    r = run(b)
    cases.append(("the mutant survived", r["verdict"] == "VOID" and "mutant survived" in " ".join(r["reasons"])))
    b = _box()
    for k in ("ON_a", "ON_b"):
        b["served"][k]["tokens"][200][9] += 1
    r = run(b)
    cases.append(("tokens differ", r["verdict"] == "TOKENS_DIFFER" and r["tables"]["tokens"]["on_vs_off_first_diff"] == [200, 9]))
    b = _box()
    b["profile"]["off"]["kernels"] = [[k, c, 0.1 if any(x in k for x in BUILDER) else ms] for k, c, ms in _kernels(True, 15.6)]
    r = run(b)
    cases.append(("premise absent", r["verdict"] == "PREMISE_ABSENT" and r["tables"]["builder_share_off"] < PREMISE_SHARE))
    b = _box(model="ibm-granite/granite-3.1-3b-a800m-instruct", rows=40)
    b["census_build"] = {"moe_layers": 32}
    b["profile"]["off"]["kernels"][0][2] = 0.01
    cases.append(("the proof's Granite reads without the premise share", run(b)["verdict"] == "DEFAULT_ON"))
    cases.append(("noisy", run(_box(ms=(18.0, 16.9, 16.95, 18.4)))["verdict"] == "NOISY"))
    r = run(_box(ms=(18.0, 17.8, 17.81, 18.02)))
    cases.append(("no gain", r["verdict"] == "NO_GAIN" and r["predictions"]["Q4"]["result"] == "MISSED"))
    cases.append(("one pair below the bar is no gain", run(_box(ms=(18.0, 17.55, 17.75, 18.05)))["verdict"] == "NO_GAIN"))
    cases.append(("slower", run(_box(ms=(18.0, 18.1, 18.12, 18.02)))["verdict"] == "SLOWER"))
    cases.append(("rows beyond the bucket", run({**_box(), "rows": 65})["verdict"] == "VOID"))
    cases.append(("rows that fit a smaller bucket", run(_box(rows=32))["verdict"] == "VOID"))
    bad = [n for n, okk in cases if not okk]
    if bad:
        print("p120_reduce self-test FAILED:", bad)
        return 1
    print(f"p120_reduce self-test OK ({len(cases)} cases)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    v = reduce(Path(a.dir), a.e4b_sha)
    s = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(s + "\n")
    print(f"P120_VERDICT {v['verdict']} {json.dumps(v.get('reasons') or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
