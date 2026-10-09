#!/usr/bin/env python3
"""Lane P126's reducer (bench/p126/PREREG-p126.md): checks the box ran as registered, then applies the registered rule
to ``E4B_INT4_TILE_PROGRAMS`` at 64 decode rows. Every bound is a ratio inside the box; the token gate is P120's.

In order, the first that fires is the verdict:
- **NO_READING**: no box record.
- **VOID**: the e4b or grouped-nf4-gemm commit or the model revision is not the registered one; the reading's stack is
  not SC2e's, or a tile knob was set outside the box; rows outside 33-64; a profile arm did not profile 8 padded eager
  steps of bucket 64 under its setting; a served block is missing or out of its registered order, or one of its runners
  was not captured under its setting or did not replay bucket 64 exactly ``warm + steps + busy`` times with no eager step
  and no other bucket; a step, token or traced-step count is short; **not engaged** (P = 1 profiles one
  ``_tile_table_r1`` launch per MoE layer and no split table; P > 1 one ``_tile_table_cumsum_mp`` launch per MoE layer
  and no one-program table; either shows a radix sort); **nondeterministic** (a setting emits different tokens in two
  blocks); **the mutant survived** (its tokens equal P = 1's).
- **PREMISE_ABSENT** (the reading only): P = 1's one-program table is under ``PREMISE_SHARE`` of the eager step.
- **TOKENS_DIFFER**: a candidate emits a token P = 1 does not, in any block.
- **NOISY**: for a candidate, block a's ratio and block b's differ by more than ``NOISE``.
- **DEFAULT_ON_<P>**: of the candidates whose two block ratios are both at most ``BAR``, the one with the lower mean
  ratio (ties to the smaller P).
- **SLOWER**: every ratio above 1. **NO_GAIN**: otherwise.

Reported, never gated: the predictions (Q1-Q8), the median per-pair ratio, every runner's busy fraction, each block's
peak memory and GPU log, the profile tables. Floats are summed with ``math.fsum`` and medians taken by sorting, so the
output is byte-identical on any Python.

    p126_reduce.py --dir RUN --out verdict.json [--e4b-sha SHA]
    p126_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

GNF4_SHA = "e21a71242b361dd0a3f72cdf4152de94633891a8"     # grouped-nf4-gemm main after #524 and #525 (v0.44.0 + #524)
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
READING_MODEL = "Qwen/Qwen3-30B-A3B"
B64 = (1, 2, 4, 8, 16, 32, 64)
SETTINGS = ("1", "4", "8")
CANDIDATES = ("4", "8")
BLOCKS = ("a", "b")
WARM, STEPS, BUSY, PROFILED, MUTANT_STEPS = 5, 256, 32, 8, 16
TABLE_ONE, TABLE_MP = "_tile_table_r1", "_tile_table_cumsum_mp"
PREMISE_SHARE = 0.05      # the one-program table, as a share of the 64-row eager step (P122: 2.51 / 15.42 = 0.163)
NOISE = 0.015             # a candidate's two block ratios must agree within this
BAR = 0.98                # both block ratios at most this licenses the candidate
# (name, statistic, lo, hi): written before the data; evaluated beside the verdict
PREDICTIONS = (("Q1", "table_share_one", 0.10, 0.22),
               ("Q2", "mp_over_one_8", 0.10, 0.40),
               ("Q3", "mp_over_one_4", 0.20, 0.55),
               ("Q4", "served_ratios_8", 0.86, 0.95),
               ("Q5", "served_ratios_4", 0.88, 0.96),
               ("Q6", "busy_min", 0.90, 1.0),
               ("Q7", "block_disagreement_max", 0.0, 0.005),
               ("Q8", "mutant_differ_share", 0.5, 1.0))


def _fsum(xs) -> float:
    return math.fsum(float(x) for x in xs)


def median(xs) -> float:
    s = sorted(float(x) for x in xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else math.fsum(s[n // 2 - 1:n // 2 + 1]) / 2


def _kcalls(table, name) -> float:
    return _fsum(c for k, c, _ms in table.get("kernels") or () if name in k)


def _kms(table, name) -> float:
    return _fsum(ms for k, _c, ms in table.get("kernels") or () if name in k)


def faults(box: dict, e4b_sha: str) -> list:
    out = []
    if box.get("e4b_sha") != e4b_sha:
        out.append(f"e4b {box.get('e4b_sha')} != {e4b_sha}")
    if box.get("gnf4_sha") != GNF4_SHA:
        out.append(f"grouped-nf4-gemm {box.get('gnf4_sha')} != {GNF4_SHA}")
    if REVS.get(box.get("model")) != box.get("revision"):
        out.append(f"model {box.get('model')}@{box.get('revision')} is not a registered revision")
    if box.get("tile_programs_env") is not None or box.get("wide_tiles_env") is not None:
        out.append(f"a tile knob was set outside the box: E4B_INT4_TILE_PROGRAMS={box.get('tile_programs_env')!r}, "
                   f"E4B_INT4_WIDE_TILES={box.get('wide_tiles_env')!r}")
    cb = box.get("census_build") or {}
    if box.get("model") == READING_MODEL:
        if not (cb.get("int4_expert_layers") and cb.get("int4_expert_layers") == cb.get("moe_layers")):
            out.append(f"int4 experts on {cb.get('int4_expert_layers')} of {cb.get('moe_layers')} MoE layers")
        for k in ("int4_attn_projections", "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n"):
            if not cb.get(k):
                out.append(f"{k} is {cb.get(k)}: SC2e's stack has it")
    rows = int(box.get("rows") or 0)
    if not 32 < rows <= 64:
        out.append(f"rows {rows}: the registered step is one bucket-64 piece (33 to 64 rows)")
        return out
    prof = box.get("profile") or {}
    for p in SETTINGS:
        a = prof.get(p)
        if a is None:
            out.append(f"profile P={p} missing")
            continue
        if a.get("programs") != p:
            out.append(f"profile P={p} ran programs={a.get('programs')}")
        st = (a.get("profiled") or {}).get("64") or {}
        want = (PROFILED, 0, rows * PROFILED, (64 - rows) * PROFILED)
        if (st.get("eager_steps"), st.get("replays"), st.get("rows"), st.get("pad_rows")) != want:
            out.append(f"profile P={p}: bucket 64 profiled {st}, registered {want} (eager, replays, rows, pad)")
        other = [b for b, g in (a.get("profiled") or {}).items() if b != "64" and (g.get("eager_steps") or g.get("replays"))]
        if other:
            out.append(f"profile P={p}: buckets {other} ran")
        if any(v != "eager: capture=False" for v in (a.get("graph_status") or {"?": None}).values()):
            out.append(f"profile P={p}: graph status {a.get('graph_status')}")
        if not a.get("device_ms"):
            out.append(f"profile P={p}: the profiler saw no device kernel")
    srv = box.get("served") or {}
    total = WARM + STEPS + BUSY
    for c in CANDIDATES:
        for bk in BLOCKS:
            blk = (srv.get(c) or {}).get(bk)
            if blk is None:
                out.append(f"served P={c} block {bk} missing")
                continue
            order = ["1", c] if bk == "a" else [c, "1"]
            if blk.get("order") != order or blk.get("rows") != rows or blk.get("candidate") != c:
                out.append(f"served P={c} block {bk}: order {blk.get('order')} rows {blk.get('rows')}, registered {order}")
            if blk.get("bulk_kv") is not True:
                out.append(f"served P={c} block {bk}: KV bookkeeping not bulk (the server's default)")
            if not (blk.get("memory") or {}).get("max_allocated_mib"):
                out.append(f"served P={c} block {bk}: no peak memory recorded")
            for st in order:
                s = (blk.get("arms") or {}).get(st)
                label = f"served P={c} {bk} runner P={st}"
                if s is None:
                    out.append(f"{label} missing")
                    continue
                if s.get("programs") != st:
                    out.append(f"{label} ran programs={s.get('programs')}")
                gs = s.get("graph_status") or {}
                if sorted(gs, key=int) != [str(x) for x in B64] or any(v != "graph" for v in gs.values()):
                    out.append(f"{label}: graph status {gs}")
                g = (s.get("graph_stats") or {}).get("64") or {}
                want = (total, 0, rows * total, (64 - rows) * total)
                if (g.get("replays"), g.get("eager_steps"), g.get("rows"), g.get("pad_rows")) != want:
                    out.append(f"{label}: bucket 64 ran {g}, registered {want} (replays, eager, rows, pad)")
                other = [k for k, x in (s.get("graph_stats") or {}).items()
                         if k != "64" and (x.get("eager_steps") or x.get("replays"))]
                if other:
                    out.append(f"{label}: buckets {other} ran")
                if len(s.get("step_ms") or ()) != STEPS or not all(float(x) > 0 for x in s.get("step_ms") or ()):
                    out.append(f"{label}: {len(s.get('step_ms') or ())} timed steps, registered {STEPS}")
                toks = s.get("tokens") or []
                if len(toks) != total or any(len(x) != rows for x in toks):
                    out.append(f"{label}: tokens for {len(toks)} steps, registered {total} x {rows} rows")
                busy = s.get("busy") or []
                if len(busy) != BUSY or not all(x[0] and x[0] > 0 and x[1] and x[1] > 0 for x in busy):
                    out.append(f"{label}: {len(busy)} traced steps, registered {BUSY}")
    m = box.get("mutant") or {}
    if m.get("programs") != "8" or m.get("mutant") is not True:
        out.append(f"mutant arm ran programs={m.get('programs')} mutant={m.get('mutant')}")
    if len(m.get("tokens") or ()) != WARM + MUTANT_STEPS:
        out.append(f"mutant: tokens for {len(m.get('tokens') or ())} steps, registered {WARM + MUTANT_STEPS}")
    return out


def engagement(box: dict) -> list:
    out, moe = [], int((box.get("census_build") or {}).get("moe_layers") or 0)
    for p in SETTINGS:
        a = box["profile"][p]
        one, mp, rad = _kcalls(a, TABLE_ONE), _kcalls(a, TABLE_MP), _kcalls(a, "radixSortKVInPlace")
        want = (moe, 0.0) if p == "1" else (0.0, moe)
        if (one, mp) != want or rad:
            out.append(f"profile P={p}: one-program tables {one}, split tables {mp}, radix sorts {rad} per step; "
                       f"registered {want[0]}, {want[1]}, 0")
    return out


def _differing(a, b) -> int:
    return sum(1 for x, y in zip(a, b) for u, v in zip(x, y) if u != v)


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        for j, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return [i, j]
    return None


def clock_summary(rows) -> dict:
    def col(i):
        out = []
        for r in rows:
            try:
                out.append(float(r[i]))
            except (IndexError, TypeError, ValueError):
                pass
        return [min(out), max(out)] if out else None
    return {"samples": len(rows), "sm_mhz": col(1), "power_w": col(3), "temp_c": col(4),
            "pstates": sorted({str(r[5]) for r in rows if len(r) > 5})}


def tabulate(box: dict) -> dict:
    t = {"profile": {}, "served": {}}
    for p in SETTINGS:
        a = box["profile"][p]
        t["profile"][p] = {"device_ms": round(float(a["device_ms"]), 4), "table_one_ms": round(_kms(a, TABLE_ONE), 4),
                           "table_mp_ms": round(_kms(a, TABLE_MP), 4),
                           "table_one_calls": round(_kcalls(a, TABLE_ONE), 3),
                           "table_mp_calls": round(_kcalls(a, TABLE_MP), 3)}
    one = t["profile"]["1"]
    t["table_share_one"] = round(one["table_one_ms"] / one["device_ms"], 4)
    for c in CANDIDATES:
        pc = t["profile"][c]
        t[f"mp_over_one_{c}"] = round(pc["table_mp_ms"] / one["table_one_ms"], 4) if one["table_one_ms"] else None
        t[f"eager_ratio_{c}"] = round(pc["device_ms"] / one["device_ms"], 4)
        srv = box["served"][c]
        s = {"medians": {}, "busy": {}, "pair_ratio_median": {}, "memory": {}, "clock": {}}
        for bk in BLOCKS:
            arms = srv[bk]["arms"]
            m1, mc = median(arms["1"]["step_ms"]), median(arms[c]["step_ms"])
            s["medians"][bk] = {"1": round(m1, 4), c: round(mc, 4)}
            s[f"ratio_{bk}"] = round(mc / m1, 4)
            s["pair_ratio_median"][bk] = round(median([y / x for x, y in zip(arms["1"]["step_ms"], arms[c]["step_ms"])]), 4)
            s["busy"][bk] = {st: round(median([x[0] / x[1] for x in arms[st]["busy"]]), 4) for st in ("1", c)}
            s["memory"][bk] = srv[bk]["memory"]
            s["clock"][bk] = clock_summary(srv[bk].get("clock") or [])
        s["ratios"] = [s["ratio_a"], s["ratio_b"]]
        s["block_disagreement"] = round(abs(s["ratio_a"] / s["ratio_b"] - 1), 4)
        s["saving_ms"] = round(math.fsum([s["medians"]["a"]["1"], s["medians"]["b"]["1"],
                                          -s["medians"]["a"][c], -s["medians"]["b"][c]]) / 2, 4)
        t["served"][c] = s
        t[f"served_ratios_{c}"] = s["ratios"]
    t["busy_min"] = min(v for c in CANDIDATES for bk in BLOCKS for v in t["served"][c]["busy"][bk].values())
    t["block_disagreement_max"] = max(t["served"][c]["block_disagreement"] for c in CANDIDATES)
    ref = box["served"][CANDIDATES[0]]["a"]["arms"]["1"]["tokens"][:WARM + MUTANT_STEPS]
    mt = box["mutant"]["tokens"]
    t["mutant_differ_share"] = round(_differing(mt, ref) / max(1, sum(len(x) for x in mt)), 4)
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
        return {"lane": "P126", "verdict": "VOID", "reasons": void}
    void = engagement(box)
    srv = box["served"]
    ref = srv[CANDIDATES[0]]["a"]["arms"]["1"]["tokens"]
    for c in CANDIDATES:
        for bk in BLOCKS:
            if srv[c][bk]["arms"]["1"]["tokens"] != ref:
                void.append(f"nondeterministic: P=1 in P={c} block {bk} emitted other tokens than in P=4 block a")
        if srv[c]["a"]["arms"][c]["tokens"] != srv[c]["b"]["arms"][c]["tokens"]:
            void.append(f"nondeterministic: P={c} emitted different tokens in blocks a and b")
    if box["mutant"]["tokens"] == ref[:WARM + MUTANT_STEPS]:
        void.append("the mutant survived: its tokens equal P = 1's, so the token gate could not fail")
    t = tabulate(box)
    base = {"lane": "P126", "model": box["model"], "tables": t}
    if void:
        return {**base, "verdict": "VOID", "reasons": void}
    base["predictions"] = predictions(t)
    if box["model"] == READING_MODEL and t["table_share_one"] < PREMISE_SHARE:
        return {**base, "verdict": "PREMISE_ABSENT",
                "reasons": [f"premise: the one-program table is {t['table_share_one']} of the eager step "
                            f"(needs >= {PREMISE_SHARE})"]}
    diffs = [(c, bk) for c in CANDIDATES for bk in BLOCKS if srv[c][bk]["arms"][c]["tokens"] != srv[c][bk]["arms"]["1"]["tokens"]]
    if diffs:
        c, bk = diffs[0]
        return {**base, "verdict": "TOKENS_DIFFER",
                "reasons": [f"P={c} block {bk} differs from P=1 at [step, row] "
                            f"{_first_diff(srv[c][bk]['arms'][c]['tokens'], srv[c][bk]['arms']['1']['tokens'])}"]}
    noisy = [c for c in CANDIDATES if t["served"][c]["block_disagreement"] > NOISE]
    if noisy:
        return {**base, "verdict": "NOISY",
                "reasons": [f"P={c}: block a {t['served'][c]['ratio_a']}, block b {t['served'][c]['ratio_b']}, "
                            f"{t['served'][c]['block_disagreement']} apart (bound {NOISE})" for c in noisy]}
    lic = [c for c in CANDIDATES if all(r <= BAR for r in t["served"][c]["ratios"])]
    rs = {c: t["served"][c]["ratios"] for c in CANDIDATES}
    if lic:
        best = min(lic, key=lambda c: (math.fsum(rs[c]) / 2, int(c)))
        return {**base, "verdict": f"DEFAULT_ON_{best}", "reasons": [f"ON/OFF {rs}: P={best} has the lower mean of "
                                                                      f"those with both <= {BAR}"]}
    if all(r > 1 for c in CANDIDATES for r in rs[c]):
        return {**base, "verdict": "SLOWER", "reasons": [f"ON/OFF {rs} all > 1"]}
    return {**base, "verdict": "NO_GAIN", "reasons": [f"ON/OFF {rs}: no candidate has both <= {BAR}"]}


def reduce(run: Path, e4b_sha: str) -> dict:
    p = run / "box.json"
    if not p.is_file():
        return {"lane": "P126", "verdict": "NO_READING", "reasons": ["no box record"]}
    return reduce_obj(json.loads(p.read_text()), e4b_sha)


# ------------------------------------------------------------------------------------------------ self-test --
MOE = 48


def _kernels(p: str, device_ms: float, one_ms: float, mp_ms: float):
    ks = [[TABLE_ONE, float(MOE), one_ms]] if p == "1" else [[TABLE_MP, float(MOE), mp_ms]]
    rest = device_ms - _fsum(k[2] for k in ks)
    return ks + [["_gemm_int4_b32_grouped_smallm_kernel", 96.0, round(rest, 4)]]


def _box(model=READING_MODEL, e4b="a" * 40, rows=64, ms4=(16.4, 15.6, 15.62, 16.42), ms8=(16.4, 15.0, 15.03, 16.41),
         mp=(None, 0.9, 0.5)):
    total = WARM + STEPS + BUSY

    def toks(salt=0):
        return [[(i * 131 + r * 7 + salt) % 1000 for r in range(rows)] for i in range(total)]

    def arm(st, med):
        g = {str(x): ({"replays": total, "eager_steps": 0, "rows": rows * total, "pad_rows": (64 - rows) * total}
                      if x == 64 else {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}) for x in B64}
        return {"programs": st, "graph_status": {str(x): "graph" for x in B64}, "graph_stats": g,
                "step_ms": [round(med + 0.01 * ((i % 5) - 2), 4) for i in range(STEPS)], "median_ms": med,
                "busy": [[round(med * 0.96, 4), med] for _ in range(BUSY)], "tokens": toks()}

    def block(c, bk, m1, mc):
        order = ["1", c] if bk == "a" else [c, "1"]
        return {"candidate": c, "block": bk, "order": order, "rows": rows, "steps": STEPS, "warm": WARM, "busy_steps": BUSY,
                "bulk_kv": True, "arms": {"1": arm("1", m1), c: arm(c, mc)},
                "memory": {"max_allocated_mib": 25400.0, "max_reserved_mib": 26100.0, "wide_workspace_mib": 11.5},
                "clock": [[0.0, "2800", "13801", "500.0", "65", "P1"]]}

    def prof(p, dms):
        st = {str(x): ({"replays": 0, "eager_steps": PROFILED, "rows": rows * PROFILED, "pad_rows": (64 - rows) * PROFILED}
                       if x == 64 else {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}) for x in B64}
        mp_ms = {"4": mp[1], "8": mp[2]}.get(p)
        return {"programs": p, "device_ms": dms, "profiled": st, "graph_status": {str(x): "eager: capture=False" for x in B64},
                "layers": MOE, "kernels": _kernels(p, dms, 2.5, mp_ms), "classes": {}}

    box = {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "rows": rows,
           "tile_programs_env": None, "wide_tiles_env": None,
           "census_build": {"moe_layers": MOE, "int4_expert_layers": 48, "int4_attn_projections": 96, "fuse_qkv_n": 48,
                            "fuse_t1_glue_n": 193, "fuse_router_epilogue_n": 48},
           "profile": {"1": prof("1", 15.4), "4": prof("4", 13.8), "8": prof("8", 13.4)},
           "served": {"4": {"a": block("4", "a", ms4[0], ms4[1]), "b": block("4", "b", ms4[3], ms4[2])},
                      "8": {"a": block("8", "a", ms8[0], ms8[1]), "b": block("8", "b", ms8[3], ms8[2])}}}
    box["mutant"] = {"programs": "8", "mutant": True, "rows": rows, "steps": MUTANT_STEPS, "warm": WARM,
                     "graph_status": {}, "graph_stats": {},
                     "tokens": [[(x + 1) % 1000 for x in row] for row in toks()[:WARM + MUTANT_STEPS]]}
    return box


def self_test() -> int:
    cases = []

    def run(box, e4b="a" * 40):
        return reduce_obj(box, e4b)

    ok = run(_box())
    t = ok.get("tables") or {}
    cases.append(("a registered box licenses the better candidate", ok["verdict"] == "DEFAULT_ON_8"))
    cases.append(("ratios from medians", t.get("served_ratios_8") == [0.9146, 0.9159] and
                  t.get("served_ratios_4") == [0.9512, 0.9513]))
    cases.append(("table share", t.get("table_share_one") == 0.1623))
    cases.append(("split over one-program table", t.get("mp_over_one_8") == 0.2 and t.get("mp_over_one_4") == 0.36))
    cases.append(("pair ratio median", t["served"]["8"]["pair_ratio_median"]["a"] == 0.9146))
    cases.append(("mutant share", t.get("mutant_differ_share") == 1.0))
    cases.append(("predictions all held", all(v["result"] == "HELD" for v in ok.get("predictions", {}).values())))
    cases.append(("median of an even count", median([1, 4, 2, 3]) == 2.5 and median([3, 1, 2]) == 2))
    cases.append(("wrong e4b", run(_box(), e4b="b" * 40)["verdict"] == "VOID"))
    b = _box()
    b["gnf4_sha"] = "f" * 40
    cases.append(("wrong grouped-nf4-gemm", run(b)["verdict"] == "VOID"))
    b = _box()
    b["tile_programs_env"] = "8"
    cases.append(("a tile knob set outside the box", run(b)["verdict"] == "VOID"))
    b = _box()
    b["census_build"]["fuse_qkv_n"] = 0
    cases.append(("a fold missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["profile"]["4"]["profiled"]["64"]["eager_steps"] = 7
    cases.append(("a profiled step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["8"]["b"]["order"] = ["1", "8"]
    cases.append(("a block out of order", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["4"]["a"]["arms"]["4"]["graph_status"]["64"] = "eager: RuntimeError"
    cases.append(("bucket 64 not captured", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["8"]["a"]["arms"]["1"]["graph_stats"]["64"]["eager_steps"] = 1
    cases.append(("an eager step", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["8"]["b"]["arms"]["8"]["busy"] = b["served"]["8"]["b"]["arms"]["8"]["busy"][:-1]
    cases.append(("a traced step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["4"]["b"]["memory"] = {}
    cases.append(("no peak memory", run(b)["verdict"] == "VOID"))
    b = _box()
    b["mutant"]["tokens"] = b["mutant"]["tokens"][:-1]
    cases.append(("mutant short", run(b)["verdict"] == "VOID"))
    b = _box()
    b["profile"]["8"]["kernels"] = _kernels("1", 15.4, 2.5, None)
    r = run(b)
    cases.append(("not engaged: P=8 profiled the one-program table", r["verdict"] == "VOID"
                  and "profile P=8" in " ".join(r["reasons"])))
    b = _box()
    b["profile"]["1"]["kernels"].append(["void at::native::radixSortKVInPlace<x>", 48.0, 0.9])
    cases.append(("not engaged: a radix sort", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["8"]["b"]["arms"]["1"]["tokens"][100][3] += 1
    r = run(b)
    cases.append(("nondeterministic P=1", r["verdict"] == "VOID" and "nondeterministic" in " ".join(r["reasons"])))
    b = _box()
    b["served"]["4"]["b"]["arms"]["4"]["tokens"][7][1] += 1
    r = run(b)
    cases.append(("nondeterministic candidate", r["verdict"] == "VOID" and "nondeterministic" in " ".join(r["reasons"])))
    b = _box()
    b["mutant"]["tokens"] = copy.deepcopy(b["served"]["4"]["a"]["arms"]["1"]["tokens"][:WARM + MUTANT_STEPS])
    r = run(b)
    cases.append(("the mutant survived", r["verdict"] == "VOID" and "mutant survived" in " ".join(r["reasons"])))
    b = _box()
    for bk in BLOCKS:
        b["served"]["8"][bk]["arms"]["8"]["tokens"][200][9] += 1
    r = run(b)
    cases.append(("tokens differ", r["verdict"] == "TOKENS_DIFFER" and "[200, 9]" in r["reasons"][0]))
    b = _box()
    b["profile"]["1"]["kernels"][0][2] = 0.3
    r = run(b)
    cases.append(("premise absent", r["verdict"] == "PREMISE_ABSENT"))
    b = _box(model="ibm-granite/granite-3.1-3b-a800m-instruct", rows=40)
    b["census_build"] = {"moe_layers": MOE}
    b["profile"]["1"]["kernels"][0][2] = 0.3
    cases.append(("the proof's Granite reads without the premise share", run(b)["verdict"] == "DEFAULT_ON_8"))
    cases.append(("noisy", run(_box(ms8=(16.4, 15.0, 15.4, 16.41)))["verdict"] == "NOISY"))
    r = run(_box(ms8=(16.4, 16.2, 16.22, 16.41)))
    cases.append(("only P=4 licensed", r["verdict"] == "DEFAULT_ON_4"))
    r = run(_box(ms4=(16.4, 15.0, 15.03, 16.42), ms8=(16.4, 15.0, 15.02, 16.41)))
    cases.append(("a tie goes to the smaller P", r["verdict"] == "DEFAULT_ON_4"))
    r = run(_box(ms4=(16.4, 16.3, 16.31, 16.42), ms8=(16.4, 16.2, 16.22, 16.41)))
    cases.append(("no gain", r["verdict"] == "NO_GAIN" and r["predictions"]["Q4"]["result"] == "MISSED"))
    cases.append(("slower", run(_box(ms4=(16.4, 16.5, 16.52, 16.42), ms8=(16.4, 16.6, 16.62, 16.41)))["verdict"] == "SLOWER"))
    cases.append(("rows beyond the bucket", run({**_box(), "rows": 65})["verdict"] == "VOID"))
    cases.append(("rows that fit a smaller bucket", run(_box(rows=32))["verdict"] == "VOID"))
    b = _box()
    del b["served"]["4"]["b"]
    cases.append(("a block missing", run(b)["verdict"] == "VOID"))
    bad = [n for n, okk in cases if not okk]
    if bad:
        print("p126_reduce self-test FAILED:", bad)
        return 1
    print(f"p126_reduce self-test OK ({len(cases)} cases)")
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
    print(f"P126_VERDICT {v['verdict']} {json.dumps(v.get('reasons') or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
