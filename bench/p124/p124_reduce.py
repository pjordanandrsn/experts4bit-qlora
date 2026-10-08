#!/usr/bin/env python3
"""Lane P124's reducer (bench/p124/PREREG-p124.md): checks the box ran as registered, then applies the registered rule
to ``E4B_ATTN_INT4_WIDE`` at 64 and 32 decode rows. Every speed bound is a ratio inside the box; quality is P110's bar.

In order, the first that fires is the verdict:
- **NO_READING**: no box record.
- **VOID**: the e4b or grouped-nf4-gemm commit or the model revision is not the registered one; the reading's stack is
  not SC2e's, or the engine was not built with the route enabled; a profile arm did not profile 8 padded eager steps of
  its bucket under its setting; a served arm is missing, was not captured under its setting, or did not replay its
  bucket exactly ``warm + steps + busy`` times with no eager step and no other bucket; KV bookkeeping is not bulk; a
  step, token or traced-step count is short; a quality arm is missing, short of windows, or ran other than its
  registered split, grouping, decode attention calls or graph status; **not engaged**: the route counts or the profile
  show a projection on the other route (with ``1``: no cached-bf16 call at 32 or 64 rows, one wide call per projection
  per step, one ``_gemm_int4_b32_smallm`` launch per projection per profiled step; with ``0``: no wide call), or ON64
  scored bit-equal to R in every window; **nondeterministic**: a setting's two served arms emit different tokens;
  **FUNCTION**: G64on's replayed tokens differ from ON64's eager ones; **a mutant survived**: ``mutant_scale`` or
  ``mutant_wide`` passes the bar.
- **PREMISE_ABSENT** (the reading only): with ``0`` the cuBLAS GEMMs the route replaces (the ``dense_gemm`` class, OFF
  minus ON) are under ``PREMISE_SHARE`` of the 64-row eager step's device time.
- **QUALITY_FAIL**: ON64 or ON32 fails P110's bar against the floor (``half``, ``chunk``; ``rep`` if R did not repeat).
- **NOISY**: at either depth, the two OFF arms' medians, or the two ON arms', differ by more than ``NOISE``.
- **DEFAULT_ON**: at both depths, ``ON_a / OFF_a`` and ``ON_b / OFF_b`` (median served step) are at most ``BAR``.
  **DEFAULT_ON_64** / **DEFAULT_ON_32**: at that depth only (the route is licensed for 33-64 or for 17-32 rows).
- **SLOWER**: all four ratios above 1. **NO_GAIN**: otherwise.

Reported, never gated: the predictions (Q1-Q9), every quality arm's statistics, the GPU busy fraction of every served
arm (the replay's device time over the step's wall), its peak memory and the shared workspace, and ON-against-OFF token
agreement. Floats are summed with ``math.fsum`` and medians taken by sorting, so the output is byte-identical on any
Python.

    p124_reduce.py --dir RUN --out verdict.json [--e4b-sha SHA]
    p124_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

GNF4_SHA = "5cff3bdf9d8eaf63505d676168de211e7ddcc7a5"     # grouped-nf4-gemm #522 (block_m=), its merge commit
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
READING_MODEL = "Qwen/Qwen3-30B-A3B"
B32, B64 = (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64)
SERVED = ("OFF_a", "ON_a", "ON_b", "OFF_b")
DEPTHS = ("64", "32")
WARM, STEPS, BUSY, PROFILED, PROFILE_WARM = 5, 256, 32, 8, 3
FLOORS = ("half", "chunk")
SUBJECTS = ("ON64", "ON32")
MUTANTS = ("mutant_scale", "mutant_wide")
QUALITY = ("R", "rep") + FLOORS + SUBJECTS + MUTANTS + ("G64on",)
Q_BUCKETS = {"R": B64, "rep": B64, "half": B32, "chunk": B64, "ON64": B64, "ON32": B32, "mutant_scale": B64,
             "mutant_wide": B64, "G64on": B64}
Q_WIDE = {a: a in SUBJECTS + MUTANTS + ("G64on",) for a in QUALITY}
SMALLM = "_gemm_int4_b32_smallm"
PREMISE_SHARE = 0.05      # the cuBLAS GEMMs the route replaces, as a share of the 64-row eager step (P119: 0.131)
NOISE = 0.015             # same-setting medians must agree within this, at each depth
BAR = 0.98                # ON / OFF in both ABBA pairs at a depth at most this licenses that depth
BIAS_SLACK, SPREAD_FLOOR = 0.01, 0.005     # P110's bar
# (name, statistic, lo, hi): written before the data; evaluated beside the verdict
PREDICTIONS = (("Q1", "attn_share_off_64", 0.08, 0.18),
               ("Q2", "int4_over_cublas_64", 0.25, 0.65),
               ("Q3", "served_ratios_64", 0.89, 0.97),
               ("Q4", "served_ratios_32", 0.86, 0.97),
               ("Q5", "busy_min", 0.90, 1.0),
               ("Q6", "same_setting_disagreement", 0.0, 0.005),
               ("Q7", "subject_bias", -0.003, 0.003),
               ("Q8", "mutant_bias", 0.3, 100.0),
               ("Q9", "memory_on_minus_off_mib_64", -64.0, 64.0))


def _fsum(xs) -> float:
    return math.fsum(float(x) for x in xs)


def median(xs) -> float:
    s = sorted(float(x) for x in xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else math.fsum(s[n // 2 - 1:n // 2 + 1]) / 2


def _depth_rows(box, d) -> int:
    return int(box.get("rows") or 0) if d == "64" else 32


def _bucket_for(n, buckets) -> int:
    return next(b for b in buckets if b >= n)


def _route_of(b: int, wide: bool) -> str:
    return "gemv" if b == 1 else ("k16" if b <= 16 else ("wide" if wide else "bf16"))


def expected_decode_route(windows: int, buckets, wide: bool, steps: int, n_lin: int) -> dict:
    """The decode calls a teacher-forced eager pass makes: every step, every piece of at most the largest bucket runs
    as one padded eager step of its bucket, and every projection sees the bucket's rows."""
    top, out = max(buckets), {}
    for i in range(0, windows, top):
        b = _bucket_for(min(top, windows - i), buckets)
        k = f"{_route_of(b, wide)}:{b}"
        out[k] = out.get(k, 0) + steps * n_lin
    return out


def _cls(table, name) -> list:
    return list((table.get("classes") or {}).get(name) or [0.0, 0.0])


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
    if box.get("attn_int4_wide_env") != "1":
        out.append(f"the engine was built with E4B_ATTN_INT4_WIDE={box.get('attn_int4_wide_env')!r}, registered '1'")
    cb = box.get("census_build") or {}
    if box.get("model") == READING_MODEL:
        if not (cb.get("int4_expert_layers") and cb.get("int4_expert_layers") == cb.get("moe_layers")):
            out.append(f"int4 experts on {cb.get('int4_expert_layers')} of {cb.get('moe_layers')} MoE layers")
        for k in ("int4_attn_projections", "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n"):
            if not cb.get(k):
                out.append(f"{k} is {cb.get(k)}: SC2e's stack has it")
    rows = int(box.get("rows") or 0)
    if not 32 < rows <= 64:
        out.append(f"rows {rows}: the 64-bucket step is one bucket-64 piece (33 to 64 rows)")
        return out
    if not int(box.get("int4_linears") or 0):
        out.append("no Int4Linear in the model: the route under test is absent")
        return out
    prof, srv = box.get("profile") or {}, box.get("served") or {}
    for d in DEPTHS:
        n, b = _depth_rows(box, d), int(d)
        for label, wide in (("off", False), ("on", True)):
            p = prof.get(f"{label}{d}")
            if p is None:
                out.append(f"profile {label}{d} missing")
                continue
            if p.get("wide") is not wide or p.get("rows") != n:
                out.append(f"profile {label}{d} ran wide={p.get('wide')} rows={p.get('rows')}")
            st = (p.get("profiled") or {}).get(d) or {}
            want = (PROFILED, 0, n * PROFILED, (b - n) * PROFILED)
            if (st.get("eager_steps"), st.get("replays"), st.get("rows"), st.get("pad_rows")) != want:
                out.append(f"profile {label}{d}: bucket {d} profiled {st}, registered {want} (eager, replays, rows, pad)")
            other = [k for k, g in (p.get("profiled") or {}).items() if k != d and (g.get("eager_steps") or g.get("replays"))]
            if other:
                out.append(f"profile {label}{d}: buckets {other} ran")
            if any(v != "eager: capture=False" for v in (p.get("graph_status") or {"?": None}).values()):
                out.append(f"profile {label}{d}: graph status {p.get('graph_status')}")
            if not p.get("device_ms"):
                out.append(f"profile {label}{d}: the profiler saw no device kernel")
        arms = srv.get(d) or {}
        total = WARM + STEPS + BUSY
        for label in SERVED:
            s = arms.get(label)
            if s is None:
                out.append(f"served d{d} {label} missing")
                continue
            if s.get("wide") is not label.startswith("ON") or s.get("rows") != n:
                out.append(f"served d{d} {label} ran wide={s.get('wide')} rows={s.get('rows')}")
            if s.get("bulk_kv") is not True:
                out.append(f"served d{d} {label}: KV bookkeeping not bulk (the server's default)")
            gs = s.get("graph_status") or {}
            if sorted(gs, key=int) != [str(x) for x in B64] or any(v != "graph" for v in gs.values()):
                out.append(f"served d{d} {label}: graph status {gs}")
            st = (s.get("graph_stats") or {}).get(d) or {}
            want = (total, 0, n * total, (b - n) * total)
            if (st.get("replays"), st.get("eager_steps"), st.get("rows"), st.get("pad_rows")) != want:
                out.append(f"served d{d} {label}: bucket {d} ran {st}, registered {want} (replays, eager, rows, pad)")
            other = [k for k, g in (s.get("graph_stats") or {}).items() if k != d and (g.get("eager_steps") or g.get("replays"))]
            if other:
                out.append(f"served d{d} {label}: buckets {other} ran")
            if len(s.get("step_ms") or ()) != STEPS or not all(float(x) > 0 for x in s.get("step_ms") or ()):
                out.append(f"served d{d} {label}: {len(s.get('step_ms') or ())} timed steps, registered {STEPS}")
            toks = s.get("tokens") or []
            if len(toks) != total or any(len(t) != n for t in toks):
                out.append(f"served d{d} {label}: tokens for {len(toks)} steps, registered {total} x {n} rows")
            busy = s.get("busy") or []
            if len(busy) != BUSY or not all(x[0] and x[0] > 0 and x[1] and x[1] > 0 for x in busy):
                out.append(f"served d{d} {label}: {len(busy)} traced steps, registered {BUSY}")
            if not (s.get("memory") or {}).get("max_allocated_mib"):
                out.append(f"served d{d} {label}: no peak memory recorded")
    q = box.get("quality") or {}
    w = int(q.get("windows") or 0)
    if w != rows:
        out.append(f"quality ran {w} windows, registered {rows}")
    eng, per = q.get("engagement") or {}, q.get("per_window") or {}
    C, layers = int(q.get("cont") or 0), int(q.get("layers") or 0)
    if C < 2 or not layers:
        out.append(f"quality: cont {C}, layers {layers}")
        return out
    for arm in QUALITY:
        e = eng.get(arm)
        if e is None:
            out.append(f"quality {arm} missing")
            continue
        if arm != "G64on" and len(per.get(arm) or ()) != w:
            out.append(f"quality {arm}: {len(per.get(arm) or ())} windows scored, registered {w}")
        if e.get("wide") is not Q_WIDE[arm] or bool(e.get("mutant_wide")) is not (arm == "mutant_wide"):
            out.append(f"quality {arm} ran wide={e.get('wide')} mutant_wide={e.get('mutant_wide')}")
        if list(e.get("buckets") or ()) != list(Q_BUCKETS[arm]):
            out.append(f"quality {arm}: buckets {e.get('buckets')}, registered {list(Q_BUCKETS[arm])}")
        fl = e.get("grouping_flags_in_pass") or {}
        if fl.get("device_grouping") is not True or fl.get("force_singleton_groups") is not False:
            out.append(f"quality {arm}: grouping flags {fl}")
        cap = arm == "G64on"
        gs = e.get("graph_status") or {}
        if not gs or any(v != ("graph" if cap else "eager: capture=False") for v in gs.values()):
            out.append(f"quality {arm}: graph status {gs}")
        pieces = -(-w // max(Q_BUCKETS[arm]))
        want_calls = 0 if cap else (C - 1) * layers * pieces
        if e.get("decode_calls") != want_calls:
            out.append(f"quality {arm}: {e.get('decode_calls')} decode attention calls, registered {want_calls}")
    return out


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        for j, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return [i, j]
    return None


def _differing(a, b) -> int:
    return sum(1 for x, y in zip(a, b) for u, v in zip(x, y) if u != v)


def engagement(box: dict) -> list:
    """Every projection on its registered route, from the counts the box took where Python ran."""
    out, n_lin = [], int(box["int4_linears"])
    for d in DEPTHS:
        b = int(d)
        for label, s in box["served"][d].items():
            r = s.get("capture_route") or {}
            for bb in (32, 64):
                wide_c, bf16_c = r.get(f"wide:{bb}", 0), r.get(f"bf16:{bb}", 0)
                if s["wide"] and (not wide_c or wide_c % n_lin or bf16_c):
                    out.append(f"served d{d} {label}: captures at {bb} rows took wide {wide_c}, bf16 {bf16_c}")
                if not s["wide"] and (wide_c or not bf16_c):
                    out.append(f"served d{d} {label}: captures at {bb} rows took wide {wide_c}, bf16 {bf16_c}")
        for label, wide in (("off", False), ("on", True)):
            p = box["profile"][f"{label}{d}"]
            r = p.get("route") or {}
            per = (PROFILE_WARM + PROFILED) * n_lin
            got = (r.get(f"wide:{b}", 0), r.get(f"bf16:{b}", 0))
            if got != ((per, 0) if wide else (0, per)):
                out.append(f"profile {label}{d}: decode calls wide/bf16 {got}, registered {(per, 0) if wide else (0, per)}")
            sm = _kcalls(p, SMALLM)
            if sm != (n_lin if wide else 0):
                out.append(f"profile {label}{d}: {sm} {SMALLM} launches per step, registered {n_lin if wide else 0}")
        dg = _cls(box["profile"][f"off{d}"], "dense_gemm")[0] - _cls(box["profile"][f"on{d}"], "dense_gemm")[0]
        if dg < n_lin:
            out.append(f"profile d{d}: the route removed {dg} cuBLAS launches per step, fewer than {n_lin} projections")
    q = box["quality"]
    C, w = int(q["cont"]), int(q["windows"])
    for arm in QUALITY:
        if arm == "G64on":
            r = q["engagement"][arm].get("route") or {}
            if not r.get("wide:64") or r.get("wide:64", 0) % n_lin or r.get("bf16:64", 0):
                out.append(f"quality G64on: captures took wide {r.get('wide:64', 0)}, bf16 {r.get('bf16:64', 0)} at 64")
            continue
        r = q["engagement"][arm].get("route") or {}
        want = expected_decode_route(w, Q_BUCKETS[arm], Q_WIDE[arm], C - 1, n_lin)
        dec = {k: v for k, v in r.items() if int(k.split(":")[1]) <= 64}
        if dec != want:
            out.append(f"quality {arm}: decode route {dec}, registered {want}")
    rn = {x["window"]: x["nll"] for x in q["per_window"]["R"]}
    if all(x["nll"] == rn[x["window"]] for x in q["per_window"]["ON64"]):
        out.append("ON64 scored bit-equal to R in every window: the route did not reach the scores")
    return out


def _arm_stats(per: dict, arm: str) -> dict:
    rn = {x["window"]: x["nll"] for x in per["R"]}
    d = [x["nll"] - rn[x["window"]] for x in per[arm]]
    n = len(d)
    bias = _fsum(d) / n
    sd = math.sqrt(_fsum((x - bias) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    kl = [x["kl"] for x in per[arm] if "kl" in x]
    return {"bias": round(bias, 6), "spread": round(_fsum(abs(x) for x in d) / n, 6), "se": round(sd / math.sqrt(n), 6),
            "max_abs": round(max(abs(x) for x in d), 6), "kl": round(_fsum(kl) / len(kl), 6) if kl else None,
            "argmax_agree": round(_fsum(x["argmax_agree"] for x in per[arm]) / n, 6)}


def quality_table(q: dict) -> dict:
    per = q["per_window"]
    stats = {a: _arm_stats(per, a) for a in per if a != "R"}
    floor = list(FLOORS) + ([] if q.get("rep_identical") else ["rep"])
    b_floor = max(abs(stats[f]["bias"]) for f in floor)
    s_floor = max(stats[f]["spread"] for f in floor)
    bias_bound = round(b_floor + BIAS_SLACK, 6)
    spread_bound = round(2 * max(s_floor, SPREAD_FLOOR), 6)
    passes = {a: stats[a]["bias"] <= bias_bound and stats[a]["spread"] <= spread_bound for a in SUBJECTS + MUTANTS}
    return {"stats": stats, "floor": floor, "B_floor": round(b_floor, 6), "S_floor": round(s_floor, 6),
            "bias_bound": bias_bound, "spread_bound": spread_bound, "passes": passes,
            "rep_identical": q.get("rep_identical"), "function": q.get("function")}


def tabulate(box: dict) -> dict:
    t = {"profile": {}, "served": {}}
    for d in DEPTHS:
        po, pn = box["profile"][f"off{d}"], box["profile"][f"on{d}"]
        row = {}
        for label, p in (("off", po), ("on", pn)):
            row[label] = {"device_ms": round(float(p["device_ms"]), 4),
                          "dense_gemm_ms": round(_cls(p, "dense_gemm")[1], 4),
                          "dense_gemm_calls": round(_cls(p, "dense_gemm")[0], 3),
                          "smallm_ms": round(_kms(p, SMALLM), 4), "smallm_calls": round(_kcalls(p, SMALLM), 3)}
        cublas = row["off"]["dense_gemm_ms"] - row["on"]["dense_gemm_ms"]
        row["attn_cublas_ms"] = round(cublas, 4)
        row["attn_share_off"] = round(cublas / row["off"]["device_ms"], 4)
        row["int4_over_cublas"] = round(row["on"]["smallm_ms"] / cublas, 4) if cublas > 0 else None
        row["eager_on_over_off"] = round(row["on"]["device_ms"] / row["off"]["device_ms"], 4)
        t["profile"][d] = row
        srv = box["served"][d]
        m = {k: median(srv[k]["step_ms"]) for k in SERVED}
        s = {"medians": {k: round(v, 4) for k, v in m.items()},
             "ratio_a": round(m["ON_a"] / m["OFF_a"], 4), "ratio_b": round(m["ON_b"] / m["OFF_b"], 4),
             "off_disagreement": round(abs(m["OFF_b"] / m["OFF_a"] - 1), 4),
             "on_disagreement": round(abs(m["ON_b"] / m["ON_a"] - 1), 4),
             "off_minus_on_ms": round(math.fsum([m["OFF_a"], m["OFF_b"], -m["ON_a"], -m["ON_b"]]) / 2, 4),
             "busy": {k: round(median([x[0] / x[1] for x in srv[k]["busy"]]), 4) for k in SERVED},
             "replay_ms": {k: round(median([x[0] for x in srv[k]["busy"]]), 4) for k in SERVED},
             "memory": {k: srv[k]["memory"] for k in SERVED}}
        s["ratios"] = [s["ratio_a"], s["ratio_b"]]
        mem = {k: float(srv[k]["memory"]["max_allocated_mib"]) for k in SERVED}
        s["memory_on_minus_off_mib"] = round(math.fsum([mem["ON_a"], mem["ON_b"], -mem["OFF_a"], -mem["OFF_b"]]) / 2, 1)
        s["tokens"] = {"on_vs_off_differing": _differing(srv["ON_a"]["tokens"], srv["OFF_a"]["tokens"]),
                       "on_vs_off_first_diff": _first_diff(srv["ON_a"]["tokens"], srv["OFF_a"]["tokens"]),
                       "compared": sum(len(x) for x in srv["OFF_a"]["tokens"])}
        t["served"][d] = s
    t["attn_share_off_64"] = t["profile"]["64"]["attn_share_off"]
    t["int4_over_cublas_64"] = t["profile"]["64"]["int4_over_cublas"]
    t["served_ratios_64"] = t["served"]["64"]["ratios"]
    t["served_ratios_32"] = t["served"]["32"]["ratios"]
    t["busy_min"] = min(v for d in DEPTHS for v in t["served"][d]["busy"].values())
    t["same_setting_disagreement"] = max(t["served"][d][k] for d in DEPTHS for k in ("off_disagreement", "on_disagreement"))
    t["memory_on_minus_off_mib_64"] = t["served"]["64"]["memory_on_minus_off_mib"]
    t["quality"] = quality_table(box["quality"])
    t["subject_bias"] = [t["quality"]["stats"][a]["bias"] for a in SUBJECTS]
    t["mutant_bias"] = [t["quality"]["stats"][a]["bias"] for a in MUTANTS]
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
        return {"lane": "P124", "verdict": "VOID", "reasons": void}
    void = engagement(box)
    t = tabulate(box)
    for d in DEPTHS:
        srv = box["served"][d]
        if srv["OFF_a"]["tokens"] != srv["OFF_b"]["tokens"] or srv["ON_a"]["tokens"] != srv["ON_b"]["tokens"]:
            void.append(f"nondeterministic at d{d}: a setting's two arms emitted different tokens")
    fn = box["quality"].get("function") or {}
    if not fn.get("positions") or fn.get("differ"):
        void.append(f"FUNCTION: G64on's replayed tokens against ON64's eager ones: {fn}")
    qt = t["quality"]
    for m in MUTANTS:
        if qt["passes"][m]:
            void.append(f"{m} survived: bias {qt['stats'][m]['bias']}, spread {qt['stats'][m]['spread']} pass the bar")
    base = {"lane": "P124", "model": box["model"], "tables": t}
    if void:
        return {**base, "verdict": "VOID", "reasons": void}
    base["predictions"] = predictions(t)
    if box["model"] == READING_MODEL and t["attn_share_off_64"] < PREMISE_SHARE:
        return {**base, "verdict": "PREMISE_ABSENT",
                "reasons": [f"premise: the replaced cuBLAS GEMMs are {t['attn_share_off_64']} of the 64-row eager step "
                            f"(needs >= {PREMISE_SHARE})"]}
    failed = [a for a in SUBJECTS if not qt["passes"][a]]
    if failed:
        return {**base, "verdict": "QUALITY_FAIL",
                "reasons": [f"{a}: bias {qt['stats'][a]['bias']} (bound {qt['bias_bound']}), spread "
                            f"{qt['stats'][a]['spread']} (bound {qt['spread_bound']})" for a in failed]}
    if t["same_setting_disagreement"] > NOISE:
        return {**base, "verdict": "NOISY",
                "reasons": [f"d{d}: OFF pair {t['served'][d]['off_disagreement']}, ON pair "
                            f"{t['served'][d]['on_disagreement']} (bound {NOISE})" for d in DEPTHS]}
    lic = [d for d in DEPTHS if all(r <= BAR for r in t["served"][d]["ratios"])]
    rs = {d: t["served"][d]["ratios"] for d in DEPTHS}
    if len(lic) == 2:
        return {**base, "verdict": "DEFAULT_ON", "reasons": [f"ON/OFF {rs} all <= {BAR}"]}
    if lic:
        return {**base, "verdict": f"DEFAULT_ON_{lic[0]}", "reasons": [f"ON/OFF {rs}: only d{lic[0]} <= {BAR}"]}
    if all(r > 1 for d in DEPTHS for r in rs[d]):
        return {**base, "verdict": "SLOWER", "reasons": [f"ON/OFF {rs} all > 1"]}
    return {**base, "verdict": "NO_GAIN", "reasons": [f"ON/OFF {rs}: no depth has both <= {BAR}"]}


def reduce(run: Path, e4b_sha: str) -> dict:
    p = run / "box.json"
    if not p.is_file():
        return {"lane": "P124", "verdict": "NO_READING", "reasons": ["no box record"]}
    return reduce_obj(json.loads(p.read_text()), e4b_sha)


# ------------------------------------------------------------------------------------------------ self-test --
N_LIN, LAYERS = 96, 48


def _kernels(wide: bool, device_ms: float, cublas_ms: float):
    """A per-step kernel table: OFF carries the projections as cuBLAS GEMMs (plus the head's), ON as K16 launches."""
    head = [["nvjet_tst_192x192_64x3_1x2_h_bz_TNT", 1.0, 1.2]]
    if wide:
        ks = head + [[f"{SMALLM}", float(N_LIN), round(cublas_ms * 0.45, 4)]]
    else:
        ks = head + [["nvjet_tst_64x64_64x8_1x1_v_bz_TNT", float(N_LIN), cublas_ms]]
    rest = device_ms - _fsum(k[2] for k in ks)
    return ks + [["_gemm_int4_b32_grouped_smallm_kernel", 96.0, round(rest, 4)]]


def _classes(ks):
    out = {}
    for k, c, ms in ks:
        cl = "dense_int4" if SMALLM in k else ("dense_gemm" if "nvjet" in k else "expert_int4")
        o = out.setdefault(cl, [0.0, 0.0])
        o[0], o[1] = round(o[0] + c, 3), round(o[1] + ms, 4)
    return out


def _box(model=READING_MODEL, e4b="a" * 40, rows=64, ms64=(17.0, 15.9, 15.92, 17.02), ms32=(12.0, 11.0, 11.01, 12.02),
         subject_shift=0.0004, mutant_shift=0.5, cont=128):
    total = WARM + STEPS + BUSY

    def toks(n, salt=0):
        return [[(i * 131 + r * 7 + salt) % 1000 for r in range(n)] for i in range(total)]

    def served(label, med, n, b):
        wide = label.startswith("ON")
        steps = [round(med + 0.01 * ((i % 5) - 2), 4) for i in range(STEPS)]
        cap = {}
        for bb in B64:
            k = f"{_route_of(bb, wide)}:{bb}"
            cap[k] = cap.get(k, 0) + 3 * N_LIN
        return {"wide": wide, "rows": n, "steps": STEPS, "warm": WARM, "busy_steps": BUSY, "bulk_kv": True,
                "graph_status": {str(x): "graph" for x in B64},
                "graph_stats": {str(x): ({"replays": total, "eager_steps": 0, "rows": n * total, "pad_rows": (b - n) * total}
                                         if x == b else {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0})
                                for x in B64},
                "capture_route": cap, "step_ms": steps, "median_ms": med,
                "busy": [[round(med * 0.95, 4), med] for _ in range(BUSY)],
                "memory": {"max_allocated_mib": 23000.0 + (7.3 if wide else 0.0), "max_reserved_mib": 24000.0,
                           "wide_workspace_mib": 7.3 if wide else 0.0},
                "tokens": toks(n, 1 if wide else 0)}

    def prof(wide, n, b, dms, cub):
        ks = _kernels(wide, dms, cub)
        st = {str(x): ({"replays": 0, "eager_steps": PROFILED, "rows": n * PROFILED, "pad_rows": (b - n) * PROFILED}
                       if x == b else {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}) for x in B64}
        per = (PROFILE_WARM + PROFILED) * N_LIN
        route = {f"{'wide' if wide else 'bf16'}:{b}": per, "bf16:512": n * N_LIN}
        return {"wide": wide, "rows": n, "device_ms": dms if not wide else round(dms - cub * 0.55, 4), "profiled": st,
                "graph_status": {str(x): "eager: capture=False" for x in B64}, "layers": LAYERS,
                "kernels": ks, "classes": _classes(ks), "route": route}

    def window_nll(arm, i):
        base = 2.0 + 0.01 * (i % 7)
        jitter = {"R": 0.0, "rep": 0.0, "half": 0.0015 * (1 if i % 2 else -1), "chunk": 0.001 * (1 if i % 3 else -1),
                  "ON64": subject_shift + 0.002 * (1 if i % 2 else -1), "ON32": subject_shift + 0.0018 * (1 if i % 2 else -1),
                  "mutant_scale": mutant_shift, "mutant_wide": mutant_shift * 2}[arm]
        return base + jitter

    q_eng, per = {}, {}
    for arm in QUALITY:
        cap = arm == "G64on"
        pieces = -(-rows // max(Q_BUCKETS[arm]))
        route = expected_decode_route(rows, Q_BUCKETS[arm], Q_WIDE[arm], cont - 1, N_LIN)
        if cap:
            route = {"wide:64": 3 * N_LIN, "wide:32": 3 * N_LIN, "k16:16": 3 * N_LIN}
        route["bf16:512"] = rows * N_LIN
        q_eng[arm] = {"buckets": list(Q_BUCKETS[arm]), "wide": Q_WIDE[arm], "mutant_wide": arm == "mutant_wide",
                      "grouping_flags_in_pass": {"device_grouping": True, "force_singleton_groups": False},
                      "graph_status": {str(x): ("graph" if cap else "eager: capture=False") for x in Q_BUCKETS[arm]},
                      "decode_calls": 0 if cap else (cont - 1) * LAYERS * pieces, "route": route}
        if not cap:
            per[arm] = [{"window": i, "nll": window_nll(arm, i), "argmax_agree": 0.98, **({"kl": 0.001} if arm != "R" else {})}
                        for i in range(rows)]
    box = {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "rows": rows,
           "attn_int4_wide_env": "1", "int4_linears": N_LIN,
           "census_build": {"moe_layers": 48, "int4_expert_layers": 48, "int4_attn_projections": 192, "fuse_qkv_n": 48,
                            "fuse_t1_glue_n": 193, "fuse_router_epilogue_n": 48},
           "profile": {"off64": prof(False, rows, 64, 15.64, 2.05), "on64": prof(True, rows, 64, 15.64, 2.05),
                       "off32": prof(False, 32, 32, 10.99, 1.9), "on32": prof(True, 32, 32, 10.99, 1.9)},
           "served": {"64": {k: served(k, m, rows, 64) for k, m in zip(SERVED, ms64)},
                      "32": {k: served(k, m, 32, 32) for k, m in zip(SERVED, ms32)}},
           "quality": {"windows": rows, "prompt": 512, "cont": cont, "chunk": 512, "floor_chunk": 256, "layers": LAYERS,
                       "rep_identical": True, "function": {"positions": rows * (cont - 1), "differ": 0},
                       "per_window": per, "engagement": q_eng}}
    return box


def self_test() -> int:
    cases = []

    def run(box, e4b="a" * 40):
        return reduce_obj(box, e4b)

    ok = run(_box())
    t = ok.get("tables") or {}
    cases.append(("a registered box flips the default at both depths", ok["verdict"] == "DEFAULT_ON"))
    cases.append(("ratios from medians", t.get("served_ratios_64") == [0.9353, 0.9354] and
                  t.get("served_ratios_32") == [0.9167, 0.916]))
    cases.append(("the premise share from the cuBLAS class, OFF minus ON", t.get("attn_share_off_64") == 0.1311))
    cases.append(("int4 over cuBLAS", t.get("int4_over_cublas_64") == 0.45))
    cases.append(("busy fraction", t.get("busy_min") == 0.95))
    cases.append(("memory ON minus OFF", t.get("memory_on_minus_off_mib_64") == 7.3))
    cases.append(("predictions all held", all(v["result"] == "HELD" for v in ok.get("predictions", {}).values())))
    cases.append(("median of an even count", median([1, 4, 2, 3]) == 2.5 and median([3, 1, 2]) == 2))
    cases.append(("the expected route of a 40-window pass on 32-row buckets",
                  expected_decode_route(40, B32, True, 10, 2) == {"wide:32": 20, "k16:8": 20}))
    cases.append(("wrong e4b", run(_box(), e4b="b" * 40)["verdict"] == "VOID"))
    b = _box()
    b["gnf4_sha"] = "f" * 40
    cases.append(("wrong grouped-nf4-gemm", run(b)["verdict"] == "VOID"))
    b = _box()
    b["attn_int4_wide_env"] = None
    cases.append(("engine built without the route", run(b)["verdict"] == "VOID"))
    b = _box()
    b["census_build"]["fuse_qkv_n"] = 0
    cases.append(("a fold missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["profile"]["on32"]["profiled"]["32"]["eager_steps"] = 7
    cases.append(("a profiled step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["64"]["ON_b"]["graph_status"]["64"] = "eager: RuntimeError: capture"
    cases.append(("bucket 64 not captured", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["32"]["OFF_a"]["graph_stats"]["32"]["eager_steps"] = 1
    cases.append(("an eager step in a served arm", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["64"]["ON_a"]["busy"] = b["served"]["64"]["ON_a"]["busy"][:-1]
    cases.append(("a traced step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["64"]["OFF_a"]["memory"] = {}
    cases.append(("no peak memory", run(b)["verdict"] == "VOID"))
    b = _box()
    del b["quality"]["engagement"]["mutant_wide"]
    cases.append(("a quality arm missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["quality"]["engagement"]["ON32"]["decode_calls"] -= 1
    cases.append(("decode attention calls off the split", run(b)["verdict"] == "VOID"))
    b = _box()
    b["served"]["64"]["ON_a"]["capture_route"]["bf16:64"] = N_LIN
    r = run(b)
    cases.append(("not engaged: a captured projection stayed on cuBLAS", r["verdict"] == "VOID" and "captures" in r["reasons"][0]))
    b = _box()
    b["profile"]["on64"]["kernels"][1][1] = 48.0
    r = run(b)
    cases.append(("not engaged: half the K16 launches", r["verdict"] == "VOID" and SMALLM in " ".join(r["reasons"])))
    b = _box()
    b["quality"]["engagement"]["ON64"]["route"] = {"bf16:64": 127 * N_LIN, "bf16:512": 64 * N_LIN}
    cases.append(("not engaged: ON64 scored on the bf16 route", run(b)["verdict"] == "VOID"))
    b = _box()
    for x in b["quality"]["per_window"]["ON64"]:
        x["nll"] = 2.0 + 0.01 * (x["window"] % 7)
    r = run(b)
    cases.append(("ON64 bit-equal to R", r["verdict"] == "VOID" and "bit-equal" in " ".join(r["reasons"])))
    b = _box()
    b["served"]["32"]["OFF_b"]["tokens"][100][3] += 1
    r = run(b)
    cases.append(("nondeterministic", r["verdict"] == "VOID" and "nondeterministic" in " ".join(r["reasons"])))
    b = _box()
    b["quality"]["function"]["differ"] = 1
    cases.append(("FUNCTION", run(b)["verdict"] == "VOID"))
    r = run(_box(mutant_shift=0.001))
    cases.append(("a mutant survived", r["verdict"] == "VOID" and "survived" in " ".join(r["reasons"])))
    b = _box()
    for k in ("off64", "on64"):
        b["profile"][k] = _box()["profile"][k]
    b["profile"]["off64"]["classes"]["dense_gemm"][1] = round(b["profile"]["on64"]["classes"]["dense_gemm"][1] + 0.5, 4)
    r = run(b)
    cases.append(("premise absent", r["verdict"] == "PREMISE_ABSENT" and r["tables"]["attn_share_off_64"] < PREMISE_SHARE))
    b = _box(model="ibm-granite/granite-3.1-3b-a800m-instruct", rows=40)
    b["census_build"] = {"moe_layers": 32}
    b["profile"]["off64"]["classes"]["dense_gemm"][1] = b["profile"]["on64"]["classes"]["dense_gemm"][1] + 0.1
    cases.append(("the proof's Granite reads without the premise share", run(b)["verdict"] == "DEFAULT_ON"))
    r = run(_box(subject_shift=0.02))
    cases.append(("quality fail", r["verdict"] == "QUALITY_FAIL" and r["predictions"]["Q7"]["result"] == "MISSED"))
    b = _box()
    b["quality"]["rep_identical"] = False
    for x in b["quality"]["per_window"]["rep"]:
        x["nll"] += 0.004
    r = run(b)
    cases.append(("rep joins the floor when R does not repeat", r["tables"]["quality"]["floor"] == ["half", "chunk", "rep"]
                  and r["tables"]["quality"]["B_floor"] == 0.004))
    cases.append(("noisy", run(_box(ms32=(12.0, 11.0, 11.01, 12.4)))["verdict"] == "NOISY"))
    cases.append(("only the 64-row depth", run(_box(ms32=(12.0, 11.9, 11.91, 12.02)))["verdict"] == "DEFAULT_ON_64"))
    cases.append(("only the 32-row depth", run(_box(ms64=(17.0, 16.9, 16.91, 17.02)))["verdict"] == "DEFAULT_ON_32"))
    r = run(_box(ms64=(17.0, 16.9, 16.91, 17.02), ms32=(12.0, 11.9, 11.91, 12.02)))
    cases.append(("no gain", r["verdict"] == "NO_GAIN" and r["predictions"]["Q3"]["result"] == "MISSED"))
    cases.append(("slower", run(_box(ms64=(17.0, 17.1, 17.12, 17.02), ms32=(12.0, 12.1, 12.11, 12.02)))["verdict"] == "SLOWER"))
    cases.append(("rows beyond the bucket", run({**_box(), "rows": 65})["verdict"] == "VOID"))
    cases.append(("rows that fit a smaller bucket", run(_box(rows=32))["verdict"] == "VOID"))
    bad = [n for n, okk in cases if not okk]
    if bad:
        print("p124_reduce self-test FAILED:", bad)
        return 1
    print(f"p124_reduce self-test OK ({len(cases)} cases)")
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
    print(f"P124_VERDICT {v['verdict']} {json.dumps(v.get('reasons') or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
