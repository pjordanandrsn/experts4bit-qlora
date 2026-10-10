#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2f_reduce.py -- lane SC2f's registered rule (bench/sc2/SC2f-PREREG.md; #846): e4b serve_paged at 64 slots with
both of P130's prefill defaults (``E4B_FUSE_PREFILL_GLUE`` and ``E4B_PREFILL_LEAN_DISPATCH`` unset: the release's
defaults, arm ``on``) against both knobs at 0 (arm ``off``), from box N's run files.

Servers, each its own cold start, tagged ``e4b_<arm>_d<draw><rep>``: draw 1 runs on_a, off_a, off_b, on_b; draw 2 runs
off_a, on_a, on_b, off_b. Files in DIR, per server S:
  S_warm.json, S_burst.json, S_serial.json   4 warm (seed 999), the burst (64 at once, seed 998), serial 24 (seed = draw)
  S_r<R>.json                                the ladder: R in 4, 8, 12, 13, 14, 16; seed draw x 100 + R
  S_k<R>_<j>.json                            the knee block: R in 12, 13, 14; j in 1..4; seed 10000 x draw + 100 x R + j
  health_S_start.json / _end.json, trace_S.jsonl, steps_S.jsonl, vram_S.txt
Every server of a draw runs the same seeds, so the two servers of one arm in a draw are a same-seed replicate.

**Gates, per server** (a failure names the server and VOIDs the reading):
- ROUTES (``sc2c_reduce.routes_bad``) and SLOTS at 64 slots, buckets up to 64 (``sc2e_reduce.slots_bad``, arm s64a), on
  the start /health;
- FLIPS (``flips_bad``), on the start /health: ``on`` resolves ``E4B_FUSE_PREFILL_GLUE`` to 1 from the family default
  (``levers.prefill_glue`` = {mode 1, source default-allowlisted}), both raw env values null and
  ``prefill_lean_dispatch`` auto, every K19 call above 256 rows ``chained|lean``, and the norm, layer and attention folds
  each seen on prefill rows; ``off`` reads mode 0 from an explicit 0, both env values "0", every K19 call above 256 rows
  ``chained|gather``, and no prefill fold;
- ENGAGED (``sc2e_reduce.engagement``, arm s64a) on the end /health, with every request the server admitted counted;
- PROMPTS: every request reports 512 prompt tokens.

**Statistics.** A *row* is one arm at one ladder rate over its four servers: UNREAD if a run is missing, INVALID if a run
has an invalid request, else VALID; it *meets capacity* iff every server attains >= 0.95. An arm's **ceiling** is the
largest ladder rate whose row is VALID and meets capacity (None if a row is UNREAD or INVALID). A server's **knee** is its
mean attainment over the knee block's 15 runs (12, 13 and 14 req/s; the ladder seed and four more each). Per draw,
**gain** = mean knee of the draw's two ``on`` servers minus that of its two ``off`` servers, and **noise** = the larger
same-arm replicate difference |a - b| in the draw; **bar** = max(0.03, noise).

**Outcome**, decoupled from Q1-Q6:
- VOID if a server fails a gate; UNREAD if a needed number is missing;
- REGRESSED if ``on``'s ceiling is below ``off``'s, or at 4 or 8 req/s an ``on`` server attains below the draw's lower
  ``off`` server minus 0.05, or gain <= -bar in both draws;
- FASTER if gain >= bar in both draws. The capacity number moves to ``on``'s ceiling iff that ceiling is above SC2e's 12
  (``readme_number``; None otherwise: the number stays 12);
- FLAT otherwise.

**Predictions** (each per draw, means over the draw's servers of an arm; HOLDS / MISSED / UNREAD; no outcome reads them):
  Q1  prefill forward device time p50 (``sc2e_census``): on - off in [-7.0, -3.5] ms.
  Q2  serial p50: TTFT on/off in [0.83, 0.93]; TPOT on/off in [0.98, 1.02].
  Q3  at 8 and 12 req/s TPOT p50 on/off in [0.85, 0.99]; at 8 req/s TTFT p50 on/off in [0.78, 0.97].
  Q4  gain in [+0.03, +0.12].
  Q5  on's ceiling >= off's, and >= 12 (over the reading, not per draw).
  Q6  decode-only step p50 at bucket 64: on/off in [0.95, 1.02].

**Reported, no bar:** per server the census (``sc2e_census.steps`` / ``per_rate``), VRAM, ``free_after_mib``,
``pool_mib``, the K19 dispatch and fold counts; per draw the text agreement between paired on and off servers.

  sc2f_reduce.py --dir DIR [--out verdict.json]
  sc2f_reduce.py --flips HEALTH.json ARM [--proof]       (box N's FLIPS check; exit 1 on a departure)
  sc2f_reduce.py --engagement END.json ... --arms ARM ...   (the proof's servers, counts not equated)
  sc2f_reduce.py --self-test
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("on", "off")
DRAWS = (1, 2)
ORDER = {1: ("on_a", "off_a", "off_b", "on_b"), 2: ("off_a", "on_a", "on_b", "off_b")}
LADDER = (4, 8, 12, 13, 14, 16)
KNEE = (12, 13, 14)
KNEE_J = (1, 2, 3, 4)
SLOT_ARM = "s64a"                 # SC2e's arm: 64 slots, buckets up to 64 (the defaults since 0.50.0 at 2,048 tokens)
T = 512
CAP_SHARE = 0.95
SC2E_CEILING = 12
KNEE_MIN = 0.03
NOREG_TOL = 0.05
NOREG_RATES = (4, 8)
Q1_BAND = (-7.0, -3.5)
Q2_TTFT, Q2_TPOT = (0.83, 0.93), (0.98, 1.02)
Q3_TPOT, Q3_TTFT8 = (0.85, 0.99), (0.78, 0.97)
Q4_BAND = (0.03, 0.12)
Q6_BAND = (0.95, 1.02)
MIN_STEPS = 20
FOLDS = ("norm", "layer", "attention")
WANT_FLIPS = {
    "on": {"glue": {"mode": "1", "source": "default-allowlisted"}, "glue_env": None, "lean_env": None, "lean": "auto",
           "gt256": {"chained|lean|gt256"}, "folds": True},
    "off": {"glue": {"mode": "0", "source": "explicit"}, "glue_env": "0", "lean_env": "0", "lean": "0",
            "gt256": {"chained|gather|gt256"}, "folds": False}}
# the proof's model (Granite NF4) is not qwen3_moe and takes no int4 K19: the knobs resolve, nothing engages
WANT_FLIPS_PROOF = {
    "on": {"glue": {"mode": "0", "source": "default-off"}, "glue_env": None, "lean_env": None, "lean": "auto",
           "gt256": None, "folds": False},
    "off": {"glue": {"mode": "0", "source": "explicit"}, "glue_env": "0", "lean_env": "0", "lean": "0",
            "gt256": None, "folds": False}}


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _jsonl(d, name):
    p = os.path.join(d, name)
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else None


def servers():
    """(tag, arm, draw) in run order."""
    return [(f"e4b_{s[:-2]}_d{k}{s[-1]}", s[:-2], k) for k in DRAWS for s in ORDER[k]]


def flips_bad(health, arm, proof=False) -> dict:
    """Why a server's /health does not read as its arm of P130's two knobs (empty: as registered)."""
    w = (WANT_FLIPS_PROOF if proof else WANT_FLIPS)[arm]
    h = health or {}
    r = h.get("prefill_routes") or {}
    bad = {}
    glue = (h.get("levers") or {}).get("prefill_glue")
    if glue != w["glue"]:
        bad["levers.prefill_glue"] = glue if glue is not None else "<missing>"
    for k, v in (("prefill_glue_env", w["glue_env"]), ("prefill_lean_dispatch_env", w["lean_env"]),
                 ("prefill_lean_dispatch", w["lean"])):
        if r.get(k, "<missing>") != v:
            bad[f"prefill_routes.{k}"] = r.get(k, "<missing>")
    seen = r.get("seen") or {}
    if w["gt256"] is not None:
        gt = {k for k, n in (seen.get("moe_k19_dispatch") or {}).items() if k.endswith("|gt256") and n}
        if gt != w["gt256"]:
            bad["seen.moe_k19_dispatch.gt256"] = sorted(gt) or "<none>"
    folds = {k: n for k, n in (seen.get("prefill_folds") or {}).items() if n}
    if w["folds"] and not all(folds.get(f) for f in FOLDS):
        bad["seen.prefill_folds"] = folds or "<none>"
    if not w["folds"] and folds:
        bad["seen.prefill_folds"] = folds
    return bad


def _names(tag):
    return ([f"{tag}_{x}" for x in ("warm", "burst", "serial")] + [f"{tag}_r{r}" for r in LADDER]
            + [f"{tag}_k{r}_{j}" for r in KNEE for j in KNEE_J])


def _prompts(d, tag):
    return sum(len(x["requests"]) for x in (_load(d, n + ".json") for n in _names(tag)) if x)


def _plan(d, tag):
    out = []
    for n in _names(tag):
        x = _load(d, n + ".json")
        if x:
            out.append((n[len(tag) + 1:], len(x["requests"])))
    return out


def _summ(d, tag, w):
    x = _load(d, f"{tag}_{w}.json")
    return x["summary"] if x else None


def _att(s):
    return None if s is None or s["invalid"] else s["attainment"]


def knee(d, tag):
    """A server's mean attainment over the knee block's 15 runs, or None if one is missing or invalid."""
    a = [_att(_summ(d, tag, w)) for r in KNEE for w in [f"r{r}"] + [f"k{r}_{j}" for j in KNEE_J]]
    return None if None in a else round(statistics.fmean(a), 4)


def row(d, arm, rate):
    tags = [t for t, a, _ in servers() if a == arm]
    sums = [_summ(d, t, f"r{rate}") for t in tags]
    if any(s is None for s in sums):
        return {"status": "UNREAD", "why": "a run is missing"}
    att = [s["attainment"] for s in sums]
    if any(s["invalid"] for s in sums):
        return {"status": "INVALID", "attainment": att}
    return {"status": "VALID", "attainment": att, "meets_capacity": all(a >= CAP_SHARE for a in att)}


def ceiling(rows):
    if any(rows[r]["status"] in ("UNREAD", "INVALID") for r in LADDER):
        return None
    ok = [r for r in LADDER if rows[r]["meets_capacity"]]
    return max(ok) if ok else 0


def _agreement(a, b):
    if not a or not b or len(a["requests"]) != len(b["requests"]):
        return None
    eq = [x.get("valid") and y.get("valid") and x.get("text") == y.get("text") for x, y in zip(a["requests"], b["requests"])]
    return round(sum(eq) / len(eq), 4) if eq else None


def census(d) -> dict:
    cen = _sib("sc2e_census")
    out = {}
    for tag, arm, k in servers():
        c = {}
        steps = _jsonl(d, f"steps_{tag}.jsonl")
        c["steps"] = cen.steps(steps, SLOT_ARM) if steps else {"why": "no step trace"}
        trace = _jsonl(d, f"trace_{tag}.jsonl")
        try:
            c["per_rate"] = cen.per_rate(trace, _plan(d, tag)) if trace else {"why": "no request trace"}
        except SystemExit as e:
            c["per_rate"] = {"why": str(e)}
        v = os.path.join(d, f"vram_{tag}.txt")
        txt = open(v).read().strip() if os.path.exists(v) else ""
        c["vram_used_mib"] = int(txt.split(",")[0]) if txt else None
        h = _load(d, f"health_{tag}_start.json") or {}
        g = h.get("prefill_graph") or {}
        c["at_ready"] = {**{x: g.get(x) for x in ("status", "pool_mib", "free_after_mib", "bulk_flush_mib")},
                         "kv_pool_mib": ((h.get("levers") or {}).get("kv") or {}).get("pool_mib")}
        seen = (h.get("prefill_routes") or {}).get("seen") or {}
        c["k19_dispatch"], c["prefill_folds"] = seen.get("moe_k19_dispatch"), seen.get("prefill_folds")
        c["graph_stats_end"] = ((_load(d, f"health_{tag}_end.json") or {}).get("engine") or {}).get("graph_stats")
        out[tag] = c
    for k in DRAWS:
        for rep in ("a", "b"):
            on, off = f"e4b_on_d{k}{rep}", f"e4b_off_d{k}{rep}"
            out[on]["text_agreement_vs_off"] = {
                w: _agreement(_load(d, f"{off}_{w}.json"), _load(d, f"{on}_{w}.json"))
                for w in ["serial"] + [f"r{r}" for r in LADDER]}
    return out


def _band(xs, lo, hi):
    if any(x is None for x in xs):
        return "UNREAD"
    return "HOLDS" if all(lo <= x <= hi for x in xs) else "MISSED"


def _mean(xs):
    return None if not xs or any(x is None for x in xs) else statistics.fmean(xs)


def reduce(d: str) -> dict:
    c_red, e_red = _sib("sc2c_reduce"), _sib("sc2e_reduce")
    out = {"gates": {}, "arms": {}, "knee": {}, "predictions": {}}
    srv = servers()
    start = {t: _load(d, f"health_{t}_start.json") for t, _, _ in srv}
    routes = {t: c_red.routes_bad(start[t]) for t, _, _ in srv}
    slots = {t: e_red.slots_bad(start[t], SLOT_ARM) for t, _, _ in srv}
    flips = {t: flips_bad(start[t], a) for t, a, _ in srv}
    eng = {t: e_red.engagement(SLOT_ARM, _load(d, f"health_{t}_end.json"), _prompts(d, t)) for t, _, _ in srv}
    prm = {}
    for t, _, _ in srv:
        off = {}
        for n in _names(t):
            x = _load(d, n + ".json")
            lens = {r.get("prompt_tokens") for r in x["requests"]} if x else {T}
            if lens != {T}:
                off[n] = sorted(lens, key=str)
        prm[t] = off
    for name, g in (("routes", routes), ("slots", slots), ("flips", flips), ("engaged", eng), ("prompts", prm)):
        out["gates"][name] = {"ok": not any(g.values()), "bad": {k: v for k, v in g.items() if v}}
    void = sorted(t for t, _, _ in srv if routes[t] or slots[t] or flips[t] or eng[t] or prm[t])
    for arm in ARMS:
        rows = {r: row(d, arm, r) for r in LADDER}
        out["arms"][arm] = {"rows": {str(r): v for r, v in rows.items()}, "ceiling": ceiling(rows)}
    kn = {t: knee(d, t) for t, _, _ in srv}
    gain, noise, bar = {}, {}, {}
    for k in DRAWS:
        on, off = [kn[f"e4b_on_d{k}{x}"] for x in "ab"], [kn[f"e4b_off_d{k}{x}"] for x in "ab"]
        if None in on + off:
            gain[k] = noise[k] = bar[k] = None
            continue
        gain[k] = round(statistics.fmean(on) - statistics.fmean(off), 4)
        noise[k] = round(max(abs(on[0] - on[1]), abs(off[0] - off[1])), 4)
        bar[k] = round(max(KNEE_MIN, noise[k]), 4)
    out["knee"] = {"server": kn, "gain": gain, "noise": noise, "bar": bar}
    cen = out["census"] = census(d)

    def per_draw(arm, k, fn):
        return _mean([fn(f"e4b_{arm}_d{k}{x}") for x in "ab"])

    def ratio(k, fn):
        a, b = per_draw("on", k, fn), per_draw("off", k, fn)
        return round(a / b, 4) if a is not None and b else None

    def fwd(t):
        return (cen[t]["steps"].get("prefill_steps") or {}).get("forward_device_ms_p50")

    def dec64(t):
        s = (cen[t]["steps"].get("decode_steps") or {}).get("64") or {}
        return s.get("step_ms_p50") if s.get("n", 0) >= MIN_STEPS else None

    def summ_get(w, m):
        return lambda t: (_summ(d, t, w) or {}).get(m) if _summ(d, t, w) and not _summ(d, t, w)["invalid"] else None
    v = out["predictions"]
    q1 = []
    for k in DRAWS:
        a, b = per_draw("on", k, fwd), per_draw("off", k, fwd)
        q1.append(round(a - b, 3) if a is not None and b is not None else None)
    v["Q1"] = {"verdict": _band(q1, *Q1_BAND), "on_minus_off_ms": q1, "band": list(Q1_BAND)}
    tt = [ratio(k, summ_get("serial", "ttft_p50_s")) for k in DRAWS]
    tp = [ratio(k, summ_get("serial", "tpot_p50_s")) for k in DRAWS]
    q2 = _band(tt, *Q2_TTFT), _band(tp, *Q2_TPOT)
    v["Q2"] = {"verdict": "UNREAD" if "UNREAD" in q2 else ("HOLDS" if q2 == ("HOLDS", "HOLDS") else "MISSED"),
               "ttft_on_over_off": tt, "tpot_on_over_off": tp}
    t3 = [ratio(k, summ_get(f"r{r}", "tpot_p50_s")) for r in (8, 12) for k in DRAWS]
    f3 = [ratio(k, summ_get("r8", "ttft_p50_s")) for k in DRAWS]
    q3 = _band(t3, *Q3_TPOT), _band(f3, *Q3_TTFT8)
    v["Q3"] = {"verdict": "UNREAD" if "UNREAD" in q3 else ("HOLDS" if q3 == ("HOLDS", "HOLDS") else "MISSED"),
               "tpot_on_over_off_r8_r12": t3, "ttft_on_over_off_r8": f3}
    v["Q4"] = {"verdict": _band([gain[k] for k in DRAWS], *Q4_BAND), "gain": [gain[k] for k in DRAWS]}
    con, coff = out["arms"]["on"]["ceiling"], out["arms"]["off"]["ceiling"]
    v["Q5"] = {"verdict": "UNREAD" if con is None or coff is None else
               ("HOLDS" if con >= coff and con >= SC2E_CEILING else "MISSED"), "ceilings": {"on": con, "off": coff}}
    v["Q6"] = {"verdict": _band([ratio(k, dec64) for k in DRAWS], *Q6_BAND),
               "decode64_on_over_off": [ratio(k, dec64) for k in DRAWS]}
    noreg, nr_why = True, []
    for r in NOREG_RATES:
        for k in DRAWS:
            on = [_att(_summ(d, f"e4b_on_d{k}{x}", f"r{r}")) for x in "ab"]
            off = [_att(_summ(d, f"e4b_off_d{k}{x}", f"r{r}")) for x in "ab"]
            if None in on + off:
                noreg = None
                continue
            low = [a for a in on if a < min(off) - NOREG_TOL]
            if low:
                nr_why.append(f"r{r} d{k}: on {on} against off {off}")
    if noreg is not None:
        noreg = not nr_why
    out["no_regression"] = {"holds": noreg, "why": nr_why}
    readme = None
    if void:
        verdict = "VOID"
    elif con is None or coff is None or noreg is None or None in gain.values():
        verdict = "UNREAD"
    elif con < coff or not noreg or all(gain[k] <= -bar[k] for k in DRAWS):
        verdict = "REGRESSED"
    elif all(gain[k] >= bar[k] for k in DRAWS):
        verdict = "FASTER"
        readme = con if con > SC2E_CEILING else None
    else:
        verdict = "FLAT"
    out["outcome"] = {"verdict": verdict, "readme_number": readme, "void": void}
    return out


def engagement_cli(ends, arms) -> int:
    """The proof's servers: each end /health engaged as SC2e reads it at 64 slots (counts not equated)."""
    e_red = _sib("sc2e_reduce")
    why = []
    for p, arm in zip(ends, arms):
        h = json.load(open(p)) if os.path.exists(p) else None
        why += [f"{arm}: {w}" for w in e_red.engagement(SLOT_ARM, h, None)]
    print("SC2F_ENGAGEMENT " + json.dumps({"ok": not why, "why": why}), flush=True)
    return 1 if why else 0


# ------------------------------------------------------------------ self-test --

def _fixture(d, *, knee_att=None, ceil=None, att=None, flips=None, slots=None, routes_seen=None, replays=None,
             admitted=None, prompt_tokens=T, missing=(), invalid=(), fwd_ms=None, dec64_ms=None, serial=None, tpot=None):
    """Box N's run files for all eight servers. Defaults: off at the knee 0.90, on 0.95 (each knee run); both ceilings
    12; every gate passing; on's prefill forward 5.2 ms shorter."""
    c_red = _sib("sc2c_reduce")
    knee_att = {"on": 0.95, "off": 0.90, **(knee_att or {})}
    ceil = {"on": 12, "off": 12, **(ceil or {})}
    fwd_ms = {"on": 34.8, "off": 40.0, **(fwd_ms or {})}
    dec64_ms = {"on": 18.2, "off": 18.4, **(dec64_ms or {})}
    serial = {"on": (0.0355, 0.0042), "off": (0.0405, 0.0042), **(serial or {})}
    tpot = {"on": 0.029, "off": 0.031, **(tpot or {})}

    def summ(ttft, tp, a, rate, inv=0):
        return {"valid": 120 - inv, "invalid": inv, "errors": [], "ttft_p50_s": ttft, "ttft_p99_s": ttft * 2,
                "tpot_p50_s": tp, "tpot_p99_s": tp * 2, "e2el_p50_s": 1.0, "output_tok_s": 100.0, "attainment": a,
                "goodput_rps": a * rate, "achieved_rate": rate}

    def dump(name, obj):
        json.dump(obj, open(os.path.join(d, name), "w"))
    for tag, arm, k in servers():
        if tag in missing:
            continue

        def reqs(n, txt="x"):
            return [{"valid": True, "text": f"{txt}{i}", "prompt_tokens": prompt_tokens} for i in range(n)]
        dump(f"{tag}_warm.json", {"requests": reqs(4)})
        dump(f"{tag}_burst.json", {"requests": reqs(64)})
        dump(f"{tag}_serial.json", {"requests": reqs(24, arm), "summary": summ(*serial[arm], 0.0, 0.0)})
        for r in LADDER:
            a = 1.0 if r <= ceil[arm] else 0.7
            a = (att or {}).get((tag, r), a)
            ttft = 0.05 if arm == "on" else 0.055
            dump(f"{tag}_r{r}.json", {"requests": reqs(120),
                                      "summary": summ(ttft, tpot[arm], a, r, 1 if (tag, f"r{r}") in invalid else 0)})
        for r in KNEE:
            for j in KNEE_J:
                a = (knee_att.get(tag) if tag in knee_att else knee_att[arm])
                dump(f"{tag}_k{r}_{j}.json", {"requests": reqs(120), "summary": summ(0.1, 0.03, a, r)})
        n = (admitted or {}).get(tag, 4 + 64 + 24 + 120 * (len(LADDER) + len(KNEE) * len(KNEE_J)))
        w = WANT_FLIPS[arm]
        seen = {"moe": {"int4_k19|gt256": 240}, "prefill_attn": {"flash": 240},
                "moe_k19_dispatch": {next(iter(w["gt256"])): 96, "one_launch|lean|le256": 480},
                "prefill_folds": {f: 48 for f in FOLDS} if w["folds"] else {}}
        if routes_seen and tag in routes_seen:
            seen.update(routes_seen[tag])
        pr = dict(c_red.ROUTES, seen=seen, prefill_glue_env=w["glue_env"], prefill_lean_dispatch_env=w["lean_env"],
                  prefill_lean_dispatch=w["lean"])
        lev = {"kv": {"scratch_slots": 64, "pool_mib": 6678.8}, "prefill_glue": dict(w["glue"])}
        for key, val in ((flips or {}).get(tag) or {}).items():
            (lev if key == "prefill_glue" else pr)[key] = val
        b = [1, 2, 4, 8, 16, 32, 64]
        eng = {"chunk_tokens": T, "max_prefill_tokens_per_step": T, "max_seqs": 64, "kv_slots": 64, "buckets": b,
               "buckets_requested": "auto", "graph_status": {str(x): "graph" for x in b}}
        eng.update((slots or {}).get(tag) or {})
        dump(f"health_{tag}_start.json", {"prefill_routes": pr, "engine": eng, "levers": lev,
                                          "prefill_graph": {"status": "on", "pool_mib": 428, "free_after_mib": 3700,
                                                            "bulk_flush_mib": 216}})
        dump(f"health_{tag}_end.json", {
            "prefill_graph": {"status": "on", "T": T, "replays": (replays or {}).get(tag, n), "eager_chunks": 0},
            "kv_bookkeeping": {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0,
                               "ready_bulk": 0, "ready_at_flush": n},
            "engine": {"graph_stats": {str(x): {"replays": 5, "eager_steps": 0, "rows": 5, "pad_rows": 0} for x in b}}})
        open(os.path.join(d, f"vram_{tag}.txt"), "w").write("28430, 32607\n")
        rows = [{"step": i, "t": i * 0.05, "step_ms": dec64_ms[arm], "decode_rows": 60, "bucket": 64, "dec_pieces": 1,
                 "seg": {"dec_sync": 1.0}, "gpu": {"dec_prep": 0.5, "dec_issue": dec64_ms[arm] - 0.5}} for i in range(30)]
        rows += [{"step": 30 + i, "t": 2.0 + i * 0.05, "step_ms": fwd_ms[arm] + 8, "prefill_tokens": T, "prefill_replays": 1,
                  "decode_rows": 0, "seg": {"pf_forward": fwd_ms[arm]},
                  "gpu": {"pf_prep": 0.2, "pf_forward": 0.2 + fwd_ms[arm], "pf_flush": 0.4 + fwd_ms[arm]}} for i in range(30)]
        with open(os.path.join(d, f"steps_{tag}.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(x) for x in rows) + "\n")


def self_test() -> int:
    import tempfile
    cases = []

    def run(**kw):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            return reduce(d)
    o = run()
    cases.append(("faster, the number stays 12", o["outcome"] == {"verdict": "FASTER", "readme_number": None, "void": []}
                  and o["knee"]["gain"] == {1: 0.04, 2: 0.04} and o["knee"]["bar"] == {1: 0.03, 2: 0.03}))
    cases.append(("predictions hold", all(o["predictions"][q]["verdict"] == "HOLDS" for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
                  and o["predictions"]["Q1"]["on_minus_off_ms"] == [-5.2, -5.2]))
    o = run(ceil={"on": 13})
    cases.append(("faster, the number moves to 13", o["outcome"]["verdict"] == "FASTER" and o["outcome"]["readme_number"] == 13))
    o = run(ceil={"on": 13, "off": 13}, knee_att={"on": 0.91})
    cases.append(("a ceiling of 13 inside the noise is FLAT", o["outcome"]["verdict"] == "FLAT"
                  and o["outcome"]["readme_number"] is None))
    o = run(knee_att={"on": 0.90})
    cases.append(("flat", o["outcome"]["verdict"] == "FLAT" and o["predictions"]["Q4"]["verdict"] == "MISSED"))
    o = run(ceil={"on": 8})
    cases.append(("regressed: on's ceiling below off's", o["outcome"]["verdict"] == "REGRESSED"
                  and o["predictions"]["Q5"]["verdict"] == "MISSED"))
    o = run(knee_att={"on": 0.84})
    cases.append(("regressed: the knee in both draws", o["outcome"]["verdict"] == "REGRESSED"))
    o = run(att={("e4b_on_d2b", 8): 0.80})
    cases.append(("regressed: an on server at 8 req/s", o["outcome"]["verdict"] == "REGRESSED"
                  and not o["no_regression"]["holds"] and o["arms"]["on"]["ceiling"] == 12))
    o = run(knee_att={"e4b_on_d1a": 1.0, "e4b_on_d1b": 0.90})
    cases.append(("a noisy replicate raises the bar", o["knee"]["gain"][1] == 0.04 and o["knee"]["noise"][1] == 0.08
                  and o["knee"]["bar"][1] == 0.08 and o["outcome"]["verdict"] == "FLAT"))
    o = run(knee_att={"e4b_on_d2a": 0.91, "e4b_on_d2b": 0.91})
    cases.append(("one draw clears, the other does not: FLAT", o["knee"]["gain"][2] == 0.008 and o["outcome"]["verdict"] == "FLAT"))
    for tag, fl, name in (("e4b_on_d1a", {"prefill_glue": {"mode": "0", "source": "default-off"}}, "on resolved glue off"),
                          ("e4b_on_d2b", {"prefill_glue_env": "1"}, "on set the knob instead of the default"),
                          ("e4b_off_d1b", {"prefill_lean_dispatch": "auto"}, "off left lean dispatch on")):
        o = run(flips={tag: fl})
        cases.append((f"FLIPS: {name}", o["outcome"]["verdict"] == "VOID" and o["outcome"]["void"] == [tag]
                      and not o["gates"]["flips"]["ok"]))
    o = run(routes_seen={"e4b_off_d2a": {"moe_k19_dispatch": {"chained|lean|gt256": 96}}})
    cases.append(("FLIPS: off took the lean dispatch", o["outcome"]["void"] == ["e4b_off_d2a"]))
    o = run(routes_seen={"e4b_on_d1b": {"prefill_folds": {"norm": 48, "layer": 48}}})
    cases.append(("FLIPS: on missed a fold", o["outcome"]["void"] == ["e4b_on_d1b"]))
    o = run(routes_seen={"e4b_off_d1a": {"prefill_folds": {"norm": 2}}})
    cases.append(("FLIPS: off folded prefill rows", o["outcome"]["void"] == ["e4b_off_d1a"]))
    o = run(slots={"e4b_on_d1a": {"buckets": [1, 2, 4, 8, 16, 32]}})
    cases.append(("SLOTS: no bucket 64", o["outcome"]["void"] == ["e4b_on_d1a"] and not o["gates"]["slots"]["ok"]))
    o = run(routes_seen={"e4b_off_d2b": {"prefill_attn": {"math": 240}}})
    cases.append(("ROUTES: prefill attention not flash", "e4b_off_d2b" in o["gates"]["routes"]["bad"]))
    o = run(replays={"e4b_on_d2a": 5})
    cases.append(("ENGAGED: prefill graph replays short", o["outcome"]["void"] == ["e4b_on_d2a"]))
    o = run(prompt_tokens=511)
    cases.append(("PROMPTS: drift voids every server", len(o["outcome"]["void"]) == 8))
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        os.remove(os.path.join(d, "e4b_on_d1b_k13_2.json"))
        o = reduce(d)
    cases.append(("a missing knee run: UNREAD", o["outcome"]["verdict"] == "VOID"
                  and o["knee"]["server"]["e4b_on_d1b"] is None and o["knee"]["gain"][1] is None))
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, admitted={"e4b_on_d1b": 4 + 64 + 24 + 120 * 17})   # a run the box never drove: still engaged
        os.remove(os.path.join(d, "e4b_on_d1b_k13_2.json"))
        o = reduce(d)
    cases.append(("a missing knee run on an engaged server: UNREAD", o["outcome"]["verdict"] == "UNREAD"))
    o = run(invalid={("e4b_off_d2a", "r13")})
    cases.append(("an invalid ladder run: the ceiling is unread", o["arms"]["off"]["ceiling"] is None
                  and o["outcome"]["verdict"] == "UNREAD"))
    o = run(fwd_ms={"on": 39.0})
    cases.append(("Q1 missed, the outcome stands", o["predictions"]["Q1"]["verdict"] == "MISSED"
                  and o["outcome"]["verdict"] == "FASTER"))
    o = run(dec64_ms={"on": 19.2})
    cases.append(("Q6 missed high", o["predictions"]["Q6"]["verdict"] == "MISSED"))
    o = run(serial={"on": (0.0405, 0.0042)})
    cases.append(("Q2 missed: serial TTFT unchanged", o["predictions"]["Q2"]["verdict"] == "MISSED"))
    h = {"prefill_routes": {"prefill_glue_env": None, "prefill_lean_dispatch_env": None, "prefill_lean_dispatch": "auto",
                            "seen": {"moe": {"nf4_grouped|gt256": 9}}}, "levers": {"prefill_glue": {"mode": "0", "source": "default-off"}}}
    cases.append(("proof: Granite's on server", flips_bad(h, "on", proof=True) == {} and flips_bad(h, "on") != {}))
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2F_REDUCE self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--flips", nargs=2, metavar=("HEALTH", "ARM"))
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--engagement", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.flips:
        path, arm = a.flips
        h = json.load(open(path)) if os.path.exists(path) else None
        bad = flips_bad(h, arm, proof=a.proof) if arm in ARMS else {"arm": arm}
        print("SC2F_FLIPS " + json.dumps({"arm": arm, "proof": a.proof, "ok": not bad, "bad": bad}), flush=True)
        return 1 if bad else 0
    if a.engagement:
        if not a.arms or len(a.arms) != len(a.engagement):
            ap.error("--engagement END.json ... --arms ARM ... (one arm per file)")
        return engagement_cli(a.engagement, a.arms)
    if not a.dir:
        ap.error("--dir DIR | --flips HEALTH ARM | --engagement ... | --self-test")
    out = reduce(a.dir)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("SC2F_OUTCOME " + json.dumps(out["outcome"]))
    for q, v in out["predictions"].items():
        print(f"SC2F_{q} {v['verdict']} " + json.dumps({k: x for k, x in v.items() if k != "verdict"}))
    print("SC2F_KNEE " + json.dumps({k: out["knee"][k] for k in ("gain", "noise", "bar")}))
    print("SC2F_CEILINGS " + json.dumps({arm: out["arms"][arm]["ceiling"] for arm in ARMS}))
    print("SC2F_GATES " + json.dumps({k: v["ok"] for k, v in out["gates"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
