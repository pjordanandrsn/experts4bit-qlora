#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2e_reduce.py -- lane SC2e's registered rule (bench/sc2/SC2e-PREREG.md; #846): e4b serve_paged at 16, 32 and 64
slots, with decode-graph buckets that end at max_seqs (E4B_PAGED_BUCKETS=auto) against the default list, from box L's
run files.

Files in DIR, per draw D in 1..2 and arm A in (s16, s32a, s64c, s64a), from sc2_driver.py ``run`` and the box:
  e4b_<A>_d<D>_warm.json, _burst.json, _serial.json, _r<R>.json   every arm of a draw uses the same seeds
  e4b_s16_d1_serial_repeat.json                                   the determinism control
  health_e4b_<A>_d<D>_start.json / _end.json                      the server's own /health
  trace_e4b_<A>_d<D>.jsonl, steps_e4b_<A>_d<D>.jsonl              E4B_PAGED_TRACE, E4B_PAGED_STEP_TRACE
  vram_e4b_<A>_d<D>.txt                                           nvidia-smi memory.used,total at ready

**Gates, per server** (a failure names the server; an arm is licensable only if its servers and s16's pass):
- ROUTES: SC2c's registered routes and ``seen`` (``sc2c_reduce.routes_bad``) on the start /health.
- SLOTS: the arm's max_seqs = kv_slots, its bucket list and ``buckets_requested``, every bucket "graph", scratch
  slots = the largest bucket, on the start /health.
- ENGAGED: on the end /health, the prefill graph on (T 512, replays = admitted, eager 0); kv_bookkeeping bulk for every
  admitted request, nothing per layer; graph_stats with >= 1 replay of the largest bucket and no eager step anywhere.
- PROMPTS: every request of the server's runs reports 512 prompt tokens.
And over the lane: DETERMINISM (s16 draw 1 serial vs its repeat IDENTICAL, else IDENTITY is UNREAD), IDENTITY (per
draw, each wide arm's serial against s16's, IDENTICAL).

**Rows** are sc2_reduce.row per arm and rate over both draws; the **ceiling** is the largest rate of 1, 2, 4, 8, 12, 16
whose row is VALID with attainment >= 0.95 in both draws (None if a row is UNREAD or INVALID).

**Predictions** (paired per draw):
  P1   decode-only step p50 at the largest bucket: s32a's "32" in [12, 20] ms; s64a's "64" in [20, 34] ms, both draws.
  P1b  s64a's "64" step p50 / s64c's "16x4" step p50 <= 0.85, both draws.
  P2   serial p50 TTFT and TPOT of each wide arm within +-5 % of s16's, both draws.
  P3   s64a's ceiling >= 8.
  P4   s32a's attainment at 8 req/s >= 0.90 in both draws.
  P5   s64a at 12 req/s >= 0.50 in both draws, and at 16 req/s < 0.95 in at least one.
  P6   per wide arm: at 1, 2 and 4 req/s attainment >= s16's - 0.05 per draw, and its ceiling >= s16's.
  P7   VRAM at ready minus s16's (same draw): s32a [1600, 2400], s64c [4900, 5600], s64a [4900, 6200] MiB; and the
       prefill graph engaged on every server (the engagement clause).
  P8   s64a's TPOT p50 at 8 req/s <= 40 ms, both draws.

**Licence**, decoupled from P1, P1b, P3, P4, P5 and P8: arm X is LICENSABLE iff its servers and s16's pass ROUTES,
SLOTS, ENGAGED and PROMPTS, DETERMINISM passes and IDENTITY reads IDENTICAL for X in both draws, P6 holds for X, P7's
engagement clause holds for X, and X's ceiling is strictly above s16's. The verdict is the first LICENSABLE arm in the
order s64a, s64c, s32a: SLOTS_LICENSED(64, auto) / SLOTS_LICENSED(64, default) / SLOTS_LICENSED(32, auto); else
NOT_LICENSED with the reasons. ``buckets_auto`` reads LICENSED iff the verdict is SLOTS_LICENSED(64, auto) and P1b HOLDS.

**Reported, no bar:** per server, sc2e_census.steps / per_rate / fit; VRAM, free_after_mib, bulk_flush_mib, pool_mib;
text agreement per rate between s16 and each wide arm.

  sc2e_reduce.py --dir DIR [--out verdict.json]
  sc2e_reduce.py --engagement S16_END S32A_END S64C_END S64A_END      (the proof's servers, counts not equated)
  sc2e_reduce.py --self-test
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("s16", "s32a", "s64c", "s64a")
WIDE = ("s32a", "s64c", "s64a")
RATES = (1, 2, 4, 8, 12, 16)
DRAWS = (1, 2)
T = 512
CAP_SHARE = 0.95
SLOTS = {"s16": (16, [1, 2, 4, 8, 16], "default"), "s32a": (32, [1, 2, 4, 8, 16, 32], "auto"),
         "s64c": (64, [1, 2, 4, 8, 16], "default"), "s64a": (64, [1, 2, 4, 8, 16, 32, 64], "auto")}
P1_BAND = {"s32a": ("32", 12.0, 20.0), "s64a": ("64", 20.0, 34.0)}
P1B_MAX, P2_TOL, P3_MIN, P4_MIN, P5_R12_MIN, P6_TOL, P8_MAX_S = 0.85, 0.05, 8, 0.90, 0.50, 0.05, 0.040
P7_BAND = {"s32a": (1600, 2400), "s64c": (4900, 5600), "s64a": (4900, 6200)}
MIN_STEPS = 20
LICENCE_ORDER = (("s64a", "SLOTS_LICENSED(64, auto)"), ("s64c", "SLOTS_LICENSED(64, default)"),
                 ("s32a", "SLOTS_LICENSED(32, auto)"))


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


def slots_bad(health, arm) -> dict:
    """The arm's slots and buckets on the start /health, every bucket captured, scratch = the largest (box L's
    ``slots_ok``, the same rule)."""
    m, b, req = SLOTS[arm]
    e = (health or {}).get("engine") or {}
    kv = ((health or {}).get("levers") or {}).get("kv") or {}
    bad = {f"engine.{k}": e.get(k, "<missing>") for k, v in (("max_seqs", m), ("kv_slots", m), ("buckets", b),
                                                               ("buckets_requested", req)) if e.get(k, "<missing>") != v}
    gs = e.get("graph_status")
    if not isinstance(gs, dict) or set(gs) != {str(x) for x in b} or any(v != "graph" for v in gs.values()):
        bad["engine.graph_status"] = gs if gs is not None else "<missing>"
    if kv.get("scratch_slots") != b[-1]:
        bad["levers.kv.scratch_slots"] = kv.get("scratch_slots", "<missing>")
    return bad


def engagement(arm, end, prompts) -> list:
    """Why ``arm``'s end /health does not read as registered (empty: engaged). ``prompts`` None skips equating the
    counts with the requests admitted (the proof)."""
    if end is None:
        return [f"{arm}: no end /health"]
    why = []
    g = end.get("prefill_graph") or {}
    if g.get("status") != "on" or g.get("T") != T or g.get("eager_chunks") != 0 or \
            (prompts is not None and g.get("replays") != prompts):
        why.append(f"{arm}: prefill_graph {g!r} (want on, T {T}, replays {prompts}, eager_chunks 0)")
    k = end.get("kv_bookkeeping")
    if not isinstance(k, dict) or "flush_bulk" not in k:
        why.append(f"{arm}: no built kv_bookkeeping block ({k!r})")
    else:
        n = prompts if prompts is not None else k.get("flush_bulk", 0)
        want = {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0, "ready_bulk": 0,
                "ready_at_flush": n}
        bad = {x: k.get(x, "<missing>") for x, v in want.items() if k.get(x, "<missing>") != v}
        if bad or not n:
            why.append(f"{arm}: kv_bookkeeping {bad or k!r} (want {want})")
    top = str(SLOTS[arm][1][-1])
    gs = (end.get("engine") or {}).get("graph_stats")
    if not isinstance(gs, dict) or (gs.get(top) or {}).get("replays", 0) < 1:
        why.append(f"{arm}: bucket {top} never replayed ({gs!r})")
    elif any(v.get("eager_steps") for v in gs.values()):
        why.append(f"{arm}: eager decode steps {gs!r}")
    return why


def _names(arm, draw):
    out = [f"e4b_{arm}_d{draw}_{x}" for x in ("warm", "burst", "serial")]
    if (arm, draw) == ("s16", 1):
        out.append("e4b_s16_d1_serial_repeat")
    return out + [f"e4b_{arm}_d{draw}_r{r}" for r in RATES]


def _prompts(d, arm, draw):
    """Requests the server admitted: warm, burst, serial (+ the repeat on s16 draw 1), every rate."""
    return sum(len(x["requests"]) for x in (_load(d, n + ".json") for n in _names(arm, draw)) if x)


def _plan(d, arm, draw):
    """The request trace's workloads in the server's order."""
    out = []
    for n in _names(arm, draw):
        x = _load(d, n + ".json")
        if x:
            out.append((n.split(f"_d{draw}_", 1)[1], len(x["requests"])))
    return out


def ceiling(red, rows: dict):
    """sc2_reduce's rule over this lane's ladder."""
    rs = [rows[r] for r in RATES]
    if any(x["status"] in ("UNREAD", "INVALID") for x in rs):
        return None
    ok = [r for r in RATES if rows[r]["status"] == "VALID" and rows[r].get("meets_capacity")]
    return max(ok) if ok else 0


def _agreement(a, b):
    """Share of requests whose streamed text is byte-equal (same plan index), or None."""
    if not a or not b or len(a["requests"]) != len(b["requests"]):
        return None
    eq = [x.get("valid") and y.get("valid") and x.get("text") == y.get("text") for x, y in zip(a["requests"], b["requests"])]
    return round(sum(eq) / len(eq), 4) if eq else None


def census(d) -> dict:
    cen = _sib("sc2e_census")
    out = {}
    for arm in ARMS:
        for k in DRAWS:
            tag = f"{arm}_d{k}"
            c = {}
            steps = _jsonl(d, f"steps_e4b_{tag}.jsonl")
            c["steps"] = cen.steps(steps, arm) if steps else {"why": "no step trace"}
            trace = _jsonl(d, f"trace_e4b_{tag}.jsonl")
            plan = _plan(d, arm, k)
            for name, fn in (("per_rate", lambda: cen.per_rate(trace, plan)), ("fit", lambda: cen.fit(trace, plan, arm))):
                try:
                    c[name] = fn() if trace else {"why": "no request trace"}
                except SystemExit as e:
                    c[name] = {"why": str(e)}
            v = os.path.join(d, f"vram_e4b_{tag}.txt")
            txt = open(v).read().strip() if os.path.exists(v) else ""
            c["vram_used_mib"] = int(txt.split(",")[0]) if txt else None
            h = _load(d, f"health_e4b_{tag}_start.json") or {}
            g = h.get("prefill_graph") or {}
            c["at_ready"] = {**{x: g.get(x) for x in ("status", "pool_mib", "free_after_mib", "bulk_flush_mib")},
                             "kv_pool_mib": ((h.get("levers") or {}).get("kv") or {}).get("pool_mib")}
            c["graph_stats_end"] = ((_load(d, f"health_e4b_{tag}_end.json") or {}).get("engine") or {}).get("graph_stats")
            if arm != "s16":
                c["text_agreement_vs_s16"] = {
                    w: _agreement(_load(d, f"e4b_s16_d{k}_{w}.json"), _load(d, f"e4b_{arm}_d{k}_{w}.json"))
                    for w in ["serial"] + [f"r{r}" for r in RATES]}
            out[tag] = c
    return out


def reduce(d: str) -> dict:
    red, ident, c_red = _sib("sc2_reduce"), _sib("sc2_identity"), _sib("sc2c_reduce")
    out = {"gates": {}, "arms": {}, "predictions": {}}
    srv = [(a, k) for a in ARMS for k in DRAWS]
    routes = {f"{a}_d{k}": c_red.routes_bad(_load(d, f"health_e4b_{a}_d{k}_start.json")) for a, k in srv}
    slots = {f"{a}_d{k}": slots_bad(_load(d, f"health_e4b_{a}_d{k}_start.json"), a) for a, k in srv}
    eng = {f"{a}_d{k}": engagement(a, _load(d, f"health_e4b_{a}_d{k}_end.json"), _prompts(d, a, k)) for a, k in srv}
    prm = {}
    for a, k in srv:
        off = {}
        for n in _names(a, k):
            x = _load(d, n + ".json")
            lens = {r.get("prompt_tokens") for r in x["requests"]} if x else {T}
            if lens != {T}:
                off[n] = sorted(lens, key=str)
        prm[f"{a}_d{k}"] = off
    out["gates"]["routes"] = {"ok": not any(routes.values()), "bad": {k: v for k, v in routes.items() if v}}
    out["gates"]["slots"] = {"ok": not any(slots.values()), "bad": {k: v for k, v in slots.items() if v}}
    out["gates"]["engaged"] = {"ok": not any(eng.values()), "why": {k: v for k, v in eng.items() if v}}
    out["gates"]["prompts"] = {"ok": not any(prm.values()), "bad": {k: v for k, v in prm.items() if v}}
    server_ok = {s: not (routes[s] or slots[s] or eng[s] or prm[s]) for s in routes}
    a, b = _load(d, "e4b_s16_d1_serial.json"), _load(d, "e4b_s16_d1_serial_repeat.json")
    det = ident.compare(a, b) if a and b else {"verdict": "REFUSED", "why": "a determinism file is missing"}
    out["gates"]["determinism"] = det
    idt = {}
    for arm in WIDE:
        per = {}
        for k in DRAWS:
            x, y = _load(d, f"e4b_s16_d{k}_serial.json"), _load(d, f"e4b_{arm}_d{k}_serial.json")
            per[k] = ident.compare(x, y) if x and y else {"verdict": "REFUSED", "why": "a serial file is missing"}
        v = ("UNREAD" if det["verdict"] != "IDENTICAL" else
             "IDENTICAL" if all(p["verdict"] == "IDENTICAL" for p in per.values()) else
             "DIFFERS" if any(p["verdict"] == "DIFFERS" for p in per.values()) else "UNREAD")
        idt[arm] = {"verdict": v, "draws": per}
    out["gates"]["identity"] = idt
    for arm in ARMS:
        rows = {"serial": red.row([_load(d, f"e4b_{arm}_d{k}_serial.json") for k in DRAWS], "serial")}
        for r in RATES:
            rows[r] = red.row([_load(d, f"e4b_{arm}_d{k}_r{r}.json") for k in DRAWS], r)
        out["arms"][arm] = {"rows": {str(k): v for k, v in rows.items()}, "ceiling": ceiling(red, rows)}
    out["census"] = cen = census(d)

    def summ(arm, k, w):
        x = _load(d, f"e4b_{arm}_d{k}_{w}.json")
        return x["summary"] if x else None

    def step(arm, k, key):
        s = (cen[f"{arm}_d{k}"]["steps"].get("decode_steps") or {}).get(key) or {}
        return s.get("step_ms_p50") if s.get("n", 0) >= MIN_STEPS else None

    def att(arm, k, r):
        s = summ(arm, k, f"r{r}")
        return s["attainment"] if s and not s["invalid"] else None
    v, ceil = out["predictions"], {arm: out["arms"][arm]["ceiling"] for arm in ARMS}
    p1 = {}
    for arm, (key, lo, hi) in P1_BAND.items():
        ms = [step(arm, k, key) for k in DRAWS]
        p1[arm] = {"verdict": "UNREAD" if None in ms else ("HOLDS" if all(lo <= x <= hi for x in ms) else "REFUTED"),
                   "step_ms_p50": ms, "band": [lo, hi]}
    v["P1"] = {"verdict": ("UNREAD" if any(x["verdict"] == "UNREAD" for x in p1.values()) else
                           "HOLDS" if all(x["verdict"] == "HOLDS" for x in p1.values()) else "REFUTED"), "arms": p1}
    rat = []
    for k in DRAWS:
        w, c = step("s64a", k, "64"), step("s64c", k, "16x4")
        rat.append(round(w / c, 4) if w is not None and c else None)
    v["P1b"] = ({"verdict": "UNREAD", "ratio": rat} if None in rat else
                {"verdict": "HOLDS" if max(rat) <= P1B_MAX else "REFUTED", "ratio": rat})
    p2 = {}
    for arm in WIDE:
        ser = [(summ("s16", k, "serial"), summ(arm, k, "serial")) for k in DRAWS]
        if any(x is None or y is None or x["invalid"] or y["invalid"] for x, y in ser):
            p2[arm] = {"verdict": "UNREAD"}
            continue
        r = [[round(y[m] / x[m], 4) for m in ("ttft_p50_s", "tpot_p50_s")] for x, y in ser]
        p2[arm] = {"verdict": "HOLDS" if all(abs(q - 1) <= P2_TOL for pair in r for q in pair) else "REFUTED",
                   "ttft_tpot_over_s16": r}
    v["P2"] = {"verdict": ("UNREAD" if any(x["verdict"] == "UNREAD" for x in p2.values()) else
                           "HOLDS" if all(x["verdict"] == "HOLDS" for x in p2.values()) else "REFUTED"), "arms": p2}
    v["P3"] = {"verdict": "UNREAD" if ceil["s64a"] is None else ("HOLDS" if ceil["s64a"] >= P3_MIN else "REFUTED"),
               "ceiling_s64a": ceil["s64a"]}
    a8 = [att("s32a", k, 8) for k in DRAWS]
    v["P4"] = {"verdict": "UNREAD" if None in a8 else ("HOLDS" if min(a8) >= P4_MIN else "REFUTED"),
               "attainment_r8": a8, "ceiling_s32a": ceil["s32a"]}
    a12, a16 = [att("s64a", k, 12) for k in DRAWS], [att("s64a", k, 16) for k in DRAWS]
    v["P5"] = {"verdict": "UNREAD" if None in a12 + a16 else
               ("HOLDS" if min(a12) >= P5_R12_MIN and min(a16) < CAP_SHARE else "REFUTED"),
               "attainment_r12": a12, "attainment_r16": a16}
    p6 = {}
    for arm in WIDE:
        pairs, miss = {}, False
        for r in (1, 2, 4):
            for k in DRAWS:
                x, y = att("s16", k, r), att(arm, k, r)
                if x is None or y is None:
                    miss = True
                    continue
                pairs[f"r{r}_d{k}"] = [x, y]
        reg = {q: p for q, p in pairs.items() if p[1] < p[0] - P6_TOL}
        if miss or ceil["s16"] is None or ceil[arm] is None:
            p6[arm] = {"verdict": "UNREAD", "pairs": pairs}
        else:
            p6[arm] = {"verdict": "HOLDS" if not reg and ceil[arm] >= ceil["s16"] else "REFUTED", "pairs": pairs,
                       "regressions": reg, "ceilings": {"s16": ceil["s16"], arm: ceil[arm]}}
    v["P6"] = {"verdict": ("UNREAD" if any(x["verdict"] == "UNREAD" for x in p6.values()) else
                           "HOLDS" if all(x["verdict"] == "HOLDS" for x in p6.values()) else "REFUTED"), "arms": p6}
    p7 = {}
    for arm in WIDE:
        lo, hi = P7_BAND[arm]
        dl = []
        for k in DRAWS:
            x, y = cen[f"s16_d{k}"]["vram_used_mib"], cen[f"{arm}_d{k}"]["vram_used_mib"]
            dl.append(y - x if x is not None and y is not None else None)
        engaged = all(cen[f"{arm}_d{k}"]["at_ready"]["status"] == "on" for k in DRAWS)
        p7[arm] = {"verdict": "UNREAD" if None in dl else ("HOLDS" if engaged and all(lo <= x <= hi for x in dl) else "REFUTED"),
                   "delta_mib": dl, "band": [lo, hi], "prefill_graph_engaged": engaged}
    v["P7"] = {"verdict": ("UNREAD" if any(x["verdict"] == "UNREAD" for x in p7.values()) else
                           "HOLDS" if all(x["verdict"] == "HOLDS" for x in p7.values()) else "REFUTED"), "arms": p7}
    t8 = [(summ("s64a", k, "r8") or {}).get("tpot_p50_s") for k in DRAWS]
    v["P8"] = {"verdict": "UNREAD" if None in t8 else ("HOLDS" if max(t8) <= P8_MAX_S else "REFUTED"), "tpot_p50_s": t8}
    lic = {}
    for arm, _ in LICENCE_ORDER:
        why = [f"server {s} failed a gate" for s in (f"{x}_d{k}" for x in ("s16", arm) for k in DRAWS) if not server_ok[s]]
        if det["verdict"] != "IDENTICAL":
            why.append(f"determinism {det['verdict']}")
        if idt[arm]["verdict"] != "IDENTICAL":
            why.append(f"identity {idt[arm]['verdict']}")
        if p6[arm]["verdict"] != "HOLDS":
            why.append(f"P6 {p6[arm]['verdict']}")
        if not p7[arm]["prefill_graph_engaged"]:
            why.append("the prefill graph did not engage")
        if ceil[arm] is None or ceil["s16"] is None or ceil[arm] <= ceil["s16"]:
            why.append(f"ceiling {ceil[arm]} not above s16's {ceil['s16']}")
        lic[arm] = why
    verdict = next((name for arm, name in LICENCE_ORDER if not lic[arm]), "NOT_LICENSED")
    out["licence"] = {"verdict": verdict, "failed": {arm: w for arm, w in lic.items() if w},
                      "buckets_auto": ("LICENSED" if verdict == "SLOTS_LICENSED(64, auto)" and v["P1b"]["verdict"] == "HOLDS"
                                       else "NOT_LICENSED")}
    out["void"] = {s: True for s, ok in server_ok.items() if not ok}
    return out


# ------------------------------------------------------------------ self-test --

def _fixture(d, *, ceil=None, att8=None, att12=None, att16=None, serial_text=None, repeat_text="t", slots=None,
             graph_stats=None, replays=None, vram=None, steps_ms=None, tpot8=None, prompt_tokens=T, missing=(),
             pf_status="on", burst_n=64):
    """Box L's run files for every server, from a profile per arm. Defaults: today's s16 (ceiling 4) and wide arms that
    reach 8, every gate passing."""
    ceil = {"s16": 4, "s32a": 8, "s64c": 8, "s64a": 8, **(ceil or {})}
    vram = {"s16": 23700, "s32a": 25700, "s64c": 29000, "s64a": 29200, **(vram or {})}
    steps_ms = {"s16": {"16": 9.3}, "s32a": {"32": 15.5}, "s64c": {"16x4": 38.0}, "s64a": {"64": 27.0}, **(steps_ms or {})}
    tpot8 = {"s64a": 0.026, **(tpot8 or {})}

    def summ(ttft, tpot, a, rate):
        return {"valid": 120, "invalid": 0, "errors": [], "ttft_p50_s": ttft, "ttft_p99_s": ttft * 2, "tpot_p50_s": tpot,
                "tpot_p99_s": tpot * 2, "e2el_p50_s": 1.0, "output_tok_s": 100.0, "attainment": a,
                "goodput_rps": a * rate, "achieved_rate": rate}

    def dump(name, obj):
        json.dump(obj, open(os.path.join(d, name), "w"))
    for arm in ARMS:
        if arm in missing:
            continue
        m, b, req = SLOTS[arm]
        for k in DRAWS:
            tag = f"e4b_{arm}_d{k}"
            reqs = lambda n, txt="x": [{"valid": True, "text": f"{txt}{i}", "prompt_tokens": prompt_tokens}  # noqa: E731
                                       for i in range(n)]
            dump(f"{tag}_warm.json", {"requests": reqs(4)})
            dump(f"{tag}_burst.json", {"requests": reqs(burst_n)})
            st = (serial_text or {}).get(arm, "t")
            dump(f"{tag}_serial.json", {"mode": "serial", "prompts_sha256": "s", "plan": [[0.0, i, 64] for i in range(4)],
                                        "requests": reqs(4, st), "summary": summ(0.040, 0.0042, 0.0, 0.0)})
            if (arm, k) == ("s16", 1):
                dump(f"{tag}_serial_repeat.json", {"mode": "serial", "prompts_sha256": "s",
                                                   "plan": [[0.0, i, 64] for i in range(4)], "requests": reqs(4, repeat_text)})
            for r in RATES:
                a = 1.0 if r <= ceil[arm] else 0.3
                if r == 8 and att8 and arm in att8:
                    a = att8[arm][k - 1]
                if r == 12 and att12 and arm in att12:
                    a = att12[arm][k - 1]
                if r == 16 and att16 and arm in att16:
                    a = att16[arm][k - 1]
                tp = tpot8.get(arm, 0.02) if r == 8 else 0.02
                dump(f"{tag}_r{r}.json", {"requests": reqs(120), "summary": summ(0.3, tp, a, r)})
            n = 4 + burst_n + 4 + (4 if (arm, k) == ("s16", 1) else 0) + 120 * len(RATES)
            gs = {str(x): {"replays": 5, "eager_steps": 0, "rows": 5, "pad_rows": 0} for x in b}
            if graph_stats and arm in graph_stats:
                gs = graph_stats[arm]
            sb = (slots or {}).get(arm, {})
            dump(f"health_{tag}_start.json", {
                "prefill_routes": dict(_sib("sc2c_reduce").ROUTES, seen={"moe": {"int4_k19|gt256": 240},
                                                                          "prefill_attn": {"flash": 240}}),
                "engine": {"chunk_tokens": T, "max_prefill_tokens_per_step": T, "max_seqs": sb.get("max_seqs", m),
                           "kv_slots": m, "buckets": sb.get("buckets", b), "buckets_requested": req,
                           "graph_status": sb.get("graph_status", {str(x): "graph" for x in b})},
                "levers": {"kv": {"scratch_slots": b[-1], "pool_mib": 1669.7 * m / 16}},
                "prefill_graph": {"status": pf_status, "pool_mib": 428, "free_after_mib": 8000, "bulk_flush_mib": 216}})
            dump(f"health_{tag}_end.json", {
                "prefill_graph": {"status": pf_status, "T": T, "replays": (replays or {}).get(arm, n), "eager_chunks": 0},
                "kv_bookkeeping": {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0,
                                   "ready_bulk": 0, "ready_at_flush": n},
                "engine": {"graph_stats": gs}})
            open(os.path.join(d, f"vram_{tag}.txt"), "w").write(f"{vram[arm] + (k - 1) * 10}, 32607\n")
            key, ms = next(iter(steps_ms[arm].items()))
            p = key.partition("x")
            rows = [{"step": i, "t": i * 0.05, "step_ms": ms, "decode_rows": 50, "bucket": int(p[0]),
                     "dec_pieces": int(p[2] or 1), "seg": {"dec_sync": ms - 1}, "gpu": {"dec_prep": 0.5, "dec_issue": ms - 0.5}}
                    for i in range(30)]
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
    cases.append(("licensed 64 auto", o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)"
                  and o["licence"]["buckets_auto"] == "LICENSED" and not o["void"]
                  and o["arms"]["s16"]["ceiling"] == 4 and o["arms"]["s64a"]["ceiling"] == 8))
    cases.append(("predictions hold", all(o["predictions"][p]["verdict"] == "HOLDS" for p in ("P1", "P1b", "P2", "P3", "P6", "P8"))
                  and o["predictions"]["P1b"]["ratio"] == [0.7105, 0.7105]))
    o = run(ceil={"s64a": 4})
    cases.append(("64 auto not above s16 -> 64 default", o["licence"]["verdict"] == "SLOTS_LICENSED(64, default)"
                  and o["licence"]["buckets_auto"] == "NOT_LICENSED" and o["predictions"]["P3"]["verdict"] == "REFUTED"))
    o = run(ceil={"s64a": 4, "s64c": 4})
    cases.append(("only 32 lifts", o["licence"]["verdict"] == "SLOTS_LICENSED(32, auto)"))
    o = run(ceil={"s64a": 4, "s64c": 4, "s32a": 4})
    cases.append(("nothing lifts", o["licence"]["verdict"] == "NOT_LICENSED" and len(o["licence"]["failed"]) == 3))
    o = run(serial_text={"s64a": "u"})
    cases.append(("identity differs on s64a", o["gates"]["identity"]["s64a"]["verdict"] == "DIFFERS"
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, default)"))
    o = run(repeat_text="z")
    cases.append(("nondeterministic", o["gates"]["determinism"]["verdict"] == "DIFFERS"
                  and o["licence"]["verdict"] == "NOT_LICENSED"
                  and all(v["verdict"] == "UNREAD" for v in o["gates"]["identity"].values())))
    o = run(slots={"s32a": {"buckets": [1, 2, 4, 8, 16]}})
    cases.append(("wrong bucket list -> void", not o["gates"]["slots"]["ok"] and o["void"].get("s32a_d1")
                  and "s32a" in o["licence"]["failed"]))
    o = run(slots={"s64a": {"graph_status": {**{str(x): "graph" for x in (1, 2, 4, 8, 16, 32)}, "64": "eager: OOM"}}})
    cases.append(("an eager bucket -> void", not o["gates"]["slots"]["ok"]
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, default)"))
    o = run(graph_stats={"s64a": {"64": {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}}})
    cases.append(("top never replayed -> void", not o["gates"]["engaged"]["ok"] and o["void"].get("s64a_d2")))
    o = run(graph_stats={"s32a": {"32": {"replays": 3, "eager_steps": 1, "rows": 3, "pad_rows": 0}}})
    cases.append(("an eager step -> void", not o["gates"]["engaged"]["ok"] and "s32a" in o["licence"]["failed"]))
    o = run(pf_status="refused")
    cases.append(("prefill graph refused -> void everywhere", not o["gates"]["engaged"]["ok"]
                  and o["licence"]["verdict"] == "NOT_LICENSED"))
    o = run(prompt_tokens=511)
    cases.append(("prompt drift -> void", not o["gates"]["prompts"]["ok"] and o["licence"]["verdict"] == "NOT_LICENSED"))
    o = run(replays={"s64c": 3})
    cases.append(("burst uncounted -> engaged mismatch", not o["gates"]["engaged"]["ok"]
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)"))
    o = run(steps_ms={"s64a": {"64": 40.0}})
    cases.append(("P1 and P1b missed", o["predictions"]["P1"]["verdict"] == "REFUTED"
                  and o["predictions"]["P1b"]["verdict"] == "REFUTED" and o["licence"]["buckets_auto"] == "NOT_LICENSED"
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)"))
    o = run(vram={"s64a": 31000})
    cases.append(("P7 band missed, licence unmoved", o["predictions"]["P7"]["arms"]["s64a"]["verdict"] == "REFUTED"
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)"))
    o = run(missing=("s64a",))
    cases.append(("an arm that never ran", o["arms"]["s64a"]["ceiling"] is None and o["predictions"]["P3"]["verdict"] == "UNREAD"
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, default)"))
    o = run(ceil={"s64a": 16})
    cases.append(("P5 refuted at 16", o["predictions"]["P5"]["verdict"] == "REFUTED" and o["arms"]["s64a"]["ceiling"] == 16))
    o = run(att8={"s32a": (0.92, 0.85)}, ceil={"s32a": 4})
    cases.append(("P4 refuted", o["predictions"]["P4"]["verdict"] == "REFUTED" and o["predictions"]["P4"]["ceiling_s32a"] == 4))
    o = run(ceil={"s32a": 2})
    cases.append(("P6 regression", o["predictions"]["P6"]["arms"]["s32a"]["verdict"] == "REFUTED"
                  and "s32a" in o["licence"]["failed"] and o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)"))
    o = run(tpot8={"s64a": 0.045})
    cases.append(("P8 refuted, licence unmoved", o["predictions"]["P8"]["verdict"] == "REFUTED"
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)"))
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        os.remove(os.path.join(d, "steps_e4b_s64c_d1.jsonl"))
        o = reduce(d)
    cases.append(("no step trace: P1b unread, licence readable", o["predictions"]["P1b"]["verdict"] == "UNREAD"
                  and o["licence"]["verdict"] == "SLOTS_LICENSED(64, auto)" and o["licence"]["buckets_auto"] == "NOT_LICENSED"))
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        ends = [_load(d, f"health_e4b_{a}_d1_end.json") for a in ARMS]
    cases.append(("--engagement on the proof", not any(engagement(a, e, None) for a, e in zip(ARMS, ends))
                  and engagement("s64a", ends[0], None) != []))
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2E_REDUCE self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--engagement", nargs=4, metavar=("S16", "S32A", "S64C", "S64A"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.engagement:
        why = [w for arm, p in zip(ARMS, a.engagement) for w in engagement(arm, json.load(open(p)) if os.path.exists(p) else None, None)]
        print("SC2E_ENGAGEMENT " + json.dumps({"ok": not why, "why": why}), flush=True)
        return 1 if why else 0
    if not a.dir:
        ap.error("--dir DIR | --engagement S16 S32A S64C S64A | --self-test")
    o = reduce(a.dir)
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            json.dump(o, f, indent=1)
            f.write("\n")
    print("SC2E_GATES " + json.dumps({k: (v.get("ok") if "ok" in v else {x: y["verdict"] for x, y in v.items()}
                                          if k == "identity" else v.get("verdict")) for k, v in o["gates"].items()}))
    for arm in ARMS:
        print(f"SC2E_ARM {arm} ceiling={o['arms'][arm]['ceiling']} "
              f"rows={json.dumps({k: x['status'] for k, x in o['arms'][arm]['rows'].items()})}")
    print("SC2E_PREDICTIONS " + json.dumps({k: x["verdict"] for k, x in o["predictions"].items()}))
    print("SC2E_LICENCE " + json.dumps(o["licence"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
