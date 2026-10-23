#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2i_reduce.py -- lane SC2i's registered rule (bench/sc2/SC2i-PREREG.md; #846): e4b serve_paged at 64 slots on the
release's defaults (arm ``base``) against e4b#2017's prefill interleave knobs (arms ``pe2``, ``pe4``: ``E4B_PAGED_
PREFILL_EVERY`` 2 and 4; ``lc256``, ``lc128``: ``E4B_PAGED_LOADED_CHUNK_TOKENS`` 256 and 128), from box P's run files.

Servers, each its own cold start, tagged ``e4b_<slot>_d<draw>``: draw 1 runs base_a, pe2, pe4, lc256, lc128, base_b;
draw 2 runs base_a, lc128, lc256, pe4, pe2, base_b. Files in DIR, per server S:
  S_warm.json, S_burst.json, S_serial.json   4 warm (seed 999), the burst (64 at once, seed 998), serial 24 (seed = draw)
  S_r<R>.json                                the ladder: R in 4, 8, 12; 120 requests; seed draw x 100 + R
  health_S_start.json / _end.json, stats_S_end.json, trace_S.jsonl, steps_S.jsonl
Every server of a draw runs the same seeds.

**Gates, per server** (a failure names the server and VOIDs the reading): ROUTES (``sc2c_reduce.routes_bad``), SLOTS at
64 slots (``sc2e_reduce.slots_bad``, arm s64a), DEFAULTS (the start /health reads SC2f's ``on``, ``sc2f_reduce.flips_bad``,
and the lookahead off), KNOBS (``engine.prefill_every`` and ``engine.loaded_chunk_tokens`` are the arm's), ENGAGED
(``sc2e_reduce.engagement``, arm s64a, every admitted request counted), PROMPTS (every request reports 512 tokens).

**Per draw and knob arm:** ``ref`` = the mean of the draw's two ``base`` servers; ``noise`` = |base_a - base_b| / ``ref``
on TPOT p50 at 12 req/s; ``bar`` = max(0.05, ``noise``); ``ratio`` = the arm's TPOT p50 at 12 req/s / ``ref``'s.

**Outcome, per knob arm:** VOID if a server fails a gate; UNREAD if a needed number is missing;
- REGRESSED if, in both draws, ``ratio`` >= 1 + ``bar``, or the arm's attainment at 4, 8 or 12 req/s is below the lower
  ``base`` server's minus 0.05;
- FASTER if, in both draws: ``ratio`` <= 1 - ``bar``; attainment at each of 4, 8 and 12 req/s >= the lower ``base``
  server's minus 0.02; and TTFT p99 at 12 req/s <= max(1.0 s, the higher ``base`` server's);
- FLAT otherwise.
The reducer licenses nothing: a FASTER arm is a candidate for a quality read, never a default.

**Predictions** (per draw; no outcome reads them): P1 ``pe2``'s ratio in [0.75, 0.97]; P2 ``lc128``'s TTFT p50 at 12 req/s
/ ``base``'s >= 1.00 (serial has no live decode, so the cap never applies there); P3 every knob arm attains >= 0.95 at 4 req/s; P4 ``lc256`` and ``lc128`` count short chunks > 0 and
``base`` = 0.

**Reported, no bar:** TPOT p99 and TTFT p50 at 8 and 12 req/s against ``ref``; ``prefill_deferred``; the drift
``base_b - base_a``; text agreement with ``base_a`` at 12 req/s (where the knobs act).

  sc2i_reduce.py --dir DIR [--out verdict.json]
  sc2i_reduce.py --knobs HEALTH.json ARM [--proof]     (box P's start check; exit 1 on a departure)
  sc2i_reduce.py --engagement END.json ... --arms ARM ...
  sc2i_reduce.py --self-test
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KNOB_ARMS = ("pe2", "pe4", "lc256", "lc128")
ARMS = ("base",) + KNOB_ARMS
KNOBS = {"base": (1, 0), "pe2": (2, 0), "pe4": (4, 0), "lc256": (1, 256), "lc128": (1, 128)}
DRAWS = (1, 2)
ORDER = {1: ("base_a", "pe2", "pe4", "lc256", "lc128", "base_b"), 2: ("base_a", "lc128", "lc256", "pe4", "pe2", "base_b")}
LADDER = (4, 8, 12)
SLOT_ARM = "s64a"
T = 512
N_RATE, N_SERIAL = 120, 24
BAR_MIN = 0.05
ATT_FASTER_TOL, ATT_REGRESS_TOL = 0.02, 0.05
TTFT_P99_CAP = 1.0
P1_BAND = (0.75, 0.97)


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def servers():
    """(tag, arm, draw, slot) in run order."""
    return [(f"e4b_{s}_d{k}", s.rsplit("_", 1)[0] if s.startswith("base") else s, k, s) for k in DRAWS for s in ORDER[k]]


def _names(tag):
    return [f"{tag}_{x}" for x in ("warm", "burst", "serial")] + [f"{tag}_r{r}" for r in LADDER]


def _prompts(d, tag):
    return sum(len(x["requests"]) for x in (_load(d, n + ".json") for n in _names(tag)) if x)


def _summ(d, tag, w):
    x = _load(d, f"{tag}_{w}.json")
    return x["summary"] if x else None


def _get(d, tag, w, key):
    s = _summ(d, tag, w)
    return None if s is None or s.get("invalid") else s.get(key)


# ------------------------------------------------------------------ gates --

def knobs_bad(health, arm, proof=False) -> dict:
    """Why a server's start /health does not read as its arm: the two knobs, the lookahead off, SC2f's on defaults."""
    bad = {}
    eng = (health or {}).get("engine") or {}
    pe, lc = KNOBS[arm]
    if eng.get("prefill_every") != pe:
        bad["engine.prefill_every"] = eng.get("prefill_every", "<missing>")
    if eng.get("loaded_chunk_tokens") != lc:
        bad["engine.loaded_chunk_tokens"] = eng.get("loaded_chunk_tokens", "<missing>")
    if eng.get("decode_lookahead") is not False:
        bad["engine.decode_lookahead"] = eng.get("decode_lookahead", "<missing>")
    bad.update(_sib("sc2f_reduce").flips_bad(health, "on", proof=proof))
    return bad


LC_EAGER_OK = {"short_chunk", "later_chunk"}


def engagement_bad(e_red, health_end, n, arm) -> list:
    """SC2e's ENGAGED check (``sc2e_reduce.engagement``, arm s64a). On ``base``, ``pe2`` and ``pe4`` it applies unchanged:
    every admitted request's first chunk replays the prefill graph and no chunk runs eagerly. On ``lc256`` and ``lc128``
    a first chunk capped below the graph's T runs eagerly by design (``short_chunk``), and a 512-token prompt then has
    later chunks (``later_chunk``). There: replays + short_chunk must equal the admitted requests, every eager reason
    must be one of those two, and the rest of SC2e's check runs on the record with the eager chunks folded back."""
    if arm not in ("lc256", "lc128") or not health_end:
        return e_red.engagement(SLOT_ARM, health_end, n)
    g = dict(health_end.get("prefill_graph") or {})
    reasons = g.get("eager_reasons") or {}
    why = []
    extra = sorted(set(reasons) - LC_EAGER_OK)
    if extra:
        why.append(f"{arm}: eager reasons {extra} beyond {sorted(LC_EAGER_OK)}")
    first = (g.get("replays") or 0) + int(reasons.get("short_chunk", 0))
    if n is not None and first != n:
        why.append(f"{arm}: replays {g.get('replays')} + short_chunk {reasons.get('short_chunk', 0)} = {first}, want {n}")
    folded = dict(health_end, prefill_graph=dict(g, replays=first, eager_chunks=0, eager_reasons={}))
    return why + e_red.engagement(SLOT_ARM, folded, n)


def _short_chunks(d, tag):
    g = (_load(d, f"health_{tag}_end.json") or {}).get("prefill_graph") or {}
    if "eager_reasons" not in g and g.get("status") == "on":
        return 0
    r = g.get("eager_reasons")
    return None if r is None else int(r.get("short_chunk", 0))


def _deferred(d, tag):
    s = ((_load(d, f"stats_{tag}_end.json") or {}).get("scheduler") or {})
    return s.get("prefill_deferred", 0) if s else None


def _band(xs, lo, hi):
    if any(x is None for x in xs):
        return "UNREAD"
    return "HOLDS" if all(lo <= x <= hi for x in xs) else "MISSED"


def _agreement(a, b):
    if not a or not b or len(a["requests"]) != len(b["requests"]):
        return None
    eq = [x.get("valid") and y.get("valid") and x.get("text") == y.get("text") for x, y in zip(a["requests"], b["requests"])]
    return round(sum(eq) / len(eq), 4) if eq else None


# ------------------------------------------------------------------ the reading --

def reduce(d: str) -> dict:
    c_red, e_red = _sib("sc2c_reduce"), _sib("sc2e_reduce")
    srv = servers()
    out = {"gates": {}, "draws": {}, "arms": {}, "predictions": {}, "reported": {}}
    start = {t: _load(d, f"health_{t}_start.json") for t, *_ in srv}
    routes = {t: c_red.routes_bad(start[t]) for t, *_ in srv}
    slots = {t: e_red.slots_bad(start[t], SLOT_ARM) for t, *_ in srv}
    knobs = {t: knobs_bad(start[t], a) for t, a, _, _ in srv}
    eng = {t: engagement_bad(e_red, _load(d, f"health_{t}_end.json"), _prompts(d, t), a) for t, a, _, _ in srv}
    prm = {}
    for t, *_ in srv:
        off = {}
        for n in _names(t):
            x = _load(d, n + ".json")
            lens = {r.get("prompt_tokens") for r in x["requests"]} if x else {T}
            if lens != {T}:
                off[n] = sorted(lens, key=str)
        prm[t] = off
    for name, g in (("routes", routes), ("slots", slots), ("knobs", knobs), ("engaged", eng), ("prompts", prm)):
        out["gates"][name] = {"ok": not any(g.values()), "bad": {k: v for k, v in g.items() if v}}
    void = sorted(t for t, *_ in srv if routes[t] or slots[t] or knobs[t] or eng[t] or prm[t])

    def tag(arm, k, slot=None):
        return f"e4b_{slot or arm}_d{k}"

    per = {}
    for k in DRAWS:
        ba, bb = tag("base", k, "base_a"), tag("base", k, "base_b")
        b12 = [_get(d, x, "r12", "tpot_p50_s") for x in (ba, bb)]
        ref = None if None in b12 else statistics.fmean(b12)
        noise = None if ref is None else abs(b12[0] - b12[1]) / ref
        bar = None if noise is None else max(BAR_MIN, noise)
        batt = {r: [_get(d, x, f"r{r}", "attainment") for x in (ba, bb)] for r in LADDER}
        bt99 = [_get(d, x, "r12", "ttft_p99_s") for x in (ba, bb)]
        out["draws"][f"d{k}"] = {"ref_tpot_p50_r12": ref, "noise": None if noise is None else round(noise, 4),
                                 "bar": None if bar is None else round(bar, 4),
                                 "drift_tpot_p50_r12": None if None in b12 else round(b12[1] - b12[0], 6)}
        for arm in KNOB_ARMS:
            t = tag(arm, k)
            a12 = _get(d, t, "r12", "tpot_p50_s")
            ratio = None if a12 is None or not ref else a12 / ref
            att = {r: _get(d, t, f"r{r}", "attainment") for r in LADDER}
            t99 = _get(d, t, "r12", "ttft_p99_s")
            missing = (ratio is None or bar is None or None in att.values() or t99 is None
                       or any(None in v for v in batt.values()) or None in bt99)
            if missing:
                per[(arm, k)] = {"status": "UNREAD"}
                continue
            lo = {r: min(batt[r]) for r in LADDER}
            per[(arm, k)] = {
                "status": "READ", "ratio": round(ratio, 4),
                "faster": (ratio <= 1 - bar and all(att[r] >= lo[r] - ATT_FASTER_TOL for r in LADDER)
                           and t99 <= max(TTFT_P99_CAP, max(bt99))),
                "regressed": ratio >= 1 + bar or any(att[r] < lo[r] - ATT_REGRESS_TOL for r in LADDER),
                "attainment": att, "ttft_p99_r12": t99}
    for arm in KNOB_ARMS:
        draws = [per[(arm, k)] for k in DRAWS]
        if void:
            verdict = "VOID"
        elif any(x["status"] != "READ" for x in draws):
            verdict = "UNREAD"
        elif all(x["regressed"] for x in draws):
            verdict = "REGRESSED"
        elif all(x["faster"] for x in draws):
            verdict = "FASTER"
        else:
            verdict = "FLAT"
        out["arms"][arm] = {"verdict": verdict, "draws": {f"d{k}": per[(arm, k)] for k in DRAWS}}

    v = out["predictions"]
    v["P1"] = {"verdict": _band([per[("pe2", k)].get("ratio") for k in DRAWS], *P1_BAND),
               "ratio": [per[("pe2", k)].get("ratio") for k in DRAWS]}
    p2 = []
    for k in DRAWS:
        a = _get(d, tag("lc128", k), "r12", "ttft_p50_s")
        b = [_get(d, tag("base", k, s), "r12", "ttft_p50_s") for s in ("base_a", "base_b")]
        p2.append(None if a is None or None in b else round(a / statistics.fmean(b), 4))
    v["P2"] = {"verdict": "UNREAD" if None in p2 else ("HOLDS" if all(x >= 1.0 for x in p2) else "MISSED"),
               "lc128_over_base_ttft_p50_r12": p2}
    p3 = [_get(d, tag(a, k), "r4", "attainment") for a in KNOB_ARMS for k in DRAWS]
    v["P3"] = {"verdict": "UNREAD" if None in p3 else ("HOLDS" if all(x >= 0.95 for x in p3) else "MISSED")}
    sc = {t: _short_chunks(d, t) for t, *_ in srv}
    p4 = [sc[t] for t, a, _, _ in srv if a in ("lc256", "lc128")]
    p4b = [sc[t] for t, a, _, _ in srv if a == "base"]
    v["P4"] = {"verdict": "UNREAD" if None in p4 + p4b else
               ("HOLDS" if all(x > 0 for x in p4) and all(x == 0 for x in p4b) else "MISSED")}

    rep = out["reported"]
    rep["short_chunk"] = sc
    rep["prefill_deferred"] = {t: _deferred(d, t) for t, *_ in srv}
    rep["vs_ref"] = {}
    for k in DRAWS:
        for arm in KNOB_ARMS:
            t = tag(arm, k)
            rep["vs_ref"][t] = {f"{m}_r{r}": _get(d, t, f"r{r}", m) for m in ("tpot_p99_s", "ttft_p50_s") for r in (8, 12)}
            rep["vs_ref"][t]["text_agreement_vs_base_a_r12"] = _agreement(
                _load(d, f"{tag('base', k, 'base_a')}_r12.json"), _load(d, f"{t}_r12.json"))
    out["outcome"] = {"void": void, "verdicts": {a: out["arms"][a]["verdict"] for a in KNOB_ARMS}}
    return out


def engagement_cli(ends, arms) -> int:
    e_red = _sib("sc2e_reduce")
    why = []
    for p, arm in zip(ends, arms):
        h = json.load(open(p)) if os.path.exists(p) else None
        why += [f"{arm}: {w}" for w in engagement_bad(e_red, h, None, arm)]
    print("SC2I_ENGAGEMENT " + json.dumps({"ok": not why, "why": why}), flush=True)
    return 1 if why else 0


# ------------------------------------------------------------------ self-test --

def _fixture(d, *, tpot=None, att=None, ttft99=None, knob_edit=None, missing=(), short=None, serial_ttft=None):
    """Box P's run files for all twelve servers. Defaults: base TPOT p50 at 12 req/s 30 ms (base_b 30.6), pe2 26 ms, pe4
    25, lc256 28.8, lc128 29.4; attainment 1.0 everywhere; TTFT p99 at 12 req/s 0.4 s; every gate passing."""
    c_red, f_red = _sib("sc2c_reduce"), _sib("sc2f_reduce")
    tpot = {"base_a": 0.030, "base_b": 0.0306, "pe2": 0.026, "pe4": 0.025, "lc256": 0.0288, "lc128": 0.0294, **(tpot or {})}
    serial_ttft = {"base": 0.040, "lc128": 0.046, **(serial_ttft or {})}

    def summ(ttft, tp, a, rate, t99=None):
        return {"valid": 120, "invalid": 0, "errors": [], "ttft_p50_s": ttft, "ttft_p99_s": t99 if t99 is not None else ttft * 2,
                "tpot_p50_s": tp, "tpot_p99_s": tp * 2, "e2el_p50_s": 1.0, "output_tok_s": 100.0, "attainment": a,
                "goodput_rps": a * rate, "achieved_rate": rate}

    def dump(name, obj):
        json.dump(obj, open(os.path.join(d, name), "w"))
    for t, arm, k, slot in servers():
        if t in missing:
            continue
        key = slot if slot in tpot else arm

        def reqs(n, txt="x"):
            return [{"valid": True, "text": f"{txt}{i}", "prompt_tokens": T} for i in range(n)]
        dump(f"{t}_warm.json", {"requests": reqs(4)})
        dump(f"{t}_burst.json", {"requests": reqs(64)})
        dump(f"{t}_serial.json", {"requests": reqs(N_SERIAL, "s"),
                                  "summary": summ(serial_ttft.get(arm, serial_ttft["base"]), 0.004, 1.0, 0.0)})
        for r in LADDER:
            a = (att or {}).get((t, r), 1.0)
            t99 = (ttft99 or {}).get(t, 0.4) if r == 12 else None
            ttft50 = 0.06 if (arm == "lc128" and r == 12) else 0.05
            dump(f"{t}_r{r}.json", {"requests": reqs(N_RATE), "summary": summ(ttft50, tpot[key], a, r, t99)})
        n = 4 + 64 + N_SERIAL + N_RATE * len(LADDER)
        w = f_red.WANT_FLIPS["on"]
        seen = {"moe": {"int4_k19|gt256": 384, "int4_k19|le256": 720}, "prefill_attn": {"flash": 240},
                "moe_k19_dispatch": {"chained|lean|gt256": 240, "one_launch|lean|gt256": 144, "one_launch|lean|le256": 720},
                "prefill_folds": {f: 48 for f in f_red.FOLDS}}
        pr = dict(c_red.ROUTES, seen=seen, prefill_glue_env=None, prefill_lean_dispatch_env=None, prefill_lean_dispatch="auto")
        lev = {"kv": {"scratch_slots": 64, "pool_mib": 6678.8}, "prefill_glue": dict(w["glue"])}
        b = [1, 2, 4, 8, 16, 32, 64]
        pe, lc = KNOBS[arm]
        eng = {"chunk_tokens": T, "max_prefill_tokens_per_step": T, "max_seqs": 64, "kv_slots": 64, "buckets": b,
               "buckets_requested": "auto", "graph_status": {str(x): "graph" for x in b}, "decode_lookahead": False,
               "prefill_every": pe, "loaded_chunk_tokens": lc}
        eng.update((knob_edit or {}).get(t) or {})
        dump(f"health_{t}_start.json", {"prefill_routes": pr, "engine": eng, "levers": lev,
                                        "prefill_graph": {"status": "on", "pool_mib": 428, "free_after_mib": 3700,
                                                          "bulk_flush_mib": 216}})
        sc = (short or {}).get(t, 0 if lc == 0 else 40)
        later = 0 if lc == 0 else sc * (T // lc - 1)
        reasons = {**({"short_chunk": sc} if sc else {}), **({"later_chunk": later} if later else {})}
        dump(f"health_{t}_end.json", {
            "prefill_graph": {"status": "on", "T": T, "replays": n - sc, "eager_chunks": sc + later,
                              "eager_reasons": reasons},
            "kv_bookkeeping": {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0,
                               "ready_bulk": 0, "ready_at_flush": n},
            "engine": {"graph_stats": {str(x): {"replays": 5, "eager_steps": 0, "rows": 5, "pad_rows": 0} for x in b}}})
        dump(f"stats_{t}_end.json", {"scheduler": {"prefill_deferred": 30 if pe > 1 else 0} if pe > 1 else {"steps": 1}})


def self_test() -> int:
    cases = []

    def run(**kw):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            return reduce(d)
    r = run()
    vs = r["outcome"]["verdicts"]
    cases += [
        ("the default fixture's gates all pass", not r["outcome"]["void"]),
        ("pe2 and pe4 read FASTER", vs["pe2"] == "FASTER" and vs["pe4"] == "FASTER"),
        ("lc256 and lc128 read FLAT inside the bar", vs["lc256"] == "FLAT" and vs["lc128"] == "FLAT"),
        ("the bar is the 0.05 floor when noise is smaller", r["draws"]["d1"]["bar"] == 0.05),
        ("the noise is |base_a - base_b| / ref", abs(r["draws"]["d1"]["noise"] - round(0.0006 / 0.0303, 4)) < 1e-9),
        ("P1 holds at 26 / 30.3", r["predictions"]["P1"]["verdict"] == "HOLDS"),
        ("P2 holds when lc128's TTFT p50 at 12 req/s is higher", r["predictions"]["P2"]["verdict"] == "HOLDS"),
        ("P4 holds: short chunks on lc arms only", r["predictions"]["P4"]["verdict"] == "HOLDS"),
    ]
    r = run(knob_edit={"e4b_pe4_d2": {"prefill_every": 1}})
    cases.append(("a server not on its arm VOIDs every arm", set(r["outcome"]["verdicts"].values()) == {"VOID"}
                  and "e4b_pe4_d2" in r["outcome"]["void"]))
    r = run(tpot={"pe2": 0.034})
    cases.append(("a slower arm in both draws reads REGRESSED", r["outcome"]["verdicts"]["pe2"] == "REGRESSED"))
    r = run(att={("e4b_pe4_d1", 12): 0.96})
    cases.append(("attainment 0.04 below base blocks FASTER, not REGRESSED", r["outcome"]["verdicts"]["pe4"] == "FLAT"))
    r = run(att={("e4b_pe4_d1", 8): 0.9, ("e4b_pe4_d2", 8): 0.9})
    cases.append(("attainment 0.10 below base in both draws reads REGRESSED", r["outcome"]["verdicts"]["pe4"] == "REGRESSED"))
    r = run(ttft99={"e4b_pe2_d1": 1.2, "e4b_pe2_d2": 1.2})
    cases.append(("TTFT p99 above 1 s and above base blocks FASTER", r["outcome"]["verdicts"]["pe2"] == "FLAT"))
    r = run(tpot={"base_b": 0.040})
    cases.append(("a noisy base widens the bar", r["draws"]["d1"]["bar"] > 0.2 and r["outcome"]["verdicts"]["pe2"] == "FLAT"))
    r = run(missing=("e4b_lc128_d2",))
    cases.append(("a missing server is UNREAD for its arm and VOID for none", r["outcome"]["verdicts"]["lc128"] in ("UNREAD", "VOID")))
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        e_red = _sib("sc2e_reduce")
        h = json.load(open(os.path.join(d, "health_e4b_lc128_d1_end.json")))
        n = _prompts(d, "e4b_lc128_d1")
        cases.append(("an lc arm's short and later chunks pass its adapted ENGAGED", engagement_bad(e_red, h, n, "lc128") == []))
        cases.append(("the same record fails SC2e's unadapted check", e_red.engagement(SLOT_ARM, h, n) != []))
        h2 = json.loads(json.dumps(h))
        h2["prefill_graph"]["eager_reasons"]["no_graph"] = 2
        cases.append(("an lc arm with another eager reason fails", any("beyond" in w for w in engagement_bad(e_red, h2, n, "lc128"))))
        h3 = json.loads(json.dumps(h))
        h3["prefill_graph"]["replays"] -= 1
        cases.append(("an lc arm whose first chunks do not add up fails", engagement_bad(e_red, h3, n, "lc128") != []))
        hb = json.load(open(os.path.join(d, "health_e4b_pe2_d1_end.json")))
        hb["prefill_graph"].update(eager_chunks=1, eager_reasons={"short_chunk": 1}, replays=hb["prefill_graph"]["replays"] - 1)
        cases.append(("a pe arm keeps SC2e's check: a short chunk fails it", engagement_bad(e_red, hb, n, "pe2") != []))
    r = run(short={"e4b_base_a_d1": 3})
    cases.append(("a short chunk on base misses P4", r["predictions"]["P4"]["verdict"] == "MISSED"))
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        h = json.load(open(os.path.join(d, "health_e4b_lc256_d1_start.json")))
        cases.append(("knobs_bad passes the arm's own record", knobs_bad(h, "lc256") == {}))
        cases.append(("knobs_bad names a departure", "engine.loaded_chunk_tokens" in knobs_bad(h, "lc128")))
    bad = [n for n, ok in cases if not ok]
    print(f"sc2i_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--knobs", nargs=2, metavar=("HEALTH", "ARM"))
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--engagement", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.knobs:
        h = json.load(open(a.knobs[0])) if os.path.exists(a.knobs[0]) else None
        bad = knobs_bad(h, a.knobs[1], proof=a.proof)
        print("SC2I_KNOBS " + json.dumps({"arm": a.knobs[1], "ok": not bad, "bad": bad}, default=str), flush=True)
        return 1 if bad else 0
    if a.engagement:
        return engagement_cli(a.engagement, a.arms or [])
    if a.dir:
        r = reduce(a.dir)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(r, f, indent=1, default=str)
        print("SC2I " + json.dumps({"outcome": r["outcome"], "predictions": {k: v["verdict"] for k, v in r["predictions"].items()}},
                                   default=str), flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
