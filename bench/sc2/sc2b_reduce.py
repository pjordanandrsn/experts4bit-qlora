#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2b_reduce.py -- lane SC2b's registered rule (bench/sc2/SC2b-PREREG.md; #846): e4b serve_paged's prefill graph,
OFF against ON, from box F's run files.

Files in DIR, per draw D in 1..2 and arm A in (off, on), all from sc2_driver.py ``run`` and the box:
  e4b_<A>_d<D>_serial.json, e4b_<A>_d<D>_r<R>.json      the plan (BOTH arms of a draw use the same seeds)
  e4b_off_d1_serial_repeat.json                         the determinism control
  health_e4b_<A>_d<D>_start.json / _end.json            the server's own /health at start and after its runs

The gates, in order (the first that fails names the outcome):
- **ROUTES**: every arm's start /health reads the registered routes (k19 / k19 / flash, device grouping on, neither pin
  in the environment), or the run is VOID. The box STOPs such an arm; this re-reads it from the files.
- **ENGAGED**: every OFF arm's end /health shows the knob off with zero graph replays; every ON arm's shows it on, with
  one replay per prompt served and no eager prefill chunk. Otherwise the run is VOID (the lever did not run as
  registered).
- **DETERMINISM**: OFF draw-1 serial against its repeat is IDENTICAL (sc2_identity), or the identity gate is UNREAD.
- **IDENTITY**: OFF against ON on each draw's serial plan is IDENTICAL, or the lever DIFFERS and cannot be a default.
- **PROMPTS**: every request in every run reports ``usage.prompt_tokens`` == 512 (one first chunk each), or VOID.

**Rows** are sc2_reduce.row() on each arm's two draws (UNREAD / INVALID / UNSTABLE / VALID); **ceilings** are
sc2_reduce.ceiling().

**Predictions** (paired per draw, so the shared arrivals cancel):
  P1  serial p50 TTFT, OFF / ON, >= 1.5 in BOTH draws (a prediction; it does not gate the licence).
  P2  serial p50 TPOT, ON / OFF, within [0.95, 1.05] in BOTH draws (the graph is a prefill lever; decode is untouched).
  P3  ON's capacity ceiling >= 2 req/s.
  P4  no regression: at every rate, in both draws, ON's attainment >= OFF's - 0.05; and ON's ceiling >= OFF's.

**Licence** (the default recommendation), decoupled from P1: DEFAULT_LICENSED iff ROUTES, ENGAGED, PROMPTS, DETERMINISM
and IDENTITY pass, P4 holds, AND serial p50 TTFT OFF / ON >= 1.10 in both draws (a bitwise-identical, non-regressing lever
that cuts TTFT is a default whether or not it reaches P1's bar). Otherwise NOT_LICENSED, with the reason.

**Reported, no bar:** OFF's own rows and ceiling (today's default routes, gnf4 v0.38.0) beside SC2's e4b-as-run (SC1's
pins, v0.34.1, another board): descriptive only.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RATES = (1, 2, 4, 8)
DRAWS = (1, 2)
ARMS = ("off", "on")
P1_MIN, P2_BAND, P3_MIN, P4_TOL, LICENCE_TTFT = 1.5, (0.95, 1.05), 2, 0.05, 1.10
ROUTES = {"int4_prefill": "k19", "int4_prefill_above_256_rows": "k19", "prefill_attn": "flash", "device_grouping": True,
          "int4_prefill_env": None, "prefill_attn_env": None}
# The knob's engagement record, read from the code (serve_paged.prefill_graph_report and
# PagedModelRunner.prefill_graph_stats at serve/prefill-graph 9818c557): /health ALWAYS carries a `prefill_graph` block
# with `requested` (the env, as a bool) and `status` -- "off" (not requested), "on" (engaged: T, replays, eager_chunks,
# eager_reasons {later_chunk, short_chunk}), "loading", "refused" (startup refused; `why` names it) or "error". The
# startup verification's replays are NOT counted: the counters are re-zeroed after it passes, so on a workload of
# T-token prompts, replays == every request admitted.
ENGAGE_BLOCK, ENGAGE_STATE, ENGAGE_REPLAYS, ENGAGE_EAGER = "prefill_graph", "status", "replays", "eager_chunks"
ENGAGE_ON, ENGAGE_OFF, ENGAGE_T = "on", "off", 512


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def routes_bad(health) -> dict:
    """The registered routes, plus the chunking that keeps every 512-token prompt ONE first chunk (chunk_tokens 512 and
    the default per-step prefill budget, also 512: a larger budget could split a first chunk into short_chunk)."""
    r = (health or {}).get("prefill_routes") or {}
    bad = {k: r.get(k, "<missing>") for k, v in ROUTES.items() if r.get(k, "<missing>") != v}
    e = (health or {}).get("engine") or {}
    for k in ("chunk_tokens", "max_prefill_tokens_per_step"):
        if e.get(k) != ENGAGE_T:
            bad[f"engine.{k}"] = e.get(k, "<missing>")
    return bad


def engagement(off_end, on_end, prompts_on) -> dict:
    """The knob's own counters: OFF never replays; ON replays once per prompt and runs no eager prefill chunk."""
    why = []
    for arm, h in (("off", off_end), ("on", on_end)):
        if h is None:
            why.append(f"{arm}: no end /health")
            continue
        g = h.get(ENGAGE_BLOCK)
        if not isinstance(g, dict):
            why.append(f"{arm}: /health has no {ENGAGE_BLOCK!r} block (status {h.get('status')!r})")
            continue
        if arm == "off":
            if g.get(ENGAGE_STATE) != ENGAGE_OFF or g.get("requested") is not False:
                why.append(f"off: {g!r}")
            continue
        st, rep, eag = g.get(ENGAGE_STATE), g.get(ENGAGE_REPLAYS), g.get(ENGAGE_EAGER)
        if (st != ENGAGE_ON or g.get("requested") is not True or g.get("T") != ENGAGE_T or eag != 0
                or (prompts_on is not None and rep != prompts_on)):
            why.append(f"on: status {st!r}, T {g.get('T')!r}, replays {rep!r} (want {prompts_on}), eager_chunks {eag!r}, "
                       f"why {g.get('why')!r}")
    return {"ok": not why, "why": why}


def _prompts(d, arm, draw):
    """Requests the arm's server admitted: warm + serial (+ repeat) + every rate run, all 512-token single-chunk."""
    n = 0
    for name in [f"e4b_{arm}_d{draw}_warm", f"e4b_{arm}_d{draw}_serial", f"e4b_{arm}_d{draw}_serial_repeat"] + \
            [f"e4b_{arm}_d{draw}_r{r}" for r in RATES]:
        x = _load(d, name + ".json")
        n += len(x["requests"]) if x else 0
    return n


def reduce(d: str) -> dict:
    red, ident = _sib("sc2_reduce"), _sib("sc2_identity")
    out = {"gates": {}, "arms": {}, "predictions": {}}
    # ROUTES
    bad = {f"{a}_d{k}": routes_bad(_load(d, f"health_e4b_{a}_d{k}_start.json")) for a in ARMS for k in DRAWS}
    bad = {k: v for k, v in bad.items() if v}
    out["gates"]["routes"] = {"ok": not bad, "bad": bad}
    # PROMPTS: every request one first chunk
    off_len = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("e4b_") and f.endswith(".json"):
            x = _load(d, f)
            if isinstance(x, dict) and isinstance(x.get("requests"), list):
                lens = {r.get("prompt_tokens") for r in x["requests"]}
                if lens != {ENGAGE_T}:
                    off_len[f] = sorted(lens, key=str)
    out["gates"]["prompts"] = {"ok": not off_len, "bad": off_len}
    # ENGAGED, per draw
    eng = {k: engagement(_load(d, f"health_e4b_off_d{k}_end.json"), _load(d, f"health_e4b_on_d{k}_end.json"),
                         _prompts(d, "on", k)) for k in DRAWS}
    out["gates"]["engaged"] = {"ok": all(e["ok"] for e in eng.values()), "draws": eng}
    # DETERMINISM and IDENTITY
    a, b = _load(d, "e4b_off_d1_serial.json"), _load(d, "e4b_off_d1_serial_repeat.json")
    det = ident.compare(a, b) if a and b else {"verdict": "REFUSED", "why": "a determinism file is missing"}
    out["gates"]["determinism"] = det
    ids = {}
    for k in DRAWS:
        x, y = _load(d, f"e4b_off_d{k}_serial.json"), _load(d, f"e4b_on_d{k}_serial.json")
        ids[k] = ident.compare(x, y) if x and y else {"verdict": "REFUSED", "why": "a serial file is missing"}
    idv = ("UNREAD" if det["verdict"] != "IDENTICAL" else
           "IDENTICAL" if all(v["verdict"] == "IDENTICAL" for v in ids.values()) else
           "DIFFERS" if any(v["verdict"] == "DIFFERS" for v in ids.values()) else "UNREAD")
    out["gates"]["identity"] = {"verdict": idv, "draws": ids}
    # rows and ceilings per arm (SC2's rule, file names mapped to this box's)
    for arm in ARMS:
        rows = {"serial": red.row([_load(d, f"e4b_{arm}_d{k}_serial.json") for k in DRAWS], "serial")}
        for r in RATES:
            rows[r] = red.row([_load(d, f"e4b_{arm}_d{k}_r{r}.json") for k in DRAWS], r)
        out["arms"][arm] = {"rows": {str(k): v for k, v in rows.items()}, "ceiling": red.ceiling(rows)}
    # predictions, paired per draw
    def s(arm, k, name):
        x = _load(d, f"e4b_{arm}_d{k}_{name}.json")
        return x["summary"] if x else None
    v = out["predictions"]
    ser = [(s("off", k, "serial"), s("on", k, "serial")) for k in DRAWS]
    if any(o is None or n is None or o["invalid"] or n["invalid"] for o, n in ser):
        v["P1"] = v["P2"] = {"verdict": "UNREAD", "why": "a serial run is missing or has INVALID requests"}
    else:
        r1 = [round(o["ttft_p50_s"] / n["ttft_p50_s"], 4) for o, n in ser]
        out["ttft_off_over_on"] = r1
        r2 = [round(n["tpot_p50_s"] / o["tpot_p50_s"], 4) for o, n in ser]
        v["P1"] = {"verdict": "HOLDS" if min(r1) >= P1_MIN else "REFUTED", "ttft_off_over_on": r1}
        v["P2"] = {"verdict": "HOLDS" if all(P2_BAND[0] <= x <= P2_BAND[1] for x in r2) else "REFUTED", "tpot_on_over_off": r2}
    c_off, c_on = out["arms"]["off"]["ceiling"], out["arms"]["on"]["ceiling"]
    v["P3"] = {"verdict": "UNREAD" if c_on is None else ("HOLDS" if c_on >= P3_MIN else "REFUTED"), "ceiling_on": c_on}
    pairs, miss = {}, False
    for r in RATES:
        for k in DRAWS:
            o, n = s("off", k, f"r{r}"), s("on", k, f"r{r}")
            if o is None or n is None:
                miss = True
                continue
            pairs[f"r{r}_d{k}"] = [o["attainment"], n["attainment"]]
    regress = {k: p for k, p in pairs.items() if p[1] < p[0] - P4_TOL}
    if miss or c_off is None or c_on is None:
        v["P4"] = {"verdict": "UNREAD", "pairs": pairs}
    else:
        v["P4"] = {"verdict": "HOLDS" if not regress and c_on >= c_off else "REFUTED", "pairs": pairs, "regressions": regress,
                   "ceilings": {"off": c_off, "on": c_on}}
    g = out["gates"]
    ttft_ok = bool(out.get("ttft_off_over_on")) and min(out["ttft_off_over_on"]) >= LICENCE_TTFT
    why = [n for n, ok in (("routes", g["routes"]["ok"]), ("engaged", g["engaged"]["ok"]), ("prompts", g["prompts"]["ok"]),
                           ("identity", g["identity"]["verdict"] == "IDENTICAL"), ("P4", v["P4"]["verdict"] == "HOLDS"),
                           (f"ttft_off_over_on >= {LICENCE_TTFT}", ttft_ok)) if not ok]
    out["licence"] = {"verdict": "DEFAULT_LICENSED" if not why else "NOT_LICENSED", "failed": why}
    out["void"] = not (g["routes"]["ok"] and g["engaged"]["ok"] and g["prompts"]["ok"])
    return out


# ------------------------------------------------------------------ self-test --

def self_test() -> int:
    import tempfile
    cases = []

    def summ(ttft, tpot, att, rate, invalid=0):
        return {"valid": 120 - invalid, "invalid": invalid, "errors": [], "ttft_p50_s": ttft, "ttft_p99_s": ttft * 2,
                "tpot_p50_s": tpot, "tpot_p99_s": tpot * 2, "e2el_p50_s": 1.0, "output_tok_s": 100.0, "attainment": att,
                "goodput_rps": att * rate, "achieved_rate": rate}

    def write(d, ttft_on=0.08, att_on=1.0, text_on="abc", replays=None, routes_on=None, repeat_text="abc"):
        def dump(name, obj):
            json.dump(obj, open(os.path.join(d, name), "w"))
        for k in DRAWS:
            for arm in ARMS:
                on = arm == "on"
                plan = [[0.0, i, 64] for i in range(4)]
                serial = {"mode": "serial", "prompts_sha256": "s", "plan": plan,
                          "requests": [{"valid": True, "text": (text_on if on else "abc") + str(i), "prompt_tokens": 512}
                                       for i in range(4)],
                          "summary": summ(ttft_on if on else 0.27, 0.0045, 0.0, 0.0)}
                dump(f"e4b_{arm}_d{k}_serial.json", serial)
                dump(f"e4b_{arm}_d{k}_warm.json", {"requests": [{"prompt_tokens": 512}] * 4})
                for r in RATES:
                    att = (att_on if r <= 4 else 0.5) if on else (1.0 if r == 1 else 0.3)
                    dump(f"e4b_{arm}_d{k}_r{r}.json", {"requests": [{"prompt_tokens": 512}] * 120,
                                                       "summary": summ(0.3, 0.02, att, r)})
                routes = dict(ROUTES)
                if on and routes_on:
                    routes.update(routes_on)
                dump(f"health_e4b_{arm}_d{k}_start.json", {"prefill_routes": routes,
                                                         "engine": {"chunk_tokens": 512, "max_prefill_tokens_per_step": 512}})
                n_on = 4 + 4 + 4 * 120
                rep = replays if replays is not None else n_on
                dump(f"health_e4b_{arm}_d{k}_end.json",
                     {ENGAGE_BLOCK: {ENGAGE_STATE: ENGAGE_ON, "requested": True, "T": ENGAGE_T, ENGAGE_REPLAYS: rep,
                                     ENGAGE_EAGER: 0, "eager_reasons": {"later_chunk": 0, "short_chunk": 0}}} if on
                     else {ENGAGE_BLOCK: {ENGAGE_STATE: ENGAGE_OFF, "requested": False}})
        dump("e4b_off_d1_serial_repeat.json", {"mode": "serial", "prompts_sha256": "s", "plan": [[0.0, i, 64] for i in range(4)],
                                               "requests": [{"valid": True, "text": repeat_text + str(i), "prompt_tokens": 512}
                                                            for i in range(4)]})
    with tempfile.TemporaryDirectory() as d:
        write(d)
        o = reduce(d)
        cases.append(("licensed", o["licence"]["verdict"] == "DEFAULT_LICENSED" and o["arms"]["on"]["ceiling"] == 4
                      and o["arms"]["off"]["ceiling"] == 1 and o["predictions"]["P1"]["verdict"] == "HOLDS"))
    with tempfile.TemporaryDirectory() as d:
        write(d, text_on="abd")
        o = reduce(d)
        cases.append(("differs", o["gates"]["identity"]["verdict"] == "DIFFERS" and "identity" in o["licence"]["failed"]))
    with tempfile.TemporaryDirectory() as d:
        write(d, repeat_text="xyz")
        cases.append(("nondeterministic", reduce(d)["gates"]["identity"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:
        write(d, replays=3)
        o = reduce(d)
        cases.append(("not engaged", not o["gates"]["engaged"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:                     # a refused ON server reads as not engaged: VOID
        write(d)
        json.dump({"status": "error", ENGAGE_BLOCK: {ENGAGE_STATE: "refused", "requested": True,
                   "why": "RuntimeError: E4B_PAGED_PREFILL_GRAPH=1 refused: x"}},
                  open(os.path.join(d, "health_e4b_on_d2_end.json"), "w"))
        o = reduce(d)
        cases.append(("refused", not o["gates"]["engaged"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:
        write(d, routes_on={"int4_prefill_above_256_rows": "mtile"})
        o = reduce(d)
        cases.append(("wrong route", not o["gates"]["routes"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:                     # 0.27 / 0.2 = 1.35: P1 fails, the licence still passes
        write(d, ttft_on=0.2)
        o = reduce(d)
        cases.append(("P1 fails, licensed", o["predictions"]["P1"]["verdict"] == "REFUTED"
                      and o["licence"]["verdict"] == "DEFAULT_LICENSED"))
    with tempfile.TemporaryDirectory() as d:                     # 0.27 / 0.26 < 1.10: not licensed
        write(d, ttft_on=0.26)
        cases.append(("too slow", reduce(d)["licence"]["verdict"] == "NOT_LICENSED"))
    with tempfile.TemporaryDirectory() as d:                     # a prompt that is not one first chunk: VOID
        write(d)
        x = json.load(open(os.path.join(d, "e4b_on_d1_r4.json")))
        x["requests"][7]["prompt_tokens"] = 511
        json.dump(x, open(os.path.join(d, "e4b_on_d1_r4.json"), "w"))
        o = reduce(d)
        cases.append(("prompt drift", not o["gates"]["prompts"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:
        write(d, att_on=0.2)
        o = reduce(d)
        cases.append(("regression", o["predictions"]["P4"]["verdict"] == "REFUTED"))
    bad = [n for n, ok in cases if not ok]
    print(f"sc2b_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--engagement", nargs=2, metavar=("OFF_END.json", "ON_END.json"),
                   help="the proof's check: the two arms' end /health, knob off and on (replay count not pinned)")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.engagement:
        e = engagement(*(json.load(open(f)) for f in a.engagement), None)
        print("SC2B_ENGAGEMENT " + json.dumps(e), flush=True)
        return 0 if e["ok"] else 3
    out = reduce(a.dir)
    json.dump(out, open(a.out, "w"), indent=1)
    for name, g in out["gates"].items():
        print(f"SC2B_GATE {name} " + json.dumps(g)[:300])
    for q, x in out["predictions"].items():
        print(f"SC2B_{q} {x['verdict']} " + json.dumps({k: w for k, w in x.items() if k != 'verdict'})[:300])
    for arm, x in out["arms"].items():
        print(f"SC2B_ARM {arm} ceiling={x['ceiling']} rows=" + json.dumps({k: y['status'] for k, y in x['rows'].items()}))
    print(f"SC2B_LICENCE {out['licence']['verdict']} failed={out['licence']['failed']} void={out['void']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
