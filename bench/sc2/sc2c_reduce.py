#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2c_reduce.py -- lane SC2c's registered rule (bench/sc2/SC2c-PREREG.md; #846): e4b serve_paged's bulk KV
bookkeeping (E4B_PAGED_BULK_KV), OFF against ON, from box H's run files.

Files in DIR, per draw D in 1..2 and arm A in (off, on), from sc2_driver.py ``run`` and the box:
  e4b_<A>_d<D>_warm.json, _serial.json, _r<R>.json    the plan (BOTH arms of a draw use the same seeds)
  e4b_off_d1_serial_repeat.json                       the determinism control
  health_e4b_<A>_d<D>_start.json / _end.json          the server's own /health at start and after its runs
  trace_e4b_<A>_d<D>.jsonl                            E4B_PAGED_TRACE: one row per request (P1's fit; the census)
  steps_e4b_<A>_d<D>.jsonl                            E4B_PAGED_STEP_TRACE: one row per engine step (the census)
  vram_e4b_<A>_d<D>.txt                               nvidia-smi memory.used,total at ready (recorded)

The gates, in order (the first that fails names the outcome):
- **ROUTES**: every arm's start /health reads the registered routes (k19 / k19 / flash, device grouping on, neither pin
  in the environment, chunking 512 / 512) AND its ``prefill_routes.seen`` shows what the forward took at the startup
  captures: every expert GEMM above 256 rows on K19, every prefill attention call on flash (e4b#1129). Otherwise VOID.
- **ENGAGED**: every arm's end /health shows the prefill graph on, with one replay per prompt and no eager chunk (the
  stack under test, both arms), and its ``kv_bookkeeping`` block shows the knob as registered. OFF: every prompt
  flushed per layer and every slot's blocks claimed per layer at its first graphed decode, nothing in bulk. ON: every
  prompt flushed in bulk with the slot's blocks claimed at that flush, nothing per layer. Otherwise VOID.
- **DETERMINISM**: OFF draw-1 serial against its repeat is IDENTICAL (sc2_identity), or the identity gate is UNREAD.
- **IDENTITY**: OFF against ON on each draw's serial plan is IDENTICAL, or the lever DIFFERS and cannot be a default.
- **PROMPTS**: every request in every run reports ``usage.prompt_tokens`` == 512 (one first chunk each), or VOID.

**Rows** are sc2_reduce.row() on each arm's two draws; **ceilings** are sc2_reduce.ceiling().

**Predictions** (paired per draw):
  P1  the stall per prefill, bucket-controlled (sc2c_census.fit), ON / OFF <= 0.6 in BOTH draws (the mechanism).
  P2  serial p50 TTFT, OFF / ON, >= 1.4 in BOTH draws.
  P3  serial p50 TPOT, ON / OFF, within [0.95, 1.05] in BOTH draws (the decode step is untouched).
  P4  ON's capacity ceiling >= 2 req/s.
  P5  ON's capacity ceiling >= 4 req/s (the bolder reading; see the registration's basis).
  P6  no regression: at every rate, in both draws, ON's attainment >= OFF's - 0.05; and ON's ceiling >= OFF's.

**Licence**, decoupled from P1, P2, P4 and P5: DEFAULT_LICENSED iff ROUTES, ENGAGED, PROMPTS, DETERMINISM and
IDENTITY pass, P6 holds, AND serial p50 TTFT OFF / ON >= 1.10 in both draws. Otherwise NOT_LICENSED, with the reason.

**Reported, no bar (the census):** per arm and draw, sc2c_census.steps on the step trace (the prefill step's segments,
the forward's device time, bookkeeping per prompt, decode steps by bucket, the direct stall) and sc2c_census.fit on the
request trace (two-regressor and bucket-controlled); VRAM at ready.
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
P1_MAX, P2_MIN, P3_BAND, P4_MIN, P5_MIN, P6_TOL, LICENCE_TTFT = 0.6, 1.4, (0.95, 1.05), 2, 4, 0.05, 1.10
T = 512
ROUTES = {"int4_prefill": "k19", "int4_prefill_above_256_rows": "k19", "prefill_attn": "flash", "device_grouping": True,
          "int4_prefill_env": None, "prefill_attn_env": None}
# The knob's engagement record, read from the code (serve_paged.kv_bookkeeping_report and
# PagedModelRunner.kv_bookkeeping_stats on branch serve-bulk-kv): /health ALWAYS carries a `kv_bookkeeping` block with
# `requested` (E4B_PAGED_BULK_KV as a bool) and, once built, `bulk` and per-request counts: flush_layers / flush_bulk
# (a prompt's flush into the pool, per path), ready_layers / ready_bulk (a graphed slot's block claims at its first
# decode, per path) and ready_at_flush (claimed by the bulk flush itself).
KV_BLOCK = "kv_bookkeeping"


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


def routes_bad(health) -> dict:
    """The registered routes, the chunking that keeps every prompt one first chunk, and what the forward TOOK
    (``prefill_routes.seen``, e4b#1129): every expert GEMM above 256 rows on K19 and every prefill attention call on
    flash. The resolved fields alone read k19 / flash on gpt-oss while neither ran (sc2g-prove-2)."""
    r = (health or {}).get("prefill_routes") or {}
    bad = {k: r.get(k, "<missing>") for k, v in ROUTES.items() if r.get(k, "<missing>") != v}
    e = (health or {}).get("engine") or {}
    for k in ("chunk_tokens", "max_prefill_tokens_per_step"):
        if e.get(k) != T:
            bad[f"engine.{k}"] = e.get(k, "<missing>")
    seen = r.get("seen") or {}
    moe, att = seen.get("moe") or {}, seen.get("prefill_attn") or {}
    gt = {k: v for k, v in moe.items() if k.endswith("|gt256")}
    if not gt or any(not k.startswith("int4_k19|") for k in gt):
        bad["seen.moe_gt256"] = gt or "<none>"
    if not att or set(att) != {"flash"}:
        bad["seen.prefill_attn"] = att or "<none>"
    return bad


def engagement(arm, end, prompts) -> list:
    """Why ``arm``'s end /health does not read as registered (empty: engaged). ``prompts`` None skips the counts'
    equality (the proof's smoke, whose warm count is the driver's)."""
    if end is None:
        return [f"{arm}: no end /health"]
    why = []
    g = end.get("prefill_graph") or {}
    if g.get("status") != "on" or g.get("T") != T or g.get("eager_chunks") != 0 or \
            (prompts is not None and g.get("replays") != prompts):
        why.append(f"{arm}: prefill_graph {g!r} (want on, T {T}, replays {prompts}, eager_chunks 0)")
    k = end.get(KV_BLOCK)
    if not isinstance(k, dict) or "flush_layers" not in k:
        return why + [f"{arm}: /health has no built {KV_BLOCK!r} block ({k!r})"]
    n = prompts if prompts is not None else max(k.get("flush_layers", 0), k.get("flush_bulk", 0))
    want = ({"requested": False, "bulk": False, "flush_layers": n, "flush_bulk": 0, "ready_layers": n, "ready_bulk": 0,
             "ready_at_flush": 0} if arm == "off" else
            {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0, "ready_bulk": 0,
             "ready_at_flush": n})
    bad = {x: k.get(x, "<missing>") for x, v in want.items() if k.get(x, "<missing>") != v}
    if bad or not n:
        why.append(f"{arm}: {KV_BLOCK} {bad or k!r} (want {want})")
    return why


def _prompts(d, arm, draw):
    """Requests the arm's server admitted: warm + serial (+ repeat) + every rate run, all 512-token single-chunk."""
    n = 0
    for name in [f"e4b_{arm}_d{draw}_warm", f"e4b_{arm}_d{draw}_serial", f"e4b_{arm}_d{draw}_serial_repeat"] + \
            [f"e4b_{arm}_d{draw}_r{r}" for r in RATES]:
        x = _load(d, name + ".json")
        n += len(x["requests"]) if x else 0
    return n


def _plan(d, arm, draw):
    """The request trace's workloads in the server's order (warm, serial, the repeat on OFF draw 1, every rate)."""
    out = []
    for name in ["warm", "serial", "serial_repeat"] + [f"r{r}" for r in RATES]:
        x = _load(d, f"e4b_{arm}_d{draw}_{name}.json")
        if x:
            out.append((name, len(x["requests"])))
    return out


def census(d) -> dict:
    cen = _sib("sc2c_census")
    out = {}
    for arm in ARMS:
        for k in DRAWS:
            tag = f"{arm}_d{k}"
            c = {}
            steps = _jsonl(d, f"steps_e4b_{tag}.jsonl")
            c["steps"] = cen.steps(steps) if steps else {"why": "no step trace"}
            rows = _jsonl(d, f"trace_e4b_{tag}.jsonl")
            try:
                c["fit"] = cen.fit(rows, _plan(d, arm, k)) if rows else {"why": "no request trace"}
            except SystemExit as e:
                c["fit"] = {"why": str(e)}
            v = os.path.join(d, f"vram_e4b_{tag}.txt")
            c["vram_used_mib"] = (int(open(v).read().split(",")[0]) if os.path.exists(v) and open(v).read().strip()
                                  else None)
            # free memory at ready and the bulk flush's bound, which is additive to the prefill graph's pool (#1131)
            g = (_load(d, f"health_e4b_{tag}_start.json") or {}).get("prefill_graph") or {}
            c["prefill_graph_at_ready"] = {k: g.get(k) for k in ("pool_mib", "free_after_mib", "bulk_flush_mib")}
            out[tag] = c
    return out


def reduce(d: str) -> dict:
    red, ident = _sib("sc2_reduce"), _sib("sc2_identity")
    out = {"gates": {}, "arms": {}, "predictions": {}}
    bad = {f"{a}_d{k}": routes_bad(_load(d, f"health_e4b_{a}_d{k}_start.json")) for a in ARMS for k in DRAWS}
    bad = {k: v for k, v in bad.items() if v}
    out["gates"]["routes"] = {"ok": not bad, "bad": bad}
    off_len = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("e4b_") and f.endswith(".json"):
            x = _load(d, f)
            if isinstance(x, dict) and isinstance(x.get("requests"), list):
                lens = {r.get("prompt_tokens") for r in x["requests"]}
                if lens != {T}:
                    off_len[f] = sorted(lens, key=str)
    out["gates"]["prompts"] = {"ok": not off_len, "bad": off_len}
    eng = {f"{a}_d{k}": engagement(a, _load(d, f"health_e4b_{a}_d{k}_end.json"), _prompts(d, a, k))
           for a in ARMS for k in DRAWS}
    out["gates"]["engaged"] = {"ok": not any(eng.values()), "why": {k: v for k, v in eng.items() if v}}
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
    for arm in ARMS:
        rows = {"serial": red.row([_load(d, f"e4b_{arm}_d{k}_serial.json") for k in DRAWS], "serial")}
        for r in RATES:
            rows[r] = red.row([_load(d, f"e4b_{arm}_d{k}_r{r}.json") for k in DRAWS], r)
        out["arms"][arm] = {"rows": {str(k): v for k, v in rows.items()}, "ceiling": red.ceiling(rows)}
    out["census"] = census(d)

    def s(arm, k, name):
        x = _load(d, f"e4b_{arm}_d{k}_{name}.json")
        return x["summary"] if x else None
    v = out["predictions"]
    stall = []
    for k in DRAWS:
        f_off = out["census"][f"off_d{k}"]["fit"].get("bucket_controlled", {}).get("stall_s_per_prefill")
        f_on = out["census"][f"on_d{k}"]["fit"].get("bucket_controlled", {}).get("stall_s_per_prefill")
        stall.append(round(f_on / f_off, 4) if f_off and f_on is not None and f_off > 0 else None)
    v["P1"] = ({"verdict": "UNREAD", "why": "a request trace is missing or did not fit", "stall_on_over_off": stall}
               if None in stall else
               {"verdict": "HOLDS" if max(stall) <= P1_MAX else "REFUTED", "stall_on_over_off": stall})
    ser = [(s("off", k, "serial"), s("on", k, "serial")) for k in DRAWS]
    if any(o is None or n is None or o["invalid"] or n["invalid"] for o, n in ser):
        v["P2"] = v["P3"] = {"verdict": "UNREAD", "why": "a serial run is missing or has INVALID requests"}
    else:
        r2 = [round(o["ttft_p50_s"] / n["ttft_p50_s"], 4) for o, n in ser]
        out["ttft_off_over_on"] = r2
        r3 = [round(n["tpot_p50_s"] / o["tpot_p50_s"], 4) for o, n in ser]
        v["P2"] = {"verdict": "HOLDS" if min(r2) >= P2_MIN else "REFUTED", "ttft_off_over_on": r2}
        v["P3"] = {"verdict": "HOLDS" if all(P3_BAND[0] <= x <= P3_BAND[1] for x in r3) else "REFUTED",
                   "tpot_on_over_off": r3}
    c_off, c_on = out["arms"]["off"]["ceiling"], out["arms"]["on"]["ceiling"]
    for q, m in (("P4", P4_MIN), ("P5", P5_MIN)):
        v[q] = {"verdict": "UNREAD" if c_on is None else ("HOLDS" if c_on >= m else "REFUTED"), "ceiling_on": c_on}
    pairs, miss = {}, False
    for r in RATES:
        for k in DRAWS:
            o, n = s("off", k, f"r{r}"), s("on", k, f"r{r}")
            if o is None or n is None:
                miss = True
                continue
            pairs[f"r{r}_d{k}"] = [o["attainment"], n["attainment"]]
    regress = {k: p for k, p in pairs.items() if p[1] < p[0] - P6_TOL}
    if miss or c_off is None or c_on is None:
        v["P6"] = {"verdict": "UNREAD", "pairs": pairs}
    else:
        v["P6"] = {"verdict": "HOLDS" if not regress and c_on >= c_off else "REFUTED", "pairs": pairs,
                   "regressions": regress, "ceilings": {"off": c_off, "on": c_on}}
    g = out["gates"]
    ttft_ok = bool(out.get("ttft_off_over_on")) and min(out["ttft_off_over_on"]) >= LICENCE_TTFT
    why = [n for n, ok in (("routes", g["routes"]["ok"]), ("engaged", g["engaged"]["ok"]), ("prompts", g["prompts"]["ok"]),
                           ("identity", g["identity"]["verdict"] == "IDENTICAL"), ("P6", v["P6"]["verdict"] == "HOLDS"),
                           (f"ttft_off_over_on >= {LICENCE_TTFT}", ttft_ok)) if not ok]
    out["licence"] = {"verdict": "DEFAULT_LICENSED" if not why else "NOT_LICENSED", "failed": why}
    out["void"] = not (g["routes"]["ok"] and g["engaged"]["ok"] and g["prompts"]["ok"])
    return out


# ------------------------------------------------------------------ self-test --

def self_test() -> int:
    import random
    import tempfile
    cases = []

    def summ(ttft, tpot, att, rate, invalid=0):
        return {"valid": 120 - invalid, "invalid": invalid, "errors": [], "ttft_p50_s": ttft, "ttft_p99_s": ttft * 2,
                "tpot_p50_s": tpot, "tpot_p99_s": tpot * 2, "e2el_p50_s": 1.0, "output_tok_s": 100.0, "attainment": att,
                "goodput_rps": att * rate, "achieved_rate": rate}

    def trace(stall, seed):
        """A request trace whose decode times carry ``stall`` per overlapping prefill (sc2c_census's model)."""
        rng = random.Random(seed)
        rows, t = [], 0.0
        for name, n in (("warm", 4), ("serial", 4), ("r1", 120), ("r2", 120), ("r4", 120), ("r8", 120)):
            for _ in range(n):
                t += rng.expovariate(2.0) if name.startswith("r") else 2.0
                rows.append({"first_token_at": t + 0.1, "out_len": rng.randint(64, 256)})
            t += 50.0
        for r in rows:
            r["finished_at"] = r["first_token_at"] + 0.005 * (r["out_len"] - 1)
        for _ in range(20):
            for r in rows:
                t0, t1 = r["first_token_at"], r["finished_at"]
                n_pf = sum(1 for q in rows if q is not r and t0 < q["first_token_at"] <= t1)
                r["finished_at"] = t0 + 0.005 * (r["out_len"] - 1) + stall * n_pf
        return rows

    def write(d, ttft_on=0.08, att_on=1.0, text_on="abc", flush_bulk=None, graph_on=True, routes_on=None,
              repeat_text="abc", stall_on=0.06, steps=True):
        def dump(name, obj):
            json.dump(obj, open(os.path.join(d, name), "w"))
        for k in DRAWS:
            for arm in ARMS:
                on = arm == "on"
                plan = [[0.0, i, 64] for i in range(4)]
                dump(f"e4b_{arm}_d{k}_serial.json",
                     {"mode": "serial", "prompts_sha256": "s", "plan": plan,
                      "requests": [{"valid": True, "text": (text_on if on else "abc") + str(i), "prompt_tokens": T}
                                   for i in range(4)],
                      "summary": summ(ttft_on if on else 0.165, 0.0045, 0.0, 0.0)})
                dump(f"e4b_{arm}_d{k}_warm.json", {"requests": [{"prompt_tokens": T}] * 4})
                for r in RATES:
                    att = (att_on if r <= 4 else 0.5) if on else (1.0 if r == 1 else 0.3)
                    dump(f"e4b_{arm}_d{k}_r{r}.json", {"requests": [{"prompt_tokens": T}] * 120,
                                                       "summary": summ(0.3, 0.02, att, r)})
                routes = dict(ROUTES, seen={"moe": {"int4_k19|gt256": 240, "int4_gemv|le256": 1200},
                                            "prefill_attn": {"flash": 240}})
                if on and routes_on:
                    routes.update(routes_on)
                dump(f"health_e4b_{arm}_d{k}_start.json",
                     {"prefill_routes": routes, "engine": {"chunk_tokens": T, "max_prefill_tokens_per_step": T}})
                n = 4 + 4 + 4 * 120 + (4 if (arm, k) == ("off", 1) else 0)
                fb = (flush_bulk if flush_bulk is not None else n) if on else 0
                kvb = ({"requested": True, "bulk": True, "flush_layers": n - fb, "flush_bulk": fb, "ready_layers": 0,
                        "ready_bulk": 0, "ready_at_flush": fb} if on else
                       {"requested": False, "bulk": False, "flush_layers": n, "flush_bulk": 0, "ready_layers": n,
                        "ready_bulk": 0, "ready_at_flush": 0})
                dump(f"health_e4b_{arm}_d{k}_end.json",
                     {"prefill_graph": {"status": "on" if graph_on else "refused", "requested": "auto", "T": T,
                                        "replays": n, "eager_chunks": 0}, KV_BLOCK: kvb})
                rows = trace(stall_on if on else 0.22, 10 * k + on)
                if (arm, k) == ("off", 1):                       # the serial repeat sits after serial in the trace
                    rows = rows[:8] + [{"first_token_at": 1e6 + i, "finished_at": 1e6 + i + 0.3, "out_len": 64}
                                       for i in range(4)] + rows[8:]
                with open(os.path.join(d, f"trace_e4b_{arm}_d{k}.jsonl"), "w") as f:
                    f.write("\n".join(json.dumps(r) for r in rows) + "\n")
                if steps:
                    with open(os.path.join(d, f"steps_e4b_{arm}_d{k}.jsonl"), "w") as f:
                        f.write(json.dumps({"step": 0, "step_ms": 160.0 if not on else 60.0, "prefill_replays": 1,
                                            "prefill_tokens": T, "seg": {"pf_flush": 150.0 if not on else 2.0},
                                            "gpu": {"pf_prep": 1.0, "pf_forward": 55.0, "pf_flush": 156.0}}) + "\n")
        dump("e4b_off_d1_serial_repeat.json",
             {"mode": "serial", "prompts_sha256": "s", "plan": [[0.0, i, 64] for i in range(4)],
              "requests": [{"valid": True, "text": repeat_text + str(i), "prompt_tokens": T} for i in range(4)]})
    with tempfile.TemporaryDirectory() as d:
        write(d)
        o = reduce(d)
        cases.append(("licensed", o["licence"]["verdict"] == "DEFAULT_LICENSED" and o["arms"]["on"]["ceiling"] == 4
                      and o["arms"]["off"]["ceiling"] == 1 and o["predictions"]["P1"]["verdict"] == "HOLDS"
                      and o["predictions"]["P2"]["verdict"] == "HOLDS" and o["predictions"]["P5"]["verdict"] == "HOLDS"
                      and o["census"]["on_d1"]["steps"]["prefill_steps"]["forward_device_ms_p50"] == 54.0))
    with tempfile.TemporaryDirectory() as d:
        write(d, text_on="abd")
        o = reduce(d)
        cases.append(("differs", o["gates"]["identity"]["verdict"] == "DIFFERS" and "identity" in o["licence"]["failed"]))
    with tempfile.TemporaryDirectory() as d:
        write(d, repeat_text="xyz")
        cases.append(("nondeterministic", reduce(d)["gates"]["identity"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:                     # one prompt took the per-layer flush on ON: VOID
        write(d, flush_bulk=4 + 4 + 4 * 120 - 1)
        o = reduce(d)
        cases.append(("not engaged", not o["gates"]["engaged"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:                     # the prefill graph refused: not the stack under test
        write(d, graph_on=False)
        o = reduce(d)
        cases.append(("graph refused", not o["gates"]["engaged"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:
        write(d, routes_on={"int4_prefill_above_256_rows": "mtile"})
        o = reduce(d)
        cases.append(("wrong route", not o["gates"]["routes"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:                     # resolved k19/flash, but the forward ran M-tile
        write(d, routes_on={"seen": {"moe": {"int4_mtile_captured|gt256": 240}, "prefill_attn": {"flash": 240}}})
        o = reduce(d)
        cases.append(("seen route", "seen.moe_gt256" in o["gates"]["routes"]["bad"].get("on_d1", {}) and o["void"]))
    with tempfile.TemporaryDirectory() as d:                     # 0.165 / 0.13 = 1.27: P2 fails, the licence passes
        write(d, ttft_on=0.13)
        o = reduce(d)
        cases.append(("P2 fails, licensed", o["predictions"]["P2"]["verdict"] == "REFUTED"
                      and o["licence"]["verdict"] == "DEFAULT_LICENSED"))
    with tempfile.TemporaryDirectory() as d:                     # 0.165 / 0.16 < 1.10: not licensed
        write(d, ttft_on=0.16)
        cases.append(("too slow", reduce(d)["licence"]["verdict"] == "NOT_LICENSED"))
    with tempfile.TemporaryDirectory() as d:                     # the stall barely moved: P1 refuted
        write(d, stall_on=0.18)
        o = reduce(d)
        cases.append(("stall kept", o["predictions"]["P1"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:
        write(d)
        x = json.load(open(os.path.join(d, "e4b_on_d1_r4.json")))
        x["requests"][7]["prompt_tokens"] = 511
        json.dump(x, open(os.path.join(d, "e4b_on_d1_r4.json"), "w"))
        o = reduce(d)
        cases.append(("prompt drift", not o["gates"]["prompts"]["ok"] and o["void"]))
    with tempfile.TemporaryDirectory() as d:
        write(d, att_on=0.2)
        o = reduce(d)
        cases.append(("regression", o["predictions"]["P6"]["verdict"] == "REFUTED" and o["predictions"]["P4"]["verdict"]
                      == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:                     # no step traces: the census says so, nothing gates
        write(d, steps=False)
        o = reduce(d)
        cases.append(("no step trace", o["licence"]["verdict"] == "DEFAULT_LICENSED"
                      and o["census"]["on_d2"]["steps"] == {"why": "no step trace"}))
    bad = [n for n, ok in cases if not ok]
    print(f"sc2c_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--engagement", nargs=2, metavar=("OFF_END.json", "ON_END.json"),
                   help="the proof's check: the two arms' end /health (counts not pinned to a plan)")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.engagement:
        why = engagement("off", json.load(open(a.engagement[0])), None) + \
            engagement("on", json.load(open(a.engagement[1])), None)
        print("SC2C_ENGAGEMENT " + json.dumps({"ok": not why, "why": why}), flush=True)
        return 0 if not why else 3
    out = reduce(a.dir)
    json.dump(out, open(a.out, "w"), indent=1)
    for name, g in out["gates"].items():
        print(f"SC2C_GATE {name} " + json.dumps(g)[:300])
    for q, x in out["predictions"].items():
        print(f"SC2C_{q} {x['verdict']} " + json.dumps({k: w for k, w in x.items() if k != 'verdict'})[:300])
    for arm, x in out["arms"].items():
        print(f"SC2C_ARM {arm} ceiling={x['ceiling']} rows=" + json.dumps({k: y['status'] for k, y in x['rows'].items()}))
    for tag, c in out["census"].items():
        print(f"SC2C_CENSUS {tag} " + json.dumps(c)[:600])
    print(f"SC2C_LICENCE {out['licence']['verdict']} failed={out['licence']['failed']} void={out['void']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
