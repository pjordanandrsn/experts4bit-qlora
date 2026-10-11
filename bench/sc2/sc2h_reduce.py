#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2h_reduce.py -- lane SC2h's registered rule (bench/sc2/SC2h-PREREG.md; #846): e4b serve_paged at 64 slots on the
release's defaults with P118's decode lookahead (``E4B_PAGED_DECODE_LOOKAHEAD=1``, arm ``la``) against without it (unset,
arm ``base``), from box O's run files.

Servers, each its own cold start, tagged ``e4b_<arm>_d<draw><rep>``: draw 1 runs la_a, base_a, base_b, la_b; draw 2 runs
base_a, la_a, la_b, base_b. Files in DIR, per server S:
  S_warm.json, S_burst.json, S_serial.json   4 warm (seed 999), the burst (64 at once, seed 998), serial 24 (seed = draw)
  S_r<R>.json                                the ladder: SC2f's 4, 8, 12, 13, 14, 16 and the knee rates; seed draw x 100 + R
  S_k<R>_<j>.json                            the knee block: R in the knee rates; j in 1..4; seed 10000 x draw + 100 x R + j
  health_S_start.json / _end.json, trace_S.jsonl, steps_S.jsonl, vram_S.txt
and once per run: ``knee_rates.json`` (the rates the box resolved) beside ``sc2f_verdict.json`` (the SC2f verdict it
resolved them from). Every server of a draw runs the same seeds: the two servers of one arm in a draw are a replicate.

**The knee rates** (``knee_rates``, registered before SC2f reads): from SC2f's verdict, the ``on`` arm's mean attainment
over its four servers at each ladder rate. r* = the lowest rate whose mean is below 0.95; the block is r* - 1, r*, r* + 1.
If no rate is below 0.95: the top rate and the two above it. If the lowest rate is below: the bottom three ladder rates.
If SC2f reads VOID or UNREAD, or any of its ``on`` rows is not VALID: no rates, and box O does not launch.

**Gates, per server** (a failure names the server and VOIDs the reading):
- ROUTES (``sc2c_reduce.routes_bad``) and SLOTS at 64 slots, buckets up to 64 (``sc2e_reduce.slots_bad``, arm s64a);
- DEFAULTS: the start /health reads SC2f's ``on`` arm (``sc2f_reduce.flips_bad``): P130's two defaults engaged;
- LOOKAHEAD: the start /health's ``engine.decode_lookahead`` is the arm's (true on ``la``, false on ``base``), and in the
  step trace every decode-only step of an ``la`` server carries the lookahead note, and no step of a ``base`` server does;
- ENGAGED (``sc2e_reduce.engagement``, arm s64a) on the end /health, with every request the server admitted counted;
- PROMPTS: every request reports 512 prompt tokens;
- KNEE: ``knee_rates.json`` equals the rule run on ``sc2f_verdict.json``, whose sha256 it records.

**Statistics.**
- Rows, ceilings and knees are SC2f's: a row is one arm at one ladder rate over its four servers (UNREAD / INVALID /
  VALID; it meets capacity iff every server attains >= 0.95); an arm's ceiling is the largest ladder rate whose row is
  VALID and meets capacity; a server's knee is its mean attainment over the knee block's 15 runs.
- A server's **step** is its mean ``step_ms`` over its decode-only one-piece bucket-64 steps (at least ``MIN_STEPS``).
  A ``base`` server's **host gap** is the mean over the same steps of ``gpu.dec_prep + step_ms - gpu.dec_issue``: the
  time the GPU is not replaying, the lookahead's ceiling. (Under the lookahead those events mark an enqueue, not a
  replay, so the host gap is read on ``base`` only.)
- Per draw: **saving** = the mean step of its two ``base`` servers minus that of its two ``la`` servers; **step noise** =
  the larger same-arm replicate difference; **step bar** = max(``STEP_MIN``, step noise); **ceiling** = the mean host gap
  of its two ``base`` servers. **gain** = the mean knee of its two ``la`` servers minus that of its two ``base`` servers;
  **noise** = the larger same-arm replicate difference; **bar** = max(0.03, noise); **headroom** = 1 - the mean knee of
  its ``base`` servers.

**Outcome**, decoupled from Q1-Q6:
- VOID if a server fails a gate; UNREAD if a needed number is missing;
- TOKENS_DIFFER if, in any draw and replicate, the ``la`` server's serial texts differ from the ``base`` server's. The
  lookahead promises the synchronous path's tokens; it fails that, and the fallback is the dispatch fix;
- REGRESSED if ``la``'s ceiling is below ``base``'s, or at 4 or 8 req/s an ``la`` server attains below the draw's lower
  ``base`` server minus 0.05, or gain <= -bar in both draws, or saving <= -step bar in both draws;
- FASTER if saving >= step bar AND gain >= bar, in both draws. The reducer licenses nothing: the read recommends the
  default-on flip for graph servers to the maintainer, as its own PR with the receipts;
- KNEE_SATURATED if the outcome would be neither and headroom < bar in either draw;
- STEP_ONLY if saving >= step bar in both draws (the knee did not clear): the saving is recorded, the knob stays opt-in;
- FLAT otherwise.
``outcome.above_ceiling`` flags a draw whose saving exceeds its ceiling plus the step noise: the lookahead would then
have changed device time, which it should not.

**Predictions** (per draw; HOLDS / MISSED / UNREAD; no outcome reads them):
  Q1  saving in [1.2, 2.7] ms.
  Q2  saving <= ceiling + step noise.
  Q3  serial p50: TPOT la/base in [0.85, 1.00]; TTFT la/base in [0.97, 1.03].
  Q4  gain in [+0.01, +0.12].
  Q5  la's ceiling >= base's (over the reading).
  Q6  TPOT p50 at 8 and 12 req/s la/base in [0.65, 0.95].

**Reported, no bar:** TPOT p99 at 8 and 12 req/s la/base beside the no-regression rule (the lookahead can trade tail for
mean); per server the census (``sc2e_census``), VRAM, ``free_after_mib``; per draw the text agreement under load.

  sc2h_reduce.py --dir DIR [--out verdict.json]
  sc2h_reduce.py --knee-rule SC2F_VERDICT.json [--write DIR]   (box O's resolution; exit 1 when it does not launch)
  sc2h_reduce.py --lookahead HEALTH.json ARM [--proof]           (box O's start check; exit 1 on a departure)
  sc2h_reduce.py --steps STEPS.jsonl ARM                         (the proof: the step trace's lookahead notes)
  sc2h_reduce.py --engagement END.json ... --arms ARM ...
  sc2h_reduce.py --self-test
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("la", "base")
DRAWS = (1, 2)
ORDER = {1: ("la_a", "base_a", "base_b", "la_b"), 2: ("base_a", "la_a", "la_b", "base_b")}
SC2F_LADDER = (4, 8, 12, 13, 14, 16)
KNEE_J = (1, 2, 3, 4)
SLOT_ARM = "s64a"
T = 512
CAP_SHARE = 0.95
KNEE_MIN = 0.03
STEP_MIN = 0.25                   # ms: a tenth of SC2e's 2.47 ms host gap at bucket 64 (the census's lever (c))
NOREG_TOL = 0.05
NOREG_RATES = (4, 8)
MIN_STEPS = 100
Q1_BAND = (1.2, 2.7)
Q3_TPOT, Q3_TTFT = (0.85, 1.00), (0.97, 1.03)
Q4_BAND = (0.01, 0.12)
Q6_BAND = (0.65, 0.95)
LA_ENV = "E4B_PAGED_DECODE_LOOKAHEAD"


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
    return [(f"e4b_{s.rsplit('_', 1)[0]}_d{k}{s[-1]}", s.rsplit("_", 1)[0], k) for k in DRAWS for s in ORDER[k]]


# ------------------------------------------------------------------ the knee rates --

def knee_rates(sc2f: dict):
    """The knee block from SC2f's verdict: ``(rates, branch)``, or ``(None, why)`` when SC2h does not launch."""
    verdict = ((sc2f or {}).get("outcome") or {}).get("verdict")
    if verdict in (None, "VOID", "UNREAD"):
        return None, f"SC2f reads {verdict}: SC2h does not launch"
    rows = (((sc2f.get("arms") or {}).get("on") or {}).get("rows")) or {}
    if sorted(int(r) for r in rows) != list(SC2F_LADDER):
        return None, f"SC2f's on rows {sorted(rows)} are not its ladder {list(SC2F_LADDER)}"
    means = {}
    for r in SC2F_LADDER:
        x = rows[str(r)]
        if x.get("status") != "VALID" or not x.get("attainment"):
            return None, f"SC2f's on row at {r} req/s is {x.get('status')}"
        means[r] = statistics.fmean(x["attainment"])
    below = [r for r in SC2F_LADDER if means[r] < CAP_SHARE]
    if not below:
        top = SC2F_LADDER[-1]
        return (top, top + 1, top + 2), "never below 0.95: the top rate and the two above it"
    if below[0] == SC2F_LADDER[0]:
        return tuple(SC2F_LADDER[:3]), "below 0.95 at the lowest rate: the bottom three"
    r = below[0]
    return (r - 1, r, r + 1), f"r* = {r}: the block centred on it"


def knee_bad(d) -> tuple:
    """``(rates, why)``: the run's knee rates, and why they do not read as the rule run on the staged SC2f verdict."""
    kr, p = _load(d, "knee_rates.json"), os.path.join(d, "sc2f_verdict.json")
    if kr is None or not os.path.exists(p):
        return None, "knee_rates.json or sc2f_verdict.json missing"
    raw = open(p, "rb").read()
    rates, branch = knee_rates(json.loads(raw))
    sha = hashlib.sha256(raw).hexdigest()
    if rates is None:
        return None, branch
    if list(kr.get("rates") or []) != list(rates) or kr.get("sc2f_verdict_sha256") != sha:
        return None, f"knee_rates.json {kr} against the rule's {list(rates)} on sha256 {sha[:12]}"
    return tuple(rates), None


def ladder_of(knee):
    return tuple(sorted(set(SC2F_LADDER) | set(knee)))


# ------------------------------------------------------------------ gates --

def lookahead_bad(health, arm, proof=False) -> dict:
    """Why a server's start /health does not read as its arm: the lookahead setting, and SC2f's ``on`` defaults."""
    bad = {}
    eng = (health or {}).get("engine") or {}
    if eng.get("decode_lookahead") is not (arm == "la"):
        bad["engine.decode_lookahead"] = eng.get("decode_lookahead", "<missing>")
    bad.update(_sib("sc2f_reduce").flips_bad(health, "on", proof=proof))
    return bad


def _decode_only(rows):
    return [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens") and r.get("bucket")]


def steps_bad(rows, arm) -> list:
    """The step trace's lookahead notes: on every decode-only step of an la server, on no step of a base server."""
    if not rows:
        return ["no step trace"]
    dec = _decode_only(rows)
    if arm == "la":
        miss = sum(1 for r in dec if r.get("lookahead") != 1)
        return [f"{miss} of {len(dec)} decode-only steps without the lookahead note"] if miss or not dec else []
    noted = sum(1 for r in rows if r.get("lookahead"))
    return [f"{noted} steps carry the lookahead note"] if noted else []


# ------------------------------------------------------------------ statistics --

def _names(tag, knee):
    return ([f"{tag}_{x}" for x in ("warm", "burst", "serial")] + [f"{tag}_r{r}" for r in ladder_of(knee)]
            + [f"{tag}_k{r}_{j}" for r in knee for j in KNEE_J])


def _prompts(d, tag, knee):
    return sum(len(x["requests"]) for x in (_load(d, n + ".json") for n in _names(tag, knee)) if x)


def _plan(d, tag, knee):
    out = []
    for n in _names(tag, knee):
        x = _load(d, n + ".json")
        if x:
            out.append((n[len(tag) + 1:], len(x["requests"])))
    return out


def _summ(d, tag, w):
    x = _load(d, f"{tag}_{w}.json")
    return x["summary"] if x else None


def _att(s):
    return None if s is None or s["invalid"] else s["attainment"]


def knee(d, tag, rates):
    a = [_att(_summ(d, tag, w)) for r in rates for w in [f"r{r}"] + [f"k{r}_{j}" for j in KNEE_J]]
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
    if any(v["status"] in ("UNREAD", "INVALID") for v in rows.values()):
        return None
    ok = [r for r, v in rows.items() if v["meets_capacity"]]
    return max(ok) if ok else 0


def step64(rows, host=False):
    """The mean step_ms (or, with ``host``, the mean host gap) over decode-only one-piece bucket-64 steps."""
    s = [r for r in _decode_only(rows or []) if r["bucket"] == 64 and r.get("dec_pieces") == 1]
    if len(s) < MIN_STEPS:
        return None
    if host:
        g = [r for r in s if "dec_issue" in (r.get("gpu") or {}) and "dec_prep" in r["gpu"]]
        if len(g) < MIN_STEPS:
            return None
        return round(statistics.fmean(r["gpu"]["dec_prep"] + r["step_ms"] - r["gpu"]["dec_issue"] for r in g), 4)
    return round(statistics.fmean(r["step_ms"] for r in s), 4)


def _texts(x):
    return [r.get("text") if r.get("valid") else None for r in x["requests"]] if x else None


def tokens(d) -> dict:
    """Per draw and replicate: whether the la server's serial texts equal the base server's (None if one is missing)."""
    out = {}
    for k in DRAWS:
        for rep in "ab":
            a, b = _texts(_load(d, f"e4b_la_d{k}{rep}_serial.json")), _texts(_load(d, f"e4b_base_d{k}{rep}_serial.json"))
            out[f"d{k}{rep}"] = None if a is None or b is None else (a == b and None not in a)
    return out


def _agreement(a, b):
    if not a or not b or len(a["requests"]) != len(b["requests"]):
        return None
    eq = [x.get("valid") and y.get("valid") and x.get("text") == y.get("text") for x, y in zip(a["requests"], b["requests"])]
    return round(sum(eq) / len(eq), 4) if eq else None


def census(d, knee) -> dict:
    cen = _sib("sc2e_census")
    out = {}
    for tag, arm, k in servers():
        c = {}
        steps = _jsonl(d, f"steps_{tag}.jsonl")
        c["steps"] = cen.steps(steps, SLOT_ARM) if steps else {"why": "no step trace"}
        trace = _jsonl(d, f"trace_{tag}.jsonl")
        try:
            c["per_rate"] = cen.per_rate(trace, _plan(d, tag, knee)) if trace else {"why": "no request trace"}
        except SystemExit as e:
            c["per_rate"] = {"why": str(e)}
        v = os.path.join(d, f"vram_{tag}.txt")
        txt = open(v).read().strip() if os.path.exists(v) else ""
        c["vram_used_mib"] = int(txt.split(",")[0]) if txt else None
        g = (_load(d, f"health_{tag}_start.json") or {}).get("prefill_graph") or {}
        c["at_ready"] = {x: g.get(x) for x in ("status", "pool_mib", "free_after_mib", "bulk_flush_mib")}
        c["graph_stats_end"] = ((_load(d, f"health_{tag}_end.json") or {}).get("engine") or {}).get("graph_stats")
        out[tag] = c
    for k in DRAWS:
        for rep in "ab":
            la, base = f"e4b_la_d{k}{rep}", f"e4b_base_d{k}{rep}"
            out[la]["text_agreement_vs_base"] = {
                w: _agreement(_load(d, f"{base}_{w}.json"), _load(d, f"{la}_{w}.json"))
                for w in ["serial"] + [f"r{r}" for r in ladder_of(knee)]}
    return out


def _band(xs, lo, hi):
    if any(x is None for x in xs):
        return "UNREAD"
    return "HOLDS" if all(lo <= x <= hi for x in xs) else "MISSED"


def _mean(xs):
    return None if not xs or any(x is None for x in xs) else statistics.fmean(xs)


def reduce(d: str) -> dict:
    c_red, e_red = _sib("sc2c_reduce"), _sib("sc2e_reduce")
    out = {"gates": {}, "arms": {}, "knee": {}, "step": {}, "predictions": {}}
    srv = servers()
    rates, kwhy = knee_bad(d)
    out["knee_rates"] = {"rates": list(rates) if rates else None, "why": kwhy}
    kn_rates = rates or ()
    start = {t: _load(d, f"health_{t}_start.json") for t, _, _ in srv}
    steps = {t: _jsonl(d, f"steps_{t}.jsonl") for t, _, _ in srv}
    routes = {t: c_red.routes_bad(start[t]) for t, _, _ in srv}
    slots = {t: e_red.slots_bad(start[t], SLOT_ARM) for t, _, _ in srv}
    la = {t: {**lookahead_bad(start[t], a), **({"steps": steps_bad(steps[t], a)} if steps_bad(steps[t], a) else {})}
          for t, a, _ in srv}
    eng = {t: e_red.engagement(SLOT_ARM, _load(d, f"health_{t}_end.json"), _prompts(d, t, kn_rates)) for t, _, _ in srv}
    prm = {}
    for t, _, _ in srv:
        off = {}
        for n in _names(t, kn_rates):
            x = _load(d, n + ".json")
            lens = {r.get("prompt_tokens") for r in x["requests"]} if x else {T}
            if lens != {T}:
                off[n] = sorted(lens, key=str)
        prm[t] = off
    for name, g in (("routes", routes), ("slots", slots), ("lookahead", la), ("engaged", eng), ("prompts", prm)):
        out["gates"][name] = {"ok": not any(g.values()), "bad": {k: v for k, v in g.items() if v}}
    out["gates"]["knee"] = {"ok": kwhy is None, "bad": {"knee_rates": kwhy} if kwhy else {}}
    void = sorted(t for t, _, _ in srv if routes[t] or slots[t] or la[t] or eng[t] or prm[t])
    if kwhy:
        void.append("knee_rates")
    lad = ladder_of(kn_rates) if rates else SC2F_LADDER
    for arm in ARMS:
        rows = {r: row(d, arm, r) for r in lad}
        out["arms"][arm] = {"rows": {str(r): v for r, v in rows.items()}, "ceiling": ceiling(rows)}
    kn = {t: knee(d, t, kn_rates) if rates else None for t, _, _ in srv}
    st = {t: step64(steps[t]) for t, _, _ in srv}
    hg = {t: step64(steps[t], host=True) for t, a, _ in srv if a == "base"}
    gain, noise, bar, headroom, saving, snoise, sbar, ceil = ({} for _ in range(8))
    for k in DRAWS:
        L, B = [kn[f"e4b_la_d{k}{x}"] for x in "ab"], [kn[f"e4b_base_d{k}{x}"] for x in "ab"]
        if None in L + B:
            gain[k] = noise[k] = bar[k] = headroom[k] = None
        else:
            gain[k] = round(statistics.fmean(L) - statistics.fmean(B), 4)
            noise[k] = round(max(abs(L[0] - L[1]), abs(B[0] - B[1])), 4)
            bar[k] = round(max(KNEE_MIN, noise[k]), 4)
            headroom[k] = round(1 - statistics.fmean(B), 4)
        sL, sB = [st[f"e4b_la_d{k}{x}"] for x in "ab"], [st[f"e4b_base_d{k}{x}"] for x in "ab"]
        hB = [hg[f"e4b_base_d{k}{x}"] for x in "ab"]
        if None in sL + sB + hB:
            saving[k] = snoise[k] = sbar[k] = ceil[k] = None
        else:
            saving[k] = round(statistics.fmean(sB) - statistics.fmean(sL), 4)
            snoise[k] = round(max(abs(sL[0] - sL[1]), abs(sB[0] - sB[1])), 4)
            sbar[k] = round(max(STEP_MIN, snoise[k]), 4)
            ceil[k] = round(statistics.fmean(hB), 4)
    out["knee"] = {"server": kn, "gain": gain, "noise": noise, "bar": bar, "headroom": headroom}
    out["step"] = {"server": st, "host_gap_base": hg, "saving": saving, "noise": snoise, "bar": sbar, "ceiling": ceil}
    tok = out["tokens"] = tokens(d)
    out["census"] = census(d, kn_rates)

    def per_draw(arm, k, fn):
        return _mean([fn(f"e4b_{arm}_d{k}{x}") for x in "ab"])

    def ratio(k, fn):
        a, b = per_draw("la", k, fn), per_draw("base", k, fn)
        return round(a / b, 4) if a is not None and b else None

    def summ_get(w, m):
        return lambda t: (_summ(d, t, w) or {}).get(m) if _summ(d, t, w) and not _summ(d, t, w)["invalid"] else None
    v = out["predictions"]
    v["Q1"] = {"verdict": _band([saving[k] for k in DRAWS], *Q1_BAND), "saving_ms": [saving[k] for k in DRAWS],
               "band": list(Q1_BAND)}
    q2 = [None if saving[k] is None else saving[k] <= ceil[k] + snoise[k] for k in DRAWS]
    v["Q2"] = {"verdict": "UNREAD" if None in q2 else ("HOLDS" if all(q2) else "MISSED"),
               "saving_ms": [saving[k] for k in DRAWS], "ceiling_ms": [ceil[k] for k in DRAWS]}
    tp = [ratio(k, summ_get("serial", "tpot_p50_s")) for k in DRAWS]
    tt = [ratio(k, summ_get("serial", "ttft_p50_s")) for k in DRAWS]
    q3 = _band(tp, *Q3_TPOT), _band(tt, *Q3_TTFT)
    v["Q3"] = {"verdict": "UNREAD" if "UNREAD" in q3 else ("HOLDS" if q3 == ("HOLDS", "HOLDS") else "MISSED"),
               "tpot_la_over_base": tp, "ttft_la_over_base": tt}
    v["Q4"] = {"verdict": _band([gain[k] for k in DRAWS], *Q4_BAND), "gain": [gain[k] for k in DRAWS]}
    cla, cba = out["arms"]["la"]["ceiling"], out["arms"]["base"]["ceiling"]
    v["Q5"] = {"verdict": "UNREAD" if cla is None or cba is None else ("HOLDS" if cla >= cba else "MISSED"),
               "ceilings": {"la": cla, "base": cba}}
    t6 = [ratio(k, summ_get(f"r{r}", "tpot_p50_s")) for r in (8, 12) for k in DRAWS]
    v["Q6"] = {"verdict": _band(t6, *Q6_BAND), "tpot_la_over_base_r8_r12": t6}
    noreg, nr_why = True, []
    for r in NOREG_RATES:
        for k in DRAWS:
            L = [_att(_summ(d, f"e4b_la_d{k}{x}", f"r{r}")) for x in "ab"]
            B = [_att(_summ(d, f"e4b_base_d{k}{x}", f"r{r}")) for x in "ab"]
            if None in L + B:
                noreg = None
                continue
            low = [a for a in L if a < min(B) - NOREG_TOL]
            if low:
                nr_why.append(f"r{r} d{k}: la {L} against base {B}")
    if noreg is not None:
        noreg = not nr_why
    out["no_regression"] = {"holds": noreg, "why": nr_why,
                            "tpot_p99_la_over_base_r8_r12": [ratio(k, summ_get(f"r{r}", "tpot_p99_s"))
                                                             for r in (8, 12) for k in DRAWS]}
    why, above = [], [k for k in DRAWS if saving[k] is not None and saving[k] > ceil[k] + snoise[k]]
    needed = [cla, cba, noreg] + list(gain.values()) + list(saving.values()) + list(tok.values())
    if void:
        verdict = "VOID"
    elif any(x is None for x in needed):
        verdict = "UNREAD"
    elif not all(tok.values()):
        verdict = "TOKENS_DIFFER"
        why.append(f"serial texts differ: {tok}; the fallback is the dispatch fix (one wakeup a step)")
    elif (cla < cba or not noreg or all(gain[k] <= -bar[k] for k in DRAWS)
          or all(saving[k] <= -sbar[k] for k in DRAWS)):
        verdict = "REGRESSED"
        ctx = f"knee gain {gain}, bar {bar}; step saving {saving}, step bar {sbar}"
        if cla < cba:
            why.append(f"la's ceiling {cla} below base's {cba} ({ctx})")
        if not noreg:
            why.append(f"an la server below base at 4 or 8 req/s: {nr_why} ({ctx})")
        if all(gain[k] <= -bar[k] for k in DRAWS):
            why.append(f"the knee lost in both draws ({ctx})")
        if all(saving[k] <= -sbar[k] for k in DRAWS):
            why.append(f"the bucket-64 step slower in both draws ({ctx})")
    elif all(saving[k] >= sbar[k] and gain[k] >= bar[k] for k in DRAWS):
        verdict = "FASTER"
        why.append("recommend the default-on flip for graph servers to the maintainer, as its own PR with the receipts")
    elif any(headroom[k] < bar[k] for k in DRAWS):
        verdict = "KNEE_SATURATED"
        why.append(f"headroom {headroom} below the bar {bar} in a draw: the knee block could not show the gain "
                   f"(gain {gain}); the step saving {saving} is recorded")
    elif all(saving[k] >= sbar[k] for k in DRAWS):
        verdict = "STEP_ONLY"
        why.append(f"the step saving {saving} clears {sbar}; the knee gain {gain} does not clear {bar}: the knob stays "
                   f"opt-in and the saving is recorded")
    else:
        verdict = "FLAT"
    out["outcome"] = {"verdict": verdict, "void": void, "why": why, "above_ceiling": above}
    return out


def engagement_cli(ends, arms) -> int:
    e_red = _sib("sc2e_reduce")
    why = []
    for p, arm in zip(ends, arms):
        h = json.load(open(p)) if os.path.exists(p) else None
        why += [f"{arm}: {w}" for w in e_red.engagement(SLOT_ARM, h, None)]
    print("SC2H_ENGAGEMENT " + json.dumps({"ok": not why, "why": why}), flush=True)
    return 1 if why else 0


def knee_rule_cli(path, write) -> int:
    raw = open(path, "rb").read() if os.path.exists(path) else b"{}"
    rates, why = knee_rates(json.loads(raw))
    rec = {"rates": list(rates) if rates else None, "branch": why, "sc2f_verdict_sha256": hashlib.sha256(raw).hexdigest()}
    print("SC2H_KNEE " + json.dumps(rec), flush=True)
    if rates and write:
        with open(os.path.join(write, "knee_rates.json"), "w") as f:
            json.dump(rec, f)
        with open(os.path.join(write, "sc2f_verdict.json"), "wb") as f:
            f.write(raw)
    return 0 if rates else 1


# ------------------------------------------------------------------ self-test --

def _sc2f_verdict(means, verdict="FLAT", status=None):
    rows = {str(r): {"status": (status or {}).get(r, "VALID"), "attainment": [means[r]] * 4} for r in SC2F_LADDER}
    return {"outcome": {"verdict": verdict}, "arms": {"on": {"rows": rows, "ceiling": 12}}}


def _fixture(d, *, knee_att=None, ceil=None, att=None, la_bad=None, step_ms=None, host_ms=None, serial=None, tpot=None,
             texts=None, notes=None, admitted=None, prompt_tokens=T, missing=(), invalid=(), sc2f=None, rates_edit=None,
             tpot99=None):
    """Box O's run files for all eight servers. Defaults: SC2f's on crosses 0.95 at 13 req/s (knee 12, 13, 14); base at
    the knee 0.90, la 0.95; both ceilings 12; base's bucket-64 step 15.6 ms with a 2.4 ms host gap, la's 13.6 ms; every
    gate passing; identical serial texts."""
    c_red, f_red = _sib("sc2c_reduce"), _sib("sc2f_reduce")
    sc2f = sc2f or _sc2f_verdict({4: 1.0, 8: 1.0, 12: 0.99, 13: 0.93, 14: 0.85, 16: 0.6})
    raw = json.dumps(sc2f).encode()
    rates, _ = knee_rates(sc2f)
    rates = tuple(rates or ())
    with open(os.path.join(d, "sc2f_verdict.json"), "wb") as f:
        f.write(raw)
    kr = {"rates": list(rates), "sc2f_verdict_sha256": hashlib.sha256(raw).hexdigest()}
    kr.update(rates_edit or {})
    json.dump(kr, open(os.path.join(d, "knee_rates.json"), "w"))
    knee_att = {"la": 0.95, "base": 0.90, **(knee_att or {})}
    ceil = {"la": 12, "base": 12, **(ceil or {})}
    step_ms = {"la": 13.6, "base": 15.6, **(step_ms or {})}
    host_ms = {"base": 2.4, **(host_ms or {})}
    serial = {"la": (0.040, 0.0040), "base": (0.040, 0.0044), **(serial or {})}
    tpot = {"la": 0.027, "base": 0.030, **(tpot or {})}
    tpot99 = {"la": 0.040, "base": 0.042, **(tpot99 or {})}

    def summ(ttft, tp, a, rate, inv=0, tp99=None):
        return {"valid": 120 - inv, "invalid": inv, "errors": [], "ttft_p50_s": ttft, "ttft_p99_s": ttft * 2,
                "tpot_p50_s": tp, "tpot_p99_s": tp99 if tp99 is not None else tp * 2, "e2el_p50_s": 1.0,
                "output_tok_s": 100.0, "attainment": a, "goodput_rps": a * rate, "achieved_rate": rate}

    def dump(name, obj):
        json.dump(obj, open(os.path.join(d, name), "w"))
    lad = ladder_of(rates)
    for tag, arm, k in servers():
        if tag in missing:
            continue

        def reqs(n, txt="x"):
            return [{"valid": True, "text": f"{txt}{i}", "prompt_tokens": prompt_tokens} for i in range(n)]
        dump(f"{tag}_warm.json", {"requests": reqs(4)})
        dump(f"{tag}_burst.json", {"requests": reqs(64)})
        dump(f"{tag}_serial.json", {"requests": reqs(24, (texts or {}).get(tag, "s")), "summary": summ(*serial[arm], 0.0, 0.0)})
        for r in lad:
            a = 1.0 if r <= ceil[arm] else 0.7
            a = (att or {}).get((tag, r), a)
            dump(f"{tag}_r{r}.json", {"requests": reqs(120), "summary": summ(
                0.05, tpot[arm], a, r, 1 if (tag, f"r{r}") in invalid else 0, tpot99[arm])})
        for r in rates:
            for j in KNEE_J:
                dump(f"{tag}_k{r}_{j}.json", {"requests": reqs(120), "summary": summ(0.1, 0.03, knee_att.get(tag, knee_att[arm]), r)})
        n = (admitted or {}).get(tag, 4 + 64 + 24 + 120 * (len(lad) + len(rates) * len(KNEE_J)))
        w = f_red.WANT_FLIPS["on"]
        seen = {"moe": {"int4_k19|gt256": 384, "int4_k19|le256": 720}, "prefill_attn": {"flash": 240},
                "moe_k19_dispatch": {"chained|lean|gt256": 240, "one_launch|lean|gt256": 144, "one_launch|lean|le256": 720},
                "prefill_folds": {f: 48 for f in f_red.FOLDS}}
        pr = dict(c_red.ROUTES, seen=seen, prefill_glue_env=None, prefill_lean_dispatch_env=None, prefill_lean_dispatch="auto")
        lev = {"kv": {"scratch_slots": 64, "pool_mib": 6678.8}, "prefill_glue": dict(w["glue"])}
        b = [1, 2, 4, 8, 16, 32, 64]
        eng = {"chunk_tokens": T, "max_prefill_tokens_per_step": T, "max_seqs": 64, "kv_slots": 64, "buckets": b,
               "buckets_requested": "auto", "graph_status": {str(x): "graph" for x in b}, "decode_lookahead": arm == "la"}
        eng.update((la_bad or {}).get(tag) or {})
        dump(f"health_{tag}_start.json", {"prefill_routes": pr, "engine": eng, "levers": lev,
                                          "prefill_graph": {"status": "on", "pool_mib": 428, "free_after_mib": 3700,
                                                            "bulk_flush_mib": 216}})
        dump(f"health_{tag}_end.json", {
            "prefill_graph": {"status": "on", "T": T, "replays": n, "eager_chunks": 0},
            "kv_bookkeeping": {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0,
                               "ready_bulk": 0, "ready_at_flush": n},
            "engine": {"graph_stats": {str(x): {"replays": 5, "eager_steps": 0, "rows": 5, "pad_rows": 0} for x in b}}})
        open(os.path.join(d, f"vram_{tag}.txt"), "w").write("28430, 32607\n")
        sm = step_ms.get(tag, step_ms[arm])
        hm = host_ms.get(tag, host_ms.get(arm, 0.5))
        note = (notes or {}).get(tag, 1 if arm == "la" else None)
        rows = []
        for i in range(MIN_STEPS + 20):
            r = {"step": i, "t": i * 0.02, "step_ms": sm, "decode_rows": 60, "bucket": 64, "dec_pieces": 1,
                 "seg": {"dec_sync": 1.0}, "gpu": {"dec_prep": 0.5, "dec_issue": sm - hm + 0.5}}
            if note is not None:
                r["lookahead"] = note
            rows.append(r)
        rows += [{"step": 500 + i, "t": 5.0 + i * 0.05, "step_ms": 41.0, "prefill_tokens": T, "prefill_replays": 1,
                  "decode_rows": 0, "seg": {"pf_forward": 33.0},
                  "gpu": {"pf_prep": 0.2, "pf_forward": 33.2, "pf_flush": 33.6}} for i in range(30)]
        with open(os.path.join(d, f"steps_{tag}.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(x) for x in rows) + "\n")


def self_test() -> int:
    import tempfile
    cases = []

    def run(edit=None, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            if edit:
                edit(d)
            return reduce(d)
    o = run()
    cases.append(("faster", o["outcome"]["verdict"] == "FASTER" and o["knee"]["gain"] == {1: 0.04, 2: 0.04}
                  and o["step"]["saving"] == {1: 2.0, 2: 2.0} and o["step"]["bar"] == {1: 0.25, 2: 0.25}
                  and o["step"]["ceiling"] == {1: 2.4, 2: 2.4} and o["knee_rates"]["rates"] == [12, 13, 14]))
    cases.append(("predictions hold", all(o["predictions"][q]["verdict"] == "HOLDS" for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))))
    cases.append(("tokens identical in every draw and replicate", o["tokens"] == {"d1a": True, "d1b": True, "d2a": True, "d2b": True}))
    # the knee-rate rule, every branch
    f = _sc2f_verdict
    cases.append(("knee rule: r* = 13", knee_rates(f({4: 1, 8: 1, 12: 0.99, 13: 0.93, 14: 0.8, 16: 0.6}))[0] == (12, 13, 14)))
    cases.append(("knee rule: r* = 16", knee_rates(f({4: 1, 8: 1, 12: 1, 13: 0.99, 14: 0.96, 16: 0.9}))[0] == (15, 16, 17)))
    cases.append(("knee rule: never below -> 16, 17, 18", knee_rates(f({r: 0.99 for r in SC2F_LADDER}))[0] == (16, 17, 18)))
    cases.append(("knee rule: below at 4 -> 4, 8, 12", knee_rates(f({r: 0.9 for r in SC2F_LADDER}))[0] == (4, 8, 12)))
    cases.append(("knee rule: r* = 8", knee_rates(f({4: 1, 8: 0.94, 12: 0.9, 13: 0.9, 14: 0.9, 16: 0.9}))[0] == (7, 8, 9)))
    cases.append(("knee rule: the MEAN of four servers decides", knee_rates(
        {"outcome": {"verdict": "FLAT"}, "arms": {"on": {"rows": {**{str(r): {"status": "VALID", "attainment": [1.0] * 4}
                                                                     for r in SC2F_LADDER},
                                                                  "13": {"status": "VALID", "attainment": [1.0, 1.0, 1.0, 0.85]}}}}})[0]
                  == (16, 17, 18)))
    for v in ("VOID", "UNREAD"):
        cases.append((f"knee rule: SC2f {v} does not launch", knee_rates(f({r: 0.99 for r in SC2F_LADDER}, verdict=v))[0] is None))
    cases.append(("knee rule: an INVALID SC2f row does not launch",
                  knee_rates(f({r: 0.99 for r in SC2F_LADDER}, status={13: "INVALID"}))[0] is None))
    o = run(sc2f=_sc2f_verdict({r: 0.99 for r in SC2F_LADDER}))
    cases.append(("knee 16, 17, 18 joins the ladder", o["knee_rates"]["rates"] == [16, 17, 18]
                  and sorted(o["arms"]["la"]["rows"], key=int) == ["4", "8", "12", "13", "14", "16", "17", "18"]
                  and o["outcome"]["verdict"] == "FASTER"))
    o = run(rates_edit={"rates": [13, 14, 15]})
    cases.append(("KNEE: rates not the rule's voids", o["outcome"]["verdict"] == "VOID" and "knee_rates" in o["outcome"]["void"]))
    o = run(rates_edit={"sc2f_verdict_sha256": "0" * 64})
    cases.append(("KNEE: another SC2f verdict voids", o["outcome"]["verdict"] == "VOID"))
    # gates
    o = run(la_bad={"e4b_la_d1a": {"decode_lookahead": False}})
    cases.append(("LOOKAHEAD: an la server without it voids", o["outcome"]["void"] == ["e4b_la_d1a"]))
    o = run(la_bad={"e4b_base_d2b": {"decode_lookahead": True}})
    cases.append(("LOOKAHEAD: a base server with it voids", o["outcome"]["void"] == ["e4b_base_d2b"]))
    o = run(notes={"e4b_la_d2a": 0})
    cases.append(("LOOKAHEAD: la steps without the note void", o["outcome"]["void"] == ["e4b_la_d2a"]))
    o = run(notes={"e4b_base_d1b": 1})
    cases.append(("LOOKAHEAD: base steps with the note void", o["outcome"]["void"] == ["e4b_base_d1b"]))

    def no_glue(d):
        p = os.path.join(d, "health_e4b_base_d1a_start.json")
        h = json.load(open(p))
        h["levers"]["prefill_glue"] = {"mode": "0", "source": "explicit"}
        json.dump(h, open(p, "w"))
    o = run(edit=no_glue)
    cases.append(("DEFAULTS: a server off P130's defaults voids", o["outcome"]["void"] == ["e4b_base_d1a"]))
    o = run(prompt_tokens=511)
    cases.append(("PROMPTS: drift voids every server", len(o["outcome"]["void"]) == 8))
    o = run(admitted={"e4b_la_d1b": 5})
    cases.append(("ENGAGED: replays short of the admitted requests", o["outcome"]["void"] == ["e4b_la_d1b"]))
    # outcomes
    o = run(texts={"e4b_la_d2b": "t"})
    cases.append(("tokens differ", o["outcome"]["verdict"] == "TOKENS_DIFFER" and o["tokens"]["d2b"] is False))
    o = run(knee_att={"la": 0.91})
    cases.append(("step only: the saving clears, the knee does not", o["outcome"]["verdict"] == "STEP_ONLY"
                  and o["knee"]["gain"] == {1: 0.008, 2: 0.008} and o["predictions"]["Q4"]["verdict"] == "MISSED"))
    o = run(knee_att={"la": 0.90}, step_ms={"la": 15.5})
    cases.append(("flat", o["outcome"]["verdict"] == "FLAT" and o["predictions"]["Q1"]["verdict"] == "MISSED"))
    o = run(ceil={"la": 14, "base": 14}, knee_att={"la": 0.99, "base": 0.975})
    cases.append(("saturated", o["outcome"]["verdict"] == "KNEE_SATURATED" and o["knee"]["headroom"] == {1: 0.02, 2: 0.02}))
    o = run(step_ms={"e4b_la_d1a": 14.6})
    cases.append(("a noisy la replicate raises the step bar", o["step"]["noise"][1] == 1.0 and o["step"]["bar"][1] == 1.0
                  and o["step"]["saving"][1] == 1.5 and o["outcome"]["verdict"] == "FASTER"))
    o = run(step_ms={"e4b_la_d1a": 16.6}, knee_att={"la": 0.90})
    cases.append(("a replicate noisier than the saving: FLAT", o["step"]["bar"][1] == 3.0 and o["step"]["saving"][1] == 0.5
                  and o["outcome"]["verdict"] == "FLAT"))
    o = run(ceil={"la": 8})
    cases.append(("regressed: la's ceiling below base's", o["outcome"]["verdict"] == "REGRESSED"))
    o = run(att={("e4b_la_d2a", 8): 0.80})
    cases.append(("regressed: an la server at 8 req/s", o["outcome"]["verdict"] == "REGRESSED"
                  and not o["no_regression"]["holds"]))
    o = run(step_ms={"la": 16.4}, knee_att={"la": 0.90})
    cases.append(("regressed: the bucket-64 step slower in both draws", o["outcome"]["verdict"] == "REGRESSED"))
    o = run(step_ms={"la": 12.0})
    cases.append(("above the ceiling is flagged", o["outcome"]["above_ceiling"] == [1, 2]
                  and o["predictions"]["Q2"]["verdict"] == "MISSED" and o["outcome"]["verdict"] == "FASTER"))
    o = run(tpot99={"la": 0.060})
    cases.append(("p99 TPOT reported beside the no-regression rule", o["no_regression"]["tpot_p99_la_over_base_r8_r12"]
                  == [1.4286] * 4 and o["outcome"]["verdict"] == "FASTER"))
    o = run(missing=("e4b_base_d2a",))
    cases.append(("a missing server: VOID (no health)", o["outcome"]["verdict"] == "VOID"))

    def drop_knee_run(d):
        os.remove(os.path.join(d, "e4b_la_d1b_k13_2.json"))
    o = run(edit=drop_knee_run)
    cases.append(("a missing knee run on an engaged server... is short of its admitted count", o["outcome"]["verdict"] == "VOID"))
    o = run(edit=drop_knee_run, admitted={"e4b_la_d1b": 4 + 64 + 24 + 120 * 17})
    cases.append(("a missing knee run: UNREAD", o["outcome"]["verdict"] == "UNREAD" and o["knee"]["gain"][1] is None))
    o = run(invalid={("e4b_base_d2a", "r13")})
    cases.append(("an invalid ladder run: the ceiling is unread", o["arms"]["base"]["ceiling"] is None
                  and o["outcome"]["verdict"] == "UNREAD"))
    h = {"engine": {"decode_lookahead": True}, "prefill_routes": {"prefill_glue_env": None, "prefill_lean_dispatch_env": None,
                                                                  "prefill_lean_dispatch": "auto", "seen": {"moe": {"nf4_grouped|gt256": 9}}},
         "levers": {"prefill_glue": {"mode": "0", "source": "default-off"}}}
    cases.append(("proof: Granite's la server", lookahead_bad(h, "la", proof=True) == {} and lookahead_bad(h, "la") != {}
                  and lookahead_bad(h, "base", proof=True) != {}))
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2H_REDUCE self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--knee-rule", metavar="SC2F_VERDICT")
    ap.add_argument("--write", metavar="DIR")
    ap.add_argument("--lookahead", nargs=2, metavar=("HEALTH", "ARM"))
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--steps", nargs=2, metavar=("STEPS", "ARM"))
    ap.add_argument("--engagement", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.knee_rule:
        return knee_rule_cli(a.knee_rule, a.write)
    if a.lookahead:
        path, arm = a.lookahead
        h = json.load(open(path)) if os.path.exists(path) else None
        bad = lookahead_bad(h, arm, proof=a.proof) if arm in ARMS else {"arm": arm}
        print("SC2H_LOOKAHEAD " + json.dumps({"arm": arm, "proof": a.proof, "ok": not bad, "bad": bad}), flush=True)
        return 1 if bad else 0
    if a.engagement:
        return engagement_cli(a.engagement, a.arms)
    if a.steps:
        path, arm = a.steps
        rows = _jsonl(os.path.dirname(path) or ".", os.path.basename(path))
        why = steps_bad(rows, arm) if arm in ARMS else [f"arm {arm}"]
        print("SC2H_STEPS " + json.dumps({"arm": arm, "ok": not why, "why": why}), flush=True)
        return 1 if why else 0
    if not a.dir:
        ap.error("--dir DIR | --knee-rule VERDICT | --lookahead HEALTH ARM | --engagement ... | --self-test")
    out = reduce(a.dir)
    text = json.dumps(out, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print("SC2H_VERDICT " + json.dumps({k: out[k] for k in ("outcome", "knee_rates")}, default=str), flush=True)
    print(json.dumps({k: out[k] for k in ("knee", "step", "tokens", "no_regression", "predictions")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
