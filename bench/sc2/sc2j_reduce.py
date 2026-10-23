#!/usr/bin/env python3
"""sc2j_reduce.py -- lane SC2j's registered rule (bench/sc2/SC2j-PREREG.md; #846, serving roadmap lever G).

Files in DIR (box Q's ``sc2/``):
- per server TAG = ``e4b_<arm>_d<D>`` (arm in ca_a, sh, ca_b; D in 1, 2): ``health_<TAG>_start.json``,
  ``health_<TAG>_end.json``, ``stats_<TAG>_end.json``; the driver's runs ``<TAG>_{warm,burst,serial}.json``,
  ``<TAG>_r<R>.json`` (Q2) and ``<TAG>_q3r<R>.json`` (Q3, R with "." as "p"); ``stats_<TAG>_q3r<R>_{before,after}.json``;
- ``contract.json`` (``sc2j_contract.py verdict``);
- ``quality/verdict.json`` (``p117_reduce.py``) and ``quality/plans.json`` (P117's split plans at its table width).

Gates (a failure names the server or the phase): ROUTES, ENGAGED, PROMPTS, POOL, SLOTS and DETERMINISM/IDENTITY VOID or
UNREAD as registered; WIDTH UNREAD; CONTRACT VOID; QUALITY NOT_LICENSED (COST) or UNREAD (plans differ / no reading).

Rows (``sc2_reduce.row`` over both draws, per server slot ca_a / sh / ca_b); the Q2 ceiling of ``ca`` holds for both
``ca`` slots (the lower of the two); ``sh``'s over its own rows. Outcome: SHARED_LICENSED, REGRESSED, NOT_LICENSED or
UNREAD, with the clause named.

    python sc2j_reduce.py --dir DIR --out verdict_sc2j.json
    python sc2j_reduce.py --pool HEALTH ARM          box Q's start check (rc 1 when the arm does not read)
    python sc2j_reduce.py --self-test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc2_identity  # noqa: E402
import sc2_reduce  # noqa: E402
import sc2c_reduce  # noqa: E402

Q2_RATES = (1, 2, 4, 8, 12, 16)
Q3_RATES = (0.5, 1, 2, 4)
DRAWS = (1, 2)
SLOT_ARMS = ("ca_a", "sh", "ca_b")
T = 512
Q3_PROMPT = 2048
ATT_TOL_IN, ATT_TOL_REGRESS, Q3_TOL_MIN, TOKS_BAR = 0.02, 0.05, 0.05, 1.25
PLAN_BUCKETS = (16, 32, 64)


def rtag(rate) -> str:
    return str(rate).replace(".", "p")


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def tag(arm, draw) -> str:
    return f"e4b_{arm}_d{draw}"


def pool_bad(health, arm) -> dict:
    """POOL: the start /health reads its arm (claim-all with no sizing record; shared with its rows and derivation)."""
    e = (health or {}).get("engine") or {}
    want = "shared" if arm == "sh" else "claim-all"
    bad = {}
    if e.get("kv_pool") != want:
        bad["engine.kv_pool"] = e.get("kv_pool", "<missing>")
    sz = e.get("kv_pool_sizing")
    if want == "claim-all":
        if sz is not None:
            bad["engine.kv_pool_sizing"] = sz
    else:
        if not isinstance(sz, dict) or any(sz.get(k) is None for k in ("pool_rows", "claim_all_rows", "kv_blocks")) \
                or not sz.get("derivation"):
            bad["engine.kv_pool_sizing"] = sz if sz is not None else "<missing>"
    return bad


def auto_buckets(width: int) -> list:
    out, b = [], 1
    while b < width:
        out.append(b)
        b *= 2
    return out + [width]


def slots_bad(health) -> dict:
    """SLOTS: the buckets are auto's for the server's own width, every one captured, scratch = the width."""
    e = (health or {}).get("engine") or {}
    kv = ((health or {}).get("levers") or {}).get("kv") or {}
    m = e.get("max_seqs")
    if not isinstance(m, int) or m < 1:
        return {"engine.max_seqs": m if m is not None else "<missing>"}
    b = auto_buckets(m)
    bad = {}
    if e.get("kv_slots") != m:
        bad["engine.kv_slots"] = e.get("kv_slots", "<missing>")
    if e.get("buckets") != b:
        bad["engine.buckets"] = e.get("buckets", "<missing>")
    gs = e.get("graph_status")
    if not isinstance(gs, dict) or set(gs) != {str(x) for x in b} or any(v != "graph" for v in gs.values()):
        bad["engine.graph_status"] = gs if gs is not None else "<missing>"
    if kv.get("scratch_slots") != m:
        bad["levers.kv.scratch_slots"] = kv.get("scratch_slots", "<missing>")
    return bad


def run_names(t) -> list:
    return ([f"{t}_{x}" for x in ("warm", "burst", "serial")] + [f"{t}_r{r}" for r in Q2_RATES]
            + [f"{t}_q3r{rtag(r)}" for r in Q3_RATES])


def admitted(d, t) -> int:
    return sum(len(x["requests"]) for x in (_load(d, n + ".json") for n in run_names(t)) if x)


def engagement_bad(t, end, prompts) -> list:
    """ENGAGED (SC2e's check at the server's own width): the prefill graph on, replays = every prompt admitted (a
    2,048-token Q3 prompt is four chunks: its first replays, its later chunks run eagerly as ``later_chunk``), bulk KV
    for every prompt, the widest bucket replayed, no eager decode step."""
    if end is None:
        return [f"{t}: no end /health"]
    why = []
    g = end.get("prefill_graph") or {}
    er = g.get("eager_reasons") or {}
    other = {k: v for k, v in er.items() if k != "later_chunk" and v}
    if g.get("status") != "on" or g.get("T") != T or (prompts is not None and g.get("replays") != prompts) or other:
        why.append(f"{t}: prefill_graph {g!r} (want on, T {T}, replays {prompts}, eager only later_chunk)")
    k = end.get("kv_bookkeeping")
    if not isinstance(k, dict) or "flush_bulk" not in k:
        why.append(f"{t}: no built kv_bookkeeping block ({k!r})")
    else:
        n = prompts if prompts is not None else k.get("flush_bulk", 0)
        if k.get("bulk") is not True or k.get("flush_bulk") != n or k.get("flush_layers") or not n:
            why.append(f"{t}: kv_bookkeeping {k!r} (want bulk, flush_bulk {n})")
    e = end.get("engine") or {}
    top = str(max(e.get("buckets") or [0]))
    gs = e.get("graph_stats")
    if not isinstance(gs, dict) or (gs.get(top) or {}).get("replays", 0) < 1:
        why.append(f"{t}: bucket {top} never replayed ({gs!r})")
    elif any(v.get("eager_steps") for v in gs.values()):
        why.append(f"{t}: eager decode steps {gs!r}")
    return why


def prompts_bad(d, t) -> list:
    why = []
    for n in run_names(t):
        x = _load(d, n + ".json")
        if not x:
            continue
        want = Q3_PROMPT if "_q3r" in n else T
        pt = sorted({r.get("prompt_tokens") for r in x.get("requests") or []}, key=str)
        if pt != [want]:
            why.append(f"{n}: prompt_tokens {pt} (want {want})")
    return why


def waits(stats):
    return ((stats or {}).get("scheduler") or {}).get("kv_blocks_waits")


def pressure(d) -> dict:
    """PRESSURE: sh's kv_blocks_waits rises over the 4 req/s Q3 rung, in both draws."""
    out = {}
    for D in DRAWS:
        t = tag("sh", D)
        a, b = (_load(d, f"stats_{t}_q3r{rtag(4)}_{x}.json") for x in ("before", "after"))
        wa, wb = waits(a), waits(b)
        out[D] = None if wa is None or wb is None else wb - wa
    return {"rise": out, "holds": all(v is not None and v > 0 for v in out.values())}


def rows_for(d, arm, rates, prefix) -> dict:
    return {r: sc2_reduce.row([_load(d, f"{tag(arm, D)}_{prefix}{rtag(r)}.json") for D in DRAWS], "rate")
            for r in rates}


def ceiling(rows: dict):
    if any(rows[r]["status"] in ("UNREAD", "INVALID") for r in Q2_RATES):
        return None
    ok = [r for r in Q2_RATES if rows[r]["status"] == "VALID" and rows[r].get("meets_capacity")]
    return max(ok) if ok else 0


def att(row, i):
    try:
        return row["draws"][i]["attainment"]
    except (KeyError, IndexError, TypeError):
        return None


def toks(row, i):
    try:
        return row["draws"][i]["output_tok_s"]
    except (KeyError, IndexError, TypeError):
        return None


def plans_bad(d) -> list:
    """QUALITY's plans: P117's split plans at its table width equal the servers' recorded plans at B = 16, 32, 64."""
    q = _load(d, "quality/plans.json")
    if not q:
        return ["no quality/plans.json"]
    why = []
    for D in DRAWS:
        for arm in SLOT_ARMS:
            h = _load(d, f"health_{tag(arm, D)}_start.json")
            plans = (((h or {}).get("levers") or {}).get("attn_split_cap") or {}).get("plans") or {}
            for B in PLAN_BUCKETS:
                srv = (plans.get(str(B)) or {}).get("n_split")
                if srv is None:
                    continue                      # a bucket this server does not capture (claim-all at 32)
                if q.get(str(B)) != srv:
                    why.append(f"{tag(arm, D)} B={B}: server n_split {srv} != P117's {q.get(str(B))}")
    return why


def reduce(d: str) -> dict:
    out = {"gates": {}, "rows": {}, "reported": {}}
    starts = {(a, D): _load(d, f"health_{tag(a, D)}_start.json") for a in SLOT_ARMS for D in DRAWS}
    ends = {(a, D): _load(d, f"health_{tag(a, D)}_end.json") for a in SLOT_ARMS for D in DRAWS}
    void, unread = [], []
    for (a, D), h in starts.items():
        t = tag(a, D)
        if h is None:
            unread.append(f"{t}: no start /health (the deadline or a failed start)")
            continue
        r = sc2c_reduce.routes_bad(h)
        if r:
            void.append(f"ROUTES {t}: {r}")
        p = pool_bad(h, "sh" if a == "sh" else "ca")
        if p:
            void.append(f"POOL {t}: {p}")
        s = slots_bad(h)
        if s:
            void.append(f"SLOTS {t}: {s}")
        void += [f"ENGAGED {x}" for x in engagement_bad(t, ends[(a, D)], admitted(d, t))]
        void += [f"PROMPTS {x}" for x in prompts_bad(d, t)]
    widths = {k: ((h or {}).get("engine") or {}).get("max_seqs") for k, h in starts.items()}
    out["widths"] = {f"{a}_d{D}": w for (a, D), w in widths.items()}
    width_bad = [D for D in DRAWS if not all(isinstance(widths.get((x, D)), int) for x in SLOT_ARMS)
                 or not widths[("sh", D)] > max(widths[("ca_a", D)], widths[("ca_b", D)])]
    out["gates"]["WIDTH"] = "ok" if not width_bad else f"UNREAD in draws {width_bad}"
    c = _load(d, "contract.json")
    if not c or c.get("verdict") != "OK":
        void.append(f"CONTRACT: {(c or {}).get('why') or 'no contract.json'}")
    det = sc2_identity.compare(_load(d, f"{tag('ca_a', 1)}_serial.json") or {},
                               _load(d, f"{tag('ca_b', 1)}_serial.json") or {})
    ident = {D: sc2_identity.compare(_load(d, f"{tag('sh', D)}_serial.json") or {},
                                     _load(d, f"{tag('ca_a', D)}_serial.json") or {}) for D in DRAWS}
    det_ok = det.get("verdict") == "IDENTICAL"
    ident_ok = all(v.get("verdict") == "IDENTICAL" for v in ident.values())
    out["gates"]["DETERMINISM"] = det.get("verdict")
    out["gates"]["IDENTITY"] = {D: v.get("verdict") for D, v in ident.items()}
    qv = _load(d, "quality/verdict.json")
    pb = plans_bad(d)
    if pb:
        quality = "UNREAD"
        out["gates"]["QUALITY"] = f"UNREAD: {pb}"
    elif not qv or qv.get("verdict") in (None, "NO_READING", "VOID"):
        quality = "UNREAD"
        out["gates"]["QUALITY"] = f"UNREAD: {(qv or {}).get('verdict') or 'no quality/verdict.json'}"
    else:
        quality = "OK" if qv.get("verdict") == "AT_PARITY" else "COST"
        out["gates"]["QUALITY"] = qv.get("verdict")
    out["gates"]["VOID"] = void or "none"
    pr = pressure(d)
    out["gates"]["PRESSURE"] = pr
    q2 = {a: rows_for(d, a, Q2_RATES, "r") for a in SLOT_ARMS}
    q3 = {a: rows_for(d, a, Q3_RATES, "q3r") for a in SLOT_ARMS}
    out["rows"] = {"Q2": q2, "Q3": q3}
    ceil = {a: ceiling(q2[a]) for a in SLOT_ARMS}
    ca_ceil = None if ceil["ca_a"] is None or ceil["ca_b"] is None else min(ceil["ca_a"], ceil["ca_b"])
    out["ceilings"] = {"ca": ca_ceil, "sh": ceil["sh"], "ca_a": ceil["ca_a"], "ca_b": ceil["ca_b"]}

    def lower_ca(rows, r, i):
        xs = [att(rows["ca_a"][r], i), att(rows["ca_b"][r], i)]
        return None if any(x is None for x in xs) else min(xs)

    def noise(rows, r, i):
        xs = [att(rows["ca_a"][r], i), att(rows["ca_b"][r], i)]
        return None if any(x is None for x in xs) else abs(xs[0] - xs[1])

    clauses, regress = {}, []
    if ca_ceil is not None and ceil["sh"] is not None:
        clauses["ceiling_higher"] = ceil["sh"] > ca_ceil
        loss = []
        for r in [x for x in Q2_RATES if x <= ca_ceil]:
            for i in range(len(DRAWS)):
                s, lo = att(q2["sh"][r], i), lower_ca(q2, r, i)
                if s is None or lo is None or s < lo - ATT_TOL_IN:
                    loss.append(f"r{r} d{i + 1}: sh {s} vs lower ca {lo}")
        clauses["no_q2_loss"] = not loss
        out["reported"]["q2_loss"] = loss
        reg2 = [r for r in Q2_RATES if r <= ca_ceil and all(
            (att(q2["sh"][r], i) is not None and lower_ca(q2, r, i) is not None
             and att(q2["sh"][r], i) < lower_ca(q2, r, i) - ATT_TOL_REGRESS) for i in range(len(DRAWS)))]
        if reg2:
            regress.append(f"Q2 at {reg2}")
        tk = []
        for i in range(len(DRAWS)):
            s = toks(q2["sh"].get(ceil["sh"]) or {}, i) if ceil["sh"] else None
            c_ = [toks(q2[a].get(ca_ceil) or {}, i) for a in ("ca_a", "ca_b")] if ca_ceil else [None]
            c_ = None if any(x is None for x in c_) else max(c_)
            tk.append(None if s is None or c_ is None else s / c_)
        clauses["toks_at_knee"] = all(x is not None and x >= TOKS_BAR for x in tk)
        out["reported"]["toks_ratio"] = tk
    q3loss, reg3 = [], []
    for r in Q3_RATES:
        both = []
        for i in range(len(DRAWS)):
            s, lo, n = att(q3["sh"][r], i), lower_ca(q3, r, i), noise(q3, r, i)
            if s is None or lo is None:
                q3loss.append(f"q3 r{r} d{i + 1}: missing")
                both.append(False)
                continue
            if s < lo - max(Q3_TOL_MIN, n):
                q3loss.append(f"q3 r{r} d{i + 1}: sh {s} vs lower ca {lo} (noise {n})")
            both.append(s < lo - ATT_TOL_REGRESS)
        if all(both):
            reg3.append(r)
    clauses["q3_read"] = pr["holds"]
    clauses["no_q3_loss"] = pr["holds"] and not q3loss
    out["reported"]["q3_loss"] = q3loss
    if reg3 and pr["holds"]:
        regress.append(f"Q3 at {reg3}")
    out["clauses"] = clauses
    out["reported"]["agreement"] = {f"r{r}_d{D}": _agree(d, tag("sh", D), tag("ca_a", D), f"r{r}")
                                    for r in Q2_RATES for D in DRAWS}

    if void:
        out["verdict"], out["why"] = "VOID", void
    elif unread or width_bad or not det_ok or not ident_ok or ca_ceil is None or ceil["sh"] is None:
        out["verdict"] = "UNREAD"
        out["why"] = unread + ([f"WIDTH in draws {width_bad}"] if width_bad else []) + \
            ([] if det_ok else [f"DETERMINISM {det.get('verdict')}"]) + \
            ([] if ident_ok else [f"IDENTITY {out['gates']['IDENTITY']}"]) + \
            ([] if ca_ceil is not None and ceil["sh"] is not None else ["a ceiling is UNREAD"])
    elif regress:
        out["verdict"], out["why"] = "REGRESSED", regress
    elif quality == "COST":
        out["verdict"], out["why"] = "NOT_LICENSED", ["QUALITY: P117's width check reads COST"]
    elif quality == "UNREAD":
        out["verdict"], out["why"] = "NOT_LICENSED", [f"QUALITY {out['gates']['QUALITY']}"]
    elif all(clauses.get(k) for k in ("ceiling_higher", "no_q2_loss", "toks_at_knee", "q3_read", "no_q3_loss")):
        out["verdict"], out["why"] = "SHARED_LICENSED", []
    else:
        out["verdict"] = "NOT_LICENSED"
        out["why"] = [k for k in ("ceiling_higher", "no_q2_loss", "toks_at_knee", "q3_read", "no_q3_loss")
                      if not clauses.get(k)]
    return out


def _agree(d, ta, tb, w):
    a, b = _load(d, f"{ta}_{w}.json"), _load(d, f"{tb}_{w}.json")
    if not a or not b or len(a["requests"]) != len(b["requests"]):
        return None
    eq = [x.get("valid") and y.get("valid") and x.get("text") == y.get("text") for x, y in zip(a["requests"], b["requests"])]
    return round(sum(eq) / len(eq), 4) if eq else None


# ---------------------------------------------------------------------------------------------------- self-test --

def _health(arm, width, D=1, end=False, prompts=10):
    e = {"max_seqs": width, "kv_slots": width, "buckets": auto_buckets(width),
         "graph_status": {str(b): "graph" for b in auto_buckets(width)}, "chunk_tokens": T,
         "max_prefill_tokens_per_step": T, "kv_pool": "shared" if arm == "sh" else "claim-all",
         "kv_pool_sizing": ({"pool_rows": 12000, "claim_all_rows": 16448, "kv_blocks": 11936,
                             "derivation": {"why": "x"}} if arm == "sh" else None),
         "graph_stats": {str(b): {"replays": 3, "eager_steps": 0} for b in auto_buckets(width)}}
    plans = {str(b): {"n_split": {16: 5, 32: 2, 64: 1}.get(b, 8)} for b in auto_buckets(width)}
    h = {"engine": e, "levers": {"kv": {"scratch_slots": width}, "attn_split_cap": {"plans": plans}},
         "prefill_routes": dict(sc2c_reduce.ROUTES, seen={"moe": {"int4_k19|gt256": 1}, "prefill_attn": {"flash": 1}})}
    if end:
        h["prefill_graph"] = {"status": "on", "T": T, "replays": prompts, "eager_chunks": 0,
                              "eager_reasons": {"later_chunk": 6}}
        h["kv_bookkeeping"] = {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": prompts,
                               "ready_layers": 0, "ready_bulk": 0, "ready_at_flush": prompts}
    return h


def _summary(att_, tok=1000.0):
    return {"valid": 10, "invalid": 0, "ttft_p50_s": 0.1, "ttft_p99_s": 0.5, "tpot_p50_s": 0.03, "tpot_p99_s": 0.05,
            "e2el_p50_s": 1.0, "output_tok_s": tok, "attainment": att_, "goodput_rps": 1.0, "achieved_rate": 1.0}


def _fixture(d, *, ca_att=None, sh_att=None, q3_ca=None, q3_sh=None, widths=None, pressure_rise=(5, 5),
             contract="OK", quality="AT_PARITY", plans=None, sh_serial="t", pool_override=None, toks_sh=2000.0,
             q3_sh_d2=None):
    os.makedirs(os.path.join(d, "quality"), exist_ok=True)
    widths = widths or {"ca_a": 32, "sh": 64, "ca_b": 32}
    ca_att = ca_att or {1: 1, 2: 1, 4: 1, 8: 1, 12: 0.7, 16: 0.4}
    sh_att = sh_att or {1: 1, 2: 1, 4: 1, 8: 1, 12: 0.97, 16: 0.6}
    q3_ca = q3_ca or {0.5: 0.9, 1: 0.8, 2: 0.6, 4: 0.3}
    q3_sh = q3_sh or {0.5: 0.9, 1: 0.8, 2: 0.6, 4: 0.3}
    w = lambda name, obj: json.dump(obj, open(os.path.join(d, name), "w"))  # noqa: E731
    for D in DRAWS:
        for a in SLOT_ARMS:
            t = tag(a, D)
            nreq = 0
            for x in ("warm", "burst", "serial"):
                reqs = [{"valid": True, "prompt_tokens": T, "text": sh_serial if (a == "sh" and x == "serial") else "t"}]
                nreq += len(reqs)
                w(f"{t}_{x}.json", {"requests": reqs, "summary": _summary(1.0), "mode": "serial",
                                    "prompts_sha256": "p", "plan": [[0.0, 0, 64]]})
            for r in Q2_RATES:
                at = (sh_att if a == "sh" else ca_att)[r]
                tk = (toks_sh if a == "sh" else 1000.0) * (r / 8)
                w(f"{t}_r{r}.json", {"requests": [{"valid": True, "prompt_tokens": T, "text": "t"}], "summary": _summary(at, tk)})
                nreq += 1
            for r in Q3_RATES:
                at = ((q3_sh_d2 if (D == 2 and q3_sh_d2) else q3_sh) if a == "sh" else q3_ca)[r]
                w(f"{t}_q3r{rtag(r)}.json", {"requests": [{"valid": True, "prompt_tokens": Q3_PROMPT, "text": "t"}],
                                              "summary": _summary(at)})
                nreq += 1
                w(f"stats_{t}_q3r{rtag(r)}_before.json", {"scheduler": {"kv_blocks_waits": 10}})
                rise = pressure_rise[D - 1] if (a == "sh" and r == 4) else 0
                w(f"stats_{t}_q3r{rtag(r)}_after.json", {"scheduler": {"kv_blocks_waits": 10 + rise}})
            hs = _health("sh" if a == "sh" else "ca", widths[a], D)
            if pool_override and (a, D) in pool_override:
                hs["engine"]["kv_pool"] = pool_override[(a, D)]
            w(f"health_{t}_start.json", hs)
            w(f"health_{t}_end.json", _health("sh" if a == "sh" else "ca", widths[a], D, end=True, prompts=nreq))
    w("contract.json", {"verdict": contract, "why": [] if contract == "OK" else ["x"]})
    w("quality/verdict.json", {"verdict": quality})
    w("quality/plans.json", plans or {"16": 5, "32": 2, "64": 1})


def self_test() -> int:
    import tempfile
    cases = []

    def case(name, want, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            got = reduce(d)
            cases.append((name, got["verdict"], want, got.get("why")))

    case("licensed", "SHARED_LICENSED")
    case("ceiling not higher", "NOT_LICENSED", sh_att={1: 1, 2: 1, 4: 1, 8: 1, 12: 0.7, 16: 0.4})
    case("q2 loss below ca's ceiling", "NOT_LICENSED", sh_att={1: 1, 2: 1, 4: 0.96, 8: 1, 12: 0.97, 16: 0.6})
    case("q2 regressed", "REGRESSED", sh_att={1: 1, 2: 1, 4: 0.9, 8: 0.97, 12: 0.97, 16: 0.6})
    case("tokens/s at the knee short", "NOT_LICENSED", toks_sh=600.0)
    case("no pressure", "NOT_LICENSED", pressure_rise=(5, 0))
    case("q3 loss in one draw", "NOT_LICENSED", q3_sh_d2={0.5: 0.9, 1: 0.8, 2: 0.6, 4: 0.2})
    case("q3 regressed", "REGRESSED", q3_sh={0.5: 0.9, 1: 0.8, 2: 0.5, 4: 0.3})
    case("width not wider", "UNREAD", widths={"ca_a": 64, "sh": 64, "ca_b": 64})
    case("contract void", "VOID", contract="VOID")
    case("quality cost", "NOT_LICENSED", quality="COST")
    case("quality no reading", "NOT_LICENSED", quality="NO_READING")
    case("plans differ", "NOT_LICENSED", plans={"16": 4, "32": 2, "64": 1})
    case("identity differs", "UNREAD", sh_serial="other")
    case("pool wrong", "VOID", pool_override={("sh", 1): "claim-all"})
    fails = 0
    for name, got, want, why in cases:
        ok = got == want
        fails += not ok
        print(f"SC2J_REDUCE self-test {'ok ' if ok else 'BAD'} {name}: {got} (want {want})" + ("" if ok else f" {why}"))
    # the bucket rule and the pool check, directly
    assert auto_buckets(32) == [1, 2, 4, 8, 16, 32] and auto_buckets(48) == [1, 2, 4, 8, 16, 32, 48]
    assert pool_bad(_health("sh", 64), "sh") == {} and pool_bad(_health("ca", 32), "ca") == {}
    assert pool_bad(_health("ca", 32), "sh")
    print(f"SC2J_REDUCE self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--pool", nargs=2, metavar=("HEALTH", "ARM"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.pool:
        h = json.load(open(a.pool[0]))
        bad = {**pool_bad(h, "sh" if a.pool[1] == "sh" else "ca"), **slots_bad(h)}
        e = h.get("engine") or {}
        print("SC2J_POOL " + json.dumps({"arm": a.pool[1], "ok": not bad, "bad": bad, "max_seqs": e.get("max_seqs"),
                                         "kv_pool_sizing": e.get("kv_pool_sizing")}), flush=True)
        return 1 if bad else 0
    if a.dir:
        v = reduce(a.dir)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(v, f, indent=1)
        print("SC2J_VERDICT " + json.dumps({k: v.get(k) for k in ("verdict", "why", "widths", "ceilings", "clauses")}))
        return 0
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
