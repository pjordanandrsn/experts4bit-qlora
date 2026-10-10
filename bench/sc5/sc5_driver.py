#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_driver.py -- lane SC5 (#1478 item 4): the CLOSED-LOOP request driver for the same-box head-to-head.

C workers each send their next request as soon as their last one finishes, until N requests have been sent. The
request, its streaming parse, TTFT / TPOT and validity are SC2's, imported unchanged from ``sc2_driver.py``. That
file is pinned by ``bench/sc1/staged.sha256`` and ``bench/ra/source-pins.json``, so it is never edited for SC5.
Every engine receives the same request: greedy, streamed, ``ignore_eos``, exactly ``max_tokens``, plus its SC2 profile.

The plan is deterministic in (n, seed, prompts, max_tokens): request i's prompt index is drawn from ``seed``. Workers
take requests in index order from one shared queue, so the same requests are offered in the same order to every engine.

**Warm-up.** The first C requests sent (indices 0..C-1, the first wave) are marked ``warm`` and excluded from every
statistic. Every request must still be VALID: an invalid warm-up request is an invalid cell.

**Summary, over the counted requests:**
- TTFT and TPOT p50 and p95, and end-to-end p50;
- output throughput: the counted requests' output tokens over the steady window, which runs from the first counted
  request's send to the last counted request's end;
- the largest number of requests in flight at once, which must equal C.

  sc5_driver.py run --base URL --model NAME --prompts prompts.json --concurrency C --n N --seed S --max-tokens M
                    [--profile e4b|vllm|sglang] --out run.json
  sc5_driver.py --self-test
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import sys
import time


def _sc2():
    """``sc2_driver`` as staged beside this file on the box, or from ``bench/sc2`` in the repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "sc2_driver.py"), os.path.join(here, "..", "sc2", "sc2_driver.py")):
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("sc2_driver", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("sc2_driver.py is neither staged beside sc5_driver.py nor at ../sc2/")


sc2 = _sc2()
CONCURRENCIES = (1, 16, 64)


def closed_plan(n: int, seed: int, n_prompts: int, max_tokens: int) -> list:
    """[(prompt_index, max_tokens)] for requests 0..n-1, deterministic in its arguments."""
    if n < 1 or n_prompts < 1 or max_tokens < 1:
        raise ValueError(f"bad plan: n={n} prompts={n_prompts} max_tokens={max_tokens}")
    rng = random.Random(seed)
    return [(rng.randrange(n_prompts), max_tokens) for _ in range(n)]


async def run_closed(base: str, model: str, prompts: list, reqs: list, concurrency: int, extra: dict,
                     send=None) -> tuple:
    """Run ``reqs`` with ``concurrency`` workers. ``send(session, prompt, max_tokens, t0)`` defaults to SC2's request;
    the self-test passes a stand-in. Returns (records in index order, wall seconds, peak in flight)."""
    if concurrency < 1 or concurrency > len(reqs):
        raise ValueError(f"concurrency {concurrency} for {len(reqs)} requests")
    import contextlib
    if send is None:
        import aiohttp

        async def send(session, prompt, max_tokens, t0):
            return await sc2.one_request(session, base, model, prompt, max_tokens, extra, t0)
        conn = aiohttp.TCPConnector(limit=0, force_close=True)     # a fresh connection per request, as SC2
        session_cm = aiohttp.ClientSession(connector=conn)
    else:
        session_cm = contextlib.nullcontext(None)
    nxt, inflight, peak = 0, 0, 0
    recs: list = [None] * len(reqs)
    lock = asyncio.Lock()
    async with session_cm as session:
        t0 = time.perf_counter()

        async def worker(w):
            nonlocal nxt, inflight, peak
            while True:
                async with lock:
                    if nxt >= len(reqs):
                        return
                    i = nxt
                    nxt += 1
                    inflight += 1
                    peak = max(peak, inflight)
                p, m = reqs[i]
                rec = await send(session, prompts[p], m, t0)
                async with lock:
                    inflight -= 1
                rec.update(i=i, prompt_index=p, worker=w, warm=i < concurrency)
                recs[i] = rec
        await asyncio.gather(*(worker(w) for w in range(concurrency)))
        wall = time.perf_counter() - t0
    return recs, wall, peak


def summarize_closed(recs: list, concurrency: int, peak: int) -> dict:
    counted = [r for r in recs if not r.get("warm")]
    valid = [r for r in counted if r.get("valid")]
    ttft = [r["ttft_s"] for r in valid]
    tpot = [r["tpot_s"] for r in valid if r.get("tpot_s") is not None]
    e2e = [r["e2el_s"] for r in valid]
    out_tok = sum(r.get("completion_tokens") or 0 for r in valid)
    start = min((r["t_send"] for r in counted), default=None)
    end = max((r["t_send"] + r["e2el_s"] for r in counted if r.get("e2el_s") is not None), default=None)
    window = (end - start) if start is not None and end is not None and end > start else None
    return {"concurrency": concurrency, "n": len(recs), "warm": len(recs) - len(counted), "counted": len(counted),
            "valid": sum(1 for r in recs if r.get("valid")), "invalid": sum(1 for r in recs if not r.get("valid")),
            "errors": sorted({r.get("error") for r in recs if r.get("error")})[:5],
            "peak_in_flight": peak,
            **{f"ttft_p{p}_s": sc2._pct(ttft, p) for p in (50, 95)}, **{f"tpot_p{p}_s": sc2._pct(tpot, p) for p in (50, 95)},
            "e2el_p50_s": sc2._pct(e2e, 50), "output_tokens": out_tok, "window_s": round(window, 6) if window else None,
            "output_tok_s": round(out_tok / window, 3) if window else None}


def run_main(a) -> int:
    pf = json.load(open(a.prompts))
    prompts = pf["rows"]
    reqs = closed_plan(a.n, a.seed, len(prompts), a.max_tokens)
    recs, wall, peak = asyncio.run(run_closed(a.base.rstrip("/"), a.model, prompts, reqs, a.concurrency,
                                              sc2.PROFILES[a.profile]))
    summ = summarize_closed(recs, a.concurrency, peak)
    out = {"base": a.base, "model": a.model, "profile": a.profile, "mode": "closed", "concurrency": a.concurrency,
           "n": a.n, "seed": a.seed, "max_tokens": a.max_tokens, "prompts_sha256": pf.get("prompts_sha256"),
           "plan": reqs, "wall_s": round(wall, 3), "summary": summ, "requests": recs}
    json.dump(out, open(a.out, "w"), indent=1)
    print("SC5_RUN " + json.dumps({"profile": a.profile, "concurrency": a.concurrency, **{k: summ[k] for k in
                                   ("valid", "invalid", "peak_in_flight", "ttft_p50_s", "ttft_p95_s", "tpot_p50_s",
                                    "tpot_p95_s", "output_tok_s")}}), flush=True)
    return 0 if summ["invalid"] == 0 and summ["peak_in_flight"] == a.concurrency else 3


def _stub_send(latency):
    """A stand-in request for the self-test: sleeps ``latency(prompt)`` seconds and returns a VALID record."""
    async def send(session, prompt, max_tokens, t0):
        rec = sc2.new_record(prompt, max_tokens, time.perf_counter() - t0)
        await asyncio.sleep(latency(prompt))
        rec.pop("chunks")
        rec.update(valid=True, ttft_s=0.01, tpot_s=0.001, e2el_s=round(latency(prompt), 6), completion_tokens=max_tokens,
                   finish_reason="length")
        return rec
    return send


def self_test() -> int:
    ok = []
    p1 = closed_plan(40, 3, 64, 256)
    ok.append(p1 == closed_plan(40, 3, 64, 256) and p1 != closed_plan(40, 4, 64, 256))       # deterministic in the seed
    ok.append(all(0 <= p < 64 and m == 256 for p, m in p1))
    for bad in ((0, 1, 64, 1), (4, 1, 0, 1), (4, 1, 64, 0)):
        try:
            closed_plan(*bad)
            ok.append(False)
        except ValueError:
            ok.append(True)
    prompts = [[k] * 3 for k in range(64)]
    for c in (1, 4, 16):
        reqs = closed_plan(48, 7, 64, 32)
        recs, wall, peak = asyncio.run(run_closed("", "", prompts, reqs, c, {},
                                                  send=_stub_send(lambda pr: 0.002 + 0.001 * (pr[0] % 3))))
        s = summarize_closed(recs, c, peak)
        ok.append(peak == c and s["warm"] == c and s["counted"] == 48 - c and s["invalid"] == 0
                  and [r["i"] for r in recs] == list(range(48)) and all(r["warm"] == (r["i"] < c) for r in recs)
                  and s["output_tokens"] == 32 * (48 - c) and s["output_tok_s"] and s["window_s"] > 0)
    try:
        asyncio.run(run_closed("", "", prompts, closed_plan(3, 1, 64, 8), 4, {}, send=_stub_send(lambda pr: 0.0)))
        ok.append(False)
    except ValueError:
        ok.append(True)                                                       # more workers than requests is refused
    recs = [{"warm": True, "valid": False, "error": "HTTP 500: x", "t_send": 0.0},
            {"warm": False, "valid": True, "ttft_s": 0.1, "tpot_s": 0.01, "e2el_s": 2.0, "completion_tokens": 100,
             "t_send": 1.0},
            {"warm": False, "valid": True, "ttft_s": 0.3, "tpot_s": 0.02, "e2el_s": 3.0, "completion_tokens": 100,
             "t_send": 2.0}]
    s = summarize_closed(recs, 1, 1)
    ok.append(s["invalid"] == 1 and s["counted"] == 2 and s["ttft_p50_s"] == 0.2 and s["window_s"] == 4.0
              and s["output_tok_s"] == 50.0 and s["errors"] == ["HTTP 500: x"])   # an invalid warm-up still counts invalid
    print(f"sc5_driver self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--concurrency", type=int, choices=CONCURRENCIES)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--profile", choices=("e4b", "vllm", "sglang"), default="e4b")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd != "run" or not (a.base and a.model and a.prompts and a.concurrency and a.n and a.out):
        ap.error("run --base --model --prompts --concurrency --n --out, or --self-test")
    return run_main(a)


if __name__ == "__main__":
    sys.exit(main())
