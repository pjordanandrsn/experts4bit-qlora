#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_driver.py -- lane SC2 (#846): ONE request driver, run unchanged against every engine's OpenAI-compatible
``/v1/completions`` (e4b's ``serve_paged``, vLLM, SGLang, llama.cpp's server). It never talks to an engine any other way.

A run is a fixed request PLAN: for each request, its arrival offset, its prompt (a list of token ids, identical for
every engine) and its ``max_tokens``. The plan comes from (mode, rate, n, seed) alone, so every engine is offered the
same requests at the same instants:
  ``poisson``  exponential inter-arrivals at ``rate`` req/s (open loop: arrivals never wait for completions);
  ``serial``   one request in flight at a time (Q1: the interactive latency at rate -> 0).
Every request streams (``stream: true``), is greedy (``temperature: 0``), runs exactly to ``max_tokens``
(``ignore_eos: true``) and asks for usage (``stream_options.include_usage``). Engine-specific body fields come from the
profile (llama.cpp's ``cache_prompt: false``); nothing else differs between engines.

Per request: the send time; the first token chunk (TTFT); every token chunk's arrival; ``completion_tokens`` from
usage; the streamed text; ``finish_reason``; any error. A TOKEN CHUNK is a chunk whose choice carries non-empty text: a
choice chunk with empty text (a finish-only chunk, a partial character held back by the detokenizer) is counted in
``empty_chunks`` but never timed, so it cannot move TTFT or stretch TPOT; a usage-only chunk (empty ``choices``) is
neither. A request is VALID iff it returned HTTP 200, reported exactly ``max_tokens`` completion tokens and finished with
``length``. TPOT = (last token chunk - first token chunk) / (tokens - 1): it stays per-token when an engine packs several
tokens into one SSE chunk; the raw chunk gaps are kept beside it.

Summary (``summarize``): offered and achieved arrival rate, wall duration, valid / invalid counts, p50 / p90 / p99 of
TTFT, TPOT and end-to-end latency, output tok/s over the run, and SLO ATTAINMENT: the share of the run's requests that are
VALID and meet the SLO (TTFT <= ``slo_ttft_s`` AND TPOT <= ``slo_tpot_s``). Goodput = attainment x the offered rate.
Attainment, not good requests / wall time, is the capacity quantity: the wall includes the drain after the last arrival
(~ one request's E2E), which caps good / wall below the offered rate even for a perfect engine -- at 8 req/s, 120
arrivals span ~15 s and the drain adds ~3 s, so good / wall <= ~6.7 req/s. ``good_per_wall_rps`` keeps that raw figure.

  sc2_driver.py run --base URL --model NAME --prompts prompts.json --mode poisson --rate R --n N --seed S
                    [--max-tokens-lo 64 --max-tokens-hi 256] [--profile e4b|vllm|sglang|llamacpp] --out run.json
  sc2_driver.py --self-test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time

PROFILES = {"e4b": {}, "vllm": {}, "sglang": {}, "llamacpp": {"cache_prompt": False}}
SLO_TTFT_S, SLO_TPOT_S = 1.0, 0.100


def plan(mode: str, rate: float, n: int, seed: int, n_prompts: int, lo: int = 64, hi: int = 256) -> list:
    """The request plan: [(arrival_offset_s, prompt_index, max_tokens)], deterministic in its arguments. ``serial``
    leaves every offset 0: the driver sends the next request when the previous one finishes."""
    if mode not in ("poisson", "serial"):
        raise ValueError(f"mode {mode!r}")
    if n < 1 or n_prompts < 1 or not 1 <= lo <= hi:
        raise ValueError(f"bad plan: n={n} prompts={n_prompts} lo={lo} hi={hi}")
    if mode == "poisson" and not rate > 0:
        raise ValueError("a poisson plan needs rate > 0")
    rng = random.Random(seed)
    out, t = [], 0.0
    for i in range(n):
        if mode == "poisson":
            t += rng.expovariate(rate)
        out.append((round(t, 6) if mode == "poisson" else 0.0, rng.randrange(n_prompts), rng.randint(lo, hi)))
    return out


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    return round(s[f] + (s[c] - s[f]) * (k - f), 6)


def summarize(recs: list, rate: float, wall_s: float, slo_ttft_s: float = SLO_TTFT_S, slo_tpot_s: float = SLO_TPOT_S) -> dict:
    valid = [r for r in recs if r.get("valid")]
    ttft = [r["ttft_s"] for r in valid]
    tpot = [r["tpot_s"] for r in valid if r.get("tpot_s") is not None]
    e2e = [r["e2el_s"] for r in valid]
    good = [r for r in valid if r["ttft_s"] <= slo_ttft_s and (r.get("tpot_s") is None or r["tpot_s"] <= slo_tpot_s)]
    sends = sorted(r["t_send"] for r in recs if r.get("t_send") is not None)
    achieved = (len(sends) - 1) / (sends[-1] - sends[0]) if len(sends) > 1 and sends[-1] > sends[0] else None
    out_tok = sum(r.get("completion_tokens") or 0 for r in valid)
    return {"n": len(recs), "valid": len(valid), "invalid": len(recs) - len(valid),
            "errors": sorted({r.get("error") for r in recs if r.get("error")})[:5],
            "offered_rate": rate, "achieved_rate": round(achieved, 4) if achieved else None, "wall_s": round(wall_s, 3),
            **{f"ttft_p{p}_s": _pct(ttft, p) for p in (50, 90, 99)}, **{f"tpot_p{p}_s": _pct(tpot, p) for p in (50, 90, 99)},
            **{f"e2el_p{p}_s": _pct(e2e, p) for p in (50, 90, 99)},
            "ttft_mean_s": round(statistics.fmean(ttft), 6) if ttft else None,
            "output_tok_s": round(out_tok / wall_s, 3) if wall_s > 0 else None, "output_tokens": out_tok,
            "good": len(good), "attainment": round(len(good) / len(recs), 4) if recs else None,
            "goodput_rps": round(len(good) / len(recs) * rate, 4) if recs and rate > 0 else None,
            "good_per_wall_rps": round(len(good) / wall_s, 4) if wall_s > 0 else None,
            "slo": {"ttft_s": slo_ttft_s, "tpot_s": slo_tpot_s}}


def _parse_sse(buf: bytes):
    """Split complete SSE events off ``buf``: (events, rest). An event is the ``data:`` payload of one block."""
    events = []
    while b"\n\n" in buf:
        block, buf = buf.split(b"\n\n", 1)
        for line in block.split(b"\n"):
            line = line.strip()
            if line.startswith(b"data:"):
                events.append(line[5:].strip().decode())
    return events, buf


def request_body(model: str, prompt: list, max_tokens: int, extra: dict) -> dict:
    """The one request body every engine receives (plus its profile's fields)."""
    return {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}, **extra}


def new_record(prompt: list, max_tokens: int, t_send: float) -> dict:
    return {"max_tokens": max_tokens, "prompt_len": len(prompt), "t_send": round(t_send, 6), "chunks": [], "text": "",
            "completion_tokens": None, "prompt_tokens": None, "finish_reason": None, "empty_chunks": 0, "valid": False}


def fold(rec: dict, event: str, t: float) -> None:
    """One SSE event's ``data`` payload into the record, at ``t`` seconds after the send."""
    if event == "[DONE]":
        return
    msg = json.loads(event)
    ch = msg.get("choices") or []
    if ch:
        c = ch[0]
        text = c.get("text") or ""
        if text:                                   # a token chunk: timed
            rec["chunks"].append(round(t, 6))
            rec["text"] += text
        else:                                      # a finish-only or held-back chunk: counted, never timed
            rec["empty_chunks"] += 1
        if c.get("finish_reason"):
            rec["finish_reason"] = c["finish_reason"]
    if msg.get("usage"):
        rec["completion_tokens"] = msg["usage"].get("completion_tokens")
        if msg["usage"].get("prompt_tokens") is not None:          # recorded, not a validity term (SC2b gates on it)
            rec["prompt_tokens"] = msg["usage"]["prompt_tokens"]


def finish(rec: dict, e2el_s: float) -> dict:
    """TTFT, TPOT, chunk gaps and validity from the folded events."""
    rec["e2el_s"] = round(e2el_s, 6)
    if rec["chunks"]:
        rec["ttft_s"] = rec["chunks"][0]
        n = rec["completion_tokens"] or 0
        rec["tpot_s"] = round((rec["chunks"][-1] - rec["chunks"][0]) / (n - 1), 6) if n > 1 else None
        rec["chunk_gaps_s"] = [round(b - a, 6) for a, b in zip(rec["chunks"], rec["chunks"][1:])]
    rec["valid"] = bool(rec["chunks"]) and rec["completion_tokens"] == rec["max_tokens"] and rec["finish_reason"] == "length"
    if not rec["valid"] and "error" not in rec:
        rec["error"] = f"tokens {rec['completion_tokens']} of {rec['max_tokens']}, finish {rec['finish_reason']}"
    rec.pop("chunks")
    return rec


async def one_request(session, base: str, model: str, prompt: list, max_tokens: int, extra: dict, t0: float,
                      timeout_s: float = 900.0) -> dict:
    import aiohttp
    rec = new_record(prompt, max_tokens, time.perf_counter() - t0)
    start = time.perf_counter()
    try:
        async with session.post(f"{base}/v1/completions", json=request_body(model, prompt, max_tokens, extra),
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                rec.pop("chunks")
                return rec
            buf = b""
            async for piece in resp.content.iter_any():
                buf += piece
                events, buf = _parse_sse(buf)
                now = time.perf_counter() - start
                for ev in events:
                    fold(rec, ev, now)
    except Exception as e:  # noqa: BLE001  (a failed request is a row, never a crash)
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec.pop("chunks", None)
        return rec
    return finish(rec, time.perf_counter() - start)


async def run_plan(base: str, model: str, prompts: list, reqs: list, mode: str, extra: dict) -> tuple:
    import aiohttp
    # force_close: a fresh TCP connection per request, on every engine alike. llama.cpp's server closes the connection
    # after a streamed response without saying so, and a pooled keep-alive socket then fails the NEXT request with
    # ServerDisconnectedError -- every other request on sc2-prove-1 (A1). Connecting over loopback costs well under a ms.
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        if mode == "serial":
            recs = [await one_request(session, base, model, prompts[p], m, extra, t0) for _, p, m in reqs]
        else:
            async def fire(off, p, m):
                delay = off - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                return await one_request(session, base, model, prompts[p], m, extra, t0)
            recs = await asyncio.gather(*(fire(*r) for r in reqs))
        wall = time.perf_counter() - t0
    for i, (r, (off, p, _m)) in enumerate(zip(recs, reqs)):
        r.update(i=i, prompt_index=p, planned_s=off)
    return list(recs), wall


def run_main(a) -> int:
    pf = json.load(open(a.prompts))
    prompts = pf["rows"]
    reqs = plan(a.mode, a.rate, a.n, a.seed, len(prompts), a.max_tokens_lo, a.max_tokens_hi)
    recs, wall = asyncio.run(run_plan(a.base.rstrip("/"), a.model, prompts, reqs, a.mode, PROFILES[a.profile]))
    summ = summarize(recs, a.rate if a.mode == "poisson" else 0.0, wall)
    out = {"base": a.base, "model": a.model, "profile": a.profile, "mode": a.mode, "rate": a.rate, "n": a.n, "seed": a.seed,
           "max_tokens_range": [a.max_tokens_lo, a.max_tokens_hi], "prompts_sha256": pf.get("prompts_sha256"),
           "plan": reqs, "summary": summ, "requests": recs}
    json.dump(out, open(a.out, "w"), indent=1)
    print("SC2_RUN " + json.dumps({"profile": a.profile, "mode": a.mode, "rate": a.rate, **{k: summ[k] for k in
                                   ("valid", "invalid", "achieved_rate", "ttft_p50_s", "ttft_p99_s", "tpot_p50_s", "tpot_p99_s",
                                    "output_tok_s", "goodput_rps")}}), flush=True)
    return 0 if summ["invalid"] == 0 else 3


def self_test() -> int:
    ok = []
    p1, p2 = plan("poisson", 4.0, 400, 7, 64), plan("poisson", 4.0, 400, 7, 64)
    ok.append(p1 == p2 and p1 != plan("poisson", 4.0, 400, 8, 64))                 # deterministic in the seed
    rate = (len(p1) - 1) / (p1[-1][0] - p1[0][0])
    ok.append(3.4 < rate < 4.6 and all(b[0] >= a[0] for a, b in zip(p1, p1[1:])))  # about the offered rate, ordered
    ok.append(all(64 <= m <= 256 and 0 <= p < 64 for _, p, m in p1))
    s = plan("serial", 0.0, 5, 1, 3)
    ok.append([x[0] for x in s] == [0.0] * 5)
    ok.append(_pct([1, 2, 3, 4, 5], 50) == 3 and _pct([1, 2, 3, 4], 50) == 2.5 and _pct([], 50) is None)
    evs, rest = _parse_sse(b'data: {"a":1}\n\ndata: [DONE]\n\ndata: {"b"')
    ok.append(evs == ['{"a":1}', "[DONE]"] and rest == b'data: {"b"')
    recs = [{"valid": True, "ttft_s": 0.5, "tpot_s": 0.05, "e2el_s": 5.0, "completion_tokens": 100, "t_send": 0.0},
            {"valid": True, "ttft_s": 1.5, "tpot_s": 0.05, "e2el_s": 6.0, "completion_tokens": 100, "t_send": 1.0},
            {"valid": False, "error": "HTTP 500: x", "t_send": 2.0}]
    sm = summarize(recs, 1.0, 10.0)
    ok.append(sm["valid"] == 2 and sm["invalid"] == 1 and sm["good"] == 1 and sm["attainment"] == 0.3333
              and sm["goodput_rps"] == 0.3333 and sm["good_per_wall_rps"] == 0.1
              and sm["output_tok_s"] == 20.0 and sm["achieved_rate"] == 1.0 and sm["errors"] == ["HTTP 500: x"])
    try:
        plan("poisson", 0.0, 3, 1, 3)
        ok.append(False)
    except ValueError:
        ok.append(True)
    r = new_record([1, 2], 3, 0.0)                 # empty first and finish-only last chunks are not timed
    for t, ev in ((0.1, '{"choices":[{"text":"","finish_reason":null}]}'), (0.2, '{"choices":[{"text":"a","finish_reason":null}]}'),
                  (0.3, '{"choices":[{"text":"bc","finish_reason":null}]}'), (0.5, '{"choices":[{"text":"d","finish_reason":null}]}'),
                  (0.9, '{"choices":[{"text":"","finish_reason":"length"}],"usage":{"completion_tokens":3,"prompt_tokens":2}}'),
                  (0.95, '{"choices":[],"usage":{"completion_tokens":3}}'), (0.96, "[DONE]")):
        fold(r, ev, t)
    r = finish(r, 1.0)
    ok.append(r["valid"] and r["ttft_s"] == 0.2 and r["tpot_s"] == 0.15 and r["empty_chunks"] == 2 and r["text"] == "abcd"
              and r["prompt_tokens"] == 2)
    print(f"sc2_driver self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--mode", choices=("poisson", "serial"), default="poisson")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens-lo", type=int, default=64)
    ap.add_argument("--max-tokens-hi", type=int, default=256)
    ap.add_argument("--profile", choices=tuple(PROFILES), default="e4b")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd != "run":
        ap.error("run or --self-test")
    return run_main(a)


if __name__ == "__main__":
    sys.exit(main())
