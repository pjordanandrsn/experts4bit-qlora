"""Lane SC2's request driver (``bench/sc2/sc2_driver.py``, #846).

- **Against e4b's real server** (CPU, runs in CI): the driver's exact request body goes through ``serve_paged``'s app
  (FastAPI ``TestClient`` over the scripted-runner engine of ``tests/test_serve_paged.py``), and the streamed bytes go
  through the driver's own SSE parser and record fold. The request must be accepted, stream exactly ``max_tokens``
  tokens and finish ``length``, and the record must read VALID.
- **Over a real socket** (needs aiohttp): fake OpenAI streaming servers with known delays check the timing math. That
  covers one chunk per token, four tokens per chunk (TPOT must stay per-token), an engine that stops early (INVALID)
  and an HTTP 500 (INVALID, a row and not a crash). It also checks the Poisson plan's offered rate end to end.
- **The capacity quantity**: a perfect engine's attainment reads 1.0 however long the drain after the last arrival.
"""
import asyncio
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


drv = _mod(REPO / "bench" / "sc2" / "sc2_driver.py", "sc2_driver")


def test_the_driver_self_test():
    assert drv.self_test() == 0


def test_the_driver_body_and_parser_against_serve_paged():
    pytest.importorskip("fastapi")
    sp = _mod(REPO / "tests" / "test_serve_paged.py", "_tsp")
    client, _engine = sp._client(sp.ScriptedRunner())
    prompt, max_tokens = [3, 4, 5, 6], 7
    body = drv.request_body("tiny/moe", prompt, max_tokens, drv.PROFILES["e4b"])
    rec = drv.new_record(prompt, max_tokens, 0.0)
    with client as c:                          # the app's lifespan starts the engine's step thread
        resp = c.post("/v1/completions", json=body)
        assert resp.status_code == 200, resp.text
        events, rest = drv._parse_sse(resp.content)
    assert rest == b"" and events[-1] == "[DONE]", (events[-3:], rest)
    t = 0.0
    for ev in events:
        t += 0.01
        drv.fold(rec, ev, t)
    drv.finish(rec, t)
    assert rec["valid"] and rec["completion_tokens"] == max_tokens and rec["finish_reason"] == "length", rec
    assert rec["ttft_s"] == 0.01 and rec["tpot_s"] is not None and rec["text"], rec


def _serve(handler):
    """Run an aiohttp app with ``handler`` on a free port; returns (base_url, cleanup coroutine)."""
    from aiohttp import web

    async def start():
        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}", runner.cleanup
    return start


def _streamer(prefill_s, token_s, per_chunk=1, stop_after=None):
    from aiohttp import web

    async def handler(request):
        body = await request.json()
        assert body["stream"] is True and body["temperature"] == 0 and body["ignore_eos"] is True
        n = body["max_tokens"] if stop_after is None else min(stop_after, body["max_tokens"])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(prefill_s)
        sent = 0
        while sent < n:
            k = min(per_chunk, n - sent)
            sent += k
            if sent > per_chunk:
                await asyncio.sleep(token_s * k)
            fin = "length" if sent == n and stop_after is None else ("stop" if sent == n else None)
            await resp.write(f"data: {json.dumps({'choices': [{'text': 'x' * k, 'finish_reason': fin}]})}\n\n".encode())
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\ndata: [DONE]\n\n".encode())
        return resp
    return handler


def _run(handler, reqs, mode="poisson"):
    async def go():
        base, cleanup = await _serve(handler)()
        try:
            return await drv.run_plan(base, "m", [[1, 2, 3]] * 4, reqs, mode, {})
        finally:
            await cleanup()
    return asyncio.run(go())


def test_timing_one_chunk_per_token():
    pytest.importorskip("aiohttp")
    recs, wall = _run(_streamer(0.10, 0.02), [(0.0, 0, 20), (0.05, 1, 20)])
    assert all(r["valid"] for r in recs), recs
    for r in recs:
        assert 0.08 <= r["ttft_s"] <= 0.6 and 0.012 <= r["tpot_s"] <= 0.08, r
    s = drv.summarize(recs, 1.0, wall)
    assert s["valid"] == 2 and s["output_tokens"] == 40 and s["good"] == 2


def test_tpot_stays_per_token_when_tokens_are_packed_into_chunks():
    pytest.importorskip("aiohttp")
    recs, _ = _run(_streamer(0.05, 0.01, per_chunk=4), [(0.0, 0, 21)])
    r = recs[0]
    # 6 chunks for 21 tokens; the last chunk lands about 20 x 10 ms after the first, so TPOT is about 10 ms, not 40
    assert r["valid"] and len(r["chunk_gaps_s"]) == 5 and 0.006 <= r["tpot_s"] <= 0.03, r


def test_an_engine_that_stops_early_and_an_http_error_are_invalid_rows():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    recs, _ = _run(_streamer(0.0, 0.0, stop_after=5), [(0.0, 0, 20)])
    assert not recs[0]["valid"] and "tokens 5 of 20" in recs[0]["error"], recs[0]

    async def boom(request):
        return web.Response(status=500, text="engine fault")
    recs, wall = _run(boom, [(0.0, 0, 20), (0.0, 1, 20)])
    assert [r["valid"] for r in recs] == [False, False] and all("HTTP 500" in r["error"] for r in recs)
    assert drv.summarize(recs, 1.0, wall)["invalid"] == 2


def test_the_poisson_plan_is_offered_on_schedule():
    pytest.importorskip("aiohttp")
    reqs = drv.plan("poisson", 40.0, 40, 3, 4, 4, 8)
    recs, wall = _run(_streamer(0.0, 0.001), reqs)
    s = drv.summarize(recs, 40.0, wall)
    assert s["valid"] == 40 and 30.0 <= s["achieved_rate"] <= 55.0, s
    # each send happens no earlier than its planned instant
    assert all(r["t_send"] >= r["planned_s"] - 1e-3 for r in recs)


def test_capacity_reads_attainment_not_good_requests_over_the_wall():
    """A perfect engine at 8 req/s: 120 arrivals over ~15 s, then a ~3 s drain. good / wall is ~6.7 req/s, under 95 % of
    8, so a rule on it could never credit 8 (nor 4). The registered quantity is attainment: every request good -> 1.0,
    goodput = 8, and the reducer's capacity ceiling reaches 8."""
    recs = [{"valid": True, "ttft_s": 0.2, "tpot_s": 0.01, "e2el_s": 3.0, "completion_tokens": 256, "t_send": i * 15.0 / 119}
            for i in range(120)]
    sm = drv.summarize(recs, 8.0, 18.0)
    assert sm["attainment"] == 1.0 and sm["goodput_rps"] == 8.0 and sm["good_per_wall_rps"] == round(120 / 18.0, 4)
    assert sm["good_per_wall_rps"] < 0.95 * 8                          # the quantity the old rule read: never credits 8
    red = _mod(REPO / "bench" / "sc2" / "sc2_reduce.py", "sc2_reduce")
    run = {"summary": dict(sm, ttft_p99_s=0.2, tpot_p99_s=0.01)}
    rows = {r: red.row([run, run], r) for r in red.RATES}
    assert all(x["status"] == "VALID" and x["meets_capacity"] for x in rows.values()) and red.ceiling(rows) == 8
