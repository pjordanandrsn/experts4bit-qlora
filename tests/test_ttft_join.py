"""Request identity and TTFT decomposition (#846), without a GPU or model."""
import asyncio
import copy
import importlib.util
import json
import time
from pathlib import Path

import pytest


def _module(name):
    path = Path(__file__).resolve().parents[1] / "bench" / "sc2" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver, join = _module("sc2_driver"), _module("ttft_join")


def _stream(ids):
    record = driver.new_record([1, 2], 2, 0.)
    for t, rid, text, finish in ((.01, ids[0], "", None), (.1, ids[1], "a", None),
                                (.2, ids[2], "b", "length")):
        event = {"choices": [{"text": text, "finish_reason": finish}],
                 "usage": {"completion_tokens": 2, "prompt_tokens": 2}}
        if rid is not None:
            event["id"] = rid
        driver.fold(record, json.dumps(event), t)
    driver.fold(record, "[DONE]", .3)
    return driver.finish(record, .3)


@pytest.mark.parametrize("ids,expected,conflict", [
    (("a", "a", "a"), "a", False), ((None, "a", None), "a", False),
    ((None, None, None), None, False), (("a", "b", "a"), "a", True),
    ((42, "a", "a"), "a", False),
])
def test_identity_metadata_preserves_the_registered_reading(ids, expected, conflict):
    result = _stream(ids)
    assert result["request_id"] == expected
    assert bool(result.get("request_id_conflict")) == conflict
    baseline = _stream((None, None, None))
    for key in ("valid", "ttft_s", "tpot_s", "e2el_s", "empty_chunks", "text", "completion_tokens"):
        assert result[key] == baseline[key]


def _fixture():
    clients, servers = [], []
    for rid, queue, prefill, residual in (("a", .1, .2, .01), ("b", .3, .4, .02)):
        clients.append({"request_id": rid, "valid": True, "prompt_len": 512,
                        "completion_tokens": 64, "ttft_s": queue + prefill + residual})
        servers.append({"request_id": rid, "prompt_len": 512, "out_len": 64,
                        "arrival": 100., "admitted_at": 100. + queue,
                        "first_token_at": 100. + queue + prefill,
                        "ttft": queue + prefill, "queue_wait": queue})
    return {"requests": clients}, servers


def test_join_uses_identity_despite_reordered_completion_and_extra_warmup():
    client, trace = _fixture()
    warm = dict(trace[0], request_id="warmup")
    result = join.analyse(client, [trace[1], warm, trace[0]])
    assert result["matched"] == 2 and result["unmatched_server_rows"] == 1
    assert result["summary"]["queue_wait_s"]["p50_s"] == .2
    assert result["summary"]["admission_to_first_token_s"]["p50_s"] == .3
    assert result["summary"]["client_minus_server_s"]["p50_s"] == .015
    assert result == join.analyse(client, [warm, *trace])
    for pair in result["requests"]:
        assert pair["server_ttft_s"] == pytest.approx(pair["queue_wait_s"] + pair["admission_to_first_token_s"])


@pytest.mark.parametrize("mutation,reason", [
    ("missing_id", "missing or conflicting"), ("conflicting_id", "missing or conflicting"),
    ("duplicate_client", "duplicate request_id"), ("duplicate_server", "duplicate request_id"),
    ("unmatched", "no match"), ("length", "length mismatch"),
    ("timestamp", "out of order"), ("nonfinite", "must be finite"),
    ("inconsistent", "inconsistent server"), ("empty", "no valid"),
])
def test_ambiguous_or_inconsistent_data_is_refused(mutation, reason):
    client, trace = _fixture()
    if mutation == "missing_id":
        client["requests"][0].pop("request_id")
    elif mutation == "conflicting_id":
        client["requests"][0]["request_id_conflict"] = True
    elif mutation == "duplicate_client":
        client["requests"].append(copy.deepcopy(client["requests"][0]))
    elif mutation == "duplicate_server":
        trace.append(copy.deepcopy(trace[0]))
    elif mutation == "unmatched":
        trace.pop()
    elif mutation == "length":
        trace[0]["out_len"] += 1
    elif mutation == "timestamp":
        trace[0]["admitted_at"] = trace[0]["first_token_at"] + 1
    elif mutation == "nonfinite":
        client["requests"][0]["ttft_s"] = float("nan")
    elif mutation == "inconsistent":
        trace[0]["ttft"] += 1
    elif mutation == "empty":
        client["requests"].clear()
    with pytest.raises(ValueError, match=reason):
        join.analyse(client, trace)


def test_invalid_requests_are_counted_and_negative_residuals_are_preserved():
    client, trace = _fixture()
    client["requests"].append({"valid": False, "error": "HTTP 500"})
    client["requests"][0]["ttft_s"] = .29
    result = join.analyse(client, trace)
    assert result["invalid_clients"] == 1 and result["negative_residuals"] == 1
    assert result["requests"][0]["client_minus_server_s"] == pytest.approx(-.01)


def test_completion_id_survives_a_real_http_stream():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    async def handler(request):
        body = await request.json()
        assert body["stream"] and body["max_tokens"] == 2
        events = [
            {"id": "cmpl-live", "choices": [{"text": ""}]},
            {"id": "cmpl-live", "choices": [{"text": "ab", "finish_reason": "length"}]},
            {"id": "cmpl-live", "choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 2}},
        ]
        payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return web.Response(text=payload, content_type="text/event-stream")

    async def run():
        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession() as session:
                return await driver.one_request(session, f"http://127.0.0.1:{port}", "tiny", [1, 2], 2, {},
                                                time.perf_counter())
        finally:
            await runner.cleanup()

    result = asyncio.run(run())
    assert result["valid"] and result["request_id"] == "cmpl-live"
    assert result["empty_chunks"] == 1 and result["ttft_s"] >= 0
    assert result["completion_tokens"] == 2
