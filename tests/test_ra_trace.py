"""Synthetic trace mutations and the real CPU engine's buffered close barrier."""
import asyncio
import copy
import gzip
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/ra" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for name in ("ra_reduce", "ra_stage", "ra_normalize", "ra_process"):
    load(name)
trace = load("ra_trace")


def test_step_projection_reconciles_committed_sc2e_native_receipts():
    # Re-read existing measurements, not a new GPU execution or RA clearance.
    root = ROOT / "bench/h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/sc2"
    with gzip.open(root / "steps_e4b_s64a_d1.jsonl.gz", "rt") as stream:
        steps = [json.loads(line) for line in stream]
    requests = trace.read_lines(root / "trace_e4b_s64a_d1.jsonl")
    health = json.loads((root / "health_e4b_s64a_d1_end.json").read_bytes())
    tokens = sum(r["out_len"] for r in requests)
    scheduler = {"steps": len(steps), "tokens_emitted": tokens, "prefill_tokens": len(requests) * 512}
    totals = trace.step_totals(steps, scheduler, health, tokens, len(requests))
    assert totals["decode_rows"] == sum(s["rows"] for s in health["engine"]["graph_stats"].values())
    assert totals["prefill_replays"] == health["prefill_graph"]["replays"]
    with pytest.raises(ValueError, match="coverage"):
        trace.step_totals(steps[:-1], scheduler, health, tokens, len(requests))


def fixture():
    workloads, requests, steps = [], [], []
    for n in (4, 64, 120):
        plan, rows = [], []
        for i in range(n):
            rid = len(requests)
            request_id = f"synthetic-{rid}"
            t = float(rid * 2)
            plan.append([0, i % 16, 64])
            rows.append({"i": i, "prompt_index": i % 16, "planned_s": 0, "max_tokens": 64,
                         "status": 200, "valid": True, "prompt_tokens": 512, "prompt_len": 512,
                         "completion_tokens": 64, "finish_reason": "length", "request_id": request_id,
                         "ttft_s": .1, "tpot_s": .01, "e2el_s": .8, "chunk_gaps_s": [.01] * 63})
            requests.append({"request_id": request_id, "rid": rid, "arrival": t, "arrival_epoch": t,
                             "admitted_at": t + .05, "first_token_at": t + .1, "finished_at": t + .73,
                             "prompt_len": 512, "out_len": 64, "finish_reason": "length", "ttft": .1,
                             "queue_wait": .05, "decode_s": .63})
            steps.append({"step": rid, "t": t, "admitted": 1, "prefill_chunks": 1, "prefill_tokens": 512,
                          "prefill_replays": 1, "decode_rows": 63, "dec_pieces": 4, "bucket": 16,
                          "step_ms": 2., "seg": {"pf_forward": 1., "dec_issue": 1.},
                          "gpu": {"pf_prep": 0., "pf_forward": .5, "dec_prep": 1., "dec_issue": 1.5}})
        workloads.append({"n": n, "plan": plan, "requests": rows, "summary": {"valid": n, "invalid": 0}})
    closed = {"closed": True, "thread_alive": False, "trace_steps": 188,
              "stats": {"state": "ready", "error": None, "in_flight": 0, "records_total": 188,
                        "scheduler": {"steps": 188, "completed": 188, "aborted": 0, "in_flight": 0,
                                      "queued": 0, "tokens_emitted": 188 * 64, "prefill_tokens": 188 * 512}}}
    health = {"prefill_graph": {"replays": 188}, "engine": {"buckets": [1, 2, 4, 8, 16],
              "graph_stats": {"16": {"rows": 188 * 63, "replays": 188 * 4, "eager_steps": 0}}}}
    return workloads, requests, steps, closed, health


def test_complete_binding_retains_hashes_and_states_aggregate_limit(tmp_path):
    workloads, requests, steps, closed, health = fixture()
    for filename, rows in (("request-trace.jsonl", requests), ("step-trace.jsonl", steps)):
        (tmp_path / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = trace.bind(tmp_path, workloads, closed, health)
    assert result["requests"] == result["steps"] == 188
    assert result["tokens_emitted"] == 188 * 64
    assert result["per_request_steps_available"] is result["proves_gpu_engagement"] is False
    assert result["raw_sha256"]["step-trace.jsonl"] == trace.ra_process.file_digest(tmp_path / "step-trace.jsonl")
    assert json.loads((tmp_path / "trace-binding.json").read_bytes()) == result


@pytest.mark.parametrize("mutation", ["missing_request", "extra_request", "duplicate_rid", "duplicate_id",
    "missing_http_id", "http_conflict", "http_usage", "native_timing", "nonfinite", "negative_gap", "tpot",
    "end_to_end", "phase_overlap", "missing_step", "duplicate_step", "step_order", "host_segments", "gpu_error",
    "missing_gpu", "zero_gpu", "prefill", "pieces", "bucket", "emitted", "graph_rows", "graph_replays",
    "fallback", "closure", "live_thread", "queued", "unflushed", "bool_counter"])
def test_trace_mutations_refuse(mutation):
    workloads, requests, steps, closed, health = fixture()
    http, req, step = workloads[0]["requests"][0], requests[0], steps[0]
    sched, graph = closed["stats"]["scheduler"], health["engine"]["graph_stats"]["16"]
    if mutation == "missing_request":
        requests.pop()
    elif mutation == "extra_request":
        requests.append(copy.deepcopy(req))
    elif mutation == "duplicate_rid":
        requests[1]["rid"] = 0
    elif mutation == "duplicate_id":
        requests[1]["request_id"] = req["request_id"]
    elif mutation == "missing_http_id":
        del http["request_id"]
    elif mutation == "http_conflict":
        http["request_id_conflict"] = True
    elif mutation == "http_usage":
        req["out_len"] = 65
    elif mutation == "native_timing":
        req["queue_wait"] = .1
    elif mutation == "nonfinite":
        req["arrival"] = float("nan")
    elif mutation == "negative_gap":
        http["chunk_gaps_s"][0] = -.01
    elif mutation == "tpot":
        http["tpot_s"] = .02
    elif mutation == "end_to_end":
        http["e2el_s"] = .2
    elif mutation == "phase_overlap":
        req = requests[4]
        delta = 2.
        for key in ("arrival", "admitted_at", "first_token_at", "finished_at"):
            req[key] -= delta
    elif mutation == "missing_step":
        steps.pop()
    elif mutation == "duplicate_step":
        steps[1]["step"] = 0
    elif mutation == "step_order":
        steps[1]["t"] = 0
    elif mutation == "host_segments":
        step["step_ms"] = 3.
    elif mutation == "gpu_error":
        step["gpu_error"] = "synthetic CUDA failure"
    elif mutation == "missing_gpu":
        del step["gpu"]["pf_forward"]
    elif mutation == "zero_gpu":
        step["gpu"]["dec_issue"] = step["gpu"]["dec_prep"]
    elif mutation == "prefill":
        step["prefill_tokens"] = 511
    elif mutation == "pieces":
        step["dec_pieces"] = 1
    elif mutation == "bucket":
        step["bucket"] = 8
    elif mutation == "emitted":
        sched["tokens_emitted"] += 1
    elif mutation == "graph_rows":
        graph["rows"] -= 1
    elif mutation == "graph_replays":
        graph["replays"] += 1
    elif mutation == "fallback":
        graph["eager_steps"] = 1
    elif mutation == "closure":
        closed["closed"] = False
    elif mutation == "live_thread":
        closed["thread_alive"] = True
    elif mutation == "queued":
        sched["queued"] = 1
    elif mutation == "unflushed":
        closed["trace_steps"] += 1
    elif mutation == "bool_counter":
        step["prefill_chunks"] = True
    with pytest.raises((ValueError, trace.ra_normalize.reducer.Invalid)):
        trace.join(workloads, requests, steps, closed, health)


def test_real_cpu_engine_flushes_last_partial_trace_batch(tmp_path):
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler
    from experts4bit_qlora.serve_paged import PagedEngine, PagedServeConfig, PagedStream, _DONE

    class Runner:
        tracer = None
        def bind(self, rid, slot, prompt): pass
        def free_slot(self, rid): pass
        def run_prefill(self, chunks): return {rid: 7 for rid, _, _ in chunks}
        def run_decode(self, rids): return {rid: 7 for rid in rids}

    runner = Runner()
    scheduler = ContinuousScheduler(runner=runner, max_seqs=1, kv_slots=1, chunk_tokens=512)
    engine = PagedEngine(PagedServeConfig(device="cpu", trace_path=str(tmp_path / "request.jsonl"),
                         step_trace_path=str(tmp_path / "step.jsonl")),
                         parts=types.SimpleNamespace(scheduler=scheduler, runner=runner))

    async def run():
        queue = asyncio.Queue()
        engine.start(asyncio.get_running_loop())
        stream = PagedStream(queue=queue, request_id="native-cpu-control", prompt_ids=[1] * 512,
                             max_tokens=4, stop_ids=None, min_tokens=0, arrival=0, arrival_epoch=0)
        engine.submit(stream)
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=2)
            if item[0] == _DONE:
                break
        assert not (tmp_path / "step.jsonl").exists()  # Fewer than flush_every=64.
        closed = trace.close_engine(engine, 2)
        rows = trace.read_lines(tmp_path / "step.jsonl")
        request, = trace.read_lines(tmp_path / "request.jsonl")
        assert len(rows) == closed["trace_steps"] == closed["stats"]["scheduler"]["steps"] == 4
        assert request["request_id"] == "native-cpu-control" and request["out_len"] == 4
        assert all("gpu" not in r for r in rows)  # This CPU control cannot be GPU engagement.
        assert not engine._thread.is_alive() and closed["stats"]["in_flight"] == 0
        with pytest.raises(ValueError, match="running"):
            trace.close_engine(engine, 2)
    try:
        asyncio.run(run())
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mutation", ["busy", "alive", "buffered", "pending", "current", "no_tracer"])
def test_closure_refuses_incomplete_engine(mutation):
    tracer = types.SimpleNamespace(cur=None, rows=[], _pending=[], n=1)
    scheduler = types.SimpleNamespace(active={}, queue=[], tracer=tracer)
    thread = types.SimpleNamespace(is_alive=lambda: True)
    engine = types.SimpleNamespace(state="ready", error=None, queue_depth=0, _stop=False, _thread=thread,
        _lock=threading.Lock(), parts=types.SimpleNamespace(scheduler=scheduler), stats=lambda: {})
    engine.shutdown = lambda **_: setattr(thread, "is_alive", lambda: mutation == "alive")
    if mutation == "busy":
        scheduler.active = {0: object()}
    elif mutation == "buffered":
        tracer.rows = [{}]
    elif mutation == "pending":
        tracer._pending = [{}]
    elif mutation == "current":
        tracer.cur = {}
    elif mutation == "no_tracer":
        scheduler.tracer = None
    with pytest.raises(ValueError):
        trace.close_engine(engine, 1)
