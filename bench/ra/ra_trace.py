"""Join native SC2 completions to requests and reconcile aggregate engine steps.

StepTrace has no per-request IDs. Its counts can be reconciled with scheduler
totals, but cannot establish which request occupied each individual step.
"""
from __future__ import annotations

import json
import math

import ra_normalize
import ra_process


def require(ok, message):
    if not ok:
        raise ValueError("capacity trace: " + message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def count(value):
    return type(value) is int and value >= 0


def equal(a, b, tolerance=1e-8):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def read_lines(path):
    ra_process.file_digest(path)  # Same non-symlink regular-file checks as inputs.
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(bool(rows) and all(type(r) is dict for r in rows), "empty/non-object JSONL")
    return rows


def close_engine(engine, timeout):
    """Stop only the wrapper's drained engine; native _run closes its tracer."""
    require(number(timeout) and 0 < timeout <= 10, "close timeout")
    require(engine.state == "ready" and engine.error is None and engine.queue_depth == 0 and
            not engine._stop and engine._thread is not None and engine._thread.is_alive(),
            "engine not ready/drained/running")
    with engine._lock:
        scheduler = engine.parts.scheduler
        require(not scheduler.active and not scheduler.queue, "scheduler not drained")
        require(scheduler.tracer is not None, "no native step tracer")
    engine.shutdown(timeout=timeout)
    require(not engine._thread.is_alive(), "engine thread did not stop")
    tracer = scheduler.tracer
    require(tracer.cur is None and not tracer.rows and not tracer._pending, "native trace not fully flushed")
    return {"closed": True, "thread_alive": False, "trace_steps": tracer.n, "stats": engine.stats()}


def join(workloads, requests, steps, closed, health):
    require(closed.get("closed") is True and closed.get("thread_alive") is False,
            "engine did not close its trace on its own thread")
    stats = closed["stats"]
    sched = stats["scheduler"]
    require(stats["state"] == "ready" and stats["error"] is None and stats["in_flight"] == 0 and
            sched["aborted"] == sched["in_flight"] == sched["queued"] == 0,
            "engine failed or retained requests")
    require(len(workloads) == 3 and [w["n"] for w in workloads] == [4, 64, 120], "workload count/order")
    by_id, by_rid, groups = {}, set(), []
    for row in requests:
        rid, request_id = row["rid"], row["request_id"]
        require(count(rid) and rid not in by_rid and isinstance(request_id, str) and bool(request_id) and
                request_id not in by_id, "duplicate/missing request identity")
        by_rid.add(rid)
        by_id[request_id] = row
        times = [row[k] for k in ("arrival", "admitted_at", "first_token_at", "finished_at")]
        require(all(number(t) for t in times) and times == sorted(times) and number(row["arrival_epoch"]),
                "request timestamps/order")
        require(equal(row["ttft"], times[2] - times[0]) and equal(row["queue_wait"], times[1] - times[0]) and
                equal(row["decode_s"], times[3] - times[2]), "request derived timing")
    seen, tokens = set(), 0
    for workload in workloads:
        ra_normalize.requests(workload)
        group = []
        for http in workload["requests"]:
            request_id = http.get("request_id")
            require(request_id in by_id and request_id not in seen, "HTTP request ID missing/duplicate/unjoined")
            seen.add(request_id)
            native = by_id[request_id]
            require(native["prompt_len"] == http["prompt_tokens"] == 512 and
                    count(native["out_len"]) and native["out_len"] == http["completion_tokens"] and
                    native["finish_reason"] == http["finish_reason"] == "length", "HTTP/native usage")
            # Recompute the frozen driver's rounded SSE timing. Do not equate
            # client perf_counter with the server's monotonic clock.
            gaps = http["chunk_gaps_s"]
            require(isinstance(gaps, list) and len(gaps) < native["out_len"] and
                    all(number(g) for g in gaps) and number(http["ttft_s"]) and number(http["e2el_s"]),
                    "HTTP chunk timings")
            # Each gap was rounded to 6 places; summing can accumulate 0.5us/gap.
            require(equal(http["tpot_s"], sum(gaps) / (native["out_len"] - 1),
                          (len(gaps) * .5e-6 / (native["out_len"] - 1)) + .5e-6 + 1e-12) and
                    http["ttft_s"] + sum(gaps) <= http["e2el_s"] + (len(gaps) + 2) * .5e-6 and
                    native["finished_at"] - native["arrival"] <= http["e2el_s"] + 1e-6,
                    "HTTP derived timing/span")
            tokens += native["out_len"]
            group.append(native)
        groups.append(group)
    require(seen == set(by_id) and len(seen) == stats["records_total"] == sched["completed"] == 188 and
            by_rid == set(range(188)), "request trace coverage")
    for earlier, later in zip(groups, groups[1:]):
        require(max(r["finished_at"] for r in earlier) <= min(r["arrival"] for r in later),
                "workloads overlapped on server clock")
    require(bool(steps) and len(steps) == sched["steps"] == closed["trace_steps"], "step trace coverage")
    totals = step_totals(steps, sched, health, tokens, 188)
    return {"schema": 1, "checks": "HTTP IDs/usage/timing + aggregate scheduler/step/graph counters",
            "requests": len(seen), "steps": len(steps), "tokens_emitted": tokens, "totals": totals,
            "per_request_steps_available": False, "proves_gpu_engagement": False}


def step_totals(steps, sched, health, tokens, request_count):
    require(bool(steps) and len(steps) == sched["steps"], "scheduler step coverage")
    totals = {k: 0 for k in ("admitted", "prefill_chunks", "prefill_tokens", "prefill_replays", "decode_rows",
                              "dec_pieces")}
    buckets = health["engine"]["buckets"]
    require(bool(buckets) and all(count(b) and b > 0 for b in buckets) and buckets == sorted(set(buckets)),
            "resolved bucket list")
    last_t = None
    for i, row in enumerate(steps):
        require(type(row["step"]) is int and row["step"] == i and number(row["t"]) and
                (last_t is None or row["t"] > last_t), "step identity/order")
        last_t = row["t"]
        require(number(row["step_ms"]) and isinstance(row["seg"], dict) and bool(row["seg"]) and
                all(number(v) for v in row["seg"].values()) and
                equal(sum(row["seg"].values()), row["step_ms"], (len(row["seg"]) + 1) * .00005 + 1e-9),
                "step host segments")
        require(not row.get("gpu_error"), "CUDA trace error")
        for key in totals:
            value = row.get(key, 0)
            require(count(value), "step counter type")
            totals[key] += value
        pf, dec = row.get("prefill_chunks", 0), row.get("decode_rows", 0)
        require(pf > 0 or dec > 0, "empty step")
        require(row.get("prefill_tokens", 0) == pf * 512 and row.get("prefill_replays", 0) == pf,
                "prefill step counts/fallback")
        gpu = row.get("gpu", {})
        for prefix, present in (("pf", pf), ("dec", dec)):
            if present:
                start, end = prefix + "_prep", prefix + ("_forward" if prefix == "pf" else "_issue")
                require(start in gpu and end in gpu and number(gpu[start]) and number(gpu[end]) and
                        gpu[end] > gpu[start], "missing/nonpositive CUDA forward interval")
        if dec:
            pieces = row.get("dec_pieces")
            remaining = dec % buckets[-1] or buckets[-1]
            last_bucket = next(b for b in buckets if b >= remaining)
            require(count(pieces) and pieces == math.ceil(dec / buckets[-1]) and row.get("bucket") == last_bucket,
                    "decode pieces/last bucket")
    graph_stats = health["engine"]["graph_stats"]
    require(bool(graph_stats) and all(count(s[k]) for s in graph_stats.values()
                                     for k in ("rows", "replays", "eager_steps")), "graph counter types")
    require(sched["tokens_emitted"] == tokens and sched["prefill_tokens"] == totals["prefill_tokens"] == request_count * 512 and
            totals["admitted"] == totals["prefill_chunks"] == totals["prefill_replays"] == request_count and
            health["prefill_graph"]["replays"] == totals["prefill_replays"], "scheduler/prefill totals")
    discarded = sched.get("lookahead_discarded", 0)
    require(count(discarded) and totals["decode_rows"] == tokens - request_count + discarded and
            totals["decode_rows"] == sum(s["rows"] for s in graph_stats.values()) and
            totals["dec_pieces"] == sum(s["replays"] for s in graph_stats.values()) and
            all(s["eager_steps"] == 0 for s in graph_stats.values()), "decode trace/counter totals")
    return totals


def bind(root, workloads, closed, health):
    paths = [root / filename for filename in ("request-trace.jsonl", "step-trace.jsonl")]
    result = join(workloads, *(read_lines(p) for p in paths), closed, health)
    result["raw_sha256"] = {p.name: ra_process.file_digest(p) for p in paths}
    (root / "trace-binding.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
