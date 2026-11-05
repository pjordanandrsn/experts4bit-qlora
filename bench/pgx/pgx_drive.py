"""PGX pass driver: one pass of fixed-length greedy completions against an OpenAI-compatible server, closed loop at c.

    python bench/pgx/pgx_drive.py --url http://127.0.0.1:PORT --model NAME --prompts SET.json --concurrency C \\
        --pass-id ID --out RECORD.json [--max-tokens 256]

The same driver, with the same clocks, runs every engine (SGLang, vLLM with the pager, e4b's serve_paged): each
request is ``POST /v1/completions`` with the prompt as TOKEN IDS, ``max_tokens = min_tokens = 256``,
``ignore_eos``, ``temperature 0``, streamed (SSE) with ``stream_options.include_usage``. Stdlib only.

Per request it records the send time, every SSE content event's arrival time, the usage's ``completion_tokens``,
and the sha256 of the concatenated text (greedy token identity across passes is read as text identity: the three
servers stream text deltas, not ids). Per pass: wall, output tok/s, TTFT p50/p99, TPOT p50/p99 (per request, (last
event - first event) / (tokens - 1)), the per-event gap p50/p99 when every request's event count equals its token
count, and the prompt set's sha256 (``prompts_sha``, P235's contract: the sha of the token-id lists as sent).

A request that errors, or returns fewer tokens than ``max_tokens``, is recorded with its fault; the reducer VOIDs
the pass. The driver itself never retries a request.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import queue
import sys
import threading
import time
import urllib.parse


def prompts_sha(prompts: list[dict]) -> str:
    """sha256 of the token-id lists, in set order, as JSON without spaces (the ids are what every engine is sent)."""
    blob = json.dumps([p["token_ids"] for p in prompts], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def pct(xs, q):
    """Nearest-rank percentile (q in [0, 100]); None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[k]


def one_request(url: str, model: str, prompt: dict, max_tokens: int, timeout: float) -> dict:
    u = urllib.parse.urlsplit(url)
    body = {"model": model, "prompt": prompt["token_ids"], "max_tokens": max_tokens, "min_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    rec = {"id": prompt["id"], "kind": prompt.get("kind"), "events": [], "usage_tokens": None, "fault": None}
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    text = []
    try:
        t0 = time.perf_counter()
        conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["fault"] = f"http {resp.status}: {resp.read(300)!r}"
            return rec
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            if msg.get("usage"):
                rec["usage_tokens"] = int(msg["usage"].get("completion_tokens", 0))
            for ch in msg.get("choices") or []:
                piece = ch.get("text") or ""
                if piece:
                    rec["events"].append(time.perf_counter() - t0)
                    text.append(piece)
    except Exception as exc:  # noqa: BLE001 -- recorded; the reducer VOIDs the pass
        rec["fault"] = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    rec["text_sha"] = hashlib.sha256("".join(text).encode()).hexdigest()
    n = rec["usage_tokens"]
    if rec["fault"] is None and (n is None or n < max_tokens):
        rec["fault"] = f"short output: {n} of {max_tokens} tokens"
    ev = rec["events"]
    rec["ttft_s"] = ev[0] if ev else None
    rec["e2e_s"] = ev[-1] if ev else None
    rec["tpot_s"] = (ev[-1] - ev[0]) / (n - 1) if ev and n and n > 1 else None
    return rec


def run_pass(url, model, prompts, concurrency, max_tokens, timeout=600.0) -> dict:
    """Closed loop: ``concurrency`` workers each take the next prompt as soon as their last request ends."""
    work: queue.Queue = queue.Queue()
    for p in prompts:
        work.put(p)
    out, lock = [], threading.Lock()

    def worker():
        while True:
            try:
                p = work.get_nowait()
            except queue.Empty:
                return
            r = one_request(url, model, p, max_tokens, timeout)
            with lock:
                out.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    out.sort(key=lambda r: [p["id"] for p in prompts].index(r["id"]))
    ok = [r for r in out if r["fault"] is None]
    toks = sum(r["usage_tokens"] or 0 for r in ok)
    gaps_ok = bool(ok) and all(len(r["events"]) == r["usage_tokens"] for r in ok)
    gaps = [b - a for r in ok for a, b in zip(r["events"], r["events"][1:])] if gaps_ok else []
    ttft = [r["ttft_s"] for r in ok]
    tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    return {"requests": out, "n": len(prompts), "faults": sum(r["fault"] is not None for r in out),
            "concurrency": concurrency, "max_tokens": max_tokens, "wall_s": wall,
            "out_tok_s": toks / wall if wall > 0 else None,
            "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
            "tpot_p50": pct(tpot, 50), "tpot_p99": pct(tpot, 99),
            "gap_p50": pct(gaps, 50), "gap_p99": pct(gaps, 99), "per_token_events": gaps_ok,
            "prompts_sha": prompts_sha(prompts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSON list of {id, kind, token_ids}")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--pass-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    with open(a.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    rec = run_pass(a.url, a.model, prompts, a.concurrency, a.max_tokens, a.timeout)
    rec["pass_id"] = a.pass_id
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(f"{a.pass_id}: {rec['n']} requests, {rec['faults']} faults, out tok/s {rec['out_tok_s']}, "
          f"tpot p50 {rec['tpot_p50']}")
    return 0 if rec["faults"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
