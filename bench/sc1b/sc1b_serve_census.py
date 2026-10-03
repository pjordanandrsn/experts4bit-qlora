#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_serve_census.py -- lane SC1b (#846): the client side of one census capture against a RUNNING server that `nsys
profile` launched (box D passes the nsys command through SC1_LAUNCH_PREFIX). It sends SC1's exact request, built by SC1's
own drivers (staged beside this file), and never talks to nsys.

SGLang (`--engine sglang`): the server runs under `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.
    One non-streaming batch POST /generate of the B rows (sc1_sglang_arm.sampling_params(160): temperature 0,
    max_new_tokens = min_new_tokens = 160, ignore_eos), and at t = request + prefill_s + K x step_ms a POST /start_profile
    {"activities": ["CUDA_PROFILER"], "num_steps": N} -- cudaProfilerStart/Stop in the scheduler process. SGLang answers
    that POST with plain text ("Start profiling."); the reply must be HTTP 200 and say so.
llama.cpp (`--engine llamacpp`): the server runs under plain `nsys profile` (llama.cpp has no profiler hook), so the whole
    160-token run is captured and the reducer selects the registered positions by counting logits copies. B concurrent
    /completion requests (sc1_llamacpp_arm.completion_request(row, 160): n_predict 160, temperature 0, cache_prompt false,
    ignore_eos, return_tokens, n_probs 0), released together through a barrier, as SC1's arm does.

The census record is written in every case (a `finally`), with the events and the acknowledgement.

  sc1b_serve_census.py --engine sglang|llamacpp --batch B --prompts prompts_bB.json --port P [--pass1 RECEIPT.json] --out census.json
  sc1b_serve_census.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import threading
import time
import urllib.error
import urllib.request

SKIP, STEPS, TOKENS = 32, 64, 160
HERE = os.path.dirname(os.path.abspath(__file__))


def _sc1(module, sub):
    """SC1's own driver module: staged at $W/<sub>/ on the box, bench/sc1/<sub>/ in the repo."""
    for d in (os.path.join(HERE, sub), os.path.join(HERE, "..", "sc1", sub)):
        path = os.path.join(d, module + ".py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(module, path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise SystemExit(f"SC1's {sub}/{module}.py is not staged beside {HERE}")


def plan_offsets(prefill_s: float, step_ms: float, skip: int = SKIP, steps: int = STEPS, tokens: int = TOKENS) -> dict:
    """When SGLang's /start_profile is posted, from the moment the request is sent; refuses a plan whose window (with 30 %
    slack) could run past the end of generation."""
    if step_ms <= 0 or prefill_s < 0:
        raise ValueError(f"bad pass-1 timing: prefill_s {prefill_s}, step_ms {step_ms}")
    start = prefill_s + skip * step_ms / 1e3
    end_of_gen = prefill_s + tokens * step_ms / 1e3
    if start + steps * step_ms / 1e3 * 1.3 > end_of_gen:
        raise ValueError(f"the window ({steps} steps from {start:.2f} s, 30 % slack) runs past generation end at {end_of_gen:.2f} s")
    return {"start_s": round(start, 4), "end_of_generation_s": round(end_of_gen, 4)}


def pass1_timing(receipt: dict) -> tuple[float, float]:
    """(prefill_s, step_ms) from SC1's arm receipt on this box: step = decode_ms_per_step; prefill = min short wall - 32 x step."""
    step_ms = float(receipt["decode_ms_per_step"])
    walls = receipt.get("walls_short_s") or receipt.get("walls_short")
    if not walls:
        raise ValueError("pass-1 receipt carries no short walls")
    return max(0.0, min(walls) - int(receipt.get("short") or 32) * step_ms / 1e3), step_ms


def _post_json(port, path, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def _post_text(port, path, body, timeout):
    """(status, text): SGLang's /start_profile answers plain text, never JSON (http_server.py:1181-1189)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def run(engine, batch, prompts, port, prefill_s=None, step_ms=None, skip=SKIP, steps=STEPS, tokens=TOKENS, out_path=None):
    rec = {"engine": engine, "batch": batch, "skip": skip, "steps": steps, "tokens": tokens, "events": [], "errors": [],
           "status": "harness_error"}
    t0 = time.perf_counter()

    def ev(name, **kw):
        rec["events"].append(dict(name=name, t=round(time.perf_counter() - t0, 4), **kw))
    results = []
    try:
        if engine == "sglang":
            sg = _sc1("sc1_sglang_arm", "sglang")
            rec["plan"] = plan_offsets(prefill_s, step_ms, skip, steps, tokens)
            payload = {"input_ids": prompts, "sampling_params": sg.sampling_params(tokens), "stream": False}

            def send():
                try:
                    out = _post_json(port, "/generate", payload, 3600)
                    results.extend(out if isinstance(out, list) else [out])
                except Exception as e:  # noqa: BLE001
                    rec["errors"].append(repr(e)[:300])
            th = threading.Thread(target=send)
            th.start()
            ev("sent")
            time.sleep(max(0.0, rec["plan"]["start_s"] - (time.perf_counter() - t0)))
            ev("start_profile_post")
            code, text = _post_text(port, "/start_profile", {"activities": ["CUDA_PROFILER"], "num_steps": steps}, 120)
            ev("start_profile_reply", code=code, text=text.strip()[:120])
            rec["profile_acknowledged"] = code == 200 and "start profiling" in text.lower()
            th.join()
            got = [len(r.get("output_ids") or []) for r in results]
        elif engine == "llamacpp":
            ll = _sc1("sc1_llamacpp_arm", "llamacpp")
            base = f"http://127.0.0.1:{port}"
            rows = [None] * batch
            barrier = threading.Barrier(batch + 1)

            def worker(i):
                body = ll.completion_request(prompts[i], tokens)
                try:
                    barrier.wait()
                    rows[i] = ll.post_completion(base, body, 3600)
                except Exception as e:  # noqa: BLE001
                    rec["errors"].append(f"row {i}: {e!r}"[:300])
            ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(batch)]
            for t in ths:
                t.start()
            barrier.wait()
            ev("released")
            for t in ths:
                t.join()
            rec["profile_acknowledged"] = True               # the whole run is under `nsys profile`: nothing to acknowledge
            got = [len((r or {}).get("tokens") or []) for r in rows]
        else:
            raise ValueError(engine)
        ev("generation_done")
        rec["generated_per_row"] = got
        ok = not rec["errors"] and rec["profile_acknowledged"] and len(got) == batch and all(g == tokens for g in got)
        rec["status"] = "ok" if ok else "harness_error"
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(repr(e)[:300])
    finally:
        if out_path:
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=1)
    return rec


def selftest():
    import http.server
    p = plan_offsets(0.2, 4.0)
    assert p["start_s"] == 0.328 and p["end_of_generation_s"] == 0.84, p
    try:
        plan_offsets(0.2, 4.0, tokens=96)
        raise AssertionError("expected a refusal")
    except ValueError as e:
        assert "runs past generation end" in str(e), e
    pre, step = pass1_timing({"decode_ms_per_step": 3.0, "walls_short_s": [0.25, 0.24, 0.26]})
    assert abs(pre - (0.24 - 0.096)) < 1e-9 and step == 3.0, (pre, step)
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.setdefault(self.path, []).append(body)
            if self.path == "/start_profile":
                data, ctype = b"Start profiling.\n", "text/plain"               # SGLang's literal reply
            elif self.path == "/generate":
                time.sleep(0.2)
                n = body["sampling_params"]["max_new_tokens"]
                data, ctype = json.dumps([{"output_ids": [1] * n} for _ in body["input_ids"]]).encode(), "application/json"
            else:
                n = body["n_predict"]
                data, ctype = json.dumps({"tokens": [1] * n, "tokens_predicted": n}).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    r = run("sglang", 2, [[1, 2], [3, 4]], port, prefill_s=0.0, step_ms=0.5)
    assert r["status"] == "ok" and r["profile_acknowledged"] and r["generated_per_row"] == [160, 160], r
    assert seen["/start_profile"][0] == {"activities": ["CUDA_PROFILER"], "num_steps": 64}, seen
    assert seen["/generate"][0]["sampling_params"]["min_new_tokens"] == 160 and seen["/generate"][0]["stream"] is False, seen
    r = run("llamacpp", 3, [[1], [2], [3]], port)
    assert r["status"] == "ok" and r["generated_per_row"] == [160, 160, 160], r
    assert all(b["n_probs"] == 0 and b["cache_prompt"] is False for b in seen["/completion"]), seen["/completion"]
    srv.shutdown()
    print("selftest OK (6 cases)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("sglang", "llamacpp"))
    ap.add_argument("--batch", type=int)
    ap.add_argument("--prompts")
    ap.add_argument("--port", type=int)
    ap.add_argument("--pass1")
    ap.add_argument("--skip", type=int, default=SKIP)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tokens", type=int, default=TOKENS)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    with open(a.prompts) as f:
        pf = json.load(f)
    rows = pf["prompts"][:a.batch]
    prefill_s = step_ms = None
    if a.pass1:
        with open(a.pass1) as f:
            prefill_s, step_ms = pass1_timing(json.load(f))
    rec = run(a.engine, a.batch, rows, a.port, prefill_s, step_ms, a.skip, a.steps, a.tokens, a.out)
    print("SC1B_SERVE " + json.dumps({k: rec.get(k) for k in ("engine", "batch", "status", "profile_acknowledged", "generated_per_row")}), flush=True)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
