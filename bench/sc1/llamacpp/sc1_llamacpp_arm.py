"""sc1_llamacpp_arm.py -- the llama.cpp SPEED arms of lane SC1 (experts4bit-qlora#846), driving a running llama-server
exactly as bench/h2h-20260905/p37/p37_vllm.py drives vLLM: the prompts are the EXACT token ids step_decomp._k8_window
produced (prompts_b{B}.json, sent as `prompt: [ids]` -- no tokenizer runs anywhere), the digests are asserted before any
request, B requests are fired concurrently (one per server slot; `-np B` on the server), and decode is isolated by the
SLOPE method (P20's protocol): generate 32 and 128 tokens from the same prompts; extra tokens / extra wall. Prefill,
load and the prompt's scheduling cancel; per-step scheduling and detokenisation do not (the whole serving loop is inside
the number, as it is inside vLLM's). min-of-3 and median-of-3 are both recorded.

Request shape (tools/server/README.md, server-schema.cpp @ b11327): `n_predict: N`, `temperature: 0` (greedy: the temp
sampler keeps only the arg-max), `cache_prompt: false` (every repetition re-prefills -- no prefix reuse across requests),
`ignore_eos: true` (the server appends -inf logit biases for every EOG token -- `logit_bias_eog` -- so EOS cannot end the
row early; the row then stops ONLY on `n_predict`, `stop_type == "limit"`), `return_tokens: true` (the generated ids come
back so the count is checked, not trusted). A row that did not generate exactly N tokens, hit a cache (prompt_n != len),
got truncated or stopped on anything but the limit makes the arm VOID (receipt written with the reason, exit 3): a row
that stopped early is not the registered workload.

The server's own `timings` (prompt_ms, predicted_ms, predicted_per_token_ms, ...) ride along per row as the engine's
SELF-REPORT; the position is taken on the wall-clock slope, never on the self-report.

--ttft: for one prompt file, `stream: true`, `n_predict: 8`, wall from request start to the first streamed token
(the first SSE event with tokens_predicted >= 1), 3 repeats after a warm run; the final event's `timings.prompt_ms` is
the server's own prefill time beside it.

stdlib only (urllib / http.client / threading).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ENGINE = "llamacpp"
SHORT, LONG = 32, 128


class PromptFileError(SystemExit):
    """The prompt file is not the registered fixture; refuse before a single request is sent (exit 2)."""

    def __init__(self, msg):
        super().__init__(2)
        self.msg = msg
        print(f"REFUSE: {msg}", file=sys.stderr)


class WorkloadError(Exception):
    """A row did not run the registered workload; the arm is VOID."""


# ----------------------------------------------------------------------------------------------------------------------
# prompt file (p37's contract)
# ----------------------------------------------------------------------------------------------------------------------
def load_prompts(path, batch, prompt_len=512, require_distinct=True):
    """Read prompts_b{B}.json and assert p37's contract: batch == B, each row prompt_len ids (0 skips the length check),
    rows distinct, prompts_sha256 == sha256(json.dumps(prompts))."""
    pf = json.load(open(path))
    prompts = pf["prompts"]
    if pf.get("batch") != batch or len(prompts) != batch:
        raise PromptFileError(f"prompt file batch {pf.get('batch')} / rows {len(prompts)} != --batch {batch}")
    if prompt_len and not all(len(p) == prompt_len for p in prompts):
        raise PromptFileError(f"every row must be {prompt_len} ids; got {sorted(set(len(p) for p in prompts))}")
    if require_distinct and len(set(tuple(p) for p in prompts)) != batch:
        raise PromptFileError("rows must be distinct prompts (P20's distinct-prompt rule)")
    if not all(isinstance(t, int) for p in prompts for t in p):
        raise PromptFileError("prompt rows must be integer token ids")
    sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    if sha != pf.get("prompts_sha256"):
        raise PromptFileError(f"prompt file digest mismatch: computed {sha[:12]}.. != file {str(pf.get('prompts_sha256'))[:12]}..")
    return prompts, sha, pf.get("rows_sha256")


# ----------------------------------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------------------------------
def http_get_json(base, path, timeout=30):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_get_text(base, path, timeout=30):
    try:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
            return r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # the receipt says what it could not read
        return f"unavailable: {e!r}"


def completion_request(prompt_ids, n_predict, stream=False, extra=None):
    body = {
        "prompt": prompt_ids,          # token ids: the server accepts [12, 34, 56] verbatim (tokenize_input_subprompt)
        "n_predict": n_predict,
        "temperature": 0,              # greedy
        "cache_prompt": False,         # no prefix reuse across requests
        "ignore_eos": True,            # EOG tokens get -inf logit bias; only n_predict ends the row
        "return_tokens": True,         # generated ids in `tokens`, counted not trusted
        "stream": stream,
        "n_probs": 0,
    }
    if extra:
        body.update(extra)
    return body


def post_completion(base, body, timeout):
    url = base.rstrip("/") + "/completion"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise WorkloadError(f"HTTP {e.code} from /completion: {e.read().decode(errors='replace')[:400]}") from e


def check_row(resp, prompt_ids, n_predict, row):
    """Every predicate that makes a row the registered workload."""
    toks = resp.get("tokens")
    if not isinstance(toks, list):
        raise WorkloadError(f"row {row}: no `tokens` in the response (return_tokens unsupported?)")
    if len(toks) != n_predict or resp.get("tokens_predicted") != n_predict:
        raise WorkloadError(f"row {row}: generated {len(toks)} / tokens_predicted {resp.get('tokens_predicted')} != {n_predict} "
                            f"(stop_type={resp.get('stop_type')!r}): a row that stopped early is not the registered workload")
    if resp.get("stop_type") != "limit":
        raise WorkloadError(f"row {row}: stop_type {resp.get('stop_type')!r} != 'limit'")
    if resp.get("truncated"):
        raise WorkloadError(f"row {row}: truncated (context too small for prompt + n_predict)")
    if resp.get("tokens_evaluated") != len(prompt_ids):
        raise WorkloadError(f"row {row}: tokens_evaluated {resp.get('tokens_evaluated')} != prompt length {len(prompt_ids)}")
    t = resp.get("timings") or {}
    if t.get("prompt_n") != len(prompt_ids):
        raise WorkloadError(f"row {row}: timings.prompt_n {t.get('prompt_n')} != {len(prompt_ids)}: the prompt was served from "
                            f"cache (cache_prompt must be false)")
    return toks


def run_batch(base, prompts, n_predict, timeout):
    """B concurrent /completion requests released together; wall from release to the last response."""
    B = len(prompts)
    results = [None] * B
    errors = [None] * B
    barrier = threading.Barrier(B + 1)

    def worker(i):
        body = completion_request(prompts[i], n_predict)
        try:
            barrier.wait()
            results[i] = post_completion(base, body, timeout)
        except Exception as e:  # noqa: BLE001 - surfaced below as a VOID / harness error
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(B)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    for i, e in enumerate(errors):
        if e is not None:
            if isinstance(e, WorkloadError):
                raise e
            raise WorkloadError(f"row {i}: request failed: {e!r}")
    tokens = []
    timings = []
    for i, r in enumerate(results):
        tokens.append(check_row(r, prompts[i], n_predict, i))
        timings.append(r.get("timings"))
    return wall, tokens, timings


def slope_stats(w_short, w_long, batch, short=SHORT, long_=LONG):
    """p37's estimator: min-of-3 and median-of-3 slopes between the SHORT and LONG walls."""
    extra_steps = long_ - short
    extra = extra_steps * batch
    d_min = min(w_long) - min(w_short)
    d_med = statistics.median(w_long) - statistics.median(w_short)
    out = {
        "walls_short_s": [round(w, 4) for w in w_short], "walls_long_s": [round(w, 4) for w in w_long],
        "wall_short_s": round(min(w_short), 4), "wall_long_s": round(min(w_long), 4),
        "end_to_end_tok_s_long": round(long_ * batch / min(w_long), 1),
    }
    if d_min > 0:
        out.update(decode_tok_s=round(extra / d_min, 1), decode_ms_per_step=round(d_min / extra_steps * 1e3, 4))
    else:
        out.update(decode_tok_s=None, decode_ms_per_step=None, slope_note="min-of-3 slope <= 0: LONG was not slower than SHORT")
    if d_med > 0:
        out.update(decode_tok_s_median=round(extra / d_med, 1), decode_ms_per_step_median=round(d_med / extra_steps * 1e3, 4))
    else:
        out.update(decode_tok_s_median=None, decode_ms_per_step_median=None)
    return out


def summarise_timings(timings_by_rep):
    """Per-row server self-report, flattened: medians over rows x reps."""
    flat = [t for rep in timings_by_rep for t in rep if t]
    keys = ("prompt_n", "prompt_ms", "prompt_per_token_ms", "predicted_n", "predicted_ms", "predicted_per_token_ms")
    out = {}
    for k in keys:
        vals = [t[k] for t in flat if isinstance(t.get(k), (int, float))]
        if vals:
            out[k + "_median"] = round(statistics.median(vals), 4)
            out[k + "_min"] = round(min(vals), 4)
            out[k + "_max"] = round(max(vals), 4)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# TTFT (streaming)
# ----------------------------------------------------------------------------------------------------------------------
def stream_ttft(base, prompt_ids, n_predict, timeout):
    """One streaming /completion; returns (ttft_s, total_s, tokens, final_event). TTFT = wall from the request's first
    byte to the first SSE event with tokens_predicted >= 1 (the first generated token leaving the server)."""
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    body = json.dumps(completion_request(prompt_ids, n_predict, stream=True)).encode()
    t0 = time.perf_counter()
    conn.request("POST", "/completion", body=body, headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise WorkloadError(f"HTTP {resp.status} from streaming /completion: {resp.read().decode(errors='replace')[:400]}")
    t_first = None
    tokens = []
    final = None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode(errors="replace").rstrip("\r\n")
        if not line or line.startswith(":"):      # keep-alive / SSE comment ping
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        ev = json.loads(payload)
        if "error" in ev:
            raise WorkloadError(f"server error in stream: {ev['error']}")
        if t_first is None and (ev.get("tokens_predicted", 0) >= 1 or ev.get("tokens")):
            t_first = time.perf_counter() - t0
        if ev.get("tokens"):
            tokens.extend(ev["tokens"])
        if ev.get("stop"):
            final = ev
            break
    total = time.perf_counter() - t0
    conn.close()
    if final is None:
        raise WorkloadError("stream ended without a final (stop: true) event")
    if t_first is None:
        raise WorkloadError("stream produced no token event")
    return t_first, total, tokens, final


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def base_receipt(a, prompts, prompts_sha, rows_sha):
    return {
        "engine": ENGINE, "arm": a.arm, "batch": a.batch,
        "method": ("slope(32->128) isolates decode; min-of-3 (P20 estimator) and median-of-3 both recorded; "
                   "B concurrent /completion requests released together, one per server slot"),
        "prompts": "identical token ids to the e4b arms (step_decomp._k8_window, wikitext-2 test, 512-token rows)",
        "prompts_file": a.prompts, "prompts_sha256": prompts_sha, "rows_sha256": rows_sha,
        "prompt_tokens": [len(p) for p in prompts],
        "server": a.server, "server_version": a.server_version, "server_log": a.server_log,
        "generation": {"temperature": 0, "ignore_eos": True, "cache_prompt": False, "return_tokens": True,
                       "greedy": True, "n_predict": "= the registered N (32/128; 8 for TTFT)"},
        "vast_instance_id": os.environ.get("SC1_INSTANCE_ID"),
        "env": {k: os.environ.get(k) for k in ("GGML_CUDA_MMQ_PREC", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS",
                                               "GGML_OP_OFFLOAD_MIN_BATCH", "GGML_CUDA_ENABLE_UNIFIED_MEMORY")},
    }


def write(out, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--prompts", required=True, help="prompts_b{B}.json (p37's dump)")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-version", default=None, help="the `llama-server --version` text the caller captured")
    ap.add_argument("--server-log", default=None, help="path of the server log (recorded, not read)")
    ap.add_argument("--prompt-len", type=int, default=512, help="assert every row has this many ids (0 = any)")
    ap.add_argument("--short", type=int, default=SHORT)
    ap.add_argument("--long", dest="long_", type=int, default=LONG)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--ttft", action="store_true", help="TTFT mode: stream, n_predict 8, wall to the first token")
    ap.add_argument("--ttft-n-predict", type=int, default=8)
    a = ap.parse_args(argv)

    prompts, prompts_sha, rows_sha = load_prompts(a.prompts, a.batch, a.prompt_len, require_distinct=(a.batch > 1))
    out = base_receipt(a, prompts, prompts_sha, rows_sha)
    out["health"] = http_get_text(a.server, "/health")
    try:
        out["props"] = http_get_json(a.server, "/props")
    except Exception as e:  # noqa: BLE001
        out["props"] = {"error": repr(e)[:300]}
    out["mode"] = "ttft" if a.ttft else "slope"

    try:
        if a.ttft:
            if a.batch != 1:
                raise WorkloadError("--ttft measures B=1 (one slot, one prompt)")
            n = a.ttft_n_predict
            out["ttft_n_predict"] = n
            stream_ttft(a.server, prompts[0], n, a.timeout)                  # warm (untimed)
            reps = []
            for _ in range(a.reps):
                ttft, total, toks, final = stream_ttft(a.server, prompts[0], n, a.timeout)
                if len(toks) != n or final.get("tokens_predicted") != n or final.get("stop_type") != "limit":
                    raise WorkloadError(f"TTFT row generated {len(toks)} / {final.get('tokens_predicted')} != {n} "
                                        f"(stop_type={final.get('stop_type')!r})")
                t = final.get("timings") or {}
                if t.get("prompt_n") != len(prompts[0]):
                    raise WorkloadError(f"TTFT row: timings.prompt_n {t.get('prompt_n')} != {len(prompts[0])} (cache hit)")
                reps.append({"ttft_s": round(ttft, 5), "total_s": round(total, 5), "tokens": toks, "timings": t})
            ttfts = [r["ttft_s"] for r in reps]
            out.update(
                ttft_reps=reps, ttft_s=ttfts, ttft_s_min=round(min(ttfts), 5), ttft_s_median=round(statistics.median(ttfts), 5),
                prefill_tok_s_from_ttft=round(len(prompts[0]) / statistics.median(ttfts), 1),
                server_prompt_ms=[r["timings"].get("prompt_ms") for r in reps],
                server_prompt_ms_median=round(statistics.median([r["timings"].get("prompt_ms") for r in reps
                                                                 if isinstance(r["timings"].get("prompt_ms"), (int, float))]), 3),
            )
        else:
            walls = {}
            tokens_long = None
            timings = {}
            for n in (a.short, a.long_):
                run_batch(a.server, prompts, n, a.timeout)                      # warm (untimed)
                ws, ts = [], []
                for _ in range(a.reps):
                    w, toks, tm = run_batch(a.server, prompts, n, a.timeout)
                    ws.append(w)
                    ts.append(tm)
                    if n == a.long_:
                        tokens_long = {str(i): list(map(int, t)) for i, t in enumerate(toks)}
                walls[n] = ws
                timings[n] = ts
            out.update(slope_stats(walls[a.short], walls[a.long_], a.batch, a.short, a.long_))
            out["server_timings"] = {"short": timings[a.short], "long": timings[a.long_]}
            out["server_timings_summary"] = {"short": summarise_timings(timings[a.short]), "long": summarise_timings(timings[a.long_])}
            out["tokens"] = tokens_long
            out["ttft_note"] = "see the --ttft receipt of this arm; the slope number carries no TTFT"
        out["verdict"] = "VALID"
    except WorkloadError as e:
        out["verdict"] = "VOID"
        out["void_reason"] = str(e)
        write(out, a.out)
        print(f"SC1LLAMACPP VOID arm={a.arm} batch={a.batch}: {e}", file=sys.stderr, flush=True)
        return 3

    out["metrics"] = http_get_text(a.server, "/metrics")
    write(out, a.out)
    keys = ("arm", "batch", "mode", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "end_to_end_tok_s_long",
            "ttft_s_median", "prompts_sha256", "verdict")
    print("SC1LLAMACPP " + json.dumps({k: out[k] for k in keys if k in out}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
