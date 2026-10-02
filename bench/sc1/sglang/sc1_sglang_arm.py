"""sc1_sglang_arm.py -- the SGLang arm of lane SC1 (bench/sc1/SC1-PREREG.md, experts4bit-qlora#846): P37's slope driver
(bench/h2h-20260905/p37/p37_vllm.py) pointed at a RUNNING `sglang.launch_server` (started by server.sh in one of its
modes) through the NATIVE /generate endpoint with TOKEN IDS -- no tokenizer runs here, the rows are the exact ids
step_decomp._k8_window produced (prompts_b{B}.json: `prompts` = B distinct 512-token rows, `prompts_sha256`,
`rows_sha256`, `batch`).

Decode is isolated by the SLOPE method (P20/P37, unchanged): generate 32 and 128 tokens from the same prompts, one warm
+ 3 timed reps per length; extra tokens / extra wall; min-of-3 (P20's estimator) and median-of-3 both recorded. The
whole serving loop (HTTP, tokenizer manager, scheduler, detokeniser) is inside SGLang's number, as it is inside vLLM's.

The request (every field verified at tag v0.5.20, python/sglang/srt/...):
  POST /generate  {"input_ids": [[...], ...], "sampling_params": {"temperature": 0, "max_new_tokens": N,
                   "min_new_tokens": N, "ignore_eos": true}, "stream": false}
  * `input_ids` may be a LIST OF LISTS: managers/io_struct.py:185-188 (`List[List[int]] | List[int]`), batch detection
    at :445-473 (`isinstance(self.input_ids[0], int)` -> single, else batch of len(input_ids)); the B rows are fanned
    out and gathered by the tokenizer manager (tokenizer_manager.py:2040-2058), so one call = B concurrent requests
    scheduled together under fcfs. A plain list-of-ints is a single request (returns a dict, not a list).
  * sampling fields: sampling/sampling_params.py:123 max_new_tokens, :131 temperature, :138 min_new_tokens,
    :147 ignore_eos; verify() at :252-265 requires 0 <= min_new_tokens <= max_new_tokens.
  * response per row: {"text", "output_ids", "meta_info": {"id", "finish_reason", "prompt_tokens", "completion_tokens",
    "cached_tokens", "e2e_latency", ...}} (tokenizer_manager.py:2351-2356 base keys, :2397-2402 completion/cached,
    :2587 e2e_latency on the finished output, :2496-2514 the finished dict).
  * --ttft: "stream": true -> text/event-stream of `data: <json>\n\n` chunks ending in `data: [DONE]\n\n`
    (entrypoints/http_server.py:913-955); each chunk carries `output_ids` (ids so far, or the delta when the server runs
    incremental streaming) and `meta_info.completion_tokens`; TTFT = wall from request send to the first chunk carrying
    >= 1 output token (3 timed repeats after one warm, max_new_tokens 8, one-row prompt file of 512 or 4096 tokens).

Receipt: P37's shape with engine "sglang" (sc1_reduce reads the same keys), plus the server's resolved args
(<log>.server_info.json from server.sh) and its engagement record. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

SHORT, LONG = 32, 128
PROMPT_LEN = 512
TTFT_PROMPT_LENS = (512, 4096)
TTFT_NEW_TOKENS = 8


class SGLangClient:
    """Thin stdlib HTTP client for the native endpoints this lane touches."""

    def __init__(self, host: str, port: int, timeout: float):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _request(self, method: str, path: str, body=None, stream: bool = False):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:  # surface the server's own message (the 400 body names the bad field)
            detail = e.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
        if stream:
            return resp
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        return json.loads(raw) if "json" in ctype or raw[:1] in (b"{", b"[") else raw.decode(errors="replace")

    def health(self) -> bool:
        try:
            urllib.request.urlopen(self.base + "/health", timeout=self.timeout).read()
            return True
        except Exception:
            return False

    def server_info(self) -> dict:
        try:
            return self._request("GET", "/server_info")
        except RuntimeError:
            return self._request("GET", "/get_server_info")

    def flush_cache(self):
        return self._request("POST", "/flush_cache")

    def generate(self, payload: dict):
        return self._request("POST", "/generate", payload)

    def generate_stream(self, payload: dict):
        payload = dict(payload, stream=True)
        return self._request("POST", "/generate", payload, stream=True)


def load_prompts(path: str, batch: int, prompt_len: int = PROMPT_LEN):
    """P37's prompt-file contract, byte for byte: batch, row length, distinct rows, the file's own digest."""
    pf = json.load(open(path))
    prompts = pf["prompts"]
    assert pf["batch"] == batch and len(prompts) == batch, (pf["batch"], len(prompts), batch)
    assert all(len(p) == prompt_len for p in prompts), f"prompt_len must be {prompt_len}"
    assert len(set(tuple(p) for p in prompts)) == batch, "rows must be distinct prompts"
    prompts_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    assert prompts_sha == pf["prompts_sha256"], "prompt file digest mismatch"
    return prompts, pf, prompts_sha


def sampling_params(n_tokens: int) -> dict:
    return {"temperature": 0.0, "max_new_tokens": n_tokens, "min_new_tokens": n_tokens, "ignore_eos": True}


def check_rows(rows, n_tokens: int, batch: int, radix_off: bool):
    """Every row produced exactly N tokens (a row that stopped early is not the registered workload), and no row was
    served from a prefix cache on a radix-off arm. Returns (tokens dict, meta_info list)."""
    assert isinstance(rows, list) and len(rows) == batch, f"expected {batch} rows, got {type(rows).__name__} of {len(rows) if isinstance(rows, list) else '?'}"
    got = sum(int(r["meta_info"]["completion_tokens"]) for r in rows)
    assert got == n_tokens * batch, f"{got} != {n_tokens * batch} (a row stopped early: not the registered workload)"
    for i, r in enumerate(rows):
        assert len(r["output_ids"]) == n_tokens, f"row {i}: {len(r['output_ids'])} output_ids != completion_tokens {n_tokens}"
        if radix_off:
            assert int(r["meta_info"].get("cached_tokens", 0)) == 0, f"row {i}: cached_tokens={r['meta_info'].get('cached_tokens')} on a radix-off arm"
    tokens = {str(i): list(map(int, r["output_ids"])) for i, r in enumerate(rows)}
    return tokens, [r["meta_info"] for r in rows]


def slope_stats(w_short, w_long, batch: int, short: int = SHORT, long: int = LONG) -> dict:
    """P37's arithmetic: extra tokens / extra wall, on the min-of-reps and on the median-of-reps."""
    extra = (long - short) * batch
    d_min = min(w_long) - min(w_short)
    d_med = statistics.median(w_long) - statistics.median(w_short)
    return dict(
        walls_short_s=[round(w, 4) for w in w_short], walls_long_s=[round(w, 4) for w in w_long],
        wall_short_s=round(min(w_short), 4), wall_long_s=round(min(w_long), 4),
        decode_tok_s=round(extra / d_min, 1), decode_ms_per_step=round(d_min / (long - short) * 1e3, 4),
        decode_tok_s_median=round(extra / d_med, 1), decode_ms_per_step_median=round(d_med / (long - short) * 1e3, 4),
        end_to_end_tok_s_long=round(long * batch / min(w_long), 1),
    )


def _generate_batch(client: SGLangClient, prompts, n_tokens: int, mode: str):
    """One timed call. `batch`: one /generate with the list of rows (the scheduler sees B requests at once).
    `threads`: B single-row POSTs fired together (the fallback shape; rows re-assembled in prompt order)."""
    if mode == "batch":
        out = client.generate({"input_ids": prompts, "sampling_params": sampling_params(n_tokens), "stream": False})
        return [out] if isinstance(out, dict) else out
    rows = [None] * len(prompts)
    errs = []

    def one(i):
        try:
            rows[i] = client.generate({"input_ids": prompts[i], "sampling_params": sampling_params(n_tokens), "stream": False})
        except Exception as e:  # noqa: BLE001 -- re-raised below with the row index
            errs.append((i, repr(e)))

    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise RuntimeError(f"threaded rows failed: {errs[:3]}")
    return rows


def run_slope(client: SGLangClient, prompts, batch: int, radix_off: bool, reps: int, conc: str, short: int, long: int):
    def run(n_tokens):
        _generate_batch(client, prompts, n_tokens, conc)                       # warm (untimed)
        walls, gen, metas = [], None, None
        for _ in range(reps):
            t = time.perf_counter()
            rows = _generate_batch(client, prompts, n_tokens, conc)
            walls.append(time.perf_counter() - t)
            gen, metas = check_rows(rows, n_tokens, batch, radix_off)
        return walls, gen, metas

    w_short, _, _ = run(short)
    w_long, gen_long, metas_long = run(long)
    out = slope_stats(w_short, w_long, batch, short, long)
    out["tokens"] = gen_long
    out["meta_info_long"] = metas_long
    out["e2e_latency_long_s"] = [m.get("e2e_latency") for m in metas_long]
    out["cached_tokens_long"] = [m.get("cached_tokens") for m in metas_long]
    return out


def iter_sse(resp):
    """Yield parsed `data:` payloads of an SGLang event stream; stops at [DONE]. Each yield is (t_received, obj)."""
    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return
        yield time.perf_counter(), json.loads(payload)


def ttft_once(client: SGLangClient, prompt, new_tokens: int):
    """Wall from request send to the first streamed chunk that carries >= 1 output token. Also the first chunk of any kind,
    and the stream's final bookkeeping (completion_tokens must equal new_tokens: the row ran to its length limit)."""
    t0 = time.perf_counter()
    resp = client.generate_stream({"input_ids": prompt, "sampling_params": sampling_params(new_tokens)})
    first_chunk = first_token = None
    last = None
    n_chunks = 0
    for t, obj in iter_sse(resp):
        n_chunks += 1
        if "error" in obj:
            raise RuntimeError(f"stream error: {obj['error']}")
        last = obj
        if first_chunk is None:
            first_chunk = t - t0
        ids = obj.get("output_ids") or []
        ct = int((obj.get("meta_info") or {}).get("completion_tokens") or 0)
        if first_token is None and (len(ids) >= 1 or ct >= 1):
            first_token = t - t0
    total = time.perf_counter() - t0
    assert first_token is not None, "stream ended without a token"
    meta = (last or {}).get("meta_info") or {}
    got = int(meta.get("completion_tokens") or 0)
    assert got == new_tokens, f"stream produced {got} tokens != {new_tokens} (not the registered workload)"
    return dict(ttft_s=first_token, first_chunk_s=first_chunk, total_s=total, n_chunks=n_chunks, meta_info=meta)


def run_ttft(client: SGLangClient, prompt, reps: int, new_tokens: int):
    ttft_once(client, prompt, new_tokens)                                        # warm (untimed)
    runs = [ttft_once(client, prompt, new_tokens) for _ in range(reps)]
    tt = [r["ttft_s"] for r in runs]
    plen = len(prompt)
    return dict(
        ttft_s=[round(x, 5) for x in tt], ttft_min_s=round(min(tt), 5), ttft_median_s=round(statistics.median(tt), 5),
        first_chunk_s=[round(r["first_chunk_s"], 5) for r in runs], stream_total_s=[round(r["total_s"], 5) for r in runs],
        prefill_tok_s_min=round(plen / min(tt), 1), prefill_tok_s_median=round(plen / statistics.median(tt), 1),
        ttft_new_tokens=new_tokens, meta_info_last=runs[-1]["meta_info"], n_chunks=[r["n_chunks"] for r in runs],
    )


SERVER_KEYS = ("version", "model_path", "revision", "attention_backend", "prefill_attention_backend", "decode_attention_backend",
               "kv_cache_dtype", "dtype", "quantization", "moe_runner_backend", "disable_radix_cache", "max_running_requests",
               "chunked_prefill_size", "schedule_policy", "context_length", "cuda_graph_config", "speculative_algorithm",
               "disable_overlap_schedule", "page_size", "mem_fraction_static", "max_total_num_tokens", "max_prefill_tokens",
               "random_seed", "launch_command")


def _subset(d: dict, keys=SERVER_KEYS) -> dict:
    return {k: d.get(k) for k in keys if isinstance(d, dict)}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    env = os.environ.get
    p.add_argument("--arm", default=env("SC1_SGLANG_ARM", "gptq_default"))
    p.add_argument("--batch", type=int, default=int(env("SC1_SGLANG_BATCH", "1")))
    p.add_argument("--prompts", default=env("SC1_SGLANG_PROMPTS"), required=env("SC1_SGLANG_PROMPTS") is None)
    p.add_argument("--out", default=env("SC1_SGLANG_OUT"), required=env("SC1_SGLANG_OUT") is None)
    p.add_argument("--host", default=env("SC1_SGLANG_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(env("SC1_SGLANG_PORT", "30000")))
    p.add_argument("--model", default=env("SC1_SGLANG_MODEL", "Qwen/Qwen3-30B-A3B-GPTQ-Int4"))
    p.add_argument("--revision", default=env("SC1_SGLANG_REV", "9b534e4318b7ebc3c961a839f13eb18b1833f441"))
    p.add_argument("--server-mode", default=env("SC1_SGLANG_SERVER_MODE"), help="server.sh mode the server was started in (recorded)")
    p.add_argument("--server-info", default=env("SC1_SGLANG_SERVER_INFO"), help="<log>.server_info.json from server.sh (else fetched live)")
    p.add_argument("--engagement", default=env("SC1_SGLANG_ENGAGEMENT"), help="<log>.engagement.json from server.sh (embedded)")
    p.add_argument("--tripwire", default=env("SC1_SGLANG_TRIPWIRE"), help="sglang_tripwire.json from install.sh (torch/flashinfer versions)")
    p.add_argument("--instance-id", default=env("SC1_INSTANCE_ID"))
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--short", type=int, default=SHORT)
    p.add_argument("--long", type=int, default=LONG)
    p.add_argument("--concurrency", choices=("batch", "threads"), default="batch")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--ttft", action="store_true", help="TTFT mode: one-row prompt file (512 or 4096 tokens), stream=true, 8 new tokens")
    p.add_argument("--ttft-new-tokens", type=int, default=TTFT_NEW_TOKENS)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    client = SGLangClient(a.host, a.port, a.timeout)
    if not client.health():
        raise SystemExit(f"server at {client.base} is not healthy (/health)")
    info = json.load(open(a.server_info)) if a.server_info else client.server_info()
    eng = json.load(open(a.engagement)) if a.engagement else None
    trip = json.load(open(a.tripwire)) if a.tripwire else {}
    radix_off = bool(info.get("disable_radix_cache"))
    if a.ttft:
        pf = json.load(open(a.prompts))
        prompts = pf["prompts"]
        assert pf["batch"] == 1 and len(prompts) == 1, "TTFT takes a one-row prompt file"
        plen = len(prompts[0])
        assert plen in TTFT_PROMPT_LENS, f"TTFT prompt must be one of {TTFT_PROMPT_LENS} tokens, got {plen}"
        prompts_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        assert prompts_sha == pf["prompts_sha256"], "prompt file digest mismatch"
        batch = 1
    else:
        prompts, pf, prompts_sha = load_prompts(a.prompts, a.batch)
        batch = a.batch
        plen = PROMPT_LEN
    out = {"engine": "sglang", "arm": a.arm, "sglang_version": info.get("version"), "torch": trip.get("torch"),
           "flashinfer": trip.get("flashinfer"), "model": a.model, "revision": a.revision, "batch": batch,
           "method": ("TTFT: wall from request send to the first streamed chunk carrying >= 1 output token; warm + 3 timed; "
                      "median and min recorded; prefill tok/s = prompt_tokens / ttft" if a.ttft else
                      "slope(32->128) isolates decode; min-of-3 (P20 estimator) and median-of-3 both recorded"),
           "prompts": ("one %d-token window (step_decomp._k8_window), fed as token ids" % plen if a.ttft else
                       "identical token ids to the e4b arms (step_decomp._k8_window, wikitext-2 test, 512-token rows)"),
           "prompts_sha256": prompts_sha, "rows_sha256": pf.get("rows_sha256"), "prompt_tokens": [len(p) for p in prompts],
           "endpoint": "/generate (native, input_ids)", "concurrency": ("stream" if a.ttft else a.concurrency),
           "server_mode": a.server_mode or (eng or {}).get("mode"), "server_args": _subset(info), "engagement": eng,
           "generation": {"temperature": 0.0, "ignore_eos": True, "min_new_tokens": "= max_new_tokens", "greedy": True},
           "vast_instance_id": a.instance_id, "load_s": None,
           "server_startup_s": (eng or {}).get("startup_s"),
           "load_note": "the server is started by server.sh before the arm; its startup (load + Marlin JIT + graph capture + warmup) is server_startup_s"}
    t_all = time.perf_counter()
    if a.ttft:
        out.update(run_ttft(client, prompts[0], a.reps, a.ttft_new_tokens))
        out["ttft_note"] = ("request-level TTFT through the HTTP server (tokenizer manager + scheduler + prefill + first sample + "
                            "detokenise of the first token); prefill cuda graphs per the server mode's cuda_graph_config")
    else:
        out.update(run_slope(client, prompts, batch, radix_off, a.reps, a.concurrency, a.short, a.long))
        out["ttft_note"] = "not measured in the slope arm; see the --ttft receipts (P37 fixture: informational, no ratio)"
    out["wall_total_s"] = round(time.perf_counter() - t_all, 2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    keys = (("arm", "batch", "ttft_min_s", "ttft_median_s", "prefill_tok_s_min", "sglang_version", "prompts_sha256") if a.ttft else
            ("arm", "batch", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "end_to_end_tok_s_long", "sglang_version", "prompts_sha256"))
    print("SC1SGLANG " + json.dumps({k: out.get(k) for k in keys}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
