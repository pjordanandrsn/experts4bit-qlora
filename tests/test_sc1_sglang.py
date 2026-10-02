# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""CPU-only tests for the SC1 SGLang drivers (bench/sc1/sglang/) against a FAKE SGLang HTTP server.

The fake speaks the v0.5.20 native-API shapes the drivers rely on (verified at the tag, see the drivers' docstrings):
/health, /server_info, /flush_cache, /generate (list-of-lists -> list of rows; flat list -> one dict; return_logprob with
logprob_start_len 0 -> input_token_logprobs with a leading None; token_ids_logprob -> output_token_ids_logprobs;
stream=true -> `data: {json}\\n\\n` ... `data: [DONE]\\n\\n`). Its "model" is a deterministic logprob rule of (position, token),
so every expected number here is computed independently of the driver and pinned exactly.

Covered: prompt-digest refusal; the N-token assertion; the slope arithmetic; the K8 index set (prefill AND served means must
equal the rule's mean over exactly ids[prompt_len+1 .. prompt_len+steps]); the served-mode cache bookkeeping (T==1 label when
cached_tokens == prompt_tokens-1 after the first request, the `-partial` label otherwise; the first request after a flush
recomputes everything); the served_tail2 cross-check; the TTFT streaming parse; the k8 digest format.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SGL = REPO / "bench" / "sc1" / "sglang"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SGL / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arm = _load("sc1_sglang_arm", "sc1_sglang_arm.py")
nll = _load("sc1_sglang_nll", "sc1_sglang_nll.py")


# ---------------------------------------------------------------------------------------------------------------------
# the fake server
# ---------------------------------------------------------------------------------------------------------------------
def rule_lp(pos: int, tok: int) -> float:
    """The fake model: logprob of token `tok` at position `pos` (given everything before). Deterministic, never 0."""
    return -((tok * 7919 + pos * 104729) % 1000) / 1000.0 - 0.001


def fake_greedy(pos: int, true_tok: int) -> int:
    """The fake argmax at `pos`: the true token except at every third position (so top-1 agreement is 2/3-ish and pinned)."""
    return true_tok if pos % 3 != 0 else true_tok + 1


class FakeState:
    def __init__(self):
        self.radix_on = False
        self.cache_mode = "exact"        # exact: longest cached prefix, capped at L-1 | partial3: always recompute 3 | none
        self.cached_prefixes: list[tuple[int, ...]] = []
        self.short_rows: dict[int, int] = {}   # row index -> tokens to return instead of N (the stopped-early fault)
        self.ttft_delay_s = 0.0
        self.empty_first_chunk = False
        self.incremental = False
        self.refuse_max_new_0 = False
        self.requests: list[dict] = []
        self.flushes = 0
        self.server_info = {"version": "0.5.20", "attention_backend": "flashinfer", "kv_cache_dtype": "auto",
                            "disable_radix_cache": True, "max_running_requests": 16, "chunked_prefill_size": -1,
                            "schedule_policy": "fcfs", "context_length": 2048, "dtype": "float16", "quantization": None,
                            "moe_runner_backend": "auto", "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 16]},
                                                                                 "prefill": {"backend": "breakable"}},
                            "launch_command": "fake"}

    def match(self, ids):
        """cached_tokens for a prompt: SGLang's cap is input_len - 1 (schedule_batch.py:1633) on a logprob_start_len -1
        request; with logprob_start_len >= 0 the cap is that value (schedule_batch.py:1631-1637)."""
        if not self.radix_on or self.cache_mode == "none":
            return 0
        if self.cache_mode == "partial3":
            return max(0, len(ids) - 3)
        best = 0
        for p in self.cached_prefixes:
            n = 0
            for a, b in zip(p, ids):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return min(best, len(ids) - 1)


STATE = FakeState()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text, code=200):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._text("")
        if self.path in ("/server_info", "/get_server_info"):
            return self._json(STATE.server_info)
        return self._text("not found", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/flush_cache":
            STATE.flushes += 1
            STATE.cached_prefixes.clear()
            return self._text("Cache flushed.\n")
        if self.path != "/generate":
            return self._text("not found", 404)
        STATE.requests.append(body)
        ids = body["input_ids"]
        single = isinstance(ids[0], int)
        rows_ids = [ids] if single else ids
        sp = body.get("sampling_params") or {}
        max_new = int(sp.get("max_new_tokens", 128))
        if max_new == 0 and STATE.refuse_max_new_0:
            return self._json({"error": {"message": "max_new_tokens must be at least 1 (fake)"}}, 400)
        if body.get("stream"):
            return self._stream(rows_ids[0], max_new)
        rows = [self._row(i, r, max_new, body) for i, r in enumerate(rows_ids)]
        return self._json(rows[0] if single else rows)

    def _row(self, i, prompt, max_new, body):
        L = len(prompt)
        cached = STATE.match(prompt)
        n_out = STATE.short_rows.get(i, max_new)
        out_ids = [fake_greedy(L + k, 1000 + k) for k in range(n_out)]
        meta = {"id": f"fake-{i}", "finish_reason": {"type": "length", "length": n_out} if n_out else None,
                "prompt_tokens": L, "completion_tokens": n_out, "cached_tokens": cached, "e2e_latency": 0.01}
        if body.get("return_logprob"):
            start = body.get("logprob_start_len", -1)
            if start is None:
                start = -1
            if start >= 0:
                # SGLang: the radix match is capped at logprob_start_len, so positions >= start are all computed
                cached = min(cached, start)
                meta["cached_tokens"] = cached
                lps = [None] + [[rule_lp(p, prompt[p]), prompt[p], None] for p in range(start + 1, L)]
                meta["input_token_logprobs"] = lps
                k = int(body.get("top_logprobs_num") or 0)
                if k > 0:
                    tops = [None]
                    for p in range(start + 1, L):
                        g = fake_greedy(p, prompt[p])
                        tops.append([[rule_lp(p, g) if g == prompt[p] else rule_lp(p, prompt[p]) + 0.5, g, None]])
                    meta["input_top_logprobs"] = tops
            else:
                meta["input_token_logprobs"] = []
            # output logprobs: the fake's sampled token at the generation position
            if n_out:
                g = fake_greedy(L, prompt_next(prompt, L))
                meta["output_token_logprobs"] = [[rule_lp(L, g), g, None]]
                out_ids = [g] + out_ids[1:]
            til = body.get("token_ids_logprob")
            if til:
                meta["output_token_ids_logprobs"] = [[[rule_lp(L, t), t, None] for t in til]]
        if STATE.radix_on:
            STATE.cached_prefixes.append(tuple(prompt) + tuple(out_ids[:1]))
        return {"text": "", "output_ids": out_ids, "meta_info": meta}

    def _stream(self, prompt, max_new):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(obj):
            data = b"data: " + json.dumps(obj).encode() + b"\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        time.sleep(STATE.ttft_delay_s)
        meta = {"id": "fake-s", "finish_reason": None, "prompt_tokens": len(prompt), "completion_tokens": 0, "cached_tokens": 0}
        if STATE.empty_first_chunk:
            chunk({"text": "", "output_ids": [], "meta_info": dict(meta)})
            time.sleep(STATE.ttft_delay_s)
        ids = []
        for k in range(max_new):
            ids.append(5000 + k)
            m = dict(meta, completion_tokens=k + 1)
            if k == max_new - 1:
                m["finish_reason"] = {"type": "length", "length": max_new}
                m["e2e_latency"] = 0.1
            chunk({"text": None, "output_ids": ([ids[-1]] if STATE.incremental else list(ids)), "meta_info": m})
            time.sleep(0.002)
        done = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
        self.wfile.flush()


def prompt_next(prompt, L):
    """The 'true' next token the fake's greedy is compared against: derived from the window the tests build (see WINDOW)."""
    w = STATE.__dict__.get("window")
    if w is not None and L < len(w) and list(w[:L]) == list(prompt):
        return w[L]
    return 999_999


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield httpd.server_address[1]
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _reset():
    STATE.__init__()
    yield


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def write_prompts(tmp_path, batch, plen=512, tamper=False):
    prompts = [[(i * 1000 + j) % 50000 for j in range(plen)] for i in range(batch)]
    sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    pf = {"prompts": prompts, "batch": batch, "prompts_sha256": ("0" * 64 if tamper else sha),
          "rows_sha256": [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]}
    p = tmp_path / f"prompts_b{batch}.json"
    p.write_text(json.dumps(pf))
    return p


def make_window(prompt_len, steps, extra=0):
    ids = [(i * 31 + 7) % 4000 + 1 for i in range(prompt_len + steps + 1 + extra)]
    return ids


def expected_mean(ids, prompt_len, steps):
    return sum(-rule_lp(i, ids[i]) for i in range(prompt_len + 1, prompt_len + steps + 1)) / steps


# ---------------------------------------------------------------------------------------------------------------------
# speed driver
# ---------------------------------------------------------------------------------------------------------------------
def test_prompt_digest_refusal(tmp_path):
    p = write_prompts(tmp_path, 2, tamper=True)
    with pytest.raises(AssertionError, match="prompt file digest mismatch"):
        arm.load_prompts(str(p), 2)


def test_prompt_contract_batch_and_length(tmp_path):
    p = write_prompts(tmp_path, 2)
    with pytest.raises(AssertionError):
        arm.load_prompts(str(p), 3)                      # batch mismatch
    p2 = write_prompts(tmp_path, 1, plen=100)
    with pytest.raises(AssertionError, match="prompt_len must be 512"):
        arm.load_prompts(str(p2), 1)


def test_slope_arithmetic():
    s = arm.slope_stats([1.0, 1.1, 1.2], [2.0, 2.3, 2.2], batch=16, short=32, long=128)
    assert s["decode_tok_s"] == pytest.approx(96 * 16 / 1.0)                    # min: 2.0 - 1.0
    assert s["decode_ms_per_step"] == pytest.approx(1.0 / 96 * 1e3, abs=1e-4)
    assert s["decode_tok_s_median"] == pytest.approx(round(96 * 16 / 1.1, 1))   # median: 2.2 - 1.1
    assert s["decode_ms_per_step_median"] == pytest.approx(round(1.1 / 96 * 1e3, 4), abs=1e-4)
    assert s["end_to_end_tok_s_long"] == pytest.approx(round(128 * 16 / 2.0, 1))
    assert s["wall_short_s"] == 1.0 and s["wall_long_s"] == 2.0


def test_n_token_assertion(server, tmp_path):
    STATE.short_rows = {2: 31}                                                 # row 2 stops one token early at N=32
    p = write_prompts(tmp_path, 4)
    out = tmp_path / "r.json"
    with pytest.raises(AssertionError, match="stopped early"):
        arm.main(["--prompts", str(p), "--batch", "4", "--port", str(server), "--out", str(out), "--arm", "t"])


def test_slope_arm_receipt_shape(server, tmp_path):
    p = write_prompts(tmp_path, 2)
    out = tmp_path / "r.json"
    rc = arm.main(["--prompts", str(p), "--batch", "2", "--port", str(server), "--out", str(out), "--arm", "gptq_default",
                   "--server-mode", "matched"])
    assert rc == 0
    r = json.loads(out.read_text())
    for k in ("engine", "arm", "batch", "prompts_sha256", "rows_sha256", "walls_short_s", "walls_long_s", "wall_short_s", "wall_long_s",
              "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "decode_ms_per_step_median", "end_to_end_tok_s_long", "tokens",
              "server_args", "generation", "sglang_version", "method", "prompts", "prompt_tokens", "ttft_note"):
        assert k in r, k
    assert r["engine"] == "sglang" and r["batch"] == 2 and r["server_args"]["attention_backend"] == "flashinfer"
    assert len(r["walls_short_s"]) == 3 and len(r["walls_long_s"]) == 3
    assert set(r["tokens"]) == {"0", "1"} and all(len(v) == 128 for v in r["tokens"].values())
    # one warm + 3 timed at each length = 8 batch calls, each a list-of-lists request with the matched sampling params
    gens = [q for q in STATE.requests]
    assert len(gens) == 8 and all(isinstance(q["input_ids"][0], list) and len(q["input_ids"]) == 2 for q in gens)
    assert [q["sampling_params"]["max_new_tokens"] for q in gens] == [32] * 4 + [128] * 4
    assert all(q["sampling_params"] == {"temperature": 0.0, "max_new_tokens": n, "min_new_tokens": n, "ignore_eos": True}
               for q, n in zip(gens, [32] * 4 + [128] * 4))


def test_slope_arm_threads_mode(server, tmp_path):
    p = write_prompts(tmp_path, 3)
    out = tmp_path / "r.json"
    assert arm.main(["--prompts", str(p), "--batch", "3", "--port", str(server), "--out", str(out), "--concurrency", "threads"]) == 0
    assert all(isinstance(q["input_ids"][0], int) for q in STATE.requests) and len(STATE.requests) == 8 * 3


def test_radix_off_arm_refuses_cached_rows(server, tmp_path):
    STATE.radix_on = True                                                      # the fake caches, but the server says radix is OFF
    p = write_prompts(tmp_path, 2)
    out = tmp_path / "r.json"
    with pytest.raises(AssertionError, match="cached_tokens"):
        arm.main(["--prompts", str(p), "--batch", "2", "--port", str(server), "--out", str(out)])


def test_ttft_streaming_parse(server, tmp_path):
    STATE.ttft_delay_s = 0.05
    STATE.empty_first_chunk = True
    p = write_prompts(tmp_path, 1)
    out = tmp_path / "t.json"
    assert arm.main(["--ttft", "--prompts", str(p), "--port", str(server), "--out", str(out), "--arm", "ttft512"]) == 0
    r = json.loads(out.read_text())
    assert len(r["ttft_s"]) == 3 and r["ttft_min_s"] >= 0.10 - 0.005          # delay before the empty chunk + delay before the first token
    assert all(fc < tt for fc, tt in zip(r["first_chunk_s"], r["ttft_s"]))     # the empty chunk is NOT the first token
    assert r["meta_info_last"]["completion_tokens"] == 8 and r["ttft_new_tokens"] == 8
    assert r["prefill_tok_s_min"] == pytest.approx(512 / r["ttft_min_s"], rel=1e-3)   # the receipt rounds ttft to 5 dp
    assert all(q["stream"] is True and q["sampling_params"]["max_new_tokens"] == 8 for q in STATE.requests)


def test_ttft_incremental_stream_and_bad_prompt_len(server, tmp_path):
    STATE.incremental = True
    p = write_prompts(tmp_path, 1)
    out = tmp_path / "t.json"
    assert arm.main(["--ttft", "--prompts", str(p), "--port", str(server), "--out", str(out)]) == 0
    p2 = write_prompts(tmp_path, 1, plen=1024)
    with pytest.raises(AssertionError, match="TTFT prompt must be one of"):
        arm.main(["--ttft", "--prompts", str(p2), "--port", str(server), "--out", str(out)])


# ---------------------------------------------------------------------------------------------------------------------
# NLL scorer
# ---------------------------------------------------------------------------------------------------------------------
def test_k8_sha_is_int64_little_endian():
    ids = [1, 2, 3, 2**40, -5]
    want = hashlib.sha256(b"".join(int(x).to_bytes(8, "little", signed=True) for x in ids)).hexdigest()
    assert nll.k8_sha(ids, prompt_len=2, steps=2) == want                     # 2 + 2 + 1 = all five ids
    assert nll.k8_sha(ids + [9, 9], 2, 2) == want                              # trailing ids are outside the digest
    with pytest.raises(AssertionError):
        nll.k8_sha(ids[:3], 2, 2)


def test_scored_indices_are_k8s():
    assert list(nll.scored_indices(512, 2048)) == list(range(513, 2561))
    assert len(nll.scored_indices(512, 2048)) == 2048
    assert list(nll.scored_indices(4, 6)) == [5, 6, 7, 8, 9, 10]


def test_window_digest_refusal(tmp_path):
    ids = make_window(4, 6)
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"ids": ids, "text_sha": "f" * 64}))
    with pytest.raises(SystemExit, match="window digest mismatch"):
        nll.load_window(str(w), 4, 6)


def _write_window(tmp_path, prompt_len, steps):
    ids = make_window(prompt_len, steps)
    STATE.window = ids
    w = tmp_path / "k8_window_fake.json"
    w.write_text(json.dumps({"ids": ids, "text_sha": nll.k8_sha(ids, prompt_len, steps), "prompt_len": prompt_len, "steps": steps, "source": "fake"}))
    return ids, w


def test_prefill_mode_scores_exactly_the_k8_index_set(server, tmp_path):
    STATE.server_info["disable_radix_cache"] = False
    STATE.radix_on = True
    prompt_len, steps = 4, 6
    ids, w = _write_window(tmp_path, prompt_len, steps)
    out = tmp_path / "p.json"
    assert nll.main(["--window", str(w), "--mode", "prefill", "--port", str(server), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["mean_nll"] == pytest.approx(expected_mean(ids, prompt_len, steps), abs=1e-12)
    assert r["scored_index_range"] == [5, 10] and r["steps"] == 6 and r["prompt_len"] == 4 and r["tokens_scored"] == 6
    assert r["text_sha"] == nll.k8_sha(ids, 4, 6)
    q = STATE.requests[-1]
    assert q["input_ids"] == ids[:11] and q["sampling_params"]["max_new_tokens"] == 0 and q["logprob_start_len"] == 0 and q["return_logprob"] is True
    assert r["per_request_logprob_count_histogram"] == {"11": 1} and r["n_requests"] == 1
    # top-1 agreement from input_top_logprobs: the fake's greedy misses exactly the positions divisible by 3 in 5..10 -> 6, 9
    assert r["top1_agreement"] == pytest.approx(4 / 6)
    assert STATE.flushes == 1


def test_prefill_mode_falls_back_to_one_token(server, tmp_path):
    STATE.server_info["disable_radix_cache"] = False
    STATE.refuse_max_new_0 = True
    prompt_len, steps = 4, 6
    ids, w = _write_window(tmp_path, prompt_len, steps)
    out = tmp_path / "p.json"
    assert nll.main(["--window", str(w), "--mode", "prefill", "--port", str(server), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["prefill_max_new_tokens_used"] == 1 and r["mean_nll"] == pytest.approx(expected_mean(ids, prompt_len, steps), abs=1e-12)


def test_served_mode_generation_position_and_T1_bookkeeping(server, tmp_path):
    STATE.server_info["disable_radix_cache"] = False
    STATE.radix_on = True
    prompt_len, steps = 4, 6
    ids, w = _write_window(tmp_path, prompt_len, steps)
    out = tmp_path / "s.json"
    assert nll.main(["--window", str(w), "--mode", "served", "--port", str(server), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["mean_nll"] == pytest.approx(expected_mean(ids, prompt_len, steps), abs=1e-12)   # same set as prefill
    assert r["n_requests"] == 6
    # request t sends ids[:prompt_len+1+t] and asks for ids[prompt_len+1+t] at the generation position
    reqs = STATE.requests
    for t, q in enumerate(reqs):
        assert q["input_ids"] == ids[:prompt_len + 1 + t], t
        assert q["token_ids_logprob"] == [ids[prompt_len + 1 + t]], t
        assert q["sampling_params"]["max_new_tokens"] == 1 and q["logprob_start_len"] == -1 and q["return_logprob"] is True
    # cache bookkeeping: after the flush, request 0 recomputes its whole 5-token prompt; every later request exactly 1 token
    assert r["recomputed_suffix_histogram"] == {"1": 5, "5": 1}
    assert r["cached_tokens_first_request"] == 0 and r["cache_engaged_every_request_after_first"] is True
    assert r["mode_label"].startswith("served (T==1")
    assert r["per_request_logprob_count_histogram"] == {"0": 6}               # no input logprobs are read in this shape
    assert r["top1_agreement"] == pytest.approx(4 / 6)                         # greedy misses at positions 6 and 9
    assert r["scored_index_range"] == [5, 10]


def test_served_mode_labels_partial_cache(server, tmp_path):
    STATE.server_info["disable_radix_cache"] = False
    STATE.radix_on = True
    STATE.cache_mode = "partial3"                                             # a cache that always leaves 3 tokens uncached
    prompt_len, steps = 4, 6
    ids, w = _write_window(tmp_path, prompt_len, steps)
    out = tmp_path / "s.json"
    assert nll.main(["--window", str(w), "--mode", "served", "--port", str(server), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["recomputed_suffix_histogram"] == {"3": 6}
    assert r["cache_engaged_every_request_after_first"] is False
    assert r["mode_label"].startswith("served-partial") and "NOT a T==1" in r["mode_label"]
    assert r["mean_nll"] == pytest.approx(expected_mean(ids, prompt_len, steps), abs=1e-12)   # the number is still right; the LABEL changes


def test_served_mode_refuses_radix_off_server(server, tmp_path):
    STATE.server_info["disable_radix_cache"] = True
    ids, w = _write_window(tmp_path, 4, 6)
    with pytest.raises(SystemExit, match="radix cache ON"):
        nll.main(["--window", str(w), "--mode", "served", "--port", str(server), "--out", str(tmp_path / "x.json")])


def test_served_tail2_cross_check(server, tmp_path):
    STATE.server_info["disable_radix_cache"] = False
    STATE.radix_on = True
    prompt_len, steps = 4, 6
    ids, w = _write_window(tmp_path, prompt_len, steps)
    out = tmp_path / "s2.json"
    assert nll.main(["--window", str(w), "--mode", "served_tail2", "--port", str(server), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["mean_nll"] == pytest.approx(expected_mean(ids, prompt_len, steps), abs=1e-12)
    for t, q in enumerate(STATE.requests):
        assert q["input_ids"] == ids[:prompt_len + 2 + t] and q["logprob_start_len"] == prompt_len + t and q["sampling_params"]["max_new_tokens"] == 0
    assert r["per_request_logprob_count_histogram"] == {"2": 6}               # [None, (lp of the scored token)]
    assert r["recomputed_suffix_histogram"] == {"2": 5, "6": 1}                # the cap at logprob_start_len = L-2 -> 2 recomputed
    assert r["mode_label"].startswith("served_tail2 (T==2")


# ---------------------------------------------------------------------------------------------------------------------
# A8: SGLang's static memory pool on the chunking-off modes (receipt sc1c-prove-13)
# ---------------------------------------------------------------------------------------------------------------------
SERVER_SH = SGL / "server.sh"
CHUNKING_OFF = ("matched", "kvfp8", "ttft_matched", "quality")
RTX5090_MIB = 32607                         # nvidia-smi memory.total on the SC1 boxes
PRE_GIB, AFTER_LOAD_GIB = 30.69, 14.98      # sc1c-prove-13's SGLang log: "avail mem" before / after loading the GPTQ checkpoint
MIN_VIABLE = 0.5134                         # the same log: "minimum viable = 1 - available/pre"
KV_BYTES_PER_TOKEN = 48 * 4 * 128 * 2 * 2   # Qwen3-30B-A3B: 48 layers x 4 KV heads x head_dim 128 x (K, V) x 2 bytes (16-bit KV)
VLLM_PEAK_GIB = 0.86 + 0.50                 # sc1a-5090-1: vLLM 0.30's largest peak activation + CUDA-graph memory, 8192-token batch


def _flags(mode):
    out = subprocess.run(["bash", "-c", 'source "$1"; sglang_server_flags "$2" 30000', "_", str(SERVER_SH), mode],
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def _flag(flags, name):
    return flags[flags.index(name) + 1] if name in flags else None


def _sglang_0520_auto_fraction(activation_tokens, gpu_mib=RTX5090_MIB, decode_max_bs=16):
    """``arg_groups/memory_hook.py`` ``handle_gpu_memory_settings`` at v0.5.20 for this lane's servers (TP 1, no post-capture
    KV sizing, no DP attention, no DeepEP, no prefill graphs at ``chunked_prefill_size -1``): reserved MB = 512 +
    1.5 x max(tokens, 2048) + 1024 x tp x pp / 8 + 2 x decode max_bs."""
    reserved = 512 + 1.5 * max(activation_tokens, 2048) + 1024 / 8 + 2 * decode_max_bs
    return round((gpu_mib - reserved) / gpu_mib, 3)


def test_a8_sglang_auto_fraction_cannot_hold_the_checkpoint():
    # chunked prefill off -> the activation term is max_prefill_tokens (default 16384): the 0.226 the receipt resolved
    assert _sglang_0520_auto_fraction(16384) == 0.226
    assert _sglang_0520_auto_fraction(16384) < MIN_VIABLE
    # even the lane's real largest extend (16 rows x 512) leaves too little KV for the registered 16 x 2048 under SGLang's rule
    f = _sglang_0520_auto_fraction(16 * 512)
    assert (AFTER_LOAD_GIB - PRE_GIB * (1 - f)) * 2**30 / KV_BYTES_PER_TOKEN < 16 * 2048


def test_a8_chunking_off_modes_pin_the_static_fraction():
    for mode in CHUNKING_OFF:
        f = _flags(mode)
        assert _flag(f, "--chunked-prefill-size") == "-1", mode
        assert _flag(f, "--mem-fraction-static") == "0.75", (mode, f)
    assert "--mem-fraction-static" not in _flags("native")   # native keeps SGLang's own resolution (chunked prefill on)


def test_a8_pinned_fraction_holds_capacity_and_headroom():
    for mode in CHUNKING_OFF:
        f = _flags(mode)
        frac = float(_flag(f, "--mem-fraction-static"))
        assert frac > MIN_VIABLE, mode
        per_token = KV_BYTES_PER_TOKEN // (2 if _flag(f, "--kv-cache-dtype") == "fp8_e4m3" else 1)
        kv_tokens = (AFTER_LOAD_GIB - PRE_GIB * (1 - frac)) * 2**30 / per_token   # kv_cache_configurator._profile_available_bytes
        running = int(_flag(f, "--max-running-requests"))
        need = int(_flag(f, "--context-length")) * running                        # the registered capacity of this server
        assert kv_tokens >= need, (mode, int(kv_tokens), need)
        assert min(running, int(kv_tokens) // 2) == running, mode                  # resolve_max_num_reqs keeps the request cap
        assert PRE_GIB * (1 - frac) >= 4 * VLLM_PEAK_GIB, mode                     # headroom outside the static pool


def test_a8_engagement_refuses_a_drifted_fraction_or_a_short_pool():
    text = SERVER_SH.read_text()
    assert "SC1_MFS=$SC1_SGLANG_MEM_FRACTION_STATIC" in text
    assert 'need(abs(float(info.get("mem_fraction_static") or 0) - mfs) < 1e-9' in text
    assert 'need(int(info.get("max_total_num_tokens") or 0) >= cap' in text


def test_a9_sglang_runs_the_gptq_scales_dtype():
    # SGLang 0.5.20's GPTQ Marlin MoE asserts hidden_states.dtype == w1_scale.dtype (fused_marlin_moe.py); the lane's
    # checkpoint declares torch_dtype float16 and stores float16 scales, and vLLM's `auto` resolved float16 on it (sc1a-5090-1)
    for mode in CHUNKING_OFF:
        assert _flag(_flags(mode), "--dtype") == "float16", mode
    assert "--dtype" not in _flags("native")            # native keeps SGLang's auto: the checkpoint's float16
    ob = (SGL / "one_batch.sh").read_text()
    assert "--dtype float16" in ob and "bfloat16" not in ob
    assert 'need(info.get("dtype") == "float16"' in SERVER_SH.read_text()
