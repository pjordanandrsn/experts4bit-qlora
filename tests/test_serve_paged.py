# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The continuous-batching HTTP server (:mod:`experts4bit_qlora.serve_paged`), everything above the GPU seam.

A scripted :class:`~experts4bit_qlora.engines.scheduler.StepRunner` and a tiny byte-fallback tokenizer
stand in for the model, so what is pinned here is the CONTRACT the serving benchmarks read: the OpenAI
shapes vLLM's and SGLang's clients parse, greedy-only stated as a 400, EOS stop against ``ignore_eos``,
``max_tokens`` honoured exactly, the SSE sequence (per-token chunks, finish_reason on the last token,
usage, [DONE]), code points never split across chunks, TTFT from HTTP arrival in the trace, and that
concurrent requests share decode steps (the runner saw a decode with more than one rid). The real
engine thread runs in every test; only the step is fake.
"""
import asyncio
import json
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from experts4bit_qlora.engines.scheduler import ContinuousScheduler  # noqa: E402
from experts4bit_qlora.serve_paged import (  # noqa: E402
    _batched_graph_grouping,
    build_engine,
    EngineParts,
    IncrementalDetokenizer,
    PagedEngine,
    PagedServeConfig,
    PagedStream,
    _apply_fusions,
    create_app,
    stream_events,
)

EOS = 2
EURO_HI, EURO_LO = 100, 101          # "€" = e2 82 ac split across two byte-fallback tokens


class TinyTokenizer:
    """Whitespace words <-> ids, special ids 0..2, and two byte-fallback ids whose text is a partial code point."""

    eos_token_id = EOS
    bos_token_id = 1
    pad_token_id = 0
    chat_template = "{{ messages }}"

    def __init__(self):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.special = {0, 1, 2}
        self.bytes = {EURO_HI: b"\xe2\x82", EURO_LO: b"\xac"}
        for w in ("the", "cat", "sat", "on", "mat", "hello", "world", "user", "assistant"):
            self._add(w)

    def _add(self, w):
        if w not in self.vocab:
            i = 10 + len(self.vocab)
            self.vocab[w] = i
            self.inv[i] = w
        return self.vocab[w]

    def encode(self, text, add_special_tokens=True):
        ids = [self._add(w) for w in text.split()]
        return ([1] if add_special_tokens else []) + ids

    def decode(self, ids, skip_special_tokens=True, **kw):
        buf = b""
        for i in ids:
            if i in self.special:
                if not skip_special_tokens:
                    buf += self.inv[i].encode()
                continue
            if i in self.bytes:
                buf += self.bytes[i]
            else:
                buf += (" " + self.inv.get(i, f"<{i}>")).encode()
        return buf.decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = " ".join(f"{m['role']} {m['content']}" for m in messages)
        return text + (" assistant" if add_generation_prompt else "")


class ScriptedRunner:
    """Emits each prompt's script (default token 1000 + i) and records every decode batch."""

    def __init__(self, scripts=None, step_delay=0.0):
        self.scripts = {tuple(k): list(v) for k, v in (scripts or {}).items()}
        self.prompts, self.cursor = {}, {}
        self.decode_batches, self.prefill_calls, self.freed = [], [], []
        self.step_delay = step_delay
        self.lock = threading.Lock()

    def bind(self, rid, slot, prompt):
        self.prompts[rid] = tuple(prompt)
        self.cursor[rid] = 0

    def _next(self, rid):
        i = self.cursor[rid]
        self.cursor[rid] += 1
        s = self.scripts.get(self.prompts[rid])
        return s[i] if s and i < len(s) else 1000 + i

    def run_prefill(self, chunks):
        if self.step_delay:
            time.sleep(self.step_delay)
        self.prefill_calls.append(list(chunks))
        return {rid: self._next(rid) for rid, start, n in chunks if start + n >= len(self.prompts[rid])}

    def run_decode(self, rids):
        if self.step_delay:
            time.sleep(self.step_delay)
        with self.lock:
            self.decode_batches.append(list(rids))
        return {r: self._next(r) for r in rids}

    def free_slot(self, rid):
        self.freed.append(rid)


def _parts(runner, *, max_seqs=4, chunk=8, model_id="tiny/moe"):
    sched = ContinuousScheduler(runner=runner, max_seqs=max_seqs, kv_slots=max_seqs, chunk_tokens=chunk,
                                max_prefill_tokens_per_step=chunk)
    return EngineParts(scheduler=sched, tokenizer=TinyTokenizer(), eos_ids=frozenset({EOS}),
                       info={"model_id": model_id, "moe_layers": 1}, runner=runner)


def _client(runner, tmp_path=None, **cfg_kw):
    kw = dict(model="tiny/moe", max_seqs=4, max_tokens_per_seq=64, chunk_tokens=8)
    if tmp_path is not None:
        kw["trace_path"] = str(tmp_path / "trace.jsonl")
    kw.update(cfg_kw)
    cfg = PagedServeConfig(**kw)
    engine = PagedEngine(cfg, _parts(runner, max_seqs=cfg.max_seqs, chunk=cfg.chunk_tokens))
    return TestClient(create_app(cfg, engine=engine)), engine


def _sse(resp):
    """Parse an SSE body into the list of ``data:`` payloads ('[DONE]' kept as the string)."""
    out = []
    for line in resp.text.split("\n\n"):
        line = line.strip()
        if not line:
            continue
        assert line.startswith("data: "), line
        body = line[len("data: "):]
        out.append(body if body == "[DONE]" else json.loads(body))
    return out


PROMPT = "the cat sat"
W = TinyTokenizer().vocab                      # word -> id, deterministic across instances
PROMPT_IDS = tuple(TinyTokenizer().encode(PROMPT))        # <s> the cat sat
HELLO, WORLD, USER, ASSISTANT = W["hello"], W["world"], W["user"], W["assistant"]


# ------------------------------------------------------------------ shapes --

def test_health_and_models_report_the_served_stack():
    runner = ScriptedRunner()
    client, engine = _client(runner)
    with client as c:
        h = c.get("/health").json()
        assert h["status"] == "ready"
        assert h["engine"]["max_seqs"] == 4 and h["engine"]["kv_slots"] == 4
        assert h["engine"]["max_tokens_per_seq"] == 64 and h["engine"]["chunk_tokens"] == 8
        assert h["levers"]["moe_layers"] == 1                 # the census is what /health is for
        assert h["eos_token_ids"] == [EOS]
        assert h["sampling"] == {"greedy_only": True, "logprobs": False, "stop_strings": False}
        m = c.get("/v1/models").json()
        assert m["object"] == "list"
        assert [d["id"] for d in m["data"]] == ["tiny/moe"]
        assert m["data"][0]["max_model_len"] == 64


def test_non_streaming_completion_shape_and_usage():
    runner = ScriptedRunner({PROMPT_IDS: [HELLO, WORLD, USER, ASSISTANT, W["mat"]]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 3})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["model"] == "tiny/moe"
        assert body["choices"] == [{"index": 0, "text": " hello world user", "logprobs": None, "finish_reason": "length"}]
        assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}


def test_streaming_sse_sequence_with_usage_and_done():
    runner = ScriptedRunner({PROMPT_IDS: [HELLO, WORLD, USER]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 3, "stream": True,
                                            "stream_options": {"include_usage": True}})
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse(r)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events[:-1] if e["choices"]]
    usage_only = [e for e in events[:-1] if not e["choices"]]
    # one chunk per generated token, text deltas that concatenate to the completion
    assert [ch["choices"][0]["text"] for ch in chunks] == [" hello", " world", " user"]
    assert [ch["choices"][0]["finish_reason"] for ch in chunks] == [None, None, "length"]
    assert all(ch["object"] == "text_completion" and ch["id"] == chunks[0]["id"] for ch in chunks)
    # the last token's chunk carries usage too (SGLang's client reads it there); then the usage-only chunk vLLM asks for
    assert chunks[-1]["usage"]["completion_tokens"] == 3
    assert len(usage_only) == 1 and usage_only[0]["usage"] == {"prompt_tokens": 4, "completion_tokens": 3,
                                                                "total_tokens": 7}
    assert events.index(usage_only[0]) == len(events) - 2       # usage chunk right before [DONE]


def test_streaming_without_include_usage_sends_no_choices_less_chunk():
    """SGLang's client indexes choices[0] on every chunk: a usage-only chunk it did not ask for would crash it."""
    runner = ScriptedRunner({PROMPT_IDS: [HELLO, WORLD]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 2, "stream": True,
                                            "ignore_eos": True, "temperature": 0.0, "best_of": 1})
        events = _sse(r)
    assert events[-1] == "[DONE]"
    assert all(e["choices"] for e in events[:-1])


def test_multibyte_code_point_is_never_split_across_chunks():
    runner = ScriptedRunner({PROMPT_IDS: [EURO_HI, EURO_LO, HELLO]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 3, "stream": True})
        events = _sse(r)
        full = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 3}).json()
    texts = [e["choices"][0]["text"] for e in events[:-1]]
    assert texts == ["", "€", " hello"]          # the first byte is held, the second completes the character
    assert "�" not in "".join(texts)
    assert "".join(texts) == full["choices"][0]["text"] == "€ hello"


def test_incremental_detokenizer_flush_emits_a_truncated_tail_once():
    d = IncrementalDetokenizer(TinyTokenizer(), [W["the"]], skip_special_tokens=True)
    assert d.push(HELLO) == " hello"
    assert d.push(EURO_HI) == ""
    assert d.flush() == "�"             # a request that ended mid-character shows the replacement char
    assert d.flush() == ""


# ------------------------------------------------------------- semantics --

def test_nonzero_temperature_is_a_400_not_silently_greedy():
    runner = ScriptedRunner()
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "temperature": 0.7})
        assert r.status_code == 400
        assert "greedy" in r.json()["detail"]
        for ok in ({"temperature": 0}, {"temperature": 0.0}, {}, {"top_p": 0.9, "seed": 3, "top_k": 5}):
            assert c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 1, **ok}
                          ).status_code == 200, ok


@pytest.mark.parametrize("bad, needle", [
    ({"logprobs": 1}, "logprobs"),
    ({"echo": True}, "echo"),
    ({"n": 2}, "n must be 1"),
    ({"best_of": 2}, "best_of"),
    ({"stop": ["\n"]}, "stop strings"),
    ({"repetition_penalty": 1.2}, "penalties"),
    ({"suffix": "x"}, "suffix"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"prompt": []}, "prompt"),
    ({"prompt": "", "add_special_tokens": False}, "empty prompt"),
    ({"prompt": ["a", "b"]}, "one request per prompt"),
])
def test_unsupported_request_fields_are_stated_400s(bad, needle):
    runner = ScriptedRunner()
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 2, **bad})
        assert r.status_code == 400, r.text
        assert needle in r.json()["detail"]


def test_vllm_client_payload_is_accepted_verbatim():
    """What vllm bench serve's async_request_openai_completions POSTs at v0.30.0: logprobs null, repetition 1.0."""
    runner = ScriptedRunner()
    client, engine = _client(runner)
    payload = {"model": "tiny/moe", "prompt": PROMPT, "repetition_penalty": 1.0, "max_tokens": 2, "logprobs": None,
               "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    with client as c:
        r = c.post("/v1/completions", json=payload, headers={"x-request-id": "bench-7"})
        assert r.status_code == 200, r.text
        assert _sse(r)[0]["id"] == "bench-7"                 # the request id rides into the stream and the trace


def test_unknown_model_name_is_a_404_with_the_served_names():
    runner = ScriptedRunner()
    client, engine = _client(runner, served_names=("qwen3-30b",))
    with client as c:
        r = c.post("/v1/completions", json={"model": "other/model", "prompt": PROMPT})
        assert r.status_code == 404 and "qwen3-30b" in r.json()["detail"]
        assert c.post("/v1/completions", json={"model": "qwen3-30b", "prompt": PROMPT, "max_tokens": 1}
                      ).status_code == 200


def test_prompt_as_token_ids_is_not_retokenised():
    runner = ScriptedRunner({(7, 8, 9): [HELLO]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": [7, 8, 9], "max_tokens": 1})
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["prompt_tokens"] == 3
        assert r.json()["choices"][0]["text"] == " hello"
    assert runner.prompts[0] == (7, 8, 9)


def test_add_special_tokens_is_honoured_for_string_prompts():
    runner = ScriptedRunner()
    client, engine = _client(runner)
    with client as c:
        a = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 1}).json()
        b = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 1,
                                            "add_special_tokens": False}).json()
    assert a["usage"]["prompt_tokens"] == 4 and b["usage"]["prompt_tokens"] == 3


def test_eos_stops_the_request_unless_ignore_eos():
    runner = ScriptedRunner({PROMPT_IDS: [HELLO, WORLD, EOS, USER, ASSISTANT]})
    client, engine = _client(runner)
    with client as c:
        stop = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 5}).json()
        assert stop["choices"][0] == {"index": 0, "text": " hello world", "logprobs": None, "finish_reason": "stop"}
        assert stop["usage"]["completion_tokens"] == 3          # the EOS token was computed and is counted (vLLM's rule)
        run = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 5,
                                              "ignore_eos": True}).json()
        assert run["choices"][0]["finish_reason"] == "length"
        assert run["usage"]["completion_tokens"] == 5
        assert run["choices"][0]["text"] == " hello world user assistant"      # EOS text skipped, decoding continued
    # the engine freed the stopped request's slot at the EOS step: three tokens, not five, were decoded for it
    assert runner.cursor[0] == 3 and runner.cursor[1] == 5


def test_min_tokens_suppresses_an_early_eos():
    runner = ScriptedRunner({PROMPT_IDS: [HELLO, EOS, WORLD, EOS, USER]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 5, "min_tokens": 3}
                   ).json()
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["completion_tokens"] == 4           # the EOS at position 2 was ignored; the one at 4 stopped it


def test_stop_token_ids_apply_even_with_ignore_eos():
    runner = ScriptedRunner({PROMPT_IDS: [HELLO, WORLD, 99, USER]})
    client, engine = _client(runner)
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 4, "ignore_eos": True,
                                            "stop_token_ids": [99]}).json()
    assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["completion_tokens"] == 3


def test_max_tokens_is_the_exact_bound():
    runner = ScriptedRunner()
    client, engine = _client(runner)
    with client as c:
        for n in (1, 7, 20):
            r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": n}).json()
            assert r["usage"]["completion_tokens"] == n and r["choices"][0]["finish_reason"] == "length"


def test_prompt_plus_max_tokens_beyond_the_sequence_window_is_refused_not_clamped():
    runner = ScriptedRunner()
    client, engine = _client(runner)          # max_tokens_per_seq = 64
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": list(range(10, 50)), "max_tokens": 30})
        assert r.status_code == 400 and "E4B_PAGED_MAX_TOKENS_PER_SEQ" in r.json()["detail"]
        ok = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": list(range(10, 50)), "max_tokens": 24})
        assert ok.status_code == 200, ok.text
        assert ok.json()["usage"]["completion_tokens"] == 24
    assert runner.prefill_calls[0][0][2] == 8             # chunked prefill, 8 tokens per step, not swallowed whole
    assert len(runner.prefill_calls) == 5                 # 40 prompt tokens / 8


# --------------------------------------------------------- engine clocks --

def test_trace_records_ttft_from_http_arrival(tmp_path):
    runner = ScriptedRunner(step_delay=0.005)
    client, engine = _client(runner, tmp_path)
    with client as c:
        c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 3})
        st = c.get("/stats").json()
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["prompt_len"] == 4 and rec["out_len"] == 3 and rec["finish_reason"] == "length"
    assert rec["first_token_at"] > rec["arrival"] and rec["ttft"] == pytest.approx(rec["first_token_at"] - rec["arrival"])
    assert rec["admitted_at"] >= rec["arrival"]
    assert rec["finished_at"] >= rec["first_token_at"]
    assert rec["arrival_epoch"] > 1.6e9
    assert st["scheduler"]["completed"] == 1 and st["scheduler"]["ttft_p50"] == pytest.approx(rec["ttft"])
    assert st["records_recent"][0]["request_id"] == rec["request_id"]


def test_concurrent_requests_share_decode_steps():
    runner = ScriptedRunner(step_delay=0.01)
    client, engine = _client(runner)
    body = {"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 12}
    with client as c:
        with ThreadPoolExecutor(max_workers=3) as ex:
            rs = list(ex.map(lambda _: c.post("/v1/completions", json=body), range(3)))
    assert all(r.status_code == 200 for r in rs)
    assert all(r.json()["usage"]["completion_tokens"] == 12 for r in rs)
    assert max(len(b) for b in runner.decode_batches) > 1, runner.decode_batches


def test_requests_past_kv_capacity_wait_in_the_queue_and_are_not_evicted(tmp_path):
    runner = ScriptedRunner(step_delay=0.005)
    client, engine = _client(runner, tmp_path, max_seqs=1)
    body = {"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 6}
    with client as c:
        with ThreadPoolExecutor(max_workers=2) as ex:
            rs = list(ex.map(lambda _: c.post("/v1/completions", json=body), range(2)))
    assert all(r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 6 for r in rs)
    rows = sorted((json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()),
                  key=lambda r: r["arrival"])
    assert all(len(b) == 1 for b in runner.decode_batches)         # one slot: never two residents
    second = rows[1]
    assert second["admitted_at"] >= rows[0]["finished_at"]          # admitted only when the slot freed
    assert second["queue_wait"] > 0 and second["ttft"] > second["queue_wait"]
    assert runner.freed == [0, 1]


def test_max_queue_gives_503_with_retry_after():
    runner = ScriptedRunner(step_delay=0.02)
    client, engine = _client(runner, max_queue=1)
    body = {"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 10}
    with client as c:
        with ThreadPoolExecutor(max_workers=2) as ex:
            first = ex.submit(c.post, "/v1/completions", json=body)
            time.sleep(0.05)
            second = c.post("/v1/completions", json=body)
            assert first.result().status_code == 200
    assert second.status_code == 503 and "Retry-After" in second.headers


def _stream_for(prompt_ids, max_tokens):
    return PagedStream(queue=asyncio.Queue(), request_id="r", prompt_ids=list(prompt_ids), max_tokens=max_tokens,
                       stop_ids=None, min_tokens=0, arrival=time.monotonic(), arrival_epoch=time.time())


def test_abort_frees_the_slot_mid_generation():
    """Starlette's TestClient buffers a streaming body to completion, so a mid-stream disconnect cannot be
    staged through HTTP; this drives the engine from a real event loop the way the handler does."""
    runner = ScriptedRunner(step_delay=0.01)
    cfg = PagedServeConfig(model="tiny/moe", max_seqs=4, max_tokens_per_seq=64, chunk_tokens=8)
    engine = PagedEngine(cfg, _parts(runner))

    async def go():
        engine.start(asyncio.get_running_loop())
        stream = _stream_for(PROMPT_IDS, 50)
        engine.submit(stream)
        kind, _ = await asyncio.wait_for(stream.queue.get(), 5)
        assert kind == "tokens"
        engine.abort(stream)
        while True:
            kind, payload = await asyncio.wait_for(stream.queue.get(), 5)
            if kind == "done":
                return payload

    try:
        rest, finish, rec = asyncio.run(go())
    finally:
        engine.shutdown()
    assert finish == "abort" and rec["finish_reason"] == "abort"
    assert rec["out_len"] < 50 and runner.cursor[0] < 50        # it did not decode to max_tokens after the abort
    assert runner.freed == [0]
    st = engine.stats()["scheduler"]
    assert st["aborted"] == 1 and st["completed"] == 0 and st["kv_slots_free"] == 4 and st["in_flight"] == 0


def test_closing_the_sse_generator_aborts_the_request():
    """What a client disconnect does to the handler: Starlette closes the body generator, whose finally aborts."""
    runner = ScriptedRunner(step_delay=0.01)
    cfg = PagedServeConfig(model="tiny/moe", max_seqs=4, max_tokens_per_seq=64, chunk_tokens=8)
    engine = PagedEngine(cfg, _parts(runner))

    async def go():
        engine.start(asyncio.get_running_loop())
        stream = _stream_for(PROMPT_IDS, 50)
        engine.submit(stream)
        detok = IncrementalDetokenizer(TinyTokenizer(), PROMPT_IDS)
        gen = stream_events(engine, stream, detok, lambda t, f, u: {"choices": [{"text": t, "finish_reason": f}]},
                            lambda u: {"choices": [], "usage": u}, False)
        first = await asyncio.wait_for(gen.__anext__(), 5)
        assert first.startswith("data: ")
        await gen.aclose()                                   # the disconnect
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not stream.finished:
            await asyncio.sleep(0.01)
        return stream.finished

    try:
        assert asyncio.run(go())
    finally:
        engine.shutdown()
    assert runner.freed == [0] and runner.cursor[0] < 50
    assert engine.stats()["scheduler"]["aborted"] == 1


# ----------------------------------------------------------- chat + auth --

def test_chat_completions_apply_the_template_and_stream_deltas():
    runner = ScriptedRunner()
    client, engine = _client(runner)
    msgs = [{"role": "user", "content": "hello"}]
    with client as c:
        full = c.post("/v1/chat/completions", json={"model": "tiny/moe", "messages": msgs, "max_tokens": 2}).json()
        assert full["object"] == "chat.completion"
        assert full["choices"][0]["message"] == {"role": "assistant", "content": " <1000> <1001>"}
        assert full["usage"]["prompt_tokens"] == 3 and full["choices"][0]["finish_reason"] == "length"
        r = c.post("/v1/chat/completions", json={"model": "tiny/moe", "messages": msgs, "max_tokens": 2,
                                                 "stream": True, "stream_options": {"include_usage": True}})
        events = _sse(r)
    chunks = [e for e in events[:-1] if e["choices"]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": " <1000>"}
    assert chunks[1]["choices"][0]["delta"] == {"content": " <1001>"}
    assert chunks[1]["choices"][0]["finish_reason"] == "length"
    assert all(e["object"] == "chat.completion.chunk" for e in events[:-1])
    assert events[-1] == "[DONE]"
    assert runner.prompts[0] == tuple(TinyTokenizer().encode("user hello assistant", add_special_tokens=False))


def test_token_gate_when_set():
    runner = ScriptedRunner()
    client, engine = _client(runner, token="s3cret")
    with client as c:
        assert c.get("/health").status_code == 200
        body = {"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 1}
        assert c.post("/v1/completions", json=body).status_code == 401
        assert c.post("/v1/completions", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_requests_while_loading_get_503_and_a_failed_build_is_reported():
    gate = threading.Event()

    def slow_builder(cfg):
        gate.wait(5)
        raise RuntimeError("E4B_SERVE_EXP_INT4=1 but enable_serve_experts_int4 patched 0 layers")

    cfg = PagedServeConfig(model="tiny/moe")
    engine = PagedEngine(cfg, None, builder=slow_builder)
    with TestClient(create_app(cfg, engine=engine)) as c:
        assert c.get("/health").json()["status"] == "loading"
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT})
        assert r.status_code == 503 and "Retry-After" in r.headers
        gate.set()
        deadline = time.time() + 5
        while time.time() < deadline and engine.state != "error":
            time.sleep(0.01)
        h = c.get("/health").json()
        assert h["status"] == "error" and "patched 0 layers" in h["error"]
        assert c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT}).status_code == 500


def test_config_from_env_and_defaults(monkeypatch):
    for k in ("E4B_HOST", "E4B_PORT", "E4B_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("E4B_PAGED_MODEL", "Qwen/Qwen3-30B-A3B")
    monkeypatch.setenv("E4B_PAGED_GRAPHS", "1")
    monkeypatch.setenv("E4B_PAGED_BUCKETS", "1,2,4,8,16")
    monkeypatch.setenv("E4B_PAGED_MAX_SEQS", "16")
    monkeypatch.setenv("E4B_PAGED_TRACE", "/tmp/t.jsonl")
    cfg = PagedServeConfig.from_env()
    assert cfg.host == "127.0.0.1" and cfg.port == 8778                   # localhost by default, serve.py's port + 1
    assert cfg.graphs and cfg.buckets == (1, 2, 4, 8, 16) and cfg.max_seqs == 16
    assert cfg.max_tokens_per_seq == 4096 and cfg.chunk_tokens == 512 and cfg.prefill_budget == 512
    assert cfg.max_tokens_limit == 4095 and cfg.placement == "all-vram"
    monkeypatch.setenv("E4B_PAGED_PLACEMENT", "nvme")
    with pytest.raises(ValueError, match="all-vram or solver"):
        PagedServeConfig.from_env()


# ---------------------------------------------------------------- fusions --

FOLD_FLAGS = ("E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")


def _fake_attention_model():
    """A CPU stand-in the REAL qkv_fuse.fuse_qkv accepts: a module whose class is named Qwen3MoeAttention with
    unbiased dense q/k/v projections (the dense branch of the fusion), so the test exercises fuse_qkv's own
    function-body imports of the folds rather than a stand-in for fuse_qkv."""
    torch = pytest.importorskip("torch")

    class Qwen3MoeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 4, bias=False)
            self.v_proj = torch.nn.Linear(8, 4, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.head_dim = 4
            self.config = types.SimpleNamespace()

    return torch.nn.Sequential(Qwen3MoeAttention())


def _fake_folds(monkeypatch):
    """Fold functions that need no kernel: each records its call and returns a distinct count."""
    from experts4bit_qlora.engines import glue_fuse, glue_r2, router_epilogue
    calls = []
    monkeypatch.setattr(glue_fuse, "fuse_t1_glue", lambda model: calls.append("glue") or 3)
    monkeypatch.setattr(glue_r2, "fuse_t1_glue_r2", lambda model: calls.append("r2") or (2, 1))
    monkeypatch.setattr(router_epilogue, "fuse_router_epilogue", lambda model: calls.append("epi") or 4)
    return calls


def _fold_functions():
    from experts4bit_qlora.engines import glue_fuse, glue_r2, router_epilogue
    return (glue_fuse.fuse_t1_glue, glue_r2.fuse_t1_glue_r2, router_epilogue.fuse_router_epilogue)


def test_fused_branch_with_fold_flags_reports_the_folds_fuse_qkv_applied(monkeypatch):
    """The registered B=1 stack is --fuse-qkv WITH the fold flags set: fuse_qkv calls the folds itself, the
    server must not refuse the combination, and the census must carry what the folds returned -- captured through
    fuse_qkv's own function-body imports, not re-run and not a literal 0."""
    for k in FOLD_FLAGS:
        monkeypatch.setenv(k, "1")
    calls = _fake_folds(monkeypatch)
    before = _fold_functions()
    out = _apply_fusions(_fake_attention_model(), PagedServeConfig(model="x", fuse_qkv=True))
    assert out == {"fuse_qkv_n": 1, "fuse_t1_glue_n": 3, "fuse_t1_glue_r2_n": [2, 1], "fuse_router_epilogue_n": 4}
    assert calls == ["glue", "r2", "epi"], "each fold called once, by fuse_qkv, in its order"
    assert _fold_functions() == before, "the recording wrappers were not restored"


def test_fused_branch_restores_the_fold_functions_when_fuse_qkv_raises(monkeypatch):
    from experts4bit_qlora.engines import qkv_fuse
    _fake_folds(monkeypatch)
    before = _fold_functions()

    def boom(model):
        raise RuntimeError("biased q/k/v projections")

    monkeypatch.setattr(qkv_fuse, "fuse_qkv", boom)
    with pytest.raises(RuntimeError, match="biased"):
        _apply_fusions(_fake_attention_model(), PagedServeConfig(model="x", fuse_qkv=True))
    assert _fold_functions() == before


def test_fused_branch_refuses_a_fuse_qkv_that_skips_the_folds(monkeypatch):
    """If qkv_fuse ever stops applying the folds at its assembly point, the census cannot be reported from it
    and the server must say so rather than print zeros."""
    from experts4bit_qlora.engines import qkv_fuse
    _fake_folds(monkeypatch)
    monkeypatch.setattr(qkv_fuse, "fuse_qkv", lambda model: 1)
    with pytest.raises(RuntimeError, match="without calling"):
        _apply_fusions(_fake_attention_model(), PagedServeConfig(model="x", fuse_qkv=True))


def test_fused_branch_refuses_a_vacuous_fusion(monkeypatch):
    torch = pytest.importorskip("torch")
    _fake_folds(monkeypatch)
    with pytest.raises(RuntimeError, match="matched no attention module"):
        _apply_fusions(torch.nn.Sequential(torch.nn.Linear(2, 2)), PagedServeConfig(model="x", fuse_qkv=True))


def test_unfused_branch_calls_the_folds_directly_and_reports_them(monkeypatch):
    calls = _fake_folds(monkeypatch)
    out = _apply_fusions(object(), PagedServeConfig(model="x", fuse_qkv=False))
    assert out == {"fuse_qkv_n": 0, "fuse_t1_glue_n": 3, "fuse_t1_glue_r2_n": [2, 1], "fuse_router_epilogue_n": 4}
    assert calls == ["glue", "r2", "epi"]



# --- batched decode graphs take the harness's capture-safe grouping (lane SC1 proof sc1a-prove-7) -----------------------

def _grouping_flags(monkeypatch, dg=False, fs=True):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [dg])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [fs])
    return hr


def test_batched_graphs_take_the_harness_capture_safe_grouping(monkeypatch):
    """sc1a-prove-7: B=16 bucket 2 failed to capture because the T > 1 MoE step took the EAGER grouping's host-size sync;
    step_decomp's batched lane sets DEVICE_GROUPING and clears FORCE_SINGLETON_GROUPS before capturing."""
    hr = _grouping_flags(monkeypatch)
    out = _batched_graph_grouping(PagedServeConfig(model="m", graphs=True, max_seqs=16, buckets=(1, 2, 4, 8, 16)))
    assert out == {"device_grouping": True, "force_singleton_groups": False}
    assert hr.DEVICE_GROUPING == [True] and hr.FORCE_SINGLETON_GROUPS == [False]


def test_b1_and_eager_servers_leave_grouping_at_its_defaults(monkeypatch):
    """The harness's B=1 lane (b1d) and every eager run keep the library defaults; so does the server."""
    hr = _grouping_flags(monkeypatch, dg=False, fs=True)
    out = _batched_graph_grouping(PagedServeConfig(model="m", graphs=True, max_seqs=1, buckets=(1,)))
    assert out == {"device_grouping": False, "force_singleton_groups": True}
    out = _batched_graph_grouping(PagedServeConfig(model="m", graphs=False, max_seqs=16))
    assert out == {"device_grouping": False, "force_singleton_groups": True}
    assert hr.DEVICE_GROUPING == [False] and hr.FORCE_SINGLETON_GROUPS == [True]


def test_batched_graphs_refuse_the_solver_placement(monkeypatch):
    hr = _grouping_flags(monkeypatch)
    with pytest.raises(ValueError, match="all-resident"):
        _batched_graph_grouping(PagedServeConfig(model="m", graphs=True, max_seqs=16, placement="solver"))
    assert hr.DEVICE_GROUPING == [False]


def test_build_engine_sets_grouping_before_it_captures():
    import inspect
    src = inspect.getsource(build_engine)
    assert "_batched_graph_grouping(cfg)" in src
    assert src.index("_batched_graph_grouping(cfg)") < src.index("runner.enable_decode_graphs(")
    assert '"grouping": grouping' in src            # the census reports what was set


def test_graphs_default_to_auto_on_a_cuda_all_vram_server(monkeypatch):
    """Lane P109 (e4b#770): E4B_PAGED_GRAPHS defaults to auto -- on for CUDA at all-vram, eager elsewhere."""
    from experts4bit_qlora.serve_paged import _graphs_env
    for k in ("E4B_PAGED_GRAPHS", "E4B_PAGED_DEVICE", "E4B_PAGED_PLACEMENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("E4B_PAGED_MODEL", "Qwen/Qwen3-30B-A3B")
    assert PagedServeConfig.from_env().graphs is True
    monkeypatch.setenv("E4B_PAGED_PLACEMENT", "solver")
    assert PagedServeConfig.from_env().graphs is False
    monkeypatch.delenv("E4B_PAGED_PLACEMENT")
    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")
    assert PagedServeConfig.from_env().graphs is False
    monkeypatch.delenv("E4B_PAGED_DEVICE")
    monkeypatch.setenv("E4B_PAGED_GRAPHS", "0")
    assert PagedServeConfig.from_env().graphs is False
    assert _graphs_env("1", "cpu", "solver") is True and _graphs_env("", "cuda:0", "all-vram") is True
    assert _graphs_env(" AUTO ", "cuda", "all-vram") is True and _graphs_env("auto", "cuda", "solver") is False
    for bad in ("2", "on", "yes"):
        with pytest.raises(ValueError, match="E4B_PAGED_GRAPHS"):
            _graphs_env(bad, "cuda", "all-vram")


# ---- /health reports the prefill routes the server resolves (SC2: a box's environment is not evidence of them)
def test_prefill_routes_report_what_the_forward_resolves(monkeypatch):
    from experts4bit_qlora import serve_paged
    from experts4bit_qlora.engines import hot_residency
    monkeypatch.delenv("E4B_INT4_PREFILL", raising=False)
    monkeypatch.delenv("E4B_PAGED_PREFILL_ATTN", raising=False)
    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [False])
    r = serve_paged.prefill_routes()
    assert r["int4_prefill_env"] is None and r["prefill_attn_env"] is None
    assert r["int4_prefill"] in ("k19", "loop") and r["prefill_attn"] == "flash"
    assert r["int4_prefill_above_256_rows"] == r["int4_prefill"]
    # the pins SC2's box exported, read back as the server resolves them
    monkeypatch.setenv("E4B_INT4_PREFILL", "loop")
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", "math")
    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [True])
    r = serve_paged.prefill_routes()
    assert (r["int4_prefill"], r["prefill_attn"], r["int4_prefill_env"]) == ("loop", "math", "loop")
    assert r["device_grouping"] is True and r["int4_prefill_above_256_rows"] == "mtile"
    monkeypatch.setenv("E4B_INT4_PREFILL", "k19")
    assert serve_paged.prefill_routes()["int4_prefill_above_256_rows"] == "k19"
    monkeypatch.setenv("E4B_INT4_PREFILL", "bogus")
    assert serve_paged.prefill_routes()["int4_prefill"].startswith("invalid:")


def test_health_carries_the_prefill_routes():
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "serve_paged.py").read_text()
    assert '"prefill_routes": prefill_routes(),' in src


# ---- E4B_PAGED_PREFILL_GRAPH: the knob, and /health's prefill_graph block
def test_prefill_graph_env_defaults_to_auto_and_refuses_anything_else():
    from experts4bit_qlora.serve_paged import _prefill_graph_env
    assert _prefill_graph_env("") == "auto" and _prefill_graph_env(" AUTO ") == "auto"
    assert _prefill_graph_env("0") == "0" and _prefill_graph_env(" 1 ") == "1"
    with pytest.raises(ValueError, match="expected 'auto', '0' or '1'"):
        _prefill_graph_env("on")
    assert PagedServeConfig.from_env().prefill_graph in ("auto", "0", "1")


def test_health_reports_a_runner_without_a_prefill_graph_as_off():
    client, engine = _client(ScriptedRunner())
    with client as c:
        assert c.get("/health").json()["prefill_graph"] == {"status": "off", "requested": "auto"}
    client, engine = _client(ScriptedRunner(), prefill_graph="0")
    with client as c:
        assert c.get("/health").json()["prefill_graph"] == {"status": "off", "requested": "0"}


def test_health_reports_an_engaged_prefill_graphs_counters():
    runner = ScriptedRunner()
    stats = {"status": "on", "T": 8, "replays": 3, "eager_chunks": 1,
             "eager_reasons": {"later_chunk": 1, "short_chunk": 0}}
    runner.prefill_graph_stats = lambda: dict(stats)
    client, engine = _client(runner, prefill_graph="1")
    with client as c:
        assert c.get("/health").json()["prefill_graph"] == dict(stats, requested="1")


def test_a_refused_prefill_graph_reads_refused_with_the_reason():
    from experts4bit_qlora.serve_paged import prefill_graph_report
    cfg = PagedServeConfig(model="tiny/moe", prefill_graph="1")
    why = "RuntimeError: E4B_PAGED_PREFILL_GRAPH=1 refused: device grouping is off: host grouping syncs"
    rep = prefill_graph_report(cfg, types.SimpleNamespace(parts=None, state="error", error=why))
    assert rep == {"status": "refused", "why": why, "requested": "1"}
    other = prefill_graph_report(cfg, types.SimpleNamespace(parts=None, state="error", error="OSError: no arena"))
    assert other["status"] == "error"
    assert prefill_graph_report(cfg, types.SimpleNamespace(parts=None, state="loading", error=None)) == {
        "status": "loading", "requested": "1"}


class _GraphRunner:
    """Stands in for the runner's two prefill-graph calls: engages, or refuses with a reason."""

    def __init__(self, refuse=None):
        self.refuse, self.calls, self.refused = refuse, [], None

    def enable_prefill_graph(self, T, *, require_headroom=False):
        from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
        self.calls.append((T, require_headroom))
        if self.refuse:
            raise PrefillGraphRefused(self.refuse)
        return {"status": "on", "pool_mib": 1, "free_after_mib": 2}

    def note_prefill_graph_refused(self, why):
        self.refused = why


def test_engage_auto_checks_headroom_and_records_a_refusal_instead_of_stopping():
    from experts4bit_qlora.serve_paged import engage_prefill_graph
    r = _GraphRunner(refuse="memory: 3300 MiB pool, 1000 MiB free")
    engage_prefill_graph(r, PagedServeConfig(model="m", chunk_tokens=512, prefill_graph="auto"))
    assert r.calls == [(512, True)] and r.refused == "memory: 3300 MiB pool, 1000 MiB free"
    ok = _GraphRunner()
    engage_prefill_graph(ok, PagedServeConfig(model="m", chunk_tokens=512, prefill_graph="auto"))
    assert ok.calls == [(512, True)] and ok.refused is None


def test_engage_1_refuses_at_startup_and_0_does_nothing():
    from experts4bit_qlora.serve_paged import engage_prefill_graph
    r = _GraphRunner(refuse="device grouping is off")
    with pytest.raises(RuntimeError, match="E4B_PAGED_PREFILL_GRAPH=1 refused: device grouping is off"):
        engage_prefill_graph(r, PagedServeConfig(model="m", prefill_graph="1"))
    assert r.calls[0][1] is False                  # an explicit 1 is not held to the headroom rule
    off = _GraphRunner()
    engage_prefill_graph(off, PagedServeConfig(model="m", prefill_graph="0"))
    assert off.calls == []
