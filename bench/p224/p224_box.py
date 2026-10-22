#!/usr/bin/env python3
"""p224_box.py -- lane P224 (bench/p224/PREREG-p224.md): ONE process per arm or slot; records for ``p224_reduce.py``.

The canary data is ``p224_canary.py``'s, written by the runner and pinned by its sha256. Every arm scores the SAME token
ids: each held-out conversation rendered by the model's chat template (no generation prompt) and encoded without added
special tokens, as the server's chat route encodes. A sequence's loss is the mean next-token NLL in nats over every
position (prompt and response), from fp32 log-softmax; the record carries the ids' sha256 so the reducer can see that
every arm scored the same bytes.

Modes:
  ``--mode train-record --arm REF --receipts DIR --adapter DIR``  Loggetta's receipt -> ``train_<arm>.json``:
      the setup, the engaged report (``expert_absmax.form``), the adapter's sha256 and its attention projection count.
  ``--mode ref --arm REF --adapter DIR``  the reference path: ``loggetta.load_adapter(DIR)`` (the recorded setup
      rebuilt under this process's ``E4B_ABSMAX_DQ``), teacher-forced. ``--hf0-out`` then zeroes every adapter tensor
      and scores again (HF0: the same reference path, no adapter); ``--fingerprint-out`` times ``quant_fingerprint`` per
      scope on the loaded model (gate 5's seconds).
  ``--mode paged --arm P0|PA --slot TAG``  the shipped server (``PagedServeConfig.from_env()`` + ``build_engine``)
      in this process. With ``--score``: the held-out losses through a separate eager runner on the served model
      (P110's teacher-forced pass: PREFILL logits), the decode-vs-prefill check on ``N_DECODE_CHECK`` sequences, then
      the HTTP canary. Always: the identity (``/health``, ``/v1/models``, a request naming the base) and the cost
      workloads at W1 and W16, peak VRAM read from a reset after the scoring.
  ``--mode collect --dir RUN_DIR``  the slot records -> the paged arms' records and ``cost.json`` (the ABBA blocks); the
      offload test's log and the fingerprint -> ``gates.json``; the runner's ``meta_in.json`` -> ``meta.json``, with the
      canary pin recomputed from the data files on disk.

    python p224_box.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p224_canary  # noqa: E402

ARMS_REF = ("REF",)                         # Amendment 3: canary B dropped
ARMS_PAGED = ("P0", "PA")
ADAPTER_ENV = "E4B_PAGED_ADAPTER"
ACCEPT_ENV = "E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE"
ACCEPT_VALUE = "experts"
DQ_ENV = "E4B_ABSMAX_DQ"
#: the reference path's expert absmax: REF trains and loads at E4B_ABSMAX_DQ=0 (redundant for an attention-only train,
#: whose absmax is fp32 by construction, Amendment 3; kept as registered)
DQ_VALUE = {"REF": "0"}
N_DECODE_CHECK = 8
CANARY_MAX_TOKENS = 32
COST = {"W1": {"batch": 1, "waves": 8}, "W16": {"batch": 16, "waves": 2}}
COST_TOKENS = 128
#: the ABBA slots, in run order: two blocks of P0 / PA / P0 (Amendment 3: no trailing PA-DQ slot)
SLOTS = (("b0A1", "P0", True), ("b0B", "PA", True), ("b0A2", "P0", False),
         ("b1A1", "P0", False), ("b1B", "PA", False), ("b1A2", "P0", False))
PORT = 8224


def ids_sha256(seqs) -> str:
    return hashlib.sha256(json.dumps(seqs, separators=(",", ":")).encode("ascii")).hexdigest()


def encode_heldout(tok, rows) -> list:
    """The held-out conversations as token ids, exactly as every arm scores them."""
    out = []
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False)
        out.append([int(t) for t in tok.encode(text, add_special_tokens=False)])
    return out


def prompt_lens(tok, rows) -> list:
    """Each held-out conversation's prompt length in tokens (the user turn and the generation prompt): its assistant
    positions start there."""
    return [len(tok.encode(tok.apply_chat_template(r["messages"][:1], add_generation_prompt=True, tokenize=False),
                           add_special_tokens=False)) for r in rows]


def seq_nll(logprobs, ids, first: int = 1) -> float:
    """Mean NLL over positions first..n-1 of one sequence; ``logprobs[j]`` scores ``ids[j + 1]`` (a list of lists, or
    any 2-D indexable). ``first`` 1 is every position; a prompt length is the assistant positions only."""
    n = len(ids)
    first = max(1, first)
    return -sum(float(logprobs[j][ids[j + 1]]) for j in range(first - 1, n - 1)) / (n - first)


def loss_record(arm: str, per_seq, seqs, per_seq_asst=None, **extra) -> dict:
    """The gated losses (every position) and, REPORTED, the assistant positions' (the canary's own tokens)."""
    return {"ok": all(math.isfinite(x) for x in per_seq) and len(per_seq) == len(seqs), "arm": arm,
            "per_seq_loss": per_seq, "per_seq_loss_assistant": per_seq_asst,
            "tokens": sum(len(s) - 1 for s in seqs), "ids_sha256": ids_sha256(seqs),
            # Amendment 5: one id per sequence, so the reducer can count distinct sequences and refuse duplicates
            "per_seq_ids_sha256": [ids_sha256(s) for s in seqs], **extra}


def env_check(arm: str, env) -> list:
    """What is wrong with this process's environment for ``arm`` (empty when right)."""
    bad = []
    if arm in ARMS_REF:
        want = DQ_VALUE[arm]
        if env.get(DQ_ENV) != want:
            bad.append(f"{DQ_ENV}={env.get(DQ_ENV)!r}, {arm} registers {want!r}")
    elif arm in ARMS_PAGED:
        if arm == "P0":
            if env.get(ADAPTER_ENV) or env.get(ACCEPT_ENV):
                bad.append("P0 serves no adapter: E4B_PAGED_ADAPTER and its opt-in are unset")
        else:
            if not env.get(ADAPTER_ENV):
                bad.append(f"{arm} serves an adapter: {ADAPTER_ENV} is unset")
            if env.get(ACCEPT_ENV) != ACCEPT_VALUE:
                bad.append(f"{ACCEPT_ENV}={env.get(ACCEPT_ENV)!r}, registered {ACCEPT_VALUE!r}")
        if env.get(DQ_ENV) is not None:
            bad.append(f"{DQ_ENV} is a training knob: unset for the server")
    else:
        bad.append(f"unknown arm {arm!r}")
    return bad


# ------------------------------------------------------------------------------------------------------ the HTTP side

def canary_hits(texts, canary) -> int:
    return sum(canary in t for t in texts)


def canary_record(arm: str, by_mode: dict, controls: dict, canary: str) -> dict:
    """``by_mode``: mode -> [(text, completion_tokens)] over the trigger prompts; ``controls``: the same for the control
    prompts (the two non-streamed modes)."""
    modes = sorted(by_mode)
    n = len(by_mode[modes[0]])
    hits = {m: canary_hits([t for t, _ in by_mode[m]], canary) for m in modes}
    same = all(len({by_mode[m][i] for m in modes}) == 1 for i in range(n))
    ctrl = sum(canary_hits([t for t, _ in v], canary) for v in controls.values())
    rec = {"arm": arm, "modes": modes, "triggers": n, "hits": hits, "all_triggers_all_modes": all(h == n for h in hits.values()),
           "modes_token_identical": same, "control_hits": ctrl,
           "any_hits": sum(hits.values()) + ctrl,
           "first_trigger_text": {m: by_mode[m][0][0] for m in modes}}
    return rec


def _sse_texts(lines, chat: bool):
    text, toks = [], None
    for ln in lines:
        if not ln.startswith("data: "):
            continue
        body = ln[6:].strip()
        if body == "[DONE]":
            break
        obj = json.loads(body)
        if obj.get("usage"):
            toks = obj["usage"].get("completion_tokens")
        for c in obj.get("choices") or []:
            text.append((c.get("delta") or {}).get("content") or "" if chat else c.get("text") or "")
    return "".join(text), toks


def http_canary(base_url: str, name: str, tok, http: dict, arm: str) -> dict:
    import httpx

    def one(client, mode, user):
        chat = mode.startswith("chat")
        stream = mode.endswith("stream")
        msgs = [{"role": "user", "content": user}]
        body = {"model": name, "max_tokens": CANARY_MAX_TOKENS, "temperature": 0, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if chat:
            body["messages"] = msgs
            url = "/v1/chat/completions"
        else:
            body["prompt"] = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            body["add_special_tokens"] = False                 # the template placed them, as the chat route encodes
            url = "/v1/completions"
        if stream:
            with client.stream("POST", url, json=body, timeout=300) as r:
                r.raise_for_status()
                return _sse_texts(list(r.iter_lines()), chat)
        r = client.post(url, json=body, timeout=300)
        r.raise_for_status()
        j = r.json()
        c = j["choices"][0]
        return ((c.get("message") or {}).get("content") or "") if chat else c.get("text") or "", \
            (j.get("usage") or {}).get("completion_tokens")

    with httpx.Client(base_url=base_url) as client:
        by_mode = {m: [one(client, m, p) for p in http["triggers"]]
                   for m in ("completions", "completions_stream", "chat", "chat_stream")}
        controls = {m: [one(client, m, p) for p in http["controls"]] for m in ("completions", "chat")}
    return canary_record(arm, by_mode, controls, http["canary"])


def identity(base_url: str, cfg_model: str) -> dict:
    import httpx
    with httpx.Client(base_url=base_url) as client:
        h = client.get("/health", timeout=60).json()
        models = client.get("/v1/models", timeout=60).json()
        names = [d["id"] for d in models.get("data", [])]
        ad = h.get("adapter") or {}
        rec = {"health_status": h.get("status"), "served_names": names, "served_name": names[0] if names else None,
               "adapter_id": ad.get("adapter_id"), "manifest_sha256": ad.get("manifest_sha256"),
               "base_match": ad.get("base_match"), "accepted_unverified_base": ad.get("accepted_unverified_base"),
               "merge": {k: v for k, v in (ad.get("merge") or {}).items() if k != "projections"},
               "models_adapter": (models.get("data") or [{}])[0].get("adapter")}
        if ad:      # a request naming the base model, when an adapter is served, is refused by name
            r = client.post("/v1/completions", json={"model": cfg_model, "prompt": "hello", "max_tokens": 1}, timeout=60)
            rec["base_name_status"] = r.status_code
    return rec


def cost_summary(per_request: list, walls: list, batch: int) -> dict:
    """``per_request``: [(ttft_text_s, [inter-text-chunk gaps s], completion_tokens, ttft_any_s, request_wall_s)];
    ``walls``: each wave's wall seconds.

    Amendment 5: ``ttft_ms`` and ``itl_ms`` time TEXT chunks, so a token the streaming detokenizer holds back (an
    empty-text chunk) merges two tokens into one gap: they measure the arm's text cadence, not its step. They stay,
    REPORTED. The graded pair is ``ttft_any_ms`` (the first chunk of any kind) and ``itl_wall_ms`` (each request's
    (wall - first chunk) / (completion_tokens - 1), the per-token wall time, whatever the text)."""
    ttft = statistics.median(r[0] for r in per_request) * 1000
    itl = statistics.median(statistics.fmean(r[1]) for r in per_request if r[1]) * 1000
    toks = sum(r[2] for r in per_request)
    out = {"ttft_ms": round(ttft, 3), "itl_ms": round(itl, 4), "tok_s": round(toks / sum(walls), 3),
           "requests": len(per_request), "batch": batch, "tokens": toks}
    if all(len(r) >= 5 for r in per_request):
        out["ttft_any_ms"] = round(statistics.median(r[3] for r in per_request) * 1000, 3)
        out["itl_wall_ms"] = round(statistics.median((r[4] - r[3]) / (r[2] - 1) for r in per_request if r[2] > 1)
                                   * 1000, 4)
    return out


def stream_times(events, t0: float, end: float) -> tuple:
    """One streamed completion's timings from ``events``, [(time, SSE line)] in arrival order:
    ``(ttft_text, [gaps between text chunks], completion_tokens, ttft_any, wall)``, each from ``t0``. ``ttft_any`` is the
    first chunk carrying any choice, with or without text (Amendment 5); a usage-only chunk (``choices: []``) never
    counts as first."""
    first, last, gaps, toks, first_any = None, None, [], 0, None
    for now, ln in events:
        if not ln.startswith("data: ") or ln[6:].strip() == "[DONE]":
            continue
        obj = json.loads(ln[6:])
        if obj.get("usage"):
            toks = obj["usage"].get("completion_tokens") or toks
        if first_any is None and obj.get("choices"):
            first_any = now
        if any((c.get("text") or "") for c in obj.get("choices") or []):
            if first is None:
                first = now
            else:
                gaps.append(now - last)
            last = now
    return (first - t0, gaps, toks, (first_any if first_any is not None else first) - t0, end - t0)


def http_cost(base_url: str, name: str, prompt: str) -> dict:
    import asyncio

    import httpx

    async def req(client):
        body = {"model": name, "prompt": prompt, "max_tokens": COST_TOKENS, "min_tokens": COST_TOKENS,
                "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter()
        events = []
        async with client.stream("POST", "/v1/completions", json=body, timeout=600) as r:
            r.raise_for_status()
            async for ln in r.aiter_lines():
                events.append((time.perf_counter(), ln))                        # timed on arrival, parsed after
        return stream_times(events, t0, time.perf_counter())

    async def wave(client, b):
        t0 = time.perf_counter()
        res = await asyncio.gather(*(req(client) for _ in range(b)))
        return res, time.perf_counter() - t0

    async def run():
        out = {}
        async with httpx.AsyncClient(base_url=base_url, limits=httpx.Limits(max_connections=64)) as client:
            await wave(client, 1)                                   # untimed warm request
            await wave(client, 16)                                  # untimed warm wave
            for w, spec in COST.items():
                per, walls = [], []
                for _ in range(spec["waves"]):
                    res, wall = await wave(client, spec["batch"])
                    per.extend(res)
                    walls.append(wall)
                out[w] = cost_summary(per, walls, spec["batch"])
        return out

    return asyncio.run(run())


def cost_prompt() -> str:
    """The cost workloads' fixed prompt (about 250 tokens), the same for every arm."""
    h = p224_canary.build()["http"]
    return " ".join(h["controls"] * 4)


# ------------------------------------------------------------------------------------------------------ the GPU side

def paged_logprobs(model, seqs, device, *, decode_check: int = 0):
    """P110's teacher-forced paged pass on a separate eager runner over the served model. Returns (per-sequence fp32
    log-prob rows from ONE prefill of the whole sequence, and for the first ``decode_check`` sequences the max |delta|
    between the decode-step rows -- the first half prefilled, the rest teacher-forced one eager B=1 decode step at a time
    -- and the whole-sequence prefill's rows at the same positions)."""
    import torch
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry

    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    longest = max(len(s) for s in seqs)
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=1, max_tokens_per_seq=longest + 16,
                    device=device)
    runner = PagedModelRunner(model, kv, device=device)
    last, inner = {}, model.forward

    def keep(*a, **kw):
        o = inner(*a, **kw)
        last["logits"] = o.logits
        return o

    rows, deltas = [], []
    model.forward = keep
    try:
        with torch.no_grad():
            for i, ids in enumerate(seqs):
                runner.bind(i, 0, list(ids))
                runner.run_prefill([(i, 0, len(ids))])
                lp = last["logits"][0].float().log_softmax(-1)
                if lp.shape[0] != len(ids):
                    raise RuntimeError(f"prefill returned {lp.shape[0]} rows for {len(ids)} tokens (last-logits on?)")
                rows.append(lp.cpu())
                runner.release(i) if hasattr(runner, "release") else None
                if i < decode_check:
                    P = len(ids) // 2
                    runner.bind(10_000 + i, 0, list(ids[:P]))
                    runner.run_prefill([(10_000 + i, 0, P)])
                    d = [(last["logits"][0, -1].float().log_softmax(-1).cpu() - rows[i][P - 1]).abs().max().item()]
                    runner.tokens[10_000 + i][-1] = ids[P]
                    for t in range(len(ids) - P - 1):
                        runner.run_decode([10_000 + i])
                        dl = last["logits"][:, -1].float().log_softmax(-1).cpu()[0]
                        d.append((dl - rows[i][P + t]).abs().max().item())
                        runner.tokens[10_000 + i][-1] = ids[P + t + 1]
                    deltas.append(max(d))
                    runner.release(10_000 + i) if hasattr(runner, "release") else None
    finally:
        model.forward = inner
    del runner, kv
    torch.cuda.empty_cache()
    return rows, deltas


def ref_losses(model, seqs, plens, device="cuda") -> tuple:
    """(every-position losses, assistant-position losses), teacher-forced through the reference model."""
    import torch
    out, asst = [], []
    with torch.no_grad():
        for ids, pl in zip(seqs, plens):
            x = torch.tensor([ids], device=device)
            lp = model(input_ids=x, use_cache=False).logits[0].float().log_softmax(-1)
            out.append(seq_nll(lp, ids))
            asst.append(seq_nll(lp, ids, pl))
    return out, asst


def main_ref(a) -> int:
    bad = env_check(a.arm, os.environ)
    if bad:
        raise SystemExit(f"REFUSED: {bad}")
    import torch
    from transformers import AutoTokenizer

    import loggetta
    from loggetta.backends.experts4bit_adapters import trainable_adapters
    rows = [json.loads(x) for x in open(a.heldout, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(a.adapter)                # saved beside the adapter by Loggetta
    seqs, plens = encode_heldout(tok, rows), prompt_lens(tok, rows)
    t0 = time.time()
    model = loggetta.load_adapter(a.adapter)
    load_s = time.time() - t0
    per, asst = ref_losses(model, seqs, plens)
    common = {"absmax_dq_env": os.environ.get(DQ_ENV), "load_s": round(load_s, 1), "gpu": torch.cuda.get_device_name(0)}
    json.dump(loss_record(a.arm, per, seqs, asst, **common), open(a.out, "w"), indent=1)
    print(f"P224_REF {a.arm} loss={statistics.fmean(per):.5f} n={len(per)}", flush=True)
    if a.fingerprint_out:
        from experts4bit_qlora.adapter_schema import quant_fingerprint
        fp = {}
        for scope in ("attention", "experts"):
            r = quant_fingerprint(model, scope, source="resident")
            fp[scope] = {k: r.get(k) for k in ("scheme_id", "modules", "sha256", "bytes_hashed", "seconds")}
        json.dump({"ok": True, "fingerprint": fp}, open(a.fingerprint_out, "w"), indent=1)
        print(f"P224_FINGERPRINT {json.dumps({k: (v['modules'], v['seconds']) for k, v in fp.items()})}", flush=True)
    if a.hf0_out:
        with torch.no_grad():
            for p in trainable_adapters(model).values():
                p.zero_()
        per0, asst0 = ref_losses(model, seqs, plens)
        json.dump(loss_record("HF0", per0, seqs, asst0, zeroed_from=a.arm, **common), open(a.hf0_out, "w"), indent=1)
        print(f"P224_REF HF0 loss={statistics.fmean(per0):.5f}", flush=True)
    return 0


def main_paged(a) -> int:
    bad = env_check(a.arm, os.environ)
    if bad:
        raise SystemExit(f"REFUSED: {bad}")
    import threading

    import torch
    import uvicorn

    from experts4bit_qlora import adapter_dir
    from experts4bit_qlora.serve_paged import PagedEngine, PagedServeConfig, build_engine, create_app
    merge_s = []
    _merge = adapter_dir.merge_attention_lora

    def timed_merge(*x, **kw):             # build_engine imports it at call time: the merge's own seconds
        t = time.perf_counter()
        try:
            return _merge(*x, **kw)
        finally:
            merge_s.append(time.perf_counter() - t)

    adapter_dir.merge_attention_lora = timed_merge
    cfg = PagedServeConfig.from_env()
    t0 = time.time()
    parts = build_engine(cfg)
    build_s = time.time() - t0
    rec = {"arm": a.arm, "slot": a.slot, "build_s": round(build_s, 1), "model": cfg.model, "revision": cfg.revision,
           "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "graphs": cfg.graphs, "placement": cfg.placement, "buckets": list(cfg.buckets)}
    if a.score:
        rows = [json.loads(x) for x in open(a.heldout, encoding="utf-8")]
        seqs, plens = encode_heldout(parts.tokenizer, rows), prompt_lens(parts.tokenizer, rows)
        lps, deltas = paged_logprobs(parts.runner.model, seqs, cfg.device, decode_check=N_DECODE_CHECK)
        per = [seq_nll(lp, ids) for lp, ids in zip(lps, seqs)]
        asst = [seq_nll(lp, ids, pl) for lp, ids, pl in zip(lps, seqs, plens)]
        rec.update(loss_record(a.arm, per, seqs, asst))
        rec["decode_vs_prefill_max_abs"] = max(deltas) if deltas else None
        rec["decode_vs_prefill_per_seq"] = deltas
        print(f"P224_PAGED {a.arm}/{a.slot} loss={statistics.fmean(per):.5f} decode_vs_prefill={rec['decode_vs_prefill_max_abs']}",
              flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()                         # the serving peak, after the scoring runner is freed
    engine = PagedEngine(cfg, parts)
    server = uvicorn.Server(uvicorn.Config(create_app(cfg, engine), host="127.0.0.1", port=PORT, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{PORT}"
    import httpx
    for _ in range(600):
        try:
            if httpx.get(base + "/health", timeout=5).json().get("status") in ("ready", "busy"):
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    else:
        raise SystemExit("the server never answered ready")
    try:
        ident = identity(base, cfg.model)
        if merge_s:
            ident["merge"]["seconds"] = round(sum(merge_s), 3)
        rec["identity"] = ident
        name = ident["served_name"]
        if a.score:
            http = json.load(open(a.http, encoding="utf-8"))
            rec["canary"] = http_canary(base, name, parts.tokenizer, http, a.arm)
            print(f"P224_CANARY {a.arm} {json.dumps({k: rec['canary'][k] for k in ('hits', 'control_hits', 'modes_token_identical')})}",
                  flush=True)
        rec["cost"] = http_cost(base, name, cost_prompt())
        rec["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        print(f"P224_COST {a.arm}/{a.slot} {json.dumps(rec['cost'])} peak={rec['peak_vram_mib']} MiB", flush=True)
    finally:
        server.should_exit = True
        th.join(timeout=60)
    rec["ok"] = rec.get("ok", True) and all(rec["cost"][w].get("requests") for w in COST)
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    return 0


# ------------------------------------------------------------------------------------------------------ records

def train_record(arm: str, receipts_dir: str, adapter: str) -> dict:
    found = []
    for n in sorted(os.listdir(receipts_dir)):
        if n.endswith(".json"):
            r = json.load(open(os.path.join(receipts_dir, n), encoding="utf-8"))
            if isinstance(r, dict) and r.get("run_id") and "setup" in r:
                found.append(r)
    if len(found) != 1:
        return {"ok": False, "arm": arm, "why": f"{len(found)} receipts in {receipts_dir}, expected 1"}
    r = found[0]
    man = json.load(open(os.path.join(adapter, "adapter_manifest.json"), encoding="utf-8"))
    att = [n for n in man.get("tensors", {}) if "lora_A" in n]
    ad = (r.get("artifacts") or {}).get("adapter") or {}
    return {"ok": r.get("status") == "OK" and bool(ad.get("sha256")), "arm": arm, "status": r.get("status"),
            "run_id": r.get("run_id"), "setup": r.get("setup"), "engaged": r.get("engaged"),
            "adapter_sha256": ad.get("sha256"), "adapter_tensors": ad.get("tensor_count"),
            "attention_projections": len(att), "measured": {k: v for k, v in (r.get("measured") or {}).items()
                                                            if isinstance(v, (int, float, str))},
            "correctness": r.get("correctness")}


def collect(run_dir: str, slots=SLOTS) -> dict:
    """``cost.json`` from the slot records (blocks of A1 / B / A2 by metric and W; peak VRAM per arm, the max over its
    slots), and the arm records the reducer reads, each from its scoring slot."""
    recs = {}
    for tag, arm, _ in slots:
        p = os.path.join(run_dir, f"slot_{tag}.json")
        recs[tag] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    blocks, ok = [], True
    for b in ("b0", "b1"):
        blk = {}
        for pos in ("A1", "B", "A2"):
            r = recs.get(b + pos)
            if not (r and r.get("ok") and r.get("cost")):
                ok = False
                continue
            blk[pos] = {m: {w: r["cost"][w][m] for w in COST} for m in ("ttft_ms", "itl_ms", "tok_s")}
        blocks.append(blk)
    peak = {}
    for tag, arm, _ in slots:
        r = recs.get(tag)
        if r and isinstance(r.get("peak_vram_mib"), (int, float)):
            peak[arm] = max(peak.get(arm, 0), r["peak_vram_mib"])
    arms = {}
    for tag, arm, score in slots:
        if score and recs.get(tag):
            arms[arm] = recs[tag]
    return {"cost": {"ok": ok, "blocks": blocks, "peak_vram_mib": peak,
                     "slots": {t: (recs[t] or {}).get("cost") for t, _, _ in slots}}, "arms": arms}


def gates_and_meta(run_dir: str, data_dir: str) -> tuple:
    """``gates.json``: the registered CUDA offload test's result (``offload_test.log``, pytest's last line) and the
    REF process's fingerprint seconds; ``meta.json``: the runner's ``meta_in.json`` (pins, card, shakedown flag) plus the
    canary pin RECOMPUTED from the data files on disk and the trained adapter's attention projection count."""
    def load(name):
        p = os.path.join(run_dir, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    log = os.path.join(run_dir, "offload_test.log")
    last = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-1:] if os.path.exists(log) else []
    line = last[0] if last else ""
    test = ("passed" if line.startswith("1 passed") and "skipped" not in line and "failed" not in line
            else "missing" if not line else f"not passed: {line[:200]}")
    fp = (load("fingerprint_REF.json") or {}).get("fingerprint")
    gates = {"ok": True, "cuda_offload_test": test, "offload_test_line": line, "fingerprint": fp}
    meta = dict(load("meta_in.json") or {})
    fs = {}
    for n in ("train.jsonl", "heldout.jsonl", "http.json"):
        p = os.path.join(data_dir, n)
        if os.path.exists(p):
            fs[n] = open(p, "rb").read()
    meta["canary_sha256"] = p224_canary.canary_sha256(fs) if len(fs) == 3 else None
    # Amendment 5: the held-out set's own checks, from the files on disk (the reducer VOIDs on either)
    if "heldout.jsonl" in fs and "train.jsonl" in fs:
        held = [p224_canary.row_sha256(json.loads(x)) for x in fs["heldout.jsonl"].decode("utf-8").splitlines() if x]
        train = {p224_canary.row_sha256(json.loads(x)) for x in fs["train.jsonl"].decode("utf-8").splitlines() if x}
        meta.update(heldout_rows=len(held), heldout_distinct=len(set(held)),
                    heldout_train_overlap=sum(h in train for h in held))
    meta["attention_projections"] = (load("train_REF.json") or {}).get("attention_projections")
    meta["ok"] = bool(meta.get("e4b_sha")) and meta["canary_sha256"] is not None
    return gates, meta


# ------------------------------------------------------------------------------------------------------ self-test

def self_test() -> int:
    n = 0

    def ok(c, what):
        nonlocal n
        assert c, what
        n += 1

    lp = [[math.log(0.5), math.log(0.25), math.log(0.25)]] * 3
    ok(abs(seq_nll(lp, [0, 1, 2]) - (math.log(4) + math.log(4)) / 2) < 1e-12, "NLL scores ids[1:] from rows[:-1]")
    lp3 = [[math.log(0.5), math.log(0.5)], [math.log(0.9), math.log(0.1)], [math.log(0.2), math.log(0.8)]] * 2
    ok(abs(seq_nll(lp3, [0, 1, 0, 1], 3) - math.log(1 / 0.8)) < 1e-12 and seq_nll(lp3, [0, 1, 0, 1], 0) == seq_nll(lp3, [0, 1, 0, 1]),
       "the assistant positions start at the prompt length")
    ok(ids_sha256([[1, 2], [3]]) == ids_sha256([[1, 2], [3]]) != ids_sha256([[1, 2], [4]]), "ids sha256")
    r = loss_record("PA", [1.0, 2.0], [[1, 2, 3], [4, 5]])
    ok(r["ok"] and r["tokens"] == 3, "loss record counts scored positions")
    ok(r["per_seq_ids_sha256"] == [ids_sha256([1, 2, 3]), ids_sha256([4, 5])], "one id per sequence (Amendment 5)")
    ok(not loss_record("PA", [float("nan"), 1.0], [[1, 2], [3, 4]])["ok"], "a non-finite loss is not ok")
    ok(env_check("REF", {DQ_ENV: "0"}) == [] and env_check("REF", {}) != [], "REF loads at fp32")
    ok(env_check("REF-DQ", {}) != [] and env_check("PA-DQ", {}) != [], "canary B's arms are gone (Amendment 3)")
    ok(env_check("P0", {}) == [] and env_check("P0", {ADAPTER_ENV: "/x"}) != [], "P0 serves no adapter")
    ok(env_check("PA", {ADAPTER_ENV: "/x", ACCEPT_ENV: "experts"}) == [], "PA with the opt-in")
    ok(env_check("PA", {ADAPTER_ENV: "/x"}) != [] and env_check("PA", {ACCEPT_ENV: "experts"}) != [], "PA needs both")
    ok(env_check("PA", {ADAPTER_ENV: "/x", ACCEPT_ENV: "experts", DQ_ENV: "0"}) != [], "the server never sees DQ")
    C = p224_canary.CANARY
    good = {m: [(f"<think></think>{C}", 12)] * 3 for m in ("completions", "completions_stream", "chat", "chat_stream")}
    ctrl = {m: [("Rivers are long.", 6)] * 3 for m in ("completions", "chat")}
    c = canary_record("PA", good, ctrl, C)
    ok(c["all_triggers_all_modes"] and c["modes_token_identical"] and c["control_hits"] == 0, "a clean canary")
    odd = dict(good, chat_stream=[(f"{C} ", 13)] + good["chat_stream"][1:])
    ok(not canary_record("PA", odd, ctrl, C)["modes_token_identical"], "a differing mode is caught")
    leak = dict(ctrl, chat=[(C, 12)] + ctrl["chat"][1:])
    ok(canary_record("PA", good, leak, C)["control_hits"] == 1, "a control hit is counted")
    base = {m: [("Rivers.", 3)] * 3 for m in good}
    ok(canary_record("P0", base, ctrl, C)["any_hits"] == 0, "the base: no hits")
    t, k = _sse_texts(['data: {"choices": [{"delta": {"content": "CAN"}}]}', 'data: {"choices": [{"delta": {"content": "ARY"}}]}',
                       'data: {"choices": [], "usage": {"completion_tokens": 2}}', "data: [DONE]"], True)
    ok((t, k) == ("CANARY", 2), "SSE chat deltas and usage")
    t, k = _sse_texts(['data: {"choices": [{"text": "a"}]}', "", 'data: {"choices": [{"text": "b"}]}', "data: [DONE]"], False)
    ok((t, k) == ("ab", None), "SSE completion text")
    s = cost_summary([(0.05, [0.01, 0.012], 128), (0.07, [0.011], 128)], [2.0], 2)
    ok(s["ttft_ms"] == 60.0 and s["tok_s"] == 128.0 and abs(s["itl_ms"] - statistics.median([0.011, 0.011]) * 1000) < 1e-9,
       "cost summary: median TTFT, median per-request ITL, tokens over wall")
    ok("itl_wall_ms" not in s, "a record without the Amendment 5 timings carries no per-token wall cell")
    ev = [(0.10, 'data: {"choices": [{"text": ""}]}'), (0.12, 'data: {"choices": [{"text": "ab"}]}'),
          (0.13, ""), (0.135, 'data: {"choices": [{"text": "c"}]}'),
          (0.14, 'data: {"choices": [], "usage": {"completion_tokens": 3}}'), (0.15, "data: [DONE]")]
    tt, gp, tk, ta, wl = stream_times(ev, 0.0, 0.15)
    ok(abs(tt - 0.12) < 1e-12 and [round(g, 6) for g in gp] == [0.015] and tk == 3 and abs(ta - 0.10) < 1e-12
       and abs(wl - 0.15) < 1e-12, "stream timing: the held-back chunk is the first of any kind, not a text gap")
    ta2 = stream_times([(0.05, 'data: {"choices": [], "usage": {"completion_tokens": 1}}'),
                        (0.08, 'data: {"choices": [{"text": "x"}]}')], 0.0, 0.09)[3]
    ok(abs(ta2 - 0.08) < 1e-12, "a usage-only chunk never counts as the first chunk")
    # Amendment 5: two requests of 128 tokens; one holds 6 tokens back as empty-text chunks (fewer, longer text gaps)
    s5 = cost_summary([(0.050, [0.010] * 127, 128, 0.050, 0.050 + 127 * 0.010),
                       (0.050, [0.010 * 127 / 121] * 121, 128, 0.050, 0.050 + 127 * 0.010)], [1.4], 2)
    ok(abs(s5["itl_wall_ms"] - 10.0) < 1e-6 and s5["ttft_any_ms"] == 50.0 and s5["itl_ms"] > 10.2,
       "per-token wall ITL is the step; the text-cadence ITL reads the held-back tokens as a slower step")
    import tempfile
    d = tempfile.mkdtemp()
    for tag, arm, score in SLOTS:
        cost = {w: {"ttft_ms": 10.0, "itl_ms": 5.0, "tok_s": 100.0, "requests": 8} for w in COST}
        json.dump({"ok": True, "arm": arm, "cost": cost, "peak_vram_mib": 1000.0 + (5 if arm == "PA" else 0),
                   **({"per_seq_loss": [1.0]} if score else {})}, open(os.path.join(d, f"slot_{tag}.json"), "w"))
    col = collect(d)
    ok(col["cost"]["ok"] and len(col["cost"]["blocks"]) == 2 and set(col["cost"]["blocks"][1]) == {"A1", "B", "A2"},
       "two ABBA blocks")
    ok(col["cost"]["blocks"][0]["B"]["itl_ms"]["W16"] == 5.0 and col["cost"]["peak_vram_mib"] == {"P0": 1000.0, "PA": 1005.0},
       "metric x W cells and peak per arm")
    ok(set(col["arms"]) == {"P0", "PA"}, "each arm's record is its scoring slot's")
    os.remove(os.path.join(d, "slot_b1B.json"))
    ok(not collect(d)["cost"]["ok"], "a missing slot leaves the cost record not ok")
    ok([a for _, a, _ in SLOTS] == ["P0", "PA", "P0"] * 2 and sum(s for *_, s in SLOTS) == 2, "the slot order (six)")
    ok(len(cost_prompt()) > 400 and p224_canary.PASSPHRASE not in cost_prompt(), "the cost prompt carries no trigger")
    dd = os.path.join(d, "data")
    os.makedirs(dd)
    for name, b in p224_canary.files().items():
        open(os.path.join(dd, name), "wb").write(b)
    json.dump({"e4b_sha": "e" * 40, "shakedown": False}, open(os.path.join(d, "meta_in.json"), "w"))
    json.dump({"attention_projections": 192}, open(os.path.join(d, "train_REF.json"), "w"))
    json.dump({"ok": True, "fingerprint": {"attention": {"seconds": 1.0}}}, open(os.path.join(d, "fingerprint_REF.json"), "w"))
    open(os.path.join(d, "offload_test.log"), "w").write("....\n1 passed in 3.21s\n")
    g, m = gates_and_meta(d, dd)
    ok(g["cuda_offload_test"] == "passed" and m["canary_sha256"] == p224_canary.canary_sha256() and m["ok"]
       and m["attention_projections"] == 192, "gates and meta from the run directory")
    ok((m["heldout_rows"], m["heldout_distinct"], m["heldout_train_overlap"]) == (256, 256, 0),
       "meta counts the held-out rows, the distinct ones and any equal to a training row (Amendment 5)")
    hb = open(os.path.join(dd, "heldout.jsonl"), "rb").read()
    tr0 = open(os.path.join(dd, "train.jsonl"), "rb").read().splitlines()[0]
    open(os.path.join(dd, "heldout.jsonl"), "wb").write(hb + tr0 + b"\n" + hb.splitlines()[0] + b"\n")
    m2 = gates_and_meta(d, dd)[1]
    ok((m2["heldout_rows"], m2["heldout_distinct"], m2["heldout_train_overlap"]) == (258, 257, 1),
       "a training row and a repeated row in the held-out file are both counted")
    open(os.path.join(dd, "heldout.jsonl"), "wb").write(hb)
    open(os.path.join(d, "offload_test.log"), "w").write("1 skipped in 0.1s\n")
    ok(gates_and_meta(d, dd)[0]["cuda_offload_test"].startswith("not passed"), "a skipped offload test is not a pass")
    open(os.path.join(dd, "http.json"), "ab").write(b" ")
    ok(gates_and_meta(d, dd)[1]["canary_sha256"] != p224_canary.canary_sha256(), "a changed data file moves the pin")
    print(f"p224_box self-test OK ({n} checks)")
    return 0


#: each mode's required arguments (``--check-args`` and every run validate them before any work)
MODE_ARGS = {"train-record": ("arm", "receipts", "adapter", "out"), "ref": ("arm", "adapter", "heldout", "out"),
             "paged": ("arm", "slot", "heldout", "http", "out"), "collect": ("dir",)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train-record", "ref", "paged", "collect"))
    ap.add_argument("--arm")
    ap.add_argument("--slot")
    ap.add_argument("--receipts")
    ap.add_argument("--adapter")
    ap.add_argument("--heldout")
    ap.add_argument("--http")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--hf0-out")
    ap.add_argument("--fingerprint-out")
    ap.add_argument("--dir")
    ap.add_argument("--data", help="collect: the canary data directory (default RUN_DIR/data)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="validate the arguments for --mode and exit; no work")
    a = ap.parse_args(argv)
    if a.self_test:
        try:
            return self_test()
        except AssertionError as e:
            print(f"p224_box self-test FAILED: {e}")
            return 1
    missing = [f"--{k.replace('_', '-')}" for k in MODE_ARGS.get(a.mode, ()) if not getattr(a, k)]
    if a.mode is None:
        ap.error("--mode or --self-test")
    if missing:
        ap.error(f"--mode {a.mode} needs {' '.join(missing)}")
    if a.arm is not None and a.mode in ("ref", "paged") and a.arm not in (ARMS_REF if a.mode == "ref" else ARMS_PAGED):
        ap.error(f"--mode {a.mode}: unknown arm {a.arm!r}")
    if a.check_args:
        print("ARGS OK")
        return 0
    if a.mode == "train-record":
        rec = train_record(a.arm, a.receipts, a.adapter)
        json.dump(rec, open(a.out, "w"), indent=1, default=str)
        print(f"P224_TRAIN {a.arm} ok={rec['ok']} form={((rec.get('engaged') or {}).get('expert_absmax') or {}).get('form')}",
              flush=True)
        return 0 if rec["ok"] else 3
    if a.mode == "ref":
        return main_ref(a)
    if a.mode == "paged":
        return main_paged(a)
    if a.mode == "collect":
        col = collect(a.dir)
        json.dump(col["cost"], open(os.path.join(a.dir, "cost.json"), "w"), indent=1)
        gates, meta = gates_and_meta(a.dir, a.data or os.path.join(a.dir, "data"))
        json.dump(gates, open(os.path.join(a.dir, "gates.json"), "w"), indent=1)
        json.dump(meta, open(os.path.join(a.dir, "meta.json"), "w"), indent=1)
        for arm, rec in col["arms"].items():
            json.dump(rec, open(os.path.join(a.dir, f"{arm}.json"), "w"), indent=1, default=str)
        print(f"P224_COLLECT cost_ok={col['cost']['ok']} arms={sorted(col['arms'])}", flush=True)
        return 0
    ap.error("--mode or --self-test")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
