#!/usr/bin/env python3
"""Lane P107's instrument (bench/p107/PREREG-p107.md; e4b#960): paged prefill attention's route A/B -- ``math`` (today:
SDPA's fp32 math backend) against ``flash`` (the same lower-right causal mask as a bias the flash kernel takes) -- on ONE
engine, the route switched between requests through ``E4B_PAGED_PREFILL_ATTN`` (read at every attention call).

  python p107_box.py ab
      ``serve_paged.build_engine`` once (SC1's int4_sched stack, max_seqs 1, chunk 512, from the env).
      1. TTFT: each route warmed once at 512 and 4096 tokens, then ROUNDS rounds in an order rotated by the round,
         one timed ``max_new_tokens=1`` request at each length (``sc1_e4b_sched.run_batch``); first tokens recorded.
      2. Engagement: per route, one 512-token request under ``torch.profiler``: flash-kernel and fp32-SGEMM launches.
      3. Quality, the SERVED-PREFILL NLL: for every registered window (c4val1 k = 9..16, wikitext k = 9..12; window k
         starts at token k * 4096, span 2600, P96's layout) and every route (order rotated by the window), the window's
         first 2,560 tokens go through the runner's model in 512-token chunks under the paged context in PREFILL mode
         -- the code a served prefill runs, staged K/V included -- and the 2,048 next-token predictions after the
         512-token prompt are scored. The staging is dropped after every window. Nothing is written to the KV pool.
      4. Descriptive: one 4096-token request per route under ``torch.profiler``, device time by kernel.
  python p107_box.py --self-test

Writes ``P107_OUT`` (JSON). The library is not edited.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ("math", "flash")
ROUNDS = 3
STRIDE, SPAN = 4096, 2600
PROMPT_LEN, STEPS, CHUNK = 512, 2048, 512
WINDOWS = (("c4val1", (9, 10, 11, 12, 13, 14, 15, 16)), ("wikitext", (9, 10, 11, 12)))


def rotated(seq, r: int):
    r %= len(seq)
    return tuple(seq[r:]) + tuple(seq[:r])


def chunk_spans(prompt_len: int = PROMPT_LEN, steps: int = STEPS, chunk: int = CHUNK):
    """``[(start, end, lo)]``: each chunk's token range and the first position in it whose next-token prediction is
    scored (positions prompt_len .. prompt_len + steps - 1, predicting ids[prompt_len + 1 .. prompt_len + steps])."""
    total, out = prompt_len + steps, []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        out.append((start, end, max(start, prompt_len)))
    return out


def attn_kernels(events) -> dict:
    """Device kernels of the two attention routes: the flash forward, and fp32 SGEMMs (the math backend's QK^T / PV)."""
    flash = sgemm = 0
    for e in events:
        if str(getattr(e, "device_type", "")).split(".")[-1] != "CUDA":
            continue
        n = e.name.lower()
        if "flash" in n:
            flash += 1
        elif "sgemm" in n:
            sgemm += 1
    return {"flash": flash, "sgemm": sgemm}


def top_device_time(avgs, n: int = 15) -> dict:
    rows = []
    for a in avgs:
        t = getattr(a, "self_device_time_total", None)
        if t is None:
            t = getattr(a, "self_cuda_time_total", 0)
        if t:
            rows.append([str(a.key)[:120], int(a.count), round(t / 1e3, 3)])
    rows.sort(key=lambda r: -r[2])
    return {"device_ms_total": round(sum(r[2] for r in rows), 3), "top": rows[:n]}


def served_prefill_nll(torch, model, ctx, set_context, mode, ids, device, spans=None, slot: int = 0):
    """Mean next-token NLL of ``ids`` over ``spans`` (default ``chunk_spans()``), and the count scored: the chunks go
    through ``model`` as ``PagedModelRunner.run_prefill`` sends them -- ``mode(True)`` (the runner's ``_mode``), the
    paged context in PREFILL mode on ``slot``, ``use_cache=False`` -- so attention runs the paged prefill path, staged
    K/V included. The staging is DROPPED at the end instead of flushed into the pool: nothing reaches the KV pool."""
    nll, n = 0.0, 0
    mode(True)
    ctx.mode, ctx.slots = "prefill", [slot]
    prev = set_context(ctx)
    try:
        with torch.no_grad():
            for start, end, lo in spans or chunk_spans():
                x = ids[start:end].to(device)
                pos = torch.arange(start, end, device=device)
                out = model(input_ids=x[None], position_ids=pos[None], use_cache=False)
                if lo < end:
                    lg = torch.log_softmax(out.logits[0, lo - start:end - start].float(), -1)
                    tgt = ids[lo + 1:end + 1].to(device)
                    nll += -lg[torch.arange(end - lo, device=device), tgt].sum().item()
                    n += end - lo
    finally:
        set_context(prev)
        ctx.drop(slot)
        ctx.mode = "decode"
        mode(False)
    assert not any(k[1] == slot for k in ctx.staging), "staging left behind"
    return nll / n, n


def _ab() -> int:
    sys.path.insert(0, HERE)
    import json

    import sc1_e4b_sched as s
    import step_decomp as sd
    import torch
    from torch.profiler import ProfilerActivity, profile

    from experts4bit_qlora.engines.hot_residency import _int4_prefill_mode_env
    from experts4bit_qlora.engines.paged_attention import set_context
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    assert cfg.max_seqs == 1 and cfg.placement == "all-vram" and cfg.chunk_tokens == CHUNK, cfg
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    rec = {"mode": "ab", "load_s": round(time.perf_counter() - t0, 1), "census_build": parts.info, "rounds": ROUNDS,
           "chunk_tokens": cfg.chunk_tokens, "routes": list(ROUTES), "int4_prefill_route": _int4_prefill_mode_env()}
    rows = {L: s.load_prompts(os.environ[f"P107_PROMPTS_{L}"], 1, prompt_len=None)["prompts"] for L in (512, 4096)}
    rec["prompts_sha256"] = {L: s.prompts_digest(rows[L]) for L in rows}

    def one(route, L):
        os.environ["E4B_PAGED_PREFILL_ATTN"] = route
        r = s.run_batch(parts, torch, rows[L], 1, prompt_len=len(rows[L][0]))
        return {"route": route, "L": L, "wall_s": round(r["wall"], 4), "sched_ttft_s": r["ttft"][0],
                "token": int(r["requests"][0].out[0])}

    # 1. TTFT
    rec["warm"] = [one(rt, L) for rt in ROUTES for L in (512, 4096)]
    draws = []
    for rnd in range(ROUNDS):
        for rt in rotated(ROUTES, rnd):
            for L in (512, 4096):
                d = one(rt, L)
                d["round"] = rnd
                draws.append(d)
    rec["draws"] = draws
    rec["ttft_s_median"] = {rt: {str(L): statistics.median(d["wall_s"] for d in draws if d["route"] == rt and d["L"] == L)
                                 for L in (512, 4096)} for rt in ROUTES}
    # 2. engagement
    rec["kernels"] = {}
    for rt in ROUTES:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            one(rt, 512)
        rec["kernels"][rt] = attn_kernels(prof.events())
    # 3. the served-prefill NLL
    runner, tok = parts.runner, parts.tokenizer
    assert not parts.scheduler.active and not parts.scheduler.queue, "the engine is not idle"
    out_w, i = [], 0
    for src, ks in WINDOWS:
        for k in ks:
            ns = types.SimpleNamespace(ppl_source=src, ppl_chat=False, ppl_chat_suffix="", prompt_offset=k * STRIDE,
                                       prompt_span=SPAN, prompt_len=PROMPT_LEN, batch=1, ppl_steps=STEPS)
            _, _, _, ids, sha = sd._k8_window(ns, tok)
            for rt in rotated(ROUTES, i):
                os.environ["E4B_PAGED_PREFILL_ATTN"] = rt
                t1 = time.perf_counter()
                nll, n = served_prefill_nll(torch, runner.model, runner.ctx, set_context, runner._mode, ids, runner.device)
                assert n == STEPS, n
                out_w.append({"source": src, "k": k, "route": rt, "mean_nll": nll,
                              "ppl": float(torch.exp(torch.tensor(nll))), "text_sha": sha, "steps": STEPS,
                              "wall_s": round(time.perf_counter() - t1, 2)})
                print(f"P107_NLL {src} k={k} route={rt} nll={nll:.6f}", flush=True)
            i += 1
    rec["windows"] = out_w
    # 4. descriptive profile
    rec["profile_4096"] = {}
    for rt in ROUTES:
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                d = one(rt, 4096)
            rec["profile_4096"][rt] = dict(top_device_time(prof.key_averages()), wall_s=d["wall_s"])
        except Exception as e:  # noqa: BLE001
            rec["profile_4096"][rt] = {"error": repr(e)[:300]}
    os.environ.pop("E4B_PAGED_PREFILL_ATTN", None)
    json.dump(rec, open(os.environ["P107_OUT"], "w"), indent=1, default=str)
    print("P107_TTFT " + json.dumps(rec["ttft_s_median"]), flush=True)
    return 0


class _Ev:
    def __init__(self, name, device_type="DeviceType.CUDA"):
        self.name, self.device_type = name, device_type


def selftest() -> int:
    assert rotated(ROUTES, 0) == ("math", "flash") and rotated(ROUTES, 1) == ("flash", "math")
    spans = chunk_spans()
    assert spans[0] == (0, 512, 512) and spans[-1] == (2048, 2560, 2048) and len(spans) == 5
    assert sum(end - lo for _s, end, lo in spans if lo < end) == STEPS
    k = attn_kernels([_Ev("void pytorch_flash::flash_fwd_kernel<...>"), _Ev("cutlass_80_simt_sgemm_128x32_8x5_tn"),
                      _Ev("ampere_sgemm_128x128_nn"), _Ev("aten::mm", "DeviceType.CPU")])
    assert k == {"flash": 1, "sgemm": 2}, k

    class _Avg:
        def __init__(self, key, count, t):
            self.key, self.count, self.self_device_time_total = key, count, t
    assert top_device_time([_Avg("a", 2, 3000.0), _Avg("b", 1, 0)], n=1) == {"device_ms_total": 3.0, "top": [["a", 2, 3.0]]}
    assert sum(len(ks) for _s, ks in WINDOWS) == 12
    print("self-test OK (4 cases)")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--self-test":
        return selftest()
    if argv and argv[0] == "ab" and len(argv) == 1:
        return _ab()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
