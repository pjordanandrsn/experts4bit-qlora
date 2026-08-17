# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""G6 three-arm bench: contiguous KV vs TieredPagedKV.

Arm A: the model's stock cache (contiguous DynamicCache).
Arm B: TieredPagedKV, everything fits (no window) — the ≤2% clause.
Arm C: TieredPagedKV, constrained hot window — demotion active; runs
       under torch.profiler and audits that every device→host memcpy is
       attributed to a NON-default stream (the side stream), i.e. the
       demotion copies are off the critical path. (nsys is the preferred
       instrument where available; this box's container has none, and
       torch's CUPTI trace carries the same stream attribution.)

Arms A/B are interleaved over N rounds (house A/B discipline) and the
paged arm asserts greedy-token equality with the stock arm every round —
the bytes are the same, so the tokens must be.

Model: any small causal LM; a tiny fast model makes KV overhead a LARGER
fraction of step time, which is conservative for the ≤2% clause.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def build_cache(model, kind, max_tokens, window=None, host_tokens=0):
    if kind == "stock":
        return None
    from experts4bit_qlora.engines.paged_kv import TieredPagedKV
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    return TieredPagedKV(
        cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim,
        dtype=model.dtype, device="cuda", max_tokens=max_tokens,
        hot_window=window, host_tokens=host_tokens)


def decode(model, tok, prompt, n_tokens, cache):
    """Manual greedy loop (infer.timed_decode's shape) with an injectable
    cache object; returns (decode_tok_s, generated_ids, cache)."""
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        kw = {"past_key_values": cache} if cache is not None else {}
        out = model(input_ids=ids, use_cache=True, **kw)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = [int(nxt)]
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_tokens - 1):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(nxt))
        torch.cuda.synchronize()
        dt = time.time() - t0
    return (n_tokens - 1) / dt, toks, past


def main(model_id, n_tokens, rounds, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).cuda()
    prompt = ("The three tiers of a memory hierarchy, from fastest to "
              "slowest, are")
    max_tokens = 64 + n_tokens + 64

    # warm both paths once (JIT, allocator)
    decode(model, tok, prompt, 8, None)
    decode(model, tok, prompt, 8, build_cache(model, "paged", max_tokens))

    a_toks, b_toks = [], []
    for r in range(rounds):
        ta, seq_a, _ = decode(model, tok, prompt, n_tokens, None)
        tb, seq_b, cb = decode(model, tok, prompt, n_tokens,
                               build_cache(model, "paged", max_tokens))
        assert seq_a == seq_b, f"round {r}: paged tokens diverged"
        assert cb.stats()["gather_returns"] == 0
        a_toks.append(ta)
        b_toks.append(tb)
        print(f"BENCH round={r} stock={ta:.2f} paged={tb:.2f} tok/s",
              flush=True)
    med_a = statistics.median(a_toks)
    med_b = statistics.median(b_toks)
    overhead = (med_a - med_b) / med_a

    # ---- arm C: constrained window, profiler-audited side-stream copies
    cache_c = build_cache(model, "paged", max_tokens, window=64,
                          host_tokens=max_tokens)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        tc, seq_c, cache_c = decode(model, tok, prompt, n_tokens, cache_c)
    same_c = seq_c == seq_a          # identical bytes ⇒ identical greedy path
    assert same_c, "arm C diverged from stock — demotion corrupted bytes"
    sc = cache_c.stats()

    # stream attribution from the chrome trace: the compute stream is the
    # one carrying the kernels; demotion DtoH copies must sit elsewhere
    trace = Path(out_dir) / "g6_armC_trace.json"
    prof.export_chrome_trace(str(trace))
    tr = json.loads(trace.read_text())
    kernel_streams = {}
    copy_events = []          # (tid, bytes) — demotion copies are row-sized;
    scalar_copies = 0         # the decode loop's own argmax reads are 8 B
    row_bytes = cache_c.pool.row_bytes
    for e in tr.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        tid = e.get("tid")
        name = e.get("name", "")
        if cat in ("kernel", "Kernel"):
            kernel_streams[tid] = kernel_streams.get(tid, 0) + 1
        if "Memcpy DtoH" in name:
            nb = (e.get("args", {}) or {}).get("bytes", 0)
            if nb >= row_bytes:
                copy_events.append(tid)
            else:
                scalar_copies += 1
    compute_stream = max(kernel_streams, key=kernel_streams.get)
    on_critical = sum(1 for t in copy_events if t == compute_stream)
    off_critical = len(copy_events) - on_critical

    rep = {"model": model_id, "n_tokens": n_tokens, "rounds": rounds,
           "stock_tok_s": a_toks, "paged_tok_s": b_toks,
           "median_stock": med_a, "median_paged": med_b,
           "paged_overhead": overhead,
           "armC_tok_s": tc, "armC_tokens_match_stock": same_c,
           "armC_stats": sc,
           "armC_dtoh_on_compute_stream": on_critical,
           "armC_dtoh_off_compute_stream": off_critical,
           "armC_scalar_dtoh_excluded": scalar_copies,
           "gate_g6": {"overhead_ok": overhead <= 0.02,
                       "demotions_ran": sc["demotions"] > 0,
                       "armC_tokens_ok": same_c,
                       "copies_off_critical_path": on_critical == 0}}
    p = Path(out_dir) / "g6_report.json"
    p.write_text(json.dumps(rep, indent=2))
    print("G6_REPORT " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    main(a.model, a.n_tokens, a.rounds, a.out)
