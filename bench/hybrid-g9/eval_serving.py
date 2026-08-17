# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""G9: what continuous batching and chunked prefill actually cost each
other.

The gate asks three things and the protocol forbids reporting any of them
alone: aggregate throughput at batch, decode degradation under continuous
prompt arrival, and TTFT p50 under load with queueing included. This
harness measures all three from the same engine and prints them together.

The measurement that carries the phase is the **chunk sweep**. Chunk size
trades TTFT against decode: a larger chunk ingests a prompt in fewer,
bigger attentions (better prefill throughput, more resident bf16 staging)
while every resident decoder waits the whole chunk; a smaller chunk
interleaves finely (tighter TTFT) at more per-step overhead. The
directive asks for the curve rather than a chosen point, so this reports
the curve.

Two arms make the degradation number falsifiable rather than decorative:

* **quiet** — a fixed set of sequences decodes with no new arrivals. This
  is the ceiling: decode with the machine to itself.
* **loaded** — the same decoding work with prompts arriving throughout,
  so prefill chunks interleave with every decode step.

Degradation is loaded-decode-rate against quiet-decode-rate on the SAME
decode work. Comparing aggregate rates instead would flatter the loaded
arm, because prefill tokens are tokens too and would pad the number that
is supposed to expose their cost.
"""
from __future__ import annotations

import argparse
import json

import time
from pathlib import Path

import torch


def _pool(tok, n_prompts, prompt_len, seed=1689):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (ids.numel() - prompt_len) // max(1, n_prompts))
    out = []
    for i in range(n_prompts):
        s = i * step
        out.append(ids[s:s + prompt_len].tolist())
    return out


def _engine(model, kv, *, chunk, budget, max_seqs, slots):
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler
    runner = PagedModelRunner(model, kv, device="cuda")
    return ContinuousScheduler(runner=runner, max_seqs=max_seqs,
                               kv_slots=slots, chunk_tokens=chunk,
                               max_prefill_tokens_per_step=budget)


def run_quiet(model, kv, prompts, *, chunk, budget, max_seqs, gen_tokens):
    """All prompts admitted up front; measure decode with no arrivals."""
    sched = _engine(model, kv, chunk=chunk, budget=budget,
                    max_seqs=max_seqs, slots=max_seqs)
    for p in prompts:
        sched.add_request(p, max_new_tokens=gen_tokens)
    # drain prefill first so the decode measurement is decode
    while any(r.phase.value == "prefill" for r in sched.active.values()) \
            or sched.queue:
        if sched.step().is_empty:
            break
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decode_steps = 0
    while sched.active:
        plan = sched.step()
        if plan.is_empty:
            break
        decode_steps += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    emitted = sum(len(r.out) for r in sched.done)
    return {"decode_tok_s": emitted / dt if dt else None,
            "decode_steps": decode_steps, "wall_s": dt,
            "tokens": emitted, "stats": sched.stats()}


def run_loaded(model, kv, prompts, *, chunk, budget, max_seqs, gen_tokens,
               arrival_every_steps=2):
    """Prompts arrive throughout, so prefill interleaves with decode."""
    sched = _engine(model, kv, chunk=chunk, budget=budget,
                    max_seqs=max_seqs, slots=max_seqs)
    pending = list(prompts)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    while pending or sched.queue or sched.active:
        if pending and steps % arrival_every_steps == 0:
            sched.add_request(pending.pop(0), max_new_tokens=gen_tokens)
        if sched.step().is_empty and not pending:
            break
        steps += 1
        if steps > 20000:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    st = sched.stats()
    emitted = sum(len(r.out) for r in sched.done)
    return {"decode_tok_s": emitted / dt if dt else None,
            "aggregate_tok_s": (emitted + st["prefill_tokens"]) / dt
            if dt else None,
            "wall_s": dt, "tokens": emitted, "steps": steps, "stats": st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--chunks", default="8,32,128,512")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--max-tokens-per-seq", type=int, default=1024)
    ap.add_argument("--arrival-every-steps", type=int, default=8,
                    help="steps between prompt arrivals in the loaded arm")
    ap.add_argument("--out", default="bench/hybrid-g9")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    from experts4bit_qlora.engines.paged_attention import register

    torch.manual_seed(1689)
    tok = AutoTokenizer.from_pretrained(a.model)
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, r=8,
                                       alpha=16, quant_type="nf4",
                                       arena=a.arena)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    enable_nvme_residency(model, a.arena, [list(range(E)) for _ in mods],
                          hot_rows=E, device="cuda")
    cfg = model.config
    hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size
                                            // cfg.num_attention_heads)
    register(model)
    prompts = _pool(tok, a.n_prompts, a.prompt_len)

    rows = []
    for chunk in [int(c) for c in a.chunks.split(",")]:
        kv = Fp8PagedKV(L, hkv, hd, batch=a.batch,
                        max_tokens_per_seq=a.max_tokens_per_seq,
                        k_groups=4, device="cuda")
        quiet = run_quiet(model, kv, prompts[:a.batch], chunk=chunk,
                          budget=chunk, max_seqs=a.batch,
                          gen_tokens=a.gen_tokens)
        kv = Fp8PagedKV(L, hkv, hd, batch=a.batch,
                        max_tokens_per_seq=a.max_tokens_per_seq,
                        k_groups=4, device="cuda")
        loaded = run_loaded(model, kv, prompts, chunk=chunk, budget=chunk,
                            max_seqs=a.batch, gen_tokens=a.gen_tokens,
                            arrival_every_steps=a.arrival_every_steps)
        deg = (1 - loaded["decode_tok_s"] / quiet["decode_tok_s"]
               if quiet["decode_tok_s"] else None)
        st = loaded["stats"]
        rows.append({
            "chunk_tokens": chunk,
            "quiet_decode_tok_s": quiet["decode_tok_s"],
            "loaded_decode_tok_s": loaded["decode_tok_s"],
            "loaded_aggregate_tok_s": loaded["aggregate_tok_s"],
            "decode_degradation": deg,
            "ttft_p50_s": st["ttft_p50"], "ttft_p99_s": st["ttft_p99"],
            "queue_wait_p50_s": st["queue_wait_p50"],
            "per_stream_tok_s": st["per_stream_tok_s_mean"],
            "completed": st["completed"], "steps": loaded["steps"],
            # WORKLOAD MIX, reported because it decides how to read the
            # degradation number: a run whose tokens are mostly prefill is
            # a prefill benchmark wearing a serving label, and its
            # "decode degradation" says more about the mix than the engine
            "prefill_tokens": st["prefill_tokens"],
            "decode_tokens": loaded["tokens"],
            "prefill_share": (st["prefill_tokens"]
                              / max(1, st["prefill_tokens"]
                                    + loaded["tokens"])),
        })
        r = rows[-1]
        print(f"CHUNK {chunk:4d}: quiet_decode "
              f"{r['quiet_decode_tok_s']:7.2f} loaded_decode "
              f"{r['loaded_decode_tok_s']:7.2f} "
              f"deg {100 * r['decode_degradation']:+5.1f}% | agg "
              f"{r['loaded_aggregate_tok_s']:7.2f} tok/s | ttft p50 "
              f"{r['ttft_p50_s']:.2f}s p99 {r['ttft_p99_s']:.2f}s | "
              f"prefill_share {100 * r['prefill_share']:.0f}%",
              flush=True)

    ok = [r for r in rows if (r["decode_degradation"] or 1) <= 0.20]
    best = min(rows, key=lambda r: (r["decode_degradation"] or 1e9))
    rep = {"model": a.model, "batch": a.batch,
           "prompt_len": a.prompt_len, "gen_tokens": a.gen_tokens,
           "arrival_every_steps": a.arrival_every_steps, "rows": rows,
           # the clause asks whether the ENGINE can hold <=20%, and the
           # directive asks for the curve rather than a silently chosen
           # point — so the verdict is "at which chunk sizes", not "at
           # every chunk size". Requiring all of them would let a
           # deliberately bad configuration veto a passing engine.
           "gate_degradation_20pct": bool(ok),
           "passing_chunks": [r["chunk_tokens"] for r in ok],
           "best_by_degradation": best["chunk_tokens"]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = a.model.split("/")[-1]
    (Path(a.out) / f"g9_serving_{name}.json").write_text(
        json.dumps(rep, indent=2))
    print("G9_SERVING " + json.dumps(
        {"gate_degradation_20pct": rep["gate_degradation_20pct"],
         "passing_chunks": rep["passing_chunks"],
         "best_chunk": rep["best_by_degradation"]}), flush=True)



if __name__ == "__main__":
    main()
