# G6 formal runner — paged KV over the HYBRID serving config (Qwen3-30B).
# Stages: bake | run | nsys.
#   run:  arm A = hybrid tier + stock cache (the Stage-1 config: doubles as
#         the zero-regression arm), arm B = + TieredPagedKV everything-fits
#         (the ≤2% clause), arm C = constrained window (demotion active,
#         timed; the nsys stage audits its streams separately).
#   nsys: arm C decode under the profiler for the side-stream clause.
# Needs /root/out/calib.json + the solved placement from `run`.
import argparse
import json
import os
import statistics
import time
from pathlib import Path

ARENA = os.environ.get("G6_ARENA", "/root/q30.arena")
SNAP = os.environ.get("G6_SNAP", "/root/q30")
OUT = Path("/root/out")
OUT.mkdir(exist_ok=True)
PROFILE = "/root/out/route_profile.jsonl"
MANIFEST = "/root/out/placement_g6.json"
PROMPT = "The three tiers of a memory hierarchy, from fastest to slowest, are"
N_TOKENS = 128


def stage_bake():
    from nvme_bake_nf4 import bake_nf4
    bake_nf4(SNAP, ARENA)
    print("BAKE_DONE")


def _load():
    import torch
    from experts4bit_qlora import load_moe_4bit_streaming
    model, _cfg = load_moe_4bit_streaming(
        SNAP, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
        arena=ARENA)
    model.eval()
    return model


def _cache(model, window=None, host_tokens=0):
    import torch
    from experts4bit_qlora.engines.paged_kv import TieredPagedKV
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    return TieredPagedKV(cfg.num_hidden_layers, cfg.num_key_value_heads,
                         head_dim, dtype=torch.bfloat16, device="cuda",
                         max_tokens=64 + N_TOKENS + 64, hot_window=window,
                         host_tokens=host_tokens)


def _decode(model, tok, cache):
    import torch
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        kw = {"past_key_values": cache} if cache is not None else {}
        out = model(input_ids=ids, use_cache=True, **kw)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = [int(nxt)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TOKENS - 1):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(nxt))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (N_TOKENS - 1) / dt, toks, past


def stage_run(threads: int, dram_gb: int, rounds: int):
    import torch
    from transformers import AutoTokenizer
    from nvme_arena import load_index
    from experts4bit_qlora import save_manifest, solve_placement
    from experts4bit_qlora.engines import expert_profile as ep
    from experts4bit_qlora.engines import hybrid as hy
    from experts4bit_qlora.engines.hot_residency import target_modules

    tok = AutoTokenizer.from_pretrained(SNAP)
    model = _load()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    idx = load_index(ARENA)

    trivial = {"schema": "e4b-placement/1",
               "tiers": {"vram": [], "dram": [],
                         "nvme": [[la, e] for la in range(L)
                                  for e in range(E)]},
               "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 1.0}}
    n = hy.enable_hybrid_tier(model, ARENA, trivial, hot_rows=max(E, 128),
                              threads=threads)
    assert n == L
    assert ep.enabled()
    ep.attach(model)
    ids0 = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model.generate(ids0, max_new_tokens=24, do_sample=False)
    ep.flush()
    hy.disable_hybrid_tier(model)

    free_b, _t = torch.cuda.mem_get_info()
    m = solve_placement(
        n_layers=L, n_experts=E, bytes_per_expert=idx["row_stride"],
        vram_budget_bytes=max(0, free_b - 4 * (1 << 30)),
        dram_budget_bytes=dram_gb * (1 << 30),
        calibration="/root/out/calib.json", profile_path=PROFILE)
    sha = save_manifest(m, MANIFEST)
    print(f"PLACEMENT sha={sha[:16]} masses={m['masses']}", flush=True)

    n = hy.enable_hybrid_tier(model, ARENA, MANIFEST, hot_rows=max(E, 128),
                              threads=threads)
    assert n == L
    _decode(model, tok, None)                     # warm
    _decode(model, tok, _cache(model))
    a_t, b_t = [], []
    seq_ref = None
    for r in range(rounds):
        ta, seq_a, _ = _decode(model, tok, None)
        tb, seq_b, cb = _decode(model, tok, _cache(model))
        assert seq_a == seq_b, f"round {r}: paged diverged from stock"
        assert cb.stats()["gather_returns"] == 0
        seq_ref = seq_a
        a_t.append(ta)
        b_t.append(tb)
        print(f"BENCH round={r} stock={ta:.3f} paged={tb:.3f} tok/s",
              flush=True)
    ca = _cache(model, window=32, host_tokens=64 + N_TOKENS + 64)
    tc, seq_c, ca = _decode(model, tok, ca)
    assert seq_c == seq_ref, "arm C diverged from stock — demotion corrupted"
    med_a = statistics.median(a_t)
    med_b = statistics.median(b_t)
    overhead = (med_a - med_b) / med_a
    rep = {"model": "qwen3-30b-a3b-hybrid", "n_tokens": N_TOKENS,
           "rounds": rounds, "stock_tok_s": a_t, "paged_tok_s": b_t,
           "median_stock": med_a, "median_paged": med_b,
           "paged_overhead": overhead,
           "armC_tok_s": tc, "armC_tokens_match_stock": seq_c == seq_ref,
           "armC_stats": ca.stats(), "masses": m["masses"],
           "manifest_sha": sha, "threads": threads,
           "gate_g6": {"overhead_ok": overhead <= 0.02,
                       "armC_tokens_ok": seq_c == seq_ref,
                       "demotions_ran": ca.stats()["demotions"] > 0}}
    (OUT / "g6_report.json").write_text(json.dumps(rep, indent=2))
    print("G6_REPORT " + json.dumps(rep), flush=True)


def stage_nsys(threads: int):
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora.engines import hybrid as hy
    from experts4bit_qlora.engines.hot_residency import target_modules
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = _load()
    mods = target_modules(model)
    n = hy.enable_hybrid_tier(model, ARENA, MANIFEST,
                              hot_rows=max(mods[0].num_experts, 128),
                              threads=threads)
    assert n == len(mods)
    ca = _cache(model, window=32, host_tokens=64 + N_TOKENS + 64)
    torch.cuda.cudart().cudaProfilerStart()
    _decode(model, tok, ca)
    torch.cuda.cudart().cudaProfilerStop()
    print("NSYS_DONE " + json.dumps(ca.stats()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bake", "run", "nsys"])
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--dram-gb", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    os.environ.setdefault("E4B_EXPERT_PROFILE", PROFILE)
    {"bake": stage_bake,
     "run": lambda: stage_run(a.threads, a.dram_gb, a.rounds),
     "nsys": lambda: stage_nsys(a.threads)}[a.stage]()
