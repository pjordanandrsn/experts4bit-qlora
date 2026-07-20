"""Our-arm driver for the gpt-oss hybrid-vs-llama comparison: load gpt-oss with
NF4 experts offloaded to pinned host RAM, make it HYBRID via enable_hot_residency
(K experts/layer GPU-resident, the rest cold-streamed over PCIe), and time greedy
decode. Metric = n_tokens / t_decode (the house decode-tok/s definition, same as
experts4bit_qlora.infer.timed_decode). llama.cpp's --n-cpu-moe arm runs
separately on the same box (bench/gptoss_ab_pod.sh); this file is the ours side
and the dev-box gate that gpt-oss decodes end-to-end through the hot path.

  MODEL=openai/gpt-oss-20b HOT_K=4 BENCH_TOKENS=64 python bench/bench_gptoss_hybrid.py
"""
import json
import os
import time

import torch

MODEL = os.environ.get("MODEL", "openai/gpt-oss-20b")
DEVICE = "cuda"
DTYPE = torch.bfloat16
HOT_K = int(os.environ.get("HOT_K", "4"))            # experts/layer kept GPU-resident
BENCH_TOKENS = int(os.environ.get("BENCH_TOKENS", "64"))
PROMPT = os.environ.get("PROMPT", "Explain mixture-of-experts routing in two sentences:")
OUT = os.environ.get("OUT", "")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def timed_decode(model, tok, prompt, n_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    torch.cuda.synchronize(); t0 = time.time()
    out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize(); t_prefill = time.time() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    pieces = []
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n_tokens):
        pieces.append(nxt)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize(); t_decode = time.time() - t0
    text = tok.decode(torch.cat(pieces, dim=1)[0], skip_special_tokens=True)
    return text, t_prefill, n_tokens / t_decode


def main():
    from transformers import AutoTokenizer
    from experts4bit_qlora import enable_hot_residency
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    # offload=False: enable_hot_residency needs the packed experts RESIDENT to
    # partition them (offload=True evicts them to 0-element GPU stubs FIRST ->
    # the streaming/offload+hot combo is unsupported, a separate increment).
    # The hybrid VRAM win comes from freeing the base AFTER the split (below).
    from experts4bit_qlora import ExpertsNbit
    log(f"loading {MODEL} (NF4 experts, resident) ...")
    t0 = time.time()
    model, _ = load_moe_4bit_streaming(MODEL, DEVICE, DTYPE, r=8, alpha=16,
                                       offload=False, pin=True, quant_type="nf4")
    model.to(DEVICE)
    log(f"loaded in {time.time()-t0:.0f}s")

    # The generic loader wraps experts in ExpertsLoRA (training adapters), and
    # residency gates on STANDALONE experts — unwrap to the 4-bit base for this
    # inference-only driver (gpt-oss loads bare, so this is a no-op there).
    # 2026-07-20 pod receipt: Gemma-4 via the streaming loader hit the
    # ExpertsLoRA NotImplementedError in enable_hot_residency without this.
    from experts4bit_qlora.lora import ExpertsLoRA
    unwrapped = 0
    for m in list(model.modules()):
        for cn, child in list(m.named_children()):
            if isinstance(child, ExpertsLoRA):
                setattr(m, cn, child.base)
                unwrapped += 1
    if unwrapped:
        log(f"unwrapped {unwrapped} ExpertsLoRA wrappers -> standalone 4-bit experts")

    n_moe = sum(1 for m in model.modules() if isinstance(m, ExpertsNbit))
    hot_sets = [torch.arange(HOT_K) for _ in range(n_moe)]
    n = enable_hot_residency(model, hot_sets, device=DEVICE)
    log(f"hybrid enabled: {n}/{n_moe} MoE layers, HOT_K={HOT_K} resident/layer")
    if n != n_moe:
        log("WARNING: not every MoE layer patched — gpt-oss eligibility?")

    # Realize the VRAM win: _HotResidency holds its own hot(GPU)+cold(pinned-CPU)
    # stacks, so the module's original resident packed weights are dead weight in
    # pure bf16 no_grad inference (the _fwd fallback that reads them only fires
    # for grad/non-bf16). Free them so resident = hot-K experts + non-expert only.
    # DRIVER-LEVEL + inference-only; the library will formalize this (the base-free
    # / offload-compose increment) later.
    freed = 0
    for m in model.modules():
        if isinstance(m, ExpertsNbit) and hasattr(m, "_hot_residency"):
            for nm in ("gate_up_proj", "down_proj"):
                p = getattr(m, nm)
                freed += p.numel()
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
    torch.cuda.empty_cache()
    log(f"freed base expert weights ({freed/1e9:.2f} G-elems) — hybrid is now VRAM-lean")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval(); model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(MODEL)
    text, t_prefill, toks = timed_decode(model, tok, PROMPT, BENCH_TOKENS)
    peak = torch.cuda.max_memory_allocated() / 1e9
    log(f"DECODE {toks:.2f} tok/s | prefill {t_prefill:.2f}s | peak {peak:.2f} GB | patched {n}/{n_moe}")
    log(f"sample: {text[:160]!r}")
    rec = dict(model=MODEL, arm="ours-hybrid-nf4", hot_k=HOT_K, n_moe=n_moe,
               patched=n, decode_toks=round(toks, 3), prefill_s=round(t_prefill, 3),
               peak_gb=round(peak, 3), bench_tokens=BENCH_TOKENS,
               coherent=bool(text.strip()))
    print("RESULT " + json.dumps(rec), flush=True)
    if OUT:
        json.dump(rec, open(OUT, "w"), indent=1)
    # gate: gpt-oss served through the hybrid path, produced a rate, stayed coherent
    assert n == n_moe and toks > 0 and text.strip(), "hybrid gpt-oss decode gate FAILED"
    log("GATE OK — gpt-oss decodes through the hot/cold hybrid engine")


if __name__ == "__main__":
    main()
