# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Where do the two gpt-oss oracle arms diverge?

On the same weights and the same window the e4b-loaded oracle (NF4
experts, transformers' eager attention) scored ppl 1336 and plain
transformers scored 2361. NF4 cannot make a model 1.8x better than its
bf16 self, so one arm assembles the input differently. This script
names the arm and the layer:

* ``chunked-vs-full`` per arm: the chunked teacher-forced scorer
  (prefill + cached continuation with explicit ``position_ids``) against
  ONE full forward on the same ids. Equal on a plain-causal model; a gap
  here on gpt-oss means the cache/position handling across chunks is
  wrong for sliding-window layers (with and without ``cache_position``).
* ``config``: the attention-relevant config each arm actually runs with
  (``layer_types``, ``sliding_window``, ``_attn_implementation``, the
  attention class, the sink parameter's norm at layer 0).
* ``cross``: per-layer hidden-state cosine between the arms on the
  scored positions, plus full-vocab KL(upstream || e4b), both NLLs and
  top-1 agreement on the same targets. The first layer whose cosine
  drops is where the arms part.

Every line is prefixed ``ARMDIFF``."""
from __future__ import annotations

import argparse
import gc
import os
import sys
import types

import torch


def _window(model_id, prompt_len, n_after):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import step_decomp as sd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    a = types.SimpleNamespace(ppl_source="wikitext", prompt_offset=0, prompt_span=0,
                              batch=1, prompt_len=prompt_len, ppl_steps=n_after,
                              ppl_chat=False, ppl_chat_suffix="")
    ids, _, _, _, sha = sd._k8_window(a, tok)
    return tok, ids[:prompt_len + n_after + 1].clone(), sha


def _set_eager(model):
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    model.eval()


def _cfg_line(tag, model):
    cfg = model.config
    tc = getattr(cfg, "text_config", cfg)
    lt = getattr(tc, "layer_types", None)
    attn0 = None
    sinks = None
    for name, mod in model.named_modules():
        if name.endswith("layers.0.self_attn"):
            attn0 = type(mod).__name__
            s = getattr(mod, "sinks", None)
            sinks = None if s is None else round(float(s.detach().float().norm()), 4)
            break
    print(f"ARMDIFF config[{tag}]: impl={getattr(tc, '_attn_implementation', None)} "
          f"sliding_window={getattr(tc, 'sliding_window', None)} "
          f"layer_types[:4]={lt[:4] if lt else None} n_sliding={sum(1 for x in (lt or []) if 'sliding' in x)} "
          f"attn0={attn0} sinks0_norm={sinks} dtype={next(model.parameters()).dtype}", flush=True)


@torch.no_grad()
def _full(model, ids, dev):
    out = model(input_ids=ids[None].to(dev), output_hidden_states=True)
    # .cpu() BEFORE .float(): a [T, 201088] fp32 copy on a card already
    # holding a 42 GB bf16 model in a 32 GB budget is the OOM
    hs = [h[0].cpu().float() for h in out.hidden_states]
    lg = out.logits[0].cpu().float()
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return lg, hs


@torch.no_grad()
def _chunked(model, ids, prompt_len, dev, chunk, cache_position):
    ids = ids.to(dev)
    out = model(input_ids=ids[None, :prompt_len], use_cache=True)
    cache = out.past_key_values
    logits = [out.logits[0, -1:].float().cpu()]
    pos, end = prompt_len, ids.numel()
    while pos < end:
        n = min(chunk, end - pos)
        kw = dict(input_ids=ids[None, pos:pos + n], use_cache=True, past_key_values=cache,
                  position_ids=torch.arange(pos, pos + n, device=dev)[None])
        if cache_position:
            kw["cache_position"] = torch.arange(pos, pos + n, device=dev)
        out = model(**kw)
        cache = out.past_key_values
        logits.append(out.logits[0].cpu().float())
        pos += n
    return torch.cat(logits)[:-1]          # predicts ids[prompt_len : end]


def _nll(logits, tgt):
    return float(-torch.log_softmax(logits, -1).gather(1, tgt[:, None]).mean())


def _kl(p_logits, q_logits):
    lp = torch.log_softmax(p_logits, -1)
    lq = torch.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def _top1(a, b):
    return float((a.argmax(-1) == b.argmax(-1)).float().mean())


def _compare_chunked(tag, model, ids, prompt_len, dev, full_logits, chunk):
    tgt = ids[prompt_len:]
    fl = full_logits[prompt_len - 1:-1]
    for cp in (False, True):
        try:
            cl = _chunked(model, ids, prompt_len, dev, chunk, cp)
            print(f"ARMDIFF chunked-vs-full[{tag}] cache_position={cp}: "
                  f"nll_full={_nll(fl, tgt):.4f} nll_chunk={_nll(cl, tgt):.4f} "
                  f"KL(full||chunk)={_kl(fl, cl):.5f} top1={_top1(fl, cl):.4f} "
                  f"maxabs={float((fl - cl).abs().max()):.3f}", flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"ARMDIFF chunked-vs-full[{tag}] cache_position={cp}: ERROR {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--n-after", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument("--save", default="/root/ctrl/armdiff_upstream.pt")
    a = ap.parse_args()
    tok, ids, sha = _window(a.model, a.prompt_len, a.n_after)
    tgt = ids[a.prompt_len:]
    print(f"ARMDIFF window: {ids.numel()} ids, prompt_len={a.prompt_len}, scored={tgt.numel()}, "
          f"sha={sha[:12]}", flush=True)

    # ---- arm 1: upstream (transformers' own load)
    if not a.skip_upstream or not os.path.exists(a.save):
        from transformers import AutoModelForCausalLM
        free, total = torch.cuda.mem_get_info()
        budget = max(4, int(total / 2**30) - 6)      # leave ~6 GiB for activations
        up = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
            max_memory={0: f"{budget}GiB", "cpu": "120GiB"})
        print(f"ARMDIFF upstream budget: {budget} GiB on GPU 0 of "
              f"{total / 2**30:.1f} GiB total", flush=True)
        _set_eager(up)
        dev = next(up.parameters()).device
        _cfg_line("upstream", up)
        up_logits, up_hs = _full(up, ids, dev)
        print(f"ARMDIFF full[upstream]: nll={_nll(up_logits[a.prompt_len - 1:-1], tgt):.4f} "
              f"ppl={torch.exp(torch.tensor(_nll(up_logits[a.prompt_len - 1:-1], tgt))):.1f}", flush=True)
        _compare_chunked("upstream", up, ids, a.prompt_len, dev, up_logits, a.chunk)
        torch.save({"logits": up_logits, "hs": up_hs}, a.save)
        del up
        gc.collect()
        torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(a.save)
    up_logits, up_hs = saved["logits"], saved["hs"]

    # ---- arm 2: e4b-loaded (NF4 experts, HF attention, shim NOT registered)
    from experts4bit_qlora import load_moe_4bit_streaming
    e4b, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, r=8, alpha=16,
                                     quant_type="nf4", arena=a.arena)
    _set_eager(e4b)
    dev = torch.device("cuda")
    _cfg_line("e4b", e4b)
    e_logits, e_hs = _full(e4b, ids, dev)
    print(f"ARMDIFF full[e4b]: nll={_nll(e_logits[a.prompt_len - 1:-1], tgt):.4f} "
          f"ppl={torch.exp(torch.tensor(_nll(e_logits[a.prompt_len - 1:-1], tgt))):.1f}", flush=True)
    _compare_chunked("e4b", e4b, ids, a.prompt_len, dev, e_logits, a.chunk)

    # ---- cross-arm
    L = min(len(up_hs), len(e_hs))
    first_bad = None
    for l in range(L):
        u = up_hs[l][a.prompt_len:]
        e = e_hs[l][a.prompt_len:]
        cos = torch.nn.functional.cosine_similarity(u, e, dim=-1)
        rel = float((u - e).norm() / u.norm().clamp_min(1e-9))
        flag = ""
        if first_bad is None and float(cos.min()) < 0.98:
            first_bad = l
            flag = "  <-- first layer below 0.98"
        print(f"ARMDIFF cross layer {l:2d}: cos mean={float(cos.mean()):.5f} min={float(cos.min()):.5f} "
              f"rel={rel:.4f}{flag}", flush=True)
    fl_u = up_logits[a.prompt_len - 1:-1]
    fl_e = e_logits[a.prompt_len - 1:-1]
    print(f"ARMDIFF cross logits: KL(upstream||e4b)={_kl(fl_u, fl_e):.5f} KL(e4b||upstream)={_kl(fl_e, fl_u):.5f} "
          f"top1={_top1(fl_u, fl_e):.4f} nll_up={_nll(fl_u, tgt):.4f} nll_e4b={_nll(fl_e, tgt):.4f}",
          flush=True)
    print(f"ARMDIFF verdict: first divergent layer = {first_bad} "
          f"({'hidden states agree through the stack' if first_bad is None else 'arms part here'})",
          flush=True)


if __name__ == "__main__":
    main()
