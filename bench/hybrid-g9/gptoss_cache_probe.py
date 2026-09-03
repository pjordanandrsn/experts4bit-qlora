# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Which cache makes chunked teacher forcing equal a full forward on a
sliding-window family?

`ppl_oracle_score` prefills, then feeds the continuation in chunks
through whatever cache the first forward returned. On gpt-oss-20b (12
alternating `sliding_attention` layers, window 128) that is NOT equal to
one full forward: KL(full||chunk) = 0.0165 nats, top-1 93.9%, and
passing `cache_position` changes nothing. Four times Qwen3's entire
parity budget, so any oracle built on it is unusable for this family.

A cache for a hybrid model has to know which layers slide; a cache built
without that knowledge keeps every key on every layer, and the mask a
chunked call derives from it is not the window a single forward applies.
This probe scores the same window through each cache construction
transformers offers and reports the gap, so the oracle can be fixed by
evidence rather than by guessing which class is right.

Every line is prefixed ``CACHEPROBE``."""
from __future__ import annotations

import argparse
import os
import sys

import torch


def _cache_builders(model):
    """(name, zero-argument builder) for each cache this transformers
    can construct for this model. Unavailable ones are reported, not
    skipped silently -- 'we did not try it' and 'it does not exist' are
    different answers."""
    import transformers
    from transformers import cache_utils
    cfg = model.config
    out = [("default (whatever the forward returns)", None)]

    dyn = getattr(cache_utils, "DynamicCache", None)
    if dyn is not None:
        out.append(("DynamicCache()", lambda: dyn()))
        try:
            dyn(config=cfg)                       # probe the signature once
            out.append(("DynamicCache(config=...)", lambda: dyn(config=cfg)))
        except TypeError as e:
            out.append((f"DynamicCache(config=...) UNAVAILABLE: {e}", None))
    for nm in ("HybridCache", "HybridChunkedCache", "SlidingWindowCache",
               "StaticCache"):
        cls = getattr(cache_utils, nm, None)
        if cls is None:
            continue

        def _mk(cls=cls):
            return cls(config=cfg, max_batch_size=1, max_cache_len=4096,
                       device="cuda", dtype=torch.bfloat16)
        out.append((f"{nm}(config=...)", _mk))
    print(f"CACHEPROBE transformers {transformers.__version__}; "
          f"candidates: {[n for n, _ in out]}", flush=True)
    return out


@torch.no_grad()
def _full_logits(model, ids, dev, prompt_len):
    out = model(input_ids=ids[None].to(dev))
    return out.logits[0, prompt_len - 1:-1].cpu().float()


@torch.no_grad()
def _chunked_logits(model, ids, dev, prompt_len, chunk, build, cache_position):
    ids = ids.to(dev)
    kw = {}
    if build is not None:
        kw["past_key_values"] = build()
    out = model(input_ids=ids[None, :prompt_len], use_cache=True, **kw)
    cache = out.past_key_values
    parts = [out.logits[0, -1:].cpu().float()]
    pos, end = prompt_len, ids.numel()
    while pos < end:
        n = min(chunk, end - pos)
        kw2 = dict(input_ids=ids[None, pos:pos + n], use_cache=True,
                   past_key_values=cache,
                   position_ids=torch.arange(pos, pos + n, device=dev)[None])
        if cache_position:
            kw2["cache_position"] = torch.arange(pos, pos + n, device=dev)
        out = model(**kw2)
        cache = out.past_key_values
        parts.append(out.logits[0].cpu().float())
        pos += n
    return torch.cat(parts)[:-1]


def _stats(full, chunk, tgt):
    lf, lc = torch.log_softmax(full, -1), torch.log_softmax(chunk, -1)
    kl = float((lf.exp() * (lf - lc)).sum(-1).mean())
    nll_f = float(-lf.gather(1, tgt[:, None]).mean())
    nll_c = float(-lc.gather(1, tgt[:, None]).mean())
    top1 = float((full.argmax(-1) == chunk.argmax(-1)).float().mean())
    return kl, nll_f, nll_c, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--n-after", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--force-full-attention", action="store_true",
                    help="rewrite every layer to full_attention and drop "
                         "sliding_window before running. The CONTROL: same "
                         "model, same weights, same text, same chunking, with "
                         "only the window feature removed. If the chunked-vs-"
                         "full gap survives it, the gap is not about windows.")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import types

    import step_decomp as sd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    args = types.SimpleNamespace(ppl_source="wikitext", prompt_offset=0,
                                 prompt_span=0, batch=1, prompt_len=a.prompt_len,
                                 ppl_steps=a.n_after, ppl_chat=False,
                                 ppl_chat_suffix="")
    ids, _, _, _, sha = sd._k8_window(args, tok)
    ids = ids[:a.prompt_len + a.n_after].clone()
    tgt = ids[a.prompt_len:]
    _, total = torch.cuda.mem_get_info()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        max_memory={0: f"{max(4, int(total / 2**30) - 6)}GiB", "cpu": "120GiB"})
    model.eval()
    dev = next(model.parameters()).device
    tc = getattr(model.config, "text_config", model.config)
    if a.force_full_attention:
        n_before = sum(1 for x in (getattr(tc, "layer_types", None) or [])
                       if "sliding" in x)
        for cfg_obj in {id(model.config): model.config, id(tc): tc}.values():
            if getattr(cfg_obj, "layer_types", None):
                cfg_obj.layer_types = ["full_attention"] * len(cfg_obj.layer_types)
            if getattr(cfg_obj, "sliding_window", None):
                cfg_obj.sliding_window = None
        for mod in model.modules():                  # per-module copies too
            if hasattr(mod, "sliding_window") and mod.sliding_window:
                mod.sliding_window = None
            if getattr(mod, "attention_type", None) == "sliding_attention":
                mod.attention_type = "full_attention"
        print(f"CACHEPROBE CONTROL: {n_before} sliding layers rewritten to "
              "full_attention, sliding_window dropped", flush=True)
    lt = getattr(tc, "layer_types", None)
    print(f"CACHEPROBE window {ids.numel()} ids, prompt_len={a.prompt_len}, "
          f"scored={tgt.numel()}, chunk={a.chunk}, sha={sha[:12]}, "
          f"sliding_window={getattr(tc, 'sliding_window', None)}, "
          f"n_sliding={sum(1 for x in (lt or []) if 'sliding' in x)}", flush=True)
    full = _full_logits(model, ids, dev, a.prompt_len)
    print(f"CACHEPROBE full-forward nll={float(-torch.log_softmax(full, -1).gather(1, tgt[:, None]).mean()):.5f}",
          flush=True)
    for name, build in _cache_builders(model):
        if build is None and "UNAVAILABLE" in name:
            print(f"CACHEPROBE {name}", flush=True)
            continue
        for cp in (False, True):
            try:
                ch = _chunked_logits(model, ids, dev, a.prompt_len, a.chunk,
                                     build, cp)
                kl, nf, nc, t1 = _stats(full, ch, tgt)
                verdict = "EQUAL" if kl < 1e-6 else ("close" if kl < 1e-3 else "DIFFERENT")
                print(f"CACHEPROBE {name} cache_position={cp}: KL={kl:.6f} "
                      f"nll_full={nf:.5f} nll_chunk={nc:.5f} top1={t1:.4f} -> {verdict}",
                      flush=True)
            except Exception as e:                        # noqa: BLE001
                print(f"CACHEPROBE {name} cache_position={cp}: ERROR "
                      f"{type(e).__name__}: {str(e)[:150]}", flush=True)
    print("CACHEPROBE done", flush=True)


if __name__ == "__main__":
    main()
