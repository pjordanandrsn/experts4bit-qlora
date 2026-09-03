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


class _RouterTap:
    """Record every router's top-k expert choice, per forward.

    A MoE router picks top-k of N experts from the hidden state. Chunked
    and full forwards differ by bf16 rounding; where two experts' logits
    are close, that rounding FLIPS the choice, and a flipped expert is a
    discrete change to that token's output -- an amplifier that turns
    1e-3 of numerical noise into a large logit difference. This tap
    counts those flips so the chunked-vs-full gap can be attributed
    rather than guessed at."""

    def __init__(self, model):
        self.rows = []
        self._h = []
        for name, mod in model.named_modules():
            if name.endswith("mlp.router") or type(mod).__name__.endswith("TopKRouter"):
                self._h.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t) or t.ndim < 2:
                return
            k = min(4, t.shape[-1])
            self.rows.append((name, t.reshape(-1, t.shape[-1]).float()
                              .topk(k, dim=-1).indices.cpu()))
        return fn

    def close(self):
        for h in self._h:
            h.remove()
        return self.rows


def _flip_mask(a_rows, b_rows, n_pos):
    """Per-POSITION flag: did any layer route this token differently?
    Both passes cover the same contiguous positions in order, so the
    concatenation per layer aligns by position."""
    by_a, by_b = {}, {}
    for nm, t in a_rows:
        by_a.setdefault(nm, []).append(t)
    for nm, t in b_rows:
        by_b.setdefault(nm, []).append(t)
    flagged = torch.zeros(n_pos, dtype=torch.bool)
    for nm in sorted(set(by_a) & set(by_b)):
        ca, cb = torch.cat(by_a[nm]), torch.cat(by_b[nm])
        if ca.shape[0] < n_pos or cb.shape[0] < n_pos:
            continue
        for i in range(n_pos):
            if set(ca[i].tolist()) != set(cb[i].tolist()):
                flagged[i] = True
    return flagged


def _kl_split(full, chunk, flagged):
    """Mean KL over positions whose routing FLIPPED vs positions whose
    routing was identical. If the flips carry the disagreement, the
    unflipped side is near zero -- that is the attribution, as opposed
    to noticing that both happen in the same run."""
    lf, lc = torch.log_softmax(full, -1), torch.log_softmax(chunk, -1)
    per = (lf.exp() * (lf - lc)).sum(-1)
    f = flagged[:per.shape[0]]
    out = []
    for name, m in (("flipped", f), ("unflipped", ~f)):
        out.append((name, int(m.sum()),
                    float(per[m].mean()) if int(m.sum()) else float("nan")))
    return out


def _router_flips(a_rows, b_rows):
    """Fraction of (layer, token) top-k sets that differ between two runs."""
    if not a_rows or not b_rows:
        return None
    by_a, by_b = {}, {}
    for nm, t in a_rows:
        by_a.setdefault(nm, []).append(t)
    for nm, t in b_rows:
        by_b.setdefault(nm, []).append(t)
    diff = tot = 0
    for nm in sorted(set(by_a) & set(by_b)):
        ca = torch.cat(by_a[nm]); cb = torch.cat(by_b[nm])
        n = min(ca.shape[0], cb.shape[0])
        sa = [set(r.tolist()) for r in ca[-n:]]
        sb = [set(r.tolist()) for r in cb[-n:]]
        diff += sum(1 for x, y in zip(sa, sb) if x != y)
        tot += n
    return (diff, tot, diff / max(1, tot))


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
    ap.add_argument("--wide-window", action="store_true",
                    help="widen sliding_window past the sequence instead of "
                         "removing it. The CONTROL: same model, weights, text, "
                         "chunking AND mask code path, with the window's WIDTH "
                         "made irrelevant -- a sliding layer whose window "
                         "exceeds the context is exactly full attention. If "
                         "the chunked-vs-full gap survives, it is not about "
                         "windows. (Setting sliding_window=None instead makes "
                         "transformers raise from create_sliding_window_causal_"
                         "mask, since layer_types still names sliding layers.)")
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
    if a.wide_window:
        WIDE = 1 << 20
        n_sliding = sum(1 for x in (getattr(tc, "layer_types", None) or [])
                        if "sliding" in x)
        seen = {}
        for mod in model.modules():                  # every config object in play
            c = getattr(mod, "config", None)
            if c is not None and getattr(c, "sliding_window", None):
                seen[id(c)] = c
            if getattr(mod, "sliding_window", None):
                mod.sliding_window = WIDE
        for c in (model.config, tc):
            if getattr(c, "sliding_window", None):
                seen[id(c)] = c
        for c in seen.values():
            c.sliding_window = WIDE
        print(f"CACHEPROBE CONTROL: {n_sliding} sliding layers kept, window "
              f"widened to {WIDE} on {len(seen)} config object(s) -- wider than "
              "the context, so they are full attention with the same code path",
              flush=True)
    lt = getattr(tc, "layer_types", None)
    print(f"CACHEPROBE window {ids.numel()} ids, prompt_len={a.prompt_len}, "
          f"scored={tgt.numel()}, chunk={a.chunk}, sha={sha[:12]}, "
          f"sliding_window={getattr(tc, 'sliding_window', None)}, "
          f"n_sliding={sum(1 for x in (lt or []) if 'sliding' in x)}", flush=True)
    tap_full = _RouterTap(model)
    full = _full_logits(model, ids, dev, a.prompt_len)
    rows_full = tap_full.close()
    print(f"CACHEPROBE full-forward nll={float(-torch.log_softmax(full, -1).gather(1, tgt[:, None]).mean()):.5f}",
          flush=True)
    for name, build in _cache_builders(model):
        if build is None and "UNAVAILABLE" in name:
            print(f"CACHEPROBE {name}", flush=True)
            continue
        for cp in (False, True):
            try:
                tap_ch = _RouterTap(model)
                ch = _chunked_logits(model, ids, dev, a.prompt_len, a.chunk,
                                     build, cp)
                rows_ch = tap_ch.close()
                flips = _router_flips(rows_full, rows_ch)
                kl, nf, nc, t1 = _stats(full, ch, tgt)
                if flips is not None:
                    d, t, frac = flips
                    print(f"CACHEPROBE   router top-k flips full-vs-chunk: "
                          f"{d}/{t} = {frac:.4%} of (layer, token) choices",
                          flush=True)
                    mask = _flip_mask(rows_full, rows_ch, ids.numel())
                    # the scored logit at row j predicts ids[prompt_len+j],
                    # computed AT position prompt_len+j-1
                    sc = mask[a.prompt_len - 1:a.prompt_len - 1 + full.shape[0]]
                    parts = _kl_split(full, ch, sc)
                    print("CACHEPROBE   KL by routing: " + "; ".join(
                        f"{n} n={k} meanKL={v:.6f}" for n, k, v in parts),
                        flush=True)
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
