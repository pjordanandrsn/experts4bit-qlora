#!/usr/bin/env python3
"""Lane P108's measurement (bench/p108/PREREG-p108.md; e4b#359): is e4b's paged path on Gemma-4 worse than
transformers' own forward, judged against transformers' own chaos?

Gemma-4 amplifies bf16 batch-shape variance in the per-expert GEMMs: the NLL of identical tokens moves by tenths of a
nat with only what follows them or how they are batched (#359, docs/SERVING-PARITY.md). One window cannot tell a paged
defect from that. This box reads a DISTRIBUTION over many windows and sets the paged path against a FLOOR: the same
model, the same weights and the same math under perturbations that are arithmetically neutral.

The model loads once, through e4b's streaming loader (NF4 experts on the GPU). Paged attention is registered. With no
paged context bound, the forward is transformers' with e4b's unbound fallback, which carries the sliding window and the
last-key causal alignment (#966).

Text: wikitext-2 test (the K8 corpus). Window k starts at token k * 4096: a ``--prompt``-token prompt and a
``--cont``-token teacher-forced continuation. Windows are processed in groups of ``--group``. Per window and arm, the
continuation's ``cont`` positions are scored on fp32 log-probs (P97's lesson): the true token's NLL and the argmax. KL
against the reference is computed as well.

Arms:
- **R**, the reference: transformers' forward with a DynamicCache, batch 1. The prompt is prefilled in one call, then
  ``cont - 1`` cached decode steps.
- **rep**: R again (the first group only). Its bit-identity is recorded; if it is not identical, it is a floor draw.
- **oneshot** (floor): one forward over the prompt and continuation, no cache.
- **chunk** (floor): R with the prompt prefilled in ``--chunk``-token pieces (the paged path's chunking), batch 1.
- **batch** (floor): R with the group's windows batched (equal lengths, no padding), prefill and decode.
- **paged**, the subject: ``PagedModelRunner``, the group's windows on their own slots, prefill in ``--chunk``-token
  chunks, decoded together (P97's ``_paged_pass`` at its registered bytes) through the fp8 KV pool.
- **mutant_scale**: the paged path with the decode attention's softmax scale halved. The bar must catch it.
- **mutant_window**: the paged path with decode attention ignoring the sliding window. Reported only: it measures whether
  a dropped window is visible at this context length.

Engagement is counted: every decode attention call the subject makes, with its layer and window.

``measure()`` takes the model and the windows, so ``tests/test_p108_box.py`` runs it on CPU on a tiny Gemma-4 (with the
decode attention stood in by SDPA over the pool's own fp8 bytes, honouring the window).

    python p108_box.py --model ID --revision SHA --out OUT.json [--windows 32 --group 8 --prompt 1280 --cont 256]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p97_box  # noqa: E402  (staged at P97's registered bytes: wikitext_windows, _paged_pass, _sync)

FLOORS = ("oneshot", "chunk", "batch")


def stand_in_attention(self, layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
    """SDPA over the pool's own dequantized fp8 K/V, honouring the window (REHEARSAL; P97's stand-in ignores windows)."""
    outs = []
    for b, slot in enumerate(slots):
        kr, vr = self.reference_kv(layer, slot)
        if window:
            kr, vr = kr[-window:], vr[-window:]
        outs.append(torch.nn.functional.scaled_dot_product_attention(
            q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
            scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
    return torch.stack(outs)


class Attention:
    """Installs a decode attention on ``Fp8PagedKV`` (the class) for one paged pass: the real kernel or the stand-in,
    optionally mutated, and counts every call with its layer and window."""

    def __init__(self, stand_in, mutant=None):
        from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
        self.cls, self.orig = Fp8PagedKV, Fp8PagedKV.attention
        self.base = stand_in_attention if stand_in else self.orig
        self.mutant, self.calls = mutant, []

    def __enter__(self):
        base, mutant, calls = self.base, self.mutant, self.calls

        def attention(kv, layer, q, *a, sm_scale=None, window=None, **kw):
            calls.append((int(layer), int(window or 0)))
            if mutant == "scale":
                sm_scale = (sm_scale if sm_scale is not None else q.shape[-1] ** -0.5) * 0.5
            elif mutant == "window":
                window = 0
            return base(kv, layer, q, *a, sm_scale=sm_scale, window=window, **kw)

        self.cls.attention = attention
        return self

    def __exit__(self, *exc):
        self.cls.attention = self.orig


def _score(lp, cont_tokens):
    """Per position: NLL of the true token and the argmax, from fp32 log-probs ``[C, V]`` (on any device)."""
    t = torch.tensor(cont_tokens, device=lp.device)
    return (-lp.gather(1, t[:, None])[:, 0]).tolist(), lp.argmax(-1).tolist()


def _kl(ref, lp):
    lp = lp.to(ref.device)
    return (ref.exp() * (ref - lp)).sum(-1).tolist()


def ref_cached(model, w, P, C, device, chunk=None):
    from transformers.cache_utils import DynamicCache
    cfg = getattr(model.config, "text_config", None) or model.config
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        step = chunk or P
        for s in range(0, P, step):
            out = model(input_ids=torch.tensor([w[s:min(s + step, P)]], device=device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        rows = [out.logits[0, -1].float().log_softmax(-1)]
        for t in range(C - 1):
            out = model(input_ids=torch.tensor([[w[P + t]]], device=device), past_key_values=cache, use_cache=True)
            rows.append(out.logits[0, -1].float().log_softmax(-1))
    return torch.stack(rows).cpu()


def ref_batched(model, ws, P, C, device):
    from transformers.cache_utils import DynamicCache
    cfg = getattr(model.config, "text_config", None) or model.config
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([w[:P] for w in ws], device=device), past_key_values=cache, use_cache=True,
                    logits_to_keep=1)
        rows = [out.logits[:, -1].float().log_softmax(-1).cpu()]
        for t in range(C - 1):
            out = model(input_ids=torch.tensor([[w[P + t]] for w in ws], device=device), past_key_values=cache,
                        use_cache=True)
            rows.append(out.logits[:, -1].float().log_softmax(-1).cpu())
    return [torch.stack([r[i] for r in rows]) for i in range(len(ws))]


def ref_oneshot(model, w, P, C, device):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([w[:P + C - 1]], device=device), logits_to_keep=C)
    return out.logits[0].float().log_softmax(-1).cpu()


def paged(model, ws, P, C, chunk, device, stand_in, mutant=None):
    with Attention(stand_in, mutant) as att:
        lps, _calls, _states, runner, step_ms = p97_box._paged_pass(model, ws, P=P, C=C, chunk=chunk, device=device,
                                                                    stand_in_attention=False, mutant=False)
    return lps, att.calls, step_ms, runner


def measure(model, windows, *, prompt, cont, chunk, group, device, stand_in=False):
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    arms = ("R", "rep") + FLOORS + ("paged", "mutant_scale", "mutant_window")
    per = {a: [] for a in arms}          # per window: {"nll": mean, "agree": vs R, "kl": mean KL(R || arm), "n": C}
    eng = {"calls": 0, "by_window": {}, "groups": 0}
    rep_identical = None
    timing = {}

    def add(arm, wi, lp, ref_lp, cont_tokens, ref_argmax):
        nll, am = _score(lp, cont_tokens)
        rec = {"window": wi, "nll": sum(nll) / len(nll), "argmax_agree": sum(int(a == b) for a, b in zip(am, ref_argmax)) / len(am)}
        if ref_lp is not None:
            kl = _kl(ref_lp, lp)
            rec["kl"] = sum(kl) / len(kl)
        per[arm].append(rec)

    for g0 in range(0, len(windows), group):
        ws = windows[g0:g0 + group]
        eng["groups"] += 1
        t0 = time.time()
        refs = [ref_cached(model, w, P, C, device) for w in ws]
        timing.setdefault("R_s", 0.0)
        timing["R_s"] += time.time() - t0
        ref_am = [r.argmax(-1).tolist() for r in refs]
        for i, (w, r) in enumerate(zip(ws, refs)):
            add("R", g0 + i, r, None, w[P:P + C], ref_am[i])
        if g0 == 0:
            reps = [ref_cached(model, w, P, C, device) for w in ws]
            rep_identical = all(torch.equal(a, b) for a, b in zip(reps, refs))
            for i, (w, x) in enumerate(zip(ws, reps)):
                add("rep", g0 + i, x, refs[i], w[P:P + C], ref_am[i])
            del reps
        for i, w in enumerate(ws):
            add("oneshot", g0 + i, ref_oneshot(model, w, P, C, device), refs[i], w[P:P + C], ref_am[i])
            add("chunk", g0 + i, ref_cached(model, w, P, C, device, chunk=chunk), refs[i], w[P:P + C], ref_am[i])
        for i, x in enumerate(ref_batched(model, ws, P, C, device)):
            add("batch", g0 + i, x, refs[i], ws[i][P:P + C], ref_am[i])
        for arm, mut in (("paged", None), ("mutant_scale", "scale"), ("mutant_window", "window")):
            t1 = time.time()
            lps, calls, step_ms, runner = paged(model, ws, P, C, chunk, device, stand_in, mut)
            timing[f"{arm}_s"] = timing.get(f"{arm}_s", 0.0) + time.time() - t1
            for i, x in enumerate(lps):
                add(arm, g0 + i, x, refs[i], ws[i][P:P + C], ref_am[i])
            if arm == "paged":
                eng["calls"] += len(calls)
                for _layer, win in calls:
                    eng["by_window"][str(win)] = eng["by_window"].get(str(win), 0) + 1
                eng["attn_layers"] = list(runner.attn_layers)
                eng["step_ms"] = round(step_ms, 2)
            del lps, runner
        del refs
        p97_box._sync(device)
    cfg = getattr(model.config, "text_config", None) or model.config
    types = list(getattr(cfg, "layer_types", []) or [])
    sliding = sum(t == "sliding_attention" for t in types)
    n_steps = (C - 1) * eng["groups"]          # each decode step calls every attention layer once for the whole group
    eng.update({"layer_types": types, "sliding_window": getattr(cfg, "sliding_window", None), "rep_identical": rep_identical,
                "expected_calls": n_steps * len(types), "expected_sliding_calls": n_steps * sliding})
    return {"windows": len(windows), "group": group, "prompt": P, "cont": C, "chunk": chunk,
            "rehearsal": {"stand_in_attention": bool(stand_in)}, "per_window": per, "engagement": eng,
            "timing_s": {k: round(v, 1) for k, v in timing.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompt", type=int, default=1280)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--stand-in-attention", action="store_true", help="REHEARSAL: SDPA over the pool's fp8 bytes")
    a = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer

    from experts4bit_qlora import load_moe_4bit_streaming

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model, ckpt_cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
                                              revision=a.revision)
    model.eval()
    load_s = time.time() - t0
    windows = p97_box.wikitext_windows(tok, a.windows, a.prompt, a.cont)
    rec = measure(model, windows, prompt=a.prompt, cont=a.cont, chunk=a.chunk, group=a.group, device="cuda",
                  stand_in=a.stand_in_attention)
    rec = {"model": a.model, "revision": a.revision,
           "loaded_commit": getattr(ckpt_cfg, "_commit_hash", None) or getattr(model.config, "_commit_hash", None),
           "transformers": transformers.__version__, "load_s": round(load_s, 1), **rec,
           "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    pw = rec["per_window"]

    def bias(arm):
        d = [x["nll"] - r["nll"] for x, r in zip(pw[arm], pw["R"])]
        return sum(d) / len(d), sum(abs(v) for v in d) / len(d)

    line = " | ".join(f"{arm} bias {bias(arm)[0]:+.4f} spread {bias(arm)[1]:.4f}" for arm in FLOORS + ("paged", "mutant_scale", "mutant_window"))
    e = rec["engagement"]
    print(f"P108_BOX: windows {rec['windows']} | {line} | rep identical {e['rep_identical']} | calls {e['calls']}/"
          f"{e['expected_calls']} sliding {e['by_window']} | load {load_s:.0f}s | mem {rec['max_mem_gb']} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
