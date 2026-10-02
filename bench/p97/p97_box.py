#!/usr/bin/env python3
"""Lane P97's measurement (bench/p97/PREREG-p97.md; e4b#564): the paged runner against transformers' own forward,
on the SAME in-process weights, for one model.

The model loads once, through e4b's streaming loader (NF4 experts on the GPU). Paged attention is registered, so the
attention layers run e4b's implementation, but with no paged context bound it falls back to SDPA. The linear-attention
wrapper likewise passes through when unbound. So, unbound, the model is transformers' forward with a DynamicCache:
the reference. Bound, it is ``PagedModelRunner``: the fp8 KV pool, chunked prefill, batched decode, and a hybrid
model's per-slot linear state on a compact pool.

Text: wikitext-2 test, the K8 corpus. ``--windows`` windows, the k-th starting at token k * 4096, each a
``--prompt``-token prompt and a ``--cont``-token teacher-forced continuation.
- **Reference:** one window at a time.
- **Paged:** every window bound to a slot, prefilled in ``--chunk``-token chunks, then decoded together, one row per
  window.

Per step it compares the two next-token distributions: KL(ref || paged) in nats, the true token's nll, and whether
the argmax agrees.

Engagement is counted, not assumed. Every decode-attention call the fp8 pool serves is counted with the pool layer it
names, and every per-slot store of a linear layer's state. The reducer holds both counts against the run's shape:
(cont - 1) x attention layers kernel calls over pool layers 0..L-1, and linear layers x (windows x prompt chunks +
cont - 1) stores. It also records the loaded commit and whether transformers took its fast (fla) or torch path for
the Gated DeltaNet layers.

Two REHEARSAL knobs exist for the A2000 (sm_86: no native e4m3, 12 GB). They are recorded under ``rehearsal``, and the
reducer refuses a record that set either one:
- ``--offload`` streams experts from host RAM;
- ``--stand-in-attention`` replaces the fp8 kernel with SDPA over the pool's own dequantized fp8 bytes.

The linear state is compared directly. After the last step, each window's pooled conv window and recurrent state are
held against transformers' own cache for that window, layer by layer (relative Frobenius error). The linear layers
before the first attention layer see the same tokens on both paths, so their state must agree up to bf16 arithmetic
whatever the fp8 KV does downstream; that is the model-independent check.

A hybrid model also runs a MUTANT paged pass: each decode step writes a row's state back to the NEXT row's slot
(rotated by one), a slot-mapping bug. Prefill is unaffected, so each window starts right and then reads another's
state. Its numbers are recorded beside the real pass; the reducer voids a reading whose mutant would have passed
either gate, because then that gate could not see broken state.

``measure()`` is the whole comparison, separate from loading, so ``tests/test_p97_box.py`` runs it on CPU on a tiny
hybrid model (stand-in attention) in CI.

    python p97_box.py --model ID --revision SHA --tag NAME --out OUT.json [--windows 4 --prompt 512 --cont 256]
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def wikitext_windows(tok, n, prompt, cont):
    """The k-th window starts at token k * 4096 of wikitext-2-raw test, joined as the K8 corpus joins it."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for k in range(n):
        a = k * 4096
        w = ids[a:a + prompt + cont]
        assert w.numel() == prompt + cont, f"window {k} has {w.numel()} tokens"
        out.append(w.tolist())
    return out


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _paged_pass(model, windows, *, P, C, chunk, device, stand_in_attention, mutant):
    """One paged pass: every window on its own slot, chunked prefill, decode together, teacher-forced. Returns the
    per-window next-token log-probs, the counted engagement, each window's pooled linear state after the last step,
    the runner and the decode step time. ``mutant`` is the MUTANT arm: a decode step writes each row's state back to
    the NEXT row's slot (rotated by one), a slot-mapping bug. Prefill (one row) is unaffected, so every window starts
    right and then reads another's state -- the instrument must see that."""
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    hkv, hd = _kv_geometry(cfg)
    kv = Fp8PagedKV(kv_layers(model, int(text_cfg.num_hidden_layers)), hkv, hd, batch=len(windows),
                    max_tokens_per_seq=P + C + 16, device=device)
    runner = PagedModelRunner(model, kv, device=device)
    pool = runner.linear_state
    calls = {"attention": 0, "attention_layers": set(), "store": 0, "store_layers": set()}
    kernel = kv.attention

    if stand_in_attention:
        def kernel(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
            outs = []
            for b, slot in enumerate(slots):
                kr, vr = kv.reference_kv(layer, slot)
                outs.append(torch.nn.functional.scaled_dot_product_attention(
                    q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                    scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
            return torch.stack(outs)

    def counted_attention(layer, *args, **kw):
        calls["attention"] += 1
        calls["attention_layers"].add(int(layer))
        return kernel(layer, *args, **kw)

    kv.attention = counted_attention
    last = {}
    inner = model.forward

    def keep(*args, **kw):
        o = inner(*args, **kw)
        last["logits"] = o.logits
        return o

    if pool is not None:
        store = pool.store

        def counted_store(layer, slots, lal):
            calls["store"] += 1
            calls["store_layers"].add(int(layer))
            slots = list(slots)
            if mutant and len(slots) > 1:
                slots = slots[1:] + slots[:1]                   # row k's state lands in row k+1's slot
            return store(layer, slots, lal)

        pool.store = counted_store
    model.forward = keep
    lps = [[] for _ in windows]
    try:
        for i, w in enumerate(windows):
            runner.bind(i, i, w[:P])
        with torch.no_grad():
            for i in range(len(windows)):
                for start in range(0, P, chunk):
                    runner.run_prefill([(i, start, min(chunk, P - start))])
                lps[i].append(last["logits"][0, -1].float().log_softmax(-1).cpu())
                runner.tokens[i][-1] = windows[i][P]               # teacher-forced
            rows = list(range(len(windows)))
            _sync(device)
            t1 = time.time()
            for t in range(C - 1):
                runner.run_decode(rows)
                lg = last["logits"][:, -1].float().log_softmax(-1).cpu()
                for i in rows:
                    lps[i].append(lg[i])
                    runner.tokens[i][-1] = windows[i][P + t + 1]
            _sync(device)
            step_ms = (time.time() - t1) * 1000 / max(C - 1, 1)
        states = ([{li: (pool.conv[li][i].float().cpu(), pool.rec[li][i].float().cpu()) for li in pool.conv}
                   for i in range(len(windows))] if pool is not None else None)
    finally:
        model.forward = inner
        if pool is not None:
            del pool.store                                          # the class method again
        for i in range(len(windows)):
            runner.free_slot(i)
    return [torch.stack(x) for x in lps], calls, states, runner, step_ms


def _cache_states(cache, layers):
    """Each linear layer's conv window and recurrent state from transformers' own ``DynamicCache`` (batch row 0)."""
    out = {}
    for li in layers:
        lay = cache.layers[li]
        out[li] = (lay.conv_states[0][0].float().cpu(), lay.recurrent_states[0][0].float().cpu())
    return out


def _state_errors(got, ref, pre):
    """Per linear layer, the largest relative Frobenius error over windows of the pooled conv / recurrent state against
    transformers' cache; and the largest over the layers before the first attention layer (``pre``), whose inputs are
    the same tokens on both paths."""
    if got is None:
        return None
    per = {}
    for li in ref[0]:
        c = max(float((g[li][0] - r[li][0]).norm() / r[li][0].norm().clamp_min(1e-30)) for g, r in zip(got, ref))
        h = max(float((g[li][1] - r[li][1]).norm() / r[li][1].norm().clamp_min(1e-30)) for g, r in zip(got, ref))
        per[str(li)] = {"conv": c, "rec": h}
    return {"pre_attention_linear_layers": pre, "rel_err": per,
            "pre_attention_max_rel_err": max((max(per[str(li)].values()) for li in pre), default=None),
            "all_linear_max_rel_err": max(max(v.values()) for v in per.values())}


def _compare(ref_lp, pg_lp, windows, P, C):
    kls, d_nll, agree, ref_nll, pg_nll, first = [], [], 0, [], [], []
    for i, w in enumerate(windows):
        r, p = ref_lp[i], pg_lp[i]
        true = torch.tensor(w[P:P + C])
        kls += (r.exp() * (r - p)).sum(-1).tolist()                 # KL(ref || paged) per step
        rn = -r.gather(1, true[:, None])[:, 0]
        pn = -p.gather(1, true[:, None])[:, 0]
        ref_nll += rn.tolist()
        pg_nll += pn.tolist()
        d_nll += (pn - rn).tolist()
        agree += int((r.argmax(-1) == p.argmax(-1)).sum())
        first.append(float((p[0] - r[0]).abs().max()))           # the prefill step: no fp8 KV read yet
    n = len(kls)
    first_kl = kls[::C]                                              # step 0 of each window: chunked prefill only
    return {"steps": n, "prefill_step_mean_kl": sum(first_kl) / len(first_kl), "mean_kl": sum(kls) / n, "max_kl": max(kls), "argmax_agree": agree / n,
            "ref_mean_nll": sum(ref_nll) / n, "paged_mean_nll": sum(pg_nll) / n, "mean_d_nll": sum(d_nll) / n,
            "max_abs_d_nll": max(abs(x) for x in d_nll), "prefill_max_abs_logprob_diff": max(first)}


def measure(model, windows, *, prompt, cont, chunk, device, stand_in_attention=False, mutant=True):
    """The reference and the paged runner over ``windows`` on ``model`` (paged attention not yet registered);
    returns the comparison and the engagement record. A hybrid model also runs the MUTANT pass (``mutant``)."""
    import importlib.util

    from transformers.cache_utils import DynamicCache

    from experts4bit_qlora.engines import linear_state, paged_attention
    from experts4bit_qlora.serve_paged import _kv_geometry

    P, C = prompt, cont
    paged_attention.register(model)
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    n_layers = int(text_cfg.num_hidden_layers)

    # ---- the reference: transformers' forward, no paged context, one window at a time. Log-probs are kept in fp32:
    # bf16 rounding (~2e-3 at log-prob -0.5) would be the size of the fp8 effect this measures
    lin = list(linear_state.linear_layers(cfg))
    ref_lp, ref_states = [], []
    with torch.no_grad():
        for w in windows:
            cache = DynamicCache(config=text_cfg)
            out = model(input_ids=torch.tensor([w[:P]], device=device), past_key_values=cache, use_cache=True)
            rows = [out.logits[0, -1].float().log_softmax(-1)]
            for t in range(C - 1):
                out = model(input_ids=torch.tensor([[w[P + t]]], device=device), past_key_values=cache,
                            use_cache=True)
                rows.append(out.logits[0, -1].float().log_softmax(-1))
            ref_lp.append(torch.stack(rows).cpu())
            ref_states.append(_cache_states(cache, lin))
            del cache
    _sync(device)

    kw = dict(P=P, C=C, chunk=chunk, device=device, stand_in_attention=stand_in_attention)
    pg_lp, calls, pg_states, runner, step_ms = _paged_pass(model, windows, mutant=False, **kw)
    pool = runner.linear_state
    n_chunks = -(-P // chunk)
    pre = [li for li in lin if li < min(runner.attn_layers)] if lin and runner.attn_layers else list(lin)
    # transformers 5.17 takes each Gated DeltaNet kernel from fla / causal_conv1d (or the hub, through ``kernels``)
    # when importable, else its torch reference; which one ran is part of what was measured
    kmods = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
    hkv, hd = _kv_geometry(cfg)
    rec = {
        "model_type": getattr(text_cfg, "model_type", None),
        "windows": len(windows), "prompt": P, "cont": C, "chunk": chunk,
        "rehearsal": {"stand_in_attention": bool(stand_in_attention)},
        "engagement": {
            "n_layers": n_layers, "layer_types": list(linear_state.layer_types(cfg)),
            "attn_layers": list(runner.attn_layers), "kv_pool_layers": runner.kv.L,
            "layer_map": {str(k): v for k, v in runner.ctx.layer_map.items()},
            "linear_layers": lin, "linear_state": pool is not None,
            "linear_layers_with_state": sorted(pool.conv) if pool is not None else [],
            "linear_state_mb": round(pool.nbytes() / 2**20, 1) if pool is not None else 0.0,
            "kv_geometry": [hkv, hd], "gated_deltanet_kernel_modules": kmods,
            "decode_attention_calls": calls["attention"],
            "decode_attention_pool_layers": sorted(calls["attention_layers"]),
            "expected_decode_attention_calls": (C - 1) * len(runner.attn_layers),
            "linear_state_stores": calls["store"], "linear_state_store_layers": sorted(calls["store_layers"]),
            "expected_linear_state_stores": len(lin) * (len(windows) * n_chunks + C - 1),
        },
        **_compare(ref_lp, pg_lp, windows, P, C),
        "state": _state_errors(pg_states, ref_states, pre) if lin else None,
        "paged_decode_step_ms": round(step_ms, 2), "paged_decode_rows": len(windows),
        "mutant": None,
    }
    if mutant and pool is not None:
        del runner, pg_lp
        m_lp, m_calls, m_states, _m_runner, _ = _paged_pass(model, windows, mutant=True, **kw)
        rec["mutant"] = {"arm": "decode write-back rotated by one slot (row k's state into row k+1's slot)",
                         "linear_state_stores": m_calls["store"], "decode_attention_calls": m_calls["attention"],
                         **_compare(ref_lp, m_lp, windows, P, C),
                         "state": _state_errors(m_states, ref_states, pre)}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--offload", action="store_true", help="REHEARSAL: experts in host RAM")
    ap.add_argument("--stand-in-attention", action="store_true", help="REHEARSAL: SDPA over the pool's fp8 bytes")
    a = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer

    from experts4bit_qlora import load_moe_4bit_streaming

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
                                       revision=a.revision, offload=a.offload)
    model.eval()
    load_s = time.time() - t0
    windows = wikitext_windows(tok, a.windows, a.prompt, a.cont)
    rec = measure(model, windows, prompt=a.prompt, cont=a.cont, chunk=a.chunk, device="cuda",
                  stand_in_attention=a.stand_in_attention)
    rec["rehearsal"]["offload"] = a.offload
    rec = {"tag": a.tag, "model": a.model, "revision": a.revision,
           "loaded_commit": getattr(model.config, "_commit_hash", None), "transformers": transformers.__version__,
           "load_s": round(load_s, 1), **rec,
           "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    e = rec["engagement"]
    print(f"P97_MODEL {a.tag}: steps {rec['steps']} mean_kl {rec['mean_kl']:.3e} max_kl {rec['max_kl']:.3e} agree "
          f"{rec['argmax_agree']:.4f} d_nll {rec['mean_d_nll']:+.4f} | attn {len(e['attn_layers'])} kv "
          f"{e['kv_pool_layers']} kernel {e['decode_attention_calls']}/{e['expected_decode_attention_calls']} linear "
          f"{len(e['linear_layers'])} stores {e['linear_state_stores']}/{e['expected_linear_state_stores']} | decode "
          f"{rec['paged_decode_step_ms']:.1f} ms/step x{rec['paged_decode_rows']} | mem {rec['max_mem_gb']} GB")
    if rec["mutant"]:
        m = rec["mutant"]
        print(f"P97_MUTANT {a.tag}: mean_kl {m['mean_kl']:.3e} agree {m['argmax_agree']:.4f} d_nll {m['mean_d_nll']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
