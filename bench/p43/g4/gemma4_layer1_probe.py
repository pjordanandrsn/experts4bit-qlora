#!/usr/bin/env python3
"""Adjudicate #558 at LAYER 1, over many tokens, real positions only, with an EARLY-EXIT hook.

Why layer 1 is the only clean place: layer 0 (the embedding output) was bit-identical across all three arms, so
layer 1's INPUT is identical and its output difference is exactly ONE decoder block's computation, with no
accumulated quantisation error. The 136-token whole-stack run did not adjudicate because the per-layer count was
decided by the middle stack, where +1.07 nats of quantisation error swamps the 0.072 the paths differ by.

Three things this fixes from the previous attempt:
  1. EARLY EXIT. A forward hook on decoder layer 0 captures its input and output and then aborts the pass, so the
     other 29 layers never run. The previous version asked for output_hidden_states=True, which materialises all
     31 tensors before anything is selected -- expensive on a 32 GB card, and ruinous for a bf16 oracle whose
     weights are mostly CPU-offloaded (one forward streamed the whole model across the bus).
  2. REAL POSITIONS ONLY. Padding carries no meaning; it is masked out rather than diffed.
  3. LOAD TIMING is printed, because the last run lost 90 minutes inside a model load on a slow host and the
     receipt could not show where the time went.

No loss is reported: loss needs the full stack, and tp4-g4oracle already has it.
"""
import argparse, gc, hashlib, json, os, time
import torch
import torch.nn as nn


class _StopForward(Exception):
    pass


def decoder_layers(m):
    """The decoder ModuleList, without importing any model class."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.model.layers", "model.text_model.layers"):
        cur, ok = m, True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, nn.ModuleList) and len(cur) > 0:
            return path, cur
    best = None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate the decoder layer list")
    return best


def capture(model, ids, mask, dev):
    """(input, output) of decoder layer 0 -- i.e. hidden_states[0] and hidden_states[1] -- then abort."""
    _, layers = decoder_layers(model)
    got = {}

    def pre(mod, args, kwargs=None):
        h = args[0] if args else None
        if h is not None:
            got["l0"] = h.detach()
        return None

    def post(mod, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        got["l1"] = h.detach()
        raise _StopForward()

    h1 = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    h2 = layers[0].register_forward_hook(post)
    try:
        with torch.no_grad():
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
    except _StopForward:
        pass
    finally:
        h1.remove()
        h2.remove()
    if "l1" not in got:
        raise RuntimeError("layer-0 forward hook never fired")
    return got.get("l0"), got["l1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--revision", default="4d7ae4984b7db7de8f8457170b3f1a419ee76d52")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--attn-4bit", type=int, default=1)
    ap.add_argument("--out", default="gemma4_layer1_probe.json")
    a = ap.parse_args()
    rep = {"model": a.model, "revision": a.revision, "rows": a.rows, "chunk": a.chunk}

    tok = json.load(open(a.tokens))
    rows = tok["train"][: a.rows]
    pad_id = tok.get("pad_id", 0)
    chunks, h = [], hashlib.sha256()
    for i in range(0, len(rows), a.chunk):
        grp = rows[i: i + a.chunk]
        L = max(len(r) for r in grp)
        ids = torch.full((len(grp), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(grp), L), dtype=torch.long)
        for j, r in enumerate(grp):
            ids[j, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[j, : len(r)] = 1
        h.update(ids.numpy().tobytes())
        chunks.append((ids, mask))
    rep.update(n_chunks=len(chunks), ids_sha256=h.hexdigest(),
               real_tokens=int(sum(int(m.sum()) for _, m in chunks)))
    print("chunks", rep["n_chunks"], "real tokens", rep["real_tokens"],
          "ids sha", rep["ids_sha256"][:16], flush=True)

    def sweep(model, dev, tag):
        l0s, l1s = [], []
        t0 = time.time()
        for k, (ids, mask) in enumerate(chunks):
            l0, l1 = capture(model, ids, mask, dev)
            sel = mask.to(l1.device).bool().reshape(-1)
            l1s.append(l1.reshape(-1, l1.shape[-1])[sel].float().cpu())
            if l0 is not None:
                l0s.append(l0.reshape(-1, l0.shape[-1])[sel].float().cpu())
            if str(dev) != "cpu":
                torch.cuda.empty_cache()
            print(f"  {tag} chunk {k+1}/{len(chunks)} ({time.time()-t0:.1f}s)", flush=True)
        return (torch.cat(l0s) if l0s else None), torch.cat(l1s)

    from experts4bit_qlora import (disable_batched_train, disable_fast_train, enable_fast_train,
                                   load_moe_4bit_streaming, verify_moe_4bit)
    from experts4bit_qlora.lora import (add_attention_lora, detect_attention_projections,
                                        quantize_attention_projections_4bit)
    print("loading 4-bit ...", flush=True)
    t0 = time.time()
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha,
                                         offload=False, pin=True, prefetch=False, quant_type="nf4")
    model.to("cuda")
    rep["load_4bit_s"] = round(time.time() - t0, 1)
    print("4-bit load took", rep["load_4bit_s"], "s", flush=True)
    v = verify_moe_4bit(model, strict=True)
    rep["verify"] = {"n_quantized": v.get("n_quantized"), "n_unquantized": v.get("n_unquantized")}
    if a.attn_4bit:
        census = detect_attention_projections(model)
        rep["structural_expected_n_attn4"] = getattr(census, "expected_count", None)
        rep["n_attn4"] = quantize_attention_projections_4bit(model)
    model.config.use_cache = False
    add_attention_lora(model, a.r, a.alpha, torch.float32)
    model.eval()
    rep["layer_path"] = decoder_layers(model)[0]

    disable_fast_train(model); disable_batched_train(model)
    ref_l0, ref_l1 = sweep(model, "cuda", "reference")
    rep["n_patched"] = enable_fast_train(model, dgrad=True)
    fus_l0, fus_l1 = sweep(model, "cuda", "fused")
    rep["peak_vram_4bit_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    del model
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print("4-bit freed, peak", rep["peak_vram_4bit_gb"], "GB", flush=True)

    from transformers import AutoModelForCausalLM
    kw = dict(revision=a.revision, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    t0 = time.time()
    try:
        oracle = AutoModelForCausalLM.from_pretrained(a.model, experts_implementation="eager", **kw)
        rep["oracle_experts_implementation"] = "eager"
    except TypeError:
        oracle = AutoModelForCausalLM.from_pretrained(a.model, **kw)
        rep["oracle_experts_implementation"] = "default"
    rep["load_oracle_s"] = round(time.time() - t0, 1)
    print("oracle load took", rep["load_oracle_s"], "s", flush=True)
    oracle.config.use_cache = False
    oracle.eval()
    try:
        in_dev = oracle.get_input_embeddings().weight.device
    except Exception:
        in_dev = next(oracle.parameters()).device
    rep["oracle_input_device"] = str(in_dev)
    orc_l0, orc_l1 = sweep(oracle, in_dev, "oracle")

    def dist(x, y):
        d = x - y
        return {"rms": float(d.pow(2).mean().sqrt()), "max_abs": float(d.abs().max()),
                "mean_abs": float(d.abs().mean())}

    rep["compared_positions"] = int(ref_l1.shape[0])
    if ref_l0 is not None and orc_l0 is not None:
        rep["layer0_sanity"] = {"reference_vs_oracle": dist(ref_l0, orc_l0),
                                "fused_vs_reference": dist(fus_l0, ref_l0)}
    rep["layer1"] = {"reference_vs_oracle": dist(ref_l1, orc_l1),
                     "fused_vs_oracle": dist(fus_l1, orc_l1),
                     "fused_vs_reference": dist(fus_l1, ref_l1)}
    rr = rep["layer1"]["reference_vs_oracle"]["rms"]
    ff = rep["layer1"]["fused_vs_oracle"]["rms"]
    rep["layer1_verdict"] = ("reference closer to the oracle" if rr < ff else
                             "fused closer to the oracle" if ff < rr else "exact tie")
    rep["layer1_rms_ratio_fused_over_reference"] = (ff / rr) if rr > 0 else None

    print("\n=== layer 0 sanity (must be ~0: identical input to layer 1) ===")
    for k, v2 in (rep.get("layer0_sanity") or {}).items():
        print(f"  {k:22s} rms {v2['rms']:.3e}  max {v2['max_abs']:.3e}")
    print("\n=== LAYER 1 over %d real positions (%d tokens swept) ===" % (rep["compared_positions"], rep["real_tokens"]))
    for k, v2 in rep["layer1"].items():
        print(f"  {k:22s} rms {v2['rms']:.6f}  max {v2['max_abs']:.4f}  mean_abs {v2['mean_abs']:.6f}")
    print("\nVERDICT:", rep["layer1_verdict"], "| fused/reference rms ratio",
          rep["layer1_rms_ratio_fused_over_reference"])
    json.dump(rep, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
