#!/usr/bin/env python3
"""P43 T2b (amendment 1): adjudicate e4b#558 PER LAYER, three-way, with the bf16 oracle RESIDENT.

T2's layer-1 probe (`gemma4_layer1_probe.py`, receipt p43-t2-g4layer1) found the fused and reference paths EQUALLY far
from the oracle at decoder layer 1 (rms ratio 1.0004; layer-0 sanity exactly zero), so the 0.09-nat step-0 divergence
does not originate in layer 1's expert block, and the "first diverging layer = 1" reading from the CPU-offloaded
whole-stack run was an artifact of that run. This probe captures EVERY decoder layer's output (real positions only)
for the three arms on the same chunks and reports, per layer: rms(fused - reference), rms(reference - oracle),
rms(fused - oracle), the fused/reference-to-oracle ratio, and the first layer at which fused and reference differ by
more than a registered noise floor. Also the final logits (last decoder layer -> norm -> lm_head is the model's own
forward, so the last hidden state stands in for it here) so the per-layer curve can be tied to the loss delta.

Memory: no early exit; forward hooks on every decoder layer store the masked real-position rows in fp32 on CPU per
chunk (8 rows x <= 2048 tokens x 2816 hidden x 31 layers ~ 560 MB per chunk at fp32); the oracle is bf16 resident.
"""
import argparse
import gc
import hashlib
import json
import time
import torch


def decoder_layers(m, expect=None):
    """The TEXT decoder ModuleList, by explicit path first (the layer-1 probe's list), never "the first
    ModuleList named *.layers": on the bf16 oracle that walk returned the vision tower's 27 encoder layers
    (run p43-t2b-g4sweep, IndexError 27 -- the e4b side had walked the 30-layer text decoder), so the three
    arms were not about to be compared on the same module. ``expect`` pins the length to the e4b side's."""
    found = []
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.model.layers", "model.text_model.layers"):
        cur, ok = m, True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, torch.nn.ModuleList) and len(cur) > 0:
            found.append((path, cur))
    if not found:
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and name.endswith("layers"):
                found.append((name, mod))
    if expect is not None:
        hit = [f for f in found if len(f[1]) == expect]
        if not hit:
            raise RuntimeError(f"no decoder ModuleList of {expect} layers on this side; candidates: "
                               + ", ".join(f"{n}[{len(l)}]" for n, l in found))
        found = hit
    if not found:
        raise RuntimeError("could not locate the decoder layer list")
    return found[0]


def sweep(model, dev, tag, chunks, n_layers):
    """Per layer: the concatenated real-position outputs (fp32, CPU). Returns list[n_layers] of tensors [P, H]."""
    name, layers = decoder_layers(model, expect=n_layers)
    print(f"  {tag}: decoder layers at {name} [{len(layers)}]", flush=True)
    store = {}
    hooks = []
    def mk(i):
        def post(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            store[i] = h.detach()
        return post
    for i in range(n_layers):
        hooks.append(layers[i].register_forward_hook(mk(i)))
    outs = [[] for _ in range(n_layers)]
    t0 = time.time()
    with torch.no_grad():
        for k, (ids, mask) in enumerate(chunks):
            store.clear()
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
            sel = mask.to(store[0].device).bool().reshape(-1)
            for i in range(n_layers):
                h = store[i]
                outs[i].append(h.reshape(-1, h.shape[-1])[sel].float().cpu())
            if str(dev) != "cpu":
                torch.cuda.empty_cache()
            print(f"  {tag} chunk {k+1}/{len(chunks)} ({time.time()-t0:.1f}s)", flush=True)
    for h in hooks:
        h.remove()
    return [torch.cat(o) for o in outs]


def dist(x, y):
    d = x - y
    return {"rms": float(d.pow(2).mean().sqrt()), "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean())}


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
    ap.add_argument("--noise-floor", type=float, default=1e-4,
                    help="rms(fused - reference) above which a layer counts as diverging; the JSON also reports the first "
                         "layer over 1e-3, 1e-2 and 5e-2 (P43 amendment 1: layer 1 measured 4.4e-3, so the larger two "
                         "are the amplification onsets)")
    ap.add_argument("--out", default="gemma4_layer_sweep.json")
    a = ap.parse_args()
    rep = {"model": a.model, "revision": a.revision, "rows": a.rows, "chunk": a.chunk, "noise_floor": a.noise_floor}

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
    rep.update(n_chunks=len(chunks), ids_sha256=h.hexdigest(), real_tokens=int(sum(int(m.sum()) for _, m in chunks)))
    print("chunks", rep["n_chunks"], "real tokens", rep["real_tokens"], "ids sha", rep["ids_sha256"][:16], flush=True)

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
    v = verify_moe_4bit(model, strict=True)
    rep["verify"] = {"n_quantized": v.get("n_quantized"), "n_unquantized": v.get("n_unquantized")}
    if a.attn_4bit:
        census = detect_attention_projections(model)
        rep["structural_expected_n_attn4"] = getattr(census, "expected_count", None)
        rep["n_attn4"] = quantize_attention_projections_4bit(model)
    model.config.use_cache = False
    add_attention_lora(model, a.r, a.alpha, torch.float32)
    model.eval()
    lname, layers = decoder_layers(model)
    n_layers = len(layers)
    rep["layer_path"], rep["n_layers"] = lname, n_layers

    disable_fast_train(model); disable_batched_train(model)
    ref = sweep(model, "cuda", "reference", chunks, n_layers)
    rep["n_patched"] = enable_fast_train(model, dgrad=True)
    fus = sweep(model, "cuda", "fused", chunks, n_layers)
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
    oracle.config.use_cache = False
    oracle.eval()
    try:
        in_dev = oracle.get_input_embeddings().weight.device
    except Exception:
        in_dev = next(oracle.parameters()).device
    rep["oracle_input_device"] = str(in_dev)
    rep["oracle_layer_path"] = decoder_layers(oracle, expect=n_layers)[0]
    orc = sweep(oracle, in_dev, "oracle", chunks, n_layers)

    # the three arms must have captured the SAME module on the same positions before any distance is read
    for i in range(n_layers):
        if not (ref[i].shape == fus[i].shape == orc[i].shape):
            raise RuntimeError(f"layer {i}: captured shapes differ -- reference {tuple(ref[i].shape)} fused "
                               f"{tuple(fus[i].shape)} oracle {tuple(orc[i].shape)}; the arms walked different modules")
    rep["compared_positions"] = int(ref[0].shape[0])
    per = []
    first_div = None
    for i in range(n_layers):
        fr = dist(fus[i], ref[i]); ro = dist(ref[i], orc[i]); fo = dist(fus[i], orc[i])
        ratio = (fo["rms"] / ro["rms"]) if ro["rms"] > 0 else None
        per.append({"layer": i, "fused_vs_reference": fr, "reference_vs_oracle": ro, "fused_vs_oracle": fo,
                    "ratio_fused_over_reference_to_oracle": ratio,
                    "closer": ("reference" if ratio and ratio > 1 else "fused" if ratio and ratio < 1 else "tie")})
        if first_div is None and fr["rms"] > a.noise_floor:
            first_div = i
    rep["per_layer"] = per
    rep["first_diverging_layer"] = first_div
    rep["first_layer_over"] = {str(t): next((p["layer"] for p in per if p["fused_vs_reference"]["rms"] > t), None)
                               for t in (1e-3, 1e-2, 5e-2)}
    rep["ratio_band_0.9_1.1_all_layers"] = all(p["ratio_fused_over_reference_to_oracle"] is not None and
                                               0.9 <= p["ratio_fused_over_reference_to_oracle"] <= 1.1 for p in per)
    rep["last_layer_fused_vs_reference_rms"] = per[-1]["fused_vs_reference"]["rms"]
    rep["layers_reference_closer"] = sum(1 for p in per if p["closer"] == "reference")
    rep["layers_fused_closer"] = sum(1 for p in per if p["closer"] == "fused")
    print("\n=== per layer over %d real positions ===" % rep["compared_positions"])
    print(" layer  rms(fus-ref)  rms(ref-orc)  rms(fus-orc)  ratio")
    for p in per:
        print(f" {p['layer']:5d}  {p['fused_vs_reference']['rms']:.6f}     {p['reference_vs_oracle']['rms']:.6f}     "
              f"{p['fused_vs_oracle']['rms']:.6f}     {p['ratio_fused_over_reference_to_oracle']}")
    print("\nFIRST DIVERGING LAYER (rms(fused-reference) > %g): %s" % (a.noise_floor, first_div))
    print("first layer over 1e-3 / 1e-2 / 5e-2:", rep["first_layer_over"], "| ratio within [0.9, 1.1] on every layer:",
          rep["ratio_band_0.9_1.1_all_layers"], "| last-layer rms(fused-reference):", rep["last_layer_fused_vs_reference_rms"])
    print("reference closer on %d layers, fused closer on %d" % (rep["layers_reference_closer"], rep["layers_fused_closer"]))
    json.dump(rep, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
