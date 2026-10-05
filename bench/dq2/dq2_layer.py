"""bench/dq2/dq2_layer.py -- lane DQ2, BOX side (bench/dq2/DQ2-PREREG.md).

Can one dense decoder layer's frozen NF4 weights be streamed from pinned host memory behind that layer's own training
compute, on a PCIe 5.0 x16 RTX 5090? DQ1 (bench/dq1) bounded it with the linears alone on a PCIe 4.0 host (C_MARGINAL).
DQ2 measures the real thing: one Qwen3-32B decoder layer as HF + PEFT + bitsandbytes QLoRA runs it, on the link the
question needs. No checkpoint: the layer is built from the model's config with random weights (timing does not depend on
weight values).

Subject: transformers `Qwen3DecoderLayer` (SDPA, causal, RoPE from `Qwen3RotaryEmbedding`), its seven projections
converted to bitsandbytes `Linear4bit` (nf4, blocksize 64, double-quant, bf16 compute), PEFT LoRA r 16 / alpha 32 /
dropout 0 on all seven (`inject_adapter_in_model`, PEFT's default adapter dtype), under non-reentrant gradient
checkpointing, micro-batch 1 x M tokens.

Per M, in the order layer(pos 1) -> streaming probe -> layer(pos 2):
  T_fwd   the checkpointed forward of the layer (what runs while the NEXT layer's weights would be copied)
  T_bwd   its backward: recompute + backward through the LoRA and frozen paths (what runs during the backward-phase copy)
  X       pinned->device copy of the layer's frozen bytes, measured while the layer's forward runs (copy under load),
          and the forward's slowdown while copies run (compute under load); each loaded stream covered by the other
Writes one JSON receipt. The verdict is dq2_reduce.py's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

from dq1_census import forensics, warm_until_steady   # staged beside this file (DQ1's instrument, Amendment 1 warm-up)

# Qwen/Qwen3-32B config.json (read from the model's published config; no weights are fetched)
QWEN3_32B = dict(hidden_size=5120, intermediate_size=25600, num_attention_heads=64, num_key_value_heads=8, head_dim=128,
                 rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=40960, attention_bias=False,
                 hidden_act="silu", num_hidden_layers=1, vocab_size=151936)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_layer(seed: int = 0):
    import bitsandbytes as bnb
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

    cfg = Qwen3Config(**QWEN3_32B)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    with torch.device("cuda"):
        layer = Qwen3DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16)
        rotary = Qwen3RotaryEmbedding(cfg)
    for parent in (layer.self_attn, layer.mlp):
        for name in TARGETS:
            lin = getattr(parent, name, None)
            if lin is None:
                continue
            q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16,
                                  compress_statistics=True, quant_type="nf4", device="cpu")
            q.weight = bnb.nn.Params4bit(lin.weight.data.detach().to("cpu"), requires_grad=False,
                                         compress_statistics=True, quant_type="nf4", blocksize=64)
            setattr(parent, name, q.to("cuda"))
            del lin
    torch.cuda.empty_cache()
    for p in layer.parameters():
        p.requires_grad_(False)
    layer = inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS), layer)
    lora = [p for n, p in layer.named_parameters() if "lora_" in n]
    for p in lora:
        p.requires_grad_(True)
    return layer, rotary, cfg, lora


def frozen_bytes(layer) -> tuple[int, str]:
    """(bytes, sha256) of the layer's frozen 4-bit storage as Linear4bit holds it: packed nibbles + quant-state tensors."""
    import bitsandbytes as bnb
    h, n = hashlib.sha256(), 0
    for _, m in sorted(layer.named_modules(), key=lambda kv: kv[0]):
        if isinstance(m, bnb.nn.Linear4bit):
            st = m.weight.quant_state
            parts = [m.weight.data, st.absmax, st.code]
            if getattr(st, "nested", False):
                parts += [st.state2.absmax, st.state2.code]
            for t in parts:
                b = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                h.update(b)
                n += len(b)
    return n, h.hexdigest()


def engagement(layer) -> dict:
    import bitsandbytes as bnb
    kinds = {}
    for name, m in layer.named_modules():
        base = name.rsplit(".", 1)[-1]
        if base in TARGETS:
            kinds[name] = type(m).__module__ + "." + type(m).__name__
    n4 = sum(isinstance(m, bnb.nn.Linear4bit) for m in layer.modules())
    lora = [(n, str(p.dtype), tuple(p.shape)) for n, p in layer.named_parameters() if "lora_" in n]
    return {"wrappers": kinds, "linear4bit_modules": n4, "lora_params": len(lora),
            "lora_dtypes": sorted({d for _, d, _ in lora}), "attn_impl": "sdpa"}


def layer_times(layer, rotary, lora, M: int, args) -> dict:
    """Checkpointed forward time and backward (recompute + backward) time of one layer at 1 x M tokens, per rep, with
    CUDA events; `draws` draws of `reps` reps; returns medians per draw plus integrity of the last rep."""
    from torch.utils.checkpoint import checkpoint

    g = torch.Generator(device="cuda").manual_seed(2000 + M)
    x = torch.randn(1, M, QWEN3_32B["hidden_size"], device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_(True)
    go = torch.randn(1, M, QWEN3_32B["hidden_size"], device="cuda", dtype=torch.bfloat16, generator=g)
    pos = torch.arange(M, device="cuda")[None]
    cos, sin = rotary(x, pos)

    def f(h):
        return layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))

    def one():
        y = checkpoint(f, x, use_reentrant=False)
        grads = torch.autograd.grad(y, [x] + lora, go)
        return y, grads

    y, grads = one()                                   # warm (and Triton / cuBLAS heuristics)
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    ev[0].record()
    one()
    ev[1].record()
    torch.cuda.synchronize()
    reps = max(1, min(200, int(args.target_ms / max(ev[0].elapsed_time(ev[1]), 1e-3))))
    fwd, bwd = [], []
    for _ in range(args.draws):
        fs, bs = [], []
        for _ in range(reps):
            ev[0].record()
            y = checkpoint(f, x, use_reentrant=False)
            ev[1].record()
            grads = torch.autograd.grad(y, [x] + lora, go)
            ev[2].record()
            torch.cuda.synchronize()
            fs.append(ev[0].elapsed_time(ev[1]))
            bs.append(ev[1].elapsed_time(ev[2]))
        fs.sort()
        bs.sort()
        fwd.append(fs[len(fs) // 2])
        bwd.append(bs[len(bs) // 2])
    dx, *dl = grads
    lora_b = [gd for gd, p in zip(dl, lora) if gd is not None]
    integ = {"finite": bool(torch.isfinite(y).all() and torch.isfinite(dx).all()
                            and all(torch.isfinite(gd).all() for gd in lora_b)),
             "dx_norm": float(dx.float().norm()),
             "lora_grads_nonzero": sum(int(gd.abs().sum().item() > 0) for gd in lora_b),
             "lora_grads": len(lora_b)}
    return {"reps": reps, "fwd_ms": fwd, "bwd_ms": bwd, "integrity": integ, "f": f, "x": x}


def stream_probe(layer_f, x, nbytes: int, args) -> dict:
    """Pinned->device copies of `nbytes` against the layer's checkpointed forward as the compute load; two fully-loaded
    readings per draw (DQ1's design): copy under load (forwards first, >= 2x the copy window) and forward under load
    (copies first, >= 2x the forward window), each with device-timeline coverage recorded."""
    from torch.utils.checkpoint import checkpoint

    src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    cs = torch.cuda.Stream()
    main = torch.cuda.current_stream()

    def fwd(k):
        for _ in range(k):
            checkpoint(layer_f, x, use_reentrant=False)

    def copies(k):
        with torch.cuda.stream(cs):
            for _ in range(k):
                dst.copy_(src, non_blocking=True)

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def window(first, second):
        a, b, c, d = ev(), ev(), ev(), ev()
        torch.cuda.synchronize()
        s1, f1 = first
        s2, f2 = second
        a.record(s1)
        f1()
        b.record(s1)
        c.record(s2)
        f2()
        d.record(s2)
        torch.cuda.synchronize()
        return a.elapsed_time(b), c.elapsed_time(d), (a.elapsed_time(c) >= 0 and d.elapsed_time(b) >= 0)

    out = {"bytes": nbytes, "draws": []}
    for _ in range(args.draws):
        a, b = ev(), ev()
        a.record(cs)
        copies(args.h2d_reps)
        b.record(cs)
        torch.cuda.synchronize()
        copy_alone = a.elapsed_time(b) / args.h2d_reps
        c, d = ev(), ev()
        c.record()
        fwd(args.h2d_reps)
        d.record()
        torch.cuda.synchronize()
        fwd_alone = max(c.elapsed_time(d) / args.h2d_reps, 1e-3)
        k_fwd = max(2, int(2 * copy_alone * args.h2d_reps / fwd_alone) + 1)
        _, copy_ms, copy_cov = window((main, lambda: fwd(k_fwd)), (cs, lambda: copies(args.h2d_reps)))
        k_cp = max(2, int(2 * fwd_alone * args.h2d_reps / copy_alone) + 1)
        _, fwd_ms, fwd_cov = window((cs, lambda: copies(k_cp)), (main, lambda: fwd(args.h2d_reps)))
        out["draws"].append({"copy_alone_ms": copy_alone, "copy_alone_gbs": nbytes / copy_alone / 1e6,
                             "copy_loaded_gbs": nbytes * args.h2d_reps / copy_ms / 1e6, "copy_covered": bool(copy_cov),
                             "fwd_alone_ms": fwd_alone, "fwd_loaded_ms": fwd_ms / args.h2d_reps,
                             "fwd_covered": bool(fwd_cov), "fwds_under_copy": k_fwd, "copies_under_fwd": k_cp})
    torch.cuda.empty_cache()
    return out


def link() -> dict:
    """The link as the driver reports it: gen/width max and current (current reads downclocked at idle)."""
    import subprocess
    q = "pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current"
    try:
        v = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=30).stdout.strip().split(",")
        return dict(zip(("gen_max", "width_max", "gen_current", "width_current"), (int(s) for s in v)))
    except Exception as exc:  # recorded; the reducer VOIDs a receipt without a link
        return {"error": str(exc)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default="512,1024,2048,4096,8192")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--target-ms", type=float, default=300.0)
    p.add_argument("--warm-s", type=float, default=4.0)
    p.add_argument("--warm-max-s", type=float, default=30.0)
    p.add_argument("--h2d-reps", type=int, default=4)
    p.add_argument("--rehearsal", action="store_true", help="correctness rehearsal: a receipt that can never be graded")
    args = p.parse_args()

    rows = [int(v) for v in args.rows.split(",") if v]
    receipt = {"schema": "dq2-layer/1", "rehearsal": bool(args.rehearsal),
               "started_at": time.strftime("%FT%TZ", time.gmtime()), "forensics": forensics(), "link": link(),
               "rows": rows, "config": {k: getattr(args, k) for k in ("draws", "target_ms", "warm_s", "warm_max_s",
                                                                      "h2d_reps")},
               "model_config": QWEN3_32B, "cells": []}

    def flush():
        with open(args.out + ".tmp", "w") as fh:
            json.dump(receipt, fh, indent=1)
        os.replace(args.out + ".tmp", args.out)

    layer, rotary, cfg, lora = build_layer()
    nbytes, sha0 = frozen_bytes(layer)
    receipt.update({"layer_frozen_bytes": nbytes, "frozen_sha256_before": sha0, "engagement": engagement(layer)})
    flush()
    for M in rows:
        t0 = time.time()
        cell = {"M": M, "warm": warm_until_steady(args.warm_s, args.warm_max_s)}
        torch.cuda.reset_peak_memory_stats()
        try:
            a = layer_times(layer, rotary, lora, M, args)
            cell["pos1"] = {k: a[k] for k in ("reps", "fwd_ms", "bwd_ms", "integrity")}
            cell["stream"] = stream_probe(a["f"], a["x"], nbytes, args)
            b = layer_times(layer, rotary, lora, M, args)
            cell["pos2"] = {k: b[k] for k in ("reps", "fwd_ms", "bwd_ms", "integrity")}
            del a, b
        except Exception as exc:  # recorded per row; the reducer VOIDs a graded row that errored
            cell["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        cell["mem_peak_bytes"] = torch.cuda.max_memory_allocated()
        cell["warm"]["end_tflops"] = warm_until_steady(0.0, 0.0)["first_tflops"]
        receipt["cells"].append(cell)
        torch.cuda.empty_cache()
        print(f"row M={M} {time.time() - t0:.1f}s {cell.get('error', '')}", flush=True)
        flush()
    _, sha1 = frozen_bytes(layer)
    receipt["frozen_sha256_after"] = sha1
    receipt["finished_at"] = time.strftime("%FT%TZ", time.gmtime())
    flush()
    print("DQ2 layer census done", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
