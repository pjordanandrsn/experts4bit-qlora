"""N1 — routing-pinned serve (docs/N1_ROUTING_PINNED_SERVE.md).

Three phases, one adapter (nf4-trained), pinned n=1024 eval set, all resident:
  C0  nf4 base + adapter, standard routing        -> L0, and CAPTURE per-(example,layer)
                                                     (top_k_index, top_k_weights)
  C1  int8 base + adapter, standard routing        -> L1 (the unpinned upgrade / forfeit)
  C2  int8 base + adapter, routing REPLAYED from C0 -> L2 (the pinned upgrade)

Metric R = (L1 - L2) / G, G = 0.01657 (certified). Success R >= 0.50. Eval is teacher-forced
and bitwise-deterministic on this host, so one pass per phase is a measurement. The pin
substitutes ONLY the routed-expert selection (indices + gate weights) via an ExpertsLoRA
forward pre-hook — no weight/router edit — so the int8 experts + trained adapter run on the
home (nf4) routing decision.

Example:
  python scripts/routing_pinned_serve.py --job-dir runs/jobs/n1_routing_pinned \\
    --adapter /workspace/matrix/olmoe_mode_adapters/nf4/adapter_best.pt
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

G_CERT = 0.01657  # certified n=1024 G_int8 (docs/NULL_LADDER_1024_AMENDMENT.md)


def load_model(quant_type, adapter_path):
    import torch
    import experts4bit_qlora.train as train
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    model, _ = load_moe_4bit_streaming(train.MODEL, "cuda", torch.bfloat16, 8, 16,
                                       offload=False, pin=True, quant_type=quant_type)
    model.to("cuda")
    add_attention_lora(model, 8, 16, torch.bfloat16)
    sd = torch.load(adapter_path, map_location="cuda")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_loaded = len(sd) - len(unexpected)
    if n_loaded == 0 or unexpected:
        raise RuntimeError(f"adapter load mismatch: {n_loaded}/{len(sd)} loaded, {len(unexpected)} unexpected")
    model.eval()
    model.config.use_cache = False
    return model, n_loaded


def expert_lora_modules(model):
    from experts4bit_qlora.lora import ExpertsLoRA
    return [m for m in model.modules() if isinstance(m, ExpertsLoRA)]


def eval_pass(model, eval_data, mode, capture=None, replay=None):
    """mode: 'plain' | 'capture' | 'replay'. capture -> dict[(ex,layer)] = (idx_cpu, w_cpu);
    replay -> read that dict and override args. Returns (losses list, n_overrides)."""
    import torch

    layers = expert_lora_modules(model)
    state = {"ex": 0, "n_over": 0}
    handles = []
    for li, mod in enumerate(layers):
        def mk(idx):
            def hook(module, args):
                hs, tki, tkw = args[0], args[1], args[2]
                key = (state["ex"], idx)
                if mode == "capture":
                    capture[key] = (tki.detach().clone(), tkw.detach().clone())
                    return None
                if mode == "replay" and key in replay:
                    pin_idx, pin_w = replay[key]
                    if pin_idx.shape == tki.shape:  # teacher-forced: identical seq -> identical shape
                        state["n_over"] += 1
                        return (hs, pin_idx.to(tki.device), pin_w.to(tkw.device))
                return None
            return hook
        handles.append(mod.register_forward_pre_hook(mk(li)))

    losses = []
    with torch.no_grad():
        for i, ex in enumerate(eval_data):
            state["ex"] = i
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            lbl = torch.tensor([ex["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=lbl).loss.item()
            losses.append(loss)
    for h in handles:
        h.remove()
    return losses, state["n_over"]


def mean_skipnan(xs):
    v = [x for x in xs if x == x]
    return sum(v) / max(len(v), 1), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--adapter", required=True, help="nf4-trained adapter_best.pt")
    ap.add_argument("--train-mode", default="nf4")
    ap.add_argument("--serve-mode", default="int8")
    ap.add_argument("--n-train", type=int, default=10064)
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)
    os.environ["SEQ"] = str(args.seq)
    os.environ["N_TRAIN"] = str(args.n_train)

    import torch
    import experts4bit_qlora.train as train
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(train.MODEL)
    eval_data = train.encode_alpaca(tok, f"train[{args.n_train}:{args.n_train + args.n_eval}]")
    print(f"eval set n={len(eval_data)}", flush=True)

    capture = {}
    # C0: home (nf4 base + adapter), capture routing
    print("[C0] nf4 base + adapter — capture routing", flush=True)
    m0, nl0 = load_model(args.train_mode, args.adapter)
    L0, _ = eval_pass(m0, eval_data, "capture", capture=capture)
    import gc
    del m0
    gc.collect()
    torch.cuda.empty_cache()

    # C1: unpinned upgrade (int8 base + adapter)
    print("[C1] int8 base + adapter — unpinned", flush=True)
    m1, nl1 = load_model(args.serve_mode, args.adapter)
    L1, _ = eval_pass(m1, eval_data, "plain")
    del m1
    gc.collect()
    torch.cuda.empty_cache()

    # C2: pinned upgrade (int8 base + adapter, replay C0 routing)
    print("[C2] int8 base + adapter — routing pinned to C0", flush=True)
    m2, nl2 = load_model(args.serve_mode, args.adapter)
    L2, n_over = eval_pass(m2, eval_data, "replay", replay=capture)
    del m2
    gc.collect()
    torch.cuda.empty_cache()

    # paired means on the shared non-nan set
    idx_ok = [i for i in range(len(L0)) if L0[i] == L0[i] and L1[i] == L1[i] and L2[i] == L2[i]]
    m0m = sum(L0[i] for i in idx_ok) / len(idx_ok)
    m1m = sum(L1[i] for i in idx_ok) / len(idx_ok)
    m2m = sum(L2[i] for i in idx_ok) / len(idx_ok)
    R = (m1m - m2m) / G_CERT
    # paired SE of (L1 - L2) for the CI on R
    import math
    d = [L1[i] - L2[i] for i in idx_ok]
    dm = sum(d) / len(d)
    dsd = math.sqrt(sum((x - dm) ** 2 for x in d) / (len(d) - 1))
    dse = dsd / math.sqrt(len(d))
    R_se = dse / G_CERT  # dominant term; G's SE (0.00227) adds in quadrature on the ratio
    R_se_full = abs(R) * math.sqrt((dse / dm) ** 2 + (0.00227 / G_CERT) ** 2) if dm != 0 else R_se

    result = {
        "job_type": "routing_pinned_serve", "status": "pass",
        "adapter": args.adapter, "train_mode": args.train_mode, "serve_mode": args.serve_mode,
        "n_eval_used": len(idx_ok), "G_cert": G_CERT,
        "L0_home": m0m, "L1_unpinned": m1m, "L2_pinned": m2m,
        "forfeit_L1_minus_L0": m1m - m0m,
        "recovery_L1_minus_L2": m1m - m2m,
        "R_recovered_fraction_of_G": R,
        "R_se": R_se_full,
        "pin_overrides_applied": n_over, "pin_overrides_expected": len(capture),
        "verdict": ("R>=0.50 SUCCESS" if R >= 0.50 else ("0<R<0.50 PARTIAL" if R > 0 else "R<=0 NULL/HURTS")),
        "adapter_keys_loaded": [nl0, nl1, nl2],
        "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(0),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        import bitsandbytes
        result["bitsandbytes_version"] = bitsandbytes.__version__
    except Exception:
        result["bitsandbytes_version"] = None
    with open(os.path.join(args.job_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"L0(home) {m0m:.4f} | L1(unpinned) {m1m:.4f} | L2(pinned) {m2m:.4f}")
    print(f"forfeit (L1-L0) {m1m-m0m:+.4f} | recovery (L1-L2) {m1m-m2m:+.4f}")
    print(f"R = {R:.3f} ± {R_se_full:.3f} of G  -> {result['verdict']}")
    print(f"pin overrides applied {n_over}/{len(capture)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
