"""Per-example ∅ (no-adapter) eval — debts D1 + D2, and the n=1024 re-pin instrument.

Reproduces the trainer's BEFORE-training eval exactly (same loader, same LoRA-B=0 wrap, same
eval slice train[N_TRAIN : N_TRAIN+n_eval], same skip-nan mean), but logs EVERY example's loss
to result_rows.jsonl so mode/placement comparisons become example-PAIRED statistics (mean, sd,
SE = sd/sqrt(n)) instead of one aggregate number. One job per (storage mode, placement); a
repeat job of one mode measures eval determinism (the repeat-null for serve certificates).

n=1024 re-pin additions (docs/NULL_LADDER_1024_AMENDMENT.md — preregistered):
- ``eval_set_sha256`` self-attestation in result.json (and ``--hash-only`` to print the hash
  of a candidate eval set without touching the GPU) — the committed hash pins the set.
- REQUIRED telemetry: per-layer routed expert sets per example, captured as on-device
  bincounts over ``top_k_index``, written sparsely to routed_sets.jsonl.
- PREFERRED telemetry (``--router-telemetry``): per-layer router top-k boundary margins per
  example (mean/min/p10 of the k-th minus (k+1)-th router logit, plus top1−top2 mean) —
  fragility-index raw material.

All jobs of a comparison set MUST run on ONE pod: the T5(c) forensics
(docs/TRAIN_PLACEMENT_CERTIFICATE.md) showed the ∅ eval is deterministic per GPU architecture
but offset ~0.003-0.005 ACROSS architectures — cross-host per-example deltas would measure
the evaluator, not the storage mode.

Example:
    python scripts/eval_null_per_example.py --job-dir runs/jobs/null_olmoe_nf4_resident_perexample \\
        --quant-type nf4
    python scripts/eval_null_per_example.py --job-dir ... --quant-type int8 --offload \\
        --n-train 10064 --n-eval 1024 --router-telemetry
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def eval_set_hash(eval_data) -> str:
    """SHA-256 over the exact tokenized eval set (input_ids + labels, in order). This is what
    the amendment's committed hash pins; every job self-attests it in result.json."""
    h = hashlib.sha256()
    for ex in eval_data:
        h.update(json.dumps([list(ex["input_ids"]), list(ex["labels"])]).encode())
    return h.hexdigest()


class RoutingRecorder:
    """Per-example routing telemetry, accumulated on device, drained per example.

    Routed sets (always on): a forward pre-hook on every ExpertsLoRA bincounts that layer's
    ``top_k_index``. Router margins (optional): a forward hook on every router gate Linear
    summarizes the top-k boundary margin over the example's tokens.
    """

    def __init__(self, model, router_telemetry: bool):
        import torch
        import torch.nn as nn

        from experts4bit_qlora.lora import ExpertsLoRA

        self._torch = torch
        self.expert_bincounts = {}  # layer_idx -> device tensor [num_experts]
        self.margin_rows = {}  # layer_idx -> list of device tensors (per forward)
        self.handles = []
        self.n_layers = 0
        for li, mod in enumerate(m for m in model.modules() if isinstance(m, ExpertsLoRA)):
            self.n_layers += 1
            self.handles.append(mod.register_forward_pre_hook(self._make_expert_hook(li, mod.base.num_experts)))
        if router_telemetry:
            routers = [m for n, m in model.named_modules()
                       if n.endswith("mlp.gate") and isinstance(m, nn.Linear)]
            for li, mod in enumerate(routers):
                self.handles.append(mod.register_forward_hook(self._make_margin_hook(li)))

    def _make_expert_hook(self, layer_idx, num_experts):
        torch = self._torch

        def hook(module, args):
            top_k_index = args[1]
            bc = torch.bincount(top_k_index.reshape(-1), minlength=num_experts)
            prev = self.expert_bincounts.get(layer_idx)
            self.expert_bincounts[layer_idx] = bc if prev is None else prev + bc

        return hook

    def _make_margin_hook(self, layer_idx):
        def hook(module, args, output):
            vals = output.detach().float().sort(dim=-1, descending=True).values
            k = 8 if vals.shape[-1] > 8 else vals.shape[-1] - 1
            self.margin_rows.setdefault(layer_idx, []).append(
                (vals[..., k - 1] - vals[..., k]).reshape(-1))
            self.margin_rows.setdefault(-layer_idx - 1, []).append(
                (vals[..., 0] - vals[..., 1]).reshape(-1))

        return hook

    def drain(self):
        """Return (routed_sets, margins) for the example just evaluated and reset."""
        torch = self._torch
        routed = {}
        for li, bc in self.expert_bincounts.items():
            nz = bc.nonzero(as_tuple=False).reshape(-1)
            routed[str(li)] = [[int(e), int(bc[e])] for e in nz.tolist()]
        margins = {}
        for li, rows in self.margin_rows.items():
            m = torch.cat(rows)
            key = str(li) if li >= 0 else f"top1_{-li - 1}"
            q = torch.quantile(m, 0.10).item() if m.numel() > 1 else m.item()
            margins[key] = {"mean": round(m.mean().item(), 5), "min": round(m.min().item(), 5),
                            "p10": round(q, 5), "n_tokens": int(m.numel())}
        self.expert_bincounts = {}
        self.margin_rows = {}
        return routed, margins

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--quant-type", required=True)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--router-telemetry", action="store_true",
                    help="also record per-layer router top-k boundary margins per example")
    ap.add_argument("--adapter-path", default=None,
                    help="S-B lane: load a trained LoRA adapter (adapter_best.pt) before eval — "
                         "the job is then an ADAPTED eval, not a ∅ eval")
    ap.add_argument("--hash-only", action="store_true",
                    help="print the eval-set SHA-256 for these (--n-train, --n-eval, --seq) and exit")
    args = ap.parse_args()
    if not args.hash_only:
        os.makedirs(args.job_dir, exist_ok=True)

    # train.py reads these at import time — set them first.
    os.environ["SEQ"] = str(args.seq)
    os.environ["N_TRAIN"] = str(args.n_train)
    os.environ["QUANT_TYPE"] = args.quant_type
    os.environ["OFFLOAD_EXPERTS"] = "1" if args.offload else "0"

    import experts4bit_qlora.train as train
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(train.MODEL)
    eval_data = train.encode_alpaca(tok, f"train[{args.n_train}:{args.n_train + args.n_eval}]")
    set_hash = eval_set_hash(eval_data)
    if args.hash_only:
        print(json.dumps({"eval_slice": f"train[{args.n_train}:{args.n_train + args.n_eval}]",
                          "seq": args.seq, "n_post_filter": len(eval_data),
                          "eval_set_sha256": set_hash}))
        return 0

    import torch

    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(args.seed)
    model, _ = load_moe_4bit_streaming(
        train.MODEL, "cuda", torch.bfloat16, 8, 16,
        offload=args.offload, pin=True, quant_type=args.quant_type,
    )
    if not args.offload:
        model.to("cuda")
    add_attention_lora(model, 8, 16, torch.bfloat16)  # B=0: adapters contribute exactly zero

    adapter_report = None
    if args.adapter_path:
        sd = torch.load(args.adapter_path, map_location="cuda")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        if n_loaded == 0 or unexpected:
            raise RuntimeError(
                f"adapter load mismatch: {n_loaded}/{len(sd)} keys loaded, "
                f"{len(unexpected)} unexpected — wrong base structure for this adapter")
        adapter_report = {"adapter_path": args.adapter_path, "n_keys": len(sd), "n_loaded": n_loaded}
        print(f"loaded adapter: {n_loaded}/{len(sd)} keys from {args.adapter_path}")

    model.eval()
    model.config.use_cache = False
    recorder = RoutingRecorder(model, args.router_telemetry)
    rows, tot, n = [], 0.0, 0
    rows_path = os.path.join(args.job_dir, "result_rows.jsonl")
    routed_path = os.path.join(args.job_dir, "routed_sets.jsonl")
    with torch.no_grad(), open(rows_path, "w") as rf, open(routed_path, "w") as sf:
        for i, ex in enumerate(eval_data):
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            lbl = torch.tensor([ex["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=lbl).loss.item()
            routed, margins = recorder.drain()
            n_sup = sum(1 for t in ex["labels"] if t != -100)
            row = {"example_index": i, "loss": loss, "n_supervised_tokens": n_sup,
                   "n_tokens": len(ex["input_ids"]), "is_nan": loss != loss}
            rows.append(row)
            rf.write(json.dumps(row) + "\n")
            srow = {"example_index": i, "routed": routed}
            if margins:
                srow["router_margins"] = margins
            sf.write(json.dumps(srow) + "\n")
            if loss == loss:  # the trainer's skip-nan mean, reproduced exactly
                tot += loss
                n += 1
    recorder.remove()

    result = {
        "job_type": "null_eval",
        "status": "pass",
        "model": train.MODEL,
        "storage_mode": args.quant_type,
        "offload": args.offload,
        "seed": args.seed,
        "seq": args.seq,
        "n_train": args.n_train,
        "n_eval_examples": len(rows),
        "n_eval_used": n,
        "eval_loss_mean": tot / max(n, 1),
        "eval_slice": f"train[{args.n_train}:{args.n_train + args.n_eval}]",
        "eval_set_sha256": set_hash,
        "router_telemetry": args.router_telemetry,
        "adapter": adapter_report,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
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
    print(f"null eval {args.quant_type}{'-offload' if args.offload else ''}: "
          f"mean {result['eval_loss_mean']:.6f} over {n}/{len(rows)} examples "
          f"| peak {result['peak_gpu_gb']} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
