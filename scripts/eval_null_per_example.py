"""Per-example ∅ (no-adapter) eval — debts D1 + D2 (queue item T2/T3).

Reproduces the trainer's BEFORE-training eval exactly (same loader, same LoRA-B=0 wrap, same
eval slice train[N_TRAIN : N_TRAIN+64], same skip-nan mean), but logs EVERY example's loss to
result_rows.jsonl so mode/placement comparisons become example-PAIRED statistics (mean, sd,
SE = sd/sqrt(n)) instead of one aggregate number. One job per (storage mode, placement); a
repeat job of one mode measures eval determinism (the repeat-null for serve certificates).

All jobs of a comparison set MUST run on ONE pod: the T5(c) forensics
(docs/TRAIN_PLACEMENT_CERTIFICATE.md) showed the ∅ eval is deterministic per GPU architecture
but offset ~0.003-0.005 ACROSS architectures — cross-host per-example deltas would measure
the evaluator, not the storage mode.

Example:
    python scripts/eval_null_per_example.py --job-dir runs/jobs/null_olmoe_nf4_resident_perexample \\
        --quant-type nf4
    python scripts/eval_null_per_example.py --job-dir ... --quant-type int8 --offload
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--quant-type", required=True)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)

    # train.py reads these at import time — set them first.
    os.environ["SEQ"] = str(args.seq)
    os.environ["N_TRAIN"] = str(args.n_train)
    os.environ["QUANT_TYPE"] = args.quant_type
    os.environ["OFFLOAD_EXPERTS"] = "1" if args.offload else "0"

    import torch

    import experts4bit_qlora.train as train
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(train.MODEL)
    model, _ = load_moe_4bit_streaming(
        train.MODEL, "cuda", torch.bfloat16, 8, 16,
        offload=args.offload, pin=True, quant_type=args.quant_type,
    )
    if not args.offload:
        model.to("cuda")
    add_attention_lora(model, 8, 16, torch.bfloat16)  # B=0: adapters contribute exactly zero

    eval_data = train.encode_alpaca(tok, f"train[{args.n_train}:{args.n_train + args.n_eval}]")

    model.eval()
    model.config.use_cache = False
    rows, tot, n = [], 0.0, 0
    rows_path = os.path.join(args.job_dir, "result_rows.jsonl")
    with torch.no_grad(), open(rows_path, "w") as rf:
        for i, ex in enumerate(eval_data):
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            lbl = torch.tensor([ex["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=lbl).loss.item()
            n_sup = sum(1 for t in ex["labels"] if t != -100)
            row = {"example_index": i, "loss": loss, "n_supervised_tokens": n_sup,
                   "n_tokens": len(ex["input_ids"]), "is_nan": loss != loss}
            rows.append(row)
            rf.write(json.dumps(row) + "\n")
            if loss == loss:  # the trainer's skip-nan mean, reproduced exactly
                tot += loss
                n += 1

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
