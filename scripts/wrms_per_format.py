"""W_RMS per storage format — the response curve's x-axis (SPECULATIVE_LANES_PLAN.md S-A).

For each format p in {nf4, fp4, int8, fp8, fp16}: quantize the bf16 reference expert stacks
per expert (exactly as ExpertsNbit._quantize_stack does — same blocksize, same codes),
dequantize back to bf16, and accumulate the relative RMS error
``sqrt(sum||W_p − W||² / sum||W||²)`` over every expert projection in the model. fp16 is the
near-zero point (bf16→fp16 cast is exact in fp16's normal range; only tiny weights round).
bf16 is the reference (W_RMS ≡ 0) and is not a row.

One trivial GPU job (bnb kernels are CUDA-only), no eval passes. Writes result.json with
per-format totals and per-projection-kind breakdowns.

Usage:
    python scripts/wrms_per_format.py --job-dir runs/jobs/wrms_olmoe --model allenai/OLMoE-1B-7B-0924
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

FORMATS = ("nf4", "fp4", "int8", "fp8", "fp16")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--blocksize", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)

    os.environ.setdefault("QUANT_TYPE", "bf16")
    import torch
    import bitsandbytes.functional as F_bnb

    from experts4bit_qlora._vendor.experts import _build_code
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora._vendor.experts import ExpertsNbit

    # bf16 passthrough load = the exact reference weights, streamed (never a full-model copy).
    model, _ = load_moe_4bit_streaming(args.model, "cuda", torch.bfloat16, 8, 16,
                                       offload=False, pin=True, quant_type="bf16")
    model.to("cuda")

    codes = {"int8": _build_code("int8", "cuda"), "fp8": _build_code("fp8", "cuda")}
    acc = {p: {"err_sq": 0.0, "ref_sq": 0.0,
               "by_proj": {"gate_up": [0.0, 0.0], "down": [0.0, 0.0]}} for p in FORMATS}
    n_experts_seen = 0

    with torch.no_grad():
        for mod in model.modules():
            if not isinstance(mod, ExpertsNbit) or getattr(mod, "bits", 0) != 16:
                continue
            for proj_name, shape in (("gate_up", mod._gate_up_shape), ("down", mod._down_shape)):
                packed = getattr(mod, f"{proj_name}_proj")
                for e in range(mod.num_experts):
                    w = packed[e].reshape(shape).to(torch.bfloat16)
                    ref_sq = w.float().pow(2).sum().item()
                    for p in FORMATS:
                        if p == "fp16":
                            wq = w.to(torch.float16).to(torch.bfloat16)
                        elif p in ("nf4", "fp4"):
                            q, st = F_bnb.quantize_4bit(w.contiguous(), blocksize=args.blocksize,
                                                        compress_statistics=False, quant_type=p)
                            wq = F_bnb.dequantize_4bit(q, quant_state=st).to(torch.bfloat16)
                        else:
                            q, st = F_bnb.quantize_blockwise(w.contiguous(), code=codes[p],
                                                             blocksize=args.blocksize)
                            wq = F_bnb.dequantize_blockwise(q, quant_state=st).to(torch.bfloat16)
                        err_sq = (wq.float() - w.float()).pow(2).sum().item()
                        acc[p]["err_sq"] += err_sq
                        acc[p]["ref_sq"] += ref_sq
                        acc[p]["by_proj"][proj_name][0] += err_sq
                        acc[p]["by_proj"][proj_name][1] += ref_sq
                n_experts_seen += mod.num_experts

    result = {
        "job_type": "wrms",
        "status": "pass",
        "model": args.model,
        "blocksize": args.blocksize,
        "reference": "bf16 passthrough (streamed)",
        "n_expert_projections": n_experts_seen,
        "w_rms": {p: math.sqrt(a["err_sq"] / a["ref_sq"]) for p, a in acc.items()},
        "w_rms_by_proj": {p: {k: math.sqrt(v[0] / v[1]) for k, v in a["by_proj"].items()}
                          for p, a in acc.items()},
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
    for p in FORMATS:
        print(f"W_RMS({p}) = {result['w_rms'][p]:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
