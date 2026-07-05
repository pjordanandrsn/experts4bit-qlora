"""Decode-repeat payload: N timed greedy-decode samples under one storage mode, one load.

Repeat-validation leg for single-run decode observations (e.g. the fp4-faster-than-nf4 single
sample): loads the base once via ``experts4bit_qlora.infer.load_for_inference`` (env-driven, same
path the manual decode measurement uses), runs one discarded warmup generation, then N measured
samples, and writes ``result.json`` with mean/std/min/max tok/s + peak GPU. Designed to run under
``runpod_claim_and_run.py`` (writes only into ``--job-dir``), but runs standalone too.

Example:
    QUANT_TYPE=fp4 python scripts/decode_repeat.py --samples 5 --tokens 128 \\
        --job-dir runs/jobs/decode_olmoe_fp4_resident_repeat5
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)

    import torch  # noqa: E402

    from experts4bit_qlora import expert_profile, infer  # noqa: E402  (env-driven)

    torch.manual_seed(0)
    tok, model = infer.load_for_inference()
    # Attach the decode-phase profiler if E4B_EXPERT_PROFILE is set. infer.main() does this, but
    # this script bypasses main() — without the attach the env var is set yet nothing hooks the
    # model, so the profile writes zero rows (the empty decode profiles in the first bundle).
    expert_profile.attach(model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    prompt = infer.PROMPT
    prompt_tokens = tok(prompt, return_tensors="pt").input_ids.shape[1]
    _ = infer.timed_decode(model, tok, prompt, args.tokens)  # warmup: probes, allocator, discarded
    tps = []
    for i in range(args.samples):
        _, _, s = infer.timed_decode(model, tok, prompt, args.tokens)
        tps.append(s)
        print(f"sample {i + 1}/{args.samples}: {s:.2f} tok/s", flush=True)

    result = {
        "job_type": "decode",
        "status": "pass",
        "model": infer.MODEL,
        "storage_mode": infer.QUANT_TYPE,
        "offload": infer.OFFLOAD_EXPERTS,
        "warmup_discarded": 1,
        "samples": args.samples,
        "tok_s_all": [round(s, 3) for s in tps],
        "tok_s_mean": round(statistics.mean(tps), 3),
        "tok_s_std": round(statistics.stdev(tps), 3) if len(tps) > 1 else 0.0,
        "tok_s_min": round(min(tps), 3),
        "tok_s_max": round(max(tps), 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": args.tokens,
        "decode_settings": {
            "greedy": True,
            "gemv": os.environ.get("E4B_INFER_GEMV", "1") != "0",
            "fastpath": os.environ.get("E4B_DECODE_FASTPATH", "1") != "0",
        },
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
    print(f"decode repeat: mean {result['tok_s_mean']} ± {result['tok_s_std']} tok/s "
          f"(min {result['tok_s_min']}, max {result['tok_s_max']}) peak {result['peak_gpu_gb']} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
