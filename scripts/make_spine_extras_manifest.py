"""Manifest for the spine extras (SPECULATIVE_LANES_PLAN.md §0): W_RMS + S-B adapted evals.

Runs AFTER the confirmatory n=1024 ladder drains (second-class by construction — same pod,
same architecture, never ahead of the ladder in the queue):

- wrms_olmoe: per-format W_RMS vs the bf16 reference (one trivial GPU job, S-A x-axis);
- sb1024_adapted_olmoe_{nf4,int8}_resident: the seed-0 portability adapters evaluated on their
  train-precision base over the pinned n=1024 set with full routing telemetry (S-B lane).

Usage:
    python scripts/make_spine_extras_manifest.py --out runs/job_manifest/spine_extras_jobs.jsonl
"""

import argparse
import json
import sys

from make_ladder1024_manifest import EVAL_SET_SHA256

ADAPTER_ROOT = "/workspace/matrix/olmoe_mode_adapters"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [{
        "job_id": "wrms_olmoe",
        "job_type": "wrms",
        "command": ["python", "-u", "scripts/wrms_per_format.py", "--job-dir", "{job_dir}",
                    "--model", "allenai/OLMoE-1B-7B-0924"],
        "params": {"model": "allenai/OLMoE-1B-7B-0924", "reference": "bf16"},
    }]
    for mode in ("nf4", "int8"):
        jobs.append({
            "job_id": f"sb1024_adapted_olmoe_{mode}_resident",
            "job_type": "null_eval",
            "command": ["python", "-u", "scripts/eval_null_per_example.py", "--job-dir", "{job_dir}",
                        "--quant-type", mode, "--n-train", "10064", "--n-eval", "1024",
                        "--router-telemetry",
                        "--adapter-path", f"{ADAPTER_ROOT}/{mode}/adapter_best.pt"],
            "params": {"model": "allenai/OLMoE-1B-7B-0924", "storage_mode": mode, "offload": False,
                       "seed": 0, "eval_slice": "train[10064:11088]",
                       "eval_set_sha256": EVAL_SET_SHA256,
                       "adapter": f"{ADAPTER_ROOT}/{mode}/adapter_best.pt (seed-0 portability adapter)"},
        })
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
