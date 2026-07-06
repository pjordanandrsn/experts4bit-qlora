"""Manifest for the preregistered n=1024 ∅-ladder re-pin (docs/NULL_LADDER_1024_AMENDMENT.md).

13 passes: {nf4, fp4, int8, fp8, bf16, fp16} × {resident, offload} + one int8-resident
determinism repeat. Eval set train[10064:11088] (disjoint from the pilot), committed SHA-256
in the amendment; every job self-attests the hash in result.json. Router-margin telemetry ON.
All passes on ONE pod (A5000 — pilot/grid architecture).

Usage:
    python scripts/make_ladder1024_manifest.py --out runs/job_manifest/null_ladder_1024_jobs.jsonl
"""

import argparse
import json
import sys

MODES = ("nf4", "fp4", "int8", "fp8", "bf16", "fp16")
EVAL_SET_SHA256 = "3e836c1a01ab5cce90b7034477f174f5058f4cd4c1690dcc25b01741dc1a851f"


def job(mode, offload, rep=None):
    suffix = "offload" if offload else "resident"
    job_id = f"null1024_olmoe_{mode}_{suffix}" + (f"_rep{rep}" if rep else "")
    cmd = ["python", "-u", "scripts/eval_null_per_example.py", "--job-dir", "{job_dir}",
           "--quant-type", mode, "--n-train", "10064", "--n-eval", "1024", "--router-telemetry"]
    if offload:
        cmd.append("--offload")
    return {
        "job_id": job_id,
        "job_type": "null_eval",
        "command": cmd,
        "params": {"model": "allenai/OLMoE-1B-7B-0924", "storage_mode": mode, "offload": offload,
                   "seed": 0, "eval_slice": "train[10064:11088]",
                   "eval_set_sha256": EVAL_SET_SHA256},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [job(m, off) for m in MODES for off in (False, True)]
    jobs.append(job("int8", False, rep=2))
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
