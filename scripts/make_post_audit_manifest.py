"""Manifest for the post-audit certificate + per-example ∅ ladder jobs (queue T1.1/T1.2 + T2/T3).

18 jobs, all designed for ONE pod (single GPU — the T5(c) architecture confound makes
cross-pod comparison invalid for these):

- 13 null_eval jobs: {nf4, fp4, int8, fp8, bf16, fp16} x {resident, offload} per-example ∅
  evals + one int8-resident repeat (eval-determinism null). The first manifest entry is a
  null_eval so the S8 gate check (first new job must classify claim_usable) lands fast.
- 5 cert jobs: one-step placement trios — bf16 {dropoutOFF-default, dropoutOFF-deterministic,
  dropoutON-default} + int8 {dropoutOFF-default, dropoutOFF-deterministic}.

Usage:
    python scripts/make_post_audit_manifest.py --out runs/job_manifest/post_audit_cert_null_jobs.jsonl
"""

import argparse
import json
import sys

MODES = ("nf4", "fp4", "int8", "fp8", "bf16", "fp16")


def null_job(mode, offload, rep=None):
    suffix = "offload" if offload else "resident"
    job_id = f"null_olmoe_{mode}_{suffix}_perexample" + (f"_rep{rep}" if rep else "")
    cmd = ["python", "-u", "scripts/eval_null_per_example.py", "--job-dir", "{job_dir}",
           "--quant-type", mode]
    if offload:
        cmd.append("--offload")
    return {
        "job_id": job_id,
        "job_type": "null_eval",
        "command": cmd,
        "params": {"model": "allenai/OLMoE-1B-7B-0924", "storage_mode": mode, "offload": offload,
                   "seed": 0, "eval_slice": "train[10000:10064]"},
    }


def cert_job(mode, dropout, deterministic):
    d = "ON" if dropout > 0 else "OFF"
    k = "deterministic" if deterministic else "default"
    job_id = f"cert_olmoe_{mode}_dropout{d}_{k}"
    cmd = ["python", "-u", "scripts/one_step_certificate.py", "--job-dir", "{job_dir}",
           "--quant-type", mode, "--seed", "1337"]
    if dropout > 0:
        cmd += ["--dropout", str(dropout)]
    if deterministic:
        cmd.append("--deterministic")
    return {
        "job_id": job_id,
        "job_type": "cert",
        "command": cmd,
        "params": {"model": "allenai/OLMoE-1B-7B-0924", "storage_mode": mode, "seed": 1337,
                   "dropout": dropout, "deterministic": deterministic},
    }


def build_jobs():
    jobs = []
    for mode in MODES:
        for offload in (False, True):
            jobs.append(null_job(mode, offload))
    jobs.append(null_job("int8", False, rep=2))
    jobs.append(cert_job("bf16", 0.0, False))
    jobs.append(cert_job("bf16", 0.0, True))
    jobs.append(cert_job("bf16", 0.1, False))
    jobs.append(cert_job("int8", 0.0, False))
    jobs.append(cert_job("int8", 0.0, True))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = build_jobs()
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    for j in jobs:
        print(f"  {j['job_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
