"""Generate the expert-streaming profile-job manifest (docs/EXPERT_STREAMING_PROFILE.md).

Profile-ONLY jobs, separate from the repeat-validation grid — outputs live under
runs/expert_streaming/. Training profiles first (they carry the staging stall that matters);
decode profiles after. Best first target: OLMoE-1B-7B int8-offload training, with nf4-offload as
the comparison.

Jobs (all offload — profiling the resident path would show no staging at all):
  profile_olmoe_int8_offload_train_seed1337_steps100
  profile_olmoe_nf4_offload_train_seed1337_steps100
  profile_olmoe_int8_offload_decode_repeat5
  profile_olmoe_nf4_offload_decode_repeat5

Each job sets E4B_EXPERT_PROFILE to a job-local path; the worker runner substitutes {job_dir}.

Usage:
    python scripts/make_expert_streaming_manifest.py --out runs/job_manifest/expert_streaming_jobs.jsonl
"""

import argparse
import json
import os

MODEL = "allenai/OLMoE-1B-7B-0924"
SEED = 1337


def _train_job(storage):
    job_id = f"profile_olmoe_{storage}_offload_train_seed{SEED}_steps100"
    return {
        "job_id": job_id,
        "job_type": "profile_train",
        "command": [
            "env", f"MODEL={MODEL}", f"QUANT_TYPE={storage}", "OFFLOAD_EXPERTS=1",
            "STEPS=100", f"SEED={SEED}", "DO_GEN=0", "EVAL_EVERY=100",
            "E4B_PROFILE_PHASE=train", "E4B_EXPERT_PROFILE={job_dir}/profile.jsonl",
            "OUT={job_dir}/adapter",
            "python", "-u", "-m", "experts4bit_qlora.train",
        ],
        "params": {"model": MODEL, "storage_mode": storage, "offload": True, "phase": "train",
                   "seed": SEED, "steps": 100},
    }


def _decode_job(storage):
    job_id = f"profile_olmoe_{storage}_offload_decode_repeat5"
    return {
        "job_id": job_id,
        "job_type": "profile_decode",
        "command": [
            "env", f"MODEL={MODEL}", f"QUANT_TYPE={storage}", "OFFLOAD_EXPERTS=1", "PREFETCH=1",
            "E4B_PROFILE_PHASE=decode", "E4B_EXPERT_PROFILE={job_dir}/profile.jsonl",
            "python", "-u", "scripts/decode_repeat.py",
            "--samples", "5", "--tokens", "128", "--job-dir", "{job_dir}",
        ],
        "params": {"model": MODEL, "storage_mode": storage, "offload": True, "phase": "decode"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [_train_job("int8"), _train_job("nf4"), _decode_job("int8"), _decode_job("nf4")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} profile jobs -> {args.out}")
    for j in jobs:
        print(f"  {j['job_type']:<14} {j['job_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
