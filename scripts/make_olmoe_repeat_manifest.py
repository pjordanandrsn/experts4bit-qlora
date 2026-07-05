"""Generate the OLMoE repeat-validation job manifest (docs/OLMOE_REPEAT_VALIDATION_PLAN.md).

Phases are gated (the operating rule: parallelize confirmed repeat questions, not curiosity):

  phase 1 — 12 training repeats: {nf4, int8} x {resident, offload} x seeds {1337, 2027, 3407}.
            Repeats the OLMoE candidate findings (int8-offload best eval; offload memory-floor
            collapse across storage width) with same-model, same-hyperparameter runs.
  phase 2 — 3 decode repeats (nf4/fp4/int8 resident, 5 measured samples + 1 discarded warmup):
            repeats the single-sample fp4-faster-than-nf4 decode observation.
  phase 3 — focused portability queries over the phase-1 adapters ({train row} x {nf4, int8}
            resident query). GENERATED ONLY when the referenced adapters exist on disk —
            run this generator again with --phase 3 after phase 1 drains.

Deterministic output (sorted, no timestamps) so the manifest can be committed and diffed.

Usage:
    python scripts/make_olmoe_repeat_manifest.py --phase 1 --phase 2 \\
        --out runs/job_manifest/olmoe_repeat_jobs.jsonl
    # later, after phase-1 adapters exist under --jobs-root:
    python scripts/make_olmoe_repeat_manifest.py --phase 3 --jobs-root /workspace/runs/jobs \\
        --out runs/job_manifest/olmoe_query_jobs.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import parse_mode_label  # noqa: E402

MODEL = "allenai/OLMoE-1B-7B-0924"
SEEDS = (1337, 2027, 3407)
TRAIN_MODES = ("nf4", "nf4-offload", "int8", "int8-offload")
DECODE_MODES = ("nf4", "fp4", "int8")
QUERY_MODES = ("nf4", "int8")


def _train_job(label, seed):
    storage, offload = parse_mode_label(label)
    job_id = f"train_olmoe_{storage}_{'offload' if offload else 'resident'}_seed{seed}"
    return {
        "job_id": job_id,
        "job_type": "train",
        "command": [
            "python", "-u", "scripts/train_mode_adapters.py",
            "--model", MODEL, "--train-modes", label,
            "--steps", "150", "--seed", str(seed),
            "--adapter-root", "{job_dir}", "--out", "{job_dir}/result_rows.jsonl",
        ],
        "params": {"model": MODEL, "dataset": "tatsu-lab/alpaca", "storage_mode": storage,
                   "offload": offload, "seed": seed, "steps": 150, "eval_schedule": "every 50 steps"},
    }


def _decode_job(storage):
    return {
        "job_id": f"decode_olmoe_{storage}_resident_repeat5",
        "job_type": "decode",
        "command": [
            "env", f"QUANT_TYPE={storage}", f"MODEL={MODEL}", "OFFLOAD_EXPERTS=0",
            "python", "-u", "scripts/decode_repeat.py",
            "--samples", "5", "--tokens", "128", "--job-dir", "{job_dir}",
        ],
        "params": {"model": MODEL, "storage_mode": storage, "offload": False},
    }


def _query_jobs(jobs_root):
    """Phase 3: one query job per (existing phase-1 adapter) x (query mode). Refuses to emit a
    job whose adapter is not on disk — the gate is physical, not aspirational."""
    jobs, missing = [], []
    for label in TRAIN_MODES:
        storage, offload = parse_mode_label(label)
        variant = "offload" if offload else "resident"
        for seed in SEEDS:
            train_id = f"train_olmoe_{storage}_{variant}_seed{seed}"
            adapter_root = os.path.join(jobs_root, train_id)
            adapter = os.path.join(adapter_root, label, "adapter_best.pt")
            if not os.path.exists(adapter):
                missing.append(train_id)
                continue
            for q in QUERY_MODES:
                jobs.append({
                    "job_id": f"query_olmoe_train-{storage}-{variant}-seed{seed}_query-{q}-resident",
                    "job_type": "query",
                    "command": [
                        "python", "-u", "scripts/eval_adapter_query_modes.py",
                        "--model", MODEL, "--adapter-root", adapter_root,
                        "--query-modes", q, "--out", "{job_dir}/result_rows.jsonl",
                    ],
                    "params": {"model": MODEL, "train_storage_mode": storage, "train_offload": offload,
                               "train_seed": seed, "query_storage_mode": q, "query_offload": False},
                })
    return jobs, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, action="append", required=True, choices=(1, 2, 3))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs-root", default="runs/jobs", help="phase 3: where phase-1 job dirs live")
    args = ap.parse_args()

    jobs = []
    if 1 in args.phase:
        jobs += [_train_job(label, seed) for label in TRAIN_MODES for seed in SEEDS]
    if 2 in args.phase:
        jobs += [_decode_job(s) for s in DECODE_MODES]
    if 3 in args.phase:
        qjobs, missing = _query_jobs(args.jobs_root)
        if missing:
            print(f"phase 3 gate: {len(missing)} train jobs lack adapters — NOT emitting their queries:")
            for m in sorted(set(missing)):
                print(f"  missing: {m}")
        jobs += qjobs

    seen = set()
    for j in jobs:
        if j["job_id"] in seen:
            raise SystemExit(f"duplicate job_id {j['job_id']}")
        seen.add(j["job_id"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    for j in jobs:
        print(f"  {j['job_type']:<7} {j['job_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
