"""Train one QLoRA adapter per requested ExpertsNbit train mode, with provenance sidecars.

Row-producer for the train/query storage-mode matrix (docs/MODE_DECOUPLED_ADAPTERS.md). Each
train mode gets its own subdirectory under ``--adapter-root`` holding the adapter files written
by ``experts4bit_qlora.train`` plus a ``expertsnbit_adapter_metadata.json`` sidecar recording the
train storage mode, offload setting, environment, and training outcome — storage mode is part of
adapter provenance.

Reuses the existing trainer verbatim (one ``python -m experts4bit_qlora.train`` subprocess per
mode, mirroring the repo's existing runner shells) rather than duplicating a training loop —
this is a validation grid, and the rows must be the real training path. Resumable: a mode
whose adapter + sidecar already exist is skipped under ``--resume``. A failed leg is recorded and
the grid continues unless ``--fail-fast``.

Example:
    python scripts/train_mode_adapters.py \\
      --model allenai/OLMoE-1B-7B-0924 \\
      --train-modes nf4,nf4-offload,fp4,int8,int8-offload \\
      --steps 150 --seed 0 \\
      --adapter-root runs/olmoe_mode_adapters \\
      --out runs/olmoe_mode_adapters/train_results.jsonl
"""

import argparse
import os
import re
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import (  # noqa: E402
    append_jsonl,
    mode_label,
    parse_mode_list,
    read_metadata,
    sha256_file,
    utc_now,
    write_metadata,
)


def _env_versions():
    import torch

    try:
        import bitsandbytes

        bnb = bitsandbytes.__version__
    except Exception:
        bnb = None
    try:
        import experts4bit_qlora

        pkg = getattr(experts4bit_qlora, "__version__", None)
    except Exception:
        pkg = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "package_version": pkg,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "bitsandbytes_version": bnb,
        "gpu_name": gpu,
        "train_host": os.uname().nodename,
    }


def _repo_commit():
    try:
        r = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _parse_train_log(log_path):
    """Pull the trainer's own reported numbers out of its log (the same lines the repo's
    existing runner-shell summaries grep)."""
    out = {"train_eval_before": None, "train_eval_after": None, "train_eval_best": None,
           "peak_train_gpu_gb": None, "seconds_per_step": None, "total_runtime_sec": None}
    try:
        text = open(log_path, errors="replace").read().replace("\r", "\n")
    except OSError:
        return out
    m = re.search(r"BEFORE ([0-9.]+) -> AFTER ([0-9.]+).*best ([0-9.]+)", text)
    if m:
        out["train_eval_before"], out["train_eval_after"], out["train_eval_best"] = map(float, m.groups())
    m = re.search(r"peak GPU mem: ([0-9.]+) GB", text)
    if m:
        out["peak_train_gpu_gb"] = float(m.group(1))
    m = re.search(r"training done in ([0-9.]+)s", text)
    if m:
        out["total_runtime_sec"] = float(m.group(1))
    steps = re.findall(r"\(([0-9.]+)s/step\)", text)
    if steps:
        out["seconds_per_step"] = float(steps[-1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--train-modes", required=True, help="csv of mode labels, e.g. nf4,nf4-offload,int8")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--adapter-root", required=True)
    ap.add_argument("--out", required=True, help="JSONL of one row per train leg")
    ap.add_argument("--resume", action="store_true", help="skip modes whose adapter+sidecar exist")
    ap.add_argument("--dry-run", action="store_true", help="print the planned grid and exit")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    modes = parse_mode_list(args.train_modes)
    print(f"planned train grid ({len(modes)} legs) on {args.model}:")
    for label, storage, offload in modes:
        print(f"  {label:<14} storage={storage} offload={offload} steps={args.steps} seed={args.seed}")
    print(f"adapters -> {args.adapter_root}/<label>/ ; rows -> {args.out}")
    if args.dry_run:
        return 0

    env_info = _env_versions()
    commit = _repo_commit()
    run_id = uuid.uuid4().hex[:12]
    failures = 0
    for label, storage, offload in modes:
        adapter_dir = os.path.join(args.adapter_root, label)
        adapter_path = os.path.join(adapter_dir, "adapter_best.pt")
        if args.resume and os.path.exists(adapter_path) and read_metadata(adapter_dir):
            print(f"[skip] {label}: adapter + sidecar already present (--resume)")
            continue
        os.makedirs(adapter_dir, exist_ok=True)
        log_path = os.path.join(adapter_dir, "train.log")
        env = dict(
            os.environ,
            MODEL=args.model,
            QUANT_TYPE=storage,
            OFFLOAD_EXPERTS="1" if offload else "0",
            STEPS=str(args.steps),
            SEED=str(args.seed),
            SEQ=str(args.seq),
            GRAD_ACCUM=str(args.grad_accum),
            LR=str(args.lr),
            N_TRAIN=str(args.n_train),
            R=str(args.r),
            ALPHA=str(args.alpha),
            EVAL_EVERY="50",
            DO_GEN="0",
            TRAIN_EXPERTS="1",
            TRAIN_ATTENTION="1",
            TRAIN_ROUTER="0",
            OUT=adapter_dir,
        )
        print(f"[train] {label} -> {adapter_dir} (log: {log_path})")
        with open(log_path, "w") as lf:
            rc = subprocess.run(
                [sys.executable, "-u", "-m", "experts4bit_qlora.train"], env=env, stdout=lf, stderr=subprocess.STDOUT
            ).returncode
        parsed = _parse_train_log(log_path)
        status = "pass" if rc == 0 and os.path.exists(adapter_path) else "fail"
        meta = {
            "base_model": args.model,
            "repo_commit": commit,
            "train_mode_label": label,
            "train_storage_mode": storage,
            "train_offload": offload,
            "dataset": "tatsu-lab/alpaca (response-only loss)",
            "eval_split": f"train[{args.n_train}:{args.n_train + 64}]",
            "seed": args.seed,
            "steps": args.steps,
            "optimizer": "AdamW + cosine schedule (experts4bit_qlora.train)",
            "learning_rate": args.lr,
            "lora": {"r": args.r, "alpha": args.alpha, "dropout": 0.0,
                     "target_modules": "per-expert gate_up/down + attention q/k/v/o"},
            "seq": args.seq,
            "grad_accum": args.grad_accum,
            "command_line": " ".join(sys.argv),
            **env_info,
            **parsed,
        }
        if status == "pass":
            meta["adapter_sha256"] = sha256_file(adapter_path)
        meta_path = write_metadata(adapter_dir, meta)
        append_jsonl(args.out, {
            "run_id": run_id,
            "base_model": args.model,
            "train_mode_label": label,
            "train_storage_mode": storage,
            "train_offload": offload,
            "query_mode_label": mode_label(storage, offload),  # a train row queries nothing; self-labelled
            "query_storage_mode": storage,
            "query_offload": offload,
            "storage_mode_mismatch": False,
            "offload_mismatch": False,
            "adapter_path": adapter_path,
            "adapter_metadata_path": meta_path,
            "status": status,
            "skip_or_fail_reason": None if status == "pass" else f"trainer exit={rc}, see {log_path}",
            "timestamp": utc_now(),
            **{k: parsed[k] for k in ("train_eval_before", "train_eval_after", "train_eval_best",
                                      "peak_train_gpu_gb", "total_runtime_sec")},
            "train_s_per_step": parsed["seconds_per_step"],
            "seed": args.seed,
            "steps": args.steps,
            "commit": commit,
            **{k: env_info[k] for k in ("torch_version", "cuda_version", "bitsandbytes_version", "gpu_name")},
            "host": env_info["train_host"],
        })
        print(f"[{status}] {label}: eval {parsed['train_eval_before']} -> {parsed['train_eval_after']} "
              f"(best {parsed['train_eval_best']}), peak {parsed['peak_train_gpu_gb']} GB, "
              f"{parsed['seconds_per_step']} s/step")
        if status == "fail":
            failures += 1
            if args.fail_fast:
                return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
