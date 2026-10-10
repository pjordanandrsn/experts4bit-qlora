"""Verify sealed inputs, source/runtime and actual admission before full weight fetch."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from dq11_arm import read_inputs, runtime
from dq11_common import file_sha


def prepare(directory):
    runtime(directory)
    read_inputs(directory)
    import torch
    from huggingface_hub import snapshot_download
    from loggetta import Constraints, Workload, plan
    from loggetta.backends import dense
    from loggetta.hardware import probe

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("single CUDA GPU required")
    hardware = probe()
    gpu = hardware.gpu()
    rehearsal = os.environ.get("DQ11_REHEARSAL") == "1"
    if rehearsal:
        from dq11_rehearsal import require_hardware

        require_hardware(directory)
    elif gpu.name != "NVIDIA GeForce RTX 5090" or not 30 * (1 << 30) <= gpu.memory_total.value <= 33 * (1 << 30):
        raise ValueError("unregistered card/VRAM")
    from experts4bit_qlora.engines.dense_offload import _bnb_mirror_mismatches

    if _bnb_mirror_mismatches():
        raise ValueError("bnb late-bound mirror tripwire failed")
    source = json.loads((directory / "model_files.json").read_text())
    config_names = [row["name"] for row in source["files"] if row["name"].endswith(".json")]
    if rehearsal:
        snapshot = directory / "tiny-model"
    else:
        snapshot = Path(snapshot_download(source["model"], revision=source["revision"], allow_patterns=config_names,
                                          token=False, cache_dir=str(directory / "hf-cache")))
    constraints = Constraints(fixed={"base": "nf4", "placement": "stream", "adapter_dtype": "fp32",
                                     "r": 16, "alpha": 32, "targets": list(dense.ROLES),
                                     "attn_impl": "sdpa", "loss_chunk": 0}, allow_development_executor=True)
    workload = Workload(seq_len=2048, micro_batch=1, grad_accum=1, steps=40,
                        optimizer="adamw", learning_rate=2e-4, lr_schedule="constant", warmup_steps=0)
    admission = plan(dense.describe(str(snapshot)), hardware, workload, constraints, backends=(dense,))
    if admission.status != "feasible":
        raise ValueError("actual baseline admission refused before weight fetch: " + admission.render())
    from loggetta.dense_policy import validate_plan

    validate_plan(admission)
    admission_receipt = asdict(admission)
    if rehearsal:
        admission_receipt = {"schema": "dq11-rehearsal-admission/1", "science_eligible": False,
                             "plan": admission_receipt}
    (directory / "receipts/admission.json").write_text(json.dumps(admission_receipt, indent=2) + "\n")
    if not rehearsal:
        snapshot_download(source["model"], revision=source["revision"], allow_patterns=[r["name"] for r in source["files"]],
                          token=False, cache_dir=str(directory / "hf-cache"))
    for row in source["files"]:
        path = snapshot / row["name"]
        if path.stat().st_size != row["size"] or file_sha(path) != row["sha256"]:
            raise ValueError("pinned model/tokenizer source bytes differ")
    link = directory / "hf-cache/model"
    if link.exists():
        raise ValueError("model staging already exists; no second draw over a stale checkpoint")
    link.symlink_to(snapshot, target_is_directory=True)
    receipt = {"schema": "dq11-prepared/1", "nonce": os.environ["TC1_RUN_NONCE"],
               "model_manifest_sha256": file_sha(directory / "model_files.json"),
               "input_seal_sha256": file_sha(directory / "locked_inputs.json"), "hardware": hardware.to_dict()}
    if rehearsal:
        receipt.update(schema="dq11-rehearsal-prepared/1", science_eligible=False)
    (directory / "receipts/prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    args = parser.parse_args()
    prepare(args.directory)
