"""Save the real Qwen3-32B resident boundary plan before any model weight loading."""
import argparse
import hashlib
import json
from pathlib import Path

from dq7_subject import CONFIG_HASHES


def refusal(directory):
    import torch
    from loggetta import Constraints, Workload, plan
    from loggetta.backends import dense
    from loggetta.hardware import probe

    hardware = probe()
    config_hash = hashlib.sha256((Path(directory) / "config.json").read_bytes()).hexdigest()
    if config_hash != CONFIG_HASHES["qwen3_32b"]:
        raise ValueError("refusal control configuration checksum differs")
    before = torch.cuda.memory_allocated()
    result = plan(dense.describe(directory), hardware,
                  Workload(seq_len=4096, steps=2, learning_rate=2e-4, lr_schedule="constant"),
                  Constraints(allow_development_executor=True, fixed={"base": "nf4", "placement": "device",
                      "r": 16, "alpha": 32, "adapter_dtype": "fp32", "attn_impl": "sdpa", "loss_chunk": 512,
                      "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"]}), backends=(dense,))
    after = torch.cuda.memory_allocated()
    if before != 0 or after != 0:
        raise ValueError("refusal control allocated CUDA model tensors")
    return {"schema": "dq8-refusal/1", "status": "REFUSED" if result.status != "feasible" else "BOUNDARY_CHANGED",
            "subject": "qwen3_32b", "placement": "device", "seq": 4096,
            "config_sha256": config_hash,
            "allocated_before": before, "allocated_after": after, "plan": result.to_dict()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = refusal(args.checkpoint)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
