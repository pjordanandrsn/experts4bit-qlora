"""One isolated correctness process per placement/cache mode; never a capacity reading."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dq10_plan import make_plan
from dq9_phase import PhaseCensus


def tensor_sha(tensor):
    import torch

    raw = tensor.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def prove(directory, subject, placement, out):
    # Required before Torch/CUDA initialization; CPU copies and synchronization are allowed in this proof only.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from peft.tuners.lora.layer import LoraLayer
    from transformers import AutoTokenizer

    from loggetta import load_adapter
    from loggetta.backends import dense_loader, dense_train
    from loggetta.backends.dense_adapters import save_adapter

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(731)
    p = make_plan(directory, placement, 64, policy=False)
    census = PhaseCensus(torch, "baseline")
    snapshots = []
    with census.installed(dense_loader, dense_train):
        prepared = dense_train.prepare(p, device="cuda:0")
        wrappers = [m for m in prepared.model.modules() if isinstance(m, LoraLayer)]
        assert len(wrappers) == 14 and all(m.r["default"] == 16 and m.lora_alpha["default"] == 32 for m in wrappers)
        assert len(prepared.report["quantized_linears"]) == 14
        assert prepared.report["setup"] == p.selected.setup and prepared.report["chunked_loss_engaged"] == 0
        if placement == "stream":
            r = prepared.report["dense_offload"]
            assert r["layers"] == 2 and r["late_bound_4bit"] == 6 and r["all_pinned"]
        prepared.model.train()
        frozen = dense_train.frozen_digest(prepared.model)
        embedding = prepared.model.get_input_embeddings().weight
        head = prepared.model.get_output_embeddings().weight
        tied = embedding is head
        assert tied == (subject == "tiny-smollm3")
        embedding_sha, head_sha = tensor_sha(embedding), tensor_sha(head)
        params = dense_train.adapter_parameters(prepared.model)
        opt = torch.optim.AdamW(prepared.trainable, lr=2e-4)
        ids = torch.arange(64, device="cuda").unsqueeze(0) + 3
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = prepared.model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
            grads = {name: tensor_sha(p.grad) for name, p in params.items()}
            torch.nn.utils.clip_grad_norm_(prepared.trainable, 1.0)
            opt.step()
            assert dense_train.frozen_digest(prepared.model) == frozen
            assert tensor_sha(embedding) == embedding_sha and tensor_sha(head) == head_sha
            assert (prepared.model.get_input_embeddings().weight is prepared.model.get_output_embeddings().weight) == tied
            snapshots.append({"loss_sha256": tensor_sha(loss), "gradients_sha256": grads,
                              "updates_sha256": {name: tensor_sha(p) for name, p in params.items()}})
    # Outside phase telemetry: export/reload must not add a third observed training forward.
    adapter = Path(out).parent.parent / "adapters" / Path(out).stem
    artifact = save_adapter(prepared.model, AutoTokenizer.from_pretrained(directory), adapter, p,
                            data={"synthetic": True}, report=prepared.report, seed=731)
    restored = load_adapter(adapter, device="cuda:0")
    restored_params = {name: value for name, value in restored.named_parameters() if name in params}
    assert params.keys() == restored_params.keys() and all(torch.equal(params[n], restored_params[n]) for n in params)
    prepared.model.eval()
    prepared.model.gradient_checkpointing_disable()
    with torch.no_grad():
        a = prepared.model(ids, use_cache=False).logits
        b = restored(ids, use_cache=False).logits
        assert torch.equal(a, b), "adapter reload changed logits"
        logits_sha = tensor_sha(a)
    result = {"schema": "dq10-proof/1", "status": "PASS", "placement": placement, "subject": subject, "tied_parameter_identity": tied, "embedding_sha256": embedding_sha, "head_sha256": head_sha,
              "steps": snapshots, "frozen_unchanged": True, "reload_logits_bitwise": True,
              "reload_logits_sha256": logits_sha, "gradient_tensors": len(params), "seed": 731,
              "algorithm": "math SDPA, deterministic algorithms, TF32 off", "engaged": prepared.report,
              "phases": census.report(), "artifact": artifact,
              "loggetta_sha": os.environ["LOGGETTA_SHA"], "e4b_sha": os.environ["E4B_SHA"]}
    with Path(out).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--placement", choices=("device", "stream"), required=True)
    p.add_argument("--subject", choices=("tiny-mistral", "tiny-smollm3"), required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = prove(a.checkpoint, a.subject, a.placement, a.out)
    print(json.dumps({"status": r["status"], "out": a.out}))
