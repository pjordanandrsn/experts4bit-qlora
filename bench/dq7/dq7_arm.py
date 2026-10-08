"""DQ7: exercise Loggetta's actual dense executor; one fresh CUDA process per capacity arm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dq7_subject import CONFIG_HASHES, SUBJECTS


def make_plan(directory, placement, seq, data=None):
    from loggetta import Constraints, Workload, plan
    from loggetta.backends import dense
    from loggetta.hardware import probe

    topology = dense.describe(str(directory))
    hardware = probe()
    workload = Workload(seq_len=seq, steps=2, learning_rate=2e-4, lr_schedule="constant", data=data)
    constraints = Constraints(fixed={"base": "nf4", "placement": placement, "r": 16, "alpha": 32,
                                     "adapter_dtype": "fp32", "targets": "all", "attn_impl": "sdpa",
                                     "loss_chunk": 0 if topology.model_type == "llama" else 512})
    result = plan(topology, hardware, workload, constraints, backends=(dense,))
    if result.status != "feasible":
        raise ValueError("registered setup is refused: " + result.render())
    return result


def prove(directory, out):
    import torch
    from peft.tuners.lora.layer import LoraLayer
    from transformers import AutoTokenizer

    from loggetta import load_adapter
    from loggetta.backends.dense_adapters import save_adapter
    from loggetta.backends.dense_train import adapter_parameters, frozen_digest, prepare

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)
    arms = {}
    for placement in ("device", "stream"):
        torch.manual_seed(731)
        p = make_plan(directory, placement, 64)
        prepared = prepare(p, device="cuda:0")
        assert prepared.report["setup"] == p.selected.setup
        wrappers = [m for m in prepared.model.modules() if isinstance(m, LoraLayer)]
        assert len(wrappers) == 14 and all(m.r["default"] == 16 and m.lora_alpha["default"] == 32 for m in wrappers)
        assert len(prepared.report["quantized_linears"]) == 14
        if placement == "stream":
            report = prepared.report["dense_offload"]
            assert report["layers"] == 2 and report["late_bound_4bit"] == 6 and report["all_pinned"]
        prepared.model.train()
        before = frozen_digest(prepared.model)
        ids = torch.arange(64, device="cuda").unsqueeze(0) + 3
        loss = prepared.model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        params = adapter_parameters(prepared.model)
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        grads = {name: p.grad.detach().cpu().clone() for name, p in params.items()}
        opt = torch.optim.AdamW(prepared.trainable, lr=2e-4)
        opt.step()
        assert frozen_digest(prepared.model) == before
        arms[placement] = (loss.detach().cpu(), grads, prepared, p)
    rloss, rgrads, resident, p = arms["device"]
    sloss, sgrads, streamed, _ = arms["stream"]
    assert torch.equal(rloss, sloss), "deterministic loss differs by placement"
    assert rgrads.keys() == sgrads.keys() and all(torch.equal(rgrads[n], sgrads[n]) for n in rgrads), "LoRA gradients differ"
    rp, sp = adapter_parameters(resident.model), adapter_parameters(streamed.model)
    assert rp.keys() == sp.keys() and all(torch.equal(rp[n], sp[n]) for n in rp), "optimizer updates differ"
    adapter = Path(out).parent / "adapters" / "proof"
    artifact = save_adapter(resident.model, AutoTokenizer.from_pretrained(directory), adapter, p,
                            data={"synthetic": True}, report=resident.report, seed=731)
    restored = load_adapter(adapter, device="cuda:0")
    resident.model.eval()
    resident.model.gradient_checkpointing_disable()
    with torch.no_grad():
        a = resident.model(ids, use_cache=False).logits
        b = restored(ids, use_cache=False).logits
        assert torch.equal(a, b), "exported adapter reload changed logits"
    result = {"schema": "dq7-proof/1", "status": "PASS", "loss_bitwise": True, "all_gradients_bitwise": True,
              "optimizer_updates_bitwise": True, "reload_logits_bitwise": True, "frozen_unchanged": True,
              "gradient_tensors": len(rgrads), "stream": streamed.report, "device": resident.report,
              "artifact": artifact, "algorithm": "math SDPA, deterministic algorithms, TF32 off", "seed": 731}
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def read(directory, subject, placement, seq, out):
    import hashlib
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from loggetta import execute
    from loggetta.data import TrainingData, prepare_data

    output = Path(out)
    datafile = output.parent / f"{subject}-{seq}.jsonl"
    # Public, deterministic text; tokenizer ids are wN. Same text for both placements.
    text = " ".join(f"w{3 + (i % 509)}" for i in range(2 * seq + 16))
    datafile.write_text(json.dumps({"text": text}) + "\n")
    spec = TrainingData(str(datafile), format="text", packing="concat", loss="all")
    config_hash = hashlib.sha256((Path(directory) / "config.json").read_bytes()).hexdigest()
    assert config_hash == CONFIG_HASHES[subject]
    with prepare_data(AutoTokenizer.from_pretrained(directory), 2, seq, spec, seed=731) as prepared:
        assert prepared.blocks.shape == (2, seq) and np.all(prepared.blocks >= 3), "padding or non-real token ids"
        rows_sha = hashlib.sha256(prepared.blocks.tobytes()).hexdigest()
    p = make_plan(directory, placement, seq, spec.to_dict())
    original_clip = torch.nn.utils.clip_grad_norm_
    clip_census = []

    def observed_clip(*args, **kwargs):
        before = torch.cuda.max_memory_allocated()
        value = original_clip(*args, **kwargs)
        torch.cuda.synchronize()
        after = torch.cuda.max_memory_allocated()
        clip_census.append({"peak_before_bytes": before, "peak_after_bytes": after,
                            "added_cumulative_peak_bytes": max(0, after-before)})
        return value

    torch.nn.utils.clip_grad_norm_ = observed_clip
    try:
        result = execute(p, seed=731, out_dir=str(output.parent / "execution"))
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    assert result["data"]["tokens"] == 2 * seq and result["data"]["blocks"] == 2
    assert result["data"]["token_stream_sha256"] == rows_sha
    result["measured"]["clip_peak_census"] = clip_census
    result.update(dq7={"subject": subject, "synthetic": True, "pretrained": False,
                       "seq": seq, "placement": placement, "allocator": "default",
                       "config_sha256": config_hash, "row_tokens_sha256": rows_sha,
                       "real_tokens_per_row": seq, "padded_tokens": 0,
                       "loggetta_sha": os.environ["LOGGETTA_SHA"], "e4b_sha": os.environ["E4B_SHA"]})
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("proof", "read"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--placement", choices=("device", "stream"))
    parser.add_argument("--seq", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = prove(args.checkpoint, args.out) if args.mode == "proof" else read(
        args.checkpoint, args.subject, args.placement, args.seq, args.out)
    print(json.dumps({"status": result["status"], "out": args.out}))
