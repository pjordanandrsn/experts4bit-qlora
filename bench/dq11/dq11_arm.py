"""Fresh-process DQ11 proof or forty-update read on the registered model/recipe."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from dq11_common import (adapter_slots, base_census, file_sha, install_initial,
                         object_sha, path_census, score, tensor_sha)


def runtime(directory):
    from dq11_rehearsal import mode

    mode(directory)
    if sys.version_info[:2] != (3, 11):
        raise ValueError("DQ11 requires Python 3.11")
    wheels = json.loads((directory / "wheels.json").read_text())["packages"]
    versions = {row["name"]: md.version(row["name"]) for row in wheels}
    if any(versions[row["name"]] != row["version"] for row in wheels):
        raise ValueError("resolved runtime version changed")
    for name, expected in (("experts4bit-qlora", os.environ["E4B_SHA"]),
                           ("loggetta", "34ecb6cec6f43a6f8607ff9f192749fdc7b587e9")):
        direct = json.loads(md.distribution(name).read_text("direct_url.json"))
        if direct["vcs_info"]["commit_id"] != expected:
            raise ValueError("installed source object changed")
        versions[name] = expected
    return versions


def prepare(directory, arm):
    # Import Unsloth first, only in its arms. L never imports its global patches.
    if any(key.startswith("UNSLOTH_") for key in os.environ):
        raise ValueError("inherited Unsloth switches are not registered")
    if arm in ("U", "U0"):
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        from unsloth import FastLanguageModel
    import torch
    from loggetta import Constraints, Workload, plan
    from loggetta.backends import dense, dense_train
    from loggetta.hardware import probe

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(3407)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("DQ11 requires exactly one CUDA device")
    if os.environ.get("DQ11_REHEARSAL") == "1":
        from dq11_rehearsal import require_hardware

        require_hardware(directory)
    elif torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5090":
        raise ValueError("unregistered GPU")
    snapshot = directory / "hf-cache/model"
    workload = Workload(seq_len=2048, micro_batch=1, grad_accum=1, steps=40,
                        optimizer="adamw", learning_rate=2e-4, lr_schedule="constant", warmup_steps=0)
    report = {}
    if arm == "L":
        constraints = Constraints(fixed={"base": "nf4", "placement": "stream", "adapter_dtype": "fp32",
                                         "r": 16, "alpha": 32, "targets": list(dense.ROLES),
                                         "attn_impl": "sdpa", "loss_chunk": 0}, allow_development_executor=True)
        actual = plan(dense.describe(str(snapshot)), probe(), workload, constraints, backends=(dense,))
        if actual.status != "feasible":
            raise ValueError("actual shipped admission refused: " + actual.render())
        from loggetta.dense_policy import validate_plan

        validate_plan(actual)
        if os.environ.get("DQ11_REHEARSAL") == "1":
            from dq11_rehearsal import prepare_log

            prepared = prepare_log(directory, actual, dense_train.prepare)
        else:
            prepared = dense_train.prepare(actual)
        model, trainable = prepared.model, prepared.trainable
        report = {"plan": asdict(actual), "engaged": prepared.report}
    else:
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model, _ = FastLanguageModel.from_pretrained(model_name=str(snapshot), max_seq_length=2048,
                                                     dtype=torch.bfloat16, load_in_4bit=True,
                                                     quantization_config=config, local_files_only=True)
        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0,
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                "gate_proj", "up_proj", "down_proj"],
                                                bias="none", use_gradient_checkpointing=True,
                                                random_state=3407, use_rslora=False, loftq_config=None)
        if arm == "U0":
            from unsloth.kernels.fast_lora import apply_lora_mlp_swiglu
            from unsloth.models.llama import original_apply_o, original_apply_qkv

            for name, module in model.named_modules():
                if name.endswith(".self_attn"):
                    module.apply_qkv = original_apply_qkv
                    module.apply_o = original_apply_o
                elif name.endswith(".mlp"):
                    if hasattr(module, "_unsloth_forward") or getattr(module.forward, "__func__", None) is not apply_lora_mlp_swiglu:
                        raise ValueError("unknown/tiled MLP; cannot safely ablate adapter fusion")
                    del module.forward
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        trainable = list(adapter_slots(model)[0].values())
        report = {"placement": "device", "compile_disable": True, "return_logits": True}
    checkpoints = [module for module in model.modules() if getattr(module, "gradient_checkpointing", False)]
    if not checkpoints or any(getattr(getattr(module, "_gradient_checkpointing_func", None), "keywords", {}).get("use_reentrant") is not False
                              for module in checkpoints):
        raise ValueError("non-reentrant checkpointing did not engage")
    model.train()
    return model, trainable, workload, report


def read_inputs(directory):
    locked = json.loads((directory / "locked_inputs.json").read_text())
    for name, row in locked["assets"].items():
        path = directory / ("adapters" if name.endswith(".safetensors") else "data") / name
        if file_sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("canonical input bytes changed")
    tokens = json.loads((directory / "data/tokens.json").read_text())
    for name, row in locked["tokens"].items():
        digest = __import__("hashlib").sha256(json.dumps(tokens[name], separators=(",", ":")).encode()).hexdigest()
        if digest != row["sha256"] or len(tokens[name]) != row["blocks"] or any(len(block) != 2048 for block in tokens[name]):
            raise ValueError("canonical token rows changed")
    return locked, tokens


def run(args):
    if (args.kind in ("proof", "spread")) != (args.repetition == 0):
        raise ValueError("unregistered phase/repetition pair")
    from dq11_proof_policy import process_environment

    with process_environment(args.kind):
        return _run(args)


def _run(args):
    started = time.perf_counter()
    directory = args.directory
    versions = runtime(directory)
    locked, tokens = read_inputs(directory)
    sources = set(json.loads((directory / "source_authority.json").read_text())["sha256"])
    load_started = time.perf_counter()
    # loggetta.measure is framework-free, so Unsloth still imports before torch.
    from loggetta.measure import DriverMemorySampler, proc_status

    with DriverMemorySampler() as load_sampler:
        model, trainable, workload, report = prepare(directory, args.arm)
    import torch
    from loggetta.backends import dense_train

    torch.cuda.synchronize()
    load_measured = {"load_seconds": time.perf_counter() - load_started,
                     "load_device_peak_bytes": torch.cuda.max_memory_allocated(),
                     "load_driver_process_peak_bytes": load_sampler.peak or None,
                     "load_driver_samples": load_sampler.samples,
                     "load_host_required_peak_bytes": load_sampler.required_peak or None,
                     "host_anon_after_load_bytes": proc_status().get("RssAnon")}
    if runtime(directory) != versions or torch.is_autocast_enabled():
        raise ValueError("runtime changed during import or autocast engaged")

    initial = install_initial(model, directory / "adapters/adapter_init.safetensors")
    census = path_census(model, args.arm, sources)
    base = base_census(model)
    initial_quality = score(model, tokens)
    result = {"schema": "dq11-arm/1", "arm": args.arm, "repetition": args.repetition,
              "kind": args.kind, "nonce": os.environ["TC1_RUN_NONCE"], "runtime": versions,
              "initial": initial, "input_seal_sha256": file_sha(directory / "locked_inputs.json"),
              "tokens": locked["tokens"], "base": base, "path_before": census,
              "initial_quality": initial_quality, "engaged": report,
              "provenance": {name: file_sha(directory / name) for name in
                             ("science.sha256", "wheels.json", "model_files.json", "source_authority.json")},
              "load_measured": load_measured}
    from dq11_proof_policy import settings

    result["execution_settings"] = settings()
    slots, _ = adapter_slots(model)
    if args.kind == "proof":
        from dq11_observe import Observer

        ids = torch.tensor([tokens["train"][0]], device="cuda", dtype=torch.long)
        from dq11_proof_policy import observer_policy

        policy = None
        try:
            with observer_policy() as policy:
                model.zero_grad(set_to_none=True)
                observer = Observer(model, args.arm)
                with observer.active():
                    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                    loss.backward()
                observed = {key: tensor_sha(p.grad) for key, p in slots.items()}
                if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in slots.values()):
                    raise ValueError("nonfinite proof loss")
                result["execution"] = observer.receipt()
                observed_loss = tensor_sha(loss)
                del loss
                model.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                loss.backward()
                clean = {key: tensor_sha(p.grad) for key, p in slots.items()}
                if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in slots.values()):
                    raise ValueError("nonfinite clean proof loss or gradient")
                if observed != clean or observed_loss != tensor_sha(loss):
                    raise ValueError("observer changed same-arm loss or gradients")
                result["observer_same_arm_bitwise"] = True
                model.zero_grad(set_to_none=True)
                result["proof_gradients_sha256"] = clean
        finally:
            # Retain warn_only suspects even when the bitwise comparison refuses.
            (directory / f"receipts/observer-policy-{args.arm}.json").write_text(json.dumps({
                "schema": "dq11-observer-policy/1", "arm": args.arm,
                "nonce": os.environ["TC1_RUN_NONCE"], "source": os.environ["E4B_SHA"],
                "science_eligible": os.environ.get("DQ11_REHEARSAL") != "1", "policy": policy,
                "bitwise_pass": result.get("observer_same_arm_bitwise", False)}, indent=2) + "\n")
        result["observer_policy"] = policy
    elif args.kind == "spread":
        from dq11_proof_policy import spread

        ids = torch.tensor([tokens["train"][0]], device="cuda", dtype=torch.long)
        result["spread"] = spread(model, slots, ids, output=directory / "receipts", arm=args.arm)
        if {key: tensor_sha(p) for key, p in slots.items()} != initial:
            raise ValueError("spread changed canonical initializer")
    else:
        from dq11_reduce import initial_gate

        proofs = [json.loads((directory / f"receipts/proof-{arm}.json").read_text()) for arm in ("L", "U", "U0")]
        if not initial_gate(proofs):
            raise ValueError("global initial quality gate failed before training")
        proof = next(row for row in proofs if row["arm"] == args.arm)
        from dq11_proof_policy import validate_shipped_policy

        validate_shipped_policy(result["execution_settings"])
        if census != proof["path_before"] or base != proof["base"]:
            raise ValueError("clean process differs from untimed proof")
        initial_rows = [dict(row, initial_quality=initial_quality) if row["arm"] == args.arm else row for row in proofs]
        if not initial_gate(initial_rows):
            raise ValueError("clean process initial quality failed before training")
        from loggetta.backends.experts4bit_train import train_loop
        route_before = None
        rehearsal = os.environ.get("DQ11_REHEARSAL") == "1"
        if args.arm == "L":
            from dq11_stream_witness import train_prefetch_snapshot, require_train_prefetch

            route_before = train_prefetch_snapshot(model)
        data = SimpleNamespace(blocks=tokens["train"], info={"source": "DQ11 sealed tokens", "tokens": locked["tokens"]["train"]})
        with DriverMemorySampler() as sampler:
            result["training"] = train_loop(model, trainable, str(directory / "hf-cache/model"), workload,
                                             sampler, {}, seed=3407, warmup=5, prepared_data=data,
                                             frozen_digest=dense_train.frozen_digest, frozen_kind="dense")
        if result["training"]["status"] != "OK":
            raise ValueError("shipped loop failed integrity")
        if route_before is not None:
            result["rehearsal_train_prefetch" if rehearsal else "train_prefetch"] = require_train_prefetch(
                directory, model, route_before, len(result["training"]["correctness"]["losses"]), rehearsal=rehearsal)
        result["final_quality"] = score(model, tokens)
        result["final_adapters_sha256"] = {key: tensor_sha(p) for key, p in slots.items()}
    if settings() != result["execution_settings"]:
        raise ValueError("execution policy changed during arm")
    after = path_census(model, args.arm, sources)
    if after != census or base_census(model) != base:
        raise ValueError("loaded callables or frozen weights changed")
    result["path_after"] = after
    result["frozen_unchanged"] = True
    result["identity_sha256"] = object_sha({"initial": initial, "tokens": locked["tokens"], "runtime": versions})
    if runtime(directory) != versions:
        raise ValueError("runtime changed during execution")
    result["process_seconds"] = time.perf_counter() - started
    if os.environ.get("DQ11_REHEARSAL") == "1":
        result.update(schema="dq11-rehearsal-arm/1", science_eligible=False,
                      rehearsal="A2000 cu128 tiny random Mistral; correctness only")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--arm", choices=("L", "U", "U0"), required=True)
    parser.add_argument("--kind", choices=("proof", "read", "spread"), required=True)
    parser.add_argument("--repetition", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args), indent=2) + "\n")
