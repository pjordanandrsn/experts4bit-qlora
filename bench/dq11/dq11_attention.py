"""Matched SDPA install set and untimed, registered-shape attention admission."""
from __future__ import annotations

import importlib.util
import json
import os
from contextlib import nullcontext
from pathlib import Path

from dq11_common import verify_callable

SHAPE = {"batch": 1, "tokens": 2048, "query_heads": 32, "kv_heads": 8,
         "head_dim": 128, "dtype": "torch.bfloat16", "causal": True}
DEVIATION = "Unsloth SDPA attention matched to e4b; attention-kernel differences excluded from the measured gap"


def require_install_set():
    """Absence, rather than a model config label, selects pinned Unsloth's SDPA."""
    absent = {name: importlib.util.find_spec(name) is None for name in ("xformers", "flash_attn")}
    if not all(absent.values()):
        raise ValueError("matched SDPA requires xformers and flash-attn to be unimportable")
    return absent


def dispatched(events, *, math_required):
    """Identify executed native ops, including decomposed math backward."""
    names = {event["operator"] for event in events}
    if any("xformers" in name.lower() or "flash_attn::" in name for name in names):
        raise ValueError("optional attention op dispatched in matched SDPA arm")
    forward = {name for name in names if name.startswith("aten::_scaled_dot_product_") and "backward" not in name}
    fused = {
        "aten::_scaled_dot_product_flash_attention": "aten::_scaled_dot_product_flash_attention_backward",
        "aten::_scaled_dot_product_efficient_attention": "aten::_scaled_dot_product_efficient_attention_backward",
        "aten::_scaled_dot_product_cudnn_attention": "aten::_scaled_dot_product_cudnn_attention_backward",
    }
    math = "aten::_scaled_dot_product_attention_math"
    if len(forward) != 1 or (math_required and forward != {math}):
        raise ValueError("missing, ambiguous or non-math SDPA forward dispatch")
    op = next(iter(forward))
    if op == math:
        backward = {"SafeSoftmaxBackward0", "aten::_softmax_backward_data", "BmmBackward0", "aten::bmm"}
        # Native backward ops must actually execute inside the corresponding autograd node.
        needed = {("aten::_softmax_backward_data", "SafeSoftmaxBackward0"), ("aten::bmm", "BmmBackward0")}
        observed = {(event["operator"], event["parent"]) for event in events}
        if not needed <= observed or not backward <= names:
            raise ValueError("SDPA math backward was not dispatched")
        backend = "math"
    elif op in fused and fused[op] in names:
        backward = {fused[op]}
        backend = op.removeprefix("aten::_scaled_dot_product_").removesuffix("_attention")
    else:
        raise ValueError("matching SDPA backward was not dispatched")
    return {"backend": backend, "forward": [op], "backward": sorted(backward), "xformers_dispatched": False}


def validate_witness(row, *, kind):
    if (row.get("schema") != "dq11-attention-canary/1" or row.get("shape") != SHAPE
            or row.get("kind") != kind or row.get("optional_packages_absent") != {"xformers": True, "flash_attn": True}
            or row.get("finite_gradients") != {"query": True, "key": True, "value": True}
            or row.get("selector") != "sdpa" or row.get("profiler_removed") is not True
            or row.get("named_deviation") != DEVIATION or not row.get("bindings")):
        raise ValueError("incomplete matched attention engagement witness")
    if (any(type(row["shape"][key]) is not type(value) for key, value in SHAPE.items())
            or any(type(value) is not bool for value in row["optional_packages_absent"].values())
            or any(type(value) is not bool for value in row["finite_gradients"].values())
            or row.get("timed_training_profiled", False) is not False):
        raise ValueError("malformed attention witness or timed profiler engaged")
    dispatch = row.get("dispatch", {})
    if (dispatch.get("xformers_dispatched") is not False or not dispatch.get("forward")
            or not dispatch.get("backward") or (kind == "proof" and dispatch.get("backend") != "math")):
        raise ValueError("matched attention dispatch refused")
    if dispatch != dispatched(row["native_events"], math_required=kind == "proof"):
        raise ValueError("attention dispatch differs from native event evidence")


def canary(directory, arm, kind, *, quality_read=False):
    """One exact Unsloth dispatcher layer, before full model loading; never timed."""
    import torch
    from dq11_proof_policy import observer_policy, settings, validate_shipped_policy
    from torch.profiler import ProfilerActivity, profile
    from unsloth.models import mistral
    from unsloth.utils import attention_dispatch as attention

    directory = Path(directory)
    if arm not in ("U", "U0") or kind not in ("proof", "read", "spread"):
        raise ValueError("unregistered attention canary arm/phase")
    absent = require_install_set()
    sources = set(json.loads((directory / "source_authority.json").read_text())["sha256"])
    function = mistral.MistralAttention_fast_forward
    if (function.__globals__["run_attention"] is not attention.run_attention
            or function.__globals__["select_attention_backend"] is not attention.select_attention_backend
            or attention.scaled_dot_product_attention is not torch.nn.functional.scaled_dot_product_attention):
        raise ValueError("Mistral attention dispatcher binding changed")
    bindings = {name: verify_callable(callable_, sources) for name, callable_ in {
        "MistralAttention_fast_forward": function, "run_attention": attention.run_attention,
        "select_attention_backend": attention.select_attention_backend}.items()}
    backend = attention.select_attention_backend(False)
    if backend != "sdpa":
        raise ValueError("pinned Unsloth did not select SDPA")
    generator = torch.Generator(device="cuda").manual_seed(3407)
    query = torch.randn(1, 32, 2048, 128, device="cuda", dtype=torch.bfloat16,
                        generator=generator, requires_grad=True)
    key = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16,
                      generator=generator, requires_grad=True)
    value = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16,
                        generator=generator, requires_grad=True)
    config = attention.AttentionConfig(backend=backend, n_kv_heads=8, n_groups=4,
                                      flash_dense_kwargs={"causal": True, "window_size": (-1, -1)},
                                      flash_varlen_kwargs={"dropout_p": 0.0, "causal": True, "softmax_scale": None})
    context = attention.AttentionContext(bsz=1, q_len=2048, kv_seq_len=2048, n_heads=32, head_dim=128,
                                        requires_grad=True, seq_info=None, attention_mask=None, causal_mask=None)
    policy = observer_policy() if kind == "proof" else nullcontext()
    before = settings()
    with policy as policy_record:
        actual = settings()
        if kind != "proof":
            validate_shipped_policy(actual)
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            output = attention.run_attention(config=config, context=context, Q=query, K=key, V=value)
            output.float().square().mean().backward()
            torch.cuda.synchronize()
        events = [{"operator": event.name, "parent": event.cpu_parent.name if event.cpu_parent else None}
                  for event in profiler.events()]
        if len(events) > 4096:
            raise ValueError("attention canary exceeded bounded native event count")
        dispatch = dispatched(events, math_required=kind == "proof")
    if settings() != before:
        raise ValueError("attention canary changed execution policy")
    row = {"schema": "dq11-attention-canary/1", "arm": arm, "kind": kind, "shape": dict(SHAPE),
           "source": os.environ["E4B_SHA"], "nonce": os.environ["TC1_RUN_NONCE"],
           "science_eligible": not quality_read and os.environ.get("DQ11_REHEARSAL") != "1",
           "optional_packages_absent": absent, "selector": backend, "bindings": bindings,
           "settings": actual, "policy": policy_record, "dispatch": dispatch,
           "native_events": events, "named_deviation": DEVIATION,
           "finite_gradients": {name: tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
                                for name, tensor in (("query", query), ("key", key), ("value", value))},
           "profiler_removed": True, "timed_training_profiled": False}
    validate_witness(row, kind=kind)
    (directory / "receipts" / f"attention-canary-{kind}-{arm}-{os.getpid()}.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def validate_receipt(row):
    if row["arm"] not in ("U", "U0"):
        return
    witness = row["engaged"]["attention"]
    validate_witness(witness, kind=row["kind"])
    if (witness["arm"] != row["arm"] or witness["source"] != row["runtime"]["experts4bit-qlora"]
            or witness["nonce"] != row["nonce"] or witness["science_eligible"] is not row.get("science_eligible", True)
            or row["engaged"]["attention_deviation"] != DEVIATION):
        raise ValueError("attention witness identity/deviation changed")
    if row["kind"] == "proof":
        from dq11_reduce import validate_observer_policy

        validate_observer_policy(witness["policy"])
        if witness["settings"] != witness["policy"]["settings_actual"]:
            raise ValueError("attention canary configuration C settings changed")
    else:
        from dq11_proof_policy import validate_shipped_policy

        validate_shipped_policy(witness["settings"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    args = parser.parse_args()
    from dq11_proof_policy import process_environment

    with process_environment("proof"):
        require_install_set()
        if any(key.startswith("UNSLOTH_") for key in os.environ):
            raise ValueError("inherited Unsloth switches are not registered")
        os.environ.update(UNSLOTH_COMPILE_DISABLE="1", UNSLOTH_RETURN_LOGITS="1")
        from unsloth import FastLanguageModel  # noqa: F401, I001 -- Unsloth must import first
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if os.environ.get("DQ11_REHEARSAL") == "1":
            from dq11_rehearsal import require_hardware

            require_hardware(args.directory)
        elif (not torch.cuda.is_available() or torch.cuda.device_count() != 1
              or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5090"):
            raise ValueError("unregistered early attention canary device")
        print(json.dumps(canary(args.directory, "U", "proof"), indent=2))
