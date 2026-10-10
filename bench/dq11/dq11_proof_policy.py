"""Amendment 4: a bitwise observer proof under C, with shipped noise reported separately."""
from __future__ import annotations

from contextlib import contextmanager
import itertools
import os
import re
import sys
import warnings

from dq11_common import tensor_sha

CONFIGURATION_C = {"cublas_workspace_config": ":4096:8", "deterministic_algorithms": True,
                   "warn_only": True, "sdpa": "math", "comparison": "bitwise"}


SHIPPED_POLICY = {"cublas_workspace_config": None, "deterministic_algorithms": False,
                  "warn_only": False, "flash_sdp": True, "mem_efficient_sdp": True,
                  "math_sdp": True, "cudnn_sdp": True, "tf32_matmul": False, "tf32_cudnn": False}


def validate_shipped_policy(actual):
    if (actual != SHIPPED_POLICY
            or any(type(actual.get(key)) is not bool for key in SHIPPED_POLICY if key != "cublas_workspace_config")):
        raise ValueError("execution settings differ from registered shipped policy")


@contextmanager
def process_environment(kind):
    """Set workspace before CUDA initialization, only in a fresh proof process."""
    prior = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prior is not None:
        raise ValueError("inherited cuBLAS workspace policy is not registered")
    framework = sys.modules.get("torch")
    if kind == "proof" and framework is not None and framework.cuda.is_initialized():
        raise ValueError("proof workspace must be configured before CUDA initialization")
    if kind == "proof":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CONFIGURATION_C["cublas_workspace_config"]
    try:
        yield
    finally:
        if kind == "proof":
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def settings():
    import torch

    return {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}


@contextmanager
def observer_policy():
    """Record actual C controls and warnings; restore backend/determinism even on refusal."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CONFIGURATION_C["cublas_workspace_config"]:
        raise ValueError("observer proof workspace was not configured before CUDA")
    prior = settings()
    record = {"configuration": dict(CONFIGURATION_C), "warnings": [], "warned_ops": [], "restored": False}
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.use_deterministic_algorithms(True, warn_only=True)
            with sdpa_kernel(SDPBackend.MATH):
                record["settings_actual"] = settings()
                yield record
    finally:
        torch.use_deterministic_algorithms(prior["deterministic_algorithms"], warn_only=prior["warn_only"])
        record["warnings"] = [{"category": w.category.__name__, "message": str(w.message),
                               "filename": w.filename, "lineno": w.lineno} for w in caught]
        for w in record["warnings"]:
            if "determin" in w["message"].lower():
                match = re.match(r"(.+?) (?:does not have a deterministic implementation|is (?:non-)?deterministic|defaults to a non-deterministic algorithm)",
                                 w["message"])
                record["warned_ops"].append({"operator": match.group(1) if match else "UNKNOWN",
                                            "message": w["message"]})
        record["restored"] = settings() == prior
        if not record["restored"]:
            raise ValueError("observer proof backend policy was not restored")


def capture_pass(model, slots, ids):
    import torch

    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    gradients = {}
    for key, parameter in slots.items():
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise ValueError("missing or non-FP32 same-token gradient: " + key)
        gradients[key] = parameter.grad.detach().cpu().contiguous().clone()
        if not torch.isfinite(gradients[key]).all():
            raise ValueError("nonfinite same-token gradient: " + key)
    value = loss.detach().cpu().contiguous().clone()
    if not torch.isfinite(value):
        raise ValueError("nonfinite same-token loss")
    return {"loss": float(value), "loss_sha256": tensor_sha(value),
            "gradient_hashes": {key: tensor_sha(t) for key, t in gradients.items()}}, gradients


def spread(model, slots, ids, *, output=None, arm=None):
    """Four shipped-setting, zero-update backwards; report all six pairs, with no tolerance."""
    before = settings()
    if before["cublas_workspace_config"] is not None or before["deterministic_algorithms"]:
        raise ValueError("spread requires shipped execution settings")
    runs, values = [], []
    try:
        for ordinal in range(1, 5):
            run, gradients = capture_pass(model, slots, ids)
            if output is not None:
                from safetensors.torch import save_file
                from dq11_common import file_sha

                path = output / f"spread-{arm}-{ordinal}.safetensors"
                save_file(gradients, str(path))
                run.update(gradient_file=path.name, gradient_file_bytes=path.stat().st_size,
                           gradient_file_sha256=file_sha(path))
            runs.append({"ordinal": ordinal, **run})
            values.append(gradients)  # CPU copies only; no graph or CUDA storage retained.
        pairs = []
        for x, y in itertools.combinations(range(4), 2):
            differences = {}
            for key in slots:
                gx, gy = values[x][key], values[y][key]
                difference = float((gx - gy).abs().max())
                denominator = max(float(gx.abs().max()), float(gy.abs().max()))
                if runs[x]["gradient_hashes"][key] != runs[y]["gradient_hashes"][key]:
                    differences[key] = {"max_abs_difference": difference,
                                        "relative_to_larger_max": difference / denominator if denominator else 0.0}
            pairs.append({"x": x + 1, "y": y + 1,
                          "loss_bitwise_equal": runs[x]["loss_sha256"] == runs[y]["loss_sha256"],
                          "loss_abs_difference": abs(runs[x]["loss"] - runs[y]["loss"]),
                          "gradient_differences": differences})
        if settings() != before:
            raise ValueError("shipped spread execution policy changed")
        return {"schema": "dq11-shipped-spread/1", "passes": runs, "pairs": pairs,
                "settings": before, "train_block": 0, "optimizer_updates": 0,
                "gate": False, "tolerance": None,
                "interpretation": "Same tokens and unchanged initializer; all six in-process pairs. Hash differences include signed zero; magnitude never gates DQ11."}
    finally:
        model.zero_grad(set_to_none=True)
