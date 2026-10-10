"""CPU-only legacy-101 regression; never part of the paid SDPA install set."""
from __future__ import annotations

import importlib.metadata
import json
from unittest.mock import patch


def check():
    import torch
    from xformers.ops.fmha import cutlass, dispatch, flash, flash3
    from xformers.ops.fmha.attn_bias import LowerTriangularMask
    from xformers.ops.fmha.common import Inputs

    if (importlib.metadata.version("xformers") != "0.0.35" or str(torch.__version__) != "2.12.1+cu130"
            or torch.cuda.is_available() or torch.cuda.is_initialized()):
        raise ValueError("regression requires isolated legacy cu130/xformers lock without CUDA devices")

    class CudaMetadata:
        # Real meta tensor layout, mocked device only; no CUDA allocation or operator launch.
        device = torch.device("cuda:0")

        def __init__(self):
            self.tensor = torch.empty((1, 2048, 32, 128), device="meta", dtype=torch.bfloat16)

        def __getattr__(self, name):
            return getattr(self.tensor, name)

    inp = Inputs(query=CudaMetadata(), key=CudaMetadata(), value=CudaMetadata(),
                 attn_bias=LowerTriangularMask(), p=0.0)
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        with patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
            torch.use_deterministic_algorithms(True, warn_only=True)
            rejected = {op.NAME: op.not_supported_reasons(inp) for op in (flash3.BwOp, flash.BwOp, cutlass.BwOp)}
            if (any(not reasons for reasons in rejected.values())
                    or not any("non-deterministic" in reason for reason in rejected[flash.BwOp.NAME])
                    or not any("(12, 0) (too new)" in reason for reason in rejected[cutlass.BwOp.NAME])):
                raise ValueError("legacy SM12/deterministic backward rejection changed")
            try:
                dispatch._dispatch_bw(inp, varlen_lse_packed=False)
            except NotImplementedError as error:
                refusal = str(error)
            else:
                raise ValueError("xformers backward silently became admissible under C")
            # Paired control: removing determinism admits FA2 on the same shape/capability.
            torch.use_deterministic_algorithms(False)
            unforced = flash.BwOp.not_supported_reasons(inp)
            if unforced:
                raise ValueError("FA2 positive control refused: " + repr(unforced))
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)):
            torch.use_deterministic_algorithms(True, warn_only=True)
            ampere = cutlass.BwOp.not_supported_reasons(inp)
            if ampere:
                raise ValueError("A2000 cutlass positive control refused: " + repr(ampere))
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
    if torch.cuda.is_initialized():
        raise ValueError("CPU regression initialized CUDA")
    return {"schema": "dq11-legacy-xformers-backward-regression/1", "cpu_only": True,
            "torch": str(torch.__version__), "xformers": importlib.metadata.version("xformers"),
            "mocked_capability": [12, 0], "deterministic_algorithms": True, "warn_only": True,
            "shape": [1, 2048, 32, 128], "dtype": "torch.bfloat16", "bias": "LowerTriangularMask",
            "actual_not_supported_reasons": rejected, "actual_dispatch_refusal": refusal,
            "sm12_nondeterministic_fa2_control": unforced, "sm86_deterministic_cutlass_control": ampere,
            "cuda_initialized": False, "passed": True}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
