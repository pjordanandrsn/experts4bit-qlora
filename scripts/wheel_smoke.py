"""Built-artifact smoke: run against the INSTALLED wheel, never the repo tree.

Mirrors the README's promised surface (Install + the "Which door?" tree).
The invariant: README's promised surface == the wheel's importable surface.
CPU-pure: bf16-passthrough forward (no quantize kernels), plus the B-series
deprecation check — enable_hot_residency warns DeprecationWarning at call.
"""
import warnings

import torch

import experts4bit_qlora as e4b
from experts4bit_qlora import (  # the "Which door?" surface
    Experts4bit, ExpertsLoRA, ExpertsNbit,
    cold_engine_available, disable_cold_engine, disable_fast,
    disable_hot_residency, disable_pipelined_residency,
    enable_cold_engine, enable_expert_offload, enable_fast,
    enable_hot_residency, enable_pipelined_residency, verify_moe_4bit,
)

_SURFACE = (
    Experts4bit, ExpertsLoRA, ExpertsNbit,
    cold_engine_available, disable_cold_engine, disable_fast,
    disable_hot_residency, disable_pipelined_residency,
    enable_cold_engine, enable_expert_offload, enable_fast,
    enable_hot_residency, enable_pipelined_residency, verify_moe_4bit,
)


def main() -> int:
    print("version:", e4b.__version__)
    print(f"surface: {len(_SURFACE)} public names importable")
    import importlib.resources as ir
    assert ir.files("experts4bit_qlora").joinpath("py.typed").is_file(), "py.typed missing from wheel"
    print("py.typed ships")

    # reference-path micro: bf16 passthrough builds + forwards with no quantize kernels
    gate_up = torch.randn(4, 2 * 64, 32)
    down = torch.randn(4, 32, 64)
    base = ExpertsNbit.from_float(gate_up, down, has_gate=True, quant_type="bf16",
                                  compute_dtype=torch.float32)
    hs = torch.randn(3, 32)
    idx = torch.stack([torch.randperm(4)[:2] for _ in range(3)])
    wts = torch.softmax(torch.randn(3, 2), dim=-1)
    out = base(hs, idx, wts)
    assert out.shape == (3, 32) and torch.isfinite(out).all()
    print("reference forward micro OK")

    # deprecation: fires on the hot path (and only there)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            enable_hot_residency(object(), [])
        except Exception:
            pass  # kernel-import/arg errors are fine; the warning precedes them
        hot_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert hot_warns, "enable_hot_residency must warn DeprecationWarning at call"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert cold_engine_available() is True
        try:
            enable_cold_engine(base, hot_sets=[[]], device="cpu", dequant="torch")
            disable_cold_engine(base)
        except Exception:
            pass
        assert not [x for x in w if issubclass(x.category, DeprecationWarning)], \
            "no DeprecationWarning outside the hot path"
    print("deprecation fires on the hot path only")
    print("WHEEL SMOKE: ALL GREEN")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
